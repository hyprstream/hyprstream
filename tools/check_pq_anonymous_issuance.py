#!/usr/bin/env python3
"""Validate the #1060 experimental PQ-anonymous issuance refusal boundary.

This tool intentionally performs no cryptography.  It validates deterministic,
repository-owned fixtures and proves that production use stays unavailable until
a reviewed PQ anonymous construction is selected.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "docs/standards/registry/pq-anonymous-issuance.json"
VECTOR_PATH = ROOT / "docs/standards/vectors/pq-anonymous-boundary-v1.json"


class BoundaryError(ValueError):
    """One exact structural refusal reason."""


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BoundaryError(f"{path}: top level must be an object")
    return value


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise BoundaryError(reason)


def validate_registry(registry: dict[str, Any]) -> None:
    require(registry.get("profile_version") == 1, "registry:profile-version")
    require(
        registry.get("profile_id") == "hs-pp-pqhybrid/experimental-boundary/v1",
        "registry:profile-id",
    )
    scope = registry.get("scope", "")
    for phrase in ("not a production wire schema", "Privacy Pass token type", "IANA allocation"):
        require(phrase in scope, "registry:scope")
    require(registry.get("production_enabled") is False, "registry:production-enabled")
    require(
        registry.get("production_refusal") == "construction-unselected",
        "registry:refusal",
    )
    require(registry.get("token_type") is None, "registry:token-type-must-be-unselected")
    require(registry.get("pq_construction") is None, "registry:pq-construction-must-be-unselected")

    composition = registry.get("composition", {})
    require(composition.get("operator") == "and", "registry:composition")
    require(composition.get("exact_leg_count") == 2, "registry:leg-count")
    require(
        composition.get("ordered_roles") == ["classical-anonymous", "pq-anonymous"],
        "registry:leg-order",
    )
    for key in ("unknown_roles", "duplicate_roles", "reordered_roles", "crossed_context"):
        require(composition.get(key) == "reject", f"registry:{key}")

    purposes = registry.get("key_purposes", {})
    require(
        purposes.get("classical-anonymous") == "privacy-pass-classical-issuance",
        "registry:classical-purpose",
    )
    require(
        purposes.get("pq-anonymous") == "privacy-pass-pq-anonymous-issuance",
        "registry:pq-purpose",
    )
    require(purposes.get("cross_role_reuse") == "reject", "registry:key-reuse")
    forbidden = purposes.get("forbidden", [])
    require(
        len(forbidden) == len(set(forbidden)) and "mesh-signing" in forbidden,
        "registry:forbidden-purposes",
    )

    schemas = registry.get("message_schemas", {})
    require(
        set(schemas) == {"challenge", "issuance-request", "issuance-response", "token", "leg"},
        "registry:message-schemas",
    )
    for name, fields in schemas.items():
        require(
            isinstance(fields, list)
            and fields
            and all(isinstance(field, str) and field for field in fields)
            and len(fields) == len(set(fields)),
            f"registry:{name}-fields",
        )


def find_forbidden_holder_field(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            key in forbidden or find_forbidden_holder_field(item, forbidden)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(find_forbidden_holder_field(item, forbidden) for item in value)
    return False


def exact_fields(message: dict[str, Any], expected: list[str], kind: str) -> None:
    require(set(message) == set(expected), f"{kind}:fields")


def is_hex(value: Any, *, minimum: int, maximum: int) -> bool:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum or len(value) % 2:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def validate_legs(
    kind: str,
    message: dict[str, Any],
    registry: dict[str, Any],
    challenge_digest: str,
) -> None:
    require(message.get("composition") == "and", f"{kind}:composition")
    legs = message.get("legs")
    require(isinstance(legs, list), f"{kind}:legs")
    expected_roles = registry["composition"]["ordered_roles"]
    roles = [leg.get("role") if isinstance(leg, dict) else None for leg in legs]
    require(all(role in expected_roles for role in roles), f"{kind}:unknown-leg")
    require(len(roles) == len(set(roles)), f"{kind}:duplicate-leg")
    require(len(legs) == registry["composition"]["exact_leg_count"], f"{kind}:leg-count")
    require(roles == expected_roles, f"{kind}:leg-order")

    key_ids: list[str] = []
    bounds = registry["bounds"]
    for leg in legs:
        exact_fields(leg, registry["message_schemas"]["leg"], kind)
        role = leg["role"]
        require(leg.get("purpose") == registry["key_purposes"][role], f"{kind}:key-purpose")
        key_id = leg.get("issuer_key_id")
        require(
            isinstance(key_id, str) and 0 < len(key_id.encode("utf-8")) <= bounds["identifier_max_bytes"],
            f"{kind}:key-id",
        )
        key_ids.append(key_id)
        require(leg.get("challenge_digest") == challenge_digest, f"{kind}:crossed-challenge")
        require(
            is_hex(
                leg.get("context_digest"),
                minimum=bounds["digest_hex_chars"],
                maximum=bounds["digest_hex_chars"],
            ),
            f"{kind}:context-digest",
        )
        artifact = leg.get("artifact")
        require(
            is_hex(artifact, minimum=0, maximum=bounds["artifact_hex_max_chars"]),
            f"{kind}:artifact",
        )
        if role == "classical-anonymous":
            require(
                leg.get("suite_id") == "rfc9578-publicly-verifiable-v1",
                f"{kind}:classical-suite",
            )
            require(bool(artifact), f"{kind}:classical-artifact")
        else:
            require(
                leg.get("suite_id") is None and artifact == "",
                f"{kind}:pq-must-remain-unselected",
            )
    require(len(key_ids) == len(set(key_ids)), f"{kind}:key-reuse")


def validate_boundary(
    baseline: dict[str, Any],
    evaluation: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    messages = baseline.get("messages")
    require(isinstance(messages, dict), "messages")
    forbidden = set(registry.get("forbidden_holder_fields", []))
    require(not find_forbidden_holder_field(messages, forbidden), "holder-identifying-field")
    expected_message_kinds = {"challenge", "issuance-request", "issuance-response", "token"}
    require(set(messages) == expected_message_kinds, "unknown-message")

    profile_id = registry["profile_id"]
    schemas = registry["message_schemas"]
    for kind in expected_message_kinds:
        message = messages[kind]
        require(isinstance(message, dict), f"{kind}:object")
        exact_fields(message, schemas[kind], kind)
        require(message.get("kind") == kind, f"{kind}:kind")
        require(message.get("version") == 1, f"{kind}:version")
        require(message.get("profile_id") == profile_id, f"{kind}:profile")

    challenge = messages["challenge"]
    bounds = registry["bounds"]
    require(challenge.get("issuer_id") == evaluation.get("issuer_id"), "challenge:issuer")
    require(challenge.get("audience") == evaluation.get("audience"), "challenge:audience")
    require(
        challenge.get("suite_id") == "unselected-pq-anonymous-construction",
        "challenge:suite",
    )
    require(
        challenge.get("accepted_state_epoch") == evaluation.get("current_accepted_state_epoch"),
        "challenge:stale-state",
    )
    require(
        challenge.get("policy_generation") == evaluation.get("current_policy_generation"),
        "challenge:policy-rollback",
    )
    require(isinstance(challenge.get("expiry"), int), "challenge:expiry-type")
    require(challenge["expiry"] > evaluation.get("now"), "challenge:expired")
    require(
        is_hex(
            challenge.get("accepted_state_digest"),
            minimum=bounds["digest_hex_chars"],
            maximum=bounds["digest_hex_chars"],
        ),
        "challenge:state-digest",
    )
    require(
        is_hex(
            challenge.get("challenge_nonce"),
            minimum=bounds["nonce_hex_min_chars"],
            maximum=bounds["nonce_hex_max_chars"],
        ),
        "challenge:nonce",
    )

    request_digest = messages["issuance-request"].get("challenge_digest")
    require(
        is_hex(request_digest, minimum=bounds["digest_hex_chars"], maximum=bounds["digest_hex_chars"]),
        "issuance-request:challenge-digest",
    )
    for kind in ("issuance-request", "issuance-response", "token"):
        message = messages[kind]
        require(message.get("challenge_digest") == request_digest, f"{kind}:crossed-challenge")
        validate_legs(kind, message, registry, request_digest)

    token = messages["token"]
    for field in (
        "issuer_id",
        "audience",
        "suite_id",
        "accepted_state_digest",
        "accepted_state_epoch",
        "policy_generation",
        "expiry",
        "redemption_context",
        "resource_profile_commitment",
    ):
        require(token.get(field) == challenge.get(field), f"token:crossed-{field}")
    require(
        token.get("production_disposition") == "refuse-construction-unselected",
        "token:production-disposition",
    )


def production_validate(registry: dict[str, Any]) -> None:
    if (
        registry.get("production_enabled") is False
        and registry.get("token_type") is None
        and registry.get("pq_construction") is None
    ):
        raise BoundaryError(registry["production_refusal"])
    raise BoundaryError("unsafe-production-state")


def resolve_parent(root: Any, path: list[Any]) -> tuple[Any, Any]:
    require(bool(path), "mutation:empty-path")
    current = root
    for part in path[:-1]:
        current = current[part]
    return current, path[-1]


def apply_mutation(baseline: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(baseline)
    parent, leaf = resolve_parent(value, mutation["path"])
    operation = mutation.get("op")
    if operation == "delete":
        del parent[leaf]
    elif operation in ("replace", "add"):
        parent[leaf] = copy.deepcopy(mutation.get("value"))
    elif operation == "duplicate":
        parent.insert(leaf + 1, copy.deepcopy(parent[leaf]))
    elif operation == "swap":
        left, right = mutation["indices"]
        parent[leaf][left], parent[leaf][right] = parent[leaf][right], parent[leaf][left]
    else:
        raise BoundaryError(f"mutation:unknown-operation:{operation}")
    return value


def main() -> int:
    try:
        registry = load_object(REGISTRY_PATH)
        vectors = load_object(VECTOR_PATH)
        validate_registry(registry)
        require(vectors.get("vector_version") == 1, "vectors:version")
        require(vectors.get("profile_id") == registry["profile_id"], "vectors:profile")
        require(vectors.get("status") == "non-cryptographic-structural-fixture", "vectors:status")

        baseline = vectors.get("baseline")
        evaluation = vectors.get("evaluation")
        require(isinstance(baseline, dict) and isinstance(evaluation, dict), "vectors:baseline")
        validate_boundary(baseline, evaluation, registry)

        digest = hashlib.sha256(canonical_json(baseline)).hexdigest()
        require(
            vectors.get("baseline_canonical_sha256") == digest,
            f"vectors:canonical-digest:{digest}",
        )

        try:
            production_validate(registry)
        except BoundaryError as exc:
            require(
                str(exc) == vectors.get("expected_production_refusal"),
                "vectors:production-refusal",
            )
        else:
            raise BoundaryError("vectors:production-was-enabled")

        mutations = vectors.get("mutations")
        require(isinstance(mutations, list) and mutations, "vectors:mutations")
        mutation_ids: set[str] = set()
        for mutation in mutations:
            mutation_id = mutation.get("id")
            require(
                isinstance(mutation_id, str) and mutation_id not in mutation_ids,
                "vectors:mutation-id",
            )
            mutation_ids.add(mutation_id)
            mutated = apply_mutation(baseline, mutation)
            try:
                validate_boundary(mutated, evaluation, registry)
            except BoundaryError as exc:
                require(str(exc) == mutation.get("expected_error"), f"{mutation_id}:got:{exc}")
            else:
                raise BoundaryError(f"{mutation_id}:mutation-not-effective")
    except (BoundaryError, KeyError, IndexError, TypeError) as exc:
        print(f"PQ anonymous issuance boundary validation failed: {exc}", file=sys.stderr)
        return 1

    print("PQ anonymous issuance experimental boundary: structurally valid")
    print("production issuance/redemption: REFUSED (construction-unselected)")
    print(f"mutation-effective negative controls: {len(mutations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
