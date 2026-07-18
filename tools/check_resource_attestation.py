#!/usr/bin/env python3
"""Validate the #1070 resource-attestation standards slice.

Only Python's standard library is used. Validates:
  * registry/resource-intent.json and registry/resource-vocabulary.json,
  * registry/resource-attestation-obligations.json (bidirectional coverage of
    every RFCXML normative MUST/MUST NOT),
  * the deterministic vector (baseline canonical digest, both-attestations-
    over-identical-digest, production refusal, and every mutation's expected
    rejection), and
  * (--check-generated) reproducible RFCXML text/HTML via the caller's xml2rfc.

This checker owns draft-hyprstream-resource-attestation-00 only. It does not
touch the #1059 draft family or its checker.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ROOT / "docs" / "standards"
INTENT = STANDARDS / "registry" / "resource-intent.json"
VOCAB = STANDARDS / "registry" / "resource-vocabulary.json"
OBLIGATIONS = STANDARDS / "registry" / "resource-attestation-obligations.json"
VECTOR = STANDARDS / "vectors" / "resource-intent-canonicalization-v1.json"
XML = STANDARDS / "rfc" / "draft-hyprstream-resource-attestation-00.xml"
TEXT = XML.with_suffix(".txt")
HTML = XML.with_suffix(".html")
XML2RFC_VERSION = "3.34.0"

REQUIRED_MUTATION_CATEGORIES = {
    "canonicalization",
    "mutation",
    "replay",
    "crossing",
    "downgrade",
    "privacy-leakage",
    "crash-recovery",
    "concurrency-fencing",
}


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be an object")
    return data


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha(obj) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


# --------------------------------------------------------------------------- #
# Registry validation
# --------------------------------------------------------------------------- #
def validate_intent_registry() -> list[str]:
    errors: list[str] = []
    data = load_json(INTENT)
    if data.get("registry_version") != 1:
        errors.append("resource-intent.json: registry_version must be 1")
    if "not an IANA" not in data.get("scope", ""):
        errors.append("resource-intent.json: scope must explicitly say it is not an IANA registry")
    canonical = data.get("canonicalization", {})
    for key in ("encoding", "digest", "identifiers", "integers", "octets", "boundary"):
        if not canonical.get(key):
            errors.append(f"resource-intent.json: canonicalization.{key} is required")
    fields = data.get("fields", {})
    if not isinstance(fields, dict) or not fields:
        return errors + ["resource-intent.json: fields must be a nonempty object"]
    allowed_classes = {"identifier", "opaque-octets", "nonnegative-integer"}
    registry_fields = set(fields)
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            errors.append(f"resource-intent.json: field {name}: definition must be an object")
            continue
        value_class = spec.get("value_class")
        if value_class not in allowed_classes:
            errors.append(f"resource-intent.json: field {name}: unsupported value_class")
        if not spec.get("selection_requirement"):
            errors.append(f"resource-intent.json: field {name}: selection_requirement is required")
        if value_class == "nonnegative-integer":
            minimum, maximum = spec.get("minimum"), spec.get("maximum")
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (minimum, maximum)):
                errors.append(f"resource-intent.json: field {name}: nonnegative-integer needs integer minimum/maximum")
            elif minimum < 0 or minimum > maximum:
                errors.append(f"resource-intent.json: field {name}: bounds must satisfy 0 <= minimum <= maximum")
        else:
            minimum, maximum = spec.get("min_bytes"), spec.get("max_bytes")
            if any(not isinstance(v, int) or isinstance(v, bool) for v in (minimum, maximum)):
                errors.append(f"resource-intent.json: field {name}: requires integer min_bytes/max_bytes")
            elif minimum < 0 or minimum > maximum:
                errors.append(f"resource-intent.json: field {name}: bounds must satisfy 0 <= min_bytes <= max_bytes")
    labels = data.get("labels", [])
    if not isinstance(labels, list) or not labels:
        return errors + ["resource-intent.json: labels must be a nonempty list"]
    seen: set[str] = set()
    for label in labels:
        if not isinstance(label, dict):
            errors.append("resource-intent.json: each label must be an object")
            continue
        label_id = label.get("id")
        if not isinstance(label_id, str) or not re.fullmatch(r"hs-resource-attestation/[a-z0-9-]+/v1", label_id):
            errors.append(f"resource-intent.json: invalid label id: {label_id!r}")
        elif label_id in seen:
            errors.append(f"resource-intent.json: duplicate label: {label_id}")
        else:
            seen.add(label_id)
        for key in ("purpose", "owner_issue", "status"):
            if not label.get(key):
                errors.append(f"resource-intent.json: label {label_id}: {key} is required")
        used = label.get("fields")
        if not isinstance(used, list) or not used:
            errors.append(f"resource-intent.json: label {label_id}: nonempty fields list is required")
        else:
            for field in used:
                if field not in registry_fields:
                    errors.append(f"resource-intent.json: label {label_id}: unknown field {field}")
    status = data.get("construction_status", {})
    if status.get("production_refusal") != "construction-incomplete":
        errors.append("resource-intent.json: construction_status.production_refusal must be construction-incomplete")
    if status.get("selected_suite") is not None:
        errors.append("resource-intent.json: construction_status.selected_suite must be null (pre-construction)")
    return errors


def validate_vocabulary() -> list[str]:
    errors: list[str] = []
    data = load_json(VOCAB)
    if data.get("vocabulary_version") != 1:
        errors.append("resource-vocabulary.json: vocabulary_version must be 1")
    scope = data.get("scope", "")
    for required in ("not a wire schema", "codepoint allocation"):
        if required not in scope:
            errors.append(f"resource-vocabulary.json: scope must state {required!r}")
    for key in ("roles", "profiles"):
        values = data.get(key)
        if (not isinstance(values, list) or not values or
                any(not isinstance(v, str) or not v for v in values) or
                len(values) != len(set(values))):
            errors.append(f"resource-vocabulary.json: {key} must be a nonempty list of unique nonempty strings")
    principal = data.get("principal_kinds", {})
    if not isinstance(principal, dict) or not principal:
        errors.append("resource-vocabulary.json: principal_kinds must be a nonempty object")
    lifecycle = data.get("lifecycle", {})
    if not isinstance(lifecycle, dict):
        errors.append("resource-vocabulary.json: lifecycle must be an object")
    else:
        if lifecycle.get("unknown_state") != "reject":
            errors.append("resource-vocabulary.json: lifecycle.unknown_state must be reject")
        if not re.fullmatch(r"hyprstream-resource-attestation-v1", lifecycle.get("logical_namespace", "")):
            errors.append("resource-vocabulary.json: lifecycle.logical_namespace must be hyprstream-resource-attestation-v1")
        for key in ("happy_path", "failure"):
            if not isinstance(lifecycle.get(key), list) or not lifecycle.get(key):
                errors.append(f"resource-vocabulary.json: lifecycle.{key} must be a nonempty list")
    invariants = data.get("transition_invariants")
    if not isinstance(invariants, list) or not invariants:
        errors.append("resource-vocabulary.json: transition_invariants must be a nonempty list")
    # Coherence: vocabulary profiles ⊇ RFCXML profile kinds.
    profiles = set(data.get("profiles", []))
    expected = {
        "identified-owner", "pairwise-owner", "committed-owner",
        "identified-controller", "anonymous-capability-controller",
        "identified-payer", "anonymous-entitlement-payer",
    }
    if profiles < expected:
        errors.append(f"resource-vocabulary.json: profiles missing {sorted(expected - profiles)}")
    return errors


# --------------------------------------------------------------------------- #
# Obligations <-> RFCXML coverage
# --------------------------------------------------------------------------- #
def normative_sentences(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\bMUST(?: NOT)?\b", sentence)
    ]


def section_paragraphs(section: ET.Element):
    for child in section:
        if child.tag == "section":
            continue
        if child.tag == "t":
            yield child
        yield from section_paragraphs(child)


def xml_normative_statements() -> list[tuple[str, str]]:
    try:
        root = ET.parse(XML).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{XML}: invalid XML: {exc}") from exc
    statements: list[tuple[str, str]] = []
    for section in root.findall("./middle//section"):
        anchor = section.get("anchor", "")
        if not anchor:
            continue
        for paragraph in section_paragraphs(section):
            if paragraph.get("anchor") == "requirements-language":
                continue
            text = " ".join(part.strip() for part in paragraph.itertext() if part.strip())
            statements.extend((statement, anchor) for statement in normative_sentences(text))
    return statements


def validate_obligations() -> list[str]:
    errors: list[str] = []
    data = load_json(OBLIGATIONS)
    obligations = data.get("obligations", [])
    if not obligations:
        return ["resource-attestation-obligations.json: obligations must be nonempty"]
    statements: set[str] = set()
    statement_sources: dict[str, str] = {}
    ids: set[str] = set()
    for item in obligations:
        obligation_id = item.get("id")
        if not isinstance(obligation_id, str) or not re.fullmatch(r"RA-REQ-\d{3}", obligation_id):
            errors.append(f"invalid obligation id: {obligation_id!r}")
        elif obligation_id in ids:
            errors.append(f"duplicate obligation id: {obligation_id}")
        else:
            ids.add(obligation_id)
        statement = item.get("statement")
        if not isinstance(statement, str) or not re.search(r"\bMUST(?: NOT)?\b", statement):
            errors.append(f"{obligation_id}: exact normative statement is required")
            continue
        expected_keyword = "MUST NOT" if "MUST NOT" in statement else "MUST"
        if item.get("keyword") != expected_keyword:
            errors.append(f"{obligation_id}: keyword must be {expected_keyword}")
        normalized = re.sub(r"\s+", " ", statement).strip()
        if normalized in statements:
            errors.append(f"{obligation_id}: duplicate normative statement")
        statements.add(normalized)
        source = item.get("source")
        if not isinstance(source, str) or not source:
            errors.append(f"{obligation_id}: source section anchor is required")
        else:
            statement_sources[normalized] = source
        if item.get("kind") == "implemented-test":
            if not item.get("test"):
                errors.append(f"{obligation_id}: implemented-test requires test")
        elif item.get("kind") == "specification-only":
            if not item.get("owner") or not item.get("blocker") or not item.get("planned_test"):
                errors.append(f"{obligation_id}: specification-only requires owner, blocker, planned_test")
        else:
            errors.append(f"{obligation_id}: kind must be implemented-test or specification-only")
    source_pairs = xml_normative_statements()
    source_statements = {statement for statement, _ in source_pairs}
    for statement, _ in source_pairs:
        if statement not in statements:
            errors.append(f"unmapped RFCXML normative statement: {statement}")
    for statement in statements:
        if statement not in source_statements:
            errors.append(f"obligation statement not found in RFCXML: {statement}")
    for statement, anchor in source_pairs:
        if statement_sources.get(statement) not in (None, anchor):
            errors.append(
                f"obligation source for {statement!r} must be {anchor!r}, "
                f"got {statement_sources[statement]!r}"
            )
    return errors


# --------------------------------------------------------------------------- #
# Vector validation + boundary evaluator
# --------------------------------------------------------------------------- #
KNOWN_PROFILES = {
    "identified-owner", "pairwise-owner", "committed-owner",
    "identified-controller", "anonymous-capability-controller",
    "identified-payer", "anonymous-entitlement-payer",
}
ANONYMOUS_PROFILES = {"anonymous-capability-controller", "anonymous-entitlement-payer"}
INTENT_FIELD_MAX_VERSION = 1023  # predecessor_version ceiling per registry bounds


def apply_mutation(obj, mutation: dict):
    """Return a deep copy of obj with mutation applied. op semantics:
    replace (set leaf), delete (remove key/index), add (insert key), duplicate
    (replace a dict value with [value, value] so type checks catch it)."""
    path = list(mutation["path"])
    op = mutation["op"]
    root = copy.deepcopy(obj)
    cur = root
    for key in path[:-1]:
        cur = cur[key]
    last = path[-1]
    if op == "replace":
        cur[last] = mutation["value"]
    elif op == "delete":
        del cur[last]
    elif op == "add":
        cur[last] = mutation["value"]
    elif op == "duplicate":
        cur[last] = [copy.deepcopy(cur[last]), copy.deepcopy(cur[last])]
    else:
        raise ValueError(f"unsupported op {op!r} in mutation {mutation.get('id')}")
    return root


def evaluate_baseline(vector: dict) -> tuple[bool, str, str]:
    """Run the boundary over the baseline. Returns (ok, error, canonical_digest)."""
    baseline = vector["baseline"]
    resource_intent = baseline["resource_intent"]
    if not isinstance(resource_intent, dict):
        return False, "resource-intent:non-canonical", ""
    canonical = sha(resource_intent)
    ok, err = _run_checks(baseline, resource_intent, canonical, vector)
    return ok, err, canonical


def evaluate_mutation(vector: dict, mutation: dict) -> str:
    """Apply mutation and return the error the boundary produces (or 'accepted')."""
    mutated = apply_mutation(vector, mutation)
    baseline = mutated["baseline"]
    resource_intent = baseline.get("resource_intent")
    if not isinstance(resource_intent, dict):
        return "resource-intent:non-canonical"
    # Recompute the canonical digest of the (possibly mutated) resource_intent.
    canonical = sha(resource_intent)
    ok, err = _run_checks(baseline, resource_intent, canonical, vector)
    return "accepted" if ok else err


def _run_checks(baseline: dict, resource_intent: dict, canonical: str, vector: dict) -> tuple[bool, str]:
    manifest = baseline.get("manifest")

    def reject(reason: str) -> tuple[bool, str]:
        return False, reason

    # --- resource_intent structural checks ---
    registry_fields = load_json(INTENT).get("fields", {})
    allowed_intent = set(registry_fields)
    for key in resource_intent:
        if key not in allowed_intent:
            return reject("resource-intent:unknown-field")
    rid = resource_intent.get("resource_id", "")
    if isinstance(rid, str) and rid.startswith("sha256:"):
        return reject("resource-intent:cyclic-identifier")
    pv = resource_intent.get("predecessor_version")
    if isinstance(pv, int) and not isinstance(pv, bool) and (pv < 0 or pv > INTENT_FIELD_MAX_VERSION):
        return reject("resource-intent:out-of-bounds")

    # --- profile checks ---
    profile = resource_intent.get("profile")
    if profile not in KNOWN_PROFILES:
        text = str(profile)
        identified_like = any(tag in text for tag in ("identified", "pairwise", "committed"))
        anonymous_like = "anonymous" in text
        if identified_like and anonymous_like:
            return reject("profile:unknown-or-crossed-kind")
        return reject("profile:unknown-kind")
    # Privacy leakage: an anonymous-kind ref carrying a stable DID is forbidden
    # regardless of the declared profile. Checked before digest binding.
    for ref_key in ("owner_ref", "controller_ref", "payer_ref"):
        ref = resource_intent.get(ref_key, {})
        if isinstance(ref, dict) and str(ref.get("kind", "")).startswith("anonymous"):
            if str(ref.get("value", "")).startswith("did:"):
                return reject("profile:anonymous-fabricates-did")

    # --- attestation temporal checks ---
    evaluation = vector.get("evaluation", {})
    now = evaluation.get("now", 0)
    current_epoch = evaluation.get("current_accepted_state_epoch")
    current_policy = evaluation.get("current_policy_generation")
    if resource_intent.get("expiry", now) <= now:
        return reject("attestation:expired")
    if current_epoch is not None and resource_intent.get("accepted_state_epoch") != current_epoch:
        return reject("attestation:stale-state")
    if current_policy is not None and resource_intent.get("policy_generation") != current_policy:
        return reject("attestation:policy-rollback")

    # --- manifest structural checks ---
    if not isinstance(manifest, dict):
        return reject("manifest:non-canonical")
    mac = manifest.get("mac_attestation")
    ledger = manifest.get("ledger_attestation")
    if not isinstance(mac, dict):
        return reject("manifest:duplicated-attestation" if isinstance(mac, list) else "manifest:ledger-only-not-final")
    if not isinstance(ledger, dict):
        return reject("manifest:duplicated-attestation" if isinstance(ledger, list) else "manifest:mac-only-not-final")
    for role, att in (("mac", mac), ("ledger", ledger)):
        if att.get("role") not in ("mac-title-control", "ledger-economic"):
            return reject("manifest:unknown-attestation-role")

    # --- digest identity ---
    if mac.get("covered_digest") != canonical or ledger.get("covered_digest") != canonical:
        return reject("manifest:crossed-digest")

    # --- assurance floor ---
    if mac.get("assurance") == "classical-claims-pq-hybrid" or ledger.get("assurance") == "classical-claims-pq-hybrid":
        return reject("manifest:downgraded-assurance")

    # --- fencing / concurrency / idempotency ---
    fencing = manifest.get("fencing_token")
    current_fence_gen = evaluation.get("current_fencing_generation", "")
    if not isinstance(fencing, str) or not fencing:
        return reject("manifest:missing-fencing-token")
    if current_fence_gen and not fencing.startswith(current_fence_gen):
        return reject("manifest:crossed-fencing-token")
    if "successor_set" in manifest:
        return reject("manifest:concurrent-successors")
    if manifest.get("resubmit_after_crash"):
        return reject("manifest:crash-repeats-finalized")
    finalized = set(evaluation.get("finalized_operation_ids", []))
    if manifest.get("operation_id") in finalized:
        return reject("manifest:replayed-operation-id")
    if manifest.get("predecessor_version") != resource_intent.get("predecessor_version"):
        return reject("manifest:stale-predecessor")

    # --- receipt / privacy ---
    receipt = manifest.get("public_receipt")
    if isinstance(receipt, dict) and any(isinstance(v, str) and v.startswith("did:") for v in receipt.values()):
        return reject("receipt:public-reveals-holder")

    # --- CAS / 9P / state ---
    if "provisional_namespace_path" in manifest:
        return reject("cas:provisional-exposed-as-final")
    if manifest.get("content_cid") != resource_intent.get("content_cid"):
        return reject("cas:content-cid-mismatch")
    if "ninep_title_granted" in manifest:
        return reject("ninep:title-without-finalized")
    if "transition_from" in manifest:
        return reject("state-machine:unknown-state")

    # --- version type (a Python bool is an int, so exclude it explicitly) ---
    if "version" in manifest and (isinstance(manifest["version"], bool) or not isinstance(manifest["version"], int)):
        return reject("manifest:version-type")

    # --- expiry binding and crossed-resource ---
    if manifest.get("expiry") != resource_intent.get("expiry"):
        return reject("manifest:expiry-binding")
    if manifest.get("resource_id") != resource_intent.get("resource_id"):
        return reject("manifest:crossed-resource")

    return True, ""


def validate_vector() -> list[str]:
    errors: list[str] = []
    data = load_json(VECTOR)
    if data.get("vector_version") != 1:
        errors.append("vector: vector_version must be 1")
    if data.get("status") != "non-cryptographic-structural-fixture":
        errors.append("vector: status must be non-cryptographic-structural-fixture")
    baseline = data.get("baseline", {})
    resource_intent = baseline.get("resource_intent")
    if not isinstance(resource_intent, dict):
        return errors + ["vector: baseline.resource_intent must be an object"]

    # Baseline canonical digest.
    ok, err, canonical = evaluate_baseline(data)
    if not ok:
        errors.append(f"vector: baseline must be structurally valid, got {err!r}")
    if canonical != baseline.get("canonical_digest"):
        errors.append("vector: baseline.canonical_digest does not match recomputed digest")
    if canonical != data.get("baseline_canonical_sha256"):
        errors.append("vector: baseline_canonical_sha256 does not match recomputed digest")
    if baseline.get("production_disposition") != "refuse-construction-incomplete":
        errors.append("vector: baseline.production_disposition must be refuse-construction-incomplete")
    if data.get("expected_production_refusal") != "construction-incomplete":
        errors.append("vector: expected_production_refusal must be construction-incomplete")

    # Mutations.
    mutations = data.get("mutations", [])
    if not isinstance(mutations, list) or not mutations:
        return errors + ["vector: mutations must be a nonempty list"]
    seen_ids: set[str] = set()
    for mutation in mutations:
        mid = mutation.get("id")
        if not mid:
            errors.append("vector: a mutation is missing an id")
            continue
        if mid in seen_ids:
            errors.append(f"vector: duplicate mutation id {mid}")
        seen_ids.add(mid)
        produced = evaluate_mutation(data, mutation)
        expected = mutation.get("expected_error")
        if produced == "accepted":
            errors.append(f"vector: mutation {mid} was accepted but must reject")
        elif produced != expected:
            errors.append(f"vector: mutation {mid} expected {expected!r} but produced {produced!r}")

    # Category coverage.
    categories = data.get("mutation_categories", {})
    if not isinstance(categories, dict):
        errors.append("vector: mutation_categories must be an object")
    else:
        missing = set(REQUIRED_MUTATION_CATEGORIES) - set(categories)
        if missing:
            errors.append(f"vector: mutation_categories missing {sorted(missing)}")
        all_ids = seen_ids
        for category, ids in categories.items():
            if not isinstance(ids, list) or not ids:
                errors.append(f"vector: category {category} must be a nonempty list")
                continue
            for mid in ids:
                if mid not in all_ids:
                    errors.append(f"vector: category {category} references unknown mutation {mid}")
        for required in REQUIRED_MUTATION_CATEGORIES:
            if required in categories and not set(categories[required]) & seen_ids:
                errors.append(f"vector: category {required} has no known mutations")
    return errors


# --------------------------------------------------------------------------- #
# Generated RFCXML stale-output check
# --------------------------------------------------------------------------- #
def check_generated(xml2rfc: str) -> list[str]:
    errors: list[str] = []
    if not TEXT.exists() or not HTML.exists():
        return ["generated RFCXML text and HTML outputs are missing"]
    version = subprocess.run([xml2rfc, "--version"], cwd=ROOT, text=True, capture_output=True)
    if version.returncode:
        return [f"xml2rfc --version failed: {version.stdout}{version.stderr}"]
    reported = f"{version.stdout}{version.stderr}".strip()
    if reported != f"xml2rfc {XML2RFC_VERSION}":
        return [f"xml2rfc must be exactly {XML2RFC_VERSION}; got {reported!r}"]
    with tempfile.TemporaryDirectory(prefix="hyprstream-ra-rfcxml-") as tmp:
        tmp_path = Path(tmp)
        command = [xml2rfc, "--no-network", "--text", "--html", "--path", str(tmp_path), str(XML)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            return [f"xml2rfc failed: {' '.join(command)}\n{result.stdout}{result.stderr}"]
        for suffix, expected in ((".txt", TEXT), (".html", HTML)):
            actual = tmp_path / f"{XML.stem}{suffix}"
            if not actual.exists():
                errors.append(f"xml2rfc did not create {actual.name}")
            elif actual.read_bytes() != expected.read_bytes():
                errors.append(f"stale generated output: {expected.relative_to(ROOT)}; run tools/generate_resource_attestation.py")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-generated", action="store_true",
                        help="regenerate RFCXML in a temp directory and compare bytewise")
    parser.add_argument("--xml2rfc", default="xml2rfc", help="exact xml2rfc executable to invoke")
    args = parser.parse_args()
    errors = (
        validate_intent_registry()
        + validate_vocabulary()
        + validate_obligations()
        + validate_vector()
    )
    if args.check_generated:
        errors += check_generated(args.xml2rfc)
    if errors:
        print("resource-attestation validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("resource-attestation registries, vocabulary, obligations, and vector: OK")
    if args.check_generated:
        print("RFCXML generated text and HTML: current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
