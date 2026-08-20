#!/usr/bin/env python3
"""Mechanical validation gate for the v16 RPC request-proof profile freeze.

This gate checks the CDDL, the private-label registry, the credential profile,
and the canonical fixtures *together* — not each in isolation — and fails
closed on any inconsistency. It is the machine-checked half of the Gate-2
freeze (v16 §19, operator disposition 2026-08-19).

What it enforces, end to end:

  1. CDDL validity + structural conformance of every fixture, using a REAL CDDL
     validator (`pycddl`, the Rust `cddl` crate behind a pinned wheel).
  2. Exact frozen private-use values, present and consistent across the CDDL,
     the registry, and the fixtures.
  3. Every frozen cap (1 MiB body, 2 MiB object, aud 128 == the shared
     MAX_SERVICE_DOMAIN_BYTES, signer group <= 255, plan 1..8 x 1..2, suite/kid
     1..64, Nonce 16..64, cti 16, credential_hash 32).
  4. The closed 4-key response-binding map, the two orthogonal enum axes, and
     the "recipient non-null iff encrypted" relation — asserted over the
     fixtures AND proven to be enforced by the CDDL itself (the relation/enum
     negatives are rejected by the validator).
  5. No collision or duplicate across the private-use allocations.
  6. Canonical, reproducible fixtures: deterministic-CBOR decode of every
     vector, digest/size agreement, signature re-verification, and a
     byte-identical regeneration from the seeded generator (no hand-editing).

Usage:
    python3 docs/standards/v16/tools/validate_profile.py

Requires `pycddl==0.3.0` (see requirements.txt). The gate FAILS if it is
missing — the CDDL layer is mandatory, not optional.

pycddl 0.3.0 limitation, handled here: `.size (LO..HI)` byte-length RANGE
controls are mis-evaluated by that version (they are checked as a uint range
against the value, not the byte length). The gate therefore strips ONLY those
range controls for the structural CDDL pass — the exact strip list is printed —
and re-checks every corresponding size cap numerically in Python. Exact
`.size N` controls and `.le` are handled correctly and are left in place.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V16 = HERE.parent
CDDL_PATH = V16 / "hyprstream-proof-cwt.cddl"
REGISTRY_PATH = V16 / "private-label-registry.md"
CREDENTIAL_PATH = V16 / "credential-profile.md"
README_PATH = V16 / "README.md"
CANONICAL_VECTORS_PATH = V16 / "canonical-vectors.md"
VECTORS_DIR = V16 / "vectors"
# Repo root is three `.parent` hops up from V16 (docs/standards/v16 -> repo).
REPO_ROOT = V16.parent.parent.parent
ENVELOPE_RS = REPO_ROOT / "crates" / "hyprstream-rpc" / "src" / "envelope.rs"

# Import the strict deterministic-CBOR decoder from the vector checker so the
# gate and the checker share one decoder.
sys.path.insert(0, str(HERE))
from check_proof_vectors import decode, StrictError  # noqa: E402

# ---- Frozen expectations (Gate-2 §19, 2026-08-19) ------------------------

PROOF_CLAIM_KEYS = {-70001, -70002, -70003, -70004}
CREDENTIAL_CLAIM_KEYS = {-70005, -70006, -70007}  # amendment 10 (integer CWT)
HEADER_PARAMS = {-70100, -70101, -70102, -70103}
KEM_ALG = -70200  # amendment 5, hs-kem-ml-kem-768-v1
UNKNOWN_TEST_KEY = -70050  # deliberately unallocated (N-8)

MAX_SERVICE_DOMAIN_BYTES = 128  # amendment 7 (shared constant)
MAX_BODY_BYTES = 1048576  # 1 MiB
MAX_OBJECT_BYTES = 2097152  # 2 MiB
MAX_SIGNER_GROUP = 255  # value 6
CTI_BYTES = 16
CREDENTIAL_HASH_BYTES = 32
NONCE_MIN, NONCE_MAX = 16, 64
SUITE_KID_MAX = 64
SUITE_CLASSICAL = "hs-cose-sign-ed25519-v1"
SUITE_HYBRID = "hs-cose-sign-ed25519-mldsa65-wns-v1"
ALG_ED25519, ALG_ML_DSA_65 = -19, -49
GROUPS_MAX, COMPONENTS_MAX = 8, 2

# CBOR/COSE labels
H_ALG, H_CRIT, H_KID, H_TYP = 1, 2, 4, 16
H_DOMAIN, H_PLAN, H_GROUP, H_KEYSET = -70100, -70101, -70102, -70103
C_AUD, C_EXP, C_IAT, C_CTI, C_NONCE = 3, 4, 6, 7, 10
C_CREDENTIAL_HASH, C_SCHEMA_ID, C_BODY_BYTES, C_RESPONSE_BINDING = (
    -70001,
    -70002,
    -70003,
    -70004,
)

# Protected typ / hs_domain literals (paired: request vs response).
TYP_REQUEST = "application/vnd.hyprstream.proof+cwt"
TYP_RESPONSE = "application/vnd.hyprstream.response-proof+cwt"
DOMAIN_REQUEST = "hs-rpc-request-proof-v1"
DOMAIN_RESPONSE = "hs-rpc-response-proof-v1"

FAILURES: list[str] = []
SECTION = ""


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(f"[{SECTION}] {msg}")


def section(name: str) -> None:
    global SECTION
    SECTION = name
    print(f"== {name} ==")


# --------------------------------------------------------------------------


def strip_size_ranges(cddl: str) -> tuple[str, list[str]]:
    """Strip ONLY `.size (LO..HI)` byte-length range controls (mis-evaluated by
    pycddl 0.3.0). Returns the stripped text and the exact strips removed."""
    pat = re.compile(r"\.size\s*\(\s*\d+\s*\.\.\s*\d+\s*\)")
    strips = pat.findall(cddl)
    return pat.sub("", cddl), strips


def load_json(name: str) -> dict:
    return json.loads((VECTORS_DIR / name).read_text())


def payload_of(cbor_hex: str) -> bytes:
    obj = decode(bytes.fromhex(cbor_hex))
    return obj[2]  # [protected, unprotected, payload, signature(s)]


def claims_of(cbor_hex: str) -> dict:
    return decode(payload_of(cbor_hex))


def valid_service_domain(s: str) -> bool:
    """Port of validate_service_domain (crates/hyprstream-rpc/src/envelope.rs):
    1..MAX bytes; first byte a lowercase ASCII letter or digit; every byte a
    lowercase ASCII letter, digit, '.', '_', '-', or '/'. The pinned pycddl does
    not enforce the `.regexp` in the CDDL, so this is the mechanical enforcement
    of the shared canonicalization syntax the profile reuses for `aud`."""
    b = s.encode()
    if not b or len(b) > MAX_SERVICE_DOMAIN_BYTES:
        return False
    lower = set(range(0x61, 0x7B))   # a-z
    digit = set(range(0x30, 0x3A))   # 0-9
    if b[0] not in (lower | digit):
        return False
    allowed = lower | digit | {ord(c) for c in "._-/"}
    return all(x in allowed for x in b)


def kids_and_suites(cbor_hex: str) -> tuple[list[bytes], list[str]]:
    """Every kid (bstr) and suite_id (tstr) that appears in one COSE object:
    the protected-header kid, the signature-plan components and their suites,
    the unattributed key-set kids, and the per-signature buckets (COSE_Sign)."""
    obj = decode(bytes.fromhex(cbor_hex))
    prot = decode(obj[0])
    kids: list[bytes] = []
    suites: list[str] = []
    if isinstance(prot.get(H_KID), (bytes, bytearray)):
        kids.append(prot[H_KID])
    for grp in prot.get(H_PLAN, []) or []:
        if isinstance(grp, dict):
            if isinstance(grp.get(2), str):
                suites.append(grp[2])
            for comp in grp.get(3, []) or []:
                if isinstance(comp, dict) and isinstance(comp.get(2), (bytes, bytearray)):
                    kids.append(comp[2])
    for key in prot.get(H_KEYSET, []) or []:
        if isinstance(key, dict) and isinstance(key.get(2), (bytes, bytearray)):
            kids.append(key[2])
    # COSE_Sign per-signature protected buckets.
    if isinstance(obj[3], list) and obj[3] and isinstance(obj[3][0], list):
        for entry in obj[3]:
            try:
                sp = decode(entry[0])
            except Exception:  # noqa: BLE001
                continue
            if isinstance(sp, dict) and isinstance(sp.get(H_KID), (bytes, bytearray)):
                kids.append(sp[H_KID])
    return kids, suites


# --------------------------------------------------------------------------
# 1. CDDL structural validation with a real validator
# --------------------------------------------------------------------------


def gate_cddl(cddl: str, positives, negatives) -> None:
    section("1. CDDL structural validation (pycddl real validator)")
    try:
        from pycddl import Schema, ValidationError
    except ImportError:
        FAILURES.append(
            "[1. CDDL] pycddl is not installed. This gate REQUIRES a real CDDL "
            "validator. Install it: python3 -m pip install -r "
            "docs/standards/v16/tools/requirements.txt"
        )
        return
    from check_proof_vectors import enc  # deterministic re-encoder for mutants

    stripped, strips = strip_size_ranges(cddl)
    print(f"   stripped {len(strips)} bstr .size range control(s) for the "
          f"structural pass (re-checked numerically in section 3):")
    for s in strips:
        print(f"     - {s}")
    # pycddl 0.3.0 (embedding the Rust `cddl` crate 0.9.1) PANICS (unwrap-on-None)
    # validating the AKP/ML-DSA-65 COSE_Key inside the unattributed key set, so
    # the key set is passed opaquely to pycddl for the protected-bucket pass ONLY.
    # It is NOT left unvalidated: section 9 (B4) enforces the full key shape
    # (closed COSE_Key field set, kty/crv or parameter set, exact public-key byte
    # length), the frozen 1..2 element ceiling (B8), exact ordered 1:1
    # correspondence with the plan, and embedded-key signature verification —
    # strictly stronger than the CDDL key-shape rule. The normative CDDL is
    # unchanged.
    stripped = stripped.replace("=> unattributed-key-set", "=> any")
    print("   embedded key set validated by section 9 / B4 (pycddl AKP panic worked around)")

    # Whole CDDL must compile under a real parser.
    try:
        Schema("start = hyprstream-proof-claims\n" + stripped)
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"[1. CDDL] normative CDDL does not compile: {exc}")
        return

    def S(root: str):
        return Schema(f"start = {root}\n" + stripped)

    req_claims = S("hyprstream-proof-claims")
    resp_claims = S("hyprstream-response-proof-claims")
    unatt_claims = S("hyprstream-unattributed-proof-claims")
    sign1_prot = S("proof-sign1-protected")
    sign_body_prot = S("proof-sign-body-protected")
    sig_entry_prot = S("proof-sign-signature-protected")
    plan_schema = S("signature-plan")

    def plan_of(cbor_hex: str):
        return decode(decode(bytes.fromhex(cbor_hex))[0]).get(H_PLAN)

    def classify(pm: dict) -> str:
        if H_KEYSET in pm:
            return "unattributed"
        return "response" if pm.get(H_TYP) == TYP_RESPONSE else "request"

    def claims_schema_for(kind: str):
        return {"response": resp_claims, "unattributed": unatt_claims}.get(kind, req_claims)

    # pycddl 0.3.0 cannot enforce a paired typ×domain choice through a
    # whole-object `bstr .cbor` (choices of `.cbor`-bearing arrays are not
    # descended). The gate therefore decodes each object and validates the
    # PROTECTED BUCKET and the PAYLOAD directly against the paired rules — a
    # strictly stronger check that does enforce the pairing (F1) and the
    # response-only claim invariants (F3).
    for v in positives["vectors"]:
        obj = decode(bytes.fromhex(v["cbor_hex"]))
        prot_bytes, payload = obj[0], obj[2]
        pm = decode(prot_bytes)
        kind = classify(pm)
        pschema = sign1_prot if v["structure"] == "COSE_Sign1" else sign_body_prot
        try:
            pschema.validate_cbor(prot_bytes)
        except ValidationError as exc:
            check(False, f"positive {v['id']} protected bucket fails CDDL: {exc}")
        cschema = claims_schema_for(kind)
        try:
            cschema.validate_cbor(payload)
        except ValidationError as exc:
            check(False, f"positive {v['id']} claims payload ({kind}) fails CDDL: {exc}")
        if v["structure"] == "COSE_Sign":
            for entry in obj[3]:
                try:
                    sig_entry_prot.validate_cbor(entry[0])
                except ValidationError as exc:
                    check(False, f"positive {v['id']} signature bucket fails CDDL: {exc}")

    by_id_pos = {v["id"]: v for v in positives["vectors"]}

    def mutate_prot(pid: str, *, dom=None, typ=None) -> bytes:
        pm = decode(decode(bytes.fromhex(by_id_pos[pid]["cbor_hex"]))[0])
        if dom is not None:
            pm[H_DOMAIN] = dom
        if typ is not None:
            pm[H_TYP] = typ
        return enc(pm)

    # F1 — the typ×hs_domain cross-product (N-4 domain-confusion shape) MUST be
    # rejected by the paired protected rules, for both COSE_Sign1 and the
    # COSE_Sign body bucket.
    f1_cases = [
        ("Sign1 request-typ + response-domain", sign1_prot,
         mutate_prot("P-4", dom=DOMAIN_RESPONSE)),
        ("Sign1 response-typ + request-domain", sign1_prot,
         mutate_prot("P-3", dom=DOMAIN_REQUEST)),
        ("Sign1 response-typ marker on a request domain", sign1_prot,
         mutate_prot("P-4", typ=TYP_RESPONSE)),
        ("Sign body request-typ + response-domain", sign_body_prot,
         mutate_prot("P-5", dom=DOMAIN_RESPONSE)),
    ]
    for label, schema, mbytes in f1_cases:
        try:
            schema.validate_cbor(mbytes)
            check(False, f"F1: {label} is NOT rejected by the CDDL")
        except ValidationError:
            print(f"   F1 rejected by CDDL: {label}")

    # F3 — the response claims rule MUST reject a present Nonce and a non-null
    # credential_hash (the invariants the old alias left unenforced).
    p3_claims = decode(decode(bytes.fromhex(by_id_pos["P-3"]["cbor_hex"]))[2])
    f3_cases = [
        ("response proof carrying a Nonce", {**p3_claims, C_NONCE: bytes(16)}),
        ("response proof with a non-null credential_hash",
         {**p3_claims, C_CREDENTIAL_HASH: bytes(32)}),
    ]
    for label, claims in f3_cases:
        try:
            resp_claims.validate_cbor(enc(claims))
            check(False, f"F3: {label} is NOT rejected by the CDDL")
        except ValidationError:
            print(f"   F3 rejected by CDDL: {label}")

    by_id = {v["id"]: v for v in negatives["vectors"]}

    # Thread #1 (ary-f) — an unattributed proof REQUIRES the server challenge:
    # the N-16 no-Nonce shape must be rejected by the unattributed claims rule
    # (previously it validated against the generic claims set where Nonce is
    # optional). Also prove the every unattributed positive requires it.
    n16 = by_id.get("N-16")
    check(n16 is not None, "N-16 (unattributed no-Nonce) is missing")
    if n16 is not None:
        try:
            unatt_claims.validate_cbor(payload_of(n16["cbor_hex"]))
            check(False, "N-16: unattributed no-Nonce is NOT rejected by the CDDL")
        except ValidationError:
            print("   #1 N-16 unattributed no-Nonce rejected by CDDL (Nonce REQUIRED)")

    # Thread #2 (ary-i) — a cleartext UNARY response_binding carried as a map has
    # no valid encoding (cleartext unary is null); only stream_setup may be a
    # cleartext map. N-27 exercises this and must be CDDL-rejected.
    n27 = by_id.get("N-27")
    check(n27 is not None, "N-27 (cleartext-unary map) is missing")
    if n27 is not None:
        try:
            req_claims.validate_cbor(payload_of(n27["cbor_hex"]))
            check(False, "N-27: cleartext-unary response_binding map is NOT rejected by the CDDL")
        except ValidationError:
            print("   #2 N-27 cleartext-unary map rejected by CDDL (stream_setup only)")

    # Thread suite-plan (ary-137X) — the suite_id is bound to its exact ordered
    # component plan, so a hybrid group with one Ed25519 component (N-28, a
    # hybrid→classical downgrade) and an unknown suite_id (N-12) both deny at the
    # signature-plan level. First confirm every positive plan validates.
    for v in positives["vectors"]:
        pl = plan_of(v["cbor_hex"])
        check(pl is not None, f"{v['id']} has no signature_plan")
        if pl is not None:
            try:
                plan_schema.validate_cbor(enc(pl))
            except ValidationError as exc:
                check(False, f"positive {v['id']} signature_plan fails CDDL: {exc}")
    suite_plan_negs = {
        "N-28": "hybrid suite with a single Ed25519 component (downgrade)",
        "N-12": "unknown suite_id outside the closed suite set",
    }
    for nid, what in suite_plan_negs.items():
        v = by_id.get(nid)
        check(v is not None, f"suite-plan negative {nid} ({what}) is missing")
        if v is None:
            continue
        pl = plan_of(v["cbor_hex"])
        try:
            plan_schema.validate_cbor(enc(pl))
            check(False, f"{nid} ({what}) plan is NOT rejected by the CDDL")
        except ValidationError:
            print(f"   #1 {nid} rejected by CDDL ({what})")

    # Thread B2 (q5d) — every (alg, kid) pair is unique across the whole plan,
    # regardless of group ID (CDDL cannot express this, so it is enforced here
    # and in the vector checker). Every positive plan is unique; N-33 repeats one
    # (alg, kid) across two groups and must be flagged.
    def plan_keys(pl):
        return [(c.get(1), c.get(2)) for g in (pl or []) if isinstance(g, dict)
                for c in (g.get(3) or []) if isinstance(c, dict)]
    for v in positives["vectors"]:
        pk = plan_keys(plan_of(v["cbor_hex"]))
        check(len(pk) == len(set(pk)),
              f"positive {v['id']} plan repeats an (alg, kid) across groups")
    n33 = by_id.get("N-33")
    check(n33 is not None, "N-33 (duplicate (alg,kid) across groups) is missing")
    if n33 is not None:
        pk = plan_keys(plan_of(n33["cbor_hex"]))
        check(len(pk) != len(set(pk)),
              "N-33 plan must repeat one (alg, kid) across two groups")
        print("   B2 N-33: (alg, kid) repeated across two groups — denies by plan-key uniqueness")

    # Thread B6 (cddl:382) — group IDs MUST be unique and strictly ascending
    # across the plan (CDDL cannot express it; enforced in gate + checker). Every
    # positive is strictly ascending; N-40 repeats a group_id, N-41 is out of
    # order — each must be flagged.
    def group_ids(pl):
        return [g.get(1) for g in (pl or []) if isinstance(g, dict)]

    def strictly_ascending_unique(ids):
        return ids == sorted(set(ids)) and len(ids) == len(set(ids))
    for v in positives["vectors"]:
        gids = group_ids(plan_of(v["cbor_hex"]))
        check(strictly_ascending_unique(gids),
              f"positive {v['id']} group IDs must be unique and strictly ascending, got {gids}")
    for nid, what in (("N-40", "duplicate group_id"), ("N-41", "out of ascending order")):
        v = by_id.get(nid)
        check(v is not None, f"group-id negative {nid} ({what}) is missing")
        if v is not None:
            gids = group_ids(plan_of(v["cbor_hex"]))
            check(not strictly_ascending_unique(gids),
                  f"{nid} plan must violate unique-strictly-ascending group IDs, got {gids}")
            print(f"   B6 {nid} rejected ({what}): group IDs {gids}")

    # Thread C4 (gen:1369) — every signature entry MUST cite a group that is in
    # the signed plan. N-20 declares a signature entry whose group is absent from
    # the plan; assert that membership violation mechanically (the checker's
    # membership predicate runs on positives only).
    n20 = by_id.get("N-20")
    check(n20 is not None, "N-20 (signature entry citing a group absent from the plan) is missing")
    if n20 is not None:
        obj = decode(bytes.fromhex(n20["cbor_hex"]))
        comps = set()
        for g in (decode(obj[0]).get(H_PLAN) or []):
            if isinstance(g, dict):
                for c in (g.get(3) or []):
                    if isinstance(c, dict):
                        comps.add((g.get(1), c.get(1), c.get(2)))
        entry_keys = []
        for entry in (obj[3] or []):
            sh = decode(entry[0])
            entry_keys.append((sh.get(H_GROUP), sh.get(H_ALG), sh.get(H_KID)))
        has_out_of_plan = any(ek not in comps for ek in entry_keys)
        check(has_out_of_plan,
              "N-20 must carry a signature entry whose (group, alg, kid) is absent from the plan")
        if has_out_of_plan:
            print("   C4 N-20: a signature entry cites a group absent from the plan (membership denies)")

    # Thread B5 (cddl:115) — an authenticated request proof is credential-bound,
    # so a null credential_hash (no key set) MUST deny structurally. N-15 is that
    # shape and must be rejected by the authenticated request claims rule.
    n15 = by_id.get("N-15")
    check(n15 is not None, "N-15 (authenticated null credential_hash) is missing")
    if n15 is not None:
        try:
            req_claims.validate_cbor(payload_of(n15["cbor_hex"]))
            check(False, "N-15: authenticated null credential_hash is NOT rejected by the CDDL")
        except ValidationError:
            print("   B5 N-15 rejected by CDDL (authenticated request requires a 32-byte credential_hash)")

    # The relation/enum negatives MUST be rejected by the CDDL itself: the
    # machine proof that the closed map, the two enum axes, and the
    # recipient/encryption relation are structural, not prose.
    relation_negs = {
        "N-23": "encrypted binding with null recipient",
        "N-24": "cleartext binding with a recipient",
        "N-25": "response_kind outside its closed enum",
    }
    for nid, what in relation_negs.items():
        v = by_id.get(nid)
        check(v is not None, f"expected negative {nid} ({what}) is missing")
        if v is None:
            continue
        try:
            req_claims.validate_cbor(payload_of(v["cbor_hex"]))
            check(False, f"{nid} ({what}) is NOT rejected by the CDDL")
        except ValidationError:
            print(f"   {nid} correctly rejected by the CDDL ({what})")


# --------------------------------------------------------------------------
# 2. Exact private-use values, consistent across CDDL / registry / fixtures
# --------------------------------------------------------------------------


def gate_values(cddl: str, registry: str, credential: str, positives, negatives) -> None:
    section("2. Exact private-use values and cross-artifact consistency")

    # 2a. CDDL declares the KEM alg and the proof claim keys exactly.
    check(
        re.search(r"alg-hs-kem-ml-kem-768-v1\s*=\s*-70200", cddl) is not None,
        "CDDL must allocate alg-hs-kem-ml-kem-768-v1 = -70200",
    )
    for k in PROOF_CLAIM_KEYS:
        check(f"{k} =>" in cddl.replace(" ", " "), f"CDDL claims map must carry key {k}")
    for k in HEADER_PARAMS:
        check(str(k) in cddl, f"CDDL must reference header param {k}")

    # 2b. Registry documents every frozen value.
    for k in sorted(PROOF_CLAIM_KEYS | CREDENTIAL_CLAIM_KEYS | HEADER_PARAMS | {KEM_ALG}):
        check(str(k) in registry, f"registry must document {k}")
    check(
        "capnp_body_bytes" in registry and "capnp_request_bytes" not in registry,
        "registry -70003 must be renamed capnp_body_bytes (no capnp_request_bytes)",
    )
    check("hs-kem-ml-kem-768-v1" in registry, "registry must name hs-kem-ml-kem-768-v1")

    # 2c. Credential profile documents the integer CWT claim keys and keeps
    #     text names JWT-only.
    for k in CREDENTIAL_CLAIM_KEYS:
        check(str(k) in credential, f"credential profile must document CWT key {k}")

    # 2d. No fixture claims-map uses a key outside the closed proof set, except
    #     the deliberate unknown-key negative (N-8, key -70050).
    allowed = PROOF_CLAIM_KEYS | {C_AUD, C_EXP, C_IAT, C_CTI, C_NONCE}
    for v in positives["vectors"]:
        claims = claims_of(v["cbor_hex"])
        extra = set(claims) - allowed
        check(not extra, f"positive {v['id']} uses non-frozen claim keys {extra}")
    n8 = next((v for v in negatives["vectors"] if v["id"] == "N-8"), None)
    check(n8 is not None, "N-8 (unknown claim key) must exist")
    if n8 is not None:
        keys = set(claims_of(n8["cbor_hex"]))
        check(UNKNOWN_TEST_KEY in keys, "N-8 must carry the unallocated key -70050")


# --------------------------------------------------------------------------
# 3. Frozen caps (numeric, over CDDL constants, fixtures, and Rust constant)
# --------------------------------------------------------------------------


def const_from_cddl(cddl: str, name: str) -> int | None:
    m = re.search(rf"^\s*{re.escape(name)}\s*=\s*(-?\d+)\b", cddl, re.MULTILINE)
    return int(m.group(1)) if m else None


def gate_caps(cddl: str, positives, negatives) -> None:
    section("3. Frozen caps")

    # 3a. CDDL-declared constants. Each cap is regex-pinned so that drift in the
    #     normative CDDL (e.g. a 64 -> 128 kid/suite widening) fails the gate —
    #     the pinned pycddl version cannot enforce byte-length `.size` ranges, so
    #     these text pins plus the numeric fixture checks in 3d are load-bearing.
    check(const_from_cddl(cddl, "max-aud-bytes") == MAX_SERVICE_DOMAIN_BYTES,
          f"CDDL max-aud-bytes must be {MAX_SERVICE_DOMAIN_BYTES}")
    check(const_from_cddl(cddl, "max-body-bytes") == MAX_BODY_BYTES,
          "CDDL max-body-bytes must be 1048576")
    check(const_from_cddl(cddl, "alg-hs-kem-ml-kem-768-v1") == KEM_ALG,
          "CDDL KEM alg must be -70200")
    check(re.search(r"tstr\s*\.size\s*\(1\.\.128\)", cddl) is not None,
          "canonical-service-domain must be tstr .size (1..128)")
    check(re.search(r"capnp-body-bytes\s*=\s*bstr\s*\.size\s*\(0\.\.1048576\)", cddl) is not None,
          "capnp-body-bytes must be bstr .size (0..1048576)")
    check(re.search(r"logical-signer-group\s*=\s*uint\s*\.le\s*255", cddl) is not None,
          "logical-signer-group must be uint .le 255")
    check(re.search(r"2097152", cddl) is not None,
          "CDDL must state the 2 MiB total-object cap (2097152)")
    check(re.search(r"signature-plan\s*=\s*\[\s*1\*8", cddl) is not None,
          "signature-plan must cap at 1*8 groups")
    # Suite-bound component plans (finding ary-137X): each suite fixes its exact
    # ordered component list, so the suite_id and component count are not
    # independent. Pin the two suite-specific groups and their exact plans.
    check(re.search(r"signer-group\s*=\s*signer-group-classical\s*/\s*signer-group-hybrid", cddl) is not None,
          "signer-group must be the closed choice of the two suite-specific groups")
    check("3 => [ signature-component-ed25519 ]," in cddl,
          "classical group must bind exactly one Ed25519 component")
    check("3 => [ signature-component-ed25519, signature-component-mldsa65 ]," in cddl,
          "hybrid group must bind exactly two ordered components (Ed25519 then ML-DSA-65)")
    # kid 1..64 cap (F2): pin the CDDL text so a widening to 128 fails.
    check(re.search(rf"kid\s*=\s*bstr\s*\.size\s*\(1\.\.{SUITE_KID_MAX}\)", cddl) is not None,
          f"CDDL kid must be bstr .size (1..{SUITE_KID_MAX})")

    # 3b. The shared Rust constant the artifact points to.
    if ENVELOPE_RS.exists():
        m = re.search(r"MAX_SERVICE_DOMAIN_BYTES:\s*usize\s*=\s*(\d+)", ENVELOPE_RS.read_text())
        check(m is not None and int(m.group(1)) == MAX_SERVICE_DOMAIN_BYTES,
              f"envelope.rs MAX_SERVICE_DOMAIN_BYTES must equal {MAX_SERVICE_DOMAIN_BYTES}")
        print(f"   envelope.rs MAX_SERVICE_DOMAIN_BYTES = {m.group(1) if m else '?'}")
    else:
        print("   NOTE: envelope.rs not found; skipping shared-constant tie-in "
              "(docs extracted standalone).")

    # The one numeric object-cap path, applied to fixtures (3c) and to the
    # constructed over-cap object (3g).
    def object_within_cap(raw: bytes) -> bool:
        return len(raw) <= MAX_OBJECT_BYTES

    # 3c. Every positive fixture respects the caps.
    for v in positives["vectors"]:
        raw = bytes.fromhex(v["cbor_hex"])
        check(object_within_cap(raw), f"{v['id']} exceeds the 2 MiB object cap")
        claims = claims_of(v["cbor_hex"])
        aud = claims[C_AUD]
        check(1 <= len(aud.encode()) <= MAX_SERVICE_DOMAIN_BYTES,
              f"{v['id']} aud length out of 1..128")
        # Finding ary-137b: aud must satisfy the shared service-domain syntax,
        # not merely the length cap.
        check(valid_service_domain(aud),
              f"{v['id']} aud {aud!r} violates the canonical service-domain syntax")
        check(len(claims[C_CTI]) == CTI_BYTES, f"{v['id']} cti must be 16 bytes")
        ch = claims[C_CREDENTIAL_HASH]
        check(ch is None or len(ch) == CREDENTIAL_HASH_BYTES,
              f"{v['id']} credential_hash must be null or 32 bytes")
        body = claims[C_BODY_BYTES]
        check(len(body) <= MAX_BODY_BYTES, f"{v['id']} capnp body exceeds 1 MiB")
        if C_NONCE in claims:
            check(NONCE_MIN <= len(claims[C_NONCE]) <= NONCE_MAX,
                  f"{v['id']} Nonce length out of 16..64")

    # 3d. Numeric kid / suite_id caps over every positive fixture (F2: the pinned
    #     pycddl version strips these size ranges, so they are enforced here).
    for v in positives["vectors"]:
        kids, suites = kids_and_suites(v["cbor_hex"])
        for kid in kids:
            check(1 <= len(kid) <= SUITE_KID_MAX,
                  f"{v['id']} kid length {len(kid)} out of 1..{SUITE_KID_MAX}")
        for suite in suites:
            check(1 <= len(suite.encode()) <= SUITE_KID_MAX,
                  f"{v['id']} suite_id length {len(suite.encode())} out of 1..{SUITE_KID_MAX}")

    # 3e. The size-cap boundary negatives deny by their numeric rule: prove each
    #     over-limit fixture actually exceeds the cap it exercises (F2). This is
    #     what makes N-12/N-13/N-26 mechanically load-bearing rather than merely
    #     labelled — cap drift to 128 would let these become in-range and the
    #     matching CDDL-text pin in 3a would fail.
    by_id = {v["id"]: v for v in negatives["vectors"]}

    def over_cap(nid, extract, cap, what):
        v = by_id.get(nid)
        check(v is not None, f"boundary negative {nid} ({what}) is missing")
        if v is None:
            return
        try:
            val = extract(v["cbor_hex"])
        except Exception as exc:  # noqa: BLE001
            check(False, f"{nid}: could not extract {what}: {exc}")
            return
        check(val > cap, f"{nid} must exceed the {what} cap {cap} (got {val})")
        if val > cap:
            print(f"   {nid}: {what} = {val} bytes > cap {cap} (denies by numeric rule)")

    # All three stripped size-limit negatives (finding arYj7) are exercised by
    # their numeric rule, not merely measured over positives: N-12 suite_id,
    # N-13 kid, N-26 aud.
    over_cap("N-12", lambda h: max((len(s.encode()) for s in kids_and_suites(h)[1]), default=0),
             SUITE_KID_MAX, "suite_id")
    over_cap("N-13", lambda h: max((len(k) for k in kids_and_suites(h)[0]), default=0),
             SUITE_KID_MAX, "kid")
    over_cap("N-26", lambda h: len(claims_of(h)[C_AUD].encode()),
             MAX_SERVICE_DOMAIN_BYTES, "aud")

    # 3f. aud lexical syntax (finding ary-137b): the CDDL declares the shared
    #     service-domain `.regexp`; the pinned pycddl does not enforce it, so pin
    #     the CDDL text and assert the causal negatives fail the ported syntax.
    check('.regexp "[a-z0-9][a-z0-9._/-]*"' in cddl,
          "canonical-service-domain must carry the shared service-domain .regexp")
    for nid, why in (("N-29", "uppercase byte"), ("N-30", "illegal first byte")):
        v = by_id.get(nid)
        check(v is not None, f"aud-syntax negative {nid} ({why}) is missing")
        if v is None:
            continue
        aud = claims_of(v["cbor_hex"])[C_AUD]
        check(not valid_service_domain(aud),
              f"{nid} aud {aud!r} must violate the service-domain syntax ({why})")
        print(f"   #2 {nid}: aud {aud!r} rejected by service-domain syntax ({why})")

    # 3g. Total-object cap (finding 4D71 / q5i): the complete-object CDDL cannot
    #     bound the signature byte strings, so the 2 MiB cap is a validator-side
    #     numeric check. Build a STRUCTURALLY VALID oversized object (a real
    #     COSE_Sign1 whose signature bstr is enlarged past 2 MiB — not trailing
    #     padding the strict decoder would reject) and run it through the SAME
    #     object-cap path the fixtures use. It must decode cleanly (so size is a
    #     distinct cause from trailing-data) yet be rejected BY SIZE.
    from check_proof_vectors import enc as _enc  # deterministic re-encoder
    sign1 = next((v for v in positives["vectors"] if v["structure"] == "COSE_Sign1"), None)
    check(sign1 is not None, "a COSE_Sign1 positive is needed to build the over-cap object")
    if sign1 is not None:
        # Enlarge the signature bstr to a FIXED size ~2.2 MiB — a fixed target,
        # NOT one sized relative to MAX_OBJECT_BYTES, so `object_within_cap` is a
        # genuine comparison (raising the cap above this size would ACCEPT it and
        # turn the check red) rather than an arithmetic truth.
        obj = decode(bytes.fromhex(sign1["cbor_hex"]))  # [protected, {}, payload, sig]
        big_sig = obj[3] + b"\x00" * (2_300_000 - len(obj[3]))  # valid CBOR bstr, fixed length
        oversized = _enc([obj[0], obj[1], obj[2], big_sig])
        # It is well-formed CBOR: the strict decoder accepts the STRUCTURE
        # (4-element array, no trailing data) — size is a separate concern.
        try:
            dec = decode(oversized)
            structurally_valid = isinstance(dec, list) and len(dec) == 4
        except StrictError:
            structurally_valid = False
        check(structurally_valid,
              "the over-cap object must be structurally valid (size is a distinct cause from decode)")
        check(len(oversized) > MAX_OBJECT_BYTES,
              "the constructed object must exceed the 2 MiB cap by its fixed size")
        check(not object_within_cap(oversized),
              "the numeric object-cap must reject a structurally-valid object over 2 MiB")
        print(f"   3g over-cap object: {len(oversized)} bytes (fixed), valid structure, "
              f"rejected by size (> {MAX_OBJECT_BYTES})")

    # 3h. Byte-range coverage meta-guard (E1). Every `.size (LO..HI)` range that
    #     strip_size_ranges() removes for the pycddl pass MUST have a causal
    #     boundary negative (or an explicit mechanical justification) for each
    #     violable end, so a newly added stripped range cannot drip-feed without
    #     coverage. Discover the ranges straight from the CDDL and require an
    #     entry per rule; a mismatch (new range, or drifted LO/HI) fails here.
    range_rules = {
        m.group(1): (int(m.group(2)), int(m.group(3)))
        for m in re.finditer(
            r"(?m)^\s*([a-z0-9-]+)\s*=.*?\.size\s*\(\s*(\d+)\s*\.\.\s*(\d+)\s*\)", cddl
        )
    }

    def kid_extremum(hex_, which):
        kids, _ = kids_and_suites(hex_)
        lens = [len(k) for k in kids] or [None]
        return (max if which == "max" else min)(lens)

    # For each stripped range: how to measure the boundary field, and which
    # vector (or justification) covers each violable end.
    RANGE_COVERAGE = {
        "canonical-service-domain": {
            "field": lambda h: len(claims_of(h)[C_AUD].encode()),
            "upper": "N-26",
            # lower end is length 0, which also fails the service-domain .regexp
            # (first byte must be [a-z0-9]); it is subsumed by the syntax rule
            # (N-29/N-30), so no independent length-only vector is meaningful.
            "lower": ("syntax-subsumed", lambda: not valid_service_domain("")),
        },
        "kid": {
            "field": lambda h: kid_extremum(h, "max"),
            "field_lower": lambda h: kid_extremum(h, "min"),
            "upper": "N-13",
            "lower": "N-47",
        },
        "server-challenge": {
            "field": lambda h: len(claims_of(h).get(C_NONCE, b"")),
            "lower": "N-45",
            "upper": "N-46",
        },
        "capnp-body-bytes": {
            "field": lambda h: len(claims_of(h)[C_BODY_BYTES]),
            "upper": "N-44",
            # lower end is length 0 — the minimum, so no shorter value exists.
            "lower": ("min-is-zero", lambda: True),
        },
    }

    check(set(range_rules) == set(RANGE_COVERAGE),
          f"every stripped .size range must have boundary coverage; "
          f"CDDL has {sorted(range_rules)}, coverage maps {sorted(RANGE_COVERAGE)}")

    for name, (lo, hi) in sorted(range_rules.items()):
        cov = RANGE_COVERAGE.get(name)
        if cov is None:
            continue  # already flagged by the set-equality guard above
        for end, target, boundary in (("lower", cov.get("lower"), lo - 1),
                                       ("upper", cov.get("upper"), hi + 1)):
            if isinstance(target, tuple):  # justified non-vector boundary
                why, predicate = target
                check(predicate(), f"{name} {end} boundary justification '{why}' does not hold")
                print(f"   3h {name} {end} boundary: {why} (no independent vector)")
                continue
            check(target is not None, f"{name} {end} boundary has no coverage")
            v = by_id.get(target)
            check(v is not None, f"{name} {end} boundary negative {target} is missing")
            if v is None:
                continue
            extractor = cov["field_lower"] if (end == "lower" and "field_lower" in cov) else cov["field"]
            try:
                measured = extractor(v["cbor_hex"])
            except Exception as exc:  # noqa: BLE001
                check(False, f"{target}: could not measure {name} {end} boundary: {exc}")
                continue
            check(measured == boundary,
                  f"{target} must sit exactly on the {name} {end} boundary "
                  f"({'LO-1' if end == 'lower' else 'HI+1'} = {boundary}); measured {measured}")
            if measured == boundary:
                print(f"   3h {name} {end} boundary: {target} field length {measured} "
                      f"(range {lo}..{hi})")


# --------------------------------------------------------------------------
# 4. Closed response map, orthogonal enums, recipient/encryption relation
# --------------------------------------------------------------------------


def gate_response_binding(positives) -> None:
    section("4. Response binding: closed map, enum axes, recipient relation")
    seen_encrypted = seen_cleartext = seen_stream = False
    for v in positives["vectors"]:
        claims = claims_of(v["cbor_hex"])
        rb = claims.get(C_RESPONSE_BINDING)
        if rb is None:
            continue
        check(set(rb.keys()) == {1, 2, 3, 4},
              f"{v['id']} response_binding must be the closed 4-key map, got {set(rb.keys())}")
        kind, protection, recipient = rb.get(2), rb.get(3), rb.get(4)
        check(kind in (1, 2), f"{v['id']} response_kind must be 1 or 2")
        check(protection in (1, 2), f"{v['id']} protection_mode must be 1 or 2")
        if protection == 2:  # encrypted
            check(isinstance(recipient, dict),
                  f"{v['id']} encrypted binding must carry a recipient")
            if isinstance(recipient, dict):
                check(recipient.get(1) == KEM_ALG,
                      f"{v['id']} recipient alg must be -70200")
                check(isinstance(recipient.get(2), bytes) and len(recipient[2]) == 1184,
                      f"{v['id']} recipient must carry a 1184-byte ML-KEM-768 key")
            seen_encrypted = True
        else:  # cleartext
            check(recipient is None,
                  f"{v['id']} cleartext binding must carry a null recipient")
            # A cleartext binding is streamed-but-not-encrypted; cleartext unary
            # is the null encoding, so a cleartext map must be stream_setup.
            check(kind == 2,
                  f"{v['id']} cleartext response_binding must be stream_setup "
                  "(cleartext unary is encoded as null, not a map)")
            seen_cleartext = True
        if kind == 2:
            seen_stream = True
    check(seen_encrypted, "an encrypted response_binding fixture must exist (P-4)")
    check(seen_cleartext, "a cleartext response_binding fixture must exist (P-6)")
    check(seen_stream, "a stream_setup response_kind fixture must exist (P-6)")


# --------------------------------------------------------------------------
# 5. Collision / duplicate allocation review
# --------------------------------------------------------------------------


def gate_collisions() -> None:
    section("5. Collision / duplicate allocation review")
    blocks = {
        "proof-claims": PROOF_CLAIM_KEYS,
        "credential-claims": CREDENTIAL_CLAIM_KEYS,
        "header-params": HEADER_PARAMS,
        "kem-alg": {KEM_ALG},
    }
    all_vals = [x for s in blocks.values() for x in s]
    check(len(all_vals) == len(set(all_vals)),
          "duplicate value across private-use allocations")
    # CWT claim keys (proof + credential) share one IANA namespace: must be disjoint.
    check(PROOF_CLAIM_KEYS.isdisjoint(CREDENTIAL_CLAIM_KEYS),
          "proof and credential CWT claim keys collide")
    check(UNKNOWN_TEST_KEY not in set(all_vals),
          "the N-8 unknown-key -70050 must not be an allocated value")
    # Contiguity/collision-free: credential keys are the next block after proof.
    check(CREDENTIAL_CLAIM_KEYS == {-70005, -70006, -70007},
          "credential CWT keys must be the next collision-free block -70005..-70007")
    print("   allocations disjoint; -70050 unallocated; credential block -70005..-70007")


# --------------------------------------------------------------------------
# 6. Canonical, reproducible fixtures
# --------------------------------------------------------------------------


def gate_canonical(positives, negatives) -> None:
    section("6. Canonical fixtures: deterministic CBOR, digests, signatures, regen")

    # 6a. Deterministic-CBOR decode + digest/size for every vector.
    for group in (positives["vectors"], negatives["vectors"]):
        for v in group:
            raw = bytes.fromhex(v["cbor_hex"])
            check(hashlib.sha256(raw).hexdigest() == v["sha256"], f"{v['id']} sha256 mismatch")
            check(len(raw) == v["size_bytes"], f"{v['id']} size mismatch")
            try:
                decode(raw)  # strict: rejects indefinite lengths, tags, floats, unsorted keys
            except StrictError:
                # Negative vectors deliberately violate deterministic encoding;
                # their whole-object may not decode. Positives MUST decode.
                if v in positives["vectors"]:
                    check(False, f"positive {v['id']} is not deterministic CBOR")

    # 6b. Signature + plan re-verification via the shared checker.
    r = subprocess.run(
        [sys.executable, str(HERE / "check_proof_vectors.py"), str(VECTORS_DIR)],
        capture_output=True, text=True,
    )
    check(r.returncode == 0, f"check_proof_vectors failed:\n{r.stdout}\n{r.stderr}")
    print(f"   {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'checker ran'}")

    # 6c. Byte-identical regeneration proves the fixtures are not hand-edited.
    with tempfile.TemporaryDirectory() as tmp:
        gen = subprocess.run(
            [sys.executable, str(HERE / "gen_proof_vectors.py"), tmp],
            capture_output=True, text=True,
        )
        if gen.returncode != 0:
            print("   NOTE: regeneration skipped/failed (needs OpenSSL ML-DSA-65): "
                  f"{gen.stderr.strip()[:120]}")
            check(False, "generator did not run for the reproducibility check")
        else:
            for name in ("proof-v1-keys.json", "proof-v1-positive.json", "proof-v1-negative.json",
                         "proof-v1-thumbprints.json", "proof-v1-credentials.json"):
                a = (VECTORS_DIR / name).read_bytes()
                b = (Path(tmp) / name).read_bytes()
                check(a == b, f"{name} is not byte-identical to a fresh regeneration "
                              "(hand-edited or generator drift)")
            print("   fixtures regenerate byte-identically from the seeded generator")


# --------------------------------------------------------------------------
# 7. Type-confusion vector quality (N-1 valid credential, N-2 correct label)
# --------------------------------------------------------------------------


def gate_type_confusion(negatives) -> None:
    section("7. Type-confusion vectors (N-1 valid credential, N-2 label)")
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from check_proof_vectors import enc

    keys = load_json("proof-v1-keys.json")["keys"]
    issuer = next((k for k in keys["ed25519"] if k["kid_ascii"] == "issuer-ed25519-1"), None)
    check(issuer is not None, "issuer test key (issuer-ed25519-1) must exist in the key set")
    by_id = {v["id"]: v for v in negatives["vectors"]}

    # N-1 (ary-137e): a GENUINE credential — typed application/cwt (not a proof
    # type) and its issuer signature actually verifies — presented in the proof
    # slot. This proves the vector exercises credential-in-proof-slot rejection
    # of a well-formed credential, not merely a malformed token.
    n1 = by_id.get("N-1")
    check(n1 is not None, "N-1 must exist")
    if n1 is not None and issuer is not None:
        obj = decode(bytes.fromhex(n1["cbor_hex"]))
        pm = decode(obj[0])
        check(pm.get(H_TYP) == "application/cwt",
              "N-1 must be typed application/cwt (a credential encoding)")
        check(pm.get(H_TYP) not in (TYP_REQUEST, TYP_RESPONSE),
              "N-1 typ must not be a request/response proof type")
        tbs = enc(["Signature1", obj[0], b"", obj[2]])
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(issuer["public_hex"]))
        try:
            pub.verify(obj[3], tbs)
            print("   N-1 issuer signature verifies (typ application/cwt)")
        except InvalidSignature:
            check(False, "N-1 issuer signature does not verify — not a well-formed credential")
        # Finding 4D7u: N-1 must be PROFILE-VALID — cnf PoP binding, tenant
        # (-70005), clearance (-70006) — so only its presentation slot is wrong.
        n1c = decode(obj[2])
        cnf = n1c.get(8)
        check(isinstance(cnf, dict) and isinstance(cnf.get(1), dict),
              "N-1 must carry a cnf (8) PoP binding with a COSE_Key confirmation")
        if isinstance(cnf, dict) and isinstance(cnf.get(1), dict):
            ck = cnf[1]
            check(ck.get(1) == 1 and ck.get(3) == -19 and isinstance(ck.get(-2), (bytes, bytearray)),
                  "N-1 cnf must be a valid OKP/Ed25519 COSE_Key (a PoP key)")
        check(isinstance(n1c.get(-70005), str), "N-1 must carry tenant (-70005)")
        check(-70006 in n1c, "N-1 must carry clearance (-70006)")
        # v16 credentials are Reusable-only: there is no use-profile field, and
        # -70008 is unallocated (OneShotTransaction deferred to a future
        # amendment). A v16 credential carries neither.
        check(-70008 not in n1c, "N-1 must carry no use-profile field (v16 is Reusable-only; -70008 unallocated)")
        print("   #2 N-1 is a profile-valid v16 Reusable credential "
              "(cnf PoP + tenant -70005 + clearance -70006; no use-profile field)")

    # N-2 (ary-137m): the two-entry COSE_Sign proof presented as a credential.
    # Its structure label must be COSE_Sign and its bytes must be a COSE_Sign.
    n2 = by_id.get("N-2")
    check(n2 is not None, "N-2 must exist")
    if n2 is not None:
        check(n2["structure"] == "COSE_Sign", "N-2 structure label must be COSE_Sign")
        obj = decode(bytes.fromhex(n2["cbor_hex"]))
        check(isinstance(obj[3], list) and bool(obj[3]) and isinstance(obj[3][0], list),
              "N-2 bytes must be a COSE_Sign (a signatures array), matching its label")
        print("   #4 N-2 labelled COSE_Sign, matching its two-entry signature array")


# --------------------------------------------------------------------------
# 8. Response-binding field-for-field equality (bound response proofs)
# --------------------------------------------------------------------------


def gate_response_binding_equality(positives, negatives) -> None:
    section("8. Response-binding equality + schema-ID binding (findings 4D78, q5V)")
    by_id = {v["id"]: v for v in positives["vectors"]}
    by_id.update({v["id"]: v for v in negatives["vectors"]})

    def rb_of(vid: str):
        return claims_of(by_id[vid]["cbor_hex"]).get(C_RESPONSE_BINDING)

    def is_response(vid: str) -> bool:
        return decode(decode(bytes.fromhex(by_id[vid]["cbor_hex"]))[0]).get(H_TYP) == TYP_RESPONSE

    # B1 (q5V): for a bound response proof, claim -70002 (response root type id)
    # MUST equal the realized response_binding root_type_id (key 1) — the two
    # denote the same schema commitment, and nothing else ties them. Every bound
    # response positive satisfies it; N-31 changes only -70002 and must violate it.
    for v in positives["vectors"]:
        if not is_response(v["id"]):
            continue
        claims = claims_of(v["cbor_hex"])
        rb = claims.get(C_RESPONSE_BINDING)
        if isinstance(rb, dict):
            check(claims.get(C_SCHEMA_ID) == rb.get(1),
                  f"{v['id']} response -70002 must equal response_binding root_type_id")
    n31 = by_id.get("N-31")
    check(n31 is not None, "N-31 (response schema-binding mismatch) is missing")
    if n31 is not None:
        c = claims_of(n31["cbor_hex"])
        rb = c.get(C_RESPONSE_BINDING)
        check(isinstance(rb, dict) and c.get(C_SCHEMA_ID) != rb.get(1),
              "N-31 must carry -70002 != response_binding root_type_id (binding otherwise equal)")
        # And the binding itself is still equal to its originating request.
        if n31.get("originating_request") in by_id:
            check(rb == rb_of(n31["originating_request"]),
                  "N-31 response_binding must still equal the originating request (only -70002 differs)")
        print("   B1 N-31: -70002 != binding root_type_id (binding still equal) — denies by schema binding")

    # A response proof's realized response_binding MUST equal the originating
    # request's map field-for-field. The bound positive (P-7) equals it; the
    # mismatch negative (N-32) differs. Both carry the originating request id, so
    # the suite tests equality against the actual request, not only map shape.
    bound = next((v for v in positives["vectors"] if v.get("originating_request")), None)
    check(bound is not None, "a bound response-proof positive (originating_request) must exist")
    if bound is not None:
        req = bound["originating_request"]
        check(req in by_id, f"{bound['id']} originating_request {req} must be a known vector")
        if req in by_id:
            check(rb_of(bound["id"]) == rb_of(req) and rb_of(bound["id"]) is not None,
                  f"{bound['id']} response_binding must equal request {req} field-for-field")
            print(f"   #4 {bound['id']} binding == {req} binding (field-for-field bound)")

    mismatch = by_id.get("N-32")   # the field-for-field binding mismatch negative
    check(mismatch is not None, "N-32 (response_binding mismatch) must exist")
    if mismatch is not None:
        req = mismatch["originating_request"]
        check(req in by_id, f"{mismatch['id']} originating_request {req} must be a known vector")
        if req in by_id:
            a, b = rb_of(mismatch["id"]), rb_of(req)
            check(a is not None and b is not None and a != b,
                  f"{mismatch['id']} response_binding must MISMATCH request {req} field-for-field")
            print(f"   #4 {mismatch['id']} binding != {req} binding (mismatch denies)")

    # C5 (cddl:1390) — a response proof's cti (claim 7) MUST echo the originating
    # request's request_id. N-22 carries the originating-request id and mutates
    # cti, so it must contextually mismatch (mirroring N-31/N-32).
    def cti_of(vid: str):
        return claims_of(by_id[vid]["cbor_hex"]).get(C_CTI)
    n22 = by_id.get("N-22")
    check(n22 is not None, "N-22 (response cti mismatch) must exist")
    if n22 is not None:
        req = n22.get("originating_request")
        check(req in by_id, f"N-22 originating_request {req!r} must be a known vector")
        if req in by_id:
            mism = (cti_of("N-22") is not None and cti_of(req) is not None
                    and cti_of("N-22") != cti_of(req))
            check(mism, f"N-22 cti must MISMATCH the originating request {req} request_id")
            if mism:
                print(f"   C5 N-22 cti != {req} request_id (contextual mismatch denies)")


# --------------------------------------------------------------------------
# 9. Unattributed key-set correspondence + embedded-key verification (B4)
# --------------------------------------------------------------------------


def gate_unattributed_keyset(positives, negatives) -> None:
    section("9. Unattributed key-set correspondence + embedded-key verification (B4)")
    from check_proof_vectors import unattributed_keyset_correspondence
    from check_proof_vectors import enc as _enc
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    def components_of(plan):
        return [(g.get(1), c.get(1), c.get(2)) for g in (plan or []) if isinstance(g, dict)
                for c in (g.get(3) or []) if isinstance(c, dict)]

    by_id = {v["id"]: v for v in positives["vectors"]}
    by_id.update({v["id"]: v for v in negatives["vectors"]})

    # Unattributed positives: the embedded key set MUST correspond 1:1 in order
    # to the plan. (Signature verification against the embedded keys is done by
    # check_proof_vectors in section 6.)
    for v in positives["vectors"]:
        pm = decode(decode(bytes.fromhex(v["cbor_hex"]))[0])
        if H_KEYSET not in pm:
            continue
        comps = components_of(pm.get(H_PLAN))
        emb, err = unattributed_keyset_correspondence(pm[H_KEYSET], comps)
        check(err is None, f"{v['id']} unattributed key set fails correspondence: {err}")
        check(emb is not None and len(emb) == len(comps),
              f"{v['id']} embedded key set must resolve one key per plan component")
        if emb is not None:
            print(f"   {v['id']} key set corresponds 1:1 to the plan ({len(emb)} embedded keys)")

    # Correspondence-failure negatives: surplus (N-18), duplicate (N-37),
    # reordered (N-38), and over-the-1..2-ceiling (N-42, B8) each break the
    # ordered/bounded correspondence.
    for nid, what in (("N-18", "surplus key"), ("N-37", "duplicate element"),
                      ("N-38", "reordered elements"), ("N-42", "over the 1..2 ceiling")):
        v = by_id.get(nid)
        check(v is not None, f"unattributed-keyset negative {nid} ({what}) is missing")
        if v is None:
            continue
        pm = decode(decode(bytes.fromhex(v["cbor_hex"]))[0])
        _emb, err = unattributed_keyset_correspondence(pm.get(H_KEYSET), components_of(pm.get(H_PLAN)))
        check(err is not None, f"{nid} ({what}) must FAIL key-set correspondence")
        if err is not None:
            print(f"   B4 {nid} rejected by correspondence ({what}): {err}")

    # Embedded-key verification failure: N-35 keeps a well-formed key set (right
    # kid/alg/crv) but a DIFFERENT public key, so its signature must not verify
    # against the embedded key. Embedded keys are keyed by (alg, kid) (B7).
    n35 = by_id.get("N-35")
    check(n35 is not None, "unattributed-keyset negative N-35 (mismatched key) is missing")
    if n35 is not None:
        obj = decode(bytes.fromhex(n35["cbor_hex"]))
        pm = decode(obj[0])
        emb, err = unattributed_keyset_correspondence(pm.get(H_KEYSET), components_of(pm.get(H_PLAN)))
        check(err is None, f"N-35 key set must be shape-valid (only the key bytes differ): {err}")
        akid = (pm.get(H_ALG), pm.get(H_KID))
        if emb is not None and emb.get(akid) is not None:
            tbs = _enc(["Signature1", obj[0], b"", obj[2]])
            try:
                Ed25519PublicKey.from_public_bytes(emb[akid]).verify(obj[3], tbs)
                check(False, "N-35 signature must NOT verify against its (mismatched) embedded key")
            except InvalidSignature:
                print("   B4 N-35 rejected: signature does not verify against the embedded key")


# --------------------------------------------------------------------------
# C2. Documented vector counts agree with the manifests
# --------------------------------------------------------------------------


def gate_readme_counts(positives, negatives) -> None:
    section("C2. Documented vector counts match the manifests")
    npos, nneg = len(positives["vectors"]), len(negatives["vectors"])
    for label, path in (("README.md", README_PATH), ("canonical-vectors.md", CANONICAL_VECTORS_PATH)):
        text = path.read_text()
        ok = False
        # Accept either "N positive, M negative" or the two-row markdown table
        # forms ("N vectors that MUST verify" / "M vectors that MUST deny").
        m = re.search(r"(\d+)\s+positive,\s*(\d+)\s+negative", text)
        if m and (int(m.group(1)), int(m.group(2))) == (npos, nneg):
            ok = True
        pmatch = re.search(r"(\d+)\s+vectors that MUST verify", text)
        nmatch = re.search(r"(\d+)\s+vectors that MUST deny", text)
        if pmatch and nmatch and (int(pmatch.group(1)), int(nmatch.group(1))) == (npos, nneg):
            ok = True
        check(ok, f"{label} vector counts must read {npos} positive / {nneg} negative")
        if ok:
            print(f"   {label} counts agree ({npos} positive / {nneg} negative)")


# --------------------------------------------------------------------------
# 10. Replay-namespace thumbprints frozen to bytes (C1)
# --------------------------------------------------------------------------


def gate_replay_thumbprints(cddl: str, positives) -> None:
    section("10. Replay-namespace thumbprints (C1)")
    import hashlib
    from check_proof_vectors import enc as _enc

    tp_path = VECTORS_DIR / "proof-v1-thumbprints.json"
    check(tp_path.exists(), "proof-v1-thumbprints.json must exist (C1 replay thumbprint vector)")
    if not tp_path.exists():
        return
    tp = json.loads(tp_path.read_text())
    sep = tp["domain_separators"]

    # The two domain separators are frozen literals in the normative CDDL.
    check(f'replay-domain-authenticated = "{sep["authenticated"]}"' in cddl,
          "CDDL must freeze the authenticated replay domain separator")
    check(f'replay-domain-key-set       = "{sep["key_set"]}"' in cddl,
          "CDDL must freeze the key-set replay domain separator")
    check(sep["authenticated"] != sep["key_set"], "the two replay domain separators must be distinct")

    # Authenticated primary thumbprint: recompute from the frozen inputs.
    a = tp["authenticated"]
    auth_pre = _enc([sep["authenticated"], a["suite_id"],
                     [bytes.fromhex(h) for h in a["component_public_keys_hex"]],
                     a["enrollment_epoch"]])
    check(auth_pre.hex() == a["preimage_hex"], "authenticated preimage encoding drifted")
    check(hashlib.sha256(auth_pre).hexdigest() == a["thumbprint_sha256"],
          "authenticated replay thumbprint does not match its inputs")

    # Unattributed thumbprint: recompute from P-1's plan + embedded key set.
    p1 = next((v for v in positives["vectors"] if v["id"] == tp["unattributed"]["from_vector"]), None)
    check(p1 is not None, "unattributed thumbprint source vector must exist")
    if p1 is not None:
        hdr = decode(decode(bytes.fromhex(p1["cbor_hex"]))[0])
        ks_pre = _enc([sep["key_set"], hdr[H_PLAN], hdr[H_KEYSET]])
        check(ks_pre.hex() == tp["unattributed"]["preimage_hex"], "unattributed preimage encoding drifted")
        check(hashlib.sha256(ks_pre).hexdigest() == tp["unattributed"]["thumbprint_sha256"],
              "unattributed replay thumbprint does not match P-1's plan/key set")
    print(f"   authenticated thumbprint {a['thumbprint_sha256'][:16]}… and unattributed "
          f"{tp['unattributed']['thumbprint_sha256'][:16]}… recompute exactly")


# --------------------------------------------------------------------------
# 11. Centralized request->response contextual binding set (D2)
# --------------------------------------------------------------------------


def response_context_bindings(vid, by_id):
    """The COMPLETE set of request-derived response fields, compared in one place:
    a response proof's aud (3), cti (7), response_binding (-70004), and -70002
    (root type, vs the binding's root_type_id) MUST bind to the originating
    request. Returns {aud_eq, cti_eq, binding_eq, schema_eq}."""
    v = by_id[vid]
    req = by_id[v["originating_request"]]
    c = claims_of(v["cbor_hex"])
    rc = claims_of(req["cbor_hex"])
    rb = c.get(C_RESPONSE_BINDING)
    return {
        "aud_eq": c.get(C_AUD) == rc.get(C_AUD),
        "cti_eq": c.get(C_CTI) == rc.get(C_CTI),
        "binding_eq": rb == rc.get(C_RESPONSE_BINDING),
        "schema_eq": (not isinstance(rb, dict)) or (c.get(C_SCHEMA_ID) == rb.get(1)),
    }


def gate_response_context(positives, negatives) -> None:
    section("11. Request->response contextual binding set (aud, cti, response_binding, -70002) — D2")
    by_id = {v["id"]: v for v in positives["vectors"]}
    by_id.update({v["id"]: v for v in negatives["vectors"]})

    # The bound positive P-7 must satisfy ALL four bindings against P-4.
    bound = next((v for v in positives["vectors"] if v.get("originating_request")), None)
    check(bound is not None, "a bound response positive must exist")
    if bound is not None:
        b = response_context_bindings(bound["id"], by_id)
        check(all(b.values()), f"{bound['id']} must bind aud/cti/response_binding/-70002 to its request: {b}")
        print(f"   {bound['id']} binds all four request-derived response fields")

    # Each response-context negative violates EXACTLY its named field.
    field_negs = {
        "N-43": "aud_eq", "N-22": "cti_eq", "N-32": "binding_eq", "N-31": "schema_eq",
    }
    for nid, field in field_negs.items():
        v = by_id.get(nid)
        check(v is not None, f"response-context negative {nid} ({field}) missing")
        if v is None or not v.get("originating_request"):
            check(bool(v and v.get("originating_request")), f"{nid} must carry originating_request")
            continue
        b = response_context_bindings(nid, by_id)
        check(b[field] is False, f"{nid} must violate {field} vs its originating request; got {b}")
        if b[field] is False:
            print(f"   D2 {nid} violates {field} (contextual response binding denies)")


# --------------------------------------------------------------------------
# 12. Causality inventory — every negative asserts its advertised violation
# --------------------------------------------------------------------------


def gate_causality_inventory(cddl, positives, negatives) -> None:
    section("12. Causality inventory (every negative's violation shape asserted)")
    try:
        from pycddl import Schema, ValidationError
    except ImportError:
        FAILURES.append("[12] pycddl required for the causality inventory")
        return
    from check_proof_vectors import enc as _enc, unattributed_keyset_correspondence
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    stripped, _ = strip_size_ranges(cddl)
    stripped = stripped.replace("=> unattributed-key-set", "=> any")

    def S(root):
        return Schema(f"start = {root}\n" + stripped)
    req_claims = S("hyprstream-proof-claims")
    sign1_prot = S("proof-sign1-protected")
    sign_body_prot = S("proof-sign-body-protected")
    plan_schema = S("signature-plan")

    by_id = {v["id"]: v for v in positives["vectors"]}
    by_id.update({v["id"]: v for v in negatives["vectors"]})
    # H1: map each credential's token hash -> its exp, to resolve a proof's mapped
    # credential from the proof's credential_hash claim.
    _creds = _load_credentials()
    import hashlib as _hl
    cred_exp_by_hash = {_hl.sha256(c["token"].encode("ascii")).digest(): c["claims"]["exp"]
                        for c in _creds["credentials"].values()}
    PROOF_TYPS = (TYP_REQUEST, TYP_RESPONSE)
    REQUIRED_CLAIMS = {C_AUD, C_EXP, C_IAT, C_CTI, C_CREDENTIAL_HASH, C_SCHEMA_ID, C_BODY_BYTES, C_RESPONSE_BINDING}
    ALLOWED_CLAIMS = REQUIRED_CLAIMS | {C_NONCE}

    def parts(vec):
        raw = bytes.fromhex(vec["cbor_hex"])
        try:
            obj = decode(raw)
            pm = decode(obj[0])
        except StrictError:
            obj, pm = None, None
        return raw, obj, pm

    def claims_or_none(obj):
        try:
            return decode(obj[2])
        except (StrictError, Exception):  # noqa: BLE001
            return None

    def plan_components(plan):
        return [(g.get(1), c.get(1), c.get(2)) for g in (plan or []) if isinstance(g, dict)
                for c in (g.get(3) or []) if isinstance(c, dict)]

    def rejected(schema, cbytes):
        try:
            schema.validate_cbor(cbytes)
            return False
        except ValidationError:
            return True

    def shape_present(vec):
        """Return (present, detail) — is this negative's advertised violation
        structurally in its bytes? Dispatches by deny_class (+ per-id where the
        class covers several vectors)."""
        dc = vec["deny_class"]
        vid = vec["id"]
        raw, obj, pm = parts(vec)
        # Non-deterministic encoding: the object or its payload fails strict decode.
        if dc == "non-deterministic-encoding":
            if obj is None:
                return True, "object not deterministic CBOR"
            try:
                decode(obj[2])
                return False, "payload decoded deterministically"
            except StrictError:
                return True, "payload not deterministic CBOR"
        # E2: a length-delimited (bstr/tstr) value whose header over-declares its
        # length must be rejected as truncated by the strict decoder.
        if dc == "cbor-truncation":
            try:
                decode(raw)
                return False, "decoded without a truncation error"
            except StrictError as e:
                return ("truncat" in str(e)), f"strict decoder: {e}"
        # G2: a truncated integer ARGUMENT inside the claims payload (the outer
        # array decodes, but the payload integer's declared bytes are not present).
        if dc == "integer-truncation":
            if obj is None:
                try:
                    decode(raw)
                    return False, "decoded without a truncation error"
                except StrictError as e:
                    return ("truncat" in str(e)), f"outer strict decoder: {e}"
            try:
                decode(obj[2])
                return False, "payload integer decoded without a truncation error"
            except StrictError as e:
                return ("truncat" in str(e)), f"payload strict decoder: {e}"
        if obj is None or pm is None:
            return False, "unexpectedly non-decodable"
        claims = claims_or_none(obj)
        plan = pm.get(H_PLAN)
        comps = plan_components(plan)
        entries = obj[3] if isinstance(obj[3], list) and obj[3] and isinstance(obj[3][0], list) else None

        if dc == "type-confusion":
            if vid == "N-1":
                return pm.get(H_TYP) not in PROOF_TYPS, "credential typ (not a proof typ) in the proof slot"
            return pm.get(H_TYP) in PROOF_TYPS, "proof typ presented in the credential slot"
        if dc == "missing-typ":
            return H_TYP not in pm, "typ absent from the protected bucket"
        if dc == "domain-separation":
            pair = (pm.get(H_TYP), pm.get(H_DOMAIN))
            good = {(TYP_REQUEST, DOMAIN_REQUEST), (TYP_RESPONSE, DOMAIN_RESPONSE)}
            return pair not in good, f"typ/domain cross-product {pair}"
        if dc == "component-stripping":  # D1 (N-5)
            ek = set()
            for e in (entries or []):
                sh = decode(e[0]); ek.add((sh.get(H_GROUP), sh.get(H_ALG), sh.get(H_KID)))
            return (set(comps) - ek) != set() and ek <= set(comps), \
                "a plan component has no signature entry (coverage violation)"
        if dc == "parser-cap":
            if vid == "N-6":
                return len(plan or []) > 8, "more than 8 signer groups"
            if vid == "N-7":
                return any(len(g.get(3, [])) > 2 for g in (plan or [])), "a group has >2 components"
            if vid == "N-13":
                kids = [k for k, _ in [((c[2]), 0) for c in comps]] + ([pm.get(H_KID)] if isinstance(pm.get(H_KID), (bytes, bytearray)) else [])
                return any(isinstance(k, (bytes, bytearray)) and len(k) > SUITE_KID_MAX for k in kids), "kid over 64 bytes"
            if vid == "N-26":
                return claims and len(claims[C_AUD].encode()) > MAX_SERVICE_DOMAIN_BYTES, "aud over 128 bytes"
            if vid == "N-44":  # E1 body upper boundary
                return claims is not None and len(claims[C_BODY_BYTES]) > MAX_BODY_BYTES, \
                    f"capnp body over the {MAX_BODY_BYTES}-byte cap"
            if vid == "N-47":  # E1 kid lower boundary (empty kid)
                kids, _ = kids_and_suites(vec["cbor_hex"])
                return any(len(k) < 1 for k in kids), "an empty (0-byte) kid, under the 1-byte floor"
        if dc == "proof-credential-expiry":  # H1
            if claims is None:
                return False, "claims did not decode"
            ch = claims.get(C_CREDENTIAL_HASH)
            cred_exp = cred_exp_by_hash.get(ch)
            if cred_exp is None:
                return False, "proof's credential_hash resolves to no mapped credential"
            return claims.get(C_EXP, 0) > cred_exp, \
                f"proof exp {claims.get(C_EXP)} exceeds mapped credential exp {cred_exp}"
        if dc == "nonce-length":  # E1 server-challenge boundary (16..64)
            nonce = claims.get(C_NONCE) if claims else None
            if nonce is None:
                return False, "no Nonce present"
            return not (NONCE_MIN <= len(nonce) <= NONCE_MAX), \
                f"Nonce length {len(nonce)} out of {NONCE_MIN}..{NONCE_MAX}"
        if dc == "closed-claim-set":
            if vid == "N-8":
                return claims is not None and bool(set(claims) - ALLOWED_CLAIMS), "unknown claim key present"
            if vid == "N-14":
                return claims is not None and bool(REQUIRED_CLAIMS - set(claims)), "a required claim is absent"
        if dc == "crit-set":
            return rejected(sign1_prot, obj[0]), "protected bucket fails the exact crit rule"
        if dc == "disposition-confusion":
            if H_KEYSET not in pm:
                return False, "no key set present"
            if vid == "N-10f":  # credential-bound proof (non-null hash) carrying a key set
                return claims is not None and claims.get(C_CREDENTIAL_HASH) is not None, \
                    "key set on a credential-bound (non-null credential_hash) proof"
            if vid == "N-19":   # response proof carrying a key set
                return pm.get(H_TYP) == TYP_RESPONSE, "key set on a response proof"
            return rejected(sign1_prot, obj[0]), "key set in a non-unattributed disposition"
        if dc == "algorithm":
            algs = {a for _, a, _ in comps} | {pm.get(H_ALG)}
            return -8 in algs, "deprecated EdDSA (-8) present"
        if dc == "credential-binding":  # N-15 / B5
            return claims is not None and claims.get(C_CREDENTIAL_HASH) is None and H_KEYSET not in pm, \
                "authenticated proof with a null credential_hash"
        if dc == "freshness":  # N-16
            return H_KEYSET in pm and (claims is None or C_NONCE not in claims), "unattributed proof without Nonce"
        if dc == "group-id-order":
            gids = [g.get(1) for g in (plan or [])]
            return gids != sorted(set(gids)) or len(gids) != len(set(gids)), "group IDs not unique+ascending"
        if dc in ("key-set-strictness", "unattributed-keyset"):
            emb, err = unattributed_keyset_correspondence(pm.get(H_KEYSET), comps)
            if err is not None:
                return True, f"key set correspondence fails: {err}"
            # N-35: correspondence ok but the embedded key does not verify.
            if vid == "N-35" and emb is not None:
                akid = (pm.get(H_ALG), pm.get(H_KID))
                tbs = _enc(["Signature1", obj[0], b"", obj[2]])
                try:
                    Ed25519PublicKey.from_public_bytes(emb[akid]).verify(obj[3], tbs)
                    return False, "signature verified against the embedded key"
                except InvalidSignature:
                    return True, "signature does not verify against the embedded key"
            return False, "key set corresponds (no violation)"
        if dc == "plan-key-uniqueness":
            keys = [(a, k) for _, a, k in comps]
            return len(keys) != len(set(keys)), "plan repeats an (alg, kid)"
        if dc == "plan-mismatch":  # N-20 (C4)
            ek = []
            for e in (entries or []):
                sh = decode(e[0]); ek.append((sh.get(H_GROUP), sh.get(H_ALG), sh.get(H_KID)))
            return any(x not in set(comps) for x in ek), "a signature entry cites a group absent from the plan"
        if dc == "response-binding":
            return rejected(req_claims, obj[2]), "response_binding structure rejected by the CDDL"
        if dc == "suite-plan":
            return rejected(plan_schema, _enc(plan)), "plan rejected (unknown suite / downgrade)"
        if dc == "unprotected-authority":  # N-17
            return H_ALG not in pm or H_KID not in pm, "alg/kid not in the protected bucket"
        if dc == "aud-syntax":
            return claims is not None and not valid_service_domain(claims[C_AUD]), "aud violates service-domain syntax"
        if dc in ("response-cti-binding", "response-binding-equality",
                  "response-schema-binding", "response-aud-binding"):
            field = {"response-aud-binding": "aud_eq", "response-cti-binding": "cti_eq",
                     "response-binding-equality": "binding_eq", "response-schema-binding": "schema_eq"}[dc]
            if not vec.get("originating_request"):
                return False, "no originating_request"
            return response_context_bindings(vid, by_id)[field] is False, f"{field} violated vs request"
        return None, f"UNHANDLED deny_class {dc!r}"

    checked = set()
    for vec in negatives["vectors"]:
        present, detail = shape_present(vec)
        check(present is not None, f"{vec['id']}: deny_class {vec['deny_class']!r} has no causality predicate")
        if present is not None:
            check(present, f"{vec['id']} ({vec['deny_class']}) does not exhibit its violation: {detail}")
            checked.add(vec["id"])
    # Meta-guard: exactly every negative vector ID is covered by a predicate.
    all_ids = {v["id"] for v in negatives["vectors"]}
    check(checked == all_ids,
          f"causality inventory must cover every negative; missing {sorted(all_ids - checked)}")
    print(f"   causality inventory covers all {len(all_ids)} negatives; each exhibits its violation shape")


# --------------------------------------------------------------------------
# F1/F2. Deterministic verifier clock + authenticated credential context
# --------------------------------------------------------------------------


def _b64u(b: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64u_dec(s: str) -> bytes:
    import base64
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _make_jwt(header: dict, claims: dict, sk) -> str:
    hp = _b64u(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
    pp = _b64u(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
    si = f"{hp}.{pp}".encode("ascii")
    return f"{hp}.{pp}.{_b64u(sk.sign(si))}"


def _validate_clearance(cl):
    """H2: the frozen two-axis credential clearance grammar (Gate-2 value 11).
    `[level, compartments]` — level a uint 0..3 (Public/Internal/Confidential/
    Secret), compartments an array of bit indices (uint 0..63), strictly
    ascending and unique (empty allowed). Assurance is structurally absent: an
    extra element, a level out of domain, an out-of-range/duplicate/non-ascending
    compartment, a bitmask integer, or names all deny. Returns the failure list."""
    errs = []
    if not (isinstance(cl, list) and len(cl) == 2):
        return [f"clearance must be a 2-element [level, compartments] array (assurance absent), got {cl!r}"]
    level, comps = cl
    if isinstance(level, bool) or not isinstance(level, int) or not (0 <= level <= 3):
        errs.append(f"level must be a uint 0..3 (Public/Internal/Confidential/Secret), got {level!r}")
    if not isinstance(comps, list):
        errs.append(f"compartments must be an array of bit indices (not a bitmask or name), got {comps!r}")
    else:
        prev = -1
        for c in comps:
            if isinstance(c, bool) or not isinstance(c, int):
                errs.append(f"compartment {c!r} must be a uint bit index (not a name)")
                continue
            if not (0 <= c <= 63):
                errs.append(f"compartment {c} out of range 0..63")
            if c <= prev:
                errs.append(f"compartments must be strictly ascending and unique; {c} follows {prev}")
            prev = c
    return errs


def _verify_credential(token, issuer_pub, issuer_kid, now, expected_aud=None):
    """Return the list of conformance failures for one at+jwt (empty == valid):
    exact at+jwt/EdDSA header, issuer Ed25519 signature, required claims,
    audience, and temporal validity at the frozen verifier clock."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    errs = []
    parts = token.split(".")
    if len(parts) != 3:
        return ["not a compact JWS"]
    hp, pp, sp = parts
    try:
        header = json.loads(_b64u_dec(hp))
        claims = json.loads(_b64u_dec(pp))
    except Exception as e:  # noqa: BLE001
        return [f"undecodable JWS: {e}"]
    if (header.get("typ") != "at+jwt" or header.get("alg") != "EdDSA"
            or header.get("kid") != issuer_kid):
        errs.append(f"header not exact at+jwt/EdDSA/{issuer_kid}")
    try:
        Ed25519PublicKey.from_public_bytes(issuer_pub).verify(
            _b64u_dec(sp), f"{hp}.{pp}".encode("ascii"))
    except InvalidSignature:
        errs.append("issuer signature invalid")
    for r in ("iss", "sub", "aud", "iat", "exp", "jti", "client_id", "tenant", "clearance", "cnf"):
        if r not in claims:
            errs.append(f"missing claim {r}")
    if not claims.get("iss") or not claims.get("sub"):
        errs.append("empty issuer/subject")
    # RFC 9068 §2.2.1: at+jwt REQUIRES a non-empty string client_id.
    if not isinstance(claims.get("client_id"), str) or not claims.get("client_id"):
        errs.append("client_id must be a non-empty string (RFC 9068)")
    if expected_aud is not None and claims.get("aud") != expected_aud:
        errs.append("audience mismatch")
    if not (claims.get("iat", 0) <= now < claims.get("exp", 0)):
        errs.append("not temporally valid at verifier_now")
    if "hs_signer_suite" not in (claims.get("cnf") or {}):
        errs.append("cnf lacks hs_signer_suite")
    errs += [f"clearance: {e}" for e in _validate_clearance(claims.get("clearance"))]
    return errs


def _load_credentials():
    return json.loads((VECTORS_DIR / "proof-v1-credentials.json").read_text())


def _keymaps():
    keys = json.loads((VECTORS_DIR / "proof-v1-keys.json").read_text())["keys"]
    ed = {bytes.fromhex(k["kid_hex"]): bytes.fromhex(k["public_hex"]) for k in keys["ed25519"]}
    ml = {bytes.fromhex(k["kid_hex"]): bytes.fromhex(k["public_hex"]) for k in keys["ml_dsa_65"]}
    return ed, ml


def _suite_thumbprint(suite, pubs):
    import hashlib
    from check_proof_vectors import enc as _enc
    return hashlib.sha256(_enc([suite, list(pubs)])).digest()


def gate_verifier_clock(positives, negatives) -> None:
    section("F1. Deterministic verifier clock (verifier_now)")
    creds = _load_credentials()
    now = creds.get("verifier_now")
    check(isinstance(now, int), "verifier_now must be a declared integer")
    if not isinstance(now, int):
        return
    # Metadata agreement across every shipped artifact (no wall-clock ambiguity).
    for fn in ("proof-v1-positive.json", "proof-v1-negative.json",
               "proof-v1-thumbprints.json", "proof-v1-keys.json", "proof-v1-credentials.json"):
        d = json.loads((VECTORS_DIR / fn).read_text())
        check(d.get("verifier_now") == now, f"{fn} must declare verifier_now == {now}")
    # Every advertised positive and every credential is temporally valid at that
    # exact instant (strict at exp: iat <= now < exp).
    for v in positives["vectors"]:
        c = claims_of(v["cbor_hex"])
        check(c[C_IAT] <= now < c[C_EXP],
              f"{v['id']} must be temporally valid at verifier_now {now} (iat {c[C_IAT]}, exp {c[C_EXP]})")
    for name, cred in creds["credentials"].items():
        cl = cred["claims"]
        check(cl["iat"] <= now < cl["exp"],
              f"credential {name} must be temporally valid at verifier_now {now}")
    # Load-bearing: pre-iat, at-exp, and a wall-clock-style instant are all outside
    # the interval, so evaluating there would deny an advertised positive.
    c0 = claims_of(positives["vectors"][0]["cbor_hex"])
    iat, exp = c0[C_IAT], c0[C_EXP]
    check(not (iat <= (iat - 1) < exp), "pre-iat instant must be rejected by the temporal rule")
    check(not (iat <= exp < exp), "an instant at exp must be rejected (strict at exp)")
    check(not (iat <= 4102444800 < exp), "a wall-clock-style instant (2100) must be outside validity")
    print(f"   verifier_now {now} agrees across artifacts; all positives/credentials valid; "
          f"pre-iat/at-exp/wall-clock rejected")


def gate_credential_context(positives, negatives) -> None:
    section("F2. Authenticated credential context (issuer-signed at+jwt, cnf, hash)")
    creds = _load_credentials()
    now = creds["verifier_now"]
    issuer_pub = bytes.fromhex(creds["issuer"]["public_hex"])
    issuer_kid = creds["issuer"]["kid"]
    ed_by_kid, ml_by_kid = _keymaps()
    pos_by_id = {v["id"]: v for v in positives["vectors"]}

    # The literal placeholder must be gone from every artifact/tool.
    for fn in ("tools/gen_proof_vectors.py",):
        txt = (VECTORS_DIR.parent / fn).read_text()
        check("FIXTURE.FIXTURE" not in txt, f"{fn} still carries the .FIXTURE.FIXTURE placeholder")

    # Both credentials verify end to end.
    for name, cred in creds["credentials"].items():
        errs = _verify_credential(cred["token"], issuer_pub, issuer_kid, now,
                                  expected_aud=cred["claims"]["aud"])
        check(not errs, f"credential {name} must be a valid at+jwt: {errs}")

    def primary_groups_matching(pid, cnf_tp):
        obj = decode(bytes.fromhex(pos_by_id[pid]["cbor_hex"]))
        plan = decode(obj[0]).get(H_PLAN) or []
        matched = []
        for grp in plan:
            pubs, ok = [], True
            for comp in grp[3]:
                alg, kid = comp[1], comp[2]
                pub = ed_by_kid.get(kid) if alg == ALG_ED25519 else ml_by_kid.get(kid)
                if pub is None:
                    ok = False
                    break
                pubs.append(pub)
            if ok and _suite_thumbprint(grp[2], pubs) == cnf_tp:
                matched.append(grp[1])
        return matched, obj

    # Each authenticated positive: credential_hash == SHA-256(exact token), and
    # cnf resolves to EXACTLY ONE plan group (the primary) by suite + ordered keys.
    import hashlib
    for pid, credname in creds["positive_to_credential"].items():
        cred = creds["credentials"][credname]
        cnf_tp = _b64u_dec(cred["claims"]["cnf"]["hs_signer_suite"])
        matched, obj = primary_groups_matching(pid, cnf_tp)
        check(len(matched) == 1, f"{pid}: cnf must resolve to exactly one primary group; matched {matched}")
        pclaims = decode(obj[2])
        ch = pclaims.get(C_CREDENTIAL_HASH)
        check(ch == hashlib.sha256(cred["token"].encode("ascii")).digest(),
              f"{pid}: credential_hash must equal SHA-256 of the mapped {credname} credential")
        # H1: a proof's exp MUST NOT exceed the mapped credential's exp.
        check(pclaims[C_EXP] <= cred["claims"]["exp"],
              f"{pid}: proof exp {pclaims[C_EXP]} must not exceed credential exp {cred['claims']['exp']}")
    print(f"   both credentials valid; {len(creds['positive_to_credential'])} positives bind their "
          f"credential hash + cnf primary group")

    # ---- Load-bearing counter-proofs (re-signed with the seeded issuer key so
    #      only the named property is wrong; each MUST produce a failure). -------
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    keys = json.loads((VECTORS_DIR / "proof-v1-keys.json").read_text())["keys"]
    issuer_seed = next(bytes.fromhex(k["seed_hex"]) for k in keys["ed25519"]
                       if k["kid_ascii"] == issuer_kid)
    sk_i = Ed25519PrivateKey.from_private_bytes(issuer_seed)
    other_seed = bytes((0x10 + i) & 0xFF for i in range(32))
    sk_other = Ed25519PrivateKey.from_private_bytes(other_seed)
    good = creds["credentials"]["classical"]
    hdr = dict(good["header"])
    base_claims = dict(good["claims"])
    aud = base_claims["aud"]

    def rejected(label, token):
        errs = _verify_credential(token, issuer_pub, issuer_kid, now, expected_aud=aud)
        check(bool(errs), f"counter-proof '{label}' must be rejected but validated clean")
        if errs:
            print(f"   F2 counter '{label}' rejected: {errs[0]}")

    # 1. flipped signature byte.
    tok = good["token"]
    hp, pp, sp = tok.split(".")
    bad_sig = bytearray(_b64u_dec(sp)); bad_sig[0] ^= 0x01
    rejected("flipped issuer signature", f"{hp}.{pp}.{_b64u(bytes(bad_sig))}")
    # 2. wrong issuer key (validly signed by a non-issuer key).
    rejected("wrong issuer key", _make_jwt(hdr, base_claims, sk_other))
    # 3. wrong audience (re-signed).
    rejected("wrong audience", _make_jwt(hdr, {**base_claims, "aud": "evil.svc.hyprstream.test"}, sk_i))
    # 4. missing required claim (re-signed).
    rejected("missing tenant claim", _make_jwt(hdr, {k: v for k, v in base_claims.items() if k != "tenant"}, sk_i))
    # 4b. I1: RFC 9068 at+jwt requires client_id — a re-signed token without it denies.
    rejected("missing client_id (RFC 9068)", _make_jwt(hdr, {k: v for k, v in base_claims.items() if k != "client_id"}, sk_i))
    rejected("empty client_id", _make_jwt(hdr, {**base_claims, "client_id": ""}, sk_i))
    # 5. wrong typ header (re-signed).
    rejected("wrong typ header", _make_jwt({**hdr, "typ": "JWT"}, base_claims, sk_i))
    # 6. clock outside validity (re-signed with an expired window).
    rejected("expired at verifier_now",
             _make_jwt(hdr, {**base_claims, "iat": now - 100, "exp": now - 1}, sk_i))
    # 7. wrong cnf: resolves to no plan group.
    bad_cnf = _make_jwt(hdr, {**base_claims, "cnf": {"hs_signer_suite": _b64u(b"\x00" * 32)}}, sk_i)
    bad_tp = _b64u_dec(json.loads(_b64u_dec(bad_cnf.split(".")[1]))["cnf"]["hs_signer_suite"])
    matched, _ = primary_groups_matching("P-4", bad_tp)
    check(matched == [], f"a wrong cnf thumbprint must resolve to no plan group; matched {matched}")
    print(f"   F2 counter 'wrong cnf' resolves to no primary group")
    # 8. tampered proof credential_hash on a positive.
    obj = decode(bytes.fromhex(pos_by_id["P-4"]["cbor_hex"]))
    tampered = decode(obj[2]); tampered[C_CREDENTIAL_HASH] = b"\x00" * 32
    check(tampered[C_CREDENTIAL_HASH] != hashlib.sha256(good["token"].encode("ascii")).digest(),
          "a tampered proof credential_hash must not match the credential")
    # 9. approver group (P-5 group 2) stays valid against its OWN enrollment and is
    #    NOT forced into the primary credential cnf.
    cnf_classical = _b64u_dec(creds["credentials"]["classical"]["claims"]["cnf"]["hs_signer_suite"])
    matched5, _ = primary_groups_matching("P-5", cnf_classical)
    check(matched5 == [1],
          f"P-5's cnf must resolve to the primary (client) group only, not the approver; matched {matched5}")
    approver_pub = ed_by_kid.get(b"approver-ed25519-1")
    if approver_pub is not None:
        check(_suite_thumbprint(SUITE_CLASSICAL, [approver_pub]) != cnf_classical,
              "the approver group must NOT match the primary credential cnf")
    print(f"   F2 approver group bound to its own enrollment (cnf resolves to the primary only)")

    # ---- G1: hybrid credentials are at+jwt-only; CWT cnf is single-key/classical.
    prof = CREDENTIAL_PATH.read_text()
    for pin in ("hybrid credentials are", "at+jwt", "no v16 CWT confirmation method",
                "single RFC 8747 `COSE_Key`"):
        check(pin in prof, f"credential-profile must state the hybrid-JWT-only disposition (missing: {pin!r})")
    # The shipped hybrid credential MUST be a compact JWT with the exact cnf member.
    hyb = creds["credentials"]["hybrid"]
    check(hyb.get("encoding") == "at+jwt" and hyb["token"].count(".") == 2,
          "the hybrid credential fixture must be a compact at+jwt (JWT)")
    check("hs_signer_suite" in hyb["claims"]["cnf"],
          "the hybrid credential cnf must use the hs_signer_suite confirmation method")
    # The ONLY shipped CWT credential (N-1) is classical: its RFC 8747 cnf (claim 8)
    # is a single COSE_Key (one key), which cannot pin a hybrid two-key group.
    n1 = next((v for v in negatives["vectors"] if v["id"] == "N-1"), None)
    check(n1 is not None, "N-1 (issuer-signed CWT credential) must exist")
    if n1 is not None:
        n1_claims = decode(decode(bytes.fromhex(n1["cbor_hex"]))[2])
        n1_cnf = n1_claims.get(8)
        check(isinstance(n1_cnf, dict) and set(n1_cnf) == {1} and isinstance(n1_cnf[1], dict),
              "the only shipped CWT credential (N-1) must carry a single RFC 8747 COSE_Key cnf")
    # Causal reject: a single-COSE_Key CWT-style cnf resolves to the CLASSICAL
    # record, which matches a classical primary (P-4) but NOT the hybrid group of a
    # hybrid proof (P-2) — so a CWT credential purporting to confirm the hybrid
    # suite via one COSE_Key denies (no primary group resolves).
    client_ed_pub = ed_by_kid.get(b"client-ed25519-1")
    check(client_ed_pub is not None, "client Ed25519 key must resolve for the G1 counter-proof")
    if client_ed_pub is not None:
        single_key_cnf = _suite_thumbprint(SUITE_CLASSICAL, [client_ed_pub])
        m_hyb, _ = primary_groups_matching("P-2", single_key_cnf)
        m_cls, _ = primary_groups_matching("P-4", single_key_cnf)
        check(m_hyb == [], f"a single-COSE_Key (CWT) cnf must NOT resolve P-2's hybrid group; matched {m_hyb}")
        check(m_cls == [1], f"a single-COSE_Key cnf must resolve a classical primary (P-4); matched {m_cls}")
        print(f"   G1 single-key CWT cnf resolves classical (P-4) but denies the hybrid suite (P-2): "
              f"hybrid is at+jwt-only")

    # ---- H2: the two-axis credential clearance grammar (Gate-2 value 11) -------
    cddl = CDDL_PATH.read_text()
    for pin in ("credential-clearance    = [ clearance-level, clearance-compartments ]",
                "clearance-level         = 0 / 1 / 2 / 3",
                "compartment-index       = uint .le 63"):
        check(pin in cddl, f"CDDL must freeze the clearance grammar (missing: {pin!r})")
    check("assurance" in prof and "structurally absent" in prof.lower(),
          "credential-profile must state assurance is structurally absent from the credential wire")
    # Every shipped credential's clearance conforms (at+jwt via claims; N-1 via CWT).
    for name, cred in creds["credentials"].items():
        e = _validate_clearance(cred["claims"].get("clearance"))
        check(not e, f"credential {name} clearance must conform to the two-axis grammar: {e}")
    n1 = next((v for v in negatives["vectors"] if v["id"] == "N-1"), None)
    if n1 is not None:
        n1_clear = decode(decode(bytes.fromhex(n1["cbor_hex"]))[2]).get(-70006)
        check(not _validate_clearance(n1_clear),
              f"the CWT credential N-1 clearance must conform to the two-axis grammar: {n1_clear!r}")
    # Load-bearing counter-proofs: a valid credential re-signed with a malformed
    # clearance must be rejected, one case per denial the grammar names.
    for label, bad in (("unknown level 4", [4, [5, 7]]),
                       ("compartment out of range 64", [2, [64]]),
                       ("duplicate compartments", [2, [5, 5]]),
                       ("descending compartments", [2, [7, 5]]),
                       ("compartment names not indices", [2, ["pii"]]),
                       ("compartments as a bitmask integer", [2, 160]),
                       ("assurance present (3rd element)", [2, [5, 7], 1])):
        rejected(f"clearance: {label}", _make_jwt(hdr, {**base_claims, "clearance": bad}, sk_i))
    check(_validate_clearance([2, [5, 7]]) == [] and _validate_clearance([0, []]) == [],
          "the conforming clearance shapes [2,[5,7]] and [0,[]] must validate")
    print(f"   H2 clearance grammar frozen; shipped clearances conform; 7 malformed clearances rejected")


# --------------------------------------------------------------------------


def main() -> None:
    cddl = CDDL_PATH.read_text()
    # Prose artifacts render private-use keys with the typographic minus sign
    # U+2212; normalize to ASCII '-' so value checks are notation-agnostic.
    registry = REGISTRY_PATH.read_text().replace("−", "-")
    credential = CREDENTIAL_PATH.read_text().replace("−", "-")
    positives = load_json("proof-v1-positive.json")
    negatives = load_json("proof-v1-negative.json")

    print("v16 profile freeze — mechanical validation gate")
    print(f"  positives: {len(positives['vectors'])}  negatives: {len(negatives['vectors'])}\n")

    gate_cddl(cddl, positives, negatives)
    gate_values(cddl, registry, credential, positives, negatives)
    gate_caps(cddl, positives, negatives)
    gate_response_binding(positives)
    gate_collisions()
    gate_type_confusion(negatives)
    gate_response_binding_equality(positives, negatives)
    gate_unattributed_keyset(positives, negatives)
    gate_replay_thumbprints(cddl, positives)
    gate_readme_counts(positives, negatives)
    gate_response_context(positives, negatives)
    gate_causality_inventory(cddl, positives, negatives)
    gate_verifier_clock(positives, negatives)
    gate_credential_context(positives, negatives)
    gate_canonical(positives, negatives)

    print()
    if FAILURES:
        for line in FAILURES:
            print(f"FAIL {line}")
        print(f"\n{len(FAILURES)} failure(s). Profile freeze is NOT mechanically valid.")
        sys.exit(1)
    print("PASS: CDDL, registry, credential profile, and fixtures are mutually "
          "consistent and canonical.")
    print("NOTE: not production-closed — the two vnd.hyprstream media-type IANA "
          "registrations remain open; -70200 stays project-private.")


if __name__ == "__main__":
    main()
