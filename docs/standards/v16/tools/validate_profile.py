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
    # pycddl 0.9.1 PANICS (Rust unwrap-on-None) validating the AKP/ML-DSA-65
    # COSE_Key inside the unattributed key set, so the key set is passed opaquely
    # to pycddl for the protected-bucket pass ONLY. It is NOT left unvalidated:
    # section 9 (B4) enforces the full key shape (closed COSE_Key field set,
    # kty/crv or parameter set, exact public-key byte length), exact ordered 1:1
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
            for name in ("proof-v1-keys.json", "proof-v1-positive.json", "proof-v1-negative.json"):
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
        # Fence: N-1 must NOT self-allocate credential_use_profile (-70008).
        check(-70008 not in n1c, "N-1 must not carry the fenced credential_use_profile (-70008)")
        print("   #2 N-1 is profile-valid (cnf PoP + tenant -70005 + clearance -70006; no -70008)")

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
    # reordered (N-38) each break the 1:1 ordered correspondence.
    for nid, what in (("N-18", "surplus key"), ("N-37", "duplicate element"),
                      ("N-38", "reordered elements")):
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
    # against the embedded key.
    n35 = by_id.get("N-35")
    check(n35 is not None, "unattributed-keyset negative N-35 (mismatched key) is missing")
    if n35 is not None:
        obj = decode(bytes.fromhex(n35["cbor_hex"]))
        pm = decode(obj[0])
        emb, err = unattributed_keyset_correspondence(pm.get(H_KEYSET), components_of(pm.get(H_PLAN)))
        check(err is None, f"N-35 key set must be shape-valid (only the key bytes differ): {err}")
        kid = pm.get(H_KID)
        if emb is not None and emb.get(kid) is not None:
            tbs = _enc(["Signature1", obj[0], b"", obj[2]])
            try:
                Ed25519PublicKey.from_public_bytes(emb[kid]).verify(obj[3], tbs)
                check(False, "N-35 signature must NOT verify against its (mismatched) embedded key")
            except InvalidSignature:
                print("   B4 N-35 rejected: signature does not verify against the embedded key")


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
