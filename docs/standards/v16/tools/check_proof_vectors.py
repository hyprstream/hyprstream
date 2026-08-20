#!/usr/bin/env python3
"""Check the checked-in proof-v1 vectors.

Verifies, for the positive vectors, that:
  * `sha256` and `size_bytes` match `cbor_hex`;
  * the object decodes as an untagged COSE_Sign1/COSE_Sign structure;
  * every payload and protected-header bucket is RFC 8949 core-deterministic
    (definite lengths, sorted unique map keys, no tags, no floats);
  * every signature verifies over the profile's `Sig_structure` with a
    zero-length `external_aad`, using the published test keys; and
  * every signature entry's `(alg, kid, group)` matches exactly one component
    of the signed `signature_plan`.

For the negative vectors it verifies only the digest and that the bytes differ
from every positive vector: their expected disposition is `deny`, which is a
statement about a verifier, not about these bytes.

Usage:  python3 check_proof_vectors.py [vectors_dir]
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)


# --------------------------------------------------------------------------
# Minimal strict CBOR decoder (rejects everything the profile forbids)
# --------------------------------------------------------------------------


class StrictError(Exception):
    pass


def decode(data: bytes, *, strict: bool = True):
    value, rest = _decode(data, strict)
    if rest:
        raise StrictError("trailing data")
    return value


def _decode(b: bytes, strict: bool):
    if not b:
        raise StrictError("truncated")
    ib = b[0]
    major, ai = ib >> 5, ib & 0x1F
    rest = b[1:]
    if ai == 31:
        raise StrictError("indefinite length")
    if ai < 24:
        val = ai
    elif ai in (24, 25, 26, 27):
        # Additional information 24/25/26/27 carry a 1/2/4/8-byte argument. The
        # argument bytes MUST be present before slicing — a header that declares
        # more bytes than remain is a truncated value and MUST fail closed, not
        # silently read a short/zero value (G2). Applies to every major type.
        n = {24: 1, 25: 2, 26: 4, 27: 8}[ai]
        if len(rest) < n:
            raise StrictError(
                f"truncated additional-information argument: header declares {n} byte(s), "
                f"{len(rest)} remain"
            )
        val, rest = int.from_bytes(rest[:n], "big"), rest[n:]
        # Minimal-argument-length: a value MUST use the shortest argument that can
        # hold it (24..255 -> ai 24; 256..65535 -> ai 25; etc.).
        floor = {24: 24, 25: 0x100, 26: 0x10000, 27: 0x100000000}[ai]
        if strict and val < floor:
            raise StrictError("non-minimal integer encoding")
    else:
        raise StrictError(f"unsupported additional information {ai}")

    if major == 7:
        # Only the three simple values the profile uses; floats (ai 25-27) and
        # every other simple value are rejected above or here.
        if ai in (20, 21, 22):
            return {20: False, 21: True, 22: None}[ai], rest
        raise StrictError("floating-point and other simple values are forbidden")
    if major == 0:
        return val, rest
    if major == 1:
        return -1 - val, rest
    if major == 2:
        # E2: a length-delimited value MUST have at least `val` bytes remaining;
        # Python slicing silently truncates, so an over-declared length (e.g. the
        # `58 41` header before only 64 bytes, or a widened signature header) would
        # otherwise "decode" and even verify. Reject truncation explicitly.
        if len(rest) < val:
            raise StrictError(
                f"byte string truncated: header declares {val} bytes, {len(rest)} remain"
            )
        return rest[:val], rest[val:]
    if major == 3:
        if len(rest) < val:
            raise StrictError(
                f"text string truncated: header declares {val} bytes, {len(rest)} remain"
            )
        return rest[:val].decode("utf-8"), rest[val:]
    if major == 4:
        out = []
        for _ in range(val):
            item, rest = _decode(rest, strict)
            out.append(item)
        return out, rest
    if major == 5:
        out = {}
        prev_key_bytes = None
        for _ in range(val):
            before = rest
            key, rest = _decode(rest, strict)
            key_bytes = before[: len(before) - len(rest)]
            if strict:
                if prev_key_bytes is not None and key_bytes <= prev_key_bytes:
                    raise StrictError("map keys not sorted / duplicated")
                prev_key_bytes = key_bytes
            value, rest = _decode(rest, strict)
            out[key] = value
        return out, rest
    if major == 6:
        raise StrictError("tags are forbidden in this profile")
    raise StrictError("unreachable")


def enc_head(major: int, value: int) -> bytes:
    if value < 24:
        return bytes([(major << 5) | value])
    if value < 0x100:
        return bytes([(major << 5) | 24, value])
    if value < 0x10000:
        return bytes([(major << 5) | 25]) + value.to_bytes(2, "big")
    if value < 0x100000000:
        return bytes([(major << 5) | 26]) + value.to_bytes(4, "big")
    return bytes([(major << 5) | 27]) + value.to_bytes(8, "big")


def enc(obj) -> bytes:
    if obj is None:
        return b"\xf6"
    if isinstance(obj, int):
        return enc_head(0, obj) if obj >= 0 else enc_head(1, -1 - obj)
    if isinstance(obj, bytes):
        return enc_head(2, len(obj)) + obj
    if isinstance(obj, str):
        raw = obj.encode()
        return enc_head(3, len(raw)) + raw
    if isinstance(obj, list):
        return enc_head(4, len(obj)) + b"".join(enc(i) for i in obj)
    if isinstance(obj, dict):
        items = sorted(((enc(k), enc(v)) for k, v in obj.items()), key=lambda kv: kv[0])
        return enc_head(5, len(items)) + b"".join(k + v for k, v in items)
    raise TypeError(type(obj))


def cross_group_key_aliases(components):
    """S1: `components` is a list of (group_id, alg, raw_public_key_bytes) for a
    proof's resolved signer components. A cryptographic public-key IDENTITY —
    (algorithm, raw public-key bytes), never kid/group_id/enrollment label/plan
    position — may participate in AT MOST ONE logical signer group. Returns the
    identities that appear in more than one group. Different algorithms are distinct
    identities (legitimate hybrid), and the ordered components inside one suite group
    are unaffected (repeats within a single group are not cross-group aliases)."""
    groups = {}
    for gid, alg, pub in components:
        groups.setdefault((alg, bytes(pub)), set()).add(gid)
    return [ident for ident, gids in groups.items() if len(gids) > 1]


def cose_key_public(k):
    """The raw public key bytes of an unattributed COSE_Key: OKP (kty 1) carries it
    at -2, AKP (kty 7) at -1."""
    if k.get(1) == 1:
        return k.get(-2)
    if k.get(1) == 7:
        return k.get(-1)
    return None


def unattributed_replay_preimage(sep, plan, keyset):
    """M1: the CONTENT-BOUND unattributed replay-namespace preimage. For each signer
    group (in plan order) it binds the group's `suite_id` and that group's public
    keys IN COMPONENT ORDER, drawn from the embedded key set (which corresponds 1:1
    to the plan components per B4). The attacker-chosen `group_id`/`kid` labels are
    normalized OUT, matching the authenticated derivation's discipline — so a
    self-asserted signer cannot mint a fresh replay namespace by permuting labels
    over identical key material. Group boundaries and order ARE preserved (a
    different suite, key byte, or group/key ordering yields a different thumbprint),
    so distinct cryptographic identities are never over-collapsed."""
    groups, idx = [], 0
    for g in (plan or []):
        n = len(g.get(3, []) or [])
        pubs = [cose_key_public(k) for k in (keyset or [])[idx:idx + n]]
        idx += n
        groups.append([g.get(2), pubs])   # [suite_id, [ordered component public keys]]
    # R1: canonical GROUP ORDER — sort the content-only group records by their
    # RFC 8949 deterministic-CBOR encoding as unsigned byte strings, so the same
    # signer set in any plan order (A,B or B,A) hashes identically. This never sorts
    # on or includes group_id/kid/plan position; component keys INSIDE each group
    # keep their suite-plan order.
    groups.sort(key=lambda gr: enc(gr))
    return enc([sep, groups])


def _enrollment_thumbprint(rec):
    """Recompute an enrollment record's signer-suite thumbprint from its OWN
    suite_id + ordered public keys — never trust the record's stored thumbprint
    field (byte-audit). Returns the base64url thumbprint, or None if malformed."""
    try:
        suite = rec["suite_id"]
        pubs = [bytes.fromhex(h) for h in rec["component_public_keys_hex"]]
    except (KeyError, TypeError, ValueError):
        return None
    return base64.urlsafe_b64encode(hashlib.sha256(enc([suite, pubs])).digest()).rstrip(b"=").decode()


def resolve_approver_enrollment(creds_doc, requested_thumbprint_b64):
    """Q1: the authoritative approver enrollment record for a signer-suite record,
    matched by its RECOMPUTED cryptographic content (suite + ordered keys), NOT by
    the attacker-chosen group_id/kid and NOT by the record's self-declared thumbprint
    field. None if no enrollment's own suite/keys hash to the requested thumbprint."""
    for e in creds_doc.get("approver_enrollments", []):
        if _enrollment_thumbprint(e) == requested_thumbprint_b64:
            return e
    return None


def validate_approver_enrollment(rec, requested_thumbprint_b64, expected_tenant, now):
    """Q1: an additional (approver) signer group MUST resolve to an authoritative
    enrollment (credential-profile §5) whose OWN suite/keys recompute to the
    requested content thumbprint (record integrity + key/suite binding), that is an
    active, unexpired `approver` role coherent with the credential tenant, with a
    non-negative integer enrollment_epoch. Returns the failure list; an unknown,
    tampered, key/suite-mismatched, inactive, expired, cross-tenant, or wrong-role
    record denies."""
    if rec is None:
        return ["no authoritative approver enrollment for this signer-suite record"]
    errs = []
    recomputed = _enrollment_thumbprint(rec)
    if recomputed is None:
        return ["approver enrollment record has a malformed suite_id / public keys"]
    # Record integrity: the stored thumbprint must equal the recomputed one.
    if rec.get("thumbprint_b64") != recomputed:
        errs.append("approver enrollment thumbprint_b64 disagrees with its own suite/keys (tampered record)")
    # Key/suite binding: the record must bind the REQUESTED group content.
    if requested_thumbprint_b64 is not None and recomputed != requested_thumbprint_b64:
        errs.append("approver enrollment key/suite does not bind the requested signer group")
    if rec.get("role") != "approver":
        errs.append(f"enrollment role {rec.get('role')!r} is not 'approver'")
    if rec.get("status") != "active":
        errs.append(f"enrollment status {rec.get('status')!r} is not active (inactive/revoked)")
    ea = rec.get("expires_at")
    if isinstance(ea, bool) or not isinstance(ea, int):
        errs.append(f"enrollment expires_at must be an integer, got {ea!r}")
    elif ea <= now:
        errs.append(f"approver enrollment expired at verifier_now {now} (expires_at {ea})")
    if expected_tenant is not None and rec.get("tenant") != expected_tenant:
        errs.append(f"enrollment tenant {rec.get('tenant')!r} != credential tenant {expected_tenant!r}")
    ee = rec.get("enrollment_epoch")
    if isinstance(ee, bool) or not isinstance(ee, int) or ee < 0:
        errs.append(f"enrollment_epoch must be a non-negative integer, got {ee!r}")
    return errs


def resolve_primary_enrollment(creds_doc, requested_thumbprint_b64):
    """T1: the authoritative PRIMARY enrollment record for a signer-suite record,
    matched by its RECOMPUTED cryptographic content (suite + ordered keys), never by
    the record's self-declared thumbprint field. None if no primary enrollment's own
    suite/keys hash to the requested thumbprint."""
    for e in creds_doc.get("primary_enrollments", []):
        if _enrollment_thumbprint(e) == requested_thumbprint_b64:
            return e
    return None


def validate_primary_enrollment(rec, requested_thumbprint_b64, expected_tenant, expected_principal, now):
    """T1: a credential's cnf primary group MUST resolve to an authoritative PRIMARY
    enrollment whose OWN suite/keys recompute to the requested content thumbprint
    (record integrity + key/suite binding), that is an active, unexpired `primary`
    role coherent with the credential tenant and principal. Returns the failure list;
    an unknown, tampered, key/suite-mismatched, wrong-role, cross-tenant/principal,
    inactive, or expired record denies."""
    if rec is None:
        return ["no authoritative primary enrollment for this signer-suite record"]
    errs = []
    recomputed = _enrollment_thumbprint(rec)
    if recomputed is None:
        return ["primary enrollment record has a malformed suite_id / public keys"]
    if rec.get("thumbprint_b64") != recomputed:
        errs.append("primary enrollment thumbprint_b64 disagrees with its own suite/keys (tampered record)")
    if requested_thumbprint_b64 is not None and recomputed != requested_thumbprint_b64:
        errs.append("primary enrollment key/suite does not bind the requested cnf primary group")
    if rec.get("role") != "primary":
        errs.append(f"enrollment role {rec.get('role')!r} is not 'primary'")
    if rec.get("status") != "active":
        errs.append(f"primary enrollment status {rec.get('status')!r} is not active")
    ea = rec.get("expires_at")
    if isinstance(ea, bool) or not isinstance(ea, int):
        errs.append(f"primary enrollment expires_at must be an integer, got {ea!r}")
    elif ea <= now:
        errs.append(f"primary enrollment expired at verifier_now {now} (expires_at {ea})")
    if expected_tenant is not None and rec.get("tenant") != expected_tenant:
        errs.append(f"primary enrollment tenant {rec.get('tenant')!r} != credential tenant {expected_tenant!r}")
    if expected_principal is not None and rec.get("principal") != expected_principal:
        errs.append(f"primary enrollment principal {rec.get('principal')!r} != credential subject {expected_principal!r}")
    ee = rec.get("enrollment_epoch")
    if isinstance(ee, bool) or not isinstance(ee, int) or ee < 0:
        errs.append(f"enrollment_epoch must be a non-negative integer, got {ee!r}")
    return errs


def authenticated_replay_thumbprint(domain, suite_id, pubs, enrollment_epoch):
    """T1/C1: the authenticated primary signer-suite replay thumbprint —
    SHA-256(det-CBOR([domain, suite_id, [ordered public keys], enrollment_epoch])),
    with the enrollment_epoch taken from the resolved primary enrollment record."""
    return hashlib.sha256(enc([domain, suite_id, list(pubs), enrollment_epoch])).digest()


def resolve_session(creds_doc, iss, sid):
    """K1: the authoritative session record keyed by exact (iss, sid) (§3.4),
    held in the credential context OUTSIDE the credential wire. None if absent."""
    for s in creds_doc.get("sessions", []):
        if s.get("iss") == iss and s.get("sid") == sid:
            return s
    return None


def validate_session(creds_doc, cred_claims, now):
    """K1: for a `sid`-bearing credential, resolve and validate its authoritative
    session. Returns (session_record_or_None, failure_list). A non-session
    credential returns (None, []). An unknown, revoked (non-active), expired
    (expiry <= now), or (iss/sub/tenant)-mismatched session denies."""
    sid = cred_claims.get("sid")
    if sid is None:
        return None, []
    s = resolve_session(creds_doc, cred_claims.get("iss"), sid)
    if s is None:
        return None, [f"no authoritative session for (iss={cred_claims.get('iss')!r}, sid={sid!r})"]
    errs = []
    if s.get("status") != "active":
        errs.append(f"session status {s.get('status')!r} is not active (revoked)")
    if not isinstance(s.get("expiry"), int) or s.get("expiry") <= now:
        errs.append(f"session is not active at verifier_now {now} (expiry {s.get('expiry')!r})")
    # T2 (§3.4): the session's creation time must be an integer and coherently
    # ordered — created <= verifier_now < expiry. A missing, non-integer,
    # future-dated, or after-expiry `created` denies.
    created = s.get("created")
    if isinstance(created, bool) or not isinstance(created, int):
        errs.append(f"session created must be an integer, got {created!r}")
    else:
        if created > now:
            errs.append(f"session created {created} is in the future (> verifier_now {now})")
        if isinstance(s.get("expiry"), int) and created >= s.get("expiry"):
            errs.append(f"session created {created} is not before expiry {s.get('expiry')}")
    for k in ("iss", "sub", "tenant"):
        if s.get(k) != cred_claims.get(k):
            errs.append(f"session {k} {s.get(k)!r} != credential {k} {cred_claims.get(k)!r}")
    # L3 (§3.4): the authoritative session MUST carry a non-negative integer
    # clearance_epoch. Missing or non-integer/negative denies.
    ce = s.get("clearance_epoch")
    if isinstance(ce, bool) or not isinstance(ce, int) or ce < 0:
        errs.append(f"session clearance_epoch must be a non-negative integer, got {ce!r}")
    # M2 (§3.4): a sid-keyed session is a user session; its session_kind MUST be the
    # exact interactive kind. Missing, wrong type, empty, or workload/service denies.
    if s.get("session_kind") != "interactive":
        errs.append(f"session_kind must be 'interactive' for a user-session credential, got {s.get('session_kind')!r}")
    return s, errs


def validate_tenant(value):
    """J1: the frozen credential tenant rule (credential-profile §2 tenant row):
    a verified tenant/Casbin domain. Missing, empty, or wildcard tenants deny.
    Shared by both the gate and the standalone checker for JWT `tenant` and the
    CWT integer key -70005. Returns the list of failures ([] == valid). Does not
    broaden tenant syntax beyond the frozen 'non-empty, not wildcard' rule."""
    if not isinstance(value, str):
        return [f"tenant must be a string, got {type(value).__name__}"]
    if value == "":
        return ["tenant must be non-empty (empty tenants deny)"]
    if value == "*":
        return ["tenant must not be the wildcard '*'"]
    return []


ALG_ED25519, ALG_ML_DSA_65 = -19, -49
H_ALG, H_CRIT, H_KID, H_TYP = 1, 2, 4, 16
H_DOMAIN, H_PLAN, H_GROUP, H_KEYSET = -70100, -70101, -70102, -70103
TYP_RESPONSE = "application/vnd.hyprstream.response-proof+cwt"
C_SCHEMA_ID, C_RESPONSE_BINDING = -70002, -70004
KTY_OKP, KTY_AKP = 1, 7
CRV_ED25519 = 6
ED25519_PUB_BYTES = 32       # OKP/Ed25519 x
ML_DSA_65_PUB_BYTES = 1952   # AKP/ML-DSA-65 pub (RFC 9964)
# Exact COSE_Key map key sets (closed, per the CDDL): OKP/Ed25519 and
# AKP/ML-DSA-65. Surplus or missing fields deny.
OKP_KEY_FIELDS = {1, 2, 3, -1, -2}
AKP_KEY_FIELDS = {1, 2, 3, -1}


def unattributed_keyset_correspondence(keyset, components):
    """For an unattributed proof, the embedded `hs_unattributed_key_set` is the
    ONLY key material. This fully validates the key shape (the pycddl AKP panic
    means the gate cannot lean on the CDDL for it): require exact ordered 1:1
    correspondence with the plan components — keyset[i] matches component i by
    kid, alg, key type / curve (OKP/Ed25519) or parameter set (AKP/ML-DSA-65),
    the closed COSE_Key field set, and the exact public-key byte length. Return
    ({kid: public_bytes}, None) on success, or (None, reason) on failure. The
    returned public keys are what each unattributed signature MUST verify
    against."""
    if not isinstance(keyset, list):
        return None, "hs_unattributed_key_set is not an array"
    # B8: the frozen cap is 1..2 key-set elements (the pycddl AKP workaround
    # drops the CDDL `1*2` occurrence bound, so re-impose it here).
    if not (1 <= len(keyset) <= 2):
        return None, f"key set has {len(keyset)} elements, must be 1..2"
    if len(keyset) != len(components):
        return None, (f"key set has {len(keyset)} elements, plan has "
                      f"{len(components)} components")
    embedded = {}
    for i, (_grp, alg, kid) in enumerate(components):
        key = keyset[i]
        if not isinstance(key, dict):
            return None, f"key set element {i} is not a COSE_Key map"
        if key.get(2) != kid:
            return None, f"key set element {i} kid does not match plan component {i}"
        if key.get(3) != alg:
            return None, f"key set element {i} alg does not match plan component {i}"
        if alg == ALG_ED25519:
            if set(key.keys()) != OKP_KEY_FIELDS:
                return None, f"key set element {i} is not a closed OKP COSE_Key"
            if key.get(1) != KTY_OKP or key.get(-1) != CRV_ED25519:
                return None, f"key set element {i} is not OKP/Ed25519"
            pub = key.get(-2)
            if not isinstance(pub, (bytes, bytearray)) or len(pub) != ED25519_PUB_BYTES:
                return None, f"key set element {i} Ed25519 x must be {ED25519_PUB_BYTES} bytes"
        elif alg == ALG_ML_DSA_65:
            if set(key.keys()) != AKP_KEY_FIELDS:
                return None, f"key set element {i} is not a closed AKP COSE_Key"
            if key.get(1) != KTY_AKP:
                return None, f"key set element {i} is not AKP/ML-DSA-65"
            pub = key.get(-1)
            if not isinstance(pub, (bytes, bytearray)) or len(pub) != ML_DSA_65_PUB_BYTES:
                return None, f"key set element {i} ML-DSA-65 pub must be {ML_DSA_65_PUB_BYTES} bytes"
        else:
            return None, f"key set element {i} uses algorithm {alg} outside the profile"
        # B7: key by (alg, kid) — the normative uniqueness key — so a hybrid plan
        # reusing one kid across algorithms does not overwrite (or crash).
        embedded[(alg, kid)] = bytes(pub)
    return embedded, None


def ml_dsa_verify(public: bytes, message: bytes, signature: bytes) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        key_path = Path(tmp) / "pub.der"
        # Rebuild the SPKI: SEQUENCE { SEQUENCE { OID }, BIT STRING }
        oid = bytes.fromhex("06096086480165030403") + b"\x12"  # id-ml-dsa-65
        alg_id = b"\x30" + bytes([len(oid)]) + oid
        bitstring_body = b"\x00" + public
        bitstring = b"\x03" + _der_len(len(bitstring_body)) + bitstring_body
        body = alg_id + bitstring
        spki = b"\x30" + _der_len(len(body)) + body
        key_path.write_bytes(spki)
        msg_path, sig_path = Path(tmp) / "m", Path(tmp) / "s"
        msg_path.write_bytes(message)
        sig_path.write_bytes(signature)
        proc = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(key_path),
                "-keyform",
                "DER",
                "-rawin",
                "-in",
                str(msg_path),
                "-sigfile",
                str(sig_path),
            ],
            capture_output=True,
        )
        return proc.returncode == 0


def _der_len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    raw = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(raw)]) + raw


def main() -> None:
    vectors_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "vectors"
    )
    keys = json.loads((vectors_dir / "proof-v1-keys.json").read_text())["keys"]
    ed_by_kid = {
        bytes.fromhex(k["kid_hex"]): bytes.fromhex(k["public_hex"])
        for k in keys["ed25519"]
    }
    ml_by_kid = {
        bytes.fromhex(k["kid_hex"]): bytes.fromhex(k["public_hex"])
        for k in keys["ml_dsa_65"]
    }

    positive = json.loads((vectors_dir / "proof-v1-positive.json").read_text())
    negative = json.loads((vectors_dir / "proof-v1-negative.json").read_text())

    def check_digest(vec):
        raw = bytes.fromhex(vec["cbor_hex"])
        if hashlib.sha256(raw).hexdigest() != vec["sha256"]:
            fail(f"{vec['id']}: sha256 mismatch")
        if len(raw) != vec["size_bytes"]:
            fail(f"{vec['id']}: size mismatch")
        return raw

    def plan_components(plan):
        out = []
        for grp in plan:
            for comp in grp[3]:
                out.append((grp[1], comp[1], comp[2]))
        return out

    def verify_component(vec, alg, kid, tbs, sig, pub_override=None):
        # pub_override forces the embedded unattributed key (self-asserted key
        # set); otherwise the enrolled/anchored test key is resolved by kid.
        if alg == ALG_ED25519:
            pub = pub_override if pub_override is not None else ed_by_kid.get(kid)
            if pub is None:
                fail(f"{vec['id']}: unknown Ed25519 kid {kid!r}")
                return
            try:
                Ed25519PublicKey.from_public_bytes(pub).verify(sig, tbs)
            except InvalidSignature:
                fail(f"{vec['id']}: Ed25519 signature does not verify")
        elif alg == ALG_ML_DSA_65:
            pub = pub_override if pub_override is not None else ml_by_kid.get(kid)
            if pub is None:
                fail(f"{vec['id']}: unknown ML-DSA-65 kid {kid!r}")
                return
            if not ml_dsa_verify(pub, tbs, sig):
                fail(f"{vec['id']}: ML-DSA-65 signature does not verify")
        else:
            fail(f"{vec['id']}: algorithm {alg} is not in the profile")

    for vec in positive["vectors"]:
        raw = check_digest(vec)
        try:
            obj = decode(raw)
        except StrictError as exc:
            fail(f"{vec['id']}: not deterministic CBOR: {exc}")
            continue
        if not isinstance(obj, list) or len(obj) != 4:
            fail(f"{vec['id']}: not a 4-element COSE structure")
            continue
        body_protected, unprotected, payload, tail = obj
        if unprotected:
            fail(f"{vec['id']}: unprotected header is not empty")
        try:
            body_hdr = decode(body_protected)
            claims = decode(payload)
        except StrictError as exc:
            fail(f"{vec['id']}: protected/payload not deterministic: {exc}")
            continue

        # B1: for a response proof carrying a non-null response_binding, claim
        # -70002 (response root type id) MUST equal the binding's root_type_id.
        if body_hdr.get(H_TYP) == TYP_RESPONSE and isinstance(claims, dict):
            rb = claims.get(C_RESPONSE_BINDING)
            if isinstance(rb, dict):
                if claims.get(C_SCHEMA_ID) != rb.get(1):
                    fail(f"{vec['id']}: response -70002 {claims.get(C_SCHEMA_ID)!r} != "
                         f"response_binding root_type_id {rb.get(1)!r}")

        plan = body_hdr.get(H_PLAN)
        if plan is None:
            fail(f"{vec['id']}: no signature_plan in the body protected headers")
            continue
        components = plan_components(plan)
        # B2: every (alg, kid) pair is unique across the whole plan, regardless
        # of group ID — one key must not sign under two logical groups.
        plan_keys = [(alg, kid) for (_grp, alg, kid) in components]
        if len(plan_keys) != len(set(plan_keys)):
            fail(f"{vec['id']}: plan repeats an (alg, kid) across groups")
        # B6: group IDs MUST be unique and strictly ascending across the plan.
        group_ids = [grp[1] for grp in plan]
        if group_ids != sorted(set(group_ids)) or len(group_ids) != len(set(group_ids)):
            fail(f"{vec['id']}: group IDs must be unique and strictly ascending, got {group_ids}")

        # B4: an unattributed proof (embedded hs_unattributed_key_set) is verified
        # against its OWN key set, which must correspond 1:1 in order to the plan.
        # `embedded` maps (alg, kid) -> the embedded public key that each signature
        # must verify against (None for authenticated proofs, which use enrolled keys).
        embedded = None
        if H_KEYSET in body_hdr:
            embedded, err = unattributed_keyset_correspondence(body_hdr[H_KEYSET], components)
            if err is not None:
                fail(f"{vec['id']}: unattributed key set {err}")

        if vec["structure"] == "COSE_Sign1":
            # A COSE_Sign1 carries exactly one signature, so it must cover the
            # plan EXACTLY: the plan must have exactly one component. This
            # rejects a hybrid plan (two components) carried in a Sign1, i.e. a
            # hybrid-to-classical downgrade that a "signature occurs in the plan"
            # test alone would miss.
            if len(components) != 1:
                fail(
                    f"{vec['id']}: COSE_Sign1 plan has {len(components)} components; "
                    "a single signature must cover a single-component plan exactly"
                )
            tbs = enc(["Signature1", body_protected, b"", payload])
            alg, kid, grp = body_hdr[H_ALG], body_hdr[H_KID], body_hdr[H_GROUP]
            if (grp, alg, kid) not in components:
                fail(f"{vec['id']}: signature does not match a plan component")
            verify_component(vec, alg, kid, tbs, tail,
                             pub_override=(embedded.get((alg, kid)) if embedded else None))
        else:
            seen = []
            seen_keys = set()
            for entry in tail:
                sprot, sunprot, sig = entry
                if sunprot:
                    fail(f"{vec['id']}: signature unprotected header is not empty")
                shdr = decode(sprot)
                alg, kid, grp = shdr[H_ALG], shdr[H_KID], shdr[H_GROUP]
                if (grp, alg, kid) not in components:
                    fail(f"{vec['id']}: entry {kid!r} matches no plan component")
                if (grp, alg, kid) in seen:
                    fail(f"{vec['id']}: duplicate plan component {kid!r}")
                # B2: a key may not sign under two groups — uniqueness is keyed
                # on (alg, kid), NOT (group, alg, kid).
                if (alg, kid) in seen_keys:
                    fail(f"{vec['id']}: (alg, kid) {kid!r} signs under two groups")
                seen.append((grp, alg, kid))
                seen_keys.add((alg, kid))
                verify_component(
                    vec, alg, kid, enc(["Signature", body_protected, sprot, b"", payload]), sig,
                    pub_override=(embedded.get((alg, kid)) if embedded else None),
                )
            if len(seen) != len(components):
                fail(f"{vec['id']}: signature entries do not cover the plan exactly")

    # S1: content-identity uniqueness across signer groups — a resolved (alg, raw
    # public key) may participate in at most one logical signer group. Every
    # positive satisfies it (distinct algorithms in a hybrid group, distinct keys
    # across P-5's groups).
    def resolve_components(vec):
        obj = decode(bytes.fromhex(vec["cbor_hex"]))
        body = decode(obj[0])
        plan = body.get(H_PLAN) or []
        ks = body.get(H_KEYSET)
        ks_by_kid = {}
        if ks:
            for k in ks:
                ks_by_kid[k.get(2)] = cose_key_public(k)
        out = []
        for g in plan:
            for comp in g[3]:
                alg, kid = comp[1], comp[2]
                if ks:
                    pub = ks_by_kid.get(kid)
                else:
                    pub = ed_by_kid.get(kid) if alg == ALG_ED25519 else ml_by_kid.get(kid)
                if pub is not None:
                    out.append((g[1], alg, pub))
        return out

    for vec in positive["vectors"]:
        aliases = cross_group_key_aliases(resolve_components(vec))
        if aliases:
            fail(f"{vec['id']}: a resolved key participates in >1 signer group (content alias)")

    positive_bytes = {v["cbor_hex"] for v in positive["vectors"]}
    for vec in negative["vectors"]:
        check_digest(vec)
        if vec["expect"] != "deny":
            fail(f"{vec['id']}: negative vector must expect deny")
        if vec["id"] != "N-2" and vec["cbor_hex"] in positive_bytes:
            fail(f"{vec['id']}: negative vector duplicates a positive vector")

    # C1: recompute the frozen replay-namespace thumbprints and assert they match
    # the shipped cross-implementation vectors — SHA-256 over the deterministic
    # encoding of a CBOR array whose first element is the domain-separator text.
    tp_path = vectors_dir / "proof-v1-thumbprints.json"
    if tp_path.exists():
        tp = json.loads(tp_path.read_text())
        sep = tp["domain_separators"]
        a = tp["authenticated"]
        auth_pre = enc([sep["authenticated"], a["suite_id"],
                        [bytes.fromhex(h) for h in a["component_public_keys_hex"]],
                        a["enrollment_epoch"]])
        if hashlib.sha256(auth_pre).hexdigest() != a["thumbprint_sha256"]:
            fail("authenticated replay thumbprint does not match its inputs")
        if auth_pre.hex() != a["preimage_hex"]:
            fail("authenticated replay preimage encoding drifted")
        # T1: the authenticated `enrollment_epoch` is derived from the cnf-resolved
        # PRIMARY enrollment record (not the vector literal) — resolve it by content,
        # validate it, and reproduce the published thumbprint from the record's epoch.
        cpath = vectors_dir / "proof-v1-credentials.json"
        if cpath.exists():
            cdoc = json.loads(cpath.read_text())
            a_pubs = [bytes.fromhex(h) for h in a["component_public_keys_hex"]]
            a_tp = base64.urlsafe_b64encode(
                hashlib.sha256(enc([a["suite_id"], a_pubs])).digest()).rstrip(b"=").decode()
            prec = resolve_primary_enrollment(cdoc, a_tp)
            hcl = cdoc["credentials"]["hybrid"]["claims"]
            for e in validate_primary_enrollment(prec, a_tp, hcl.get("tenant"), hcl.get("sub"),
                                                 cdoc.get("verifier_now")):
                fail(f"authenticated primary enrollment: {e}")
            if prec is not None and authenticated_replay_thumbprint(
                    sep["authenticated"], a["suite_id"], a_pubs,
                    prec["enrollment_epoch"]).hex() != a["thumbprint_sha256"]:
                fail("authenticated replay thumbprint must be reproduced from the primary record's epoch")
        # Unattributed: recompute from P-1's plan + embedded key set.
        p1 = next((v for v in positive["vectors"] if v["id"] == tp["unattributed"]["from_vector"]), None)
        if p1 is None:
            fail("unattributed thumbprint source vector missing")
        else:
            import copy as _copy
            hdr = decode(decode(bytes.fromhex(p1["cbor_hex"]))[0])
            plan0, ks0 = hdr[H_PLAN], hdr[H_KEYSET]
            # M1: content-bound derivation (per-group suite + ordered public keys;
            # group_id/kid normalized out).
            ks_pre = unattributed_replay_preimage(sep["key_set"], plan0, ks0)
            base_tp = hashlib.sha256(ks_pre).hexdigest()
            if base_tp != tp["unattributed"]["thumbprint_sha256"]:
                fail("unattributed replay thumbprint does not match P-1's suite/ordered keys")
            if ks_pre.hex() != tp["unattributed"]["preimage_hex"]:
                fail("unattributed replay preimage encoding drifted")

            def _tp(plan, keyset):
                return hashlib.sha256(unattributed_replay_preimage(sep["key_set"], plan, keyset)).hexdigest()

            # Collapse (relabel -> SAME thumbprint), each label mutated independently.
            gid_only = _copy.deepcopy(plan0); gid_only[0][1] = gid_only[0][1] + 1000
            if _tp(gid_only, ks0) != base_tp:
                fail("M1: a group_id-only relabel must map to the SAME replay thumbprint")
            kid_only_plan = _copy.deepcopy(plan0); kid_only_ks = _copy.deepcopy(ks0)
            kid_only_plan[0][3][0][2] = b"relabel-kid"
            kid_only_ks[0][2] = b"relabel-kid"
            if _tp(kid_only_plan, kid_only_ks) != base_tp:
                fail("M1: a kid-only relabel must map to the SAME replay thumbprint")
            # No over-collapse (distinct crypto identity -> DIFFERENT thumbprint).
            bad_key = _copy.deepcopy(ks0)
            pk = bytearray(cose_key_public(bad_key[0])); pk[0] ^= 0x01
            bad_key[0][-2 if bad_key[0].get(1) == 1 else -1] = bytes(pk)
            if _tp(plan0, bad_key) == base_tp:
                fail("M1: a public-key byte change must change the replay thumbprint")
            bad_suite = _copy.deepcopy(plan0); bad_suite[0][2] = "hs-cose-sign-ed25519-mldsa65-wns-v1"
            if _tp(bad_suite, ks0) == base_tp:
                fail("M1: a suite change must change the replay thumbprint")

            # R1: canonical group-order evidence — two cryptographically valid
            # 2-group unattributed proofs (A,B and B,A) map to the SAME namespace.
            ev = tp.get("group_order_canonicalization")
            if ev is None:
                fail("R1: group_order_canonicalization evidence is missing")
            else:
                order_tps = []
                for name in ("order_ab", "order_ba"):
                    o = decode(bytes.fromhex(ev[name]["cbor_hex"]))
                    hdr = decode(o[0])
                    embedded, err = unattributed_keyset_correspondence(hdr[H_KEYSET], plan_components(hdr[H_PLAN]))
                    if err is not None:
                        fail(f"R1 {name}: key-set correspondence fails: {err}")
                    # every entry signature verifies against its embedded key.
                    for entry in o[3]:
                        sh = decode(entry[0])
                        a, k = sh[H_ALG], sh[H_KID]
                        tbs = enc(["Signature", o[0], entry[0], b"", o[2]])
                        if a == ALG_ED25519 and embedded is not None:
                            try:
                                Ed25519PublicKey.from_public_bytes(embedded[(a, k)]).verify(entry[2], tbs)
                            except InvalidSignature:
                                fail(f"R1 {name}: entry {k!r} signature must verify (valid proof)")
                    pre = unattributed_replay_preimage(sep["key_set"], hdr[H_PLAN], hdr[H_KEYSET])
                    if pre.hex() != ev[name]["preimage_hex"]:
                        fail(f"R1 {name}: preimage encoding drifted")
                    order_tps.append(hashlib.sha256(pre).hexdigest())
                if order_tps[0] != order_tps[1]:
                    fail("R1: A,B and B,A group orders MUST map to the same replay namespace")
                if not ev.get("same_namespace"):
                    fail("R1: group_order evidence must declare same_namespace true")
    else:
        fail("proof-v1-thumbprints.json is missing (C1 replay thumbprint vector)")

    # F1/F2: authenticated credential context. Every advertised credential is a
    # real issuer-signed at+jwt, temporally valid at the frozen verifier_now, and
    # each authenticated positive's credential_hash is SHA-256 over the exact
    # credential bytes with a cnf resolving to that proof's PRIMARY signer group.
    cred_path = vectors_dir / "proof-v1-credentials.json"
    if not cred_path.exists():
        fail("proof-v1-credentials.json is missing (F2 authenticated credential context)")
    else:
        cd = json.loads(cred_path.read_text())
        now = cd.get("verifier_now")
        if not isinstance(now, int):
            fail("verifier_now must be a declared integer")
        issuer_pub = bytes.fromhex(cd["issuer"]["public_hex"])
        issuer_kid = cd["issuer"]["kid"]
        pos_by_id = {v["id"]: v for v in positive["vectors"]}

        def b64u_dec(s: str) -> bytes:
            return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

        def signer_suite_tp(suite, pubs) -> bytes:
            return hashlib.sha256(enc([suite, list(pubs)])).digest()

        def verify_at_jwt(token: str, label: str):
            parts = token.split(".")
            if len(parts) != 3:
                fail(f"{label}: not a compact JWS"); return None
            hp, pp, sp = parts
            header = json.loads(b64u_dec(hp))
            claims = json.loads(b64u_dec(pp))
            if (header.get("typ") != "at+jwt" or header.get("alg") != "EdDSA"
                    or header.get("kid") != issuer_kid):
                fail(f"{label}: protected header not exact at+jwt/EdDSA/{issuer_kid}: {header}")
            try:
                Ed25519PublicKey.from_public_bytes(issuer_pub).verify(
                    b64u_dec(sp), f"{hp}.{pp}".encode("ascii"))
            except InvalidSignature:
                fail(f"{label}: issuer Ed25519 signature does not verify")
            for req in ("iss", "sub", "aud", "iat", "exp", "jti", "client_id", "tenant", "clearance", "cnf"):
                if req not in claims:
                    fail(f"{label}: missing required claim {req!r}")
            if not claims.get("iss") or not claims.get("sub"):
                fail(f"{label}: empty issuer/subject")
            if not isinstance(claims.get("client_id"), str) or not claims.get("client_id"):
                fail(f"{label}: client_id must be a non-empty string (RFC 9068)")
            for te in validate_tenant(claims.get("tenant")):
                fail(f"{label}: {te}")
            if not (claims.get("iat", 0) <= now < claims.get("exp", 0)):
                fail(f"{label}: not temporally valid at verifier_now {now}")
            if "hs_signer_suite" not in (claims.get("cnf") or {}):
                fail(f"{label}: cnf lacks the hs_signer_suite confirmation")
            return claims

        for name, cred in cd["credentials"].items():
            claims = verify_at_jwt(cred["token"], f"credential {name}")
            if hashlib.sha256(cred["token"].encode("ascii")).hexdigest() != cred["token_sha256"]:
                fail(f"credential {name}: published token_sha256 does not match the token bytes")
            # K1: a sid-bearing credential must map to a valid authoritative session.
            _sess, serrs = validate_session(cd, claims or {}, now)
            for se in serrs:
                fail(f"credential {name} session: {se}")

        for pid, credname in cd["positive_to_credential"].items():
            cred = cd["credentials"][credname]
            v = pos_by_id.get(pid)
            if v is None:
                fail(f"credential mapping names unknown positive {pid}"); continue
            obj = decode(bytes.fromhex(v["cbor_hex"]))
            pclaims = decode(obj[2])
            ch = pclaims.get(-70001)  # credential_hash
            if ch != hashlib.sha256(cred["token"].encode("ascii")).digest():
                fail(f"{pid}: credential_hash != SHA-256(mapped {credname} credential)")
            cred_claims = json.loads(b64u_dec(cred["token"].split(".")[1]))
            if cred_claims.get("aud") != pclaims.get(3):
                fail(f"{pid}: credential aud != the proof's aud")
            cnf_tp = b64u_dec(cred_claims["cnf"]["hs_signer_suite"])
            # The cnf MUST resolve to exactly one plan group (the primary), by
            # suite ID + exact ordered component keys (credential-profile §5).
            plan = decode(obj[0]).get(H_PLAN) or []
            matches = []
            for grp in plan:
                pubs, ok = [], True
                for comp in grp[3]:
                    alg, kid = comp[1], comp[2]
                    pub = ed_by_kid.get(kid) if alg == ALG_ED25519 else ml_by_kid.get(kid)
                    if pub is None:
                        ok = False; break
                    pubs.append(pub)
                if ok and signer_suite_tp(grp[2], pubs) == cnf_tp:
                    matches.append(grp[1])
            if len(matches) != 1:
                fail(f"{pid}: cnf must resolve to exactly one primary plan group; matched {matches}")
            # T1: the cnf primary group MUST resolve to an authoritative PRIMARY
            # enrollment record (active, role=primary, tenant/principal-coherent).
            cnf_b64 = cred_claims["cnf"]["hs_signer_suite"]
            prec = resolve_primary_enrollment(cd, cnf_b64)
            for e in validate_primary_enrollment(prec, cnf_b64, cred_claims.get("tenant"),
                                                 cred_claims.get("sub"), now):
                fail(f"{pid} primary enrollment: {e}")
            # Q1: every ADDITIONAL (approver) signer group — content-resolved, not the
            # primary cnf group — must validate against an authoritative approver
            # enrollment record (§5).
            cred_tenant = cred_claims.get("tenant")
            for grp in plan:
                pubs, ok = [], True
                for comp in grp[3]:
                    alg, kid = comp[1], comp[2]
                    p = ed_by_kid.get(kid) if alg == ALG_ED25519 else ml_by_kid.get(kid)
                    if p is None:
                        ok = False; break
                    pubs.append(p)
                if not ok:
                    continue
                gtp = base64.urlsafe_b64encode(signer_suite_tp(grp[2], pubs)).rstrip(b"=").decode()
                if gtp == cred_claims["cnf"]["hs_signer_suite"]:
                    continue  # the primary cnf group
                rec = resolve_approver_enrollment(cd, gtp)
                for e in validate_approver_enrollment(rec, gtp, cred_tenant, now):
                    fail(f"{pid} approver group_id {grp[1]}: {e}")
            # K1: proof exp <= credential exp, and for a sid-bearing credential also
            # <= the authoritative session exp.
            pexp = pclaims.get(4)
            if isinstance(pexp, int) and pexp > cred_claims.get("exp", 0):
                fail(f"{pid}: proof exp {pexp} exceeds credential exp {cred_claims.get('exp')}")
            sess, _ = validate_session(cd, cred_claims, now)
            if sess is not None and isinstance(pexp, int) and pexp > sess.get("expiry", 0):
                fail(f"{pid}: proof exp {pexp} exceeds session exp {sess.get('expiry')}")

    total = len(positive["vectors"]) + len(negative["vectors"])
    if FAILURES:
        for line in FAILURES:
            print(f"FAIL {line}")
        print(f"{len(FAILURES)} failure(s) across {total} vectors")
        sys.exit(1)
    print(f"OK: {len(positive['vectors'])} positive, {len(negative['vectors'])} negative vectors check out")


if __name__ == "__main__":
    main()
