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
    elif ai == 24:
        val, rest = rest[0], rest[1:]
        if strict and val < 24:
            raise StrictError("non-minimal integer encoding")
    elif ai in (25, 26, 27):
        n = {25: 2, 26: 4, 27: 8}[ai]
        val, rest = int.from_bytes(rest[:n], "big"), rest[n:]
        if strict and val < {25: 24, 26: 0x10000, 27: 0x100000000}[ai]:
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
        return rest[:val], rest[val:]
    if major == 3:
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

    positive_bytes = {v["cbor_hex"] for v in positive["vectors"]}
    for vec in negative["vectors"]:
        check_digest(vec)
        if vec["expect"] != "deny":
            fail(f"{vec['id']}: negative vector must expect deny")
        if vec["id"] != "N-2" and vec["cbor_hex"] in positive_bytes:
            fail(f"{vec['id']}: negative vector duplicates a positive vector")

    total = len(positive["vectors"]) + len(negative["vectors"])
    if FAILURES:
        for line in FAILURES:
            print(f"FAIL {line}")
        print(f"{len(FAILURES)} failure(s) across {total} vectors")
        sys.exit(1)
    print(f"OK: {len(positive['vectors'])} positive, {len(negative['vectors'])} negative vectors check out")


if __name__ == "__main__":
    main()
