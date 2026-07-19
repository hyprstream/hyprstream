# Encrypted receipt and selective-disclosure profile

**Status:** pre-construction (2026-07-17). Conformance closure view. The selective-disclosure and encrypted-receipt proof-plane path is owned by #1072, with pseudonymity/KYC-minimization constraints from #928.

## Requirement

Public proof records use commitments; detailed receipts are selectively disclosed or encrypted to authorized parties. An encrypted receipt is addressed only to a party authorized by the active profile and is bound to the same canonical `ResourceIntent` digest as the manifest it accompanies. A public receipt or checkpoint must not reveal a raw anonymous token, a holder DID, a stable client key, or a linkable entitlement identifier.

Traced obligations: RA-REQ-021 (public proof uses commitments; receipts selective/encrypted — spec-only), RA-REQ-022 (public receipt reveals nothing linkable — implemented via vector), RA-REQ-023 (encrypted receipt authorized recipient + digest binding — spec-only).

## Profile shape

```
PublicReceipt = {
  resource_id, manifest_cid, profile,
  owner_commitment?,        ; present for committed/anonymous profiles
  entitlement_commitment?,  ; present for anonymous-entitlement payer
  issuer_liability_aggregate?,
  canonical_digest,
}

EncryptedReceipt = {
  recipient: authorized-party,       ; per active profile
  canonical_digest: bstr .size 32,   ; == manifest canonical ResourceIntent digest
  ciphertext,                        ; sealed detailed receipt
}
```

The public receipt carries commitments and aggregates only. The encrypted receipt is bound to the identical canonical digest and addressed to a profile-authorized recipient.

## Negative controls (structural, in the vector set)

- `public-receipt-reveals-holder`: a public receipt containing a `holder_did` rejects (`receipt:public-reveals-holder`).
- `anonymous-fabricates-did`: an anonymous profile carrying a stable holder DID rejects (`profile:anonymous-fabricates-did`).

## Selective-disclosure constraints (from #928)

- Reveal only the field minimum required by the recipient's authority.
- Holder-controlled disclosure: the holder chooses which claims to present; the origin learns only authorized operations.
- No stable holder DID crosses into an origin audit record, a public receipt, or a trust store under an anonymous profile.

## Status

The encrypted-receipt addressing, digest binding, and selective-disclosure proof-plane implementation are pending #1072/#928. The structural controls above are enforced by the boundary checker; the cryptographic encryption is specification-only until those issues land.
