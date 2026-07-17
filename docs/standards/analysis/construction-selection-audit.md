# PQ anonymous issuance construction-selection audit

**Assessment date:** 2026-07-17. **Disposition:** no construction selected;
production issuance and redemption remain disabled. This is an evidence record,
not a cryptographic review, IETF adoption claim, or request for a codepoint.

## Required bar

A selectable PQ leg has to provide all of the following as one reviewable
package:

1. post-quantum authorization unforgeability and issuance/redemption
   unlinkability under an explicit concurrency and collusion model;
2. a blindness-preserving issuance protocol, not an ordinary signature over a
   visible token, a hybrid-signed classical entitlement, or an opaque wrapper;
3. fixed algorithms and parameters, canonical request/response/token encodings,
   test vectors, and an implementation suitable for independent review;
4. exact Privacy Pass challenge and token semantics, including accepted-current
   issuer/origin/state/policy bindings and dedicated issuer-key purposes; and
5. an analyzed AND composition with the classical Privacy Pass leg in which
   neither leg can be stripped, crossed, substituted, or used to upgrade a
   merely classical entitlement.

Meeting only the primitive-level blindness definition is insufficient. The
profile also needs an implementation and composition review before it can mint
the `PqHybrid` authorization assurance used by Hyprstream.

## Primary-source audit

| Candidate/source | Evidence | Disposition for this profile |
|---|---|---|
| [RFC 9578](https://www.rfc-editor.org/rfc/rfc9578.html) and [RFC 9474](https://www.rfc-editor.org/rfc/rfc9474.html) | RFC 9578 specifies a P-384 VOPRF token and RSA blind-signature token. RFC 9474 Section 7.7 explicitly says its RSA construction is not post-quantum ready. | Retain as the standard classical leg only. It cannot be the PQ anonymous leg. |
| [FIPS 204](https://doi.org/10.6028/NIST.FIPS.204) | ML-DSA is a standardized post-quantum *ordinary digital signature*. It has no blindness or issuance/redemption unlinkability protocol. | Reject visible ML-DSA signing, including a hybrid signature over a classical entitlement. |
| [Post-Quantum Privacy Pass via Post-Quantum Anonymous Credentials](https://eprint.iacr.org/2023/414) and its [reference implementation](https://github.com/guruvamsi-policharla/zkdilithium) | The ePrint calls the construction “plausibly post-quantum,” uses a modified Dilithium and a conjectured 115-bit STARK setting, and reports 85–175 KB proofs. Section 4 says the benchmarked Winterfell backend did not support zero knowledge. The implementation labels itself an academic proof of concept that has not received careful code review and is not production ready. | Valuable Privacy Pass-specific research, but it does not supply a reviewed, production-capable zero-knowledge implementation or a current Privacy Pass token-type profile. Do not select it. |
| [Improved Lattice Blind Signatures from Recycled Entropy](https://doi.org/10.1007/978-3-032-01855-7_16), CRYPTO 2025 | Peer-reviewed blind-signature research with standard lattice assumptions and a reported implementation; signatures remain roughly 40 KB. | A serious primitive candidate, but no CFRG/Privacy Pass profile, interoperable maintained implementation, fixed Hyprstream parameter set, canonical Privacy Pass encoding, vectors, side-channel review, or analysis of this exact AND composition was found. Primitive review alone is not the required profile review. |
| [Concretely Efficient Blind Signatures Based on VOLE-in-the-Head Proofs and the MAYO Trapdoor](https://www.usenix.org/conference/usenixsecurity26/presentation/baum), USENIX Security 2026 | PoMFRIT is described by its authors as “plausibly post-quantum.” It combines VOLE-in-the-head with MAYO; the paper discusses a conjectured one-more-MAYO variant and implementation/security trade-offs. MAYO remained a NIST additional-signature candidate at this assessment date. | Not a standardized or reviewed Privacy Pass construction and not an acceptable basis for a production assurance claim. Track as research only. |
| [Anonymous Rate-Limited Credentials Cryptography](https://datatracker.ietf.org/doc/html/draft-ietf-privacypass-arc-crypto-01) | An active Privacy Pass WG draft with concrete encodings and vectors, but its group construction is classical. Section 7.3 explains that a quantum discrete-log attack can partition/link clients. | Useful protocol-shape input, not the PQ anonymous leg. |
| [Late binding with PQ privacy](https://datatracker.ietf.org/meeting/interim-2026-privacypass-01/materials/slides-interim-2026-privacypass-01-sessa-late-binding-with-pq-unlinkability-00), Privacy Pass interim, 2026-05-13 | Records that current ACT is PQ-unlinkable but forgeable after breaking BBS, and says the proposed late-binding idea lacks concrete unforgeability/unlinkability bounds and needs end-to-end analysis. | Confirms the standards community is still separating PQ privacy from PQ unforgeability. It is not a selected construction. |
| [PACT workshop summary](https://datatracker.ietf.org/meeting/interim-2026-privacypass-01/materials/slides-interim-2026-privacypass-01-sessa-pact-workshop-update-00), 2026-05-13 | Places post-quantum adversaries in scope for unlinkability but presents a problem statement and preliminary realization directions rather than a complete PQ-unforgeable issuance construction. | Relevant requirements input; no implementable PQ token suite follows from it. |

The active [Privacy Pass WG document list](https://datatracker.ietf.org/wg/privacypass/)
was retrieved at 2026-07-17T17:43:08Z (datatracker release 12.65.0,
`3203f91`). Its complete active-document inventory was
`draft-ietf-privacypass-arc-crypto-01`,
`draft-ietf-privacypass-arc-protocol-01`,
`draft-ietf-privacypass-auth-scheme-extensions-03`,
`draft-ietf-privacypass-batched-tokens-08`,
`draft-ietf-privacypass-expiration-extension-00`, and
`draft-ietf-privacypass-public-metadata-issuance-03`. None specifies an adopted
post-quantum-unforgeable anonymous token issuance suite; the two ARC drafts are
the only anonymous-credential construction in that inventory and use the
classical group construction discussed above. This enumerated snapshot makes
the negative finding reproducible even as the live WG page changes. Recent
research is promising, but selecting its algorithms, parameters, encodings, and
composition here would invent a protocol rather than implement a reviewed one.

## Implemented boundary

`../registry/pq-anonymous-issuance.json` defines a repository-local experimental
record boundary for challenge, issuance request, issuance response, and token
fixtures. It fixes the exact logical field set and exact ordered AND leg roles,
but deliberately assigns no Privacy Pass token type, PQ algorithm, PQ parameters,
or production wire encoding. `../vectors/pq-anonymous-boundary-v1.json` is a
deterministic non-cryptographic fixture. Its baseline is structurally valid and
then reaches the mandatory `construction-unselected` refusal; no vector is a
mintable or redeemable token.

The checker proves the boundary rejects missing, duplicate, reordered, unknown,
crossed, or stripped legs; holder-identifying fields; stale or crossed state;
policy rollback; expired or crossed expiry; wrong origin; crossed redemption or
resource-profile bindings; malformed hex; and key-purpose or key-reuse
substitutions. These are parser/profile negative controls, not evidence of
cryptographic issuance.

## Precise dependency

Production work remains blocked until a candidate supplies: (1) a stable paper
and security model covering concurrent blindness/unlinkability and one-more
unforgeability against quantum adversaries; (2) fixed reviewed parameters and a
maintained constant-time implementation; (3) canonical Privacy Pass request,
response, token, configuration, key-rotation/drain, and error encodings with
vectors; (4) exact accepted-current state/policy and purpose-separated key
bindings; (5) an analysis of the classical-plus-PQ AND composition and the stated
issuer/origin/attester collusion model; and (6) fresh independent cryptographic,
privacy, and implementation review. CFRG and Privacy Pass WG engagement is the
standards dependency; it is not replaced by locally assigning an experimental
codepoint.
