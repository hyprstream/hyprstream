# Standards-gap matrix

**Assessment date:** 2026-07-17. Links are primary sources. “Gap” is not a request for an allocation and does not claim WG adoption.

| Source | Status at assessment | Reused surface | Gap / boundary for this work | Disposition / owner |
|---|---|---|---|---|
| [RFC 9576](https://www.rfc-editor.org/info/rfc9576), Privacy Pass Architecture | Informational RFC | client/issuer/origin roles; issuance/redemption separation | Does not select a PQ anonymous issuance construction for this profile | #1060 must select/review one; no selection here |
| [RFC 9577](https://www.rfc-editor.org/info/rfc9577), Privacy Pass HTTP Authentication Scheme | Proposed Standard RFC | classical HTTP auth/challenge/token use where applicable | HTTP authentication is not a carrier-neutral inner capability and cannot convey `PqHybrid` by itself | outer/coarse admission only; #1063/#726 |
| [RFC 9578](https://www.rfc-editor.org/info/rfc9578), VOPRF and Blind RSA Tokens | Proposed Standard RFC | classical token terminology and existing token types | Listed issuance protocols are classical; a signature wrapper does not provide PQ anonymous authorization | #1060 construction/review blocker |
| [RFC 9794](https://www.rfc-editor.org/info/rfc9794), PQ/T terminology | Informational RFC | hybrid terminology and claim discipline | Does not specify anonymous tokens, binding, or MOQT authorization | terminology only; #1059 |
| [draft-ietf-moq-transport-19](https://www.ietf.org/archive/id/draft-ietf-moq-transport-19.html) | IETF Internet-Draft, work in progress | ordinary MOQT Objects, forwarding, caching, session/track mechanics | No mandatory Hyprstream PQ-anonymous extension is selected | use ordinary payloads; #726 |
| [draft-ietf-moq-privacy-pass-auth-03](https://www.ietf.org/archive/id/draft-ietf-moq-privacy-pass-auth-03.html) | IETF Internet-Draft, work in progress | standard authorization-token/challenge surfaces for participating relays | Classical relay authorization remains coarse and cannot satisfy origin MAC/key release | outer plane only; #726 |
| [draft-guo-privacypass-token-binding-02](https://www.ietf.org/archive/id/draft-guo-privacypass-token-binding-02.html) | Individual Internet-Draft, not adopted | binding design input | Not adopted and does not establish this profile's PQ assurance or HyKEM semantics | #1061 review/question; no normative dependency yet |
| [RFC 9955](https://www.rfc-editor.org/info/rfc9955), Hybrid Signature Spectrums | Informational RFC | composition/non-separability terminology | Signatures are not a blind anonymous issuance primitive | terminology only; #1059/#1060 |
| [draft-ietf-privacypass-arc-crypto-01](https://datatracker.ietf.org/doc/html/draft-ietf-privacypass-arc-crypto-01), ARC Cryptography | Privacy Pass WG Internet-Draft, work in progress | anonymous-credential protocol shape, canonical encodings, vectors | Uses classical group assumptions; its security considerations describe quantum discrete-log linkability/partitioning | research/protocol input only; not the PQ leg |
| [Post-Quantum Privacy Pass via Post-Quantum Anonymous Credentials](https://eprint.iacr.org/2023/414) | IACR ePrint research preprint | Privacy Pass-specific PQ anonymous-credential design and prototype | Calls the result plausibly PQ; the evaluated backend lacked zero knowledge, and the reference code warns it is unreviewed and not production ready | do not select; track research and implementation review |
| [Improved Lattice Blind Signatures from Recycled Entropy](https://doi.org/10.1007/978-3-032-01855-7_16) | Peer-reviewed CRYPTO 2025 paper | reviewed lattice blind-signature candidate | No adopted CFRG/Privacy Pass profile, maintained interoperable implementation, fixed profile parameters/vectors, or exact AND-composition review | primitive candidate only; #1060 remains blocked |
| [PoMFRIT](https://www.usenix.org/conference/usenixsecurity26/presentation/baum) | USENIX Security 2026 paper | implemented blind-signature research | Described as plausibly PQ, depends on MAYO/security variants, and has no reviewed Privacy Pass profile or this composition analysis | research only; do not select |
| [Late binding with PQ privacy](https://datatracker.ietf.org/meeting/interim-2026-privacypass-01/materials/slides-interim-2026-privacypass-01-sessa-late-binding-with-pq-unlinkability-00) | Privacy Pass interim presentation, 2026-05-13 | separates PQ unlinkability from PQ unforgeability | Records classical credential forgery after quantum key recovery and no concrete bounds/end-to-end analysis for the proposed direction | confirms open research dependency |

## Explicit non-allocations

This scaffold is a local, pre-I-D implementation profile. `domain-separation.json` is **not** an IANA registry, a Privacy Pass token-type allocation, a MOQT parameter, an extension, a URI scheme, or an assigned codepoint. The included RFCXML draft makes no codepoint request. Any future token type, extension, or registry action requires construction selection, working-group engagement, and separate review.

## Construction-selection questions

1. What PQ anonymous/blind issuance construction has a reviewable proof and implementation path under the collusion model in `threat-model.md`?
2. Is AND-composition with the classical leg defined without introducing a linkable wrapper or an ambiguous assurance upgrade?
3. Is a token-binding extension needed after evaluating the individual binding draft, and how are PoP/HyKEM/resource commitments canonically encoded?
4. Can existing MOQT Objects plus standard authorization-token surfaces carry all relay needs, leaving private data inside encrypted payloads?
5. What exact replay/spend, audit, ledger, and state-rollback behavior is needed before a restricted key release is safe?

The detailed 2026-07-17 evidence and selection bar are recorded in
`construction-selection-audit.md`. No candidate cleared the complete bar. The
answers therefore still block #1060/#1061 production crypto and restricted
anonymous portions of #1062/#1063/#726/#554/#555. They do not block
independently scoped identified stream/group work.
