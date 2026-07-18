# Resource ownership standards gap and disposition matrix

**Status:** pre-standardization analysis for #1067/#1070. Nothing here claims IETF submission, adoption, endorsement, allocation, or consensus. `draft-hyprstream-resource-attestation-00` is an initial individual-draft **candidate outline**, not a submitted Internet-Draft.

**Assessment date:** 2026-07-18. Links are primary sources; statuses are as of that date and must be rechecked before any #1070 disposition. Dependency classes: **normative dependency** (referenced for correctness), **informative prior art**, **mutable work in progress** (pin revision on use), **repository-local profile** (no standardization claim).

## Disposition matrix

| Material | Existing standard/prior art (primary source, exact status) | Dependency class | Gap | Disposition / owner |
|---|---|---|---|---|
| Canonical resource intent and dual attestation | [RFC 8949](https://www.rfc-editor.org/info/rfc8949) CBOR (Internet Standard, STD 94); [RFC 8610](https://www.rfc-editor.org/info/rfc8610) CDDL (Proposed Standard); [CID/multiformats specs](https://github.com/multiformats/cid) (community specifications, no SDO) | normative dependency (CBOR/CDDL); informative prior art (CID) | no standard claim joining independent title/economic attestations with registrar fencing | candidate `draft-hyprstream-resource-attestation-00`; #1070, informed by #1065 |
| Signature envelope (COSE) | [RFC 9052](https://www.rfc-editor.org/info/rfc9052) COSE structures/process (Internet Standard, STD 96); [RFC 9053](https://www.rfc-editor.org/info/rfc9053) COSE algorithms (Proposed Standard) | normative dependency | envelope carries structures, not this profile's signer-authority/epoch semantics | #1065 references the structures; profile semantics are repository-local |
| Hybrid signatures | [RFC 9955](https://www.rfc-editor.org/info/rfc9955) Hybrid Signature Spectrums (Informational RFC); [RFC 9794](https://www.rfc-editor.org/info/rfc9794) PQ/T terminology (Informational RFC); `hyprstream-crypto` composite `id-MLDSA65-Ed25519` v1 (repository-local profile) | informative prior art (RFCs); repository-local profile (suite) | exact project signature profile/domain binding | reference existing crypto profile; do not redefine algorithms; #550/#1058 owners |
| DID/key state | [W3C DID Core 1.0](https://www.w3.org/TR/did-core/) (W3C Recommendation, 2022-07-19); did:web/did:key methods (W3C CCG community drafts, not Recommendations) | normative dependency (DID Core); informative (methods) | accepted-current did:at9p state is project-specific | informative dependency, internal profile unless separately standardized; #1039 |
| MAC title/control | [UCAN specs](https://github.com/ucan-wg/spec) (community specification, no SDO); [RFC 6749](https://www.rfc-editor.org/info/rfc6749) OAuth 2.0 (Proposed Standard); SELinux (implementation, not a protocol standard) | informative prior art | project TE/lattice and content-label semantics | internal architecture profile; no claim UCAN/SELinux standardizes title |
| Cellular reserve/post/void | double-entry accounting practice; TigerBeetle design documents (project documentation, not a standard); [Interledger protocol RFCs](https://interledger.org/developers/rfcs/interledger-protocol/) (community RFCs, not IETF) | informative prior art | resource-intent-bound issuer liabilities and compensation | internal ledger profile (#922/#1072); resource draft references semantics only |
| Privacy Pass anonymous issuance/binding | [RFC 9576](https://www.rfc-editor.org/info/rfc9576) (Informational), [RFC 9577](https://www.rfc-editor.org/info/rfc9577) / [RFC 9578](https://www.rfc-editor.org/info/rfc9578) (Proposed Standards); [draft-ietf-privacypass-arc-crypto-01](https://datatracker.ietf.org/doc/html/draft-ietf-privacypass-arc-crypto-01) / [draft-ietf-privacypass-arc-protocol-01](https://datatracker.ietf.org/doc/draft-ietf-privacypass-arc-protocol/) (Privacy Pass WG drafts, work in progress, classical assumptions); #1058 pre-I-D work | mutable work in progress (ARC); informative prior art (RFCs) | PQ anonymous construction and one-use HyKEM binding unresolved | owned exclusively by #1059–#1061; resource draft carries application context only. ARC was assessed as classical prior art in the #1059 [source/status audit](construction-selection-audit.md); this matrix deliberately states only the resource application-context delta and copies no wire/construction content |
| Anonymous MAC principal | no standard identity; #1058 typed principal | repository-local profile | authorization principal mapping into project MAC | #1062 internal/application profile; resource draft treats as opaque verifier output |
| atproto/PDS publication | [atproto specifications](https://atproto.com/specs/atp) (project/community specifications, versioned by lexicon) | informative prior art | resource manifest/receipt lexicons are project records | repository lexicons/internal profile; publication is evidence, not finality |
| CAS/CID binding | [CID spec](https://github.com/multiformats/cid) / [multihash spec](https://github.com/multiformats/multihash) (community specifications) | informative prior art | exact manifest/signature-cycle avoidance and label field | candidate resource draft encoding section after #1065 vectors |
| Registrar saga/fencing | distributed-systems patterns (linearizable CAS, fencing tokens; no interoperable standard) | informative prior art | interoperable wire behavior not required for local registrar | internal profile/ADR; draft states externally visible finalization semantics only |
| 9P ownership semantics | 9P2000 (Plan 9 manual/protocol documentation, no SDO standard); 9P2000.L (Linux community extension documentation) | informative prior art | stable title identity, hard-link/rename/bind semantics absent | internal #1071 profile first; separate individual-draft candidate only after interop proves broader value |
| MoQT carriage | [draft-ietf-moq-transport-19](https://datatracker.ietf.org/doc/draft-ietf-moq-transport/) (IETF Internet-Draft, work in progress); [draft-ietf-moq-privacy-pass-auth-03](https://datatracker.ietf.org/doc/draft-ietf-moq-privacy-pass-auth/) (IETF Internet-Draft, work in progress) | mutable work in progress | resource intent application-context carriage | feed requirements to #1058 MoQ draft owner; no duplicate resource transport draft |
| OpenTimestamps anchoring | [OpenTimestamps](https://opentimestamps.org/) format/protocol (project specification) | informative prior art | no new format needed | documented optional backend; no standards artifact |

## Non-duplication boundary

The resource draft MUST NOT specify a PQ anonymous token, blind issuance, token-binding transcript, HyKEM construction, MoQT codepoint, or unary redemption protocol. It consumes opaque verified outputs and contributes a canonical resource-intent application-context digest to #1058 drafts. Any needed change is proposed to those owners rather than copied.

It MUST NOT imply that UCAN, atproto, 9P, CID, Privacy Pass, MoQT, COSE, or OpenTimestamps endorses Hyprstream's profile. Experimental/private-use values are identified as such; no IANA allocation is assumed.

## Versioning and envelope contract scaffold

Wire encoding, CDDL, and wire-format versioning remain #1065-owned. The `hyprstream-resource` semantic scaffold nevertheless freezes explicit version/suite coordinates so no field is silently ambiguous before the codec exists: `ContractVersion` (semantic contract revision), `IntentFormatVersion` (canonical intent format), `DigestSuite` (domain-separated intent digest algorithm), and `SignatureSuite` (composite signature profile reference). `Assurance` has a checked `u8` conversion mirroring the canonical `hyprstream-rpc` owner; #1065 MUST define the checked conversion plus byte-level compatibility vectors rather than maintain two serializations.

The attestation envelope contract (architecture §5/§8) requires verifiers to authenticate, and #1065's envelope MUST carry or unambiguously reference, all of:

| Envelope field | Binds / defeats | Architecture source |
|---|---|---|
| contract version | version confusion/rollback | §5, finalization check 2 |
| signature suite/profile reference | suite stripping, cross-algorithm substitution | §5; RFC 9955 terminology only, algorithms referenced not redefined |
| signer identity (`signer`) + key identifier (`key_id`) | signer substitution; selects verification key | §5 check 2, §8 |
| signer-key epoch (`key_epoch`) | historical verification across rotation/compromise | §8 `(signer, key_id, key_epoch)` mapping |
| key purpose (MAC / ledger / registrar-finalization) | cross-purpose signature reuse (attestation-as-finalization) | §5 |
| signed-byte domain separator | cross-protocol replay | §4 domain separation |
| accepted-current state evidence coordinates (`policy_epoch`, `state_epoch`) | stale authority/policy | §5 checks 2–3 |

Attestation bodies are digest-only with respect to the intent: MAC and ledger attestations embed `intent_digest` plus attester-owned fields (epochs, controller context, capability binding, content-label commitment, assurance, expiry for MAC; ledger/cell, payer profile, issuer, unit, amount, transfer ID, phase, assurance for ledger) — never a caller-supplied full intent reference. #1065 MUST produce substitution vectors for signer, key purpose, suite, epoch, controller, label, issuer/operator, unit, phase, and amount. No arbitrary byte string is labelled "hybrid-COSE" unless the exact maintained project profile is normatively identified.

## Candidate draft outline

`draft-hyprstream-resource-attestation-00`:

1. Status, scope, and non-endorsement.
2. Conventions and terminology (see below).
3. Architecture and one-owner authority model.
4. Typed roles and privacy profiles.
5. Protocol data model: canonical ResourceIntent requirements, domain separation, attestation envelopes, finalization statement, resolution record.
6. Verification procedure: dual-attestation join, finalization checks, assurance clamp.
7. Version, extension, and suite negotiation and migration: unknown-version/critical-extension rejection, suite agility, historical verification.
8. Registrar finalization, predecessor CAS, fencing, idempotency, terminal resolutions.
9. Namespace projection ordering and stale-effect rejection.
10. Manifest reference model and CID/signature-cycle avoidance.
11. Bounds and processing-error handling.
12. Failure, quarantine, compensation, and recovery considerations.
13. Receipt and privacy projection (public/private split, per-observer leakage).
14. Privacy considerations (collusion/metadata limits).
15. Security considerations (substitution/replay/downgrade/authority compromise).
16. 9P/CAS application considerations (informative until reports exist).
17. Conformance and requirements-traceability mapping to the normative inventory below.
18. IANA considerations: none in -00; no allocations requested.
19. Implementation status and disposition.

## Terminology decisions for #1070

- **MAC** is expanded on first use as *Mandatory Access Control*; it never means message authentication code in this document family.
- **Title**/**owner** denotes a protocol-local control/ownership claim over a resource. The protocol carries no legal identity and does not adjudicate legal property rights.
- **Attestation** here means a signed MAC-authorization or ledger-accounting statement over a resource-intent digest. It is distinct from the Privacy Pass `Attester` role (RFC 9576) and from RATS attestation; the draft defines the term and does not import those roles.
- **Registrar**, **fence** (registrar term + per-resource generation), **cell ledger**, **finality**, **PDS**, **CID**, and **privacy profile** are defined in the terminology section before first use.

## Normative requirements inventory

Each requirement has a stable ID, exact source, owning issue/subsystem, and disposition (`test` = deterministic positive **and** negative evidence; `spec-only` = explicitly specification-only with a named owner). #1070 CI MUST fail on any normative sentence in the draft unmapped to an ID, and on any inventory entry without an owner/disposition, following the #1059 obligations model.

| ID | Requirement (BCP 14) | Source | Owner | Disposition |
|---|---|---|---|---|
| RR-01 | A final manifest MUST reference valid MAC and ledger attestations over identical canonical intent bytes. | arch §1 inv 1, §5 | #1065 | test |
| RR-02 | `ResourceIntent` MUST be bounded, versioned, and carry the minimum field set of arch §4. | arch §4 | #1065 | test (bounds vectors) |
| RR-03 | Verifiers MUST reject non-canonical encoding, unknown critical fields, unsupported versions, and unsupported resource kinds. | arch §4–§5 | #1065 | test |
| RR-04 | The intent digest MUST be computed over full canonical bytes with a versioned domain separator. | arch §4 | #1065 | test |
| RR-05 | `operation_id` MUST be deterministic per logical transition; reuse with non-identical canonical bytes MUST reject as an invariant violation. | arch §1 inv 5, §4 | #1065/#1069 | test |
| RR-06 | Ledger transfer IDs and storage staging IDs MUST be deterministic domain-separated derivatives of `operation_id`. | arch §4 | #1065 | test |
| RR-07 | Attestation envelopes MUST carry contract version, suite, signer, key ID, signer-key epoch, key purpose, and signed-byte domain. | arch §5, §8 | #1065 | test (substitution vectors) |
| RR-08 | Attestations MUST embed the intent digest plus attester-owned fields only; a caller-supplied full intent reference MUST NOT be embedded. | arch §5 | #1065 | test |
| RR-09 | The registrar MUST verify attester signatures against accepted-current signer authority at the pinned epochs. | arch §5 check 2 | #1068/#1072 | test |
| RR-10 | The registrar MUST revalidate MAC authorization immediately before the finalization CAS. | arch §5 check 3 | #1068 | test |
| RR-11 | Finalization MUST require ledger phase `Posted` (or an explicitly represented zero-cost phase) and unit/amount within intent bounds; reserve is a maximum. | arch §5 check 4 | #1072 | test |
| RR-12 | Content/manifest binding and label resolution MUST be verified at finalization. | arch §5 check 5 | #1068 | test |
| RR-13 | Effective assurance MUST be the minimum of all verified legs; a missing leg MUST clamp to `Unverified`; no leg may be skipped. | arch §1 inv 9, §5 check 6 | #1065 | test |
| RR-14 | Finalization MUST use expected predecessor/version CAS plus the per-resource fence; a stale fence/predecessor MUST lose. | arch §1 inv 3, §5 check 7 | #1069 | test |
| RR-15 | A terminal outcome already recorded for a different claim under the same operation ID MUST reject. | arch §5 check 8 | #1069 | test |
| RR-16 | Terminal outcomes MUST be exactly one typed immutable `Resolution` record (`Finalized`/`Voided`/`Compensated`/`RejectedConflict`) with total visibility/billing/GC consequences; quarantine/manual-review MUST NOT be treated as terminal. | arch §6 | #1069 | test |
| RR-17 | Every state and outbound intent MUST be persisted before its external effect. | arch §6 | #1069 | test |
| RR-18 | An identical retry MUST return the recorded outcome; recovery MUST NOT re-derive it. | arch §1 inv 5, §6 | #1069 | test |
| RR-19 | Projection effects MUST carry operation ID, fencing token (term + generation), and resource version, and MUST be applied by compare-and-apply that rejects stale generations; withdrawal MUST identify the transition/generation withdrawn. | arch §6 | #1069/#1071 | test |
| RR-20 | Provisional, quarantined, and manual-review material MUST NOT appear in the ordinary namespace. | arch §1 inv 6, §6 | #1071 | test |
| RR-21 | A finalized resource value MUST be constructible only from a registrar-signed finalization statement verified against accepted-current registrar authority. | arch §6 | #1069 | test (forgery negative) |
| RR-22 | The public namespace projection MUST contain commitments and minimum routing/verifiability data only: no attestation CIDs, payer, issuer, unit, exact amount, transfer ID, capability binding, or signer-key coordinates. | arch §6, §9 | #1066/#1071 | test (projection negative) |
| RR-23 | Anonymous profiles MUST bind capabilities as blinded, operation-scoped commitments; no holder-stable or cross-operation-linkable capability value may appear in any attestation or public projection. | arch §5, §9 | #1062 | test |
| RR-24 | Anonymous profiles MUST NOT fall back to identified or unauthenticated principals on any failure. | arch §7, §9 | #1062 | spec-only + test |
| RR-25 | Origin audit/trust stores MUST NOT contain raw anonymous tokens, holder/root DIDs, handles, stable client keys, holder-stable capability roots, or directly linkable allocation IDs. | arch §9, privacy-analysis | #1062 | test (schema/log scans) |
| RR-26 | No schema, receipt, grant, or record MUST carry legal identity; KYC remains optional and external at a fiat boundary. | arch §9, privacy-analysis | #1070 | spec-only |
| RR-27 | Payment MUST NOT be interpreted as title; MAC authorization MUST NOT be interpreted as entitlement. | arch §1 inv 2 | #1070 | test |
| RR-28 | Missing/unresolvable label, principal, accepted-current state, entitlement, attestation, or fence MUST fail closed; `Unavailable` MUST NOT trigger a weaker fallback. | arch §7 | all | test |
| RR-29 | Ordinary reads and fine-grained leased writes MUST NOT synchronously depend on ledger or proof-plane I/O. | arch §1 inv 7 | #1071 | spec-only + test |
| RR-30 | Posted-but-unfinalized accounting MUST be corrected by an immutable compensating entry; history MUST NOT be mutated or deleted. | arch §1 inv 10, §8 | #1072 | test |
| RR-31 | Historical verification MUST retain the accepted-current key/state evidence to map `(signer, key_id, key_epoch)` across rotation and revocation; rotation MUST NOT invalidate correctly verified historical signatures. | arch §8 | #1039 | spec-only + test |
| RR-32 | The resource draft MUST NOT specify a PQ anonymous token, blind issuance, token-binding transcript, HyKEM construction, MoQT codepoint, or unary redemption protocol, and MUST NOT imply ecosystem endorsement or assume any IANA allocation. | non-duplication boundary | #1070 | spec-only |

#1070 must map each requirement to a deterministic test or explicit specification-only obligation, publish CDDL/vectors from #1065 (native and WASM vectors covering canonical bytes, version changes, unknown critical fields/kinds, bounds, suite/profile confusion, and algorithm migration), generate reproducible RFCXML/text/HTML, and record disposition as submitted, presented for feedback, superseded, or retained as an implementation profile.

## Review/engagement targets

Architecture/storage and 9P implementers; independent security/privacy reviewers; CBOR/COSE/CID implementers; atproto community for lexicon/proof publication; IETF Privacy Pass and MoQ participants only for application-context overlap; PQUIP/CFRG only for terminology/construction coordination owned by #1058. Feedback is logged without claiming adoption.
