# Standards profile: PQ-hybrid anonymous authorization (pre-construction)

This directory is the reviewable #1059 documentation/tooling slice for #1058. It does **not** select a PQ anonymous primitive, define a Privacy Pass token type, allocate an IANA/MOQT/Privacy Pass codepoint, or claim IETF endorsement/adoption. Its RFCXML skeleton is a **pre-I-D work in progress** named `draft-hyprstream-privacypass-pqhybrid-00`.

The generated text/HTML intentionally exercise xml2rfc's standard Internet-Draft boilerplate. The source has not been submitted, and the rendered boilerplate is not evidence of submission, adoption, endorsement, or an allocation.

## Contents

- `analysis/threat-model.md` — distinct security properties, trust boundary, full issuer/origin/relay/attester and ledger/PDS/storage/cross-cell collusion model.
- `analysis/privacy-analysis.md` — data flows, metadata limits, and ledger/PDS/storage observability requirements.
- `analysis/downgrade-matrix.md` — outer/inner assurance separation and negative-control matrix.
- `analysis/standards-gap-matrix.md` — standards status, explicit non-allocations, construction questions.
- `registry/domain-separation.json` — local labels, transcript fields, canonicalization, bounds, and owning issues; it is not an external registry.
- `registry/profile-vocabulary.json` — bounded roles, principal kinds, carrier profiles, inner control message kinds, state names, and safety ceilings shared by later drafts.
- `registry/obligations.json` — machine-readable mapping for every RFCXML `MUST`/`MUST NOT` to a test or explicit specification-only owner/blocker.
- `rfc/draft-hyprstream-privacypass-pqhybrid-00.xml` and generated `.txt`/`.html` — reproducible RFCXML v3 scaffold.

## Primary sources and status (2026-07-17)

| Source | Exact primary link | Status |
|---|---|---|
| RFC 9576 — Privacy Pass Architecture | <https://www.rfc-editor.org/info/rfc9576> | Published Informational RFC |
| RFC 9577 — Privacy Pass HTTP Authentication Scheme | <https://www.rfc-editor.org/info/rfc9577> | Published Proposed Standard RFC |
| RFC 9578 — Privacy Pass Issuance Protocols | <https://www.rfc-editor.org/info/rfc9578> | Published Proposed Standard RFC |
| RFC 9794 — PQ/T hybrid terminology | <https://www.rfc-editor.org/info/rfc9794> | Published Informational RFC |
| `draft-ietf-moq-transport-19` | <https://www.ietf.org/archive/id/draft-ietf-moq-transport-19.html> | IETF Internet-Draft, work in progress |
| `draft-ietf-moq-privacy-pass-auth-03` | <https://www.ietf.org/archive/id/draft-ietf-moq-privacy-pass-auth-03.html> | IETF Internet-Draft, work in progress |
| `draft-guo-privacypass-token-binding-02` | <https://www.ietf.org/archive/id/draft-guo-privacypass-token-binding-02.html> | Individual Internet-Draft, not adopted |
| RFC 9955, Hybrid Signature Spectrums | <https://www.rfc-editor.org/info/rfc9955> | Informational RFC |

In particular, MOQT -19 Section 11.2 describes the Object payload as opaque to relays and Section 10.2.2 defines the `AUTHORIZATION TOKEN` message parameter. The MoQ Privacy Pass draft Section 3.5 profiles Privacy Pass on those authorization surfaces. Both MOQT documents remain works in progress; this profile does not treat their current text as an allocated Hyprstream extension.

## Profile boundary and state machines

The bounded vocabulary is: **outer admission** (classical, carrier-local coarse routing/rate control), **inner authorization** (origin-verified encrypted application Object), **anonymous capability** (future typed authorization result), **key release** (sealed unary/stream/group key package), and **stock relay** (unmodified forward/cache only).

Client states: `Idle → EntitlementChecked? → IssuancePrepared → TokenHeld → OuterSessionReady? → InnerControlSent → Released → SpentOrExpired`; errors move to `Denied` or `SpentOrExpired`. The outer-session step is optional because a stock forwarding relay need not participate in authorization. Issuer states: `ConfigurationResolved → IssuanceReceived → IssuanceAccepted|IssuanceRejected`, with `KeyDraining` preventing new issuance. Origin states: `ChallengeIssued → OuterSessionObserved? → InnerControlReceived → StateResolved → BindingVerified → MACAuthorized → SpendCommitted → ReleaseCommitted`; any failed check moves to `Denied`. Spend is durably committed only after all authorization checks and before dispatch or key release. A crash after `SpendCommitted` may burn the one-use capability but cannot repeat authorization. There is deliberately no outer-only transition to application dispatch, plaintext, or key release.

The outer plane uses ordinary classical QUIC/WebTransport/MOQT and, when a relay participates, standard authorization-token/challenge surfaces described by the MoQ Privacy Pass work. It may admit/rate-limit/rout ciphertext. The inner plane stays in encrypted ordinary application Objects addressed to the Hyprstream origin; only it can eventually establish `AnonymousCapability`, MAC clearance, or response/epoch release. A relay route, NodeId, URL, namespace, topic, ciphertext, or outer token is never inner authority.

The bounded inner control message kinds are `challenge`, `issuance-request`, `issuance-response`, `redemption`, `authorization-result`, `key-release-request`, `key-release-response`, and `error`. The local safety ceiling is 65,536 bytes per encrypted Object payload, 16 Objects per exchange, and 8 concurrent exchanges per session. Unknown kinds or versions reject. These are repository-local application bounds, not a wire schema, mandatory MOQT track namespace, or external allocation; construction selection may lower them and any increase requires profile review. `registry/profile-vocabulary.json` is the machine-readable source of truth.

## Stock classical MOQT relay interoperability plan

**Goal:** demonstrate that a stock relay forwards and caches byte-identical opaque MOQT `Object` payloads, and that outer admission and inner authorization have separate assurance boundaries. This is a plan, not a completion claim; #726 owns the nested control implementation.

1. **Topology.** Run an unmodified classical MOQT relay (WebTransport/QUIC), a controlled Hyprstream origin, one authorized client, and one unauthorized client. Capture relay ingress, relay egress, and cache-read bytes with local repository-owned test fixtures. The relay has no Hyprstream private key, capability verifier, MAC authority, response key, traffic key, or epoch key.
2. **Outer admission.** Configure the relay with a standard authorization-token/challenge surface if it supports one. Run an accepted and rejected coarse-admission case. Record that this result is only relay routing/rate control; it does not invoke origin capability verification or release a key.
3. **Inner control.** The client sends encrypted challenge/issuance/redemption/key-release control material only as ordinary application Object payload bytes. Bind the future inner transcript to origin, outer session digest, carrier profile, operation/resource, accepted-current state/policy, one-use PoP, and HyKEM recipient. Do not define a mandatory MOQT extension or codepoint.
4. **Byte identity.** Hash exact serialized Object payload bytes at client publish, relay ingress, relay cache storage/read, relay egress, and authorized client receive. Assert equal length and SHA-256 digest at every hop. Repeat after a cache hit. The relay is permitted to see ordinary routing metadata and ciphertext length/timing, never plaintext or private control fields.
5. **Opaque cache test.** Store an Object in the relay cache, restart/reconnect the authorized client, read the cached Object, and assert identical payload bytes. Record the stock relay binary/configuration hash and its complete test key inventory. With every relay-resident secret as input, assert that authenticated decryption and epoch-key derivation fail and that no known-plaintext sentinel appears in relay logs. This establishes the test boundary; it is not a general proof against a compromised relay host.
6. **Outer/inner separation controls.** (a) valid outer token plus missing/expired/replayed/cross-bound inner capability: ciphertext-only/no release; (b) absent or invalid outer token where relay policy denies: no relay route, while no inference is drawn about inner authorization; (c) valid inner fixture routed through an outer relay that cannot validate it: origin alone evaluates inner authorization. In no case can outer success mint `PqHybrid`, MAC clearance, or a key.
7. **Negative substitutions.** Before origin release, mutate one bound input each: origin, carrier profile, session digest, operation, resource, accepted state, PoP, HyKEM recipient, track/group/epoch. Assert denial before handler/key release. Mutate Object bytes in cache and assert authenticated decryption/verification failure at the client/origin.
8. **Key isolation.** For stream/group follow-up, assert the relay never receives a traffic/epoch key and that a revoked/stale/replayed client receives no next package. #554/#555 own epoch lifecycle; this plan does not assert it exists today.
9. **Evidence.** Save sanitized hashes, byte lengths, relay version/config hash, pcaps or fixture captures, cache trace, key-set assertion, origin decision trace with opaque IDs only, and negative-control outcomes. Do not retain raw token, holder DID/handle, plaintext, PoP private key, or epoch secret.

Success is interoperability of ordinary Objects and byte identity, not confidentiality from metadata or a claim that classical transport is PQ. A failed owned-hybrid transport negotiation cannot silently select `standard-public-relay`; selection is an accepted-current bound profile decision (#1051/#557).

## Generation and validation

Generated text and HTML are deterministic outputs of exactly xml2rfc 3.34.0:

```bash
python3 tools/generate_standards.py
python3 tools/check_standards.py
python3 tools/check_standards.py --check-generated \
  --xml2rfc "$(uvx --from xml2rfc==3.34.0 which xml2rfc)"
```

For a direct local command when `xml2rfc` is already installed at the exact version:

```bash
xml2rfc --version                     # must report 3.34.0
python3 tools/check_standards.py --check-generated --xml2rfc xml2rfc
```

The `Standards artifacts` GitHub Actions workflow installs `xml2rfc==3.34.0`, validates all three JSON files using only Python stdlib, parses RFCXML, ensures every normative RFCXML sentence is manifested, and fails if regenerated text or HTML differs byte-for-byte.

## Construction-selection blockers

1. #1060 must select a reviewable PQ anonymous/blind issuance construction with honest-case unlinkability and explicit collusion limits; an ordinary signature or classical-only wrapper is insufficient.
2. #1061 must specify one-use binding to fresh PoP and HyKEM recipient with canonical encoding and no issuer-visible stable holder key; the individual token-binding draft is input, not adopted authority.
3. #1062 must derive an anonymous typed principal/MAC clearance from verified material and prevent `Public`/unlabeled objects from becoming bearer access.
4. #726 must implement the ordinary Object control profile and standard outer authorization surface without turning a relay into a key-release authority.
5. #1051 must provision accepted-current browser recipient material; #554/#555 must finish identified epoch lifecycle before any restricted anonymous epoch release is claimed.
6. Independent cryptographic and privacy review, vectors, implementation conformance, and stock-relay evidence are production gates.
