# Resource-attestation standards-gap matrix and draft disposition

**Assessment date:** 2026-07-17. "Gap" is not a request for an allocation and does not claim WG adoption. Links are primary sources.

## Ownership boundary with #1058/#1059

#1070 owns `draft-hyprstream-resource-attestation-00`, its resource-specific CDDL/vectors/traceability, and the CAS/9P implementation reports. #1058/#1059 retain ownership of `draft-hyprstream-privacypass-*` and `draft-hyprstream-moq-*` sources, the threat/property vocabulary, and stock-relay evidence. #1070 supplies reviewed resource-intent requirements, vectors, and proposed application-context changes to those documents; it does not fork or independently publish overlapping Privacy Pass/MoQ drafts.

## Primary sources

| Source | Status at assessment | Reused surface | Gap / boundary | Disposition |
|---|---|---|---|---|
| [RFC 6920](https://www.rfc-editor.org/info/rfc6920), Naming Things with Hashes | Proposed Standard | content CID / naming | Does not define resource title, attestation, or registrar semantics | reference only; #1066 |
| [RFC 8949](https://www.rfc-editor.org/info/rfc8949), CBOR | Proposed Standard | deterministic encoding | DAG-CBOR profile (atproto) is the production target; pending #1065 | reference; #1065 |
| [RFC 2119](https://www.rfc-editor.org/info/rfc2119)/[8174](https://www.rfc-editor.org/info/rfc8174) | BCP 14 | requirement language | none | reference |
| [RFC 9576](https://www.rfc-editor.org/info/rfc9576)/[9578](https://www.rfc-editor.org/info/rfc9578), Privacy Pass | Informational/Proposed Standard | anonymous-entitlement/application-context input | resource-intent is NOT a Privacy Pass token type; no codepoint requested | #1070 supplies reviewed requirements to #1058/#1059 only |
| [draft-ietf-moq-transport-19](https://www.ietf.org/archive/id/draft-ietf-moq-transport-19.html) | IETF I-D, WIP | ordinary Object carriage | no MOQT extension required for resource title | reference; #726 |
| [RFC 9794](https://www.rfc-editor.org/info/rfc9794), PQ/T terminology | Informational | hybrid assurance terminology | does not define resource attestation | terminology |

## Explicit non-allocations

This scaffold is a local, pre-I-D implementation profile. `resource-intent.json` and `resource-vocabulary.json` are **not** IANA registries, 9P opcodes, MoQ track namespaces, Privacy Pass token types, or codepoints. The RFCXML draft makes no codepoint request. The logical application label `hyprstream-resource-attestation-v1` is carried inside registrar/proof-plane records only.

## Draft disposition (recorded per acceptance criterion)

| Draft | Disposition | Evidence |
|---|---|---|
| `draft-hyprstream-resource-attestation-00` | **retained as an implementation profile** (pre-submission) | RFCXML source + generated text/HTML in this repo; not submitted; no endorsement claimed |
| Privacy Pass extension (new token wire semantics) | **not created** | resource-intent adds no new Privacy Pass token wire semantics; requirements are supplied to #1058/#1059 |
| 9P ownership Internet-Draft | **not created** | the standards-gap review concludes external interoperability does not warrant a separate 9P draft; the profile is repository-local |

## Standards-gap questions

1. Does any existing standard define dual-attested resource title joining a MAC attestation and an economic-ledger attestation over an identical canonical intent digest? — No; this is the profile's scope.
2. Is a 9P ownership extension needed for external interoperability? — No conclusion of external interop is warranted at this stage; the profile remains repository-local.
3. Are resource-intent application-context updates to the Privacy Pass/MoQ drafts new wire semantics? — No; #1070 supplies reviewed requirements only to #1058/#1059.

The canonical production encoding, signature suite, and anonymous-construction selection remain blocked on the sibling issues and independent review.
