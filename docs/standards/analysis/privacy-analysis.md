# Privacy analysis: outer MOQT admission and inner anonymous authorization

**Status:** pre-construction analysis for #1059. It records limits rather than claiming that a PQ anonymous primitive or a Privacy Pass extension has been selected.

## Data-flow inventory

| Stage | Parties that should see it | Data that must not cross the boundary |
|---|---|---|
| Optional entitlement | holder, attester | Redemption token, one-use binding private key, origin action details beyond policy need |
| Anonymous issuance | holder, issuer | Holder DID/ATProto handle at redemption; future binding public key under the honest-case unlinkability target |
| Outer MOQT admission | client, relay, possibly origin edge | Inner capability, MAC clearance, plaintext, response/epoch key |
| Inner encrypted control Object | client, Hyprstream origin | Stable holder identifier, raw attestation/entitlement, relay-derived authority |
| Capability/MAC decision | origin/controller and durable audit minimum | Raw token, holder DID, stable client key, unnecessary entitlement data |
| Response or epoch release | origin and authorized client recipient | Epoch secret or plaintext to relay/cache/PDS/ledger observers |
| MOQT Object forwarding/cache | relay/cache/client | Plaintext, capability, clearance, private control fields, epoch secrets |

The explicit separation is intentional: **outer classical admission** is carrier-local coarse admission/rate control and can use standard Privacy Pass/MOQT authorization-token surfaces. **Inner PQ anonymous authorization/key release** is end-to-end inside encrypted ordinary application Objects and is verified only at the origin. The outer layer cannot authorize a method, create an anonymous principal, establish MAC clearance, or release a key.

## Privacy goals and non-goals

Goals are: no stable holder DID is required to redeem; the issuer does not learn the holder's one-use binding key during honest issuance; the relay can forward/cache without learning plaintext or epoch keys; and raw capability/entitlement material is excluded from general audit, PDS, storage, and ledger data.

This profile does **not** claim anonymity against issuer-origin collusion, origin-relay timing correlation, a global observer, device compromise, a malicious client, unique resource/amount classes, or cross-cell log correlation. It does not claim traffic-analysis resistance, padding, cover traffic, ORAM, private payments, or anonymous group membership. A later construction may make narrower, evidence-backed claims only after independent cryptographic and privacy review.

## Metadata and observability controls

| Signal | Exposed to | Current posture | Required design/deployment question |
|---|---|---|---|
| IP/reach/session timing | relay and network observer | unavoidable carrier metadata | Can relay choice, batching, or delayed redemption reduce correlation? |
| Track/namespace/Object identity | relay/cache | ordinary MOQT routing metadata | Use opaque epoch-specific locators; do not call them traffic-analysis protection |
| Object size and cadence | relay/cache/observer | visible even for ciphertext | Are padding classes feasible and what availability cost is acceptable? |
| Issuance time/count | issuer, possibly attester | visible | Can issuance be batched or separated from redemption route? |
| Redemption/spend time | origin and audit | visible for one-use enforcement | What minimum durable spend/audit evidence is needed? |
| Amount/unit/ceiling | origin and possible ledger | potentially distinguishing | Use fixed/coarse denominations and batch/delay publication if a ledger profile is introduced |
| Retry/nullifier result | origin/ledger | can create linkability | Define idempotency and spend result privacy before any public ledger profile |
| Accepted-state/policy generation | origin; possibly bound ciphertext | necessary authorization input | Keep it inside authenticated/encrypted inner data where relay does not need it |
| Browser storage/logs | device/app telemetry | high compromise/leak risk | Erase one-use material on spend/cancel/failure; prohibit raw token/key logging |

## Entitlement, PDS, storage, and ledger variants

ATProto/did:plc/did:web is optional entitlement ingress, not a redeemed identity. Accepted-current `did:at9p` state anchors issuer/origin/controller/policy/signature/KEM authority. PDS access must not inject a handle, DID, account identifier, or attestation transcript into a redemption, Object payload, trust store, or general audit record.

A future resource-entitlement profile must separately assess issuer-origin, origin-ledger, issuer-ledger, PDS/storage, and cross-cell collusion. It must define a commitment/receipt boundary: a public commitment can leak amount, timing, retry status, or linkable nullifiers even when its receipt is encrypted. Fixed/coarse unit and ceiling classes, batching, delayed publication, and minimal encrypted receipts are mitigations to evaluate, not guarantees supplied by this scaffold.

## Audit and retention requirements

Audit is necessary for policy, spend, and incident evidence but must use an event-local opaque correlation value, reason code, policy/state generation, and outcome. It must not contain raw tokens, holder DIDs/handles, binding secrets, stable client keys, plaintext, or epoch keys. Retention duration, access control, aggregation, and cross-cell export remain explicit policy decisions. A durable audit failure denies a restricted key release; an audit pipeline must not silently downgrade a decision to bearer access.

## Privacy review gates

Before production restricted-anonymous issuance/redemption: (1) select and independently review a blind PQ anonymous construction (#1060); (2) review one-use binding/HyKEM transcript privacy (#1061); (3) test the storage/audit/telemetry redaction boundary (#1062/#1051); (4) perform stock-relay packet/cache observations using the plan in `../README.md`; and (5) publish an updated threat model that names assumptions, deployment topology, and residual linkability. Until then, only the documented classical outer admission profile is available and it does not imply `PqHybrid`.
