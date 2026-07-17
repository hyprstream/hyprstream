# PQ-hybrid anonymous authorization threat model

**Status:** design input for #1058/#1059; not an Internet-Draft, protocol specification, construction selection, or assurance claim. **Owners:** #1059 (profile), #1060 (anonymous issuance), #1061 (binding), #1062 (MAC principal), #726 (MoQT carriage), #554/#555 (key release), and #1051 (browser lifecycle).

## Security properties are distinct

| Property | Claim boundary | Not implied by it |
|---|---|---|
| Transport confidentiality | QUIC/WebTransport or another carrier protects a hop | Object/content confidentiality from a relay; authorization; PQ assurance |
| Object/content confidentiality | A sending endpoint encrypts an application Object for the exact end receiver before MOQT publication | Relay authentication; sender identity; authorization |
| Authorization unforgeability | An origin accepts only a valid, unspent capability for the exact action | Issuance/redemption unlinkability or identity authentication |
| Issuance/redemption unlinkability | Under the stated non-collusion assumptions, an issuer cannot link an honest issuance to a later redemption | Traffic-analysis resistance or unlinkability under issuer-origin collusion |
| Proof of possession (PoP) | A redeemed one-use capability requires its fresh binding private key | Stable identity, a DID, or an issuer-visible binding key |
| Identity authentication | A resolved identified subject proves the applicable identity protocol | Anonymous-capability authorization |
| MAC clearance | The origin compares verified capability clearance with a content-bound label and policy | Access merely because a label is `Public` |
| Metadata privacy | Identifiers, timing, sizes, routes, and audit records are minimized/partitioned | Content confidentiality or full traffic-analysis resistance |
| Availability / abuse control | Rate limits, quotas, and relay admission constrain load | Authorization, clearance, or key release |

`PqHybrid` is an authorization-assurance result, not a transport property and not a synonym for a hybrid signature wrapper. Its eventual definition is blocked on a reviewed blind PQ anonymous issuance construction (#1060), binding (#1061), and independent review. A classical entitlement, classical Privacy Pass token, URL, topic, NodeId, route, ciphertext, or outer admission token cannot raise assurance or authorize inner access.

## Assets, boundaries, and trust anchors

The assets are capability secrecy before presentation; one-use spend state; entitlement privacy; PoP/HyKEM private material; content plaintext; stream/group epoch keys; MAC labels and policy; accepted-current `did:at9p` state; and minimal audit evidence. The security boundary is the Hyprstream origin/controlled key-release service. It resolves issuer, origin, controller, policies, and KEM recipients from accepted-current state; an asserted DID, relay endpoint, tenant string, track name, or self-advertised key is never authority.

The outer plane is ordinary classical QUIC/WebTransport/MOQT plus standard authorization-token surfaces. It may perform coarse relay admission, routing, caching, and rate control. The inner plane is an encrypted ordinary application Object between client and origin. It carries future PQ anonymous issuance/redemption/binding material and is the only plane that can produce an `AnonymousCapability`, MAC decision, or release a response/epoch key. No new mandatory MOQT extension or codepoint is selected here.

## Actors and adversaries

- **Client/holder:** obtains an entitlement if required, obtains a token, creates one-use PoP and HyKEM material, and redeems once. A compromised holder can spend its own capability but must not gain another holder's capability or key.
- **Attester:** checks an optional entitlement (for example ATProto) and supplies an issuance input. It is not the redeemed subject. Its classical evidence remains Classical unless a future reviewed profile proves every required leg.
- **Issuer:** issues an anonymous token under a purpose-separated issuer key from accepted-current state. It must not reuse mesh signing, mesh KEM, PDS, response, or controller keys.
- **Origin/controller:** validates the inner capability, performs spend/MAC/policy checks, and releases a response or epoch package. Anonymous protocol inputs must not carry a stable holder identity; the origin can still correlate metadata under the limits below.
- **Relay:** a stock MOQT relay forwards and may cache ordinary Object bytes. It is outside plaintext, decoded capability, clearance, and epoch-key trust boundaries.
- **Observers:** network, ledger, PDS, storage/cache, audit, browser/device, and availability observers can correlate metadata even when payloads are encrypted.

Attackers may replay, strip a leg, substitute transcript fields, cross a resource/method/track/profile/epoch, delay or reorder objects, serve stale accepted state, exhaust resources, or observe multiple systems. Cryptographic compromise assumptions are construction-specific and deliberately unclaimed pending #1060/#1061.

## Collusion model and honest-case limits

The table states what must be mitigated and what cannot honestly be promised. “Link” means correlate an issuance, redemption, resource action, or holder using data available to the colluding set.

| Colluding set / observer variant | Data jointly visible | Required boundary | Residual limitation / required mitigation |
|---|---|---|---|
| Issuer alone | Issuance transcript, issuer configuration | Blind issuance and no issuer-visible future binding key | Issuer can count issuance and correlate its own account/entitlement metadata |
| Origin alone | Inner redemption, resource, spend result, key-release request | No holder DID/ATProto handle/raw entitlement in redemption; one-use fresh PoP | Origin can correlate retries, timing, sizes, and same-session actions |
| Relay alone | MOQT session/track metadata, Object bytes, timing, size, cache reads | Payload encryption; no capability/clearance/key in outer plane | Relay can traffic-analyze routes, timing, sizes, and cache access |
| Attester alone | Entitlement and attestation request | Attester output must not become a redeemed subject identifier | Attester knows the entitlement relationship and timing |
| Issuer + origin | Issuance and redemption logs | **No issuance/redemption unlinkability claim** unless a reviewed construction explicitly proves this case | Treat as linkable; minimize logs, partition operations, batch/delay where practical |
| Issuer + attester | Issuance plus entitlement records | Do not expose redemption material | They identify entitlement-to-issuance, not necessarily redemption absent other observations |
| Origin + relay | Redemption/key-release timing plus transport/cache metadata | Encrypted inner Object and metadata minimization | Usually linkable at session/timing/size granularity; no traffic-analysis-resistance claim |
| Issuer + relay | Issuance timing plus carrier metadata | Separate issuance/redemption routes where deployable | Timing/route correlation may link; batching and cover/aggregation are open deployment questions |
| Attester + origin | Entitlement and redemption records | No stable attester identifier in inner capability | Treat as linkable if timing, amount, or unique entitlement class correlates |
| Issuer + origin + relay and/or attester | Nearly all protocol and network observations | No anonymity claim against this set | Only content cryptography and explicit authorization checks remain meaningful |
| Origin + ledger observer | Redemption result, charge/commitment, amount/time | Fixed/coarse denominations, batch/delayed publication, encrypted receipts | Amount, retry, nullifier/spend timing can correlate; no economic unlinkability claim yet |
| Issuer + ledger observer | Issuance and public/ledger records | Avoid token/raw-holder data in commitments | Timing/amount and unique class correlate; resource-entitlement extension is blocked |
| PDS observer alone | ATProto/PDS entitlement fetch/publication data | Keep it out of redemption and inner audit | Can identify entitlement activity and timing |
| Origin + PDS observer | Redemption plus entitlement access timing | Partition flows and avoid holder fields | Treat as linkable where timing/account data aligns |
| Storage/cache observer alone | Ciphertext Objects, Object identity, reads, retention | Immutable encrypted Objects and opaque epoch locators | Sees object existence, size, read/replay/caching patterns |
| Origin/relay + storage observer | Key-release/session and cache access patterns | Cache policy, bounded retention, padding/batching research | Strong timing/access-pattern correlation; no ORAM-like claim |
| Cross-cell observer (issuer/origin/relay/ledger/PDS/storage) | Correlated logs across administrative cells | Administrative/log separation and retention controls | Assume linkage when identifiers, timing, amount, or route overlap |

A malicious issuer, origin, controller, relay, or attester is not made harmless merely because other roles are honest. Any privacy assertion must name the exact non-collusion set it assumes.

## Required fail-closed decisions

Before dispatch, plaintext, or key release, the origin must reject a missing/expired/replayed capability; stale/revoked/forked accepted state; wrong issuer/origin/service/method/resource/carrier profile/track/epoch; missing or substituted PoP/HyKEM recipient; unauthorised MAC label/policy; and unsupported suite. A relay-only outer token may permit ciphertext routing but must never pass this gate. Current production network release remains gated by #726 and, for browser paths, #1051; identified stream/group epochs remain separately tracked by #554/#555.

## Open construction and deployment questions

1. Which reviewed PQ anonymous/blind primitive can satisfy the exact collusion model without selecting an unallocated Privacy Pass token type (#1060)?
2. Can a generic binding profile cover the needed HyKEM and resource intent fields, or does a new semantics-bearing extension require standards review (#1061/#1064)?
3. What public spend/nullifier/ledger design, if any, avoids retry/amount/timing leakage (#1062/#1064)?
4. What batching, denomination, padding, retention, and audit policies are acceptable for the deployments that claim metadata reduction?
