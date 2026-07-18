# Dual-attested resource ownership architecture

**Status:** proposed canonical semantic contract for #1064, produced by #1067. Wire encoding, CDDL, vectors, and cryptographic constructions remain owned by #1065 and #1059–#1062. Production enablement is blocked on the independent reviews listed in [Review and freeze gates](#review-and-freeze-gates).

## 1. Purpose and invariants

A resource transition is one canonical claim, independently authorized by MAC and funded by the cellular ledger, then serialized by one durable registrar. “Dual attestation” means two independently verifiable signatures over the **same domain-separated `ResourceIntent` digest**. It never means two title databases, a payment-derived ownership record, or a mutable join in the PDS.

The following are invariant:

1. No final manifest exists without valid MAC and ledger attestations over byte-identical intent bytes and therefore the same digest.
2. MAC authorization does not prove payment; ledger payment does not confer title.
3. The registrar alone advances canonical resource state and selects one successor using predecessor/version CAS plus a per-resource fence.
4. Stable resource identity, operation identity, path, content CID, and manifest CID are separate.
5. Every external effect is idempotent under one deterministic operation ID. An identical retry returns the stored outcome.
6. Provisional material is absent from ordinary namespace resolution. Visibility is registrar state, not byte existence.
7. Reads are MAC-only. Ledger and proof-plane I/O never enters ordinary reads or fine-grained leased writes.
8. Anonymous capability and entitlement profiles carry typed commitments, never fabricated DIDs.
9. Effective assurance is the minimum of every verified authorization, entitlement, binding, accounting, and key-release leg.
10. History is corrected by a new compensating fact, never mutation or deletion of an attested fact.

The dependency-light Rust mirror is `hyprstream-resource`. It freezes names and subsystem boundaries, not #1065's encoding.

## 2. Exactly-one-owner authority matrix

| Authoritative statement | Sole owner | Evidence / interface | Explicit non-owners |
|---|---|---|---|
| Principal may create, control, mutate, transfer, or delete this resource under the pinned label/policy/capability state | Native MAC reference monitor | `MacAttestation` over `ResourceIntent` digest | Ledger, registrar, storage, namespace, PDS |
| Payer has a valid entitlement and amount is reserved, posted, voided, or compensated in this issuer unit | Cell ledger | `LedgerAttestation`; immutable journal and receipt | MAC, registrar, PDS, storage |
| Operation is in lifecycle state S | Resource registrar | durable saga row and immutable resolution record | MAC, ledger, namespace, PDS |
| Resource/version V has canonical successor M | Resource registrar | predecessor/version CAS and fencing token | MAC, ledger, namespace, PDS |
| Bytes reconstruct to content CID C | Resource store/CAS | seal result and content verification | registrar, ledger, PDS |
| Provisional bytes exist under operation O | Resource store/CAS | private staging metadata | byte existence does not imply visibility or title |
| Finalized desired state is projected at a path/export | Namespace/VFS adapter | ordered compare-and-apply from registrar outbox | path does not own title; storage does not decide visibility |
| Source manifest's signed sensitivity/assurance label is L | Manifest signer/content producer | content-bound `security_label` inside signed/CID-addressed manifest | caller, token, ledger |
| Effective local label for imported/bound object is L' | Local MAC perimeter/reference monitor | import-floor/binder-floor join and vocabulary mapping | remote face value is only input evidence |
| Artifact A was published at proof location P | PDS/proof plane | signed commit, receipt/checkpoint/manifest record | PDS is not title, balance, or synchronous lock authority |
| Issuer authorized creation of unit U | Grant issuer | issuer-signed grant and accepted-current issuer key evidence | ledger/operator/payer cannot mint issuer authority |
| Liability for issued unit U is recorded as amount N | Cell ledger | issuer-named `UnitId`, liability account and journal | grant alone is not a balance |
| Identity/key state was accepted-current at epoch E | Accepted-current identity authority (#1039) | resolved, pinned key/state evidence | a DID string, relay, or stale cache is not authority |
| Anonymous capability is valid for exact operation O | #1058/#1062 typed anonymous authorization verifier | verified operation-scoped capability commitment | ledger bridge and origin cannot synthesize identity/reinterpret token crypto |
| Anonymous entitlement redemption/nullifier has economic state S | #1072 ledger redemption bridge | one-use durable redemption state consuming #1059–#1061 output | MAC capability validity does not prove payment |

A subsystem may cache or publish another subsystem's assertion, but must preserve its owner and provenance. Cache loss affects availability, not authority transfer.

## 3. Typed roles

Roles are never collapsed because one deployment happens to operate several of them:

- **Owner**: title principal. `Identified(Did)`, `Pairwise(Did)`, or `Committed(commitment)`.
- **Controller**: currently authorized actor or delegate. Identified/pairwise DID or typed anonymous-capability commitment.
- **Payer**: identified/pairwise account principal or anonymous-entitlement commitment.
- **Issuer**: signs the entitlement/grant and carries the liability for its unit.
- **Ledger operator**: serializes accounting within a cell. It may differ from issuer.
- **Registrar operator**: serializes lifecycle transitions. It cannot manufacture either attestation.
- **Storage custodian**: holds bytes and may deduplicate only inside the authorized domain.
- **Namespace operator**: projects finalized resources and mediates path operations.
- **Proof publisher/PDS operator**: publishes evidence without acquiring title or control.

Delegation records delegator and actor separately. Clearance is their meet; assurance is clamped to actual verified key material. A payer may pay for another owner without becoming owner. A controller may act without being owner. An issuer/ledger operator may coincide operationally but remains two typed fields.

## 4. Canonical semantic claim

#1065 MUST define a bounded, versioned canonical DAG-CBOR `ResourceIntent` containing at least:

- format version and critical-extension policy;
- stable `resource_id` and deterministic `operation_id`;
- exact operation and resource kind;
- expected predecessor manifest CID/version (absent only for create);
- proposed owner, controller, and payer references as separate typed fields;
- origin/registrar context and carrier/privacy profile;
- content commitment or explicit “content determined at seal” marker;
- content-bound security-label commitment and lattice/policy version;
- policy epoch, accepted-current state epoch, capability binding (capability-chain root in identified profiles, blinded operation-scoped commitment in anonymous profiles), and grant/entitlement commitment;
- issuer-named resource unit, maximum charge, reservation/lease bounds;
- nonce, issued-at, not-before, and expiry;
- required assurance and the verified assurance inputs that can clamp it.

The digest is computed over the full canonical bytes with a versioned domain separator. Both attesters MUST receive or independently reconstruct those exact bytes, reject unknown critical fields and non-canonical encodings, and return an attestation carrying the digest plus enough pinned state to revalidate. The final manifest references attestation CIDs; signatures do not sit inside the bytes whose CID they sign, preventing a CID/signature cycle.

`resource_id` is an opaque, randomly generated 128-bit identity allocated for `Create`. It is not derived from owner, path, content, or manifest. Random generation prevents namespace/owner leakage and permits rename, links, mutation, and title transfer without identity change. Collision is a hard create failure; no “same ID means retry” rule exists—`operation_id` carries retries.

`operation_id` is deterministically derived by #1065 from the canonical request/retry context and is unique to one logical transition. Reusing it with non-identical canonical bytes is an invariant violation and rejects. Ledger transfer IDs and storage staging IDs are deterministic domain-separated derivatives of it.

## 5. Attestation contracts

### MAC attestation

MAC owns authorization only. It binds the intent digest, policy and accepted-current state epochs, the capability binding, derived controller context, content label commitment, assurance, expiry, and signer identity. The capability binding is profile-dependent: identified/pairwise profiles MAY bind the holder's capability-chain root, while anonymous profiles MUST bind a blinded, operation-scoped commitment that is neither holder-stable nor linkable across operations. Immediately before finalization the registrar asks MAC to revalidate the original attestation against current revocation and policy state. Missing label/context, stale or unreachable authority, policy advance requiring reevaluation, crossed intent, or insufficient assurance denies.

### Ledger attestation

The ledger owns economics only. It binds the same intent digest, payer profile, issuer, ledger/cell, unit, reservation/transfer ID, amount, phase, assurance, and signer identity. Reserve is a maximum. After materialization, post records actual usage and releases the remainder. Void releases a reservation. A posted-but-unfinalized operation is corrected through a new compensation transfer. Neither balance nor receipt is title evidence.

### Finalization check

The registrar MUST verify:

1. exact intent equality, not only caller-supplied digest equality;
2. valid attester signatures and accepted-current signer authority;
3. MAC revalidation immediately before CAS;
4. ledger phase `Posted` (or a resource/profile-specific zero-cost attestation explicitly represented by #1065), unit and amount bounds;
5. content/manifest binding and label resolution;
6. effective assurance = minimum of all verified legs and not below intent requirement;
7. expected predecessor/version and per-resource fence;
8. no terminal outcome already recorded for a different claim under this operation ID.

The Rust mirror makes the load-bearing equalities unrepresentable-to-violate rather than re-checkable: a canonical intent exists only after canonical decoding and digest verification, both attestations embed the digest only and join through a validating constructor that rejects crossed digests, unposted phase, and unit/amount mismatch, and `FinalizedResource` is constructible only by verifying a registrar-signed finalization statement against accepted-current registrar authority.

## 6. Registrar state and visibility

```text
Requested -> MacAuthorized -> LedgerReserved -> Materialized
          -> LedgerPosted -> Finalized

Any pre-post failure: -> Voiding -> Voided
Unknown or posted failure: -> Quarantined -> ManualReview
Quarantined/ManualReview exit only by committing a terminal Resolution:
  Finalized | Voided | Compensated | RejectedConflict
```

Every state and outbound intent is persisted before its external effect. State names describe confirmed effects, never requested effects. The registrar stores attestations by immutable reference and a private recovery record; public records are projections.

Terminal outcomes are explicit, never a generic "reconciled". A terminal outcome is durably committed as an immutable, content-addressed **`Resolution` record** — `Finalized`, `Voided`, `Compensated`, or `RejectedConflict` — and the registration snapshot references that record. The record's consequences are total and fixed by its variant, so two conforming implementations cannot disagree about visibility, bytes, or billing:

| Resolution | Normal namespace | Bytes | Billing |
|---|---|---|---|
| `Finalized` | visible through ordered idempotent projection | retained | posted |
| `Voided` | hidden; any projection withdrawn | eligible for GC | reservation released, never posted |
| `Compensated` | hidden; any provisional projection withdrawn | eligible for GC after evidence | posted charge corrected by an immutable compensating transfer |
| `RejectedConflict` | hidden; never projected for the losing claim | losing provisional bytes eligible for GC | losing side's disposition recorded as a typed void/compensation transfer reference (`ConflictAccountingResolution`); winner named in the record |

A `Resolution` that references an accounting effect — `Voided`, `Compensated`, or the losing side's disposition inside `RejectedConflict` — MUST be committed only after the referenced transfer is durably confirmed by the ledger; until then the operation remains nonterminal. The registration snapshot's state and resolution reference form one checked value: a terminal state requires exactly one matching resolution, and a nonterminal state forbids one — any other pair is rejected at construction (`RegistrationSnapshot::new`).

`Quarantined` and `ManualReview` are **nonterminal**: they permit investigation and authority-approved recovery, but commit no externally visible consequence until exactly one `Resolution` is committed. An identical retry or recovery query reads the snapshot's resolution reference and returns the recorded outcome; it never re-derives one.

| State | Normal namespace | Bytes | Billing | Allowed recovery action |
|---|---|---|---|---|
| Requested / MacAuthorized | hidden | none | none | retry or reject |
| LedgerReserved | hidden | none or staging not begun | pending reservation | stage, void |
| Materialized | hidden | sealed provisional | pending reservation | post, discard+void |
| LedgerPosted | hidden | sealed provisional | posted | finalize or compensate; never ordinary delete |
| Finalized | visible through ordered idempotent projection | retained | posted | publish retry; future transition only |
| Voiding | hidden | provisional only | void/compensation pending | retry effect |
| Voided | hidden | eligible for GC | released | terminal resolution |
| Compensated | hidden | eligible for GC after evidence | immutable compensation recorded | terminal resolution |
| RejectedConflict | hidden | losing bytes eligible for GC | loser voided or compensated per the recorded disposition | terminal resolution |
| Quarantined (nonterminal) | hidden from ordinary namespace; restricted recovery namespace only | retained, immutable | unchanged pending investigation | reconciler/manual review, then exactly one Resolution |
| ManualReview (nonterminal) | hidden by default | retained | unchanged | authority-approved recovery, then exactly one Resolution |

Quarantine is bounded operationally by configured age/depth alarms and admission circuit breakers, but **not** by unsafe automatic publication or history deletion. A timeout moves unknown state to quarantine/manual review; it does not guess an external outcome, and it never commits a `Resolution` by default.

### Namespace projection ordering

The namespace projects finalized registrar state through a durable outbox, and plain duplicate-idempotence is not sufficient: every projection effect is an **ordered desired-state event** carrying the deterministic operation ID, the registrar fencing token (registrar term plus per-resource generation), the resource version, and entry/path identity where applicable. The registrar term is **monotonically increasing** across leadership/authority epochs — the backend term-allocation rule is #1069-owned — and the generation is monotonically increasing per resource within a term, so numeric `(registrar term, generation, version)` tuples are totally ordered. Adapters compare-and-apply: they durably record the newest applied `(registrar term, generation, version)` per resource/entry, return the recorded outcome for exact duplicates, and **reject any effect that is not strictly newer**. A withdrawal identifies the transition/generation it withdraws — never the bare stable resource ID — so a delayed withdraw from an older delete/unlink cannot remove a newer finalized projection, and a delayed publish cannot recreate or regress a projection superseded by a newer transition. Stale-effect rejection is part of this semantic contract (ADRs RO-004/RO-005); adapter mechanics remain child-owned by #1071.

### Private evidence versus public projection

Finalization produces two distinct artifacts, and the boundary is normative:

- The **private finalized record** (registrar output, `FinalizedResource` plus the stored attestation CIDs and recovery record) is authorized-party evidence only. It is constructible solely from a registrar-signed finalization statement verified against accepted-current registrar authority; namespace, storage, PDS, and recovery callers cannot fabricate it. Attestation CIDs resolve only inside access-gated evidence domains — a public or storage observer resolving one MUST NOT obtain attestation detail.
- The **public namespace projection** (`PublicResourceProjection`) is the only input to publication. It carries exactly the stable resource ID, resource version, manifest CID, optional content CID, and an opaque public evidence commitment — commitments and minimum routing/verifiability data only. It never carries attestation CIDs or content, payer, issuer, unit, exact amount, transfer ID, capability root/binding, or signer-key coordinates.

This separation is what makes the per-observer leakage model in `docs/security/resource-ownership-privacy-analysis.md` enforceable: exact amount/issuer/unit remain scoped to origin, issuer, and ledger, and a passive namespace/PDS/storage observer sees only the minimized projection.

## 7. Availability and dependency-failure policy

| Dependency/failure | Before reservation | Reserved/materialized | Posted, not finalized | After finalization |
|---|---|---|---|---|
| MAC unavailable/unverifiable | deny/retry; create no bytes | stop, attempt void; quarantine only if void outcome unknown | quarantine; compensate once accounting available | historical title stands; future control denies until authority recovers |
| MAC denial/revocation | deny | void and discard | compensate; never publish | no history rewrite; future control removed, existing visibility follows current read policy |
| Ledger unavailable | deny/retry | retain hidden, bounded retry; reservation expiry resolves or unknown goes quarantine | quarantine; do not assume post/compensation | reads continue MAC-only; new paid transitions deny when local entitlement state is cold/debt-gated |
| Insufficient/exhausted entitlement | deny with retryable funding signal | void/discard | invariant breach -> quarantine/manual review | no retroactive title effect |
| Store/CAS unavailable | do not reserve if known beforehand; otherwise continue saga | retry bounded; then void | quarantine until materialization evidence and compensation reconcile | existing finalized resource availability follows storage SLA; title unchanged |
| Seal/hash/label failure | n/a | discard and void | quarantine + compensate | never replace finalized bytes in place |
| Registrar store unavailable/corrupt | deny all mutations | deny/recovery-only | deny/recovery-only | reads from last verified finalized projection may continue; no mutation |
| Fence/CAS conflict | n/a | loser voids | loser compensates then terminal conflict | winner remains sole successor |
| Namespace publication failure | n/a | n/a | n/a | finalized but temporarily undiscoverable; durable outbox retries idempotently |
| PDS/proof publication failure | no synchronous dependency | outbox debt only | durable outbox debt | resource remains finalized; threshold stops new receipt-requiring transitions, never erases history |
| Identity/key resolver unavailable | use only valid pinned accepted-current state; otherwise deny | stop and void/quarantine | quarantine | verification uses retained historical key evidence; current control may deny |
| Anonymous issuance/redemption unavailable | anonymous profile denies; identified profile unaffected | same recovery as owning attester | quarantine if outcome unknown | no fallback to identified or unauthenticated principal |
| Relay/exchange/remote cell unavailable | no authority fallback | local lease rules only | reconciliation delayed within prefunded bound | local finalized state stands; cross-cell proof/settlement retries |
| Clock skew/rollback | deny new time-bound operations | conservative expiry; no extension | quarantine uncertain outcomes | historical monotonic timestamps preserved |

Strict finalization is mandatory: publication waits for both attestations and registrar CAS. “Bounded quarantine” bounds operator exposure and new admission, not correctness; it never converts incomplete evidence into finalization.

## 8. Revocation, compensation, keys, and history

- **Before finalization:** revocation or policy advance prevents publication. Reserved entitlement is voided. If already posted, a distinct compensating transfer is required. Bytes remain hidden and are discarded only after recovery evidence permits.
- **After finalization:** finalization is historical fact. Revocation removes future control/delegation and can make future reads deny under current MAC policy, but cannot mutate the old manifest, receipt, or ledger entry. Delete/tombstone is a new transition.
- **Payer/owner disagreement:** payer cancellation can stop an unposted transition but cannot transfer title. Owner/controller cannot force posting without valid entitlement. After posting, disagreement follows compensation/dispute policy; registrar never infers one role from another.
- **Issuer default:** existing issuer-denominated history remains verifiable and its liability does not become another issuer's. New reservations fail. Recovery/settlement is policy/legal territory and cannot rewrite title.
- **Key compromise/rotation:** every attestation and finalization envelope carries signer identity, key identifier, signer-key epoch, key purpose, and suite. Historical verification maps `(signer, key_id, key_epoch)` to retained accepted-current key evidence (#1039) as of the attestation's pinned state epoch: the verifier resolves which key was authorized for that purpose at that epoch, then verifies the signature against that key. The state epoch pins policy/authority state; it is not a substitute for the signer-key epoch. Verification retains historical accepted-current key material and compromise/rotation events. Rotation authorizes future signatures; it does not invalidate correctly verified historical signatures. A key proven compromised at signing time is a dispute marker, not silent history deletion.

## 9. Privacy profiles

| Profile | Owner | Controller | Payer | Origin necessarily learns |
|---|---|---|---|---|
| Identified | DID | DID/delegate | DID/account | role DIDs, exact operation/unit/ceiling/timing |
| Pairwise | pairwise DID | pairwise DID/delegate | pairwise DID/account | per-origin pseudonyms; issuer may know mapping |
| Committed owner | opaque commitment | identified/pairwise or anonymous capability | any payer profile | commitment and operation; reveal rules are profile-owned |
| Anonymous controller | identified/pairwise/committed owner | verified capability commitment | any payer | exact permitted operation, nullifier/commitment, assurance; no stable holder DID |
| Anonymous payer | any owner | any controller | entitlement commitment/nullifier | unit, bounded amount, timing, issuer/ledger context; no holder DID/raw token |

Public manifests and namespace projections contain commitments and minimum routing/verifiability data only (§6 public projection). Detailed receipts and attestations are encrypted/selectively disclosed to authorized parties and never resolve through public CIDs. In anonymous profiles, no field surfaced in any attestation or public projection may be holder-stable or holder-linkable: the capability binding is a blinded, operation-scoped commitment rather than the holder's capability-chain root, and transfer IDs are deterministic per-operation derivatives of the operation ID, so they are per-operation by construction. Origin audit/trust stores MUST NOT contain raw anonymous tokens, holder/root DIDs, ATProto handles, stable client keys, holder-stable capability roots, or directly linkable allocation identifiers. No protocol field carries legal identity; KYC remains optional and external at a fiat conversion boundary.

Pairwise identities prevent trivial cross-cell joins, not traffic analysis. Fixed/coarse denominations, batching, delayed/fixed-cadence publication, aggregate reconciliation, encrypted details, and pseudonymous transport reduce linkage. V1 does not claim resistance to issuer-origin-ledger collusion or a global passive adversary.

Commitment bytes are deliberately opaque here. Hash/Pedersen/BBS+/Privacy-Pass/nullifier constructions are not interchangeable and are not selected by this issue. Production anonymous finalization remains blocked on #1059–#1062 and independent cryptographic/privacy review; there is no classical or identified fallback.

## 10. Mutable 9P decisions

These decisions unblock interface design while #1071 remains activation-gated on the immutable pilot:

- **Path versus resource:** path identifies a directory entry; `resource_id` identifies title-bearing object. Rename changes an entry, not title.
- **Directories:** a directory owns its entries and creation/link authority, not child resource title. Non-empty removal follows namespace semantics; child title/history is unaffected.
- **Hard links:** multiple entries may reference one `resource_id`. Link creation requires source-resource control plus destination-directory mutation authority. Link count is namespace state, not title duplication.
- **Rename:** same-domain rename is an atomic entry move under complete MAC mediation. Cross-ownership/security-domain rename follows a deterministic, fail-closed selection rule: policy MUST authorize exactly one named mode — `relink` (unlink+link) or `copy-on-write` — and absent or ambiguous policy denies the operation. For `relink`, the transition requires source- and destination-directory mutation authority and preserves the same `resource_id` and title; entry changes are one ordered transition under predecessor CAS. For `copy-on-write`, the destination resource MUST complete dual-attested finalization first; only then does an atomic namespace compare-and-swap replace the destination entry and remove/replace the source entry, and the new content lives under a new title-bearing resource lifecycle. Crash or stale-fence mid-sequence leaves entries unchanged or fully swapped — never duplicated or lost — because each step is an idempotent ordered projection effect (§6); recovery replays the recorded step for the operation ID. Neither mode is ever an implicit title transfer.
- **Replacement:** replacing a destination entry does not transfer either object's title. It tombstones/unlinks the old entry and links a separately finalized resource/version.
- **Unlink/tombstone:** unlink removes an entry. Last unlink does not erase title or attested history; retention/GC requires an explicit delete/tombstone transition and policy.
- **Bind/mount/export:** namespace composition delegates reachability/control and is clamped at the binder/import floor. It never transfers title. Remote labels are hints until perimeter-mapped.
- **Writes:** mutable content advances a resource version under predecessor CAS. Bounded local leases settle at close/commit/exhaustion/checkpoint/expiry; fine-grained writes do not call the ledger.

## 11. Interface placement

- `hyprstream-resource`: dependency-light semantic types and traits only.
- #1065: canonical CDDL/DAG-CBOR, full `ResourceIntent`, manifests, vectors, compatibility policy.
- #1068: MAC implementation of `MacAuthority` using content labels and accepted-current state.
- #1072: service-layer implementation of `EconomicAuthority`; `hyprstream-ledger` remains pure accounting and RPC/VFS/PDS-free.
- #1069: durable registrar implementation and persistence/outbox/reconciler.
- #1066: CAS implementation of `ResourceMaterializer` and finalized publication adapter.
- #1071: VFS/9P namespace adapter consuming finalized resources.
- PDS: proof record codec/store only, never registrar state.

## 12. Review and freeze gates

This design is reviewable, not self-ratifying. Before canonical wire/persistence freeze and production enablement, record independent reviews on the exact commit:

1. **Architecture:** one-owner table, lifecycle, fencing, 9P semantics, dependency layering.
2. **Security:** replay/substitution, complete mediation, accepted-current keys, downgrade and recovery behavior.
3. **Privacy:** per-profile data flow, public/private fields, collusion and timing/amount leakage.
4. **Standards/protocol:** terminology, standards overlap, non-allocation/non-endorsement, schema evolvability.
5. **Cryptography (anonymous profiles):** separately required by #1059–#1062 before restricted anonymous enablement.

Findings must be resolved or recorded as accepted residual risk with an owner. Until all four #1067 reviews pass, schema work is provisional and production feature flags remain off.
