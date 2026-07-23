# Resource ownership threat model

**Scope:** #1064 lifecycle and #1067 authority contract. Anonymous issuance cryptanalysis remains in #1059–#1062.

## Assets and security properties

- uniqueness of canonical successor per resource predecessor/version;
- separation of title/control from payment and storage custody;
- exact binding of both attestations to one intent;
- hidden provisional/quarantined material;
- accounting conservation and immutable compensation;
- role, pseudonym, token, and detailed-receipt confidentiality;
- historical verification across key rotation/compromise records;
- availability without weaker authorization fallback.

## Trust boundaries and adversaries

| Actor/collusion | Capabilities | Required containment | Residual |
|---|---|---|---|
| Malicious requester/controller | replay/substitute intent, assert role/label, race transitions | canonical digest, verified subject/capability, operation idempotency, registrar CAS | authorized actor can create harmful content within granted scope |
| MAC operator alone | deny or issue improper title authorization | independent ledger cannot be forged; signed/audited policy/state epoch | malicious MAC authority can mis-authorize control inside its policy domain |
| Ledger operator alone | deny/reserve/post improperly, correlate payer | cannot confer title; registrar requires MAC; signed receipts/checkpoints | can censor accounting and observe cell-local amounts/timing |
| Registrar operator alone | reorder/withhold lifecycle, attempt equivocation | cannot forge attestations; per-resource CAS/fence; external proof copies | can censor and delay; backend compromise before external evidence can damage availability |
| Storage operator alone | lose/withhold/corrupt/probe bytes, exploit dedup | content verification, hidden staging, domain-scoped dedup, no title inference | observes size/timing/access unless encrypted/padded |
| PDS observer/operator | correlate publications, suppress records | commitments/encrypted details/fixed cadence; PDS not lock authority | publication timing/size and public role metadata |
| Relay/network observer | traffic analysis, replay/drop opaque messages | authenticated binding, nonce/expiry, opaque content, pseudonymous transport | global passive correlation out of scope |
| Issuer + origin | join issuance and redemption metadata | pairwise IDs, blind issuance lane, batching/coarse denomination | v1 timing/amount correlation; no collusion-resistance claim |
| Origin + ledger | join operation and charge/nullifier | aggregate/pseudonymous accounts, encrypted details | exact local operation/amount is inherently visible to origin |
| Issuer + ledger | join holder/allocation/account data | origin sees commitment only in anonymous profile | issuer already knows its bilateral underwritten holders; timing join remains |
| Issuer + origin + ledger | near-complete transaction view | structural data minimization only | v1 anonymity may collapse by timing/amount; explicitly residual |
| Cross-cell counterparties | settlement manipulation, receipt omission, correlation | prefunded bounds, dual-signed receipts, set reconciliation | bilateral volume and tranche metadata |
| Compromised key | forge current signatures | accepted-current epochs, rotation/revocation, historical key evidence | signatures made during undetected compromise require dispute handling |

## Attack classes and controls

1. **Cross-claim substitution:** resource, operation, origin, unit, payer, policy, predecessor, carrier, or profile is changed between attesters. Control: both independently derive exact canonical bytes; accepted-current verification mints private evidence before `verify_and_join` rejects crossed digests, wrong key purpose, unposted phase, and unit/amount mismatch.
2. **Payment-as-title / authorization-as-credit:** one valid attestation is reused as both. Control: distinct raw/verified types, signer authorities, domain separators, mandatory dual verification, and an independence policy that rejects shared roots, signers, or keys.
3. **Replay/idempotency collision:** same operation ID carries different bytes or one attestation is replayed. Control: durable outcome index; non-identical reuse is invariant violation; transfer/staging IDs derive from operation ID.
4. **Double finalization:** concurrent successors race. Control: expected predecessor/version CAS plus monotonic per-resource fence; one winner.
5. **Premature visibility:** staged or posted bytes leak via CAS/9P. Control: non-public staging identity; namespace consumes only a registrar-minted projection effect originating from `FinalizedResource`, which checks an accepted-current registrar statement and matching verified attestation CIDs. Unauthenticated or invalid finalizations cannot publish. Valid signatures made with an undetected compromised current key remain subject to dispute/revocation; the public projection is privacy-minimized and quarantine namespace separately authorized.
6. **Revocation race:** authority revoked after early MAC check. Control: pin epochs and revalidate immediately before finalization; compensate if post preceded failure.
7. **Confused deputy/declassification:** caller supplies role/label or bind re-exports lower. Control: derive verified context; labels content-bound and perimeter-clamped; binder-floor join; delegation records both principals.
8. **Assurance laundering:** classical outer/session wraps stronger inner claim. Control: total-order minimum over every leg; missing leg is `Unverified`; no fallback profile.
9. **Recovery ambiguity:** timeout is treated as failure and effect repeats. Control: persist-before-effect, idempotent queries/retries, unknown state quarantined.
10. **Dedup possession oracle:** cross-principal bytes-stored result reveals existing content. Control: dedup accounting and response scoped to an authorized privacy/storage domain; no title transfer; consider constant/coarsened responses in anonymous profiles.

## Availability threat posture

Authority unavailable means deny or retain hidden state; never use a weaker fallback. Pinned accepted-current state may be used only through its validity window. Circuit breakers stop new transitions on registrar corruption, cold ledger state, receipt/audit debt, or quarantine overflow. Existing finalized reads can remain available when their local MAC decision and bytes are valid because ordinary reads do not depend on ledger/PDS availability.

## Security review obligations

Independent review must challenge canonicalization, domain separation, role confusion, policy/key epoch semantics, registrar backend linearizability, every crash boundary, label provenance, compensation, anonymous downgrade, and historical verification. This document does not close those reviews by asserting them.
