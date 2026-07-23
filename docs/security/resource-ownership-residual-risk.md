# Resource ownership residual risk

This document records limits that remain after the #1067 architecture controls. Acceptance requires an explicit owner; it is not evidence that the risk is solved.

| Residual risk | Why it remains | Current treatment / owner |
|---|---|---|
| Issuer-origin-ledger timing/amount deanonymization | metadata can correlate even blind issuance | no v1 collusion-resistance claim; batching/coarse profiles; #1059/#1072 + privacy review |
| Global passive traffic analysis | Tor/Snowflake does not defeat a GPA | out of scope; transport profile documents floor |
| Malicious authority censorship | MAC, ledger, registrar, storage, or PDS can refuse service | fail closed, HA/caching/recovery; no weaker fallback |
| Malicious MAC authority mis-authorization | MAC is authoritative inside its policy domain | signed policy/epoch, audit, independent ledger; governance remains trusted |
| Ledger equivocation/history rewrite before externally held evidence | local ledger is tamper-evident, not tamper-proof | receipts/checkpoints/PDS copies; optional fixed-cadence anchoring |
| Registrar backend compromise | sole canonical writer is a concentration | linearizable CAS, audit/proof copies, recovery; deployment HA review |
| Storage loss and access-pattern leakage | content addressing proves integrity, not availability/privacy | replication/encryption/padding outside this contract |
| Dedup possession oracle | storage savings/timing may reveal existing bytes | domain-scoped dedup and coarsened responses; CAS privacy tests |
| Key compromise before detection | forged facts may verify under then-current key | historical accepted-state evidence and dispute markers; cannot silently erase facts |
| Issuer insolvency/default | credits are liabilities, not guaranteed external value | stop new spend; retain issuer identity/history; external recovery policy |
| Commitment construction weakness | #1067 deliberately selects none | production anonymous/committed profiles blocked on #1059–#1062/#1065 review |
| PQ algorithm transition | current hybrid can age or one leg fail | pinned suite/version, no in-band downgrade, deliberate migration |
| Resource-ID random collision | 128-bit collision is improbable, not impossible | hard create rejection; never retry inference |
| Quarantine accumulation / disk exhaustion | safety retains unknown/post material | bounded admission circuit breaker, alerts, operator recovery; never unsafe auto-delete |
| Historical public commitment linkability | stable resource history is intentionally verifiable | minimize fields; profile warns committed owner may link versions |
| 9P client semantic differences | kernel/FUSE/browser adapters may expose different atomicity | #1071 conformance; mutable activation waits for immutable pilot |
| Standards evolution/codepoint collision | cited MoQ/Privacy Pass work is not final/allocated | private-use/local profile, explicit disposition, standards review |

Production flags remain disabled until architecture, security, privacy, and standards reviews are recorded on the exact schema/implementation head, plus cryptographic review for anonymous profiles.

## Security review F4–F7 dispositions (#1067 revision)

- **F4 (fencing term/effector coverage) — fixed in the scaffold.** `FencingToken` now carries a registrar term alongside the per-resource generation, and the fence is threaded to the effectors (`ResourceMaterializer::stage/seal/discard`, namespace `ProjectionEffect` compare-and-apply), so a deposed primary's effects are distinguishable and rejected. Owner of the backend term/leadership design: #1069.
- **F5 (amount/phase invariants at the seam) — partially fixed, remainder accepted with owner.** `TransferId` is now a newtype whose derivation from `operation_id` is architecture-owned (§4). `DualAttestation::verify_and_join` verifies both evidence legs before enforcing `amount <= max_charge`, unit equality, and posted phase. A `BoundedCharge` newtype over `(reservation, actual)` was judged redundant with this seam; the `EconomicAuthority::post`/`compensate` phase preconditions remain documented MUSTs on the trait. Enforcement inside the ledger implementation: #1072.
- **F6 (role-enum divergence from epic #1064 text) — recorded, no code change.** The crate's `Pairwise(Did)` owner/controller/payer variants are intentional; the epic text predates them. ADR RO-002 records that #1065 freezes the wire schema from this crate's vocabulary, and the epic text should be amended rather than the variants dropped.
- **F7 (tautological tests / unused dep / unpinned time) — fixed in the scaffold.** Negative controls cover intent/predecessor shape, attestation phase/charge/purpose/independence, finalization purpose/provenance/content, projection digest binding, and terminal-resolution target/reason binding. `expires_at` is pinned to Unix seconds on the registrar's monotonic-clamped time basis. #1065/#1069 inherit the requirement that validating constructors ship with mutation-effective negative tests.
