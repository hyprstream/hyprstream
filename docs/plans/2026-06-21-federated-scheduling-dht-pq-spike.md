# Spike — Federated inference scheduling at 10k–1M nodes: DHT/overlay + PQ-readiness

**Date:** 2026-06-21 · **Status:** research spike (investigation + design; **NO production code**) ·
**Branch:** `ewindisch/dht-scheduling-spike` (off `ewindisch/310-multi-gpu` @ `27d702ce1`).
**Scope:** the **global tier** above #322's bounded leaf-tier cell router. Answers: which structured
overlay (DHT), whether it is post-quantum-safe to use, whether it lets us *avoid* SWIM, and how
session affinity / cross-cell replication look at scale.

> All file:line citations are against this worktree's tree (`.worktrees/ewindisch/dht-scheduling-spike`,
> = `ewindisch/310-multi-gpu` @ `27d702ce1`). Versions are read from `Cargo.lock`, not built.

---

## TL;DR

- **A single broadcast heartbeat topic + a global HRW ring do not scale to 1M nodes** (O(N)/router,
  constant churn). The architecture must be **hierarchical/federated**: schedule **within** a trust
  domain/cell (a PDS) using #322's bounded gossip+HRW, and route **between** cells via a structured
  overlay. This spike designs the between-cells tier.
- **Two DHT candidates are in-tree.** **iroh/pkarr** (`iroh = 1.0.0-rc.0`) is **resolution-only**
  (NodeId→address, self-published on the BitTorrent Mainline DHT via BEP44) — it **cannot do
  key→owner placement** because BEP44 mutable-item keys are `SHA1(pubkey‖salt)`, i.e. you can only
  publish at addresses you hold the private key for. **libp2p-kad** (`libp2p = 0.56`, `libp2p-kad
  0.48`) is a general Kademlia with `PUT_VALUE`/`GET_VALUE` at **arbitrary keys** → it **is
  placement-capable**, but it is currently wired **only inside `gittorrent`** and carries a **second
  identity** (`PeerId`) disjoint from our iroh `node_id`/did:key.
- **RECOMMENDATION (substrate):** **do not adopt a general placement-DHT as the inter-cell store
  yet.** Build the global tier as a thin **cell-directory** layered on the **identity and transport
  we already have** (iroh dial + the `Resolver` trait + the federation admission gate), using
  Kademlia/HRW *as an algorithm over a bounded set of cells*, not over all nodes. If/when a true
  placement-DHT is needed (very large cell counts, no central PDS-of-record), **reuse libp2p-kad** —
  it is the better-proven Kademlia and already compiled — and bridge `PeerId↔node_id` via a
  COSE-signed binding record. pkarr stays what it is: NodeId→address resolution (already our
  discovery path, #282).
- **PQ verdict:** **classical-DHT-as-untrusted-hints is safe — PQ-signed DHT records are NOT
  required for confidentiality/integrity of tenant data.** Every session is authenticated by our
  **hybrid-PQ COSE** (EdDSA + ML-DSA-65, `cose_sign.rs`) and sealed by **AES-256-GCM** (#321,
  256-bit = PQ-fine). A (quantum-)forged DHT record only achieves **misdirection → denial**: the
  attacker node still fails mesh authz (#319/#328 — unknown pubkey → `anonymous`, deny-by-default)
  and cannot decrypt tenant data → **no compromise, no HNDL on tenant content.** The one residual is
  **routing-poisoning DoS** (forged records that black-hole lookups). That is a DoS, not a breach,
  and is mitigable *without* PQ-signing the whole DHT (see §4). **So: classical DHT as hints + the
  existing PQ work-plane is the recommended posture; PQ-sign records only at the directory layer we
  control, not the underlying Mainline/Kademlia keyspace.**
- **SWIM verdict:** **we can drop a separate SWIM/epidemic membership protocol for the global
  tier.** Kademlia's intrinsic k-bucket maintenance (lazy liveness, long-lived-peer bias) +
  **reactive** failure detection (lookup/dial fail → recompute → reassign; the per-request
  token-stream stall watchdog already designed for #322) is sufficient *because cells, not nodes, are
  the global-tier members* (10k–100k cells, not 1M nodes) and **git ref-CAS remains the only
  hard-consistency point.** SWIM stays *optional, intra-cell* (#322) where proactive capacity
  rebalancing has value; it is **not** warranted at the global tier.

---

## 1. The tiered architecture (leaf #322 ↔ DHT global)

```
                        ┌─────────────────────────────────────────────┐
   client / ingress     │  GLOBAL TIER  (inter-cell)                   │
        │               │  cell-directory over a structured overlay    │
        │  cached?──yes──┼─► direct dial cell (ARP-cache hit)          │
        │   │           │  cold/failed ─► one overlay hop ─► home cell │
        │   no          │  members = CELLS (~10k–100k), not nodes      │
        ▼   ▼           │  liveness = k-bucket + reactive (no SWIM)    │
   ┌──────────┐         └───────────────┬─────────────────────────────┘
   │ home cell│ ◄───────────────────────┘
   │  (= PDS, │   LEAF TIER (intra-cell, #322 — NOT this spike)
   │  trust   │   capacity-weighted rendezvous hashing (HRW) over a
   │  domain) │   bounded set (tens–low-thousands), session→owner
   └────┬─────┘   affinity (KV-cache stickiness = perf hint, not
        │         correctness), router-side/gossiped load, 2-layer
        ▼         failure detection (dial-fail + token-stream stall)
   InferenceService (node) ── owns 1+ GPUs ── serves the request
        │
        └─ durable session writes (TTT/LoRA deltas) ─► git ref-CAS
           reconciled by the #326 delta reducer  ◄── ONLY hard-consistency point
```

**Cell = trust domain = a PDS.** hyprstream is already federated via did:web / atproto /
`federation:register` (admission gate at `crates/hyprstream-rpc/src/admission.rs:1–200`: two-stage
fail-closed — RFC 6454 origin → `federation:register` policy, then Ed25519 channel-key↔DID-VM
binding). A cell is the natural unit of trust, authz, and capacity accounting; the leaf tier (#322)
schedules *within* it; the global tier routes *between* cells.

**The two tiers have different members and different consistency:**

| | Leaf tier (#322) | Global tier (this spike) |
|---|---|---|
| Members | nodes (InferenceServices) | **cells** (PDSes) |
| Cardinality | tens–low-thousands per cell | ~10k–100k cells (gets us to 1M nodes) |
| Routing | capacity-weighted HRW | overlay lookup (Kademlia/HRW over cells) |
| Liveness | gossip + dial-fail + stall watchdog | k-bucket + reactive (no SWIM) |
| Affinity | session→owner (KV-cache hint) | session→**home cell** (durable binding) |
| Consistency | none (HRW is deterministic; load is a hint) | none, except git ref-CAS at the leaf |

The decisive scaling move is **federating membership by cell**: a global HRW/Kademlia keyspace over
~10⁴–10⁵ cells is tractable (O(log N) lookup, bounded routing table); a global ring over 10⁶ nodes is
not. Within a cell, #322's bounded router is unchanged.

---

## 2. DHT / structured-overlay options

### 2.1 The placement-vs-resolution distinction (the crux)

The single most important property for the global tier is: **does the overlay give *placement*
(arbitrary `key → node`, consistent-hash-style — e.g. `session_id → owning cell`) or only
*resolution* (`name → address`, where the name *is* the owner's key)?**

- **Resolution-only** overlays (pkarr/iroh DNS, BEP44 mutable items): the DHT key is
  *derived from* the publisher's public key — `target = SHA1(ed25519_pubkey ‖ salt)`
  (BEP44). You can **only** publish at addresses you own the key for. This is perfect for "where is
  node X?" and **useless** for "who owns session S?" — there is no node that holds the private key
  for `H(session_id)`. *Needs validation against pkarr internals, but BEP44 is unambiguous on this.*
- **Placement-capable** overlays (libp2p-kad `PUT_VALUE`/`GET_VALUE`, classic Kademlia): a record at
  an *arbitrary* key is stored on the *k closest nodes to that key by XOR distance*. This is exactly
  consistent-hash placement; it is what lets `H(session_id) → set of owning peers` work.

> hyprstream's leaf tier already uses **HRW (rendezvous hashing)** for placement within a cell.
> HRW *is* placement; the question is purely whether we need a *distributed* placement store at the
> global tier, or whether a directory + bounded HRW-over-cells suffices (see §6 recommendation).

### 2.2 Options table

| Property | **iroh / pkarr** | **libp2p-kad (+gossipsub)** | Mainline DHT (direct) | custom Kademlia |
|---|---|---|---|---|
| In-tree? | **yes** — `iroh 1.0.0-rc.0`, used by transport substrate (`iroh_substrate.rs`) | **yes** — `libp2p 0.56` / `libp2p-kad 0.48`, **kad/mdns/identify instantiated** in `gittorrent` only (`crates/gittorrent/src/dht/behaviour.rs:14–62`) | via pkarr (Mainline = pkarr's backend) | no |
| **Placement** (key→owner)? | **NO** — BEP44 key = `H(pubkey‖salt)`; self-publish only | **YES** — `PUT_VALUE`/`GET_VALUE` at arbitrary keys, k-closest-by-XOR | NO (BEP44, same constraint) | YES (by design) |
| **Resolution** (name→addr)? | **YES** — NodeId→relay/addr DNS packet on Mainline, Ed25519-signed; **already our discovery path** (`presets::N0`, #282) | YES (peer-routing) | YES | YES |
| Lookup complexity | O(log N) (Mainline) | O(log N) | O(log N) | O(log N) |
| Churn handling | Mainline absorbs churn (10M+ nodes, 15yr) but **records are short-TTL, must republish** | k-bucket eviction/refresh; long-lived-peer bias | as pkarr | DIY |
| Intrinsic membership/liveness | k-bucket (Mainline) — but we don't run the k-buckets, we publish to it | **k-bucket maintenance built-in** (eviction on unresponsive, periodic refresh) | k-bucket | DIY |
| gossipsub (epidemic pubsub)? | n/a | **available in libp2p, but NOT enabled in our feature set** (`gittorrent/Cargo.toml:72` = `kad,mdns,noise,tcp,tokio,yamux,identify,request-response,macros` — **no `gossipsub`**); not in `Cargo.lock` | n/a | DIY |
| Identity fit | **native** — iroh `node_id` = Ed25519 = our mesh root; same key family as did:key | **second identity** — `PeerId = hash(libp2p pubkey)` (`behaviour.rs:24`), **disjoint** from iroh node_id; needs a bridge (see §2.3) | Ed25519 (Mainline) | choose ours |
| Maturity | iroh `1.0.0-rc.0` (RC); Mainline very mature | libp2p-kad mature, widely deployed (IPFS/Filecoin/eth2) | Mainline very mature | none |
| PQ surface | classical (Ed25519 records, X25519 transport) — but see §3 (QUIC plane can do hybrid KEM today) | classical (Ed25519/secp256k1 PeerId, Noise X25519, classical record sigs) | classical | our choice |

### 2.3 The libp2p `PeerId` ↔ iroh `node_id` bridge — quantified

We have **4 distinct identity types** today (agent-confirmed):
`iroh node_id` (Ed25519, transport, `iroh_substrate.rs:85`), `did:web` (Ed25519 `#mesh` VM),
`did:key` (Ed25519, self-certifying), and **`libp2p PeerId`** (`PeerId::from(keypair.public())`,
`gittorrent/.../behaviour.rs:24`). The first three are one Ed25519 key family bound through the
admission gate; **`PeerId` is a separate universe** used only by gittorrent.

Adopting libp2p-kad for the service mesh adds `PeerId` as a *fifth* live identity on the mesh path.
Bridge cost = **a signed binding record** `{ peer_id, node_id, did }` published into the directory and
verified with the **existing hybrid COSE** (`cose_sign.rs`) so the binding inherits our PQ posture
(the *record* is PQ-signed even though Kademlia's transport/PeerId stay classical — exactly the
"PQ-sign the directory layer we control" posture in §4). This is real but bounded work; it is the
main reason the recommendation is "directory-first, libp2p-kad only if a true placement-DHT is
needed." A libp2p Ed25519 keypair *can* be derived from the same node root, but `PeerId` is
`multihash(pubkey)`, not the raw 32 bytes, so it is still a distinct on-wire identifier requiring the
binding record for authz.

---

## 3. Post-quantum readiness assessment

### 3.1 Classical-crypto surface each option introduces

| Surface | iroh/pkarr | libp2p-kad | hyprstream today |
|---|---|---|---|
| Node ID | Ed25519 (classical) | Ed25519/secp256k1 `PeerId` (classical) | Ed25519 mesh root (classical) **+ ML-DSA-65 `#mesh-pq`** derived per-host (`derive_mesh_mldsa_key`) |
| Record signing | Ed25519 (BEP44) | Ed25519 record sigs | **hybrid COSE** EdDSA+ML-DSA-65 (`cose_sign.rs`) |
| Transport handshake / KEM | QUIC: rustls — **`rustls 0.23.40` in lock → X25519MLKEM768 default since 0.23.27** (hybrid KEM available **today**); Mainline UDP itself unencrypted | Noise XX over **X25519 only** (classical KEM; rust-libp2p Noise has **no** ML-KEM as of 2026 — *needs validation, but the 2021 PQ issue #2168 closed "not planned"* and the current Noise crate documents X25519 only) | stream plane: **Ristretto255 DH + AES-256-GCM** (`key_exchange.rs`, #321); RPC: QUIC (rustls 0.23, hybrid-capable) |

**PQ roadmaps (cited, mark uncertain as noted):**
- **pkarr / Mainline:** classical only. Mainline (BEP44) "exclusively supports Ed25519"; pkarr's
  minimalist design leaves crypto upgrades to the application layer. No PQ roadmap.
  *(pkarr docs / pubky.github.io/pkarr; bittorrent.org BEP44.)*
- **libp2p:** the 2021 PQ tracking issue (rust-libp2p #2168) was **closed "not planned."** Noise
  still documents X25519 only. No shipped ML-KEM in libp2p Noise as of mid-2026. **Needs
  validation** against the latest libp2p-noise CHANGELOG, but treat libp2p transport as
  **classical, no near-term PQ.**
- **iroh / quinn / rustls:** **best-positioned.** `rustls-post-quantum` X25519MLKEM768 moved into
  rustls core; **default-enabled since rustls 0.23.27**; our lock has **rustls 0.23.40**. QUIC/HTTP3
  hybrid KEM is in production elsewhere (Chrome, Cloudflare). So our **QUIC RPC plane can negotiate a
  hybrid KEM today**; pre-standardization → treat as experimental, but the dependency is present.
  *(crates.io rustls-post-quantum; rustls 0.23.27/0.23.40; IETF draft-ietf-tls-ecdhe-mlkem-05.)*

**Crucial grounding:** the PQ primitives are **not a new dependency** — `hyprstream-rpc` already
depends on **both `ml-dsa 0.1.0-rc.11` and `ml-kem`** (`Cargo.lock`, `hyprstream-rpc` deps list).
PQ-protecting a record or a handshake is **new *usage*, not a new crate.**

### 3.2 Threat model — untrusted hints validated by a PQ work-plane

**Claim under test:** *if the DHT is used only as untrusted discovery/routing **hints**, and every
session is authenticated by hybrid-PQ COSE + encrypted by AES-256-GCM, then a (quantum-)forged DHT
record achieves only **misdirection → denial**, never compromise or HNDL of tenant data.*

**Walk the attack with a forged/altered DHT record (classical-DHT, post-quantum adversary):**

1. **Forged "cell C owns session S" or forged "node X is at address A".** The client follows the
   hint and dials the attacker.
2. **Attacker must pass mesh authz to be served / to serve.** Admission is fail-closed
   (`admission.rs`): origin→`federation:register`, then Ed25519 channel-key↔DID-VM binding. Mesh RPC
   resolves the peer's Ed25519 to `service:inference:host-<label>`; **unknown pubkey →
   `Subject::anonymous()` → deny-by-default** (`node_identity.rs` `register_peer`,
   `mesh_host_subject`). A forged *routing* record does not forge an *enrolled identity*. ⇒ attacker
   **cannot join the work plane.**
3. **Attacker cannot read tenant data.** The stream is sealed under AES-256-GCM with keys derived
   from the **client-bound ephemeral DH** (`derive_stream_keys`, Ristretto255); 256-bit AEAD is
   PQ-fine. Misdirection to an attacker yields ciphertext under a key the attacker doesn't hold. ⇒
   **no confidentiality break, no HNDL on tenant content.**
4. **Provenance is hybrid-signed.** StreamBlocks / envelopes carry the EdDSA+ML-DSA-65 composite
   (`cose_sign.rs`, `streaming.rs` WNS test): forged content fails the inner EdDSA *and* (when the
   signer is anchored) the outer ML-DSA layer. ⇒ **no integrity break.**

**Verdict on the framing:** it **holds for confidentiality and integrity of tenant data.** The
attacker who forges a DHT record gains **only misdirection, which is a denial-of-service** (the
client wastes a dial, fails authz/decrypt, must retry). It does **not** yield compromise or HNDL.

**Where the framing does NOT fully hold — the one residual:** **routing-poisoning DoS.** A
sufficiently capable adversary (esp. one with future quantum forgery of Ed25519 DHT-record sigs, or
classical Sybil/eclipse on Mainline today) can **black-hole or eclipse lookups** so that *legitimate*
cells become unreachable. That degrades availability, not confidentiality. It does **not** force
PQ-signing the *underlying* Mainline/Kademlia keyspace (which we cannot do anyway — Mainline is
Ed25519-only and globally shared). It **does** argue for:
- **PQ-signing the records *we* control** at the directory layer (the cell-directory record and the
  `peer_id↔node_id` binding), using the existing hybrid COSE — so the *payload* is PQ-authentic even
  while the overlay's transport/keyspace stays classical; and
- **defense-in-depth against eclipse**: redundant lookups (k-closest, multiple disjoint paths),
  cell-replica diversity (§5), and the reactive watchdog that treats a black-hole as a dial-fail and
  recomputes. These are availability mitigations, independent of PQ.

### 3.3 Concrete PQ-safe-use recommendation

1. **Treat the DHT/overlay strictly as *untrusted hints*.** Never let a DHT record grant authz or
   carry plaintext. Authz = the existing fail-closed admission + mesh-host-subject roster
   (#319/#328). Confidentiality = AES-256-GCM under client-bound DH (#321).
2. **PQ-sign the records *we* own** (cell-directory entries, `peer_id↔node_id`/`did` bindings) with
   the **hybrid COSE** already in `cose_sign.rs` — WNS posture: enforce ML-DSA for anchored
   directory signers, classical floor otherwise, **never blanket fail-closed.** This is new *usage*
   of existing primitives (`ml-dsa` already a dep).
3. **Prefer the QUIC plane for any sensitive inter-cell control** so we ride **rustls
   X25519MLKEM768** (present in `rustls 0.23.40`) — hybrid KEM today, no libp2p Noise PQ gap.
4. **Do not gate availability on PQ-signed Mainline records** (impossible) — mitigate
   routing-poisoning with redundancy + reactive recompute, not by trying to PQ-harden a globally
   shared classical DHT.

**Bottom line:** **classical-DHT-as-hints is safe; PQ-signed DHT records are NOT required for
security of tenant data.** PQ-sign only the directory-layer records we control (cheap, existing
crate). The residual is availability (routing-poisoning DoS), handled by redundancy, not by PQ.

---

## 4. Can we AVOID SWIM if we use the DHT?

**Question:** is the overlay's intrinsic routing-table liveness (k-bucket eviction/refresh) +
**reactive** failure detection (lookup/dial fail → recompute → reassign; per-request token-stream
stall watchdog) sufficient to **drop a separate SWIM/epidemic membership protocol** at the global
tier?

**Mechanics:**
- **Kademlia k-buckets** give *lazy, passive* liveness: unresponsive contacts are evicted on use,
  buckets refresh periodically, long-lived peers are preferred (stability bias). This is
  **maintenance**, not a fast failure *detector*.
- **SWIM** gives *proactive* death detection: randomized ping / indirect ping-req, fast
  dissemination — detection in O(seconds) independent of whether anyone is routing to the dead node.

**Verdict: YES — drop SWIM at the global tier.** Justification grounded in this architecture:
1. **Global-tier members are CELLS (~10⁴–10⁵), not nodes (10⁶).** Death detection of a *cell* is
   coarse and rare; we don't need sub-second fleet-wide node death detection at the global tier —
   node death is a **leaf-tier** concern (#322), where load is local and the stall watchdog already
   fires per-request.
2. **Detection is already reactive end-to-end.** #322 specifies dial-fail + token-stream **stall
   watchdog**; the global tier inherits this — a dead/eclipsed cell manifests as a dial-fail or
   stall on the *next request*, triggering recompute (HRW-over-cells / next-k-closest) and reassign.
   We pay for liveness **only when we route**, which is exactly when we care.
3. **No hard-consistency to protect.** The only hard-consistency point is **git ref-CAS** at the
   leaf (#326 reducer reconciles). Stale membership cannot corrupt state — a write that lands on the
   wrong/old owner loses the CAS and retries. Membership is a *performance hint*, mirroring the
   KV-cache affinity stance. SWIM's value (a crisp authoritative view) is unnecessary when
   correctness never depends on the view.

**Cost of dropping SWIM (state it honestly):**
- **Lazy vs proactive death detection.** Without SWIM, a dead cell/node is discovered *on demand*
  (first failed request eats one dial-timeout / one stall-timeout before recompute), not *ahead of
  time*. Worst case: a burst of cold requests to a just-died cell each pay one timeout before the
  ARP-cache invalidates. Mitigation: short negative-cache TTL on dial-fail; the stall watchdog
  bounds the per-request penalty.
- **No proactive capacity rebalancing.** SWIM-style gossip can also carry *load/capacity* for
  pre-emptive rebalancing. Reactive routing only rebalances *after* a hot node degrades. This is the
  one place SWIM (or a lightweight load-gossip) **would still be warranted — and it belongs
  intra-cell (#322), not at the global tier.** #322 already contemplates router-side or gossiped
  load; keep that option *within* a cell where the set is bounded and proactive rebalancing pays off.

**Recommendation:** **No SWIM at the global tier.** Use k-bucket maintenance + reactive
failure-detection + short negative-cache. Keep **optional, bounded, intra-cell load-gossip** in #322
for proactive capacity rebalancing — that is the only justified proactive-membership use, and it
operates over tens–thousands of nodes, not the global fleet.

---

## 5. Hierarchy & session affinity at scale

### 5.1 Session → home-cell binding

A session must resolve to a **home cell** (its trust domain / PDS) deterministically and cheaply:
- **Option A — encode the cell in the `session_id`** (e.g. `session_id = cell_id ‖ random`). Zero
  lookups: the client/ingress reads the home cell directly from the id. Simplest; the home cell is
  fixed at creation and is the unit that owns the git refs for that session's deltas.
- **Option B — a sharded session-directory** mapping `session_id → home_cell`, itself placed by the
  overlay (`H(session_id) → directory shard`). One overlay hop on a cold lookup; allows
  **rebinding** a session to a new home cell (migration) without changing the id. Needs the
  placement-capable overlay (libp2p-kad) **or** the directory-over-cells (§6 rec).

> Recommendation leans **A for v1** (no global store needed, composes with the directory-first
> substrate), with **B as the migration story** once session rebinding/cross-cell move is required.

### 5.2 Client / ingress caching — the ARP-cache principle

```
resolve(session_id):
  if cache.hit(session_id) and not cache.stale: dial cell directly      # hot path, 0 hops
  else:
    home = decode_cell(session_id)        # Option A: 0 hops
         | dir_lookup(H(session_id))       # Option B: 1 overlay hop
    cell -> intra-cell HRW -> node         # #322 leaf tier
    cache.put(session_id -> cell)          # ARP-cache fill
  on dial-fail / stall:
    cache.invalidate(session_id); negative-cache(short TTL); recompute   # reactive
```

- **Hot path = direct dial** (cache hit), no overlay involvement → O(1) and overlay load stays flat
  as sessions/sec grow.
- **Cold/failed = one overlay hop** to the home cell, then intra-cell HRW to the node.
- **Failure = reactive invalidate + recompute** (the §4 verdict in action; no SWIM).

This is exactly the ARP-cache discipline: cached→direct; cold→one resolution; failed→invalidate &
re-resolve. It keeps the overlay off the request hot path entirely under steady state.

### 5.3 Cross-cell model replication as the availability unit

- **The replicated, addressable thing is the MODEL (and its session/delta state), not the node.** A
  model is **replicated across cells**; each replica cell runs #322 internally over its own nodes.
- **Availability** at the global tier = "which cells host a replica of model M?" → a directory entry
  `model_id → {cell replicas}` (PQ-signed, §3.3), resolved like §5.2. Client picks a replica
  (locality/load), then drops into that cell's leaf router.
- **Durability** of session state = **git ref-CAS** in the home cell, reconciled by the **#326 delta
  reducer**. Cross-cell replication of *deltas* is a git-fetch/CAS between cell replicas — still the
  same single hard-consistency primitive, just exercised across cells.

### 5.4 git-CAS remains the only global hard-consistency point — confirmed

Nothing in the global tier introduces a second consensus/consistency requirement: the overlay is
hints (eventually-consistent, untrusted), membership is hints (reactive), affinity is hints (perf).
**The only place two writers must agree is a durable delta write, and that is `git ref-CAS`** (loser
retries; #326 reducer merges). **No RAFT, no global consensus, no SWIM-as-truth.** This is the
property that makes 1M-node scale tractable.

---

## 6. RECOMMENDATION

**Substrate:**
- **v1 — Directory-first, NOT a general placement-DHT.** Build the global tier as a **PQ-signed
  cell-directory** (`model_id → {cell replicas}`, optional `session_id → home_cell`) layered on what
  we already have: **iroh dial + the `Resolver` trait** (`resolver.rs`, pluggable
  `EndpointRegistry → DiscoveryService`) + the **federation admission gate**. Use **HRW/Kademlia as
  an *algorithm* over the bounded set of CELLS**, not a DHT over all nodes. Keep **pkarr/iroh as the
  NodeId→address resolution layer it already is** (#282) — do not try to make it do placement (it
  structurally cannot, §2.1).
- **v2 (only if needed) — reuse libp2p-kad for a true distributed placement store** when cell counts
  or the absence of a PDS-of-record demand `H(key)→owner` at global scale. It is the better-proven
  Kademlia, **already compiled in-tree** (gittorrent), and placement-capable. Pay the
  `PeerId↔node_id` bridge cost with a **COSE-signed binding record** (§2.3) so authz/PQ posture is
  preserved. **Do not enable gossipsub** unless a concrete epidemic-pubsub need appears (it is not
  even in our feature set today).

**Membership approach:**
- **No SWIM at the global tier.** k-bucket maintenance + reactive failure-detection (dial-fail +
  token-stream stall watchdog) + short negative-cache. Optional **intra-cell load-gossip in #322**
  for proactive capacity rebalancing (the only justified proactive-membership use).

**PQ handling:**
- **Classical DHT/overlay = untrusted hints; the PQ work-plane validates everything.** PQ-sign only
  the **directory-layer records we control** with the existing **hybrid COSE** (WNS: enforce-for-
  anchored, classical floor otherwise, never blanket fail-closed). Ride **rustls X25519MLKEM768**
  (present, 0.23.40) on the QUIC control plane. **PQ-signed underlying-DHT records are NOT
  required** — a forged record = misdirection→DoS, not compromise/HNDL.

---

## 7. HUMAN-DECISION MENU (open choices)

1. **Global substrate:** (a) **directory-first over iroh+Resolver+admission, HRW-over-cells [REC for
   v1]**; (b) adopt **libp2p-kad** as the placement-DHT now (pay the PeerId bridge up front); (c)
   wait — leaf tier (#322) only, defer global tier until cell count demands it.
2. **Session→home-cell binding:** (a) **encode cell in `session_id` [REC v1, 0 lookups]**; (b)
   **sharded session-directory** (enables migration, 1 hop, needs placement store or directory).
3. **gossipsub:** (a) **do not enable [REC — not in feature set, no concrete need]**; (b) enable if
   an epidemic-pubsub need (e.g. cell-wide capacity broadcast) is confirmed.
4. **SWIM:** (a) **none at global tier [REC]**; (b) optional **intra-cell load-gossip in #322** for
   proactive rebalancing — decide when #322 starts; (c) full SWIM (rejected — no hard-consistency to
   protect, members are cells not nodes).
5. **PQ posture on directory records:** (a) **hybrid COSE, WNS per-identity [REC]**; (b)
   classical-only directory records (relies entirely on the work-plane; accepts forged-record DoS);
   (c) mandatory-PQ directory (rejected — violates "never blanket fail-closed").
6. **Inter-cell control transport:** (a) **QUIC (rustls hybrid KEM available today) [REC]**; (b)
   libp2p Noise (classical X25519, no PQ — only if we adopt libp2p-kad and accept classical
   transport for hints).

---

## 8. What becomes its own epic / tickets (human-gated — capnp/RPC surface)

The following are **net-new wire/RPC surface** and are **explicitly human-gated** (capnp is the only
gate in this program). This spike does **not** define schemas or file tickets — it scopes the work:

- **Cell-directory record schema (capnp).** `model_id → {cell replicas}`, optional `session_id →
  home_cell`, each **PQ-signed (hybrid COSE)**. Human-gated capnp decision. *(Epic candidate:
  "global cell directory".)*
- **`peer_id ↔ node_id ↔ did` binding record (capnp)** — required *only if* v2/libp2p-kad is
  adopted; COSE-signed. Human-gated capnp decision.
- **`Resolver`/`DiscoveryService` extension** to resolve **cell-level** reach (the existing trait is
  node/endpoint-level; the directory adds a cell tier above it). Mostly pure-Rust over the existing
  trait, but the directory *record* it returns is the capnp item above.
- **Inter-cell delta replication** (git-fetch/CAS between cell replicas) — reuses #326 reducer +
  git-CAS; coordination/triggering surface is a separate epic, **not** a new consistency primitive.

These compose with #320 (reach via Resolver), #322 (leaf router), #326 (delta reducer), and
#319/#328 (mesh authz) without changing any of their settled decisions.

---

## Appendix — grounding (file:line and versions)

**In-tree code (this worktree):**
- iroh substrate / node_id (Ed25519): `crates/hyprstream-rpc/src/transport/iroh_substrate.rs:41,85`
  (`presets::N0` = n0 DNS + pkarr discovery + relay).
- TransportConfig variants (Inproc/Ipc/Quic/Iroh): `crates/hyprstream-rpc/src/transport/mod.rs:152,157,178,199–206`.
- pkarr = deferred (#282), resolution-only; "empty = direct/pkarr (#282)" `transport/mod.rs:200`;
  dial seam `dial.rs`, `moq_stream.rs` ("pkarr dial-by-node_id alone deferred to #282"). **No
  standalone `pkarr`/`mainline` crate in `Cargo.lock`** (pulled via iroh internals).
- libp2p (gittorrent only): `crates/gittorrent/Cargo.toml:72` =
  `libp2p = { version = "0.56", features = ["kad","mdns","noise","tcp","tokio","yamux","identify","request-response","macros"] }`
  (**no gossipsub**); behaviour `crates/gittorrent/src/dht/behaviour.rs:14–62`; `PeerId::from(keypair.public())` `:24`.
  Only crate depending on libp2p: `crates/gittorrent/Cargo.toml`.
- #320 reach: `crates/hyprstream/src/services/model.rs:68–72` (`transport: TransportConfig`,
  resolved via Resolver; pkarr discovery for cross-host); `IrohReach`/`Destination` capnp
  `crates/hyprstream-rpc/schema/streaming.capnp:154–157,192–201`; Resolver trait `resolver.rs`.
- Identity bridging: 4 types — iroh node_id (`iroh_substrate.rs:85`), did:web (`did_web.rs:53–96`),
  did:key (`did_web.rs:290–303`), libp2p PeerId (`behaviour.rs:24`). Admission gate
  `crates/hyprstream-rpc/src/admission.rs:1–200` (two-stage fail-closed).
- Hybrid COSE (EdDSA + ML-DSA-65 composite, fail-closed when anchored): `crypto/cose_sign.rs:47–54,113,129,429–548`;
  WNS posture (enforce-for-anchored / classical floor / never blanket fail-closed)
  `streaming.rs:1024–1089`.
- Mesh authz: per-host subject `node_identity.rs:28–36` (`mesh_host_subject`), enrolled
  Ed25519→subject else `anonymous` (`register_peer`); ML-DSA derive `derive_mesh_mldsa_key`
  `node_identity.rs:56–81`; trust store `envelope.rs` `KeyedPqTrustStore`.
- Stream DH + AEAD: Ristretto255 `crypto/key_exchange.rs:6–14,150–201,348–619`; AES-256-GCM
  `crypto/event_crypto.rs`, `enc_key` (#321) `key_exchange.rs:54–59`.

**Versions (`Cargo.lock`):** `iroh = 1.0.0-rc.0`, `libp2p = 0.56.0`, `libp2p-kad = 0.48.0`,
`libp2p-noise = 0.46.1`, `quinn = 0.11.9`, **`rustls = 0.23.40`** (X25519MLKEM768 default since
0.23.27), `ml-dsa = 0.1.0-rc.11` (**dep of `hyprstream-rpc`**), **`ml-kem`** (**also dep of
`hyprstream-rpc`**), `aes-gcm = 0.10.3`, `coset = 0.4.2`, `curve25519-dalek = 4.1.3 / 5.0.0-pre.6`,
`ed25519-dalek = 2.2.0 / 3.0.0-pre.7`. No `pkarr`/`mainline`/`libp2p-gossipsub` in lock.

**External sources (DHT properties + PQ roadmaps):**
- pkarr / Mainline / BEP44 (resolution-only, key = `H(pubkey‖salt)`, Ed25519-only): pkarr docs
  (pubky.github.io/pkarr), bittorrent.org BEP44, iroh "Iroh global node discovery" /
  "Dial by NodeID" blogs (NodeId→addr, self-signed).
- libp2p PQ status (no near-term ML-KEM in Noise; #2168 closed "not planned"): rust-libp2p issue
  #2168, libp2p-noise crate docs. **Needs validation** against latest libp2p-noise CHANGELOG.
- rustls/quinn hybrid KEM (X25519MLKEM768 default since 0.23.27): crates.io `rustls-post-quantum`,
  rustls 0.23.27/0.23.40 notes, IETF `draft-ietf-tls-ecdhe-mlkem-05`.
- Kademlia k-bucket (lazy liveness) vs SWIM (proactive): Kademlia spec (xlattice), SWIM
  (Das/Gupta/Motivala 2002; en.wikipedia.org/wiki/SWIM_Protocol), Lifeguard (arXiv 1707.00788).

*Items marked "needs validation": pkarr-internal placement impossibility (inferred from BEP44, very
high confidence); latest libp2p-noise PQ status (2021 issue + current docs, treat as classical).*
