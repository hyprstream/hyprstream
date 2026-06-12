# M2 — Streaming plane: ZMQ event/stream bus → moq-net

Part of epic **#131**. Lands on the `ewindisch/epic-moq` integration branch via
per-ticket worktrees + PRs (same git-flow as M1). Supersedes the per-ticket
framing where it conflicts with ground truth below.

> **Status when written (2026-06-12):** M1 (RPC/IPC cutover) is merged onto
> `epic-moq`. M2a's *wire layer* already landed in **PR #147**
> (`OriginShared`, `IrohMoqProtocolHandler`, `MoqStreamOrigin`,
> `MoqStreamPublisher`, `verify_moq_frame`). What remains is the **service
> refactor** (flip the primary path off ZMQ), the **producer rewiring**, the
> **relay duplex**, and the **security hardening**.

---

## Ground truth (current code in the `epic-moq` tree)

| Area | State | Key locations |
|---|---|---|
| moq wire layer (#147) | **landed** | `transport/iroh_moq.rs` (`OriginShared`, `IrohMoqProtocolHandler`), `moq_stream.rs` (`MoqStreamOrigin`, `MoqStreamPublisher`, `verify_moq_frame`) |
| Remote SUBSCRIBE / announce duplex | **stubbed** | `iroh_moq.rs:140` — `Server::with_subscribe(None)` |
| StreamService primary transport | **still ZMQ** | `service/streaming.rs` — ZMQ PULL (in) → XPUB (out); moq-lite is a *parallel overlay* via `with_moq_origin()` |
| §7.5 chained-HMAC frame | **byte-identical** ZMQ↔moq | `streaming.rs` builder + `moq_stream.rs:223-292` publisher; `StreamVerifier` in `stream_consumer.rs:57` |
| Custom late-join rejoin | present, to delete | `streaming.rs:~1553-1620` (pre-registration + cancel token + JWT expiry) |
| Consumer Group ordering / terminal-block | **gap (#163)** | `verify_moq_frame` (`moq_stream.rs:301`) discards moq Group sequence; `StreamVerifier` trusts its own `prev_mac`, never compares `block.prev_mac`; no terminal-payload requirement |
| Event bus | **ZMQ XSUB/XPUB** | `ProxyService` (`hyprstream-service/.../spawner/service.rs:38`), `inproc://hyprstream/events/{pub,sub,ctrl}` (`hyprstream-workers/src/events/endpoints.rs`), `EventPublisher`/`EventSubscriber`, `SecureEvent{Publisher,Subscriber}` (Ristretto255 DH + AES-GCM) |
| Bootstrap signals (load-bearing) | ZMQ event publish | `main.rs:~2090` `system.{svc}.ready`, `main.rs:~2120` `system.{svc}.stopping` |
| Token publishers (~15–20 sites) | ZMQ StreamChannel | `inference.rs` (×~12: 981,1005,1061,1321,1336,1351,1365,1394,1410…), `model.rs` notif (497,1378,1394), `secure_publisher.rs` (171,324) |
| Stream key exchange | **Ristretto255 only** (#153) | `crypto/key_exchange.rs:136 derive_stream_keys`, `ristretto_dh` (481/535); HNDL-vulnerable on relay'd path |
| Callback control plane (zb2) | ZMQ ROUTER/DEALER | `services/callback.rs` (`CallbackRouter`), `model.rs:138/262`; worker fd path placeholder (`run_fd_streaming_task`), 9P/VFS Mount greenfield |

---

## Sequencing & dependencies

```
Phase A (substrate + integrity) ── critical path, blocks everything
  #134-M2a   StreamService holds a moq Origin; flip primary path; port §7.5; delete rejoin
  #163       consumer Group-ordering + terminal-block guards  ← lands in the same consumer
        │
        ▼
Phase B (rewire producers) ── two parallel tracks off post-A epic-moq
  #167       Event bus XSUB/XPUB → moq Live preset; rewire ALL bus users + bootstrap signals
  #169       Token streaming Job/Log presets; at-least-once + dedup + offset resume; 21 call-sites
        │
        ▼
Phase C (relay + PQ + review)
  #168       Relay duplex + peering (fill the with_subscribe(None) slot); moq-relay interchange
  #153       ML-KEM-768 hybrid on the relay'd DH-topic path
  #174       Adversarial security review of relay-side retention
        │
        ▼
Phase D (remaining semantics)
  #135       PUSH/PULL worker-queue lease (app-level claim over RPC plane)
  #170       zb2 duplex lossless Pipe (Kata console I/O, 9P-over-Pipe, VFS Mount) + migrate callback DEALER plane
```

**Why this order:**
- #163 is HIGH-sec and defines what a *correct consumer* is; it must land with the
  substrate (#134-M2a) so every later consumer inherits the ordering/terminal guards
  rather than retrofitting them. #169 explicitly "Depends on #134 substrate, #163."
- #167 and #169 touch disjoint call-sites (event bus vs token publishers) → parallelizable
  once the substrate exists. Each branches off the **merged** Phase-A tip.
- #168's duplex is only safe to expose after consumers enforce integrity (#163) — opening
  remote subscribe before that widens the replay/truncation surface.
- #153 + #174 harden the *relay'd* path, which only becomes real in #168.
- #135/#170 are semantics on top of the substrate; #170 also carries the last ZMQ control
  plane (callback DEALER), so it gates M5 teardown of `zmq`/`tmq`.

---

## Git-flow (per ticket)

Each ticket gets its own worktree branched off the **current** `epic-moq` tip and a
PR **targeting `ewindisch/epic-moq`**, merged via merge-commit (no rebase — keeps the
stack intact, per the M1 lesson). Dependent tickets branch only **after** their
prerequisite PR merges, so we never restack. Parallel tickets (#167 / #169) both branch
off the post-A tip; the second to merge resolves any overlap.

```
git worktree add .worktrees/ewindisch/moq-<n> -b ewindisch/moq-<n> <epic-moq-tip>
# build + cargo check/clippy/test with the standard env
# (OPENSSL_NO_VENDOR=1 LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 + LD_LIBRARY_PATH)
# commit; user pushes (SK key); open PR --base ewindisch/epic-moq
```

Every ticket runs the mandated cycle: **code review + adversarial security review**
(subagents) before merge; unresolved out-of-scope findings → new tickets.

---

## Phase A — `#134-M2a` substrate + `#163` integrity (one PR, or two stacked)

**Goal:** StreamService becomes a moq-lite relay node; the moq path is *primary*;
consumers enforce ordering + completeness.

1. **StreamService owns `MoqStreamOrigin`.** Replace the ZMQ PULL→XPUB proxy
   (`service/streaming.rs`) with: in-process publishers append to the Origin via the
   Rust API (`authorize_signer` gate); external subscribers consume on the `moql` ALPN
   through `IrohMoqProtocolHandler`. Keep the blind-forwarder trust model (service does
   not see plaintext; HMAC is end-to-end).
2. **Port §7.5 1:1 to Track/Group/Frame.** One `StreamBlock` = one moq Group (already
   the shape in `moq_stream.rs:286`); confirm Track-name scheme `{tenant}/{service}/{topic}/{instance}`.
3. **Delete the custom late-join rejoin** (`streaming.rs:~1553-1620`) and collapse the
   `notification`-side pre-registration — moq Groups give native late-join (subscriber
   starts at latest Group, catches up within retention).
4. **#163 consumer guards** (lands here so all consumers inherit them):
   - feed the moq Group **sequence** into `verify_moq_frame` and assert
     `group.sequence == expected_next` (gap = fatal); optionally bind sequence into the MAC input.
   - require a terminal `Complete`/`Error` payload before treating a stream as complete;
     **EOF-without-terminal = reject** (truncation defense).
5. **Retention window** replaces the rejoin buffer; expose a `StreamPolicy` knob
   (preset substrate referenced by #169). Caps tie into #162/#174 thinking.

**Exit:** tokenstream end-to-end (in-proc publisher → Origin → external `moql` subscriber);
native late-join; `authorize_signer` enforced; consumer rejects reordered/truncated streams.
Keep ZMQ StreamService temporarily behind a flag for differential testing, removed in M5.

**Verify:** e2e test (publisher → Origin → subscriber) with byte-identical frames to the
ZMQ path; adversarial test feeding out-of-order / dropped-terminal Groups must reject.

---

## Phase B — producers (parallel)

### `#167` Event bus → moq `Live` preset
- Stand up the event bus as a moq `Live` (fan-out, unbounded, at-most-once) Track,
  replacing `ProxyService` XSUB/XPUB and `inproc://hyprstream/events/`.
- **Rewire ALL bus users** (the review flagged the original list as under-enumerated):
  compositor, TUI, WorkerService, WorkflowService + gh_adapter, `SecureEvent*` encrypted
  variants, CLI SUB consumers (shell/tui/git handlers), and — carefully — the
  **`system.{svc}.ready/.stopping` bootstrap signals** (`main.rs:~2090/2120`, load-bearing
  for staged startup; migrate these first behind a compatibility shim, verify ordering).
- Delete the ZMQ ProxyService event path once all users are moved.
- **Security:** `SecureEvent*` keeps its DH+AEAD; ensure it composes with the moq Track
  (encryption is payload-level, transport carries opaque frames).

### `#169` Token streaming — at-least-once + dedup + offset resume
- Move the ~15–20 publisher call-sites (InferenceService ×~12, Registry clone, Metrics,
  Notification, Workers attach, TUI) onto the moq publisher via **StreamPolicy presets**
  (mostly `Job`; mobile clients = `Log`).
- Delivery: at-least-once + **client dedup**; offset = moq Group sequence; reconnect
  resumes from last-acked offset within the relay retention window.
- Preserve the HWM `EAGAIN` non-blocking-publish contract as an explicit backpressure mode.
- Depends on Phase A (substrate + #163 dedup/ordering).

---

## Phase C — relay duplex + PQ + review

### `#168` Relay duplex + peering
- Fill the `with_subscribe(None)` slot (`iroh_moq.rs:140`): accept remote announce/subscribe;
  add optional upstream `moq_net::Client` peering → two StreamServices peer as a relay mesh,
  wire-interchangeable with moq-relay + Cloudflare. Consolidates old #139/#140/#142.
- CDN-portability invariants: moq-lite primitives only, standard ALPN, opaque hierarchical
  track names, opaque frame payloads, respect upstream Group cache consts (#141).

### `#153` ML-KEM-768 hybrid on the relay'd path
- Add ML-KEM-768 hybrid to `StreamInfo` key derivation (`derive_stream_keys`), mirroring the
  RPC envelope. **Scope to the relay'd DH-topic path only** — direct peer↔peer streams inherit
  PQ from the transport; the DH-topic + payload encryption matters only when egressing through
  an untrusted relay/CDN.

### `#174` Relay-retention security review (adversarial subagent)
- The retention window is a new attack surface (signed frames on an untrusted relay). Review:
  retention DoS caps (tie to #162), retention floor vs worst-case subscriber gap, MAC-indexed
  resume integrity, interaction with #153 (ML-KEM) and #163 (truncation/ordering).

---

## Phase D — remaining semantics

### `#135` PUSH/PULL worker-queue lease (blocked by #134)
- moq fans out, doesn't round-robin. Workers subscribe `work/{queue}` (Group = job); race to
  claim via side-channel `claim/{job-id}` over the RPC plane (`hyprstream-rpc/1`); first wins.
- Exit: ResponseStream backplane (`streaming.rs:~1754-1790`) ports with no loss under a
  kill-worker-mid-job chaos test. (Fallback: raw bidi PUSH/PULL if moq ergonomics disappoint.)

### `#170` zb2 duplex lossless `Pipe` + migrate callback DEALER plane
- New duplex, lossless `Pipe` stream kind. Implement Kata vsock/serial **console I/O** (replace
  `run_fd_streaming_task` placeholder), 9P-over-Pipe (non-WASM transport), wire hyprstream-vfs
  Mount/proxy, TUI stdin/stdout.
- **Migrate the model↔worker callback DEALER plane** (`callback.rs` / `CallbackRouter`,
  `inference.rs:256`) here — it's a control plane, not on RequestLoop, and is the **last** ZMQ
  user gating M5's `zmq`/`tmq` removal.

---

## Cross-cutting / coordination

- **#162 / #165** (connection caps) are M1-sec but their retention/DoS thinking feeds #174.
- **#141** (moq-net cache consts upstream PR) should land before #168 relay'd caching is tuned.
- After Phase D, **M5 (#138/#171/#172/#173)** can delete `zmq`/`tmq` and add the
  `cargo-deny` no-zmq CI gate — nothing should publish/subscribe on ZMQ by then.

## Definition of done for M2
Streaming + events ride moq-net end-to-end (in-proc publish → Origin → local & remote
subscribers); late-join native; consumers enforce ordering + terminal completeness; relay'd
path is ML-KEM-hybrid and retention-reviewed; worker-queue + container-pipe semantics ported;
the only remaining ZMQ is whatever M5 explicitly deletes.
