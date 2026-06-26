# Spike — `UdsSession`: one `web_transport_trait::Session` over UnixStream

> **STATUS (2026-06-11): BUILT** as `moq-151g`
> (`crates/hyprstream-rpc/src/transport/uds_session.rs`, yamux-backed). 9 tests
> green (incl. real RPC plane via `SessionRpcTransport`/`serve_rpc_connection`
> and full moq publish/subscribe over UDS); release clippy clean. Code +
> security review done — fixes applied (concurrent-stream cap, tag-read timeout,
> fail-closed plane validation, observability). One follow-up filed: **#207**
> (socket perms + `SO_PEERCRED`, daemon/#136 scope). Feeds #136's `dial()` ipc
> arm + daemon `UnixListener` provisioning.
>
> Review caught a real defect: `SendStream::closed()` must *detect* closure, not
> *cause* it — an early impl called `close()` inside `closed()`, so moq's
> `select! { biased; _ = stream.closed() => cancel }` in `serve_group` aborted
> every group right after its header, truncating all frames. Fixed.

2026-06-11. Decision input for the ipc-transport re-platform (ZMQ-over-`ipc://` →
SignedEnvelope-over-UDS) and #136's `dial()` ipc arm. Question answered:
**can the same UDS transport carry both the RPC plane and the moq streaming
plane?** Answer: **yes**, via one `web_transport_trait::Session` impl.

## Why one impl covers both planes

`moq-net 0.1.8` is fully generic over `web_transport_trait::Session`:

- `Server::accept<S: web_transport_trait::Session>(&self, session: S)` — `server.rs:54`
- `Client::connect<S: web_transport_trait::Session>(&self, session: S)` — `client.rs:54`
- `moq_net::Session` type-erases via blanket `impl<S: web_transport_trait::Session>
  SessionInner for S` → `Arc<dyn SessionInner>` — `session.rs:144`
- moq-net's own tests ship a hand-rolled `FakeSession` (non-quinn/non-iroh) and run
  full SETUP negotiation over it — the genericity is real, not incidental.

So one `UdsSession: web_transport_trait::Session` serves:
- **RPC plane** → `SessionRpcTransport` (`open_bi` → SignedEnvelope req/resp)
- **Streaming plane** → `moq_net::Client::connect(uds_session)` / `Server::accept`

exactly as quinn/iroh do today. Everything layered above the trait is transport-agnostic.

## What moq-lite actually requires of the Session (the contract)

| Capability | Required by moq-lite? | Evidence | UDS impl |
|---|---|---|---|
| `open_bi` / `accept_bi` | **Yes** — one SETUP control stream | `coding/stream.rs:25` `Stream::open`→`open_bi`; client negotiates Lite version over it (`client.rs:180+`) | mux bidi stream |
| `open_uni` / `accept_uni` | **Yes** — one per Group | `lite/publisher.rs:484`, `lite/subscriber.rs:93` | mux uni stream |
| `send_datagram` / `recv_datagram` / `max_datagram_size` | **No** — zero datagram refs in `lite/`; datagrams are IETF-draft-only (`ietf/adapter.rs`) | trait still requires the methods | stub: `max_datagram_size()→0`, `send/recv → Err(Closed)` |
| ALPN `protocol()` | optional | `None` → manual version negotiation arm `Some(ALPN_LITE) | None` (`client.rs:174`) | return `None`; moq negotiates Lite over the control stream |

Net: genuine concurrent stream multiplexing (1 bidi + N uni), **no datagrams, no ALPN**.

## Multiplexer: yamux (already in-tree)

A `UnixStream` is one ordered byte-stream; QUIC/iroh provide stream multiplexing
natively, UDS does not. `UdsSession` carries a multiplexer.

- **yamux 0.13.7 / 0.12.1 already in `Cargo.lock`** (via libp2p/iroh) — adding
  `yamux` to `hyprstream-rpc` pulls in nothing new.
- `yamux::Stream: futures::io::AsyncRead + AsyncWrite` → wrap as the trait's
  `SendStream`/`RecvStream`.
- yamux gives **flow control + HOL-correct concurrent streams** for free — the
  exact thing a raw socket lacks and we'd otherwise hand-roll incorrectly.
- Rejected alternatives: hand-rolled framing mux (~350 LOC, reinvents flow
  control); `SOCK_SEQPACKET`/fd-passing (no independent flow-controlled streams).

### The one real integration cost — actor wrapper

`yamux::Connection<T>` is a single-owner, poll-driven driver: **not `Clone`**, must
be polled by exactly one task (`poll_next_inbound` also drives outbound). But
`web_transport_trait::Session: Clone` and is called concurrently from many tasks.

Standard fix (what `libp2p-yamux` does): one **driver task** owns the `Connection`;
`open_bi`/`open_uni`/`accept_bi`/`accept_uni` are channel RPCs to it.

```
UdsSession (Clone)            driver task (owns yamux::Connection<Compat<UnixStream>>)
  open_bi  ─ mpsc ─────────▶  poll_new_outbound → tag BIDI → oneshot back (send,recv)
  open_uni ─ mpsc ─────────▶  poll_new_outbound → tag UNI  → oneshot back (send)
  accept_* ◀── mpsc queues ── poll_next_inbound → read 1-byte tag → route uni|bidi queue
```

Sizing: ~150 LOC actor + ~120 LOC stream-type wrappers (`SendStream::{write,
write_buf, finish, reset, set_priority}`, `RecvStream::{read, read_buf, stop}`).

### Two wire conventions to lock in (trivial)

1. **uni-vs-bidi tag** — yamux substreams are all bidi; prefix each new substream
   with a 1-byte `UNI`/`BIDI` marker. Acceptor reads it first and routes to the
   matching `accept_*` queue. Uni opener simply never reads its half.
2. **plane select** — 1-byte handshake at socket-open (`RPC` vs `MOQ`) in place of
   ALPN, since UDS has none. Server's `UnixListener` accept loop dispatches the
   connection to the RPC dispatcher or `moq_net::Server::accept` on that byte.

### Adapter glue

- `tokio::net::UnixStream` is `tokio::io::AsyncRead/Write`; yamux wants
  `futures::io::AsyncRead/Write` → bridge with
  `tokio_util::compat::TokioAsyncReadCompatExt` (`tokio-util` already a dep).
- Error-code mapping is lossy and that's fine (local socket): `finish`→close write
  half; `reset`/`stop`→drop/close; session `close(code,reason)`→`poll_close`.

## Build outline (its own increment, feeds #136 ipc arm)

1. `transport/uds_session.rs` — `UdsSession` + driver actor + `UdsSendStream`/
   `UdsRecvStream`; `impl web_transport_trait::Session`.
2. Client `connect(path) -> UdsSession` (yamux `Mode::Client`); server
   `UnixListener` accept loop → 1-byte plane select → `Mode::Server` `UdsSession`.
3. RPC plane: reuse `SessionRpcTransport` over `UdsSession` (no new RPC code).
4. Streaming plane: `moq_net::{Client,Server}` over the same `UdsSession`.
5. `dial()` ipc arm constructs `UdsSession`; daemon binds the `UnixListener`.
6. Tests: loopback `UdsSession` pair — bidi echo, N concurrent uni streams,
   SETUP negotiation, then a full moq publish/subscribe over UDS.

## Open question deferred → filed as #207

- **Peer auth on UDS** — same-host identity via `SO_PEERCRED` (uid/pid) plus
  socket-file permissions is defense-in-depth; the SignedEnvelope app-layer
  identity already authenticates the caller regardless of transport. Owned by
  the daemon listener bootstrap (where the bind path + mode are chosen), not the
  Session primitive. Tracked in **#207** (coordinate with #136).

## Review outcomes applied in moq-151g

- **Tag-read timeout** (`TAG_READ_TIMEOUT`) on `classify_inbound` — a substream
  that never sends its UNI/BIDI tag can't pin a task + yamux slot forever
  (pre-dispatch slowloris that bypassed rpc_session's `REQUEST_READ_TIMEOUT`).
- **Concurrent-substream cap** (`MAX_CONCURRENT_STREAMS` via yamux
  `set_max_num_streams`) — bounds the inbound classification fan-out.
- **Fail-closed plane validation** in `accept_uds` — unknown selector → error.
- **Observability** — warn/debug logs on unknown tag, untagged-timeout, FIN-flush
  failure.
- Rejected (by-design, evidenced): `closed()` pends rather than reporting
  spurious closure (moq relies on `read()==None` for EOF); the
  `tokio::sync::Mutex` around the accept receiver is held across `await` by
  design (single-consumer mpsc). The detached `finish()` close task is correct —
  writes are queued into yamux's send buffer before `finish`, proven by the moq
  round-trip delivering the full frame.
