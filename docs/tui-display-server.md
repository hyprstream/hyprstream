# TUI Display Server

Three-layer architecture with Cap'n Proto RPC + 9P-style virtual filesystem.

General infrastructure for all TUI applications (dashboards, terminal multiplexing, MCP-controllable display). X11-layered model (Xlib → Cairo → GTK).

## Three-Path Architecture

Three rendering/testing paths coexist. The same waxterm `TerminalApp` implementations work across all paths — the backend is selected at launch time, not compile time.

### Path A: Production (multiplexer — default for CLI users)

```
hyprstream service start tui        # start TUI service (background)
hyprstream service start tui -f     # start TUI service (foreground)
hyprstream tui attach               # attach to default session
hyprstream tui attach --session 3   # attach to specific session
hyprstream tui new                  # create new session and attach
hyprstream tui list                 # list sessions
```

Apps render into panes via `TuiPaneBackend` (direct) or `StructuredBackend` (WASI). The multiplexer provides:
- Session persistence (detach/reattach)
- Multi-window/pane layout (tmux-style splits)
- Remote access via ZMQ streaming
- MCP-controllable display surfaces
- Multiple concurrent viewers per session

### Path B: App testing (direct WASI — Playwright)

waxterm apps compile to WASI and run directly in the browser via Wanix + xterm.js. `TerminalHost` (`packages/waxterm/src/index.ts`) boots Wanix, creates a WASI task, and connects stdin/stdout. It supports two modes:

- **Bundle mode** (minerva): `new Wanix({ bundle })`, streaming API (`openReadable`/`openWritable`/`appendFile`), configurable `postMessageType`.
- **Direct mode** (wizard): `new Wanix({})`, kernel readiness poll, fd-based I/O (`open`/`read`/`write`).
- **Independent options** (either mode): `fitAddon` for responsive FitAddon sizing (vs CSS zoom), `resizeIpc` for `ESC ] Z` resize escape prefix to stdin on window resize.

Both modes share: RAF-batched stdout drain, ~60Hz input flush via `setInterval`, configurable mouse regions with negative-index support, `postMessage` IPC for external input. `minerva.html` and `minerva-test.html` both use `TerminalHost` (no more inline JS).

Playwright drives the browser and inspects the terminal via shared helpers in `packages/waxterm/src/testing/`:
- `helpers.ts`: `waitForTerminalText`, `getTerminalText`, `sendKey`, `sendKeys`, `typeText`, `selectOption`
- `quality.ts`: `hookXtermWrite`, `stats`, `percentile`, `WriteLog`, `QualityStats`
- `multiplexer.ts`: `connectViewer`, `sendViewerInput`, `destroyViewer`, `waitForFirstFrame`, `fetchCertHash`

No display server involved. No ZMQ, no Cap'n Proto, no 9P. The WASI binary writes ANSI to stdout, xterm.js renders it. This is the path used by the existing test suites in `web-minerva-game/tests/wizard.spec.ts`.

**Tests**: Application logic, widget behavior, phase transitions, user flows.

### Path C: Multiplexer testing (WebTransport — Playwright)

The TUI service runs as a native subprocess. The browser connects via the `WebTransport` API to the TUI service's QUIC endpoint (provided by the existing ZMTP-over-QUIC transport in `transport/zmtp_quic.rs`). The client opens a bidirectional QUIC stream (server accepts via `accept_bi()`): frames flow server→client as length-prefixed Cap'n Proto `TuiFrame` messages, input flows client→server as raw key bytes. TLS 1.3 is built-in. Chrome 97+ and Firefox 114+ support WebTransport (Chromium-only for Playwright tests — WebKit does not yet ship WebTransport).

A thin browser client (`TuiViewerClient`) connects via WebTransport and deserializes `TuiFrame` messages using generated TypeScript types (Phase 3). For `AnsiStream` mode, the ANSI payload is written to xterm.js. Playwright drives the browser and inspects the terminal via the same DOM polling helpers used in Path B.

**WebTransport certificate**: The W3C spec requires `serverCertificateHashes` certificates to have ≤14 days validity and use ECDSA P-256 (not RSA). TUI service generates a separate short-lived self-signed cert via a new `generate_wt_cert(validity_days)` function (not the 365-day `SHARED_TLS` cert from `server/tls.rs`). The cert hash is served at `/.well-known/tui-cert-hash` — requires extending `handle_http_request()` in `zmtp_quic.rs` (currently only serves `/health` and `/.well-known/oauth-protected-resource`).

```
Playwright                  Browser                    Hyprstream (subprocess)
    │                          │                              │
    │  beforeAll: spawn        │                              │
    │  hyprstream service      │                              │
    │  start tui -f            │                              │
    │                          │                              │
    │  goto('/tui?session=0')  │  WebTransport connect (QUIC) │
    │                          │─────────────────────────────>│
    │                          │  client opens bidi stream     │
    │                          │─────────────────────────────>│  accept_bi()
    │                          │                              │
    │                          │  TuiFrame (Cap'n Proto)      │
    │                          │<─────────────────────────────│  TuiService frame loop
    │                          │                              │  encode TuiFrame
    │                          │  deserialize → xterm.js      │
    │                          │                              │
    │  sendKey(page, 'Enter')  │  raw key bytes on bidi       │
    │─────────────────────────>│─────────────────────────────>│  → stdin of pane
    │                          │                              │
    │  waitForTerminalText()   │  polls .xterm-rows DOM       │
    │─────────────────────────>│                              │
    │                          │                              │
    │  afterAll: kill process  │                              │
```

**Tests**: Multiplexer startup/shutdown, session create/attach/detach, multi-window layout, pane splits, viewer connect/disconnect, slow viewer eviction, frame delivery, input routing.

**Reuses**: Same xterm.js DOM polling helpers (`getTerminalText`, `waitForTerminalText`, `sendKey`). Same Playwright assertions. The browser sees xterm.js rows either way — the source of data is transparent. Playwright tests via DOM polling (no protocol-level WebTransport API in Playwright — testing is through UI behavior). Tests run Chromium only (WebKit lacks WebTransport support).

### Path comparison

| Concern | A: Production | B: App testing | C: Multiplexer testing |
|---------|--------------|----------------|----------------------|
| Runtime | Native (tokio, zmq) | WASI (Wanix, browser) | Native subprocess + browser |
| Data flow | ZMQ streaming | ANSI → Wanix pipe → `openReadable` → ReadableStream → RAF-batched `term.write()` (bundle); ANSI → fd read → RAF-batched `term.write()` (direct) | QUIC bidi stream (Cap'n Proto TuiFrame) → xterm.js |
| Backend | `TuiPaneBackend` / `StructuredBackend` | `AnsiBackend` (waxterm) | `TuiPaneBackend` (server-side) |
| Input | ZMQ PUSH/PULL | stdin via `openWritable` (held open) + `appendFile` flush (bundle) or fd write (direct), plus configurable `postMessage` IPC and mouse regions | bidi stream (raw bytes) → ZMQ |
| Multiplexing | Yes | No (single app) | Yes |
| What it tests | — | App logic, widgets, UX | Multiplexer infra, sessions, streaming |
| Needs service | Yes | No | Yes (subprocess) |

### Crate responsibilities

- **waxterm** (Rust crate: `crates/waxterm/`) — Framework crate. `TerminalApp` trait, `AnsiBackend` (path B), `StructuredBackend` (path A ingestion), `InputParser`, `run_sync`/`run_async`, SGR utilities. Compiles for both native and `wasm32-wasip1`.
- **waxterm** (TypeScript package: `packages/waxterm/`) — Browser-side host and test infrastructure:
  - `src/index.ts` — `TerminalHost` class (dual-mode: bundle/streaming + direct/fd)
  - `src/testing/` — Playwright test helpers shared by Path B and Path C tests (`helpers.ts`, `quality.ts`, `multiplexer.ts`, `index.ts` barrel)
  - `src/vite-plugin.ts` — `crossOriginIsolation()` Vite plugin for SharedArrayBuffer
  - `src/viewer.ts` — `TuiViewerClient` (path C WebTransport + xterm.js browser client)
  - `src/tui-frame.ts` — hand-written `parseTuiFrame()` using generated Cap'n Proto types
- **hyprstream-tui** — Application crate. `WizardApp`, phase state machines, UI widgets. WASM-targeted. Uses waxterm's `TerminalApp` trait. No tokio, no zmq.
- **hyprstream/src/tui/** — Display server (multiplexer). Native only. `TuiState`, `TuiService`, diff engine, 9P virtual FS, VTE parser, `TuiPaneBackend`. Depends on tokio, zmq, Cap'n Proto.
No code duplication between layers. `TuiViewerClient` is ~60 lines of WebTransport + xterm.js glue. Playwright helpers for multiplexer testing reuse the same DOM polling functions as path B.

---

## CLI Commands

### Service management

The TUI display server is a registered service (`#[service_factory("tui", ...)]`), managed like all other hyprstream services:

```bash
hyprstream service start tui            # start as background process (systemd or PID-tracked)
hyprstream service start tui -f         # start in foreground (blocks)
hyprstream service stop tui             # stop the service
hyprstream service status               # show all service status including tui
```

The TUI service uses ZMQ REP as primary protocol (Cap'n Proto RPC for session/window/pane management) with QUIC/WebTransport for browser viewer delivery (via `ctx.into_spawnable_quic()`).

### Session management (`hyprstream tui`)

Top-level `tui` subcommand (like `wizard`, `flight`) for interactive session operations:

```bash
hyprstream tui attach                   # attach to default/last session
hyprstream tui attach --session <id>    # attach to specific session
hyprstream tui new                      # create new session and attach
hyprstream tui list                     # list active sessions (id, name, windows, viewers)
hyprstream tui detach                   # detach from current session (ctrl-b d)
```

`tui attach` connects as a ZMQ viewer (path A). The CLI creates a `StreamHandle`, subscribes to the session's XPUB topic, and renders received ANSI frames to the local terminal via `AnsiBackend`. Keyboard input flows back via the dedicated PUSH socket as `TuiControlMessage::input`.

### Window/pane management (within attached session)

Keybindings (tmux-style prefix, default `ctrl-b`):

| Key | Action |
|-----|--------|
| `ctrl-b c` | Create new window |
| `ctrl-b n` / `ctrl-b p` | Next / previous window |
| `ctrl-b "` | Split pane horizontally |
| `ctrl-b %` | Split pane vertically |
| `ctrl-b o` | Cycle pane focus |
| `ctrl-b x` | Close current pane |
| `ctrl-b d` | Detach from session |
| `ctrl-b w` | Window list (interactive) |

These are also available via 9P filesystem writes (see below).

### Wizard integration

The existing `hyprstream wizard --tui` flag could route through the multiplexer when the TUI service is running:

```bash
# Direct (current behavior — spawns WizardApp on blocking thread)
hyprstream wizard --tui

# Through multiplexer (when tui service is running)
# Wizard app renders into a TUI pane via TuiPaneBackend
hyprstream wizard --tui  # auto-detects running tui service
```

---

## Architectural Decisions

1. **Display server lives in `crates/hyprstream/src/tui/`** — NOT hyprstream-tui (WASM-targeted, no tokio/zmq). hyprstream-tui remains a thin TerminalApp rendered via waxterm, usable in all three paths.

2. **Does NOT extend the registry's 9P.** Registry is git-backed (`ContainedFs`, pathrs). TUI state is ephemeral. Own standalone 9P endpoint.

3. **Does NOT extend `StreamControl`.** `spawn_ctrl_listener` is a blind cancel dispatcher — it doesn't parse union variants. TUI input flows over a dedicated PUSH/PULL socket with its own DH-derived keys. `TuiControlMessage` defined in `tui.capnp`, not `streaming.capnp`.

4. **Uses `ratatui::buffer::Buffer` and `Cell` directly.** No custom `TuiCell`. `Buffer::diff()` produces the exact iterator `AnsiBackend::draw()` consumes. `CompactString` for EGC is built in.

5. **Three ingestion modes per pane** (multiplexer only):
   - **Direct** — In-process ratatui apps via `TuiPaneBackend` (zero serialization, ~8ns/cell)
   - **Structured stream** — waxterm WASI apps via `StructuredBackend` (packed binary cell tuples, ~30ns/cell)
   - **ANSI stream** — Raw PTY programs via VTE parser (legacy/compat path, ~120ns/cell)

6. **Per-viewer encoding mode.** `ViewerHandle` stores `encoding_mode: DisplayMode`. Server encodes once per distinct mode requested (typically 1-2 formats, not per-viewer). ANSI encoding runs server-side via `encode_ansi()` on `compute_diff()` output.

7. **Response variants use `Result` suffix** per `generate_rpc_service!` pairing convention.

8. **No `TuiHello`** — fold into `TuiRequest::connect`. `ephemeralPubkey` travels in `RequestEnvelope`.

9. **Session ≠ stream.** Sessions persist with zero viewers. Each viewer gets independent DH + dedicated `StreamPublisher`. Frame loop is per-session, publishes to all active viewers.

10. **Dedicated PUSH sockets per publisher** (streaming infrastructure improvement). Each `StreamPublisher` gets its own owned `tmq::push::Push` socket (stored as `Option<Push>` for `Drop` to `.take()` into spawned cleanup task) — no shared `Arc<Mutex<Push>>`. Independent HWM per viewer. Non-blocking send via `tokio::time::timeout(Duration::ZERO, send_async())` — tmq converts ZMQ EAGAIN to `Poll::Pending` (not an error), so timeout fires if the inner future is not immediately ready. This is correct: `Duration::ZERO` polls the inner future first, then fires timeout if it returned `Pending`.

11. **No heartbeats in TUI frame loop.** `StreamPayload::Heartbeat` is a dead schema variant — no Rust code sends it, and `publish_heartbeat()` does not exist on `StreamPublisher`. Dedicated PUSH/PULL sockets don't need liveness probes. The Wanix pipe heartbeat in waxterm's `AnsiBackend` is a WASI-only concern, irrelevant to the multiplexer. **Note**: Quinn does not enable QUIC keepalive by default (30s idle timeout). TUI service must set `transport_config.keep_alive_interval(Some(Duration::from_secs(15)))` on its QUIC endpoint to prevent idle connections from closing when a user stares at a static screen.

12. **WebTransport for browser viewers** (path C). The TUI service's QUIC endpoint (via `ctx.into_spawnable_quic()`) natively supports WebTransport. Browser clients open a **client-initiated bidirectional stream** (server accepts via `accept_bi()` in `zmtp_quic.rs`). Frames flow server→client as **4-byte length-prefixed Cap'n Proto `TuiFrame`** messages; input flows client→server as raw key bytes. TLS 1.3 built-in. Short-lived cert (≤14 days per W3C spec) for `serverCertificateHashes`. Chrome 97+ and Firefox 114+ (WebKit does not yet support WebTransport). No SSE — SSE remains the appropriate pattern for OpenAI-compatible streaming but not for bidirectional TUI I/O.

13. **TUI service uses QUIC + ZMQ.** Primary: ZMQ REP for Cap'n Proto RPC (session/window/pane management). QUIC: WebTransport endpoint via `UnifiedServiceConfig` (from `ctx.into_spawnable_quic()`). Both share the same `TuiState` via `Arc<RwLock<>>`. WebTransport session lifecycle managed in new `tui/wt_viewer.rs`, which implements a **new stream handler** distinct from the existing `handle_wt_stream()` (which is single-shot RPC: read request → process → respond → shutdown). TUI bidi streams are long-lived. `handle_webtransport_session()` routes TUI streams to `wt_viewer.rs` based on an initial discriminator byte or URL path.

14. **Auto-discovery via EndpointRegistry.** TUI service endpoints are registered automatically: ZMQ REP at spawn time (RAII), QUIC after bind in `unified_loop()` (registers `SocketKind::Quic` with actual port). No explicit registration call needed. `hyprstream tui attach` queries discovery via `DiscoveryClient::get_endpoints("tui")` to find the TUI service endpoint — no hardcoded ports.

15. **TypeScript codegen in Phase 3.** Once `tui.capnp` is compiled and CGR metadata generated, run `hyprstream-ts-codegen` immediately. The codegen produces `parseTuiResponse()`, `TuiClient` class, builders, and struct interfaces — but **not** standalone parsers for non-response structs like `TuiFrame`. A hand-written `parseTuiFrame()` wrapper (using the generated `CapnpReader` + struct interfaces) is needed for Phase 9's frame deserialization. The generated `TuiClient` and `ConnectRequest` builder are used for the initial RPC handshake.

---

## Phase 0: Streaming Infrastructure (prerequisite)

**Goal**: Add `StreamPublisherConfig`, dedicated sockets, and `try_publish_data()` to hyprstream-rpc. Generic improvement — benefits all services, not just TUI.

**Modify**: `crates/hyprstream-rpc/src/streaming.rs`

```rust
pub struct StreamPublisherConfig {
    pub sndhwm: i32,              // default: 1000
    pub dedicated: bool,          // default: true (own socket per publisher)
}

impl Default for StreamPublisherConfig {
    fn default() -> Self {
        Self { sndhwm: 1000, dedicated: true }
    }
}
```

**Dedicated socket creation** — replace `OnceCell<Arc<Mutex<Push>>>` shared socket with per-publisher socket:

```rust
// In StreamChannel
pub fn create_publisher_socket(&self, config: &StreamPublisherConfig) -> Result<tmq::push::Push> {
    let socket = tmq::push::push(&self.context)
        .set_sndhwm(config.sndhwm)
        .connect(&self.push_endpoint)?;
    Ok(socket)
}
// Note: sndtimeo omitted — tmq always uses DONTWAIT internally and converts
// ZMQ EAGAIN to Poll::Pending. Non-blocking behavior is achieved via
// tokio::time::timeout(Duration::ZERO, ...) in try_publish_data().
```

Each `prepare_stream` / `prepare_stream_with_claims` call creates a dedicated socket. No mutex contention between publishers. On inproc transport, ZMQ sockets are cheap (no kernel fd).

**Non-blocking publish** — owned `Option<Push>` socket (no `Arc<Mutex<Push>>`), timeout-based:

```rust
impl StreamPublisher {
    // socket: Option<tmq::push::Push>  — Option for Drop to .take() into spawned task

    /// Non-blocking publish. Returns Ok(false) if send would block (HWM full).
    pub async fn try_publish_data(&mut self, data: &[u8], rate: f32) -> Result<bool> {
        if self.cancel_token.is_cancelled() {
            anyhow::bail!("stream cancelled");
        }
        if let Some(frames) = self.builder.add_data(data, rate)? {
            let socket = self.socket.as_mut().expect("publisher not dropped");
            // tmq converts ZMQ EAGAIN → Poll::Pending (not an error).
            // Duration::ZERO polls the inner future once, then fires timeout if Pending.
            match tokio::time::timeout(Duration::ZERO, frames.send_async(socket)).await {
                Ok(Ok(())) => Ok(true),           // sent
                Err(_elapsed) => Ok(false),       // HWM full (inner returned Pending)
                Ok(Err(e)) => Err(e.into()),      // real ZMQ error
            }
        } else {
            Ok(true) // still batching, no send needed
        }
    }
}
```

**`Option<Push>` for Drop**: The socket is `Option<tmq::push::Push>` so `Drop` can `.take()` it into a spawned async task for graceful stream termination (sending error/complete frames). This eliminates the current `Arc<Mutex<Push>>` + 200ms lock timeout in `Drop`.

**Backward compatible**: Existing `publish_data()` / `publish_data_with_rate()` unchanged. `try_publish_data()` is additive. Existing services (inference) use defaults — behavior identical to today but with dedicated sockets (eliminates shared mutex contention).

**Tests**: `try_publish_data` returns `false` when HWM hit (set `sndhwm=1`, send without consumer), dedicated socket creates independently, existing publish_data still works with defaults.

---

## Phase 1: TuiState and Cell Grid (Foundation)

**New files**: `crates/hyprstream/src/tui/mod.rs`, `crates/hyprstream/src/tui/state.rs`
**Modify**: `crates/hyprstream/src/lib.rs` — add `pub mod tui;`
**Modify**: `crates/hyprstream/Cargo.toml` — add `ratatui = { version = "0.30", default-features = false }`, promote `dashmap.workspace = true` to unconditional `[dependencies]`

```rust
use ratatui::buffer::{Buffer, Cell};
use ratatui::layout::Rect;

pub struct PaneBuffer {
    pub buf: Buffer,                    // ratatui buffer — Cell uses CompactString
    pub scroll_log: Vec<ScrollOp>,      // appended by VTE performer, consumed by diff; capped at 256
    damage_rows: u128,                  // bitmask for rows 0-127 (covers practical terminal heights)
    pub force_full_frame: bool,         // set on alt screen switch, scroll_log overflow, resize
}

pub struct TuiPane {
    pub id: u32,
    pub primary: PaneBuffer,
    pub alternate: Option<PaneBuffer>,  // lazy — allocated on first ?1049h
    pub active_screen: ScreenSelector,
    pub scrollback: VecDeque<Vec<Cell>>, // 2000 lines default
    pub cursor: CursorState,
    pub saved_cursor: Option<CursorState>,
    pub title: String,
    pub ingestion: IngestionMode,       // Direct | Structured | Ansi
    pub vte: Option<VteParserState>,    // only for Ansi mode
}

pub struct TuiWindow {
    pub id: u32,
    pub name: String,
    pub layout: LayoutNode,
    pub active_pane_id: u32,
    pub panes: Vec<TuiPane>,
}

pub struct TuiSession {
    pub id: u32,
    pub name: String,
    pub windows: Vec<TuiWindow>,
    pub active_window: usize,
}

pub enum LayoutNode {
    Leaf { pane_id: u32 },
    HSplit { ratio_num: u16, ratio_den: u16, first: Box<LayoutNode>, second: Box<LayoutNode> },
    VSplit { ratio_num: u16, ratio_den: u16, first: Box<LayoutNode>, second: Box<LayoutNode> },
}

pub struct CursorState {
    pub x: u16, pub y: u16,
    pub visible: bool,
    pub shape: CursorShape,
}

pub enum IngestionMode { Direct, Structured, Ansi }
pub enum ScreenSelector { Primary, Alternate }

pub struct TuiState {
    sessions: Vec<TuiSession>,
    composite: Buffer,               // flattened output (ratatui Buffer)
    generation: u64,                 // incremented on any mutation
    window_generation: u32,          // for 9P WindowsDir qid.version
    events: broadcast::Sender<TuiEvent>,
}
```

**Locking**: `RwLock<TuiPane>` (not just `PaneBuffer`) — protects `active_screen`, `alternate`, `cursor`, and buffers atomically. Frame loop acquires pane read locks in ascending id order, drops each before ZMQ send. VTE performer batches cell writes, acquires write lock briefly to flush batch (not per-byte).

**Double-buffer for direct mode**: `TuiPaneBackend::draw()` writes into a private staging buffer, then takes a brief write lock to swap the reference into `PaneBuffer.buf`. Read lock contention with frame loop is bounded to the swap (~memcpy of Buffer pointer), not the full cell iteration.

**Scroll log cap**: `scroll_log` capped at 256 entries. On overflow, clear log and set `force_full_frame = true`.

**Deps**: `ratatui = { version = "0.30", default-features = false }` (new), `vte = "0.13"` (new, for Phase 7), `unicode-width` (transitive via ratatui)

**Tests**: PaneBuffer cell write + damage bitmask (including row > 63), compose two panes in split, resize, alternate screen switch sets `force_full_frame`, scroll_log overflow triggers full frame

---

## Phase 2: Diff Engine

**New file**: `crates/hyprstream/src/tui/diff.rs`

```rust
pub struct ScrollOp { pub region_top: u16, pub region_bottom: u16, pub lines: i16 }

pub enum FramePayload {
    Incremental { scrolls: Vec<ScrollOp>, deltas: Vec<(u16, u16, Cell)> },
    Full { width: u16, height: u16, buf: Buffer },
}

pub struct FrameDiff {
    pub generation: u64,
    pub payload: FramePayload,
    pub cursor: CursorState,
}
```

**Full-frame triggers**: `force_full_frame` flag set, or `delta_count > total_cells / 2`, or generation 0 (reconnect). Diff engine checks flag first, clears it after emitting full frame.

**ScrollOp**: Recorded in `PaneBuffer.scroll_log` by VTE performer. Diff engine holds the pane write lock for the atomic sequence: drain `scroll_log` → read `buf` for cell diff → clear `damage_rows`. This prevents compose from seeing scroll_log entries that haven't been consumed yet.

**Continuation cell skip**: Use `to_skip` counter from `cell.symbol().width().saturating_sub(1)`, matching `Buffer::diff()` internals. NOT `cell.symbol() == ""` (which never matches — ratatui resets continuation cells to `" "`).

**Wide character cursor**: After emitting a wide glyph, advance cursor tracking by `unicode_width::UnicodeWidthStr::width(cell.symbol())`, not always 1.

**SGR reuse**: Extract `write_combined_sgr`, `write_fg_params`, `write_bg_params` from `crates/waxterm/src/backend.rs` into `crates/waxterm/src/sgr.rs` as `pub` functions. Both `AnsiBackend::draw()` and `encode_ansi()` import from there.

**Tests**: Empty diff, single cell delta, ScrollOp from log, full-frame threshold, `force_full_frame` flag, wide char `to_skip` counter, continuation cell skipped, ANSI roundtrip

---

## Phase 3: Cap'n Proto Schema (tui.capnp)

**New file**: `crates/hyprstream/schema/tui.capnp`

**Does NOT modify `streaming.capnp`.**

```capnp
# Convention: Response variants suffixed with "Result" for generate_rpc_service! pairing.

struct TuiRequest {
  id @0 :UInt64;
  union {
    connect      @1 :ConnectRequest;    # returns StreamInfo
    disconnect   @2 :Text;              # sessionId
    createWindow @3 :Void;
    closeWindow  @4 :Text;
    listWindows  @5 :Void;
    focusWindow  @6 :Text;
    splitPane    @7 :SplitRequest;
    closePane    @8 :Text;
    focusPane    @9 :Text;
    snapshot    @10 :Text;
    resize      @11 :ResizeEvent;
  }
}

struct TuiResponse {
  requestId @0 :UInt64;
  union {
    error              @1 :ErrorInfo;
    connectResult      @2 :StreamInfo;
    disconnectResult   @3 :Void;
    createWindowResult @4 :WindowInfo;
    closeWindowResult  @5 :Void;
    listWindowsResult  @6 :List(WindowInfo);
    focusWindowResult  @7 :Void;
    splitPaneResult    @8 :PaneInfo;
    closePaneResult    @9 :Void;
    focusPaneResult   @10 :Void;
    snapshotResult    @11 :TuiSnapshot;
    resizeResult      @12 :Void;
  }
}

struct ConnectRequest {
  mode   @0 :DisplayMode;   # structured / ansiStream
  width  @1 :UInt16;
  height @2 :UInt16;
  # ephemeralPubkey in RequestEnvelope — not duplicated here
}

# TUI-specific client→server messages (dedicated PUSH/PULL socket, NOT StreamControl)
struct TuiControlMessage {
  union {
    input  @0 :TuiInputEvent;
    ack    @1 :TuiAck;
    resize @2 :ResizeEvent;
  }
}

# TuiFrame packed into StreamPayload.data bytes (same pattern as inference streaming)
# Also sent length-prefixed on WebTransport bidi streams (Phase 9)
# No heartbeat variant — QUIC keepalive handles liveness (see decision 11)
struct TuiFrame {
  generation @0 :UInt64;
  union {
    incremental @1 :IncrementalFrame;
    full        @2 :Data;       # complete frame bytes (interpretation depends on DisplayMode from ConnectRequest)
  }
}

struct IncrementalFrame {
  scrollOps @0 :List(ScrollOp);
  deltas    @1 :List(CellDelta);
  cursor    @2 :CursorState;
}
```

Plus: `ScrollOp`, `CellDelta` (with `width :UInt8` for wide chars), `CursorState`, `CursorShape`, `DisplayMode`, `SplitDirection`, `SplitRequest`, `ResizeEvent`, `WindowInfo`, `PaneInfo`, `TuiSnapshot`, `TuiInputEvent`, `TuiAck`, `ErrorInfo`.

**Build + Codegen**:
1. Add `tui.capnp` to `crates/hyprstream/build.rs` schema list (alongside inference, registry, etc.)
2. `cargo build -p hyprstream` — compiles schema, regenerates CGR metadata at `codegen-out/tui.cgr`
3. Run TypeScript codegen: `cargo run --bin hyprstream-ts-codegen -- --input-dir codegen-out --output-dir packages/waxterm/src/generated/`

The binary takes `--input-dir` (directory of `.cgr` files) and `--output-dir` (output directory), not single file args. This produces `parseTuiResponse()`, `TuiClient` class, request builders, and struct interfaces. **Note**: the codegen only generates parsers for the top-level response struct (`parseTuiResponse`), not standalone parsers for streaming structs like `TuiFrame`. A hand-written `parseTuiFrame()` using the generated `CapnpReader` + struct interfaces is needed in Phase 9.

**Modify**: `crates/hyprstream/build.rs` — add `"tui"` to schema list

---

## Phase 4: Standalone 9P Virtual Filesystem

**New file**: `crates/hyprstream/src/tui/ninep.rs`

**Lightweight structural identifiers, not embedded Arc clones:**

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum TuiNodeId {
    Root, GlobalCtl, NewWindow, EventFile, WindowsDir,
    WindowDir { window_id: u32 },
    WindowCtl { window_id: u32 },
    WindowCons { window_id: u32 },
    WindowScreen { window_id: u32 },
    WindowWinName { window_id: u32 },
    WindowLayout { window_id: u32 },
    PanesDir { window_id: u32 },
    PaneDir { window_id: u32, pane_id: u32 },
    PaneCons { window_id: u32, pane_id: u32 },
    PaneScreen { window_id: u32, pane_id: u32 },
}

pub enum TuiFidState {
    Walked { node: TuiNodeId },
    Opened { node: TuiNodeId, mode: u8, cons_cursor: u64,
             event_rx: Option<broadcast::Receiver<TuiEvent>> },
}
```

`TuiState` resolved at read/write time under lock, not at walk time.

**`qid_path` encoding** — explicit bit layout to prevent collisions:
```
bits 62-56: kind discriminant (7 bits = 128 kinds)
bits 55-32: window_id (24 bits = 16M windows)
bits 31-0:  pane_id (32 bits)
```
Single-id nodes (WindowDir, WindowCtl, etc.) use only bits 55-32. No-id nodes (Root, GlobalCtl, etc.) use only the discriminant. Collision-free by construction.

Directory children enumerated live at read time (offset 0 only, empty at offset > 0 — matches registry pattern).

**`/tui/new` creation semantics**: Walk to `/tui/new` atomically allocates a new window (Plan 9 `/dev/new` pattern — creation at walk time, not open). The returned qid embeds the new window ID. Open is a no-op. Read returns window ID as decimal string.

**Filesystem layout** (Plan 9 rio):
```
/tui/
  ctl             # write: global commands ("close 0", "focus 1")
  new             # walk creates window; open no-op; read → window ID decimal
  event           # read: blocks until event available (idiomatic 9P)
  windows/
    0/
      ctl         # write: "split h 0.5", "title foo", "delete"
      cons        # read: ANSI stream (ring buffer), write: keyboard input
      screen      # read: binary framebuffer snapshot at offset 0
      winname     # read/write: window title
      layout      # read: pane tree description
      panes/
        0/cons, screen
        1/cons, screen
```

**ConsBuffer** for `/cons` streaming reads:
```rust
pub struct ConsBuffer {
    ring: Box<[u8; 256 * 1024]>,  // 256KB fixed ring
    write_offset: u64,             // monotonic, never wraps
    notify: Arc<Notify>,
}
```
`cons_cursor` initialized to current `write_offset` at open time (new data only). Gap detection via seqlock pattern: read `write_offset` before and after memcpy; if gap exceeds ring size after copy, retry or return synthetic `\x1b[?gap:{offset}\x07`. `ConsBuffer` protected by `RwLock` — writer holds write lock during ring append, reader holds read lock during copy.

**`/tui/event` read semantics**: Blocks until event available (idiomatic 9P — matches Plan 9 event files). Implemented as `event_rx.recv().await` inside a `tokio::task::spawn_blocking` bridge (since 9P handlers run in ZMQ REQ/REP context). On `broadcast::RecvError::Lagged(n)`, return synthetic `"lagged {n}\n"` event line so client knows events were missed.

**Reuses** wire format types from `services/types.rs` (Qid, QTDIR, OREAD, etc.) and `insert_counted`/`take`/`replace` fid discipline from registry. No registry code modified.

**Tests**: Walk `/tui/windows/0/cons`, read returns data; walk `/tui/new` creates window, read returns ID; directory read enumerates live windows; ConsBuffer gap detection; event file blocking + lagged handling

---

## Phase 5: TuiService (ZmqService + QUIC)

**New files**: `crates/hyprstream/src/tui/service.rs`, `crates/hyprstream/src/tui/wt_viewer.rs`

The TUI service uses `ctx.into_spawnable_quic()` (established by discovery, oauth, mcp services) to run both ZMQ REP and QUIC listeners in a `UnifiedServiceConfig`.

```rust
pub struct TuiService {
    state: Arc<RwLock<TuiState>>,
    sessions: DashMap<u32, SessionHandle>,
    fid_table: Arc<TuiFidTable>,
    context: Arc<zmq::Context>,
    transport: TransportConfig,
    signing_key: SigningKey,
}

pub struct SessionHandle {
    viewers: Vec<ViewerHandle>,
    frame_cancel: CancellationToken,
}

pub struct ViewerHandle {
    viewer_id: String,
    publisher: StreamPublisher,       // dedicated socket per viewer (Phase 0)
    encoding_mode: DisplayMode,       // structured or ansiStream
    cancel_token: CancellationToken,
    consecutive_skips: u32,
}
```

**Session lifecycle** (separate from viewer streams — does NOT reuse worker attach pattern):
1. `create_session(config)` → `session_id`, spawns `session_frame_loop`
2. `attach_viewer(session_id, client_pubkey, mode)` → DH exchange with `StreamPublisherConfig { sndhwm: 4, dedicated: true }` → `(StreamInfo, Continuation)`
3. Continuation adds viewer to session's publisher list, monitors disconnect
4. Sessions persist with zero viewers

**Frame loop** (`tokio::time::interval` + `biased select!` — clean idle behavior, no stacking waits):
```rust
let mut interval = tokio::time::interval(Duration::from_millis(33));
interval.set_missed_tick_behavior(MissedTickBehavior::Skip);
let mut damage_pending = false;

loop {
    // biased: check cancel first, then damage, then tick
    tokio::select! {
        biased;

        _ = cancel_token.cancelled() => break,

        _ = damage_rx.recv(), if !damage_pending => {
            damage_pending = true;
            // drain any additional pending notifications
            while damage_rx.try_recv().is_ok() {}
        }

        _ = interval.tick() => {
            if !damage_pending { continue; }

            // Extract diff under WRITE lock (drains scroll_log, clears damage_rows)
            // then drop lock before encoding/publishing (no lock held across await)
            let diff = {
                let mut state = state.write().await;
                extract_diff(&mut state)  // drain scroll_log + snapshot damage + clear
            }; // RwLock dropped here

            // Encode once per distinct DisplayMode requested (no lock held)
            let ansi_bytes = if needs_ansi { Some(encode_ansi(&diff)) } else { None };
            let capnp_bytes = if needs_structured { Some(encode_capnp(&diff)) } else { None };

            // Per-viewer publish — errors handled per-viewer, not propagated
            for viewer in &mut session.viewers {
                let bytes = match viewer.encoding_mode {
                    DisplayMode::AnsiStream => ansi_bytes.as_ref().unwrap(),
                    DisplayMode::Structured => capnp_bytes.as_ref().unwrap(),
                };
                match viewer.publisher.try_publish_data(bytes, 30.0).await {
                    Ok(true) => viewer.consecutive_skips = 0,
                    Ok(false) => {
                        viewer.consecutive_skips += 1;
                        if viewer.consecutive_skips > 30 { // 1s of drops
                            viewer.cancel_token.cancel();  // mark for eviction
                        }
                    }
                    Err(_) => viewer.cancel_token.cancel(), // stream error → evict
                }
            }
            // Remove evicted viewers so they don't accumulate
            session.viewers.retain(|v| !v.cancel_token.is_cancelled());
            damage_pending = false;
        }
    }
}
```

No heartbeats — QUIC transport-level keepalive handles liveness (see decision 11).

**Dedicated input socket**: PULL socket at `tui-input` endpoint. Receives `TuiControlMessage` (keyboard, ack, resize). Demuxed by session ID. Does NOT touch `StreamControl` or `streaming.rs`.

**Service factory** (in `crates/hyprstream/src/services/factories.rs`):

```rust
#[service_factory("tui", schema = "../../schema/tui.capnp",
                 metadata = crate::services::generated::tui_client::schema_metadata)]
fn create_tui_service(ctx: &ServiceContext) -> anyhow::Result<Box<dyn Spawnable>> {
    let config = load_config();  // returns HyprConfig directly (not Result, not generic)
    let state = Arc::new(RwLock::new(TuiState::new(config.tui.clone())));
    let service = TuiService::new(ctx, state.clone())?;

    // ZMQ REP + QUIC (WebTransport) in one Spawnable — second arg is Option<u16>
    // Endpoints auto-registered: ZMQ REP at spawn (RAII), QUIC after bind in unified_loop()
    Ok(ctx.into_spawnable_quic(service, config.tui.quic_port))
}
```

The `ctx.into_spawnable_quic()` pattern (established by discovery, oauth, and mcp services) wraps the `ZmqService` in a `UnifiedServiceConfig` that runs both ZMQ REQ/REP and QUIC listeners. The QUIC endpoint accepts WebTransport connections from browsers (Path C) and ZMTP-over-QUIC from native clients. `unified_loop()` registers the QUIC endpoint with `SocketKind::Quic` in `EndpointRegistry` after binding the actual port.

**TuiServiceConfig** (follows naming convention: `DiscoveryServiceConfig`, `RegistryServiceConfig`, etc.):

```rust
// In crates/hyprstream/src/config/mod.rs
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TuiServiceConfig {
    pub quic_port: Option<u16>,       // WebTransport port (None = auto-assign)
    pub max_sessions: u32,            // default: 64
    pub scrollback_lines: u32,        // default: 2000
    pub wt_cert_validity_days: u16,   // default: 14 (W3C max for serverCertificateHashes)
}

impl Default for TuiServiceConfig {
    fn default() -> Self {
        Self { quic_port: None, max_sessions: 64, scrollback_lines: 2000, wt_cert_validity_days: 14 }
    }
}

// Add to HyprConfig:
pub struct HyprConfig {
    // ... existing fields ...
    #[serde(default)]
    pub tui: TuiServiceConfig,
}
```

**CLI handler** (in `crates/hyprstream/src/cli/` — new `tui_handlers.rs`):

```rust
pub async fn handle_tui_attach(session_id: Option<u32>) -> Result<()> {
    // 1. Query EndpointRegistry via DiscoveryClient for TUI endpoint
    // (with_expected_audience is server-side only — client does not set audience)
    let discovery = DiscoveryClient::new(signing_key.clone(), identity.clone());
    let endpoints = discovery.get_endpoints("tui").await?;
    let zmq_endpoint = endpoints.iter().find(|e| e.kind == SocketKind::Rep)
        .ok_or_else(|| anyhow!("TUI service not running"))?;

    // 2. Connect to TuiService via ZMQ
    let client = TuiZmqClient::new_at(&zmq_endpoint.address)?;

    // 3. Request connection with AnsiStream mode
    let stream_info = client.connect(ConnectRequest {
        mode: DisplayMode::AnsiStream,
        width: term_cols,
        height: term_rows,
    }).await?;

    // 4. Create StreamHandle (subscribe to XPUB)
    let mut handle = StreamHandle::new(&zmq_ctx, &stream_info, ...)?;

    // 5. Enable raw mode, render frames to local terminal
    let original_termios = enable_raw_mode()?;
    loop {
        tokio::select! {
            frame = handle.next() => { /* write ANSI to stdout */ }
            key = stdin_reader.next() => { /* send via PUSH socket */ }
        }
    }
    disable_raw_mode(original_termios);
}
```

**Modify**: `crates/hyprstream/src/services/factories.rs`, `services/generated.rs`, `cli/mod.rs` (add `tui` subcommand)

---

## Phase 6: StructuredBackend (waxterm)

**New file**: `crates/waxterm/src/structured.rs`
**New file**: `crates/waxterm/src/sgr.rs` (extracted from `backend.rs`)

```rust
/// Backend that emits packed binary cell tuples instead of ANSI.
/// For WASI apps (Minerva, wizard) communicating with TUI display server.
/// ~30ns/cell vs ~120ns/cell for ANSI→VTE round-trip.
pub struct StructuredBackend<W: Write> {
    writer: W,
    width: u16,
    height: u16,
}

impl<W: Write> Backend for StructuredBackend<W> {
    fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error>
    where I: Iterator<Item = (u16, u16, &'a Cell)>
    {
        // Wire format per cell: x(u16) + y(u16) + symbol_len(u8) + symbol(utf8)
        //   + fg(4 bytes) + bg(4 bytes) + modifiers(u8) = ~14 bytes typical
        // Frame: count(u32) + cells + generation(u64)
        // Display server deserializes directly into PaneBuffer — no VTE.
    }
}
```

ANSI output for clients is produced by `encode_ansi()` in the diff engine (Phase 2), applied to `compute_diff()` output. `StructuredBackend` eliminates the ANSI→VTE→cells round-trip for waxterm apps. SGR functions extracted to `sgr.rs` in Phase 2.

---

## Phase 7: VTE Parser (PTY Compatibility)

**Add to**: `crates/hyprstream/src/tui/state.rs`

```rust
pub struct VteParserState {
    parser: vte::Parser,
    performer: PanePerformer,
}
```

`PanePerformer` implements `vte::Perform`. Batches cell writes into a local staging area, then acquires the `TuiPane` write lock briefly to flush the batch into `PaneBuffer.buf` and mark `damage_rows`. Records `ScrollOp` to `scroll_log` on SU/SD/IL/DL (not inferred by diff).

**`damage_rows` bounds check**: Since `damage_rows` is a `u128` bitmask covering rows 0-127, the VTE performer must check `row_y < 128` before setting the bit. For rows >= 128, set `force_full_frame = true` instead. In practice, terminals rarely exceed 128 rows, but the guard prevents silent bit truncation.

**Minimum CSI set:**
- Tier 1 (ratatui emits): SGR (`m`), CUP (`H`/`f`), ED (`J`), EL (`K`)
- Tier 2 (scroll — DECSTBM is prerequisite for SU/SD correctness): DECSTBM (`r`), SU (`S`), SD (`T`), IL (`L`), DL (`M`)
- Tier 3 (cursor): CUU/CUD/CUF/CUB (`A`/`B`/`C`/`D`), CHA (`G`), VPA (`d`), DECTCEM (`?25h`/`?25l`)
- Tier 4 (alt screen): DECSET/DECRST 1049 (`?1049h`/`?1049l`)
- Unknown sequences: silently ignored (`_ => {}`)

**Alt screen**: On `?1049h` — save cursor, set `active_screen = Alternate`, lazy-allocate alt buffer if needed, clear alt buffer, set `force_full_frame = true`. On `?1049l` — restore cursor, set `active_screen = Primary`, set `force_full_frame = true`.

**Dedicated reader thread per ANSI-mode pane**: Reads PTY fd, sends bytes via `rtrb` (lockfree SPSC ring buffer, new dep) to VTE performer task. Frame loop never blocks on fd reads.

**Deps**: `vte = "0.13"`, `rtrb = "0.3"` in `crates/hyprstream/Cargo.toml`

---

## Phase 8: Integration Wiring

### TuiPaneBackend (ratatui → TUI pane, direct mode)

**New in**: `crates/hyprstream/src/tui/backend.rs`

```rust
pub struct TuiPaneBackend {
    pane: Arc<RwLock<TuiPane>>,
    staging: Buffer,                 // private — no lock during cell writes
}
impl ratatui::backend::Backend for TuiPaneBackend {
    fn draw<'a, I>(&mut self, content: I) -> Result<(), Self::Error> {
        // Write cells into self.staging (no lock, ~8ns/cell)
        for (x, y, cell) in content {
            self.staging[(x, y)] = cell.clone();
        }
        // Brief write lock to swap staging into pane buffer
        let mut pane = self.pane.write();
        std::mem::swap(&mut pane.active_buffer_mut().buf, &mut self.staging);
        pane.active_buffer_mut().damage_rows = u128::MAX; // mark all dirty
        // drop write lock immediately
    }
}
```

Lock contention with frame loop bounded to the swap duration (~nanoseconds), not the full cell iteration.

### Structured stream ingestion

Display server reads packed cell tuples from WASI app stdout (via worker fd or Wanix pipe). Deserializes into `PaneBuffer` cells (~30ns/cell). No VTE parser needed.

### Worker output → TUI pane

Subscribe to worker FD stream via `StreamHandle`. Route to appropriate ingestion mode based on pane's `IngestionMode`. TuiService manages DH + publisher independently — does NOT delegate to worker's `prepare_attach()` (which is 1:1 session-to-stream).

---

## Phase 9: WebTransport Client and Multiplexer Testing (Path C)

**New files**:
- `web-minerva-game/tests/multiplexer.spec.ts` — Playwright test suites

**Note**: `packages/waxterm/` already exists as a TypeScript package (the Rust crate `waxterm` is at `crates/waxterm/`). The following files are already present: `package.json`, `tsconfig.json`, `src/index.ts` (TerminalHost), `src/testing/helpers.ts`, `src/testing/quality.ts`, `src/testing/multiplexer.ts`, `src/testing/index.ts`, `src/vite-plugin.ts`, `src/viewer.ts`, `src/tui-frame.ts`. Phase 9 adds the Playwright multiplexer test suite, not the package itself.

**Modify**:
- `crates/hyprstream-rpc/src/transport/zmtp_quic.rs` — extend `handle_http_request()` with `/.well-known/tui-cert-hash`, route TUI bidi streams in `handle_webtransport_session()`
- `web-minerva-game/playwright.config.ts` — add multiplexer project with subprocess globalSetup

The QUIC/WebTransport endpoint is part of TuiService (Phase 5, via `ctx.into_spawnable_quic()`). WebTransport session lifecycle is handled by `tui/wt_viewer.rs` (Phase 5). Phase 9 adds the browser client and Playwright test infrastructure.

**WebTransport endpoint** (via QUIC listener from Phase 5):
- Browser connects via `new WebTransport(url, { serverCertificateHashes })` to TUI service QUIC port
- Client opens a **bidirectional stream** (server accepts via `accept_bi()` in `zmtp_quic.rs`)
- Server writes **4-byte length-prefixed Cap'n Proto `TuiFrame`** messages on the server→client half
- Client writes raw key bytes on the client→server half
- TLS 1.3 built-in. Short-lived ECDSA P-256 cert (≤14 days) via `generate_wt_cert()`, separate from the 365-day `SHARED_TLS` cert. Cert hash served at `/.well-known/tui-cert-hash` (requires extending `handle_http_request()` in `zmtp_quic.rs`)
- `tui/wt_viewer.rs` implements a **new long-lived stream handler** (distinct from `handle_wt_stream()` which is single-shot RPC). `handle_webtransport_session()` routes to `wt_viewer.rs` based on initial discriminator byte. Lifecycle: accept bidi stream, read initial `ConnectRequest`, authenticate, pipe frames from session publisher, pipe input to pane stdin

**Browser client** (`TuiViewerClient`): Uses generated TypeScript types from Phase 3 codegen to deserialize `TuiFrame` messages from the bidi stream and to serialize `ConnectRequest` for the initial handshake. Renders into xterm.js. Same terminal DOM structure as `TerminalHost` — Playwright helpers work unchanged.

```typescript
// parseTuiFrame is hand-written using generated CapnpReader + struct interfaces
// (codegen only generates parseTuiResponse, not standalone struct parsers)
import { parseTuiFrame, type TuiFrame } from './tui-frame';

export class TuiViewerClient {
    private term: Terminal;
    private transport: WebTransport;
    private writer: WritableStreamDefaultWriter | null = null;

    static async connect(
        container: HTMLElement,
        endpoint: string,
        sessionId: string,
        certHash: ArrayBuffer,
    ): Promise<TuiViewerClient> {
        const client = new TuiViewerClient(container);
        client.transport = new WebTransport(`https://${endpoint}/tui?session=${sessionId}`, {
            serverCertificateHashes: [{ algorithm: 'sha-256', value: certHash }],
        });
        await client.transport.ready;

        // Client-initiated bidirectional stream (server accepts via accept_bi)
        const bidi = await client.transport.createBidirectionalStream();
        client.writer = bidi.writable.getWriter();
        client.pipeFramesToXterm(bidi.readable).catch((e) =>
            console.error('[TuiViewerClient] frame pipe error:', e));

        return client;
    }

    private async pipeFramesToXterm(readable: ReadableStream<Uint8Array>): Promise<void> {
        const reader = readable.getReader();
        let buffer = new Uint8Array(0);
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            // Accumulate and parse length-prefixed Cap'n Proto TuiFrame messages
            buffer = concat(buffer, value);
            while (buffer.length >= 4) {
                // Big-endian length prefix (matches server: u32::to_be_bytes)
                const len = new DataView(buffer.buffer, buffer.byteOffset).getUint32(0);
                if (buffer.length < 4 + len) break;
                const frame = parseTuiFrame(buffer.subarray(4, 4 + len));
                buffer = buffer.subarray(4 + len);
                // TuiFrame union: 'full' contains ANSI bytes for AnsiStream mode
                if (frame.variant === 'full' && frame.data) {
                    this.term.write(frame.data as Uint8Array);
                }
                // TODO: handle 'incremental' variant for Structured mode
            }
        }
    }

    async sendInput(bytes: Uint8Array): Promise<void> {
        // Writer held for connection lifetime — no getWriter/releaseLock per call
        await this.writer?.write(bytes);
    }

    destroy(): void {
        // close() gracefully finishes writes; transport.close() aborts everything
        this.writer?.close();
        this.transport.close();
        this.term.dispose();
    }
}
```

**Playwright globalSetup**: Spawns `hyprstream service start tui -f` (foreground), captures readiness signal from stdout, stores QUIC port in `process.env.MULTIPLEXER_QUIC_PORT`. globalTeardown sends SIGTERM.

**Test suites** (`multiplexer.spec.ts`) — **Chromium only** (WebKit lacks WebTransport):
- Service lifecycle: start, connect viewer, receive first frame
- Session management: create session, detach, reattach (state preserved)
- Multi-pane: split, focus, close — layout changes reflected in frames
- Input routing: sendKey → app state change → frame update
- Viewer eviction: pause consumption, verify disconnect after ~1s
- Concurrent viewers: two browser tabs, same session, independent streams

**Note on Playwright + WebTransport**: Playwright has no protocol-level WebTransport API (no `route()`, no interception). All assertions use DOM polling (`getTerminalText`, `waitForTerminalText`) — identical to Path B testing. The transport is transparent to test assertions. Configure Playwright for Chromium only:

```typescript
// playwright.config.ts — multiplexer project
{ name: 'multiplexer', use: { ...devices['Desktop Chrome'] }, testMatch: 'multiplexer.spec.ts' }
```

**Deps**: No new Rust deps (QUIC/WebTransport already merged). TypeScript: `WebTransport` is a built-in browser API. Generated TS types from Phase 3.

---

## Performance Budget (80x24, 20% dirty, 30fps)

| Operation | Per frame | % of 33ms |
|-----------|----------|-----------|
| Buffer::diff() | ~15μs | 0.05% |
| encode_ansi() | ~25μs | 0.08% |
| encode_capnp() | ~10μs | 0.03% |
| HMAC per viewer | ~2μs | <0.01% |
| ZMQ send per viewer (dedicated socket, no mutex) | ~3μs | <0.01% |
| **Total (1 viewer)** | **~55μs** | **0.17%** |
| **Total (10 viewers)** | **~105μs** | **0.32%** |

Memory per pane: ~330KB typical (primary 61KB + cons ring 256KB + VTE ~few KB). Excludes scrollback — at 2000 lines x 80 cols x ~56 bytes/Cell = ~9MB per pane if fully populated. Scrollback is the dominant memory driver for long-running sessions.

Bottleneck: fd I/O for ANSI-mode panes (PTY read latency 50-200μs). Mitigated by dedicated `rtrb` reader threads.

---

## Dependency Graph

```
Phase 0 (StreamPublisher)   Phase 1 (TuiState)      Phase 3 (tui.capnp + TS codegen)
         │                       │                        │
         │                       ├──── Phase 2 (Diff) ───┘
         │                       │          │
         │                       ├──── Phase 4 (9P) ─── Phase 7 (VTE)
         │                       │          │                │
         └───────────────── Phase 5 (TuiService + QUIC) ─── Phase 6 (StructuredBackend)
                                      │
                                 Phase 8 (Integration)
                                      │
                                 Phase 9 (WebTransport + Playwright)
```

**Parallel**: Phases 0, 1, 3 have no cross-deps. Phase 6 (waxterm) and Phase 7 (VTE) are independent. Phase 9 depends on Phase 5 (needs running TuiService) and Phase 3 (uses generated TS types).

---

## Files Summary

| Phase | New Files | Modified Files |
|-------|-----------|----------------|
| 0 | | `hyprstream-rpc/src/streaming.rs` |
| 1 | `tui/mod.rs`, `tui/state.rs` | `lib.rs`, `hyprstream/Cargo.toml` |
| 2 | `tui/diff.rs`, `waxterm/src/sgr.rs` | `waxterm/src/backend.rs` |
| 3 | `schema/tui.capnp`, `packages/waxterm/src/generated/tui.ts` | `hyprstream/build.rs` |
| 4 | `tui/ninep.rs` | |
| 5 | `tui/service.rs`, `tui/wt_viewer.rs`, `cli/tui_handlers.rs` | `services/factories.rs`, `services/generated.rs`, `cli/mod.rs`, `config/mod.rs` |
| 6 | `waxterm/src/structured.rs` | |
| 7 | | `tui/state.rs`, `hyprstream/Cargo.toml` |
| 8 | `tui/backend.rs` | |
| 9 | `web-minerva-game/tests/multiplexer.spec.ts` | `zmtp_quic.rs` (cert hash + WT routing), `playwright.config.ts` |

**Note**: `packages/waxterm/` already exists with scaffolding and files: `package.json`, `tsconfig.json`, `src/index.ts` (TerminalHost), `src/testing/helpers.ts`, `src/testing/quality.ts`, `src/testing/multiplexer.ts`, `src/testing/index.ts`, `src/vite-plugin.ts`, `src/viewer.ts`, `src/tui-frame.ts`. Phase 9 adds the Playwright multiplexer test suite.

---

## Verification

### Build gates
1. `cargo build -p hyprstream-rpc` — StreamPublisherConfig compiles, try_publish_data works
2. `cargo build -p hyprstream` — schema compiles, CGR metadata generated
3. `cargo build -p hyprstream-tui --target wasm32-wasip1` — WASI boundary preserved (no tokio/zmq leaked)
4. `cargo clippy --workspace --all-features`

### Unit tests
5. `cargo test -p hyprstream-rpc streaming` — dedicated socket + non-blocking send tests pass
6. `cargo test -p hyprstream tui` — all TUI unit tests pass
7. `cargo test --workspace` — no regressions

### Path A: Production (multiplexer)
8. `hyprstream service start tui -f` — service starts, binds ZMQ + QUIC endpoints
9. `hyprstream tui attach` — connects, receives frames, renders to terminal
10. 9P walk to `/tui/windows/0/cons`, verify read returns data
11. StructuredBackend WASI app → pane renders without VTE
12. Slow viewer evicted after 30 consecutive skips (1s)

### Path B: App testing (direct WASI)
13. `cargo build -p hyprstream-tui --target wasm32-wasip1` — WASM binary produced
14. Existing Playwright tests pass (`npx playwright test` in web-minerva-game) — validates that waxterm `AnsiBackend` path is unbroken by SGR extraction and StructuredBackend additions
15. `WizardApp` renders identically through both `AnsiBackend` (direct) and `TuiPaneBackend` (multiplexer) — verified by snapshot comparison of final Buffer state

### Path C: Multiplexer testing (WebTransport + Playwright)
16. `npx playwright test multiplexer.spec.ts` — all multiplexer Playwright suites pass
17. WebTransport delivers frames: browser connects via QUIC, xterm.js renders correctly
18. Input roundtrip: input via WebTransport stream → app state change → next frame reflects update
19. Session persistence: disconnect viewer, reconnect, verify terminal state preserved
20. Concurrent viewers: two browser contexts subscribe to same session, both receive frames independently
21. Generated TS types (`packages/waxterm/src/generated/tui.ts`) compile and match schema
