# Compositor Architecture

The TUI chrome (tab strip, model list, modals, focus management) lives in a pure Rust
`hyprstream-compositor` crate that compiles to both native x86_64/aarch64 and
`wasm32-wasip1` (for Wanix/browser). No wasmtime required for the CLI — the compositor
is a plain Rust library linked directly into the `hyprstream` binary.

## Layer Map

```
TuiService (server — hyprstream crate, no WASM)
  session registry · PTY lifecycle · pane frame publishing keyed by pane_id
        │
        │ ANSI frames per pane_id  (native: ZMQ XPUB | browser: WebTransport)
        ▼
hyprstream-compositor  (pure Rust — no I/O, no ZMQ, no tokio, no libc)
  LayoutTree · PaneState(avt::Vt) per pane_id
  ShellChrome: tab strip, model list, modals, focus
  CompositorInput / CompositorOutput / RpcRequest  (plain enums, WASM-safe)
        │
        │ CompositorOutput::Frame(Vec<u8>) → terminal / xterm.js
        │ CompositorOutput::Rpc(RpcRequest) → event loop dispatches
        │ CompositorOutput::RouteInput { app_id, data } → ChatApp
        ▼
Event loop
  native: shell_handlers.rs (tokio, ZMQ, hyprstream crate)
  WASM:   run_compositor_wasi() in hyprstream-tui main.rs (stdin frame loop)
  Owns:   ChatApp instances, TuiClient, PrivateStore, ChatFactory
  Drives: Compositor::handle(input) → dispatch Vec<CompositorOutput>
```

**Hard rule**: `hyprstream-compositor` has zero dependencies on ZMQ, tokio, libc,
hyprstream-rpc, or any I/O. It is a pure state machine.

## How the Native CLI Uses the Compositor

No WASM toolchain, no wasmtime. The compositor is a Rust library crate:

```
cargo build --release
  → links hyprstream-compositor into hyprstream binary
  → same binary as before; compositor replaces ShellClientState

hyprstream tui shell
  → shell_handlers.rs event loop
  → Compositor::handle(CompositorInput) → Vec<CompositorOutput>
  → CompositorOutput::Frame → AnsiBackend → terminal stdout
  → CompositorOutput::Rpc  → ShellRpcAdapter → ZMQ → TuiService
```

`hyprstream-tui` (standalone binary, used for Playwright testing) also drives the
compositor natively via `run_compositor()` in `main.rs`.

## How the Browser Uses the Compositor

The same `hyprstream-compositor` crate compiled to `wasm32-wasip1` runs as a Wanix
task. Wanix provides a WASI environment with fd 0/1/2 only (no network, no fd 3+).
All I/O passes through stdin/stdout via a framed protocol.

```
Browser
  TuiViewerClient (WebTransport) ─→ TerminalHost (TS)
  keyboard input                 ─→ TerminalHost
                                       │ framed stdin writes (9-byte header)
                                       ▼
                              compositor.wasm  (Wanix task)
                              run_compositor_wasi(): reassembly buffer loop
                              Compositor::handle() → CompositorOutput
                                       │ raw ANSI → stdout → xterm.js
                                       │ OSC IPC (0xFE/0xFF) → TerminalHost
                                       ▼
                              TerminalHost parses OSC → WebTransport RPC calls
```

## Framed Stdin Protocol (Phase W2)

All integers little-endian. Fixed 9-byte header: `[type:u8][id:u32][len:u32][data:len]`.

### Inbound (TerminalHost → compositor stdin)

| Type | id field | Data |
|------|----------|------|
| `0x01` | `pane_id` | ANSI bytes from TuiService |
| `0x02` | `0` | Keyboard bytes |
| `0x03` | `0` | `cols:u16 rows:u16 pad:4` (resize) |
| `0x04` | `app_id` | ANSI bytes from ChatApp task (W3) |
| `0x05` | `0` | JSON: `Vec<WindowSummary>` (window list update) |
| `0x06` | `pane_id` | *(empty)* — pane closed |
| `0x07` | `0` | *(empty)* — session reset / reconnect |

### Outbound (compositor stdout → TerminalHost)

Raw ANSI bytes flow directly to xterm.js. IPC is OSC-framed so TerminalHost can
strip it without a binary parser state machine:

```
ESC ] 0xFE len:u32 json_bytes ST   →  RpcRequest (serde_json) → WebTransport RPC
ESC ] 0xFF app_id:u32 len:u32 data ST  →  RouteInput to ChatApp Wanix task stdin
```

xterm.js passes unrecognised OSC sequences through harmlessly.

## Compositor API

```rust
// crates/hyprstream-compositor/src/lib.rs

pub struct Compositor { /* LayoutTree, panes, ShellChrome, focused, cols, rows */ }

pub enum PaneSource {
    Server { pane_id: u32 },   // frame from TuiService
    Private { app_id: u32 },   // rendered ANSI from event-loop-owned ChatApp
}

pub enum CompositorInput {
    ServerFrame { pane_id: u32, ansi: Vec<u8> },
    AppFrame    { app_id: u32,  ansi: Vec<u8> },
    WindowList(Vec<WindowSummary>),
    PaneClosed  { pane_id: u32 },
    KeyPress(KeyPress),
    Resize(u16, u16),
    AppExited   { app_id: u32 },
}

/// Pure typed enum — no ZMQ types, no capnp types, WASM-safe.
pub enum RpcRequest {
    SpawnShell        { session_id: u32, window_id: u32, cwd: Option<String> },
    FocusWindow       { session_id: u32, window_id: u32 },
    CreatePrivatePane { session_id: u32, window_id: u32, cols: u16, rows: u16, name: String },
    SpawnServerChat   { session_id: u32, window_id: u32, model_ref: String },
}

pub enum CompositorOutput {
    Frame(Vec<u8>),                             // composed ANSI → terminal / xterm.js
    RouteInput { app_id: u32, data: Vec<u8> }, // key bytes → ChatApp instance
    Rpc(RpcRequest),                            // dispatch to ShellRpcAdapter or OSC IPC
}

impl Compositor {
    pub fn new(cols: u16, rows: u16) -> Self;
    pub fn handle(&mut self, input: CompositorInput) -> Vec<CompositorOutput>;
}
```

## ChatApp Ownership

`ChatApp` is **never** stored inside the compositor. The compositor holds only
`avt::Vt` cell buffers. The event loop owns `ChatApp` instances in
`active_apps: HashMap<u32, ChatApp>`. This is required because:

- `mpsc::sync_channel` requires OS thread parking — unavailable on WASI.
- ZMQ/tokio dependencies in ChatApp would break WASM purity of the compositor.

The event loop renders each `ChatApp` into `AnsiBackend<Vec<u8>>` and feeds the
bytes as `CompositorInput::AppFrame { app_id, ansi }`. The compositor treats it
like any other ANSI frame.

## Server-Side: `PaneBackend` (not `PaneSource`)

To avoid naming collision with the compositor's `PaneSource`, the server-side
(`crates/hyprstream/src/tui/state.rs`) uses a separate enum:

```rust
pub enum PaneBackend {
    Managed,   // server-side process (default)
    Private,   // publish [PRIVATE] placeholder; no content
}
```

`PaneBackend::Private` causes `service.rs` to substitute a hardcoded `[PRIVATE]`
ANSI placeholder instead of diff-encoding real pane cells.

## UX

```
F10 → model list → C   → Private chat  (🔒, inference bytes never leave client)
                   ⇧C  → Server chat   (existing server-side path)
```

Tab strip shows `🔒 chat:model` for private panes. Other session viewers see a
centred, dimmed `[PRIVATE]` placeholder — no inference content.

## Private Chat Storage

Chat history is encrypted at rest using AES-256-GCM.

**Native**: storage key derived from signing key via HKDF-SHA256:
```
derive_key("hyprstream-storage-session-enc-v1", signing_key_seed) → Zeroizing<[u8;32]>
```
History files: `$XDG_DATA_HOME/hyprstream/private/v1/{session-uuid}.enc`
Wire format: `nonce(12B, OsRng) || AES-GCM(serde_json(history), aad=uuid_bytes)`
Atomic writes: write to `{uuid}.enc.tmp`, rename to `{uuid}.enc`.

**Browser (V1)**: Web Crypto non-extractable AES-GCM key generated per session.
Key is session-scoped (lost on page reload). TUI shows `[⚠ PRIVATE HISTORY: SESSION ONLY]`.
Cross-session persistence (V1+) deferred — requires PolicyService key-wrapping key.

## Browser Authentication

The root Ed25519 signing key never enters browser memory. Browser auth uses the
existing OAuth 2.1 PKCE flow:

```
Browser → OAuth 2.1 Authorization Code (PKCE) → hyprstream OAuth server
        → PolicyService issues JWT(sub, aud, exp)

Browser generates ephemeral Ed25519 keypair (wasm_api.rs::generate_signing_keypair())

Each RPC call:
  build_signed_envelope_with_token(payload, ephemeral_seed, eph_pub, req_id, jwt)
  → RequestIdentity::ApiToken { user: "alice", token_name: "jwt" }
  → Server verifies JWT → Casbin checks policy
```

Required Casbin rules for browser TUI users:
```
p, alice, *, tui:connect,   write, allow
p, alice, *, tui:private,   create, allow
p, alice, *, tui:session/*, write, allow
p, alice, *, inference:*,   infer, allow   # W3 only
```

## DRY Deletions (Phase 0)

| Deleted file | Replaced by | Lines |
|---|---|---|
| `crates/hyprstream-tui/src/shell_app.rs` | Compositor chrome | ~597 |
| `crates/hyprstream/src/tui/shell_ui.rs` | Unified `hyprstream-tui/src/shell_ui.rs` | ~400 |
| Fat `ShellClientState` in `shell_client.rs` | Thin `ShellRpcAdapter` | ~600 → ~30 |

Duplicate types eliminated: `ShellMode`, `ModelEntry`, `keypress_to_bytes()`,
`WindowSummary`, modal state — all consolidated into `hyprstream-compositor/src/chrome.rs`.

## Phase Sequence

```
Phase 0 (compositor crate + DRY — all-or-nothing, Playwright gate)
    ├─────────────────────────────────────────────────┐
    ↓                                                 ↓ Wanix track
Phase 1 (PaneBackend::Private + createPrivatePane)  Phase W1 (OPFS verification)
    ↓                                               Phase W2 (compositor as Wanix task)
Phase 2 (PrivateStore) ──┐                                ↓
Phase 3 (ChatApp hooks)  ├ parallel                Phase W3 (ChatApp isolated Wanix task)
Phase 4 (event loop owns ChatApp) ──┘
    ↓
Phase 5 (ChatFactory + make_chat_spawner)
    ↓
Phase 6 (C/⇧C key wiring)
    ↓
Phase 7 (🔒 chrome indicators)
    ↓
Phase 8 (remove server-side ChatApp path — after W3)
```

Phase 0 is all-or-nothing: Playwright tests boot via `SpawnChromeShell` → server-side
`ShellApp`. Removing `shell_app.rs` without a working compositor path breaks all 24 tests.
The compositor chrome must be fully functional in the same commit.

## Cargo Dependencies

```toml
# crates/hyprstream-compositor/Cargo.toml
[dependencies]
avt     = "0.17"
ratatui = { version = "0.30", default-features = false, features = ["layout-cache"] }
waxterm = { path = "../waxterm" }   # KeyPress type only
serde   = { version = "1", features = ["derive"] }
serde_json = "1"
```

No ZMQ, no tokio, no libc, no hyprstream-rpc. Compiles to `wasm32-wasip1` without
any platform-specific conditional compilation.

## Related Docs

- `docs/tui-display-server.md` — TuiService architecture, session/pane/window model
- `docs/cryptography-architecture.md` — Ed25519 signing, CURVE transport, JWT auth
- `docs/rpc-architecture.md` — Cap'n Proto over ZMQ, generate_rpc_service!, schema pipeline
