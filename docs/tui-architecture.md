# TUI Architecture

The collaborative TUI is split across a server-side display server and a
client-side compositor:

- **Server**: `TuiService` (`crates/hyprstream/src/tui/`) — a terminal
  multiplexer with session persistence, multi-pane layouts, remote viewers,
  and a 9P control filesystem. Native only (tokio, PTYs, RPC).
- **Client**: `hyprstream-compositor` — a pure-Rust, WASM-safe state machine
  for the shell chrome (tab strip, model list, modals, focus). The same crate
  is linked into the native CLI and compiled to `wasm32-wasip1` for the
  browser (Wanix).
- **Apps**: `hyprstream-tui` (wizard, chat, shell, console, container apps),
  `waxterm` (run ratatui apps natively and in the browser via WASI), and
  `chat-core` (shared chat orchestration state machine).

```
App (ratatui)  ──→  TuiPaneBackend  ──→  TuiPane (cell buffer)
                                               │
                    Frame loop (33ms) ─────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
       ANSI viewer   capnp viewer  structured viewer
       (terminal)    (WebTransport) (StructuredBackend)
```

## Server Side: TuiService

### Service registration and delivery

`TuiService` is a registered service
(`#[service_factory("tui", schema = "../../schema/tui.capnp", depends_on = ["policy", "discovery"])]`
in `crates/hyprstream/src/services/factories.rs`). The factory:

- calls `init_local_moq_stream_plane("tui")` so the process has a MoQ
  streaming plane even in per-process deployments (idempotent) — terminal
  frames and stdin are published over MoQ via `StreamChannel::publisher()`,
  and the service returns its per-PID MoQ UDS path to the client;
- builds the VFS namespace used by RPC-spawned ChatApps
  (`tui::vfs::build_chat_vfs_namespace`);
- wraps the service with `ctx.into_spawnable_quic(...)`, so one `Spawnable`
  runs both the Cap'n Proto request/reply endpoint and a QUIC endpoint. The
  QUIC endpoint accepts WebTransport connections from browser viewers
  (`tui/wt_viewer.rs` handles the long-lived bidirectional streams).

In `crates/hyprstream/src/tui/service.rs`, per-viewer frame publishing goes
through `AnyStreamPublisher` (`hyprstream_rpc::moq_stream`), which wraps a
`MoqStreamPublisher` on the MoQ path. Viewers in `Capnp` display mode also
get a dedicated stdin publisher.

### Session / window / pane model (`tui/state.rs`)

```
TuiState
  └── TuiSession            # persists with zero viewers
        └── TuiWindow       # LayoutNode tree of H/V splits
              └── TuiPane   # primary + lazy alternate PaneBuffer,
                            # scrollback, cursor, ingestion mode
```

Each pane also has a server-side `PaneBackend`
(`crates/hyprstream/src/tui/state.rs:186`):

- `Managed` — the server owns the PTY / process (default).
- `Private` — content is owned by the client; the server publishes a
  `[PRIVATE]` placeholder frame to other viewers instead of real cell data.
  This backs the client-local private chat path (see below).

### Three ingestion modes per pane

| Mode | Source | Path |
|------|--------|------|
| `Direct` | In-process ratatui apps | `TuiPaneBackend` (`tui/backend.rs`) writes cells straight into the pane buffer |
| `Structured` | waxterm WASI apps | packed binary cell tuples from waxterm's `StructuredBackend`, deserialized without a VTE round-trip |
| `Ansi` | Raw PTY programs | byte stream parsed by the VTE parser (`tui/vte_parser.rs`); PTYs spawned via `tui/process.rs` |

A 33ms frame loop diffs pane buffers (`tui/diff.rs`) and encodes once per
distinct viewer display mode (ANSI for terminals, Cap'n Proto `TuiFrame` for
the native compositor and WebTransport viewers).

### RPC surface (`crates/hyprstream/schema/tui.capnp`)

Methods on `TuiRequest`: `connect` / `disconnect` (viewer lifecycle),
`createWindow` / `closeWindow` / `listWindows` / `focusWindow`,
`splitPane` / `closePane` / `focusPane`, `snapshot`, `resize`, `sendInput`,
`pollStdin`, and the spawn family:

- `spawnShell` — spawn a shell process (PTY) in a pane.
- `spawnChromeShell` — spawn the `ShellApp` chrome renderer inside
  TuiService, so no fork occurs in the client process.
- `spawnChatApp` — spawn a server-side ChatApp inference pane connected to a
  loaded model.
- `createPrivatePane` — create a pane whose content is client-owned; the
  server publishes only the `[PRIVATE]` placeholder to other viewers.

Methods carry `$scope(...)` and `$mcpDescription(...)` annotations, making
the display MCP-controllable.

### 9P control filesystem (`tui/ninep.rs`)

TUI state is exposed as a standalone 9P tree (Plan 9 rio style — separate
from the git-backed registry 9P):

```
/tui/
  ctl             # write: global commands
  new             # walk atomically allocates a window (Plan 9 /dev/new);
                  # read returns the window ID
  event           # read blocks until an event is available
  windows/
    0/
      ctl         # window control
      cons        # read: ANSI stream (ring buffer); write: keyboard input
      screen      # framebuffer snapshot
      winname     # window title
      layout      # pane tree description
      panes/
        0/cons, screen
```

`cons` reads stream from a fixed ring buffer (`ConsBuffer`) with monotonic
offsets and gap detection.

### Module map (`crates/hyprstream/src/tui/`)

| Module | Purpose |
|--------|---------|
| `state.rs` | `TuiState`/`TuiSession`/`TuiWindow`/`TuiPane`, `PaneBackend`, `IngestionMode`, layout tree |
| `diff.rs` | Frame diff engine (incremental vs full frames, scroll ops) |
| `service.rs` | `TuiService`: RPC handlers, frame loop, viewer management, MoQ publishing |
| `wt_viewer.rs` | WebTransport viewer sessions over the QUIC endpoint |
| `backend.rs` | `TuiPaneBackend` — ratatui `Backend` writing into a pane (Direct mode) |
| `vte_parser.rs` | VTE parser for ANSI-mode panes |
| `process.rs` | PTY / app process spawning (`spawn_pty_process`, `spawn_app_process`) |
| `ninep.rs` | The `/tui` 9P filesystem |
| `shell_client.rs` | Thin typed wrappers over the generated `TuiClient` (spawnShell, pollStdin, …) |
| `vfs.rs` | VFS namespace construction for RPC-spawned ChatApps (`/bin`, `/env`, `/srv/model`, `/lang/tcl`) + dedicated-thread VFS proxy |
| `rpc_transport.rs` | `make_chat_spawner` / `make_tool_caller` — RPC-backed inference streaming for ChatApp |

## Client Side: hyprstream-compositor

### Purity rule

`hyprstream-compositor` is a plain Rust library with **zero** I/O
dependencies — no tokio, no libc, no RPC types. Its only dependencies are
`avt`, `ratatui`, `waxterm` (types), and serde. It compiles unchanged to
native targets and `wasm32-wasip1`. It is a pure state machine:

```rust
impl Compositor {
    pub fn handle(&mut self, input: CompositorInput) -> Vec<CompositorOutput>;
}
```

The chrome (tab strip, model list, settings, service/worker manager modals,
toasts, console log) lives in `ShellChrome`
(`crates/hyprstream-compositor/src/chrome.rs`) and emits `ChromeOutput`,
which the compositor lifts into `CompositorOutput`.

### Inputs and outputs

`CompositorInput` (`lib.rs`):

- `ServerFrame { pane_id, ansi }` — ANSI frame from TuiService (WASM path)
- `ServerFrameCapnp { frame }` — decoded `TuiFrame` (native CLI, Capnp mode)
- `AppFrame { app_id, ansi }` — rendered ANSI from a client-owned ChatApp
- `WindowList(...)`, `PaneClosed`, `AppExited`
- `KeyPress(...)`, `MouseClick { col, row }`, `Resize(cols, rows)`
- `ServiceList(...)`, `WorkerList { sandboxes, pool_summary }`,
  `WorkerImageList { images }` — polled state for the manager modals

`CompositorOutput`:

- `Redraw` — state changed; the event loop re-renders
- `Rpc(RpcRequest)` — dispatch to the transport (native RPC or OSC IPC)
- `RouteInput { app_id, data }` — key bytes for a client-owned ChatApp
- `Quit`

There is no `Frame` output variant — the compositor never produces bytes for
the terminal; rendering is driven by the event loop on `Redraw`.

`RpcRequest` (`chrome.rs:371`) is a pure serde enum with several dozen
variants, grouped by concern:

- **Input / session**: `SendInput`, window create/close/focus, `SpawnShell`
- **Chat**: `SpawnServerChat`, `LocalPrivateChat` (client-local private chat,
  optional conversation resume), `DeleteConversation`, `ListConversations`,
  `LoadModel` / `UnloadModel`
- **Service management**: start/stop/restart/install, start-all/stop-all
- **Worker / container management**: sandbox create/destroy, container
  create/start/stop/remove/attach/exec, image list/pull/remove
- **Model repository**: `CloneModel`, `PullModel`, `PushModel`,
  `FetchModelStatus`
- `Quit`

### Private chat and ChatApp ownership

`RpcRequest::LocalPrivateChat` starts a chat whose inference bytes never
leave the client. The server side is a `createPrivatePane` RPC, giving the
pane `PaneBackend::Private` (other viewers see the `[PRIVATE]` placeholder).

`ChatApp` is **never** stored inside the compositor. The event loop owns the
`ChatApp` instances; the compositor holds only `avt::Vt` cell buffers per
pane. The event loop renders each ChatApp to ANSI and feeds it back as
`CompositorInput::AppFrame`, and routes keys to it via
`CompositorOutput::RouteInput`.

Shared chat logic — `ChatState`, streaming-token handling, tool-call
detection/parsing (`ToolCallStyle`), the agentic loop — lives in the
`chat-core` crate, which compiles for both native and WASI.

Encrypted private-chat history persistence is handled by
`hyprstream-tui/src/private_store.rs` (AES-256-GCM at rest; key derived from
the signing key via HKDF on native).

### Entry points (`crates/hyprstream-tui/src/main.rs`)

The standalone `hyprstream-tui` binary dispatches on its first argument:

- `shell` → `run_shell`
- `wizard` → `run_wizard` (default on native)
- `chat` → `run_chat_wasi_v2`
- default on `wasm32-wasip1` → `run_compositor_wasi` — the framed-stdin event
  loop that drives the compositor as a Wanix task (fd 0/1/2 only; RPC
  requests leave via OSC-framed IPC on stdout)

On the native CLI, `hyprstream tui shell` drives the same compositor from
the event loop in `crates/hyprstream/src/cli/shell_handlers.rs`, dispatching
`RpcRequest`s over the real RPC transport.

### Inference spawner

`crate::tui::rpc_transport::make_chat_spawner` builds the `StreamSpawner` a
ChatApp uses to run inference: per user message it spawns a dedicated OS
thread with a single-threaded tokio runtime, applies the chat template
(including the tool list), opens an authenticated inference stream via
`ModelClient`, and forwards `ChatEvent`s back over a channel.
`make_tool_caller` provides the matching tool-execution hook. Both the CLI
shell event loop and `TuiService` (for `spawnChatApp`) use this module.

### Dedicated-thread async shell drivers

`crates/hyprstream-tui/src/shell_driver.rs` hosts `!Send` language shells
(the molt-based `TclShell`) on one dedicated OS thread each (`ShellOwner`),
running a single-threaded tokio runtime with a `LocalSet`. Slash-command VFS
futures that legitimately block (e.g. reading an `/exec/.../exit` file that
waits on worker completion) park the thread with a real waker instead of
hot-spinning a noop-waker poll loop. `NamespaceDriver` provides the same
pattern for direct `Namespace` access without an interpreter. The blocking
`eval()` leg uses `std::sync::mpsc` so it is safe both on bare OS threads
(TuiService-hosted apps) and inside a live tokio runtime (the CLI shell).

`shell_app.rs` (the server-hosted `ShellApp` chrome renderer, used by
`spawnChromeShell`) still exists alongside the compositor.

### waxterm

`waxterm` (`crates/waxterm/`) is the framework crate for running ratatui
apps natively and in the browser: the `TerminalApp` trait, `AnsiBackend`
(direct ANSI output), `StructuredBackend` (`structured.rs` — packed binary
cell tuples for Structured-mode ingestion into TuiService panes), input
parsing, and shared SGR encoding in `sgr.rs` used by both the ANSI backend
and the server's frame encoder.

## Related Docs

- `docs/vfs.md` — the Plan 9 VFS namespace the TUI shell operates on
- `docs/rpc-architecture.md` — Cap'n Proto RPC, transports, `generate_rpc_service!`
- `docs/cryptography-architecture.md` — signed envelopes, JWT auth
- `docs/streaming-service-architecture.md` — the MoQ streaming plane
