//! ChatApp — LLM chat terminal application.
//!
//! On native, implements [`waxterm::app::TerminalApp`] so it can be spawned via
//! `spawn_app_process` inside the TuiService frame loop.  The inference
//! background thread is injected as a [`StreamSpawner`] closure.
//!
//! On WASM, works as a push-based state machine: the host calls `on_token()`,
//! `on_stream_complete()`, etc. directly, and `submit_message_payload()` returns
//! the serialized JSON request for the host to forward.

use std::collections::HashMap;
use std::collections::VecDeque;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use waxterm::input::KeyPress;

#[cfg(not(target_os = "wasi"))]
use std::sync::Arc;
#[cfg(not(target_os = "wasi"))]
use std::sync::mpsc;
#[cfg(not(target_os = "wasi"))]
use std::time::Instant;
#[cfg(not(target_os = "wasi"))]
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
#[cfg(not(target_os = "wasi"))]
use ratatui_textarea::TextArea;
#[cfg(not(target_os = "wasi"))]
use waxterm::app::TerminalApp;

#[cfg(not(target_os = "wasi"))]
fn keypress_to_crossterm(key: KeyPress) -> Option<KeyEvent> {
    let (code, mods) = match key {
        KeyPress::Char(b) if b.is_ascii_graphic() || b == b' ' => {
            (KeyCode::Char(b as char), KeyModifiers::NONE)
        }
        KeyPress::Char(0x0A) => (KeyCode::Enter, KeyModifiers::NONE), // Ctrl-J
        KeyPress::Enter      => (KeyCode::Enter, KeyModifiers::NONE),
        KeyPress::Backspace  => (KeyCode::Backspace, KeyModifiers::NONE),
        KeyPress::ArrowLeft  => (KeyCode::Left, KeyModifiers::NONE),
        KeyPress::ArrowRight => (KeyCode::Right, KeyModifiers::NONE),
        KeyPress::ArrowUp    => (KeyCode::Up, KeyModifiers::NONE),
        KeyPress::ArrowDown  => (KeyCode::Down, KeyModifiers::NONE),
        KeyPress::Tab        => (KeyCode::Tab, KeyModifiers::NONE),
        // Ctrl-letter keys (0x01–0x1A, excluding already handled)
        KeyPress::Char(b) if b < 0x20 => {
            let ch = (b + b'a' - 1) as char;
            (KeyCode::Char(ch), KeyModifiers::CONTROL)
        }
        _ => return None,
    };
    Some(KeyEvent {
        code,
        modifiers: mods,
        kind: KeyEventKind::Press,
        state: KeyEventState::NONE,
    })
}


// ============================================================================
// Public types
// ============================================================================

/// Per-session inference configuration, adjustable via the Settings modal.
#[derive(Clone)]
pub struct ChatGenConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    /// None = use model default; Some(n) = override context window on next load.
    pub context_window: Option<usize>,
}

impl Default for ChatGenConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            context_window: None,
        }
    }
}

/// Tool-call output format used by a model family.
///
/// Determines which markers to scan for in the streaming token output.
/// Defined here (rather than importing from `hyprstream`) to keep this crate
/// free of the main binary's dependency graph.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallFormat {
    /// Qwen3 XML: `<tool_call>{"name":…,"arguments":…}</tool_call>`
    Qwen3Xml,
    /// Qwen3.5 XML parameter format:
    /// `<tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>`
    Qwen35XmlParam,
    /// Llama 3.1+: `<|python_tag|>` prefix + JSON
    LlamaJson,
    /// Mistral: `[TOOL_CALLS]` prefix + JSON array
    MistralJson,
    /// Model does not support tool calling (default).
    #[default]
    None,
}

/// Events sent from the inference background thread to `ChatApp::tick`.
#[cfg(not(target_os = "wasi"))]
pub enum ChatEvent {
    Token(String),
    StreamComplete,
    /// Inference was cancelled by the user — partial content was already written.
    StreamCancelled,
    StreamError(String),
    TemplateError(String),
    /// A complete tool call block was parsed from the model output.
    ToolCallDetected {
        /// Per-invocation correlation ID.
        id: String,
        uuid: String,
        description: String,
        arguments: String,
    },
    /// Result from executing a tool call.
    ToolCallResult {
        /// Per-invocation correlation ID (matches `ToolCallDetected::id`).
        id: String,
        uuid: String,
        result: String,
    },
}

/// Role of a chat history entry.
#[derive(Serialize, Deserialize, Clone)]
pub enum ChatRole {
    User,
    Assistant,
    /// Tool result injected back into the conversation.
    Tool,
}

/// A recorded tool call and its result (or in-flight state).
#[derive(Serialize, Deserialize, Clone)]
pub struct ToolCallRecord {
    /// Per-invocation correlation ID (UUID v4).  Used to match `ToolCallResult`
    /// events back to this record.  Two calls to the same tool have different IDs.
    #[serde(default)]
    pub id: String,
    /// The UUID name used on the wire (what the model outputs as `name`).
    pub uuid: String,
    /// Human-readable label fetched from `list_tools()`.
    pub description: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
    /// Execution result. `None` while the call is in-flight.
    pub result: Option<String>,
}

/// A single message in the conversation history.
#[derive(Serialize, Deserialize, Clone)]
pub struct ChatHistoryEntry {
    pub role: ChatRole,
    /// Visible response content (what gets sent back to the model as context).
    pub content: String,
    /// Reasoning / thinking tokens, hidden by default (not sent to model).
    #[serde(default)]
    pub thinking: String,
    /// Tool calls made by this assistant turn.  Non-empty only on Assistant messages.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    /// For Tool messages: the UUID of the corresponding assistant tool call.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// One-shot cancellation handle returned by `StreamSpawner`.
/// Calling it signals the background thread to abort the current stream.
#[cfg(not(target_os = "wasi"))]
pub type CancelHandle = Box<dyn FnOnce() + Send + 'static>;

/// Injected by `service.rs` — captures signing_key + model_ref.
///
/// Called with the current history and event channel every time the user
/// submits a message or a tool result re-triggers generation.
/// Returns a [`CancelHandle`] that aborts the stream.
/// Must be `Send + 'static` because it is moved into the background thread
/// spawned by `spawn_app_process`.
#[cfg(not(target_os = "wasi"))]
pub type StreamSpawner =
    Box<dyn Fn(Vec<ChatHistoryEntry>, mpsc::SyncSender<ChatEvent>) -> CancelHandle + Send + 'static>;

/// Tool executor injected at construction time.
///
/// Called with `(id, uuid, arguments_json, event_tx)` where `id` is the
/// per-invocation correlation ID to echo back in `ChatEvent::ToolCallResult`.
/// Spawns a background thread internally and sends `ChatEvent::ToolCallResult`
/// when done.
#[cfg(not(target_os = "wasi"))]
pub type ToolCaller =
    Arc<dyn Fn(String, String, String, mpsc::SyncSender<ChatEvent>) + Send + Sync + 'static>;

/// Called after `StreamComplete` to persist the current history.
pub type SaveHook = Box<dyn Fn(&[ChatHistoryEntry]) + Send + 'static>;

/// Called once at construction to restore prior history (returns `None` if no
/// history exists yet).
pub type LoadHook = Box<dyn FnOnce() -> Option<Vec<ChatHistoryEntry>> + Send + 'static>;

// ============================================================================
// App mode
// ============================================================================

pub enum ChatMode {
    Input,
    Streaming,
    #[cfg(not(target_os = "wasi"))]
    Editor,
    /// Settings modal: ↑/↓ select field, ←/→ adjust, Enter save, Esc discard.
    #[cfg(not(target_os = "wasi"))]
    Settings { selected_field: usize },
}

// ============================================================================
// ChatApp
// ============================================================================

/// Create a no-op waker for polling futures that don't need wake notifications.
///
/// Safe for TclShell/VFS futures which are purely computational (no IO reactor,
/// no timers) — they always make progress on each poll without needing external
/// wake-ups.
fn noop_waker() -> std::task::Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> std::task::RawWaker {
        std::task::RawWaker::new(p, &VTABLE)
    }
    static VTABLE: std::task::RawWakerVTable =
        std::task::RawWakerVTable::new(clone, noop, noop, noop);
    // SAFETY: The vtable functions are valid no-ops. The data pointer is never dereferenced.
    unsafe { std::task::Waker::from_raw(std::task::RawWaker::new(std::ptr::null(), &VTABLE)) }
}

/// Wrapper that asserts `Send` for a `!Send` value.
///
/// **Not thread-local storage** — this is a Send bypass for values that are
/// constructed and accessed on a single thread but must transit a thread boundary
/// wrapped in `Option<AssertSend<T>>` where the Option is `None` during the move.
///
/// SAFETY: This is used exclusively for `TclShell` which uses `Rc` internally
/// (making it `!Send`). The wrapper is safe because:
/// - The value is always wrapped in `Option` that is `None` during cross-thread moves
/// - It is only populated via `ensure_tcl_shell()` on the app thread
/// - It is only accessed from that same thread thereafter
/// - `UnsafeCell` ensures `AssertSend<T>` is `!Sync` (cannot be shared across threads)
struct AssertSend<T>(std::cell::UnsafeCell<T>);

// SAFETY: See struct-level documentation.
unsafe impl<T> Send for AssertSend<T> {}

impl<T> std::ops::Deref for AssertSend<T> {
    type Target = T;
    fn deref(&self) -> &T { unsafe { &*self.0.get() } }
}
impl<T> std::ops::DerefMut for AssertSend<T> {
    fn deref_mut(&mut self) -> &mut T { unsafe { &mut *self.0.get() } }
}

/// Double-Esc close window: two Esc presses within this interval close the app.
#[cfg(not(target_os = "wasi"))]
const DOUBLE_ESC_MS: u128 = 1500;

pub struct ChatApp {
    pub model_name: String,
    pub history: Vec<ChatHistoryEntry>,
    #[cfg(not(target_os = "wasi"))]
    pub textarea: TextArea<'static>,
    /// Simple input buffer for WASM (no TextArea available).
    #[cfg(target_os = "wasi")]
    pub input_buf: String,
    pub mode: ChatMode,
    /// Current text content held by the editor (set on Ctrl-E, read on Ctrl-S).
    #[cfg(not(target_os = "wasi"))]
    pub editor_text: String,
    /// Number of lines from the tail to scroll back (0 = show tail).
    pub scroll_offset: usize,
    pub status: Option<String>,
    #[cfg(not(target_os = "wasi"))]
    event_rx: mpsc::Receiver<ChatEvent>,
    #[cfg(not(target_os = "wasi"))]
    event_tx: mpsc::SyncSender<ChatEvent>,
    #[cfg(not(target_os = "wasi"))]
    spawner: StreamSpawner,
    pub cols: u16,
    pub rows: u16,
    pub quit: bool,
    /// UUID for this private session (used as the storage filename key).
    pub session_id: Option<Uuid>,
    /// Fired after `StreamComplete` to persist `history`.
    save_hook: Option<SaveHook>,
    /// True while currently inside a `<think>…</think>` block in the stream.
    in_thinking: bool,
    /// Whether to display thinking blocks expanded (Ctrl-O to toggle).
    pub show_thinking: bool,
    /// Error messages queued for the outer compositor to surface as toasts.
    /// Drained by the event loop each tick (see shell_handlers.rs).
    pub pending_toasts: Vec<String>,
    /// Cancel handle for the current in-flight stream. `None` when idle.
    #[cfg(not(target_os = "wasi"))]
    cancel_handle: Option<CancelHandle>,
    /// Prompts queued while an inference is in flight.
    pub pending_prompts: VecDeque<String>,
    /// Timestamp of the last Esc press (for double-Esc-to-close detection).
    #[cfg(not(target_os = "wasi"))]
    last_esc: Option<Instant>,

    // ── Generation config + settings modal ───────────────────────────────────

    /// Shared generation config (read by spawner each invocation).
    #[cfg(not(target_os = "wasi"))]
    pub gen_config: Arc<parking_lot::RwLock<ChatGenConfig>>,
    /// In-progress edits to generation config (committed on Enter, discarded on Esc).
    #[cfg(not(target_os = "wasi"))]
    pub settings_draft: ChatGenConfig,
    /// Set when user saves a new context_window; polled by shell_handlers to
    /// trigger model reload.  0 = use model default.
    #[cfg(not(target_os = "wasi"))]
    pub requested_context_window: Option<usize>,
    /// True when this ChatApp is hosted inside TuiService (server-spawned).
    /// In that mode, context window changes are surfaced as a toast rather than
    /// triggering an automatic model reload.
    #[cfg(not(target_os = "wasi"))]
    pub is_server_spawned: bool,

    // ── Tool calling ─────────────────────────────────────────────────────────

    /// Optional tool executor (None when MCP service is unavailable).
    #[cfg(not(target_os = "wasi"))]
    tool_caller: Option<ToolCaller>,
    /// UUID → human-readable description map built once from `list_tools()`.
    pub tool_descriptions: HashMap<String, String>,
    /// Format-specific markers to detect in the token stream.
    pub tool_call_format: ToolCallFormat,
    /// True while buffering inside a tool call block.
    in_tool_call: bool,
    /// Accumulated bytes inside the current tool call block.
    tool_call_buf: String,
    /// Number of tool calls dispatched but not yet resolved.
    #[cfg(not(target_os = "wasi"))]
    pub pending_tool_calls: usize,
    /// Spinner frame index — incremented each tick while tool calls are in flight.
    pub spinner_tick: u32,

    // ── VFS namespace for / command routing ──────────────────────────────────

    /// VFS namespace for / path commands. None on WASM.
    vfs: Option<std::sync::Arc<hyprstream_vfs::Namespace>>,
    /// Caller identity for VFS operations.
    vfs_subject: hyprstream_vfs::Subject,
    /// Tcl shell for / command evaluation. Created lazily on first use because
    /// TclShell is !Send (molt Value uses Rc) and ChatApp is moved across threads
    /// in spawn_app_process. The init data (vfs_tx + subject) IS Send.
    ///
    /// SAFETY: TclShell is wrapped in `AssertSend` (unsafe Send impl) because:
    /// 1. It is always `None` when ChatApp crosses a thread boundary
    /// 2. It is only constructed inside `ensure_tcl_shell()` on the app thread
    /// 3. It is only accessed from that same thread thereafter
    tcl_shell: Option<AssertSend<hyprstream_tcl::TclShell>>,
    /// Deferred init data for tcl_shell (Send-safe). Consumed on first access.
    tcl_shell_init: Option<(hyprstream_vfs::Subject, std::sync::Arc<hyprstream_vfs::Namespace>)>,
    /// Receiver for TclMount commands (polled in tick()). Shared across tabs.
    #[cfg(not(target_os = "wasi"))]
    tcl_mount_rx: Option<std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<hyprstream_tcl::TclCommand>>>>,
}

impl ChatApp {
    #[cfg(not(target_os = "wasi"))]
    pub fn new(model_name: String, cols: u16, rows: u16, spawner: StreamSpawner) -> Self {
        let (event_tx, event_rx) = mpsc::sync_channel::<ChatEvent>(256);
        Self {
            model_name,
            history: Vec::new(),
            textarea: Self::make_textarea(),
            mode: ChatMode::Input,
            editor_text: String::new(),
            scroll_offset: 0,
            status: None,
            event_rx,
            event_tx,
            spawner,
            cols,
            rows,
            quit: false,
            session_id: None,
            save_hook: None,
            in_thinking: false,
            show_thinking: false,
            pending_toasts: Vec::new(),
            cancel_handle: None,
            pending_prompts: VecDeque::new(),
            last_esc: None,
            gen_config: Arc::new(parking_lot::RwLock::new(ChatGenConfig::default())),
            settings_draft: ChatGenConfig::default(),
            requested_context_window: None,
            is_server_spawned: false,
            tool_caller: None,
            tool_descriptions: HashMap::new(),
            tool_call_format: ToolCallFormat::None,
            in_tool_call: false,
            tool_call_buf: String::new(),
            pending_tool_calls: 0,
            spinner_tick: 0,
            vfs: None,
            vfs_subject: hyprstream_vfs::Subject::anonymous(),
            tcl_shell: None,
            tcl_shell_init: None,
            #[cfg(not(target_os = "wasi"))]
            tcl_mount_rx: None,
        }
    }

    /// Create a WASM-compatible ChatApp (no StreamSpawner, push-based events).
    #[cfg(target_os = "wasi")]
    pub fn new_wasm(model_name: String, cols: u16, rows: u16) -> Self {
        Self {
            model_name,
            history: Vec::new(),
            input_buf: String::new(),
            mode: ChatMode::Input,
            scroll_offset: 0,
            status: None,
            cols,
            rows,
            quit: false,
            session_id: None,
            save_hook: None,
            in_thinking: false,
            show_thinking: false,
            pending_toasts: Vec::new(),
            pending_prompts: VecDeque::new(),
            tool_descriptions: HashMap::new(),
            tool_call_format: ToolCallFormat::None,
            in_tool_call: false,
            tool_call_buf: String::new(),
            spinner_tick: 0,
            vfs: None,
            vfs_subject: hyprstream_vfs::Subject::anonymous(),
            tcl_shell: None,
            tcl_shell_init: None,
            #[cfg(not(target_os = "wasi"))]
            tcl_mount_rx: None,
        }
    }

    #[cfg(not(target_os = "wasi"))]
    fn make_textarea() -> TextArea<'static> {
        use ratatui::style::{Color, Style};
        let mut ta = TextArea::default();
        ta.set_block(hyprstream_compositor::theme::input_block());
        ta.set_line_number_style(hyprstream_compositor::theme::gutter_style());
        ta.set_cursor_line_style(Style::default().bg(hyprstream_compositor::theme::BG).fg(Color::White));
        ta.set_style(Style::default().bg(hyprstream_compositor::theme::BG).fg(Color::White));
        ta.set_placeholder_text("Type a message…");
        ta
    }

    /// Create a private chat app with encrypted history persistence.
    ///
    /// `load_hook` is called immediately to restore any prior conversation.
    /// `save_hook` is called after each `StreamComplete` to persist history.
    #[cfg(not(target_os = "wasi"))]
    pub fn new_private(
        model_name: String,
        cols: u16,
        rows: u16,
        spawner: StreamSpawner,
        session_id: Uuid,
        load_hook: LoadHook,
        save_hook: SaveHook,
    ) -> Self {
        let (event_tx, event_rx) = mpsc::sync_channel::<ChatEvent>(256);
        let mut history = load_hook().unwrap_or_default();
        // Clean up tool call records whose result is None — these survive
        // serialization when the session crashes mid-execution.
        for entry in history.iter_mut() {
            for tc in entry.tool_calls.iter_mut() {
                if tc.result.is_none() {
                    tc.result = Some("[interrupted]".to_owned());
                }
            }
        }
        Self {
            model_name,
            history,
            textarea: Self::make_textarea(),
            mode: ChatMode::Input,
            editor_text: String::new(),
            scroll_offset: 0,
            status: None,
            event_rx,
            event_tx,
            spawner,
            cols,
            rows,
            quit: false,
            session_id: Some(session_id),
            save_hook: Some(save_hook),
            in_thinking: false,
            show_thinking: false,
            pending_toasts: Vec::new(),
            cancel_handle: None,
            pending_prompts: VecDeque::new(),
            last_esc: None,
            gen_config: Arc::new(parking_lot::RwLock::new(ChatGenConfig::default())),
            settings_draft: ChatGenConfig::default(),
            requested_context_window: None,
            is_server_spawned: false,
            tool_caller: None,
            tool_descriptions: HashMap::new(),
            tool_call_format: ToolCallFormat::None,
            in_tool_call: false,
            tool_call_buf: String::new(),
            pending_tool_calls: 0,
            spinner_tick: 0,
            vfs: None,
            vfs_subject: hyprstream_vfs::Subject::anonymous(),
            tcl_shell: None,
            tcl_shell_init: None,
            #[cfg(not(target_os = "wasi"))]
            tcl_mount_rx: None,
        }
    }

    /// Replace the default gen_config with a pre-created Arc (to share with the spawner).
    #[cfg(not(target_os = "wasi"))]
    pub fn with_gen_config(mut self, gen_config: Arc<parking_lot::RwLock<ChatGenConfig>>) -> Self {
        self.gen_config = gen_config;
        self
    }

    /// Mark this ChatApp as server-spawned (hosted inside TuiService).
    ///
    /// When true, context window changes from the Settings modal push a toast
    /// with CLI instructions rather than triggering an automatic model reload.
    #[cfg(not(target_os = "wasi"))]
    pub fn with_server_spawned(mut self) -> Self {
        self.is_server_spawned = true;
        self
    }

    /// Attach a VFS namespace for `/path` command routing.
    ///
    /// The TclShell is created lazily on first use (inside the app thread)
    /// because it is `!Send` and ChatApp must cross a thread boundary in
    /// `spawn_app_process`.
    pub fn with_vfs(mut self, ns: std::sync::Arc<hyprstream_vfs::Namespace>, subject: hyprstream_vfs::Subject) -> Self {
        self.tcl_shell_init = Some((subject.clone(), std::sync::Arc::clone(&ns)));
        self.vfs = Some(ns);
        self.vfs_subject = subject;
        self
    }

    /// Attach VFS namespace for `/path` command routing (compat shim).
    ///
    /// The `_vfs_tx` parameter is ignored — TclShell now awaits namespace
    /// operations directly. This method exists for call-site compatibility;
    /// prefer `with_vfs` for new code.
    pub fn with_vfs_proxy(
        mut self,
        ns: std::sync::Arc<hyprstream_vfs::Namespace>,
        subject: hyprstream_vfs::Subject,
        _vfs_tx: tokio::sync::mpsc::Sender<hyprstream_vfs::proxy::VfsRequest>,
    ) -> Self {
        self.tcl_shell_init = Some((subject.clone(), std::sync::Arc::clone(&ns)));
        self.vfs = Some(ns);
        self.vfs_subject = subject;
        self
    }

    /// Attach the receiver end of a `/lang/tcl` mount channel.
    ///
    /// The corresponding [`hyprstream_tcl::TclMount`] must be mounted in the
    /// namespace at `/lang/tcl` before this is called. The ChatApp drains this
    /// channel in `tick()`, forwarding commands to the Tcl interpreter.
    #[cfg(not(target_os = "wasi"))]
    pub fn with_tcl_mount_rx(mut self, rx: std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<hyprstream_tcl::TclCommand>>>) -> Self {
        self.tcl_mount_rx = Some(rx);
        self
    }

    /// Configure tool calling.  Call after construction if the MCP service is available.
    ///
    /// `tool_caller` executes a tool by UUID and sends `ToolCallResult` back.
    /// `tool_descriptions` maps UUID → human label for display purposes.
    /// `format` selects the model-specific tool call syntax.
    #[cfg(not(target_os = "wasi"))]
    pub fn with_tool_caller(
        mut self,
        tool_caller: ToolCaller,
        tool_descriptions: HashMap<String, String>,
        tool_call_format: ToolCallFormat,
    ) -> Self {
        self.tool_caller = Some(tool_caller);
        self.tool_descriptions = tool_descriptions;
        self.tool_call_format = tool_call_format;
        self
    }

    // ── Settings helpers ──────────────────────────────────────────────────────

    /// Adjust a single settings field in `settings_draft` by `delta` steps.
    ///
    /// Field indices:
    /// 0 = max_tokens  (step 64, range 64–32768)
    /// 1 = temperature (step 0.05, range 0.0–2.0)
    /// 2 = top_p       (step 0.05, range 0.05–1.0)
    /// 3 = top_k       (step 1, range None/off or 1–200)
    /// 4 = context_window (step 512, range None/"model default" or 512–131072)
    #[cfg(not(target_os = "wasi"))]
    fn apply_settings_field_delta(&mut self, field: usize, delta: i32) {
        match field {
            0 => {
                let v = self.settings_draft.max_tokens as i32 + delta * 64;
                self.settings_draft.max_tokens = v.clamp(64, 32768) as usize;
            }
            1 => {
                let v = self.settings_draft.temperature + delta as f32 * 0.05;
                let rounded = (v.clamp(0.0, 2.0) * 100.0).round() / 100.0;
                self.settings_draft.temperature = rounded;
            }
            2 => {
                let v = self.settings_draft.top_p + delta as f32 * 0.05;
                let rounded = (v.clamp(0.05, 1.0) * 100.0).round() / 100.0;
                self.settings_draft.top_p = rounded;
            }
            3 => match self.settings_draft.top_k {
                None if delta > 0 => self.settings_draft.top_k = Some(1),
                Some(1) if delta < 0 => self.settings_draft.top_k = None,
                Some(v) => {
                    self.settings_draft.top_k =
                        Some((v as i32 + delta).clamp(1, 200) as usize);
                }
                _ => {}
            },
            4 => match self.settings_draft.context_window {
                None if delta > 0 => self.settings_draft.context_window = Some(512),
                Some(512) if delta < 0 => self.settings_draft.context_window = None,
                Some(v) => {
                    self.settings_draft.context_window =
                        Some((v as i32 + delta * 512).clamp(512, 131072) as usize);
                }
                _ => {}
            },
            _ => {}
        }
    }

    // ── Tool call state machine helpers ──────────────────────────────────────

    /// Parse the buffered tool call content and return `(function_name, arguments_json)`.
    ///
    /// Tries JSON first (Hermes/nous format), then falls back to the Qwen3.5
    /// XML parameter format (`<function=NAME><parameter=KEY>value</parameter></function>`).
    fn parse_tool_call_buf(buf: &str) -> Option<(String, String)> {
        let trimmed = buf.trim();

        // Try JSON format first (Qwen3/Hermes: {"name": ..., "arguments": ...})
        if let Ok(call_data) = serde_json::from_str::<serde_json::Value>(trimmed) {
            let name = call_data["name"].as_str()?.to_owned();
            let arguments = serde_json::to_string(&call_data["arguments"]).ok()?;
            return Some((name, arguments));
        }

        // Fall back to Qwen3.5 XML parameter format:
        //   <function=get_weather>
        //   <parameter=location>NYC</parameter>
        //   </function>
        if trimmed.contains("<function=") {
            return Self::parse_xml_param_tool_call(trimmed);
        }

        None
    }

    /// Parse a single Qwen3.5 XML parameter-format tool call block.
    fn parse_xml_param_tool_call(buf: &str) -> Option<(String, String)> {
        // Extract function name from <function=NAME>
        let func_start = buf.find("<function=")?;
        let name_start = func_start + "<function=".len();
        let name_end = buf[name_start..].find('>')? + name_start;
        let name = buf[name_start..name_end].trim().to_owned();
        if name.is_empty() {
            return None;
        }

        // Extract the body between <function=NAME> and </function>
        let body_start = name_end + 1;
        let body_end = buf.find("</function>").unwrap_or(buf.len());
        let body = &buf[body_start..body_end];

        // Extract <parameter=KEY>VALUE</parameter> pairs
        let mut args = serde_json::Map::new();
        let mut search_from = 0;
        while let Some(param_start) = body[search_from..].find("<parameter=") {
            let abs_start = search_from + param_start;
            let key_start = abs_start + "<parameter=".len();
            let key_end = match body[key_start..].find('>') {
                Some(p) => key_start + p,
                None => break,
            };
            let key = body[key_start..key_end].trim().to_owned();

            let value_start = key_end + 1;
            let value_end = match body[value_start..].find("</parameter>") {
                Some(p) => value_start + p,
                None => break,
            };
            let value = body[value_start..value_end].trim();

            if !key.is_empty() {
                // Try JSON parse for numbers/bools/objects, fall back to string
                let json_val = serde_json::from_str::<serde_json::Value>(value)
                    .unwrap_or_else(|_| serde_json::Value::String(value.to_owned()));
                args.insert(key, json_val);
            }

            search_from = value_end + "</parameter>".len();
        }

        let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
        Some((name, arguments))
    }

    // ── ingest_token ─────────────────────────────────────────────────────────

    /// Route a streamed token into the correct buffer.
    ///
    /// Handles `<think>`/`</think>` and (when configured) tool-call markers.
    /// Returns a list of `(uuid, arguments_json)` for any complete tool call
    /// blocks that were just closed.  Callers are responsible for emitting
    /// `ChatEvent::ToolCallDetected` and firing the cancel handle.
    fn ingest_token(&mut self, token: &str) -> Vec<(String, String)> {
        let mut detected: Vec<(String, String)> = Vec::new();

        // Cache format-derived markers before mutably borrowing history.
        let open_marker: Option<(&'static str, &'static str)> = match self.tool_call_format {
            ToolCallFormat::Qwen3Xml      => Some(("<tool_call>", "</tool_call>")),
            ToolCallFormat::Qwen35XmlParam => Some(("<tool_call>", "</tool_call>")),
            ToolCallFormat::LlamaJson     => Some(("<|python_tag|>", "")),
            ToolCallFormat::MistralJson   => Some(("[TOOL_CALLS]", "")),
            ToolCallFormat::None          => None,
        };

        let Some(last) = self.history.last_mut() else { return detected };
        let mut remaining = token;

        while !remaining.is_empty() {
            if self.in_thinking {
                if let Some(end) = remaining.find("</think>") {
                    last.thinking.push_str(&remaining[..end]);
                    self.in_thinking = false;
                    remaining = &remaining[end + "</think>".len()..];
                } else {
                    last.thinking.push_str(remaining);
                    break;
                }
            } else if self.in_tool_call {
                let close = open_marker.map(|(_, c)| c).unwrap_or("");
                if close.is_empty() {
                    // Llama/Mistral: no closing tag — buffer everything.
                    self.tool_call_buf.push_str(remaining);
                    break;
                }
                if let Some(end) = remaining.find(close) {
                    self.tool_call_buf.push_str(&remaining[..end]);
                    remaining = &remaining[end + close.len()..];
                    self.in_tool_call = false;
                    let buf = std::mem::take(&mut self.tool_call_buf);
                    if let Some(pair) = Self::parse_tool_call_buf(&buf) {
                        detected.push(pair);
                    }
                } else {
                    self.tool_call_buf.push_str(remaining);
                    break;
                }
            } else if let Some((open, _)) = open_marker {
                if let Some(start) = remaining.find(open) {
                    last.content.push_str(&remaining[..start]);
                    self.in_tool_call = true;
                    self.tool_call_buf.clear();
                    remaining = &remaining[start + open.len()..];
                } else if let Some(start) = remaining.find("<think>") {
                    last.content.push_str(&remaining[..start]);
                    self.in_thinking = true;
                    remaining = &remaining[start + "<think>".len()..];
                } else {
                    last.content.push_str(remaining);
                    break;
                }
            } else if let Some(start) = remaining.find("<think>") {
                last.content.push_str(&remaining[..start]);
                self.in_thinking = true;
                remaining = &remaining[start + "<think>".len()..];
            } else {
                last.content.push_str(remaining);
                break;
            }
        }

        detected
    }

    // ── VFS helpers ──────────────────────────────────────────────────────────

    /// Display a system-level message in the chat history (VFS output, errors).
    fn push_system_message(&mut self, msg: &str) {
        self.history.push(ChatHistoryEntry {
            role: ChatRole::Assistant,
            content: msg.to_owned(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
        self.scroll_offset = 0;
    }

    /// Lazily initialize the TclShell. Must be called from the app thread
    /// (after the ChatApp has been moved into spawn_app_process).
    fn ensure_tcl_shell(&mut self) {
        if self.tcl_shell.is_none() {
            if let Some((subject, namespace)) = self.tcl_shell_init.take() {
                self.tcl_shell = Some(AssertSend(std::cell::UnsafeCell::new(
                    hyprstream_tcl::TclShell::new(subject, namespace),
                )));
            }
        }
    }

    /// Poll a `!Send` future to completion on the current thread.
    ///
    /// Works in both contexts: inside an existing tokio runtime (shell_handlers
    /// compositor loop) and on a bare OS thread (spawn_app_process). Uses a
    /// noop-waker poll loop — safe because TclShell/VFS futures are pure async
    /// (no IO reactor, no timers — just trait method dispatch).
    fn poll_local<F: std::future::Future>(fut: F) -> F::Output {
        let mut fut = std::pin::pin!(fut);
        let waker = noop_waker();
        let mut cx = std::task::Context::from_waker(&waker);
        loop {
            match fut.as_mut().poll(&mut cx) {
                std::task::Poll::Ready(val) => return val,
                std::task::Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    /// Handle a `/`-prefixed command by evaluating it as Tcl against the VFS.
    ///
    /// Bare paths (e.g. `/srv/model/status`) are treated as `cat /srv/model/status`.
    /// Commands (e.g. `/ls /srv/model`) are evaluated directly.
    fn handle_vfs_command(&mut self, input: &str) {
        self.ensure_tcl_shell();
        let Some(shell) = &mut self.tcl_shell else {
            self.push_system_message("VFS not available");
            return;
        };

        // Show the command in history.
        self.history.push(ChatHistoryEntry {
            role: ChatRole::User,
            content: input.to_owned(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });

        // Strip leading '/' and decide: bare path vs command.
        let stripped = input.trim_start_matches('/');
        let script = if !stripped.contains(' ') && stripped.contains('/') {
            // Bare path like /srv/model/status → cat it.
            format!("cat {input}")
        } else {
            // Command like /ls /srv or /help → eval as-is.
            stripped.to_owned()
        };

        let result = Self::poll_local(shell.eval(&script));
        match result {
            Ok(output) if !output.is_empty() => {
                self.push_system_message(&output);
            }
            Ok(_) => {} // Empty output — nothing to display.
            Err(e) => {
                self.push_system_message(&format!("error: {e}"));
            }
        }
    }

    // ── Submit helpers ────────────────────────────────────────────────────────

    /// Build history pairs (role, content) for the model, appending a new user
    /// message, and fire the spawner.  Stores the returned `CancelHandle`.
    #[cfg(not(target_os = "wasi"))]
    fn submit_message(&mut self, user_text: String) {
        self.history.push(ChatHistoryEntry {
            role: ChatRole::User,
            content: user_text,
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
        self.start_generation();
    }

    /// Push an empty assistant placeholder and start the spawner.
    ///
    /// Used by both `submit_message` (first turn) and `restart_generation`
    /// (agentic re-invoke after tool results).
    #[cfg(not(target_os = "wasi"))]
    fn start_generation(&mut self) {
        self.history.push(ChatHistoryEntry {
            role: ChatRole::Assistant,
            content: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });

        let handle = (self.spawner)(self.history.clone(), self.event_tx.clone());
        self.cancel_handle = Some(handle);
        self.mode = ChatMode::Streaming;
        self.status = Some("Generating\u{2026}".to_owned());
        self.scroll_offset = 0;
        self.in_thinking = false;
        self.in_tool_call = false;
        self.tool_call_buf.clear();
    }

    /// Signal the current stream to cancel.
    #[cfg(not(target_os = "wasi"))]
    fn cancel_stream(&mut self) {
        if let Some(cancel) = self.cancel_handle.take() {
            cancel();
        }
        self.status = Some("Cancelling\u{2026}".to_owned());
    }

    /// Called on StreamComplete / StreamCancelled from user action (not tool
    /// cancellation): persist, then drain the pending queue if any prompts
    /// were typed while we were streaming.
    fn on_stream_done(&mut self) {
        #[cfg(not(target_os = "wasi"))]
        { self.cancel_handle = None; }
        self.in_thinking = false;
        self.in_tool_call = false;
        self.tool_call_buf.clear();
        if let Some(_next_text) = self.pending_prompts.pop_front() {
            #[cfg(not(target_os = "wasi"))]
            self.submit_message(_next_text);
        } else {
            self.mode = ChatMode::Input;
            self.status = None;
        }
    }

    // ====================================================================
    // Push-based event methods (for WASM and direct callers)
    // ====================================================================

    /// Ingest a streamed token from an external event source.
    pub fn on_token(&mut self, token: &str) {
        let detected = self.ingest_token(token);
        #[cfg(not(target_os = "wasi"))]
        for (uuid, arguments) in detected {
            let description = self.tool_descriptions.get(&uuid).cloned()
                .unwrap_or_else(|| uuid.clone());
            // Generate a unique correlation ID per invocation so two calls to the
            // same tool function can be distinguished in ToolCallResult matching.
            let id = Uuid::new_v4().to_string();
            // Enqueue the event BEFORE cancelling.  The cancel handle is fired
            // in tick() when ToolCallDetected is dequeued, which guarantees that
            // StreamCancelled (sent by the spawner thread after it sees the cancel
            // signal) cannot arrive before ToolCallDetected in the event channel.
            let _ = self.event_tx.send(ChatEvent::ToolCallDetected {
                id,
                uuid,
                description,
                arguments,
            });
        }
    }

    /// Signal that the inference stream completed successfully.
    pub fn on_stream_complete(&mut self) {
        #[cfg(not(target_os = "wasi"))]
        {
            // Always persist — Tool messages are already in history at this point.
            if let Some(ref hook) = self.save_hook {
                hook(&self.history);
            }
            if self.pending_tool_calls > 0 {
                // Mid-agentic loop: generation ended (naturally or before the cancel
                // signal arrived).  Don't drain the prompt queue or transition to Input
                // mode — ToolCallResult will re-invoke the spawner when all results arrive.
                self.cancel_handle = None;
                self.in_thinking = false;
                self.in_tool_call = false;
                self.tool_call_buf.clear();
                return;
            }
        }
        #[cfg(target_os = "wasi")]
        {
            if let Some(ref hook) = self.save_hook {
                hook(&self.history);
            }
        }
        self.on_stream_done();
    }

    /// Signal that the inference stream was cancelled.
    pub fn on_stream_cancelled(&mut self) {
        #[cfg(not(target_os = "wasi"))]
        {
            // If we cancelled because of a tool call, don't reset mode — the
            // agentic dispatch in tick() will re-invoke the spawner.
            if self.pending_tool_calls > 0 {
                self.cancel_handle = None;
                self.in_thinking = false;
                self.in_tool_call = false;
                self.tool_call_buf.clear();
                return;
            }
        }
        self.on_stream_done();
    }

    /// Signal an inference error.
    pub fn on_stream_error(&mut self, msg: String) {
        self.pending_toasts.push(format!("Inference error: {msg}"));
        if let Some(last) = self.history.last_mut() {
            last.content = format!("[Error: {msg}]");
        }
        #[cfg(not(target_os = "wasi"))]
        { self.cancel_handle = None; }
        self.in_thinking = false;
        self.in_tool_call = false;
        self.tool_call_buf.clear();
        #[cfg(not(target_os = "wasi"))]
        {
            if self.pending_tool_calls > 0 {
                // Tool executor threads may still be running; discard queued prompts
                // to avoid starting a new generation against a corrupt history state.
                self.pending_prompts.clear();
            }
            self.pending_tool_calls = 0;
        }
        self.on_stream_done();
    }

    /// Signal a template application error.
    pub fn on_template_error(&mut self, msg: String) {
        // The last entry is always the empty assistant placeholder pushed by
        // start_generation().  Pop only that entry to preserve any Tool messages
        // from a preceding agentic round (so context is intact on retry).
        // Fall back to popping until User if the history shape is unexpected.
        let tail_is_empty_assistant = self.history.last().map(|e| {
            matches!(e.role, ChatRole::Assistant)
                && e.content.is_empty()
                && e.tool_calls.is_empty()
        }).unwrap_or(false);
        if tail_is_empty_assistant {
            self.history.pop();
        } else {
            while let Some(entry) = self.history.pop() {
                if matches!(entry.role, ChatRole::User) {
                    // Put the user message back — we only want to remove
                    // assistant/tool entries, not the user's input.
                    self.history.push(entry);
                    break;
                }
            }
        }
        self.pending_toasts.push(format!("Model error: {msg}"));
        #[cfg(not(target_os = "wasi"))]
        {
            self.cancel_handle = None;
            self.pending_tool_calls = 0;
        }
        self.in_thinking = false;
        self.in_tool_call = false;
        self.tool_call_buf.clear();
        self.mode = ChatMode::Input;
        self.status = Some(msg);
    }

    /// Build the inference request payload as a JSON string (for WASM OSC output).
    pub fn submit_message_payload(&mut self, user_text: String) -> String {
        self.history.push(ChatHistoryEntry {
            role: ChatRole::User,
            content: user_text,
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
        self.history.push(ChatHistoryEntry {
            role: ChatRole::Assistant,
            content: String::new(),
            thinking: String::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        });
        self.mode = ChatMode::Streaming;
        self.status = Some("Generating\u{2026}".to_owned());
        self.scroll_offset = 0;
        self.in_thinking = false;

        let messages: Vec<serde_json::Value> = self
            .history
            .iter()
            .filter(|e| !e.content.is_empty())
            .map(|e| {
                let role = match e.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                    ChatRole::Tool => "tool",
                };
                serde_json::json!({ "role": role, "content": e.content })
            })
            .collect();

        serde_json::json!({
            "model": self.model_name,
            "messages": messages,
            "stream": true,
        })
        .to_string()
    }

    /// Handle a keyboard key in WASM mode.
    #[cfg(target_os = "wasi")]
    pub fn handle_key(&mut self, key: KeyPress) -> Option<String> {
        if matches!(self.mode, ChatMode::Streaming) {
            match key {
                KeyPress::Escape | KeyPress::F(10) => { self.quit = true; }
                _ => {}
            }
            return None;
        }

        match key {
            KeyPress::Escape | KeyPress::F(10) => {
                self.quit = true;
                None
            }
            KeyPress::Enter => {
                if self.input_buf.is_empty() {
                    return None;
                }
                let user_msg = std::mem::take(&mut self.input_buf);
                Some(self.submit_message_payload(user_msg))
            }
            KeyPress::Backspace => {
                self.input_buf.pop();
                None
            }
            KeyPress::Char(b) if b.is_ascii_graphic() || b == b' ' => {
                self.input_buf.push(b as char);
                None
            }
            KeyPress::ArrowUp => {
                self.scroll_offset = self.scroll_offset.saturating_add(1);
                None
            }
            KeyPress::ArrowDown => {
                self.scroll_offset = self.scroll_offset.saturating_sub(1);
                None
            }
            _ => None,
        }
    }
}

// ============================================================================
// TerminalApp impl (native only)
// ============================================================================

#[cfg(not(target_os = "wasi"))]
impl TerminalApp for ChatApp {
    type Command = KeyPress;

    fn render(&self, frame: &mut ratatui::Frame) {
        crate::chat_ui::draw(frame, self);
    }

    fn handle_input(&mut self, key: KeyPress) -> bool {
        // Ctrl-O toggles thinking visibility in all modes.
        if matches!(key, KeyPress::Char(0x0F)) {
            self.show_thinking = !self.show_thinking;
            return true;
        }

        match self.mode {
            // ── Input mode ──────────────────────────────────────────────────
            ChatMode::Input => match key {
                KeyPress::Escape => {
                    // No-op. Use Ctrl+Space → Q to close via compositor menu.
                    false
                }
                KeyPress::Char(0x05) => {
                    self.editor_text = self.textarea.lines().join("\n");
                    self.mode = ChatMode::Editor;
                    true
                }
                KeyPress::Enter => {
                    let user_content = self.textarea.lines().join("\n");
                    let user_content = user_content.trim().to_owned();
                    if user_content.is_empty() {
                        return false;
                    }
                    self.textarea = Self::make_textarea();
                    self.last_esc = None;

                    // VFS path routing: /path → cat from namespace.
                    if user_content.starts_with('/') && !user_content.starts_with("//") {
                        self.handle_vfs_command(&user_content);
                        return true;
                    }

                    // Escape: // → / (send to model with leading slash).
                    let content = if let Some(stripped) = user_content.strip_prefix("//") {
                        format!("/{stripped}")
                    } else {
                        user_content
                    };

                    self.submit_message(content);
                    true
                }
                KeyPress::Backspace => {
                    self.textarea.delete_char();
                    true
                }
                KeyPress::Char(b) => {
                    if b.is_ascii_graphic() || b == b' ' {
                        self.textarea.insert_char(b as char);
                        true
                    } else if b == 0x0A {
                        self.textarea.insert_newline();
                        true
                    } else {
                        false
                    }
                }
                KeyPress::ArrowLeft => {
                    self.textarea.move_cursor(ratatui_textarea::CursorMove::Back);
                    true
                }
                KeyPress::ArrowRight => {
                    self.textarea.move_cursor(ratatui_textarea::CursorMove::Forward);
                    true
                }
                KeyPress::ArrowUp | KeyPress::ScrollUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(
                        if matches!(key, KeyPress::ScrollUp) { 3 } else { 1 },
                    );
                    true
                }
                KeyPress::ArrowDown | KeyPress::ScrollDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(
                        if matches!(key, KeyPress::ScrollDown) { 3 } else { 1 },
                    );
                    true
                }
                _ => false,
            },

            // ── Streaming mode — allow typing + queueing ─────────────────────
            ChatMode::Streaming => match key {
                KeyPress::Escape => {
                    // Don't attempt to cancel when tool executor threads are running —
                    // cancel_handle is already consumed and there's no way to abort them.
                    // The Esc hint is hidden from the fkey bar in this state.
                    if self.pending_tool_calls == 0 {
                        self.cancel_stream();
                    }
                    true
                }
                KeyPress::F(10) => {
                    self.quit = true;
                    true
                }
                KeyPress::Enter => {
                    let user_content = self.textarea.lines().join("\n");
                    let user_content = user_content.trim().to_owned();
                    if user_content.is_empty() {
                        return false;
                    }
                    self.textarea = Self::make_textarea();
                    self.pending_prompts.push_back(user_content);
                    true
                }
                KeyPress::Backspace => {
                    self.textarea.delete_char();
                    true
                }
                KeyPress::Char(b) => {
                    if b.is_ascii_graphic() || b == b' ' {
                        self.textarea.insert_char(b as char);
                        true
                    } else if b == 0x0A {
                        self.textarea.insert_newline();
                        true
                    } else {
                        false
                    }
                }
                KeyPress::ArrowLeft => {
                    self.textarea.move_cursor(ratatui_textarea::CursorMove::Back);
                    true
                }
                KeyPress::ArrowRight => {
                    self.textarea.move_cursor(ratatui_textarea::CursorMove::Forward);
                    true
                }
                KeyPress::ArrowUp | KeyPress::ScrollUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(
                        if matches!(key, KeyPress::ScrollUp) { 3 } else { 1 },
                    );
                    true
                }
                KeyPress::ArrowDown | KeyPress::ScrollDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(
                        if matches!(key, KeyPress::ScrollDown) { 3 } else { 1 },
                    );
                    true
                }
                _ => false,
            },

            // ── Settings mode ─────────────────────────────────────────────────
            ChatMode::Settings { selected_field } => match key {
                KeyPress::Escape => {
                    self.mode = ChatMode::Input;
                    true
                }
                KeyPress::Enter => {
                    let draft = self.settings_draft.clone();
                    let old_context = self.gen_config.read().context_window;
                    *self.gen_config.write() = draft.clone();
                    if draft.context_window != old_context {
                        if self.is_server_spawned {
                            let ctx_str = draft.context_window
                                .map_or("default".to_owned(), |n| n.to_string());
                            self.pending_toasts.push(format!(
                                "Context window set to {ctx_str}. Reload model to apply: \
                                 hyprstream quick load {} --max-context {ctx_str}",
                                self.model_name,
                            ));
                        } else {
                            // Shell_handlers will reload the model on the next bg_tick.
                            self.requested_context_window = Some(draft.context_window.unwrap_or(0));
                        }
                    }
                    self.mode = ChatMode::Input;
                    true
                }
                KeyPress::ArrowUp => {
                    if selected_field > 0 {
                        self.mode = ChatMode::Settings { selected_field: selected_field - 1 };
                    }
                    true
                }
                KeyPress::ArrowDown => {
                    if selected_field < 4 {
                        self.mode = ChatMode::Settings { selected_field: selected_field + 1 };
                    }
                    true
                }
                KeyPress::ArrowLeft => {
                    self.apply_settings_field_delta(selected_field, -1);
                    true
                }
                KeyPress::ArrowRight => {
                    self.apply_settings_field_delta(selected_field, 1);
                    true
                }
                _ => false,
            },

            // ── Editor mode ──────────────────────────────────────────────────
            ChatMode::Editor => match key {
                KeyPress::Escape => {
                    self.editor_text = String::new();
                    self.mode = ChatMode::Input;
                    true
                }
                KeyPress::Char(0x13) => {
                    let content = crate::chat_ui::take_editor_text();
                    self.textarea = Self::make_textarea();
                    for (i, line) in content.lines().enumerate() {
                        if i > 0 {
                            self.textarea.insert_newline();
                        }
                        for ch in line.chars() {
                            self.textarea.insert_char(ch);
                        }
                    }
                    self.editor_text = String::new();
                    self.mode = ChatMode::Input;
                    true
                }
                _ => {
                    if let Some(crossterm_key) = keypress_to_crossterm(key) {
                        crate::chat_ui::EDITOR_STATE.with(|cell| {
                            if let Some(ref mut state) = *cell.borrow_mut() {
                                edtui::EditorEventHandler::default()
                                    .on_key_event(crossterm_key, state);
                            }
                        });
                        true
                    } else {
                        false
                    }
                }
            },
        }
    }

    fn tick(&mut self, _delta_ms: u64) -> bool {
        // Clear the single-Esc hint if it's been sitting long enough.
        if let Some(prev) = self.last_esc {
            if prev.elapsed().as_millis() >= DOUBLE_ESC_MS {
                self.last_esc = None;
                if self.status.as_deref() == Some("Press Esc again to close") {
                    self.status = None;
                }
            }
        }

        let mut redraw = false;

        // Process pending TclMount requests from /lang/tcl VFS mount.
        if self.tcl_mount_rx.is_some() {
            self.ensure_tcl_shell();
            if let (Some(ref rx_arc), Some(ref mut shell)) =
                (&self.tcl_mount_rx, &mut self.tcl_shell)
            {
                if let Ok(mut rx) = rx_arc.try_lock() {
                    while let Ok(cmd) = rx.try_recv() {
                        Self::poll_local(shell.process_command(cmd));
                    }
                }
            }
        }

        // Advance spinner while tool calls are in flight.
        if self.pending_tool_calls > 0 {
            self.spinner_tick = self.spinner_tick.wrapping_add(1);
            redraw = true;
        }

        loop {
            match self.event_rx.try_recv() {
                Ok(ChatEvent::Token(s)) => {
                    self.on_token(&s);
                    redraw = true;
                }
                Ok(ChatEvent::StreamComplete) => {
                    self.on_stream_complete();
                    redraw = true;
                }
                Ok(ChatEvent::StreamCancelled) => {
                    self.on_stream_cancelled();
                    redraw = true;
                }
                Ok(ChatEvent::StreamError(s)) => {
                    self.on_stream_error(s);
                    redraw = true;
                }
                Ok(ChatEvent::TemplateError(s)) => {
                    self.on_template_error(s);
                    redraw = true;
                }
                Ok(ChatEvent::ToolCallDetected { id, uuid, description, arguments }) => {
                    // Security: only dispatch tools present in the allowlist fetched
                    // from list_tools().  Models cannot invoke arbitrary endpoints.
                    let allowed = self.tool_caller.is_none()
                        || self.tool_descriptions.contains_key(&uuid);

                    // Record the tool call on the last assistant entry.
                    if let Some(last) = self.history.last_mut() {
                        last.tool_calls.push(ToolCallRecord {
                            id: id.clone(),
                            uuid: uuid.clone(),
                            description: description.clone(),
                            arguments: arguments.clone(),
                            result: if allowed { None } else {
                                Some(format!("[Tool not found: {uuid}]"))
                            },
                        });
                    }
                    self.pending_tool_calls += 1;

                    // Cancel the stream NOW — after ToolCallDetected is already
                    // dequeued from the channel.  This guarantees that StreamCancelled
                    // (sent by the spawner on the other side of the oneshot) arrives
                    // AFTER this event, never before it.
                    if let Some(cancel) = self.cancel_handle.take() {
                        cancel();
                    }

                    if !allowed {
                        // Unknown UUID — send error result immediately so the loop completes.
                        let _ = self.event_tx.send(ChatEvent::ToolCallResult {
                            id,
                            uuid,
                            result: "[Tool not found]".to_string(),
                        });
                    } else if let Some(ref caller) = self.tool_caller {
                        // Dispatch to the tool executor.
                        let caller_arc = Arc::clone(caller);
                        let tx = self.event_tx.clone();
                        let id_c = id.clone();
                        let uuid_c = uuid.clone();
                        let args_c = arguments.clone();
                        std::thread::spawn(move || {
                            caller_arc(id_c, uuid_c, args_c, tx);
                        });
                    } else {
                        // No MCP client — emit error result immediately.
                        let _ = self.event_tx.send(ChatEvent::ToolCallResult {
                            id,
                            uuid,
                            result: "[Tool calling not available]".to_owned(),
                        });
                    }
                    redraw = true;
                }
                Ok(ChatEvent::ToolCallResult { id, uuid: _, result }) => {
                    // Guard: ignore stale/duplicate results (e.g. after on_stream_error
                    // reset pending_tool_calls to 0, or a duplicate delivery).
                    if self.pending_tool_calls == 0 {
                        redraw = true;
                        continue;
                    }

                    // Fill in the result on the matching ToolCallRecord by correlation ID.
                    for entry in self.history.iter_mut() {
                        for tc in entry.tool_calls.iter_mut() {
                            if tc.id == id {
                                tc.result = Some(result.clone());
                                break;
                            }
                        }
                    }

                    // Sanitize tool result: strip tool-call markers so a compromised
                    // MCP service cannot inject fake tool calls into the next turn.
                    let sanitized_result = result
                        .replace("<tool_call>", "")
                        .replace("</tool_call>", "")
                        .replace("<|python_tag|>", "")
                        .replace("[TOOL_CALLS]", "");

                    // Inject a Tool message into history.  `tool_call_id` must match
                    // `ToolCall.id` in the preceding assistant message — both use the
                    // per-invocation correlation ID, not the tool-name UUID.
                    self.history.push(ChatHistoryEntry {
                        role: ChatRole::Tool,
                        content: sanitized_result,
                        thinking: String::new(),
                        tool_calls: Vec::new(),
                        tool_call_id: Some(id),
                    });

                    // Snap scroll to bottom so the newly-arrived result is visible.
                    self.scroll_offset = 0;

                    self.pending_tool_calls -= 1;

                    // When all pending tool calls are resolved, re-invoke the spawner.
                    if self.pending_tool_calls == 0 {
                        self.start_generation();
                    }
                    redraw = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if matches!(self.mode, ChatMode::Streaming) {
                        self.cancel_handle = None;
                        self.in_thinking = false;
                        self.in_tool_call = false;
                        self.tool_call_buf.clear();
                        self.pending_tool_calls = 0;
                        self.on_stream_done();
                        redraw = true;
                    }
                    break;
                }
            }
        }
        redraw
    }

    fn should_quit(&self) -> bool {
        self.quit
    }
}
