//! ChatApp — LLM chat terminal application.
//!
//! On native, implements [`waxterm::app::TerminalApp`] so it can be spawned via
//! `spawn_app_process` inside the TuiService frame loop.  The inference
//! background thread is injected as a [`StreamSpawner`] closure.
//!
//! On WASM, works as a push-based state machine: the host calls `on_token()`,
//! `on_stream_complete()`, etc. directly, and `submit_message_payload()` returns
//! the serialized JSON request for the host to forward.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use waxterm::input::KeyPress;

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

/// Events sent from the inference background thread to `ChatApp::tick`.
#[cfg(not(target_os = "wasi"))]
pub enum ChatEvent {
    Token(String),
    StreamComplete,
    /// Inference was cancelled by the user — partial content was already written.
    StreamCancelled,
    StreamError(String),
    TemplateError(String),
}

/// Role of a chat history entry.
#[derive(Serialize, Deserialize, Clone)]
pub enum ChatRole {
    User,
    Assistant,
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
}

/// One-shot cancellation handle returned by `StreamSpawner`.
/// Calling it signals the background thread to abort the current stream.
#[cfg(not(target_os = "wasi"))]
pub type CancelHandle = Box<dyn FnOnce() + Send + 'static>;

/// Injected by `service.rs` — captures signing_key + model_ref.
///
/// Called with `(history_as_role_content_pairs, event_tx)` every time the
/// user submits a message.  Returns a [`CancelHandle`] that aborts the stream.
/// Must be `Send + 'static` because it is moved into the background thread
/// spawned by `spawn_app_process`.
#[cfg(not(target_os = "wasi"))]
pub type StreamSpawner =
    Box<dyn Fn(Vec<(String, String)>, mpsc::SyncSender<ChatEvent>) -> CancelHandle + Send + 'static>;

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
}

// ============================================================================
// ChatApp
// ============================================================================

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
    /// Each entry is the raw user text; it will be appended to history and
    /// dispatched as the next inference once the current stream completes.
    pub pending_prompts: VecDeque<String>,
    /// Timestamp of the last Esc press (for double-Esc-to-close detection).
    #[cfg(not(target_os = "wasi"))]
    last_esc: Option<Instant>,
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
        let history = load_hook().unwrap_or_default();
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
        }
    }

    /// Route a streamed token into `entry.thinking` or `entry.content` based on
    /// `<think>`/`</think>` markers.  Handles markers appearing mid-token.
    fn ingest_token(&mut self, token: &str) {
        let Some(last) = self.history.last_mut() else { return };
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
            } else if let Some(start) = remaining.find("<think>") {
                last.content.push_str(&remaining[..start]);
                self.in_thinking = true;
                remaining = &remaining[start + "<think>".len()..];
            } else {
                last.content.push_str(remaining);
                break;
            }
        }
    }

    /// Build history pairs (role, content) for the model, appending a new user
    /// message, and fire the spawner.  Stores the returned `CancelHandle`.
    #[cfg(not(target_os = "wasi"))]
    fn submit_message(&mut self, user_text: String) {
        self.history.push(ChatHistoryEntry {
            role: ChatRole::User,
            content: user_text,
            thinking: String::new(),
        });
        self.history.push(ChatHistoryEntry {
            role: ChatRole::Assistant,
            content: String::new(),
            thinking: String::new(),
        });

        let pairs: Vec<(String, String)> = self
            .history
            .iter()
            .map(|e| {
                let role = match e.role {
                    ChatRole::User      => "user",
                    ChatRole::Assistant => "assistant",
                };
                (role.to_owned(), e.content.clone())
            })
            .collect();

        let handle = (self.spawner)(pairs, self.event_tx.clone());
        self.cancel_handle = Some(handle);
        self.mode = ChatMode::Streaming;
        self.status = Some("Generating\u{2026}".to_owned());
        self.scroll_offset = 0;
        self.in_thinking = false;
    }

    /// Signal the current stream to cancel.
    ///
    /// History is left intact — whatever was generated so far remains as the
    /// assistant reply.  The mode stays `Streaming` until the spawner thread
    /// acknowledges via `ChatEvent::StreamCancelled`, at which point `tick()`
    /// calls `on_stream_done()` to transition to `Input` / drain the queue.
    #[cfg(not(target_os = "wasi"))]
    fn cancel_stream(&mut self) {
        if let Some(cancel) = self.cancel_handle.take() {
            cancel();
        }
        self.status = Some("Cancelling\u{2026}".to_owned());
    }

    /// Called on StreamComplete / StreamCancelled: persist, then drain the
    /// pending queue if any prompts were typed while we were streaming.
    fn on_stream_done(&mut self) {
        #[cfg(not(target_os = "wasi"))]
        { self.cancel_handle = None; }
        self.in_thinking = false;
        if let Some(_next_text) = self.pending_prompts.pop_front() {
            #[cfg(not(target_os = "wasi"))]
            self.submit_message(_next_text);
            // On WASM, queue draining is handled by the host event loop.
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
        self.ingest_token(token);
    }

    /// Signal that the inference stream completed successfully.
    pub fn on_stream_complete(&mut self) {
        if let Some(ref hook) = self.save_hook {
            hook(&self.history);
        }
        self.on_stream_done();
    }

    /// Signal that the inference stream was cancelled by the user.
    pub fn on_stream_cancelled(&mut self) {
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
        self.on_stream_done();
    }

    /// Signal a template application error.
    pub fn on_template_error(&mut self, msg: String) {
        // Remove the empty assistant placeholder + user message.
        self.history.pop();
        self.history.pop();
        self.pending_toasts.push(format!("Model error: {msg}"));
        #[cfg(not(target_os = "wasi"))]
        { self.cancel_handle = None; }
        self.in_thinking = false;
        self.mode = ChatMode::Input;
        self.status = Some(msg);
    }

    /// Build the inference request payload as a JSON string (for WASM OSC output).
    ///
    /// This returns the serialized request that the WASM host should forward to
    /// the inference service.  Does NOT call the StreamSpawner.
    pub fn submit_message_payload(&mut self, user_text: String) -> String {
        self.history.push(ChatHistoryEntry {
            role: ChatRole::User,
            content: user_text,
            thinking: String::new(),
        });
        self.history.push(ChatHistoryEntry {
            role: ChatRole::Assistant,
            content: String::new(),
            thinking: String::new(),
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
    ///
    /// Returns `Some(json)` when the user submits — the host should write the
    /// JSON as an OSC 0xFD inference request.
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
                    // Double-Esc within DOUBLE_ESC_MS closes the app.
                    if let Some(prev) = self.last_esc {
                        if prev.elapsed().as_millis() < DOUBLE_ESC_MS {
                            self.quit = true;
                            return true;
                        }
                    }
                    self.last_esc = Some(Instant::now());
                    self.status = Some("Press Esc again to close".to_owned());
                    true
                }
                KeyPress::F(10) => {
                    self.quit = true;
                    true
                }
                // Ctrl-E — open full-screen editor with current textarea content.
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
                    self.submit_message(user_content);
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
                        // Ctrl-J — insert newline in textarea.
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
                KeyPress::ArrowUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                    true
                }
                KeyPress::ArrowDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(1);
                    true
                }
                _ => false,
            },

            // ── Streaming mode — allow typing + queueing ─────────────────────
            ChatMode::Streaming => match key {
                KeyPress::Escape => {
                    // Single Esc cancels current stream; pending queue is preserved.
                    self.cancel_stream();
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
                KeyPress::ArrowUp => {
                    self.scroll_offset = self.scroll_offset.saturating_add(1);
                    true
                }
                KeyPress::ArrowDown => {
                    self.scroll_offset = self.scroll_offset.saturating_sub(1);
                    true
                }
                _ => false,
            },

            // ── Editor mode ──────────────────────────────────────────────────
            ChatMode::Editor => match key {
                KeyPress::Escape => {
                    // Cancel — discard editor content, return to Input mode.
                    self.editor_text = String::new();
                    self.mode = ChatMode::Input;
                    true
                }
                // Ctrl-S — save editor content (from thread_local EditorState) into textarea.
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
                    // Forward all other keys to edtui via the thread_local EditorState.
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
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if matches!(self.mode, ChatMode::Streaming) {
                        self.cancel_handle = None;
                        self.in_thinking = false;
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
