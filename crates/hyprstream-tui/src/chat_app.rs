//! ChatApp — server-side LLM chat terminal application.
//!
//! Implements [`waxterm::app::TerminalApp`] so it can be spawned via
//! `spawn_app_process` inside the TuiService frame loop.
//!
//! The inference background thread is injected as a [`StreamSpawner`] closure
//! built in `service.rs`, keeping `hyprstream-tui` free of `hyprstream` types.

use std::sync::mpsc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use waxterm::app::TerminalApp;
use waxterm::input::KeyPress;

// ============================================================================
// Public types
// ============================================================================

/// Events sent from the inference background thread to `ChatApp::tick`.
pub enum ChatEvent {
    Token(String),
    StreamComplete,
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

/// Injected by `service.rs` — captures signing_key + model_ref.
///
/// Called with `(history_as_role_content_pairs, event_tx)` every time the
/// user submits a message.  Must be `Send + 'static` because it is moved
/// into the background thread spawned by `spawn_app_process`.
pub type StreamSpawner =
    Box<dyn Fn(Vec<(String, String)>, mpsc::SyncSender<ChatEvent>) + Send + 'static>;

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
}

// ============================================================================
// ChatApp
// ============================================================================

pub struct ChatApp {
    pub model_name: String,
    pub history: Vec<ChatHistoryEntry>,
    pub input_buf: String,
    pub mode: ChatMode,
    /// Number of lines from the tail to scroll back (0 = show tail).
    pub scroll_offset: usize,
    pub status: Option<String>,
    event_rx: mpsc::Receiver<ChatEvent>,
    event_tx: mpsc::SyncSender<ChatEvent>,
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
}

impl ChatApp {
    pub fn new(model_name: String, cols: u16, rows: u16, spawner: StreamSpawner) -> Self {
        let (event_tx, event_rx) = mpsc::sync_channel::<ChatEvent>(256);
        Self {
            model_name,
            history: Vec::new(),
            input_buf: String::new(),
            mode: ChatMode::Input,
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
        }
    }

    /// Create a private chat app with encrypted history persistence.
    ///
    /// `load_hook` is called immediately to restore any prior conversation.
    /// `save_hook` is called after each `StreamComplete` to persist history.
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
            input_buf: String::new(),
            mode: ChatMode::Input,
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
}

// ============================================================================
// TerminalApp impl
// ============================================================================

impl TerminalApp for ChatApp {
    type Command = KeyPress;

    fn render(&self, frame: &mut ratatui::Frame) {
        crate::chat_ui::draw(frame, self);
    }

    fn handle_input(&mut self, key: KeyPress) -> bool {
        // Ctrl-O toggles thinking visibility in both modes.
        if matches!(key, KeyPress::Char(0x0F)) {
            self.show_thinking = !self.show_thinking;
            return true;
        }

        match self.mode {
            ChatMode::Input => match key {
                KeyPress::Escape | KeyPress::F(10) => {
                    self.quit = true;
                    true
                }
                KeyPress::Enter => {
                    if self.input_buf.is_empty() {
                        return false;
                    }
                    let user_content = std::mem::take(&mut self.input_buf);
                    self.history.push(ChatHistoryEntry {
                        role: ChatRole::User,
                        content: user_content,
                        thinking: String::new(),
                    });
                    self.history.push(ChatHistoryEntry {
                        role: ChatRole::Assistant,
                        content: String::new(),
                        thinking: String::new(),
                    });

                    // Only send content (not thinking) as model context.
                    let pairs: Vec<(String, String)> = self
                        .history
                        .iter()
                        .map(|e| {
                            let role = match e.role {
                                ChatRole::User => "user",
                                ChatRole::Assistant => "assistant",
                            };
                            (role.to_owned(), e.content.clone())
                        })
                        .collect();

                    (self.spawner)(pairs, self.event_tx.clone());
                    self.mode = ChatMode::Streaming;
                    self.status = Some("Generating\u{2026}".to_owned());
                    self.scroll_offset = 0;
                    self.in_thinking = false;
                    true
                }
                KeyPress::Backspace => {
                    self.input_buf.pop();
                    true
                }
                KeyPress::Char(b) => {
                    if b.is_ascii_graphic() || b == b' ' {
                        self.input_buf.push(b as char);
                        true
                    } else {
                        false
                    }
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
            ChatMode::Streaming => match key {
                KeyPress::Escape | KeyPress::F(10) => {
                    self.quit = true;
                    true
                }
                _ => false,
            },
        }
    }

    fn tick(&mut self, _delta_ms: u64) -> bool {
        let mut redraw = false;
        loop {
            match self.event_rx.try_recv() {
                Ok(ChatEvent::Token(s)) => {
                    self.ingest_token(&s);
                    redraw = true;
                }
                Ok(ChatEvent::StreamComplete) => {
                    self.mode = ChatMode::Input;
                    self.status = None;
                    self.in_thinking = false;
                    if let Some(ref hook) = self.save_hook {
                        hook(&self.history);
                    }
                    redraw = true;
                }
                Ok(ChatEvent::StreamError(s)) => {
                    self.pending_toasts.push(format!("Inference error: {s}"));
                    if let Some(last) = self.history.last_mut() {
                        last.content = format!("[Error: {s}]");
                    }
                    self.mode = ChatMode::Input;
                    self.status = None;
                    self.in_thinking = false;
                    redraw = true;
                }
                Ok(ChatEvent::TemplateError(s)) => {
                    // Remove the empty assistant placeholder + user message.
                    self.history.pop();
                    self.history.pop();
                    self.pending_toasts.push(format!("Model error: {s}"));
                    self.status = Some(s);
                    self.mode = ChatMode::Input;
                    self.in_thinking = false;
                    redraw = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if matches!(self.mode, ChatMode::Streaming) {
                        self.mode = ChatMode::Input;
                        self.in_thinking = false;
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
