//! `CastPlayerApp` — a `TerminalApp` implementation for asciicast v2 playback.

use waxterm::input::KeyPress;
use waxterm::TerminalApp;

use crate::cast_player::{CastFile, CastPlayer};

/// Idle time limit: cap gaps between events to 100ms for smoother playback.
const IDLE_TIME_LIMIT_MS: u64 = 100;

/// A log entry with a name and its cast file content.
pub struct LogEntry {
    pub name: String,
    pub cast_content: String,
}

/// Cast player application state.
pub struct CastPlayerApp {
    pub logs: Vec<LogEntry>,
    pub selected: usize,
    pub player: CastPlayer,
    quit: bool,
    loop_playback: bool,
}

/// Commands for the cast player.
pub enum CastCommand {
    Key(KeyPress),
}

impl From<KeyPress> for CastCommand {
    fn from(k: KeyPress) -> Self {
        CastCommand::Key(k)
    }
}

fn parse_cast(content: &str) -> CastFile {
    CastFile::parse_with_idle_limit(content, Some(IDLE_TIME_LIMIT_MS)).unwrap_or_else(|e| {
        eprintln!("Failed to parse cast: {}", e);
        CastFile {
            header: crate::cast_player::CastHeader {
                version: 2,
                width: 80,
                height: 24,
                title: Some("Error".to_string()),
            },
            events: vec![],
            duration_ms: 0,
        }
    })
}

fn empty_cast() -> CastFile {
    CastFile {
        header: crate::cast_player::CastHeader {
            version: 2,
            width: 80,
            height: 24,
            title: Some("No logs".to_string()),
        },
        events: vec![],
        duration_ms: 0,
    }
}

impl CastPlayerApp {
    /// Create a new cast player with the given log entries.
    pub fn new(logs: Vec<LogEntry>) -> Self {
        let cast = if !logs.is_empty() {
            parse_cast(&logs[0].cast_content)
        } else {
            empty_cast()
        };

        let player = CastPlayer::new(cast);

        CastPlayerApp {
            logs,
            selected: 0,
            player,
            quit: false,
            loop_playback: false,
        }
    }

    /// Create a cast player from a single cast file content string.
    pub fn from_content(name: String, content: String) -> Self {
        Self::new(vec![LogEntry {
            name,
            cast_content: content,
        }])
    }

    /// Enable or disable loop playback.
    pub fn set_loop(&mut self, loop_playback: bool) {
        self.loop_playback = loop_playback;
    }

    /// Switch to a different log entry.
    pub fn select_log(&mut self, index: usize) {
        if index >= self.logs.len() || index == self.selected {
            return;
        }
        self.selected = index;
        let cast = parse_cast(&self.logs[index].cast_content);
        self.player = CastPlayer::new(cast);
    }

    /// Cycle to next log. If there is only one log, restart it.
    pub fn next_log(&mut self) {
        if self.logs.is_empty() {
            return;
        }
        let next = (self.selected + 1) % self.logs.len();
        if next == self.selected {
            let cast = parse_cast(&self.logs[self.selected].cast_content);
            self.player = CastPlayer::new(cast);
        } else {
            self.select_log(next);
        }
    }
}

impl TerminalApp for CastPlayerApp {
    type Command = CastCommand;

    fn render(&self, frame: &mut ratatui::Frame) {
        crate::cast_ui::draw(frame, self);
    }

    fn handle_input(&mut self, cmd: CastCommand) -> bool {
        match cmd {
            CastCommand::Key(key) => match key {
                KeyPress::Char(b'q' | b'Q') => {
                    self.quit = true;
                    false
                }
                KeyPress::Tab => {
                    self.next_log();
                    true
                }
                KeyPress::Char(b' ') => {
                    self.player.toggle_pause();
                    true
                }
                KeyPress::ArrowRight => {
                    self.player.seek_forward(3000);
                    true
                }
                KeyPress::ArrowLeft => {
                    self.player.seek_backward(3000);
                    true
                }
                KeyPress::Char(b'1') => { self.select_log(0); true }
                KeyPress::Char(b'2') => { self.select_log(1); true }
                KeyPress::Char(b'3') => { self.select_log(2); true }
                _ => false,
            },
        }
    }

    fn tick(&mut self, delta_ms: u64) -> bool {
        let changed = self.player.tick(delta_ms);
        if self.player.is_finished() {
            if self.loop_playback {
                self.next_log();
                return true;
            }
            self.quit = true;
        }
        changed
    }

    fn should_quit(&self) -> bool {
        self.quit
    }
}
