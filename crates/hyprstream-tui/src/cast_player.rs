//! Asciicast v2 parser and timed playback engine.
//!
//! Parses asciicast v2 files (JSON header + newline-delimited event tuples)
//! and provides a `CastPlayer` that drives an `avt::Vt` virtual terminal
//! with timed event replay, seeking, and pause.

use serde::Deserialize;

/// Strip OSC, DCS, APC, PM sequences from event data.
///
/// These have no role in terminal animation playback and could inject
/// title-sets, clipboard writes, or other side-effects if a cast file
/// were malicious.
fn strip_dangerous_sequences(data: String) -> String {
    // Fast path: vast majority of events contain none of these prefixes.
    if !data.contains("\x1b]")
        && !data.contains("\x1bP")
        && !data.contains("\x1b_")
        && !data.contains("\x1b^")
    {
        return data;
    }
    let mut out = String::with_capacity(data.len());
    let mut chars = data.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            match chars.peek().copied() {
                Some(']') | Some('P') | Some('_') | Some('^') => {
                    chars.next(); // consume the introducer
                    // Skip until BEL or ST (\x1b\\)
                    let mut prev = '\0';
                    for ch in chars.by_ref() {
                        if ch == '\x07' {
                            break;
                        }
                        if prev == '\x1b' && ch == '\\' {
                            break;
                        }
                        prev = ch;
                    }
                    // drop the sequence — push nothing
                }
                _ => out.push(c),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Asciicast v2 header.
#[derive(Debug, Deserialize)]
pub struct CastHeader {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub title: Option<String>,
}

/// A single output event: `[time, "o", data]`.
#[derive(Debug, Clone)]
pub struct CastEvent {
    pub time_ms: u64,
    pub data: String,
}

/// Parsed cast file ready for playback.
#[derive(Debug)]
pub struct CastFile {
    pub header: CastHeader,
    pub events: Vec<CastEvent>,
    pub duration_ms: u64,
}

/// Manages timed replay of a cast file.
pub struct CastPlayer {
    pub cast: CastFile,
    pub elapsed_ms: u64,
    pub event_index: usize,
    pub paused: bool,
    vt: avt::Vt,
}

impl CastFile {
    /// Parse asciicast v2 content (JSON header + newline-delimited event lines).
    pub fn parse(content: &str) -> Result<Self, String> {
        Self::parse_with_idle_limit(content, None)
    }

    /// Parse with an optional idle time limit (in ms).
    /// Gaps between consecutive events exceeding the limit are compressed.
    pub fn parse_with_idle_limit(content: &str, idle_limit_ms: Option<u64>) -> Result<Self, String> {
        let mut lines = content.lines();

        let header_line = lines.next().ok_or("Empty cast file")?;
        let header: CastHeader =
            serde_json::from_str(header_line).map_err(|e| format!("Invalid header: {}", e))?;

        if header.version != 2 {
            return Err(format!("Unsupported version: {}", header.version));
        }

        let mut events = Vec::new();

        for line in lines {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let (time_secs, event_type, data): (f64, String, String) =
                serde_json::from_str(line).map_err(|e| format!("Invalid event: {}", e))?;

            if event_type != "o" {
                continue;
            }

            let time_ms = (time_secs * 1000.0) as u64;
            events.push(CastEvent {
                time_ms,
                data: strip_dangerous_sequences(data),
            });
        }

        // Apply idle time limit: cap gaps between consecutive events
        if let Some(limit) = idle_limit_ms {
            let mut offset: u64 = 0;
            for i in 1..events.len() {
                let gap = events[i].time_ms.saturating_sub(events[i - 1].time_ms + offset);
                if gap > limit {
                    offset += gap - limit;
                }
                events[i].time_ms = events[i].time_ms.saturating_sub(offset);
            }
        }

        let duration_ms = events.last().map(|e| e.time_ms).unwrap_or(0);

        Ok(CastFile {
            header,
            events,
            duration_ms,
        })
    }
}

impl CastPlayer {
    pub fn new(cast: CastFile) -> Self {
        let vt = avt::Vt::builder()
            .size(cast.header.width as usize, cast.header.height as usize)
            .scrollback_limit(0)
            .build();

        CastPlayer {
            cast,
            elapsed_ms: 0,
            event_index: 0,
            paused: false,
            vt,
        }
    }

    /// Advance playback by delta_ms. Returns true if new output was produced.
    pub fn tick(&mut self, delta_ms: u64) -> bool {
        if self.paused {
            return false;
        }

        self.elapsed_ms += delta_ms;
        let mut produced = false;

        while self.event_index < self.cast.events.len() {
            let event = &self.cast.events[self.event_index];
            if event.time_ms <= self.elapsed_ms {
                self.vt.feed_str(&event.data);
                self.event_index += 1;
                produced = true;
            } else {
                break;
            }
        }

        produced
    }

    /// Access the virtual terminal for styled rendering.
    pub fn vt(&self) -> &avt::Vt {
        &self.vt
    }

    /// Returns true if all events have been played.
    pub fn is_finished(&self) -> bool {
        self.event_index >= self.cast.events.len() && self.elapsed_ms >= self.cast.duration_ms
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn seek_forward(&mut self, ms: u64) {
        if self.paused {
            self.elapsed_ms += ms;
            self.replay_to(self.elapsed_ms);
            self.paused = true;
        } else {
            self.elapsed_ms += ms;
        }
    }

    pub fn seek_backward(&mut self, ms: u64) {
        let target = self.elapsed_ms.saturating_sub(ms);
        self.replay_to(target);
    }

    /// Seek to an absolute time position, replaying from start.
    pub fn seek_to(&mut self, target_ms: u64) {
        self.replay_to(target_ms);
    }

    /// Replay from the beginning to a target time, rebuilding vt state.
    fn replay_to(&mut self, target_ms: u64) {
        let was_paused = self.paused;
        self.vt = avt::Vt::builder()
            .size(
                self.cast.header.width as usize,
                self.cast.header.height as usize,
            )
            .scrollback_limit(0)
            .build();
        self.event_index = 0;
        self.elapsed_ms = 0;
        self.paused = false;
        self.tick(target_ms);
        self.paused = was_paused;
    }

    pub fn progress(&self) -> f64 {
        if self.cast.duration_ms == 0 {
            return 0.0;
        }
        (self.elapsed_ms as f64 / self.cast.duration_ms as f64).min(1.0)
    }

    pub fn elapsed_display(&self) -> String {
        format_time(self.elapsed_ms)
    }

    pub fn duration_display(&self) -> String {
        format_time(self.cast.duration_ms)
    }
}

/// Format milliseconds as `MM:SS`.
pub fn format_time(ms: u64) -> String {
    let total_secs = ms / 1000;
    let mins = total_secs / 60;
    let secs = total_secs % 60;
    format!("{:02}:{:02}", mins, secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CAST: &str = r#"{"version": 2, "width": 80, "height": 24, "title": "test"}
[0.0, "o", "Hello "]
[0.5, "o", "World"]
[1.0, "o", "\r\nLine 2"]
[5.0, "o", "\r\nAfter gap"]"#;

    #[test]
    fn test_parse_basic() {
        let cast = CastFile::parse(SAMPLE_CAST).unwrap();
        assert_eq!(cast.header.width, 80);
        assert_eq!(cast.header.height, 24);
        assert_eq!(cast.events.len(), 4);
        assert_eq!(cast.duration_ms, 5000);
    }

    #[test]
    fn test_playback() {
        let cast = CastFile::parse(SAMPLE_CAST).unwrap();
        let mut player = CastPlayer::new(cast);
        player.tick(600);

        let text: String = player.vt().text().join("\n");
        assert!(text.contains("Hello World"));
    }

    #[test]
    fn test_idle_limit() {
        let cast_full = CastFile::parse(SAMPLE_CAST).unwrap();
        let full_dur = cast_full.duration_ms;

        let cast_limited = CastFile::parse_with_idle_limit(SAMPLE_CAST, Some(500)).unwrap();
        let limited_dur = cast_limited.duration_ms;

        assert!(limited_dur < full_dur);
    }

    #[test]
    fn test_seek() {
        let cast = CastFile::parse(SAMPLE_CAST).unwrap();
        let mut player = CastPlayer::new(cast);
        player.tick(5000);
        assert!(player.is_finished());

        player.seek_backward(4000);
        assert!(!player.is_finished());
    }

    #[test]
    fn test_strip_dangerous() {
        let input = "hello\x1b]0;evil title\x07world".to_string();
        let result = strip_dangerous_sequences(input);
        assert_eq!(result, "helloworld");
    }
}
