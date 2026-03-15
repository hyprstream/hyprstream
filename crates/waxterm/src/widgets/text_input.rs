use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::input::KeyPress;

use super::WidgetResult;

/// Single-line text input with cursor.
pub struct TextInput {
    label: String,
    value: Vec<u8>,
    cursor: usize,
    default: Option<String>,
}

impl TextInput {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: Vec::new(),
            cursor: 0,
            default: None,
        }
    }

    pub fn with_default(mut self, default: impl Into<String>) -> Self {
        self.default = Some(default.into());
        self
    }

    /// Current text value.
    pub fn value(&self) -> String {
        if self.value.is_empty() {
            if let Some(ref d) = self.default {
                return d.clone();
            }
        }
        String::from_utf8_lossy(&self.value).into_owned()
    }

    pub fn handle_key(&mut self, key: &KeyPress) -> WidgetResult<String> {
        match key {
            KeyPress::Enter => WidgetResult::Confirmed(self.value()),
            KeyPress::Escape => WidgetResult::Cancelled,
            KeyPress::Backspace => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                    self.value.remove(self.cursor);
                }
                WidgetResult::Pending
            }
            KeyPress::ArrowLeft => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                }
                WidgetResult::Pending
            }
            KeyPress::ArrowRight => {
                if self.cursor < self.value.len() {
                    self.cursor += 1;
                }
                WidgetResult::Pending
            }
            KeyPress::Char(b) if b.is_ascii_graphic() || *b == b' ' => {
                self.value.insert(self.cursor, *b);
                self.cursor += 1;
                WidgetResult::Pending
            }
            _ => WidgetResult::Pending,
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let display_val = String::from_utf8_lossy(&self.value);

        let mut spans = vec![
            Span::styled(&self.label, Style::default().fg(Color::Cyan)),
            Span::raw(" "),
        ];

        if self.value.is_empty() {
            if let Some(ref d) = self.default {
                spans.push(Span::styled(
                    format!("({d}) "),
                    Style::default().fg(Color::DarkGray),
                ));
            }
        }

        // Text before cursor
        if self.cursor > 0 {
            let before = &display_val[..self.cursor];
            spans.push(Span::raw(before.to_string()));
        }

        // Cursor character
        let cursor_ch = if self.cursor < display_val.len() {
            display_val.chars().nth(self.cursor).unwrap_or(' ')
        } else {
            ' '
        };
        spans.push(Span::styled(
            cursor_ch.to_string(),
            Style::default().fg(Color::Black).bg(Color::White),
        ));

        // Text after cursor
        if self.cursor < display_val.len() {
            let after = &display_val[self.cursor + 1..];
            if !after.is_empty() {
                spans.push(Span::raw(after.to_string()));
            }
        }

        let paragraph = Paragraph::new(Line::from(spans));
        frame.render_widget(paragraph, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_and_confirm() {
        let mut input = TextInput::new("Name:");
        assert_eq!(input.handle_key(&KeyPress::Char(b'h')), WidgetResult::Pending);
        assert_eq!(input.handle_key(&KeyPress::Char(b'i')), WidgetResult::Pending);
        assert_eq!(
            input.handle_key(&KeyPress::Enter),
            WidgetResult::Confirmed("hi".to_string())
        );
    }

    #[test]
    fn test_backspace() {
        let mut input = TextInput::new(">");
        input.handle_key(&KeyPress::Char(b'a'));
        input.handle_key(&KeyPress::Char(b'b'));
        input.handle_key(&KeyPress::Backspace);
        assert_eq!(
            input.handle_key(&KeyPress::Enter),
            WidgetResult::Confirmed("a".to_string())
        );
    }

    #[test]
    fn test_default_value() {
        let input = TextInput::new(">").with_default("hello");
        assert_eq!(input.value(), "hello");
    }

    #[test]
    fn test_escape_cancels() {
        let mut input = TextInput::new(">");
        input.handle_key(&KeyPress::Char(b'x'));
        assert_eq!(input.handle_key(&KeyPress::Escape), WidgetResult::Cancelled);
    }

    #[test]
    fn test_cursor_movement() {
        let mut input = TextInput::new(">");
        input.handle_key(&KeyPress::Char(b'a'));
        input.handle_key(&KeyPress::Char(b'c'));
        input.handle_key(&KeyPress::ArrowLeft);
        input.handle_key(&KeyPress::Char(b'b'));
        assert_eq!(
            input.handle_key(&KeyPress::Enter),
            WidgetResult::Confirmed("abc".to_string())
        );
    }
}
