//! Single numeric value with browse (arrow incr/decr) and edit (digit entry) modes.

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::input::KeyPress;
use super::WidgetResult;

/// Whether the field holds an integer or a float.
#[derive(Clone, Copy, Debug)]
pub enum NumericKind {
    Integer,
    Float { decimals: u8 },
}

/// A single numeric field supporting browse mode (arrow incr/decr) and
/// edit mode (direct digit entry).
pub struct NumericField {
    label: String,
    kind: NumericKind,
    value: Option<f64>,
    min: f64,
    max: f64,
    step: f64,
    nullable: bool,
    none_label: &'static str,
    // Edit mode state
    editing: bool,
    edit_buf: String,
    edit_cursor: usize,
    pre_edit_value: Option<f64>,
}

impl NumericField {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            kind: NumericKind::Integer,
            value: None,
            min: 0.0,
            max: f64::MAX,
            step: 1.0,
            nullable: true,
            none_label: "default",
            editing: false,
            edit_buf: String::new(),
            edit_cursor: 0,
            pre_edit_value: None,
        }
    }

    /// Set the field to integer mode.
    pub fn integer(mut self) -> Self {
        self.kind = NumericKind::Integer;
        self
    }

    /// Set the field to float mode with the given number of decimal places.
    pub fn float(mut self, decimals: u8) -> Self {
        self.kind = NumericKind::Float { decimals };
        self
    }

    /// Set the valid range (inclusive).
    pub fn range(mut self, min: f64, max: f64) -> Self {
        self.min = min;
        self.max = max;
        self
    }

    /// Set the increment/decrement step size.
    pub fn step(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Set the initial value. None means "use default".
    pub fn value(mut self, v: Option<f64>) -> Self {
        self.value = v;
        self
    }

    /// Set the label shown when value is None.
    pub fn none_label(mut self, label: &'static str) -> Self {
        self.none_label = label;
        self
    }

    /// Whether the field can be set to None.
    pub fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Current value.
    pub fn get_value(&self) -> Option<f64> {
        self.value
    }

    /// Whether the field is in edit (text entry) mode.
    pub fn is_editing(&self) -> bool {
        self.editing
    }

    /// Format the current value for display.
    pub fn display_value(&self) -> String {
        match self.value {
            None => self.none_label.to_owned(),
            Some(v) => match self.kind {
                NumericKind::Integer => format!("{}", v as i64),
                NumericKind::Float { decimals } => format!("{:.prec$}", v, prec = decimals as usize),
            },
        }
    }

    fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.min, self.max)
    }

    fn round_for_kind(&self, v: f64) -> f64 {
        match self.kind {
            NumericKind::Integer => v.round(),
            NumericKind::Float { decimals } => {
                let factor = 10f64.powi(decimals as i32);
                (v * factor).round() / factor
            }
        }
    }

    fn enter_edit_mode(&mut self, initial_char: Option<u8>) {
        self.pre_edit_value = self.value;
        self.editing = true;
        self.edit_buf = match initial_char {
            Some(ch) => {
                self.edit_cursor = 1;
                String::from(ch as char)
            }
            None => {
                let s = self.display_value();
                self.edit_cursor = s.len();
                if self.value.is_none() { String::new() } else { s }
            }
        };
    }

    fn commit_edit(&mut self) {
        if self.edit_buf.is_empty() && self.nullable {
            self.value = None;
        } else if let Ok(v) = self.edit_buf.parse::<f64>() {
            self.value = Some(self.round_for_kind(self.clamp(v)));
        }
        // else: invalid input — keep previous value
        self.editing = false;
        self.edit_buf.clear();
    }

    fn cancel_edit(&mut self) {
        self.value = self.pre_edit_value;
        self.editing = false;
        self.edit_buf.clear();
    }

    fn is_valid_edit_char(&self, b: u8) -> bool {
        b.is_ascii_digit()
            || b == b'.'
            || b == b'-'
    }

    /// Handle a key press. Returns `Confirmed` when Enter is pressed in browse mode,
    /// `Cancelled` when Escape is pressed in browse mode.
    pub fn handle_key(&mut self, key: &KeyPress) -> WidgetResult<Option<f64>> {
        if self.editing {
            match key {
                KeyPress::Enter => {
                    self.commit_edit();
                    WidgetResult::Pending
                }
                KeyPress::Escape => {
                    self.cancel_edit();
                    WidgetResult::Pending
                }
                KeyPress::Backspace => {
                    if self.edit_cursor > 0 {
                        self.edit_cursor -= 1;
                        self.edit_buf.remove(self.edit_cursor);
                    }
                    WidgetResult::Pending
                }
                KeyPress::ArrowLeft => {
                    if self.edit_cursor > 0 {
                        self.edit_cursor -= 1;
                    }
                    WidgetResult::Pending
                }
                KeyPress::ArrowRight => {
                    if self.edit_cursor < self.edit_buf.len() {
                        self.edit_cursor += 1;
                    }
                    WidgetResult::Pending
                }
                KeyPress::Char(b) if self.is_valid_edit_char(*b) => {
                    self.edit_buf.insert(self.edit_cursor, *b as char);
                    self.edit_cursor += 1;
                    WidgetResult::Pending
                }
                _ => WidgetResult::Pending,
            }
        } else {
            // Browse mode
            match key {
                KeyPress::ArrowLeft => {
                    match self.value {
                        Some(v) => {
                            let new_v = self.round_for_kind(v - self.step);
                            if new_v < self.min && self.nullable {
                                self.value = None;
                            } else {
                                self.value = Some(self.clamp(new_v));
                            }
                        }
                        None if !self.nullable => {
                            self.value = Some(self.min);
                        }
                        None => {} // already None, can't go lower
                    }
                    WidgetResult::Pending
                }
                KeyPress::ArrowRight => {
                    match self.value {
                        Some(v) => {
                            let new_v = self.round_for_kind(v + self.step);
                            self.value = Some(self.clamp(new_v));
                        }
                        None => {
                            self.value = Some(self.min);
                        }
                    }
                    WidgetResult::Pending
                }
                KeyPress::Backspace if self.nullable => {
                    self.value = None;
                    WidgetResult::Pending
                }
                KeyPress::Char(b) if self.is_valid_edit_char(*b) => {
                    self.enter_edit_mode(Some(*b));
                    WidgetResult::Pending
                }
                KeyPress::Enter => WidgetResult::Confirmed(self.value),
                KeyPress::Escape => WidgetResult::Cancelled,
                _ => WidgetResult::Pending,
            }
        }
    }

    /// Render the field on a single row.
    pub fn render(&self, frame: &mut Frame, area: Rect, focused: bool) {
        let label_style = Style::default().fg(Color::Rgb(112, 128, 160)); // DIM
        let focus_style = Style::default().fg(Color::Rgb(0, 232, 252)).add_modifier(Modifier::BOLD); // CYAN
        let value_style = if focused { Style::default().fg(Color::White) } else { label_style };

        let cursor_marker = if focused { "\u{25b6} " } else { "  " };

        let mut spans: Vec<Span> = vec![
            Span::styled(cursor_marker, if focused { focus_style } else { label_style }),
            Span::styled(format!("{:<16}", self.label), label_style),
        ];

        if self.editing {
            // Edit mode: show text buffer with cursor
            let buf = &self.edit_buf;
            if self.edit_cursor > 0 && self.edit_cursor <= buf.len() {
                spans.push(Span::styled(&buf[..self.edit_cursor], value_style));
            }
            let cursor_ch = if self.edit_cursor < buf.len() {
                &buf[self.edit_cursor..self.edit_cursor + 1]
            } else {
                " "
            };
            spans.push(Span::styled(
                cursor_ch.to_owned(),
                Style::default().fg(Color::Black).bg(Color::White),
            ));
            if self.edit_cursor + 1 < buf.len() {
                spans.push(Span::styled(&buf[self.edit_cursor + 1..], value_style));
            }
        } else {
            // Browse mode: show ‹ value ›
            let display = self.display_value();
            if focused {
                spans.push(Span::styled("\u{2039} ", focus_style));
                spans.push(Span::styled(display, value_style));
                spans.push(Span::styled(" \u{203a}", focus_style));
            } else {
                spans.push(Span::styled(display, value_style));
            }
        }

        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_field() -> NumericField {
        NumericField::new("Tokens")
            .integer()
            .range(64.0, 32768.0)
            .step(64.0)
            .value(Some(2048.0))
    }

    fn float_field() -> NumericField {
        NumericField::new("Temp")
            .float(2)
            .range(0.0, 2.0)
            .step(0.05)
            .value(Some(0.7))
    }

    #[test]
    fn test_increment_decrement() {
        let mut f = int_field();
        f.handle_key(&KeyPress::ArrowRight);
        assert_eq!(f.get_value(), Some(2112.0));
        f.handle_key(&KeyPress::ArrowLeft);
        assert_eq!(f.get_value(), Some(2048.0));
    }

    #[test]
    fn test_clamp_to_range() {
        let mut f = int_field();
        f.value = Some(32768.0);
        f.handle_key(&KeyPress::ArrowRight);
        assert_eq!(f.get_value(), Some(32768.0)); // clamped at max
    }

    #[test]
    fn test_digit_entry_switches_to_edit() {
        let mut f = int_field();
        assert!(!f.is_editing());
        f.handle_key(&KeyPress::Char(b'5'));
        assert!(f.is_editing());
        assert_eq!(f.edit_buf, "5");
    }

    #[test]
    fn test_edit_confirm_parses() {
        let mut f = int_field();
        f.handle_key(&KeyPress::Char(b'4'));
        f.handle_key(&KeyPress::Char(b'0'));
        f.handle_key(&KeyPress::Char(b'9'));
        f.handle_key(&KeyPress::Char(b'6'));
        f.handle_key(&KeyPress::Enter);
        assert!(!f.is_editing());
        assert_eq!(f.get_value(), Some(4096.0));
    }

    #[test]
    fn test_edit_clamps_parsed_value() {
        let mut f = int_field();
        f.handle_key(&KeyPress::Char(b'9'));
        f.handle_key(&KeyPress::Char(b'9'));
        f.handle_key(&KeyPress::Char(b'9'));
        f.handle_key(&KeyPress::Char(b'9'));
        f.handle_key(&KeyPress::Char(b'9'));
        f.handle_key(&KeyPress::Enter);
        assert_eq!(f.get_value(), Some(32768.0)); // clamped
    }

    #[test]
    fn test_edit_escape_reverts() {
        let mut f = int_field();
        assert_eq!(f.get_value(), Some(2048.0));
        f.handle_key(&KeyPress::Char(b'1'));
        f.handle_key(&KeyPress::Escape);
        assert!(!f.is_editing());
        assert_eq!(f.get_value(), Some(2048.0)); // reverted
    }

    #[test]
    fn test_none_toggle() {
        let mut f = int_field().nullable(true);
        f.handle_key(&KeyPress::Backspace);
        assert_eq!(f.get_value(), None);
        f.handle_key(&KeyPress::ArrowRight);
        assert_eq!(f.get_value(), Some(64.0)); // min value
    }

    #[test]
    fn test_float_formatting() {
        let f = float_field();
        assert_eq!(f.display_value(), "0.70");
    }

    #[test]
    fn test_float_step() {
        let mut f = float_field();
        f.handle_key(&KeyPress::ArrowRight);
        assert_eq!(f.display_value(), "0.75");
        f.handle_key(&KeyPress::ArrowLeft);
        assert_eq!(f.display_value(), "0.70");
    }

    #[test]
    fn test_invalid_input_ignored() {
        let mut f = int_field();
        f.handle_key(&KeyPress::Char(b'a')); // not a digit
        assert!(!f.is_editing()); // should not enter edit mode
    }

    #[test]
    fn test_browse_enter_confirms() {
        let mut f = int_field();
        assert_eq!(f.handle_key(&KeyPress::Enter), WidgetResult::Confirmed(Some(2048.0)));
    }

    #[test]
    fn test_browse_escape_cancels() {
        let mut f = int_field();
        assert_eq!(f.handle_key(&KeyPress::Escape), WidgetResult::Cancelled);
    }

    #[test]
    fn test_non_nullable_cannot_be_none() {
        let mut f = int_field().nullable(false);
        f.handle_key(&KeyPress::Backspace);
        assert_eq!(f.get_value(), Some(2048.0)); // unchanged
    }

    #[test]
    fn test_decrement_past_min_to_none() {
        let mut f = NumericField::new("X")
            .integer()
            .range(1.0, 10.0)
            .step(1.0)
            .value(Some(1.0))
            .nullable(true);
        f.handle_key(&KeyPress::ArrowLeft);
        assert_eq!(f.get_value(), None); // went below min → None
    }
}
