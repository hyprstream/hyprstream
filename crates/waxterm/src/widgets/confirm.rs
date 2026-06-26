use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::input::KeyPress;

use super::WidgetResult;

/// Yes/No confirmation dialog.
pub struct ConfirmDialog {
    label: String,
    value: bool,
}

impl ConfirmDialog {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            value: true,
        }
    }

    pub fn with_default(mut self, default: bool) -> Self {
        self.value = default;
        self
    }

    /// Current selection.
    pub fn value(&self) -> bool {
        self.value
    }

    pub fn handle_key(&mut self, key: &KeyPress) -> WidgetResult<bool> {
        match key {
            KeyPress::ArrowLeft | KeyPress::ArrowRight => {
                self.value = !self.value;
                WidgetResult::Pending
            }
            KeyPress::Char(b'y') | KeyPress::Char(b'Y') => {
                self.value = true;
                WidgetResult::Confirmed(true)
            }
            KeyPress::Char(b'n') | KeyPress::Char(b'N') => {
                self.value = false;
                WidgetResult::Confirmed(false)
            }
            KeyPress::Enter => WidgetResult::Confirmed(self.value),
            KeyPress::Escape => WidgetResult::Cancelled,
            _ => WidgetResult::Pending,
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let yes_style = if self.value {
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let no_style = if !self.value {
            Style::default()
                .fg(Color::Red)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::DarkGray)
        };

        let line = Line::from(vec![
            Span::styled(&self.label, Style::default().fg(Color::Cyan)),
            Span::raw(" "),
            Span::styled(if self.value { "[Yes]" } else { " Yes " }, yes_style),
            Span::raw(" / "),
            Span::styled(if !self.value { "[No]" } else { " No " }, no_style),
        ]);

        let paragraph = Paragraph::new(line);
        frame.render_widget(paragraph, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_yes() {
        let mut dialog = ConfirmDialog::new("Continue?");
        assert!(dialog.value());
        assert_eq!(dialog.handle_key(&KeyPress::Enter), WidgetResult::Confirmed(true));
    }

    #[test]
    fn test_default_no() {
        let mut dialog = ConfirmDialog::new("Delete?").with_default(false);
        assert!(!dialog.value());
        assert_eq!(dialog.handle_key(&KeyPress::Enter), WidgetResult::Confirmed(false));
    }

    #[test]
    fn test_arrow_toggles() {
        let mut dialog = ConfirmDialog::new("?");
        assert!(dialog.value());
        dialog.handle_key(&KeyPress::ArrowRight);
        assert!(!dialog.value());
        dialog.handle_key(&KeyPress::ArrowLeft);
        assert!(dialog.value());
    }

    #[test]
    fn test_y_n_shortcuts() {
        let mut dialog = ConfirmDialog::new("?");
        assert_eq!(dialog.handle_key(&KeyPress::Char(b'n')), WidgetResult::Confirmed(false));

        let mut dialog2 = ConfirmDialog::new("?").with_default(false);
        assert_eq!(dialog2.handle_key(&KeyPress::Char(b'y')), WidgetResult::Confirmed(true));
    }

    #[test]
    fn test_escape() {
        let mut dialog = ConfirmDialog::new("?");
        assert_eq!(dialog.handle_key(&KeyPress::Escape), WidgetResult::Cancelled);
    }
}
