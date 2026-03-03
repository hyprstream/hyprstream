use std::fmt::Display;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::input::KeyPress;

use super::WidgetResult;

/// Single-selection list widget (arrow keys + Enter).
pub struct SelectList<T> {
    label: String,
    items: Vec<T>,
    selected: usize,
}

impl<T: Display + Clone> SelectList<T> {
    pub fn new(label: impl Into<String>, items: Vec<T>) -> Self {
        Self {
            label: label.into(),
            items,
            selected: 0,
        }
    }

    /// Index of the currently highlighted item.
    pub fn selected_index(&self) -> usize {
        self.selected
    }

    pub fn handle_key(&mut self, key: &KeyPress) -> WidgetResult<T> {
        match key {
            KeyPress::ArrowUp => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                WidgetResult::Pending
            }
            KeyPress::ArrowDown => {
                if self.selected + 1 < self.items.len() {
                    self.selected += 1;
                }
                WidgetResult::Pending
            }
            KeyPress::Enter => {
                if let Some(item) = self.items.get(self.selected) {
                    WidgetResult::Confirmed(item.clone())
                } else {
                    WidgetResult::Pending
                }
            }
            KeyPress::Escape => WidgetResult::Cancelled,
            _ => WidgetResult::Pending,
        }
    }

    /// Render the list. Uses up to `items.len() + 1` rows (label + items).
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let mut lines = Vec::with_capacity(self.items.len() + 1);

        lines.push(Line::from(Span::styled(
            &self.label,
            Style::default().fg(Color::Cyan),
        )));

        for (i, item) in self.items.iter().enumerate() {
            let marker = if i == self.selected { "> " } else { "  " };
            let style = if i == self.selected {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(
                format!("{marker}{item}"),
                style,
            )));
        }

        let paragraph = Paragraph::new(lines);
        frame.render_widget(paragraph, area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigate_and_select() {
        let mut list = SelectList::new("Pick:", vec!["a", "b", "c"]);
        assert_eq!(list.selected_index(), 0);
        list.handle_key(&KeyPress::ArrowDown);
        assert_eq!(list.selected_index(), 1);
        list.handle_key(&KeyPress::ArrowDown);
        assert_eq!(list.selected_index(), 2);
        // Can't go past end
        list.handle_key(&KeyPress::ArrowDown);
        assert_eq!(list.selected_index(), 2);
        assert_eq!(list.handle_key(&KeyPress::Enter), WidgetResult::Confirmed("c"));
    }

    #[test]
    fn test_arrow_up_clamps() {
        let mut list = SelectList::new("Pick:", vec!["a", "b"]);
        list.handle_key(&KeyPress::ArrowUp);
        assert_eq!(list.selected_index(), 0);
    }

    #[test]
    fn test_escape() {
        let mut list = SelectList::new("Pick:", vec!["a"]);
        assert_eq!(list.handle_key(&KeyPress::Escape), WidgetResult::<&str>::Cancelled);
    }
}
