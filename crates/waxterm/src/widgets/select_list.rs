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

    /// Reference to the currently highlighted item, if any.
    pub fn selected_item(&self) -> Option<&T> {
        self.items.get(self.selected)
    }

    /// Builder: set the initial selection index.
    pub fn with_selected(mut self, idx: usize) -> Self {
        if idx < self.items.len() {
            self.selected = idx;
        }
        self
    }

    /// Mutable reference to items, for updating status in place.
    pub fn items_mut(&mut self) -> &mut Vec<T> {
        &mut self.items
    }

    /// Set the selected index (clamped to bounds).
    pub fn set_selected(&mut self, idx: usize) {
        if idx < self.items.len() {
            self.selected = idx;
        }
    }

    /// Clamp the selected index to be within bounds after items are removed.
    pub fn clamp_selected(&mut self) {
        if !self.items.is_empty() {
            self.selected = self.selected.min(self.items.len() - 1);
        }
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

    #[test]
    fn test_set_selected() {
        let mut list = SelectList::new("Pick:", vec!["a", "b", "c"]);
        list.set_selected(2);
        assert_eq!(list.selected_index(), 2);
        // Beyond bounds — should be ignored.
        list.set_selected(99);
        assert_eq!(list.selected_index(), 2);
        // Set back to 0.
        list.set_selected(0);
        assert_eq!(list.selected_index(), 0);
    }
}
