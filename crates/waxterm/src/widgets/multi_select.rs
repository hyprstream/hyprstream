use std::fmt::Display;

use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::input::KeyPress;

use super::WidgetResult;

/// Multi-selection checkbox list (arrow keys + Space to toggle + Enter to confirm).
pub struct MultiSelectList<T> {
    label: String,
    items: Vec<T>,
    checked: Vec<bool>,
    cursor: usize,
}

impl<T: Display + Clone> MultiSelectList<T> {
    pub fn new(label: impl Into<String>, items: Vec<T>) -> Self {
        let len = items.len();
        Self {
            label: label.into(),
            items,
            checked: vec![false; len],
            cursor: 0,
        }
    }

    /// Returns indices of all checked items.
    pub fn selected_indices(&self) -> Vec<usize> {
        self.checked
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| if c { Some(i) } else { None })
            .collect()
    }

    pub fn handle_key(&mut self, key: &KeyPress) -> WidgetResult<Vec<T>> {
        match key {
            KeyPress::ArrowUp => {
                if self.cursor > 0 {
                    self.cursor -= 1;
                }
                WidgetResult::Pending
            }
            KeyPress::ArrowDown => {
                if self.cursor + 1 < self.items.len() {
                    self.cursor += 1;
                }
                WidgetResult::Pending
            }
            KeyPress::Char(b' ') => {
                if self.cursor < self.checked.len() {
                    self.checked[self.cursor] = !self.checked[self.cursor];
                }
                WidgetResult::Pending
            }
            KeyPress::Enter => {
                let selected: Vec<T> = self
                    .items
                    .iter()
                    .zip(self.checked.iter())
                    .filter_map(|(item, &c)| if c { Some(item.clone()) } else { None })
                    .collect();
                WidgetResult::Confirmed(selected)
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
            let check = if self.checked[i] { "[x]" } else { "[ ]" };
            let marker = if i == self.cursor { "> " } else { "  " };
            let style = if i == self.cursor {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(
                format!("{marker}{check} {item}"),
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
    fn test_toggle_and_confirm() {
        let mut ms = MultiSelectList::new("Pick:", vec!["a", "b", "c"]);
        // Toggle first item
        ms.handle_key(&KeyPress::Char(b' '));
        // Move down, toggle third
        ms.handle_key(&KeyPress::ArrowDown);
        ms.handle_key(&KeyPress::ArrowDown);
        ms.handle_key(&KeyPress::Char(b' '));
        assert_eq!(
            ms.handle_key(&KeyPress::Enter),
            WidgetResult::Confirmed(vec!["a", "c"])
        );
    }

    #[test]
    fn test_empty_confirm() {
        let mut ms = MultiSelectList::new("Pick:", vec!["a", "b"]);
        assert_eq!(
            ms.handle_key(&KeyPress::Enter),
            WidgetResult::Confirmed(Vec::<&str>::new())
        );
    }

    #[test]
    fn test_selected_indices() {
        let mut ms = MultiSelectList::new("Pick:", vec!["a", "b", "c"]);
        ms.handle_key(&KeyPress::ArrowDown);
        ms.handle_key(&KeyPress::Char(b' '));
        assert_eq!(ms.selected_indices(), vec![1]);
    }
}
