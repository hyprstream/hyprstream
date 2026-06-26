//! Vertical stack of [`NumericField`] instances with field navigation.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::input::KeyPress;
use super::WidgetResult;
use super::numeric_field::NumericField;

/// A vertical form of numeric fields with arrow-key navigation.
///
/// Arrow up/down navigates between fields. All other keys delegate to the
/// currently selected field. When a field is in edit mode, up/down are
/// blocked until the user confirms (Enter) or cancels (Escape) the edit.
pub struct SettingsForm {
    fields: Vec<NumericField>,
    selected: usize,
}

impl SettingsForm {
    pub fn new(fields: Vec<NumericField>) -> Self {
        Self { fields, selected: 0 }
    }

    /// Index of the currently selected field.
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Whether the form has no fields.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Whether the currently selected field is in edit (text entry) mode.
    pub fn is_editing(&self) -> bool {
        self.fields.get(self.selected).is_some_and(|f| f.is_editing())
    }

    /// Get all current field values.
    pub fn values(&self) -> Vec<Option<f64>> {
        self.fields.iter().map(|f| f.get_value()).collect()
    }

    /// Handle a key press.
    ///
    /// Returns `Confirmed(())` when Enter is pressed in browse mode (all fields committed).
    /// Returns `Cancelled` when Escape is pressed in browse mode.
    /// Returns `Pending` for all navigation and edit operations.
    pub fn handle_key(&mut self, key: &KeyPress) -> WidgetResult<()> {
        if self.fields.is_empty() {
            return match key {
                KeyPress::Enter => WidgetResult::Confirmed(()),
                KeyPress::Escape => WidgetResult::Cancelled,
                _ => WidgetResult::Pending,
            };
        }

        let field = &mut self.fields[self.selected];

        // If the field is in edit mode, delegate everything to it.
        if field.is_editing() {
            field.handle_key(key);
            return WidgetResult::Pending;
        }

        // Browse mode: arrow up/down navigates fields.
        match key {
            KeyPress::ArrowUp => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                WidgetResult::Pending
            }
            KeyPress::ArrowDown => {
                if self.selected + 1 < self.fields.len() {
                    self.selected += 1;
                }
                WidgetResult::Pending
            }
            key => {
                // Delegate to the field. Map field-level Confirmed/Cancelled
                // to form-level signals.
                match field.handle_key(key) {
                    WidgetResult::Confirmed(_) => WidgetResult::Confirmed(()),
                    WidgetResult::Cancelled => WidgetResult::Cancelled,
                    WidgetResult::Pending => WidgetResult::Pending,
                }
            }
        }
    }

    /// Render the form. Each field occupies one row.
    pub fn render(&self, frame: &mut Frame, area: Rect) {
        for (i, field) in self.fields.iter().enumerate() {
            if i as u16 >= area.height {
                break;
            }
            let row = Rect {
                y: area.y + i as u16,
                height: 1,
                ..area
            };
            field.render(frame, row, i == self.selected);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::numeric_field::NumericField;

    fn make_form() -> SettingsForm {
        SettingsForm::new(vec![
            NumericField::new("A").integer().range(0.0, 100.0).step(1.0).value(Some(10.0)),
            NumericField::new("B").float(2).range(0.0, 1.0).step(0.1).value(Some(0.5)),
            NumericField::new("C").integer().range(0.0, 50.0).step(5.0).value(Some(25.0)),
        ])
    }

    #[test]
    fn test_navigation() {
        let mut form = make_form();
        assert_eq!(form.selected(), 0);
        form.handle_key(&KeyPress::ArrowDown);
        assert_eq!(form.selected(), 1);
        form.handle_key(&KeyPress::ArrowDown);
        assert_eq!(form.selected(), 2);
        form.handle_key(&KeyPress::ArrowDown); // can't go past end
        assert_eq!(form.selected(), 2);
        form.handle_key(&KeyPress::ArrowUp);
        assert_eq!(form.selected(), 1);
    }

    #[test]
    fn test_enter_confirms_form() {
        let mut form = make_form();
        assert_eq!(form.handle_key(&KeyPress::Enter), WidgetResult::Confirmed(()));
    }

    #[test]
    fn test_escape_cancels_form() {
        let mut form = make_form();
        assert_eq!(form.handle_key(&KeyPress::Escape), WidgetResult::Cancelled);
    }

    #[test]
    fn test_arrow_adjusts_selected_field() {
        let mut form = make_form();
        form.handle_key(&KeyPress::ArrowRight); // increment field A
        assert_eq!(form.values()[0], Some(11.0));
        assert_eq!(form.values()[1], Some(0.5)); // B unchanged
    }

    #[test]
    fn test_edit_mode_blocks_navigation() {
        let mut form = make_form();
        form.handle_key(&KeyPress::Char(b'5')); // enter edit on field A
        assert!(form.is_editing());
        form.handle_key(&KeyPress::ArrowDown); // should NOT navigate — goes to field edit cursor
        assert_eq!(form.selected(), 0); // still on field A
    }

    #[test]
    fn test_values() {
        let form = make_form();
        assert_eq!(form.values(), vec![Some(10.0), Some(0.5), Some(25.0)]);
    }
}
