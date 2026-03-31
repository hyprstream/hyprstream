//! Reusable form widgets for terminal applications.
//!
//! Each widget handles keyboard input via [`handle_key`] and renders via a method
//! taking `&mut Frame` and `Rect`. All widgets return [`WidgetResult<T>`] to signal
//! whether input is still pending, confirmed, or cancelled.

mod confirm;
mod multi_select;
pub mod numeric_field;
mod select_list;
pub mod settings_form;
mod text_input;

pub use confirm::ConfirmDialog;
pub use multi_select::MultiSelectList;
pub use numeric_field::NumericField;
pub use select_list::SelectList;
pub use settings_form::SettingsForm;
pub use text_input::TextInput;

/// Result of a widget's key handling.
#[derive(Debug, PartialEq, Eq)]
pub enum WidgetResult<T> {
    /// Still accepting input.
    Pending,
    /// User confirmed a value (Enter).
    Confirmed(T),
    /// User cancelled (Escape).
    Cancelled,
}
