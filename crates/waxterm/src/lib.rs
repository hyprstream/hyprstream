pub mod app;
#[cfg(feature = "avt")]
pub mod avt_style;
pub mod backend;
pub mod input;
pub mod runtime;
pub mod sgr;
pub mod structured;
pub mod widgets;

pub use app::{TerminalApp, TerminalConfig};
pub use backend::AnsiBackend;
pub use input::{InputParser, KeyPress, OscHandler, StdinPoller};
pub use runtime::run_sync;

#[cfg(feature = "tokio")]
pub use runtime::run_async;
