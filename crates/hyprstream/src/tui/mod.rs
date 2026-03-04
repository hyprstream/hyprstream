//! TUI display server — terminal multiplexer over ZMQ RPC.
//!
//! Provides session persistence, multi-pane layouts, remote access,
//! and MCP-controllable display surfaces.
//!
//! # Architecture
//!
//! ```text
//! App (ratatui)  ──→  TuiPaneBackend  ──→  TuiPane (cell buffer)
//!                                                │
//!                     Frame loop (33ms) ─────────┘
//!                           │
//!             ┌─────────────┼─────────────┐
//!             ▼             ▼             ▼
//!        ANSI viewer   capnp viewer  structured viewer
//!        (terminal)    (WebTransport) (StructuredBackend)
//! ```

pub mod state;
pub mod diff;
pub mod ninep;
pub mod process;
pub mod service;
pub mod wt_viewer;
pub mod backend;
pub mod vte_parser;

pub use state::{
    CursorShape, CursorState, IngestionMode, LayoutNode, PaneBuffer, ScrollOp, TuiEvent,
    TuiPane, TuiSession, TuiState, TuiWindow,
};
