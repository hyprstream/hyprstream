//! # Hyprstream
//!
//! Hyprstream is a real-time metrics aggregation and caching service built on Apache Arrow Flight SQL.
//!
//! ## Features
//!
//! - Real-time metrics aggregation
//! - Multiple storage backends (DuckDB, ADBC)
//! - Intelligent caching
//! - Arrow Flight SQL interface
//!
//! ## Library Usage
//!
//! ```rust,no_run
//! use hyprstream::service::FlightServiceImpl;
//! use hyprstream::storage::duckdb::DuckDbBackend;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a DuckDB backend with in-memory storage
//!     let backend = DuckDbBackend::new(":memory:")?;
//!     let backend = Arc::new(backend);
//!     backend.init().await?;
//!
//!     // Create the Flight SQL service
//!     let service = FlightServiceImpl::new(backend);
//!
//!     // Use the service in your application...
//!     Ok(())
//! }
//! ```

pub mod config;
pub mod metrics;
pub mod service;
pub mod storage;

// Re-export commonly used items
pub use crate::metrics::MetricRecord;
pub use crate::service::FlightServiceImpl;
pub use crate::storage::{
    adbc::AdbcBackend,
    cached::CachedStorageBackend,
    duckdb::DuckDbBackend,
    StorageBackend,
}; 