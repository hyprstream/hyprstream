pub mod aggregation;
pub mod cli;
pub mod config;
pub mod metrics;
pub mod models;
pub mod service;
pub mod storage;

// Re-export commonly used types
pub use service::FlightSqlService;
pub use aggregation::{AggregateFunction, GroupBy, AggregateResult, TimeWindow};