pub mod aggregation;
pub mod cli;
pub mod config;
pub mod metrics;
pub mod models;
pub mod query;
pub mod service;
pub mod storage;

// Re-export commonly used types
pub use aggregation::{AggregateFunction, AggregateResult, GroupBy, TimeWindow};
pub use query::{
    DataFusionExecutor, DataFusionPlanner, ExecutorConfig, OptimizationHint, Query, QueryEngine,
    QueryEngineBuilder, QueryExecutor, QueryPlanner,
};
pub use service::FlightSqlService;
