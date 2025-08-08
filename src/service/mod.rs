//! VDB-first service module for adaptive ML inference

pub mod embedding_flight;
pub mod flight_metric;

// Re-export main services
pub use embedding_flight::{EmbeddingFlightService, create_embedding_flight_server};
pub use flight_metric::MetricFlightSqlService;