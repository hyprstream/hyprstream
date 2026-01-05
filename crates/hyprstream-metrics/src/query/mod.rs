//! Query planning and execution module.
//!
//! This module provides the core query processing functionality:
//! - Query planning and optimization
//! - Physical plan execution
//! - Query optimization rules
//! - Query orchestration with statement caching

pub mod executor;
pub mod orchestrator;
pub mod physical;
pub mod planner;
pub mod rules;

pub use executor::{DataFusionExecutor, ExecutorConfig, QueryExecutor};
pub use orchestrator::{CachedStatement, QueryOrchestrator};
pub use physical::{PhysicalOperator, VectorizedOperator};
pub use planner::{
    DataFusionPlanner, OptimizationHint, OptimizerContext, Query, QueryPlanner, Statistics,
};
pub use rules::ViewOptimizationRule;
