//! Query planning and execution module.
//!
//! This module provides the core query processing functionality:
//! - Query planning and optimization
//! - Physical plan execution
//! - Query optimization rules
//! - Query execution context

pub mod executor;
pub mod physical;
pub mod planner;
pub mod rules;

pub use executor::{DataFusionExecutor, ExecutorConfig, QueryExecutor};
pub use physical::{PhysicalOperator, VectorizedOperator};
pub use planner::{
    DataFusionPlanner, OptimizationHint, OptimizerContext, Query, QueryPlanner, Statistics,
};
pub use rules::ViewOptimizationRule;
