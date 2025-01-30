//! Query optimization rules for the query planner.
//!
//! This module contains optimization rules that can be applied to logical plans
//! to improve query performance. The rules are applied in sequence during the
//! optimization phase of query planning.

pub mod view;

pub use view::ViewOptimizationRule;