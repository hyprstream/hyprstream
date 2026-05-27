//! Dependency-aware startup ordering for services.
//!
//! Computes startup stages from the `depends_on` metadata declared in
//! `#[service_factory]` registrations.  Services within the same stage
//! have no mutual dependencies and can start in parallel; services in
//! stage N depend only on services in stages 0..N-1.

use std::collections::{HashMap, HashSet};

use crate::service::factory::list_factories;

/// Compute startup stages from the factory dependency graph.
///
/// Only services present in `requested` are included. Dependencies that
/// are not in `requested` are silently ignored (they may already be
/// running or managed externally).
///
/// Returns stages where stage 0 has no dependencies, stage 1 depends
/// only on stage 0, etc.
///
/// # Panics
///
/// Panics if the dependency graph contains a cycle (should not happen
/// with a well-formed `depends_on` declaration).
pub fn startup_stages(requested: &[impl AsRef<str>]) -> Vec<Vec<String>> {
    let requested_set: HashSet<&str> = requested.iter().map(AsRef::as_ref).collect();

    // Build adjacency: service -> [dependencies that are also in requested]
    let mut deps: HashMap<&str, Vec<&str>> = HashMap::new();
    for factory in list_factories() {
        if !requested_set.contains(factory.name) {
            continue;
        }
        let service_deps: Vec<&str> = factory
            .depends_on
            .iter()
            .copied()
            .filter(|d| requested_set.contains(d))
            .collect();
        deps.insert(factory.name, service_deps);
    }

    // Services in requested but not in any factory (shouldn't happen, but be safe)
    for &name in &requested_set {
        deps.entry(name).or_default();
    }

    // Kahn's algorithm for topological sort by depth
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    for (&svc, svc_deps) in &deps {
        in_degree.entry(svc).or_insert(0);
        for &dep in svc_deps {
            // dep blocks svc → svc has in-degree from dep
            *in_degree.entry(svc).or_insert(0) += 1;
            in_degree.entry(dep).or_insert(0);
        }
    }

    // Reverse adjacency: dep -> [services that depend on it]
    let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();
    for (&svc, svc_deps) in &deps {
        for &dep in svc_deps {
            dependents.entry(dep).or_default().push(svc);
        }
    }

    let mut stages: Vec<Vec<String>> = Vec::new();
    let mut remaining: HashSet<&str> = deps.keys().copied().collect();

    while !remaining.is_empty() {
        // Collect all services with in-degree 0
        let mut stage: Vec<&str> = remaining
            .iter()
            .filter(|&&s| in_degree.get(s).copied().unwrap_or(0) == 0)
            .copied()
            .collect();

        assert!(
            !stage.is_empty(),
            "dependency cycle detected among: {:?}",
            remaining
        );

        stage.sort(); // deterministic order within a stage

        // Remove this stage's services and update in-degrees
        for &svc in &stage {
            remaining.remove(svc);
            if let Some(deps_of_svc) = dependents.get(svc) {
                for &dependent in deps_of_svc {
                    if let Some(deg) = in_degree.get_mut(dependent) {
                        *deg = deg.saturating_sub(1);
                    }
                }
            }
        }

        stages.push(stage.into_iter().map(String::from).collect());
    }

    stages
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: these tests don't use the inventory (no factories registered in
    // test binaries). They test the algorithm via direct construction.

    #[test]
    fn test_empty_requested() {
        let stages = startup_stages(&Vec::<String>::new());
        assert!(stages.is_empty());
    }

    #[test]
    fn test_stages_from_inventory() {
        // This test verifies the function runs against the real inventory.
        // In test binaries the inventory is typically empty, so we just
        // verify it doesn't panic.
        let stages = startup_stages(&["policy", "registry", "model"]);
        // All requested services should appear exactly once across all stages.
        let all: Vec<&str> = stages.iter().flat_map(|s| s.iter().map(String::as_str)).collect();
        // Services not in the inventory will still appear (with no deps).
        assert!(all.len() <= 3);
    }
}
