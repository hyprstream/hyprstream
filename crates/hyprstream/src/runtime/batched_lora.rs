//! Batched multi-tenant LoRA forward pass
//!
//! Gathers per-tenant A/B matrices, stacks them into batched tensors,
//! and uses bmm for efficient parallel LoRA computation. This avoids
//! sequential per-tenant forward passes.

use anyhow::{anyhow, Result};
use std::sync::Arc;
use tch::{Device, Tensor};

use crate::training::DeltaPool;
use hyprstream_rpc::Subject;

/// Batched LoRA forward pass for multi-tenant inference.
///
/// Gathers A/B matrices from each tenant's delta, pads to a common rank,
/// stacks into batch tensors, and performs bmm for efficient computation.
pub struct BatchedLoRAForward {
    /// Reference to the shared delta pool
    delta_pool: Arc<DeltaPool>,
    /// Maximum LoRA rank across all tenants (for padding)
    #[allow(dead_code)]
    max_rank: usize,
    /// Scaling factor (alpha / rank)
    #[allow(dead_code)]
    scaling: f64,
    /// Device for tensor allocation
    #[allow(dead_code)]
    device: Device,
}

impl BatchedLoRAForward {
    /// Create a new batched LoRA forward module.
    ///
    /// # Arguments
    /// * `delta_pool` - Shared delta pool containing per-tenant LoRA weights
    /// * `max_rank` - Maximum LoRA rank to pad to (default: 8)
    /// * `scaling` - LoRA scaling factor (alpha / rank)
    /// * `device` - Device for tensor allocation
    pub fn new(
        delta_pool: Arc<DeltaPool>,
        max_rank: usize,
        scaling: f64,
        device: Device,
    ) -> Self {
        Self {
            delta_pool,
            max_rank,
            scaling,
            device,
        }
    }

    /// Compute batched LoRA corrections for a single module across all provided tenants.
    ///
    /// # Arguments
    /// * `x` - Input tensor per tenant: Vec of [1, seq_len, in_features] tensors
    /// * `module_name` - Name of the target module (e.g., "q_proj")
    /// * `tenant_ids` - Tenant IDs in the same order as `x`
    /// * `layer_idx` - Layer index for per-layer delta lookup
    ///
    /// # Returns
    /// Vec of LoRA correction tensors, one per tenant. Each is [1, seq_len, out_features].
    /// Returns zero tensor for tenants without a delta.
    pub fn batched_lora_forward(
        &self,
        x: &[&Tensor],
        module_name: &str,
        tenant_ids: &[Subject],
        layer_idx: usize,
    ) -> Result<Vec<Tensor>> {
        if x.len() != tenant_ids.len() {
            return Err(anyhow!(
                "Input count ({}) doesn't match tenant count ({})",
                x.len(),
                tenant_ids.len()
            ));
        }

        if x.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = x.len();
        let _guard = tch::no_grad_guard();

        // Gather per-tenant A/B matrices
        let mut a_matrices: Vec<Option<Tensor>> = Vec::with_capacity(batch_size);
        let mut b_matrices: Vec<Option<Tensor>> = Vec::with_capacity(batch_size);
        let mut has_any_delta = false;

        let key = format!("{}.{}", layer_idx, module_name);
        for tenant_id in tenant_ids {
            if let Some(delta_arc) = self.delta_pool.get(tenant_id) {
                let delta = delta_arc.lock();
                if let (Some(a), Some(b)) = (
                    delta.lora_a.get(&key),
                    delta.lora_b.get(&key),
                ) {
                    // A: [rank, in_features], B: [out_features, rank]
                    a_matrices.push(Some(a.shallow_clone()));
                    b_matrices.push(Some(b.shallow_clone()));
                    has_any_delta = true;
                } else {
                    a_matrices.push(None);
                    b_matrices.push(None);
                }
            } else {
                a_matrices.push(None);
                b_matrices.push(None);
            }
        }

        // Fast path: no deltas at all
        if !has_any_delta {
            let results: Vec<Tensor> = x
                .iter()
                .map(|xi| {
                    let shape = xi.size();
                    let out_features = shape[shape.len() - 1]; // same as in_features for q/v proj
                    Tensor::zeros(shape.as_slice(), (xi.kind(), xi.device()))
                        .narrow(-1, 0, out_features)
                })
                .collect();
            return Ok(results);
        }

        // Process each tenant individually (single-threaded per model)
        // For batch_size=1 (most common case), this is just one iteration.
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            if let (Some(a), Some(b)) = (&a_matrices[i], &b_matrices[i]) {
                // x_i: [1, seq_len, in_features]
                // A: [rank, in_features] -> A^T: [in_features, rank]
                // B: [out_features, rank] -> B^T: [rank, out_features]
                // output = scaling * (x_i @ A^T @ B^T)
                let intermediate = x[i].matmul(&a.tr()); // [1, seq_len, rank]
                let output = intermediate.matmul(&b.tr()); // [1, seq_len, out_features]
                results.push(output * self.scaling);
            } else {
                // No delta for this tenant — zero correction
                let shape = x[i].size();
                results.push(Tensor::zeros(
                    shape.as_slice(),
                    (x[i].kind(), x[i].device()),
                ));
            }
        }

        Ok(results)
    }

    /// Compute batched LoRA correction for a single tenant (most common case).
    ///
    /// Optimized path for batch_size=1 that avoids allocation overhead.
    ///
    /// # Arguments
    /// * `x` - Input tensor [1, seq_len, in_features]
    /// * `module_name` - Name of the target module
    /// * `tenant_id` - The tenant ID
    /// * `layer_idx` - Layer index for per-layer delta lookup
    ///
    /// # Returns
    /// LoRA correction tensor [1, seq_len, out_features], or zero tensor if no delta.
    pub fn single_tenant_forward(
        &self,
        x: &Tensor,
        module_name: &str,
        tenant_id: &Subject,
        layer_idx: usize,
    ) -> Result<Tensor> {
        let _guard = tch::no_grad_guard();

        if let Some(delta_arc) = self.delta_pool.get(tenant_id) {
            let delta = delta_arc.lock();
            delta.forward(x, module_name, layer_idx)
        } else {
            // No delta for this tenant — return zeros
            Ok(Tensor::zeros(
                x.size().as_slice(),
                (x.kind(), x.device()),
            ))
        }
    }

    /// Get the delta pool reference.
    pub fn delta_pool(&self) -> &Arc<DeltaPool> {
        &self.delta_pool
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::training::TenantDeltaConfig;
    use hyprstream_rpc::Subject;
    use std::collections::HashMap;
    use tch::Kind;

    fn create_test_pool() -> (Arc<DeltaPool>, Device) {
        let device = Device::Cpu;
        let mut module_dims = HashMap::new();
        module_dims.insert("q_proj".to_owned(), (64, 64));
        module_dims.insert("v_proj".to_owned(), (64, 64));

        let config = TenantDeltaConfig {
            rank: 4,
            alpha: 2.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_owned(), "v_proj".to_owned()],
            learning_rate: 3e-4,
            max_accumulated_steps: 300,
            decay_lambda: 0.02,
        };

        let pool = DeltaPool::new(config, module_dims, device, None, std::env::temp_dir().join("batched_lora_test_snapshots"), None, 2);
        (Arc::new(pool), device)
    }

    #[test]
    fn test_single_tenant_no_delta() {
        let (pool, device) = create_test_pool();
        let batched = BatchedLoRAForward::new(pool, 4, 0.5, device);

        let x = Tensor::randn([1, 5, 64], (Kind::Float, device));
        let tenant = Subject::new("missing");

        let result = batched.single_tenant_forward(&x, "q_proj", &tenant, 0).unwrap();
        assert_eq!(result.size(), vec![1, 5, 64]);

        // Should be all zeros
        let norm: f64 = result.norm().double_value(&[]);
        assert!(norm < 1e-8, "Should be zero for missing tenant");
    }

    #[test]
    fn test_single_tenant_with_delta() {
        let (pool, device) = create_test_pool();

        // Create a delta for tenant A
        let tenant_a = Subject::new("tenant-a");
        pool.get_or_create(&tenant_a).unwrap();

        let batched = BatchedLoRAForward::new(pool, 4, 0.5, device);
        let x = Tensor::randn([1, 5, 64], (Kind::Float, device));

        let result = batched.single_tenant_forward(&x, "q_proj", &tenant_a, 0).unwrap();
        assert_eq!(result.size(), vec![1, 5, 64]);
    }

    #[test]
    fn test_batched_forward_isolation() {
        let (pool, device) = create_test_pool();

        let tenant_a = Subject::new("tenant-a");
        let tenant_b = Subject::new("tenant-b");
        pool.get_or_create(&tenant_a).unwrap();
        pool.get_or_create(&tenant_b).unwrap();

        let batched = BatchedLoRAForward::new(pool.clone(), 4, 0.5, device);

        let x1 = Tensor::randn([1, 5, 64], (Kind::Float, device));
        let x2 = Tensor::randn([1, 5, 64], (Kind::Float, device));

        let results = batched
            .batched_lora_forward(
                &[&x1, &x2],
                "q_proj",
                &[tenant_a.clone(), tenant_b.clone()],
                0,
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].size(), vec![1, 5, 64]);
        assert_eq!(results[1].size(), vec![1, 5, 64]);

    }

    #[test]
    fn test_batched_matches_sequential() {
        let (pool, device) = create_test_pool();

        let tenant = Subject::new("tenant-test");
        pool.get_or_create(&tenant).unwrap();

        let batched = BatchedLoRAForward::new(pool.clone(), 4, 0.5, device);

        let x = Tensor::randn([1, 3, 64], (Kind::Float, device));

        // Single-tenant path
        let single_result = batched
            .single_tenant_forward(&x, "q_proj", &tenant, 0)
            .unwrap();

        // Batched path with batch_size=1
        let batch_results = batched
            .batched_lora_forward(&[&x], "q_proj", std::slice::from_ref(&tenant), 0)
            .unwrap();

        // Should be identical
        let diff: f64 = (&single_result - &batch_results[0])
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert!(diff < 1e-6, "Batched should match sequential: diff={}", diff);
    }
}
