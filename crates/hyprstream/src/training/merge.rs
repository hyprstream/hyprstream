//! LoRA adapter merging strategies
//!
//! Implements multiple strategies for merging accumulated tenant deltas
//! into permanent adapter files:
//!
//! - **Replace**: Completely replace the existing adapter weights
//! - **Additive**: Weighted sum of existing and new weights
//! - **DO-Merge**: Direction-Only Merging (Yang et al. 2025) — merges magnitude
//!   and direction independently for better quality preservation.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tch::Tensor;

/// Merge strategy for combining tenant delta with existing adapter.
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Replace existing weights entirely with new weights.
    Replace,
    /// Weighted additive merge: result = (1-w) * existing + w * new
    Additive { weight: f64 },
    /// Direction-Only Merge (Yang et al. 2025):
    /// Decomposes into magnitude + direction, merges independently.
    DoMerge { weight: f64 },
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::DoMerge { weight: 0.3 }
    }
}

impl MergeStrategy {
    /// Parse strategy from string name.
    pub fn from_name(name: &str, weight: f64) -> Result<Self> {
        match name {
            "replace" => Ok(Self::Replace),
            "additive" => Ok(Self::Additive { weight }),
            "do_merge" | "doMerge" | "DO-Merge" => Ok(Self::DoMerge { weight }),
            _ => Err(anyhow!("Unknown merge strategy: '{}'. Use 'replace', 'additive', or 'do_merge'.", name)),
        }
    }
}

/// Merge two state dicts using the specified strategy.
///
/// # Arguments
/// * `existing` - Current adapter weights (module_name -> tensor)
/// * `new` - New delta weights to merge in (module_name -> tensor)
/// * `strategy` - Merge strategy to use
///
/// # Returns
/// Merged state dict with the same keys as `existing`.
pub fn merge_state_dicts(
    existing: &HashMap<String, Tensor>,
    new: &HashMap<String, Tensor>,
    strategy: &MergeStrategy,
) -> Result<HashMap<String, Tensor>> {
    let _guard = tch::no_grad_guard();
    let mut result = HashMap::new();

    for (key, existing_tensor) in existing {
        let merged = if let Some(new_tensor) = new.get(key) {
            // Both exist — merge
            apply_strategy(existing_tensor, new_tensor, strategy)?
        } else {
            // Only existing — keep as-is
            existing_tensor.copy()
        };
        result.insert(key.clone(), merged);
    }

    // Also include keys only in `new`
    for (key, new_tensor) in new {
        if !existing.contains_key(key) {
            result.insert(key.clone(), new_tensor.copy());
        }
    }

    Ok(result)
}

/// Apply merge strategy to a single tensor pair.
fn apply_strategy(existing: &Tensor, new: &Tensor, strategy: &MergeStrategy) -> Result<Tensor> {
    match strategy {
        MergeStrategy::Replace => replace_merge(existing, new),
        MergeStrategy::Additive { weight } => additive_merge(existing, new, *weight),
        MergeStrategy::DoMerge { weight } => do_merge(existing, new, *weight),
    }
}

/// Replace merge: return a copy of the new tensor.
pub fn replace_merge(_existing: &Tensor, new: &Tensor) -> Result<Tensor> {
    Ok(new.copy())
}

/// Additive merge: result = (1-weight) * existing + weight * new
pub fn additive_merge(existing: &Tensor, new: &Tensor, weight: f64) -> Result<Tensor> {
    if existing.size() != new.size() {
        return Err(anyhow!(
            "Shape mismatch in additive merge: {:?} vs {:?}",
            existing.size(),
            new.size()
        ));
    }
    Ok(existing * (1.0 - weight) + new * weight)
}

/// Direction-Only Merge (DO-Merge) from Yang et al. 2025.
///
/// Decomposes each tensor into magnitude (Frobenius norm) and direction
/// (unit-norm tensor), merges them independently, then reconstructs.
///
/// This preserves the geometric structure of weight matrices better than
/// simple additive merging.
///
/// Formula:
///   magnitude_merged = (1-w) * ||existing|| + w * ||new||
///   direction_merged = normalize((1-w) * dir(existing) + w * dir(new))
///   result = magnitude_merged * direction_merged
pub fn do_merge(existing: &Tensor, new: &Tensor, weight: f64) -> Result<Tensor> {
    if existing.size() != new.size() {
        return Err(anyhow!(
            "Shape mismatch in DO-merge: {:?} vs {:?}",
            existing.size(),
            new.size()
        ));
    }

    // Compute magnitudes (Frobenius norms)
    let mag_existing: f64 = existing.norm().double_value(&[]);
    let mag_new: f64 = new.norm().double_value(&[]);

    // Handle zero-norm cases
    let eps = 1e-8;
    if mag_existing < eps && mag_new < eps {
        return Ok(Tensor::zeros(
            existing.size().as_slice(),
            (existing.kind(), existing.device()),
        ));
    }

    if mag_existing < eps {
        // Only new has content — scale by weight
        return Ok(new * weight);
    }
    if mag_new < eps {
        // Only existing has content — scale by (1-weight)
        return Ok(existing * (1.0 - weight));
    }

    // Compute direction vectors (unit norm)
    let dir_existing = existing / mag_existing;
    let dir_new = new / mag_new;

    // Merge magnitude: linear interpolation
    let mag_merged = (1.0 - weight) * mag_existing + weight * mag_new;

    // Merge direction: interpolate then re-normalize
    let dir_combined = &dir_existing * (1.0 - weight) + &dir_new * weight;
    let dir_combined_norm: f64 = dir_combined.norm().double_value(&[]);

    let dir_merged = if dir_combined_norm > eps {
        dir_combined / dir_combined_norm
    } else {
        // Directions cancelled out — fall back to existing direction
        dir_existing
    };

    // Reconstruct: magnitude * direction
    Ok(dir_merged * mag_merged)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use tch::Kind;

    #[test]
    fn test_replace_merge() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);

        let result = replace_merge(&a, &b).unwrap();
        let expected = vec![4.0f32, 5.0, 6.0];
        let result_vec: Vec<f32> = Vec::try_from(result).unwrap();
        assert_eq!(result_vec, expected);
    }

    #[test]
    fn test_additive_merge_equal_weight() {
        let a = Tensor::from_slice(&[2.0f32, 4.0, 6.0]);
        let b = Tensor::from_slice(&[4.0f32, 6.0, 8.0]);

        let result = additive_merge(&a, &b, 0.5).unwrap();
        let result_vec: Vec<f32> = Vec::try_from(result).unwrap();
        // (1-0.5)*[2,4,6] + 0.5*[4,6,8] = [1,2,3] + [2,3,4] = [3,5,7]
        assert_eq!(result_vec, vec![3.0f32, 5.0, 7.0]);
    }

    #[test]
    fn test_additive_merge_zero_weight() {
        let a = Tensor::from_slice(&[1.0f32, 2.0]);
        let b = Tensor::from_slice(&[9.0f32, 9.0]);

        let result = additive_merge(&a, &b, 0.0).unwrap();
        let result_vec: Vec<f32> = Vec::try_from(result).unwrap();
        assert_eq!(result_vec, vec![1.0f32, 2.0]);
    }

    #[test]
    fn test_additive_merge_full_weight() {
        let a = Tensor::from_slice(&[1.0f32, 2.0]);
        let b = Tensor::from_slice(&[9.0f32, 9.0]);

        let result = additive_merge(&a, &b, 1.0).unwrap();
        let result_vec: Vec<f32> = Vec::try_from(result).unwrap();
        assert_eq!(result_vec, vec![9.0f32, 9.0]);
    }

    #[test]
    fn test_do_merge_self_half_weight() {
        // Merging a tensor with itself at w=0.5 should give the same tensor
        let a = Tensor::from_slice(&[3.0f32, 4.0]); // norm = 5
        let result = do_merge(&a, &a, 0.5).unwrap();

        let diff: f64 = (&result - &a).abs().sum(Kind::Float).double_value(&[]);
        assert!(diff < 1e-5, "DO-merge with self at w=0.5 should be identity, diff={}", diff);
    }

    #[test]
    fn test_do_merge_full_weight_is_replacement() {
        let a = Tensor::from_slice(&[1.0f32, 0.0]);
        let b = Tensor::from_slice(&[0.0f32, 2.0]);

        let result = do_merge(&a, &b, 1.0).unwrap();
        let diff: f64 = (&result - &b).abs().sum(Kind::Float).double_value(&[]);
        assert!(diff < 1e-5, "DO-merge at w=1.0 should be replacement, diff={}", diff);
    }

    #[test]
    fn test_do_merge_preserves_scale() {
        // DO-merge should interpolate magnitudes linearly
        let a = Tensor::from_slice(&[3.0f32, 4.0]); // norm = 5
        let b = Tensor::from_slice(&[6.0f32, 8.0]); // norm = 10, same direction

        let result = do_merge(&a, &b, 0.5).unwrap();
        let result_norm: f64 = result.norm().double_value(&[]);

        // Expected magnitude: (1-0.5)*5 + 0.5*10 = 7.5
        assert!(
            (result_norm - 7.5).abs() < 0.01,
            "Expected norm ~7.5, got {}",
            result_norm
        );
    }

    #[test]
    fn test_do_merge_zero_tensors() {
        let a = Tensor::zeros([3], (Kind::Float, tch::Device::Cpu));
        let b = Tensor::zeros([3], (Kind::Float, tch::Device::Cpu));

        let result = do_merge(&a, &b, 0.5).unwrap();
        let norm: f64 = result.norm().double_value(&[]);
        assert!(norm < 1e-8, "Merging zeros should produce zeros");
    }

    #[test]
    fn test_merge_state_dicts() {
        let mut existing = HashMap::new();
        existing.insert("a".to_owned(), Tensor::from_slice(&[1.0f32, 2.0]));
        existing.insert("b".to_owned(), Tensor::from_slice(&[3.0f32, 4.0]));

        let mut new = HashMap::new();
        new.insert("a".to_owned(), Tensor::from_slice(&[5.0f32, 6.0]));
        // "b" not in new — should keep existing

        let strategy = MergeStrategy::Additive { weight: 1.0 };
        let result = merge_state_dicts(&existing, &new, &strategy).unwrap();

        assert_eq!(result.len(), 2);
        // "a" should be fully replaced (weight=1.0)
        let a_vec: Vec<f32> = Vec::try_from(result["a"].shallow_clone()).unwrap();
        assert_eq!(a_vec, vec![5.0f32, 6.0]);
        // "b" should be unchanged
        let b_vec: Vec<f32> = Vec::try_from(result["b"].shallow_clone()).unwrap();
        assert_eq!(b_vec, vec![3.0f32, 4.0]);
    }

    #[test]
    fn test_shape_mismatch() {
        let a = Tensor::from_slice(&[1.0f32, 2.0]);
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);

        assert!(additive_merge(&a, &b, 0.5).is_err());
        assert!(do_merge(&a, &b, 0.5).is_err());
    }

    #[test]
    fn test_strategy_from_name() {
        assert!(matches!(MergeStrategy::from_name("replace", 0.5).unwrap(), MergeStrategy::Replace));
        assert!(matches!(MergeStrategy::from_name("additive", 0.5).unwrap(), MergeStrategy::Additive { .. }));
        assert!(matches!(MergeStrategy::from_name("do_merge", 0.3).unwrap(), MergeStrategy::DoMerge { .. }));
        assert!(MergeStrategy::from_name("unknown", 0.5).is_err());
    }
}
