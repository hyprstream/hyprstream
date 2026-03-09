//! Muon optimizer: MomentUm Orthogonalized by Newton-Schulz.
//!
//! Designed for 2D weight matrices (LoRA A/B). Orthogonalizes gradient
//! updates so all singular values → 1, making each step maximally effective.
//!
//! Reference: https://github.com/vukrosic/muon-optimizer-guide
//! Algorithm: Nesterov momentum + Newton-Schulz (Polar Express variant)

use std::collections::HashMap;
use tch::{Kind, Tensor};

/// Polar Express coefficients for Newton-Schulz iteration.
/// Pre-optimized for fast convergence in bfloat16.
const POLAR_EXPRESS_COEFFS: [(f64, f64, f64); 5] = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
];

/// Newton-Schulz iteration for matrix orthogonalization.
///
/// Maps G → UV^T (its orthogonal factor) by converging all singular values
/// to 1, without ever computing an expensive SVD.
///
/// Uses the Polar Express variant with pre-optimized coefficients for
/// stable bfloat16 computation.
///
/// # Arguments
/// * `g` - 2D gradient matrix [rows, cols]
/// * `steps` - Number of Newton-Schulz iterations (default: 5)
pub fn newton_schulz_orthogonalize(g: &Tensor, steps: usize) -> Tensor {
    let _guard = tch::no_grad_guard();
    assert!(
        g.dim() == 2,
        "Newton-Schulz requires 2D matrix, got {}D",
        g.dim()
    );
    assert!(steps <= POLAR_EXPRESS_COEFFS.len());

    let mut x = g.to_kind(Kind::Float);

    // Algorithm assumes wide (rows ≤ cols) matrices — transpose tall ones
    let transposed = x.size()[0] > x.size()[1];
    if transposed {
        x = x.tr();
    }

    // Normalize so largest singular value starts near 1
    let norm = x.norm_scalaropt_dim(2i64, &[-2i64, -1i64][..], true);
    x = &x / (&norm * 1.01 + 1e-7);

    // Newton-Schulz iterations — converge singular values to 1
    for &(a, b, c) in &POLAR_EXPRESS_COEFFS[..steps] {
        let a_mat = x.matmul(&x.tr()); // A = X @ X^T (small: [min_dim, min_dim])
        x = a * &x + (b * &a_mat + c * a_mat.matmul(&a_mat)).matmul(&x);
    }

    if transposed {
        x.tr()
    } else {
        x
    }
}

/// Muon optimizer configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MuonConfig {
    /// Learning rate (default: 0.02 — much larger than AdamW's 3e-4)
    #[serde(default = "default_muon_lr")]
    pub lr: f64,
    /// Nesterov momentum coefficient (default: 0.95)
    #[serde(default = "default_muon_momentum")]
    pub momentum: f64,
    /// Weight decay (default: 0.0 — rarely needed for Muon-optimized params)
    #[serde(default)]
    pub weight_decay: f64,
    /// Number of Newton-Schulz iterations (default: 5)
    #[serde(default = "default_ns_steps")]
    pub ns_steps: usize,
}

fn default_muon_lr() -> f64 {
    0.02
}
pub(crate) fn default_muon_momentum() -> f64 {
    0.95
}
fn default_ns_steps() -> usize {
    5
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self {
            lr: default_muon_lr(),
            momentum: default_muon_momentum(),
            weight_decay: 0.0,
            ns_steps: default_ns_steps(),
        }
    }
}

/// Per-parameter state for Muon optimizer.
#[derive(Debug, Default)]
pub struct MuonState {
    pub momentum_buffer: Option<Tensor>,
}

impl MuonState {
    pub fn new() -> Self {
        Self {
            momentum_buffer: None,
        }
    }
}

/// Perform a single Muon optimization step on a 2D parameter.
///
/// Algorithm:
/// 1. Nesterov momentum: buf = lerp(buf, grad, 1-momentum); g = lerp(grad, buf, momentum)
/// 2. Orthogonalize: g = newton_schulz(g) → UV^T (all SVs = 1)
/// 3. Weight decay: p *= (1 - lr * wd)
/// 4. Scale for rectangularity: scale = sqrt(max(1, rows/cols))
/// 5. Update: p -= lr * scale * g
pub fn muon_step(param: &Tensor, state: &mut MuonState, config: &MuonConfig) {
    let _guard = tch::no_grad_guard();
    let grad = param.grad();
    if !grad.defined() {
        return;
    }

    // 1. Nesterov momentum
    let buf = state
        .momentum_buffer
        .get_or_insert_with(|| Tensor::zeros_like(&grad));
    let _ = buf.lerp_(&grad, 1.0 - config.momentum);
    let g = grad.lerp(buf, config.momentum);

    // 2. Orthogonalize
    let g_flat = g.view([g.size()[0], -1]);
    let g_orth = newton_schulz_orthogonalize(&g_flat, config.ns_steps);
    let g_orth = g_orth.view_as(&g).to_kind(param.kind());

    // 3. Weight decay (decoupled) + 4. Scale for rectangularity + 5. Update
    // Combined into a single fused operation to minimize tensor allocations.
    let (rows, cols) = (param.size()[0] as f64, param.size()[1] as f64);
    let scale = (1.0f64.max(rows / cols)).sqrt();
    let decay = 1.0 - config.lr * config.weight_decay; // 1.0 when weight_decay=0
    // p = p * decay - lr * scale * g_orth
    let updated = &param.data() * decay - &g_orth * (config.lr * scale);
    param.data().copy_(&updated);
}

/// Snapshot all Muon momentum buffers for rollback.
pub fn snapshot_muon_states(states: &HashMap<String, MuonState>) -> HashMap<String, Tensor> {
    let _guard = tch::no_grad_guard();
    states
        .iter()
        .filter_map(|(k, s)| s.momentum_buffer.as_ref().map(|b| (k.clone(), b.copy())))
        .collect()
}

/// Restore Muon momentum buffers from a snapshot.
pub fn restore_muon_states(
    states: &mut HashMap<String, MuonState>,
    snapshot: &HashMap<String, Tensor>,
) {
    let _guard = tch::no_grad_guard();
    for (k, snap_buf) in snapshot {
        if let Some(state) = states.get_mut(k) {
            if let Some(ref mut buf) = state.momentum_buffer {
                buf.copy_(snap_buf);
            } else {
                state.momentum_buffer = Some(snap_buf.copy());
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_schulz_produces_orthogonal() {
        let _guard = tch::no_grad_guard();
        let g = Tensor::randn([8, 64], (Kind::Float, tch::Device::Cpu));
        let q = newton_schulz_orthogonalize(&g, 5);
        // Q should satisfy Q @ Q^T ≈ I (orthonormal rows)
        let qqt = q.matmul(&q.tr());
        let eye = Tensor::eye(8, (Kind::Float, tch::Device::Cpu));
        let diff = (&qqt - &eye).abs().max().double_value(&[]);
        // Polar Express coefficients are optimized for bfloat16; in float32
        // convergence is slightly looser but still well-orthogonal
        assert!(
            diff < 0.2,
            "Q@Q^T should be near identity, max diff = {diff}"
        );
    }

    #[test]
    fn test_newton_schulz_tall_matrix() {
        let _guard = tch::no_grad_guard();
        // Tall matrix [64, 8] — should transpose internally
        let g = Tensor::randn([64, 8], (Kind::Float, tch::Device::Cpu));
        let q = newton_schulz_orthogonalize(&g, 5);
        assert_eq!(q.size(), &[64, 8]);
        // Q^T @ Q ≈ I (orthonormal columns for tall)
        let qtq = q.tr().matmul(&q);
        let eye = Tensor::eye(8, (Kind::Float, tch::Device::Cpu));
        let diff = (&qtq - &eye).abs().max().double_value(&[]);
        assert!(
            diff < 0.3,
            "Q^T@Q should be near identity, max diff = {diff}"
        );
    }

    #[test]
    fn test_muon_step_updates_params() {
        let p = Tensor::randn([8, 64], (Kind::Float, tch::Device::Cpu)).set_requires_grad(true);
        let p_orig = p.data().copy();

        // Generate gradient via backward
        let loss = p.sum(Kind::Float);
        loss.backward();

        let mut state = MuonState::new();
        let config = MuonConfig::default();
        muon_step(&p, &mut state, &config);

        let diff = (&p.data() - &p_orig)
            .abs()
            .sum(Kind::Double)
            .double_value(&[]);
        assert!(diff > 0.0, "Parameter should have been updated");
    }

    #[test]
    fn test_muon_momentum_accumulates() {
        let p = Tensor::randn([8, 64], (Kind::Float, tch::Device::Cpu)).set_requires_grad(true);
        let config = MuonConfig::default();
        let mut state = MuonState::new();

        // Step 1
        let loss1 = p.sum(Kind::Float);
        loss1.backward();
        muon_step(&p, &mut state, &config);
        assert!(state.momentum_buffer.is_some());

        // Step 2 — use different gradient
        let _ = p.grad().zero_();
        let loss2 = (&p * 2.0).sum(Kind::Float);
        loss2.backward();
        let buf_before = state.momentum_buffer.as_ref().unwrap().copy();
        muon_step(&p, &mut state, &config);
        let buf_after = state.momentum_buffer.as_ref().unwrap();
        let diff = (&buf_before - buf_after)
            .abs()
            .sum(Kind::Double)
            .double_value(&[]);
        assert!(diff > 0.0, "Momentum buffer should change across steps");
    }

    #[test]
    fn test_muon_config_defaults() {
        let config = MuonConfig::default();
        assert!((config.lr - 0.02).abs() < 1e-9);
        assert!((config.momentum - 0.95).abs() < 1e-9);
        assert_eq!(config.ns_steps, 5);
    }

    #[test]
    fn test_muon_weight_decay() {
        let p = Tensor::ones([4, 8], (Kind::Float, tch::Device::Cpu)).set_requires_grad(true);

        // Create gradient, then zero it — only weight decay should take effect
        let loss = p.sum(Kind::Float);
        loss.backward();
        let _ = p.grad().zero_();

        let config = MuonConfig {
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut state = MuonState::new();
        muon_step(&p, &mut state, &config);

        // With zero grad and weight decay, parameter should shrink
        let max_val = p.data().max().double_value(&[]);
        assert!(max_val < 1.0, "Weight decay should shrink parameters");
    }
}
