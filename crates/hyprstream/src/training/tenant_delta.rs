//! Per-tenant LoRA delta for isolated Test-Time Training
//!
//! Each tenant gets their own LoRA delta (A/B weight matrices) that accumulates
//! adaptations across requests. Deltas are trained via SGD and can be persisted
//! to content-addressed storage or merged into permanent adapter files.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};
use safetensors::tensor::{TensorView, SafeTensors};
use super::muon::{MuonConfig, MuonState};

/// Extract LoRA components from hierarchical tensor names
/// e.g., "base_model.model.layers.5.self_attn.q_proj.lora_A.weight" -> Some((5, "q_proj", "lora_a"))
pub fn extract_lora_components(tensor_name: &str) -> Option<(usize, String, String)> {
    let parts: Vec<&str> = tensor_name.split('.').collect();

    // Find the lora_a/lora_b/lora_A/lora_B component
    for (i, &part) in parts.iter().enumerate() {
        let normalized = part.to_lowercase();
        if (normalized == "lora_a" || normalized == "lora_b") && i > 0 {
            let module = parts[i - 1];
            if matches!(
                module,
                "q_proj"
                    | "k_proj"
                    | "v_proj"
                    | "o_proj"
                    | "gate_proj"
                    | "up_proj"
                    | "down_proj"
            ) {
                // Find layer index: look for "layers" followed by a number
                let layer_idx = parts.iter()
                    .zip(parts.iter().skip(1))
                    .find_map(|(&a, &b)| {
                        if a == "layers" { b.parse::<usize>().ok() } else { None }
                    })
                    .unwrap_or(0);
                return Some((layer_idx, module.to_owned(), normalized));
            }
        }
    }

    None
}

/// Per-layer LoRA rank and target module override.
///
/// When present in `TenantDeltaConfig::layer_overrides`, these values replace
/// the top-level `rank` and `target_modules` for the specified layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDeltaConfig {
    pub rank: usize,
    pub target_modules: Vec<String>,
}

/// Configuration for creating a new tenant delta (also used by the create_lora MCP tool)
///
/// This is the single LoRA configuration type. TTT and PEFT differ only in lifecycle
/// (online vs frozen), not structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantDeltaConfig {
    /// LoRA rank (default: 8)
    #[serde(default = "default_rank")]
    pub rank: usize,

    /// LoRA alpha scaling factor (default: 4.0, < rank for regularization)
    #[serde(default = "default_alpha")]
    pub alpha: f32,

    /// Dropout probability (default: 0.0, unused by TTT but accepted by create_lora RPC)
    #[serde(default)]
    pub dropout: f32,

    /// Target modules to apply LoRA to (default: ["q_proj", "v_proj"])
    #[serde(default = "default_target_modules")]
    pub target_modules: Vec<String>,

    /// Learning rate for optimizer (default: 3e-4)
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,

    /// Maximum accumulated gradient steps before requiring save/merge (default: 300)
    #[serde(default = "default_max_accumulated_steps")]
    pub max_accumulated_steps: u64,

    /// Weight decay factor for optimizer (default: 0.02)
    #[serde(default = "default_decay_lambda")]
    pub decay_lambda: f64,

    /// Muon momentum coefficient (default: 0.95)
    #[serde(default = "default_muon_momentum_config")]
    pub muon_momentum: f64,

    /// Per-layer rank/module override. Key = layer index.
    /// Layers not in map use the top-level `rank` and `target_modules`.
    #[serde(default)]
    pub layer_overrides: Option<HashMap<usize, LayerDeltaConfig>>,
}

impl TenantDeltaConfig {
    /// Create config with non-uniform per-layer ranks from a TTN analysis profile.
    ///
    /// Works for any model — the profile is produced by `ttn_profile::get_layer_profile()`.
    pub fn from_profile(
        base: &TenantDeltaConfig,
        profile: &crate::runtime::ttn_profile::LayerProfile,
    ) -> Self {
        let mut overrides = HashMap::new();
        for la in &profile.layers {
            overrides.insert(
                la.layer_idx,
                LayerDeltaConfig {
                    rank: la.recommended_rank,
                    target_modules: la.target_modules.clone(),
                },
            );
        }
        Self {
            layer_overrides: Some(overrides),
            ..base.clone()
        }
    }
}

fn default_rank() -> usize {
    8
}
fn default_alpha() -> f32 {
    4.0
}
fn default_target_modules() -> Vec<String> {
    vec!["q_proj".to_owned(), "v_proj".to_owned()]
}
fn default_learning_rate() -> f64 {
    3e-4
}
fn default_max_accumulated_steps() -> u64 {
    300
}
fn default_decay_lambda() -> f64 {
    0.02
}
fn default_muon_momentum_config() -> f64 {
    0.95
}

impl Default for TenantDeltaConfig {
    fn default() -> Self {
        Self {
            rank: default_rank(),
            alpha: default_alpha(),
            dropout: 0.0,
            target_modules: default_target_modules(),
            learning_rate: default_learning_rate(),
            max_accumulated_steps: default_max_accumulated_steps(),
            decay_lambda: default_decay_lambda(),
            muon_momentum: default_muon_momentum_config(),
            layer_overrides: None,
        }
    }
}

/// Per-tenant LoRA delta with per-layer accumulation tracking
///
/// Each tenant gets isolated per-layer A/B weight matrices. Keys use the
/// format `"layer_idx.module_name"` (e.g., `"0.q_proj"`, `"27.v_proj"`).
/// The delta accumulates across requests and can be periodically saved or
/// merged into a permanent adapter.
pub struct TenantDelta {
    /// LoRA A matrices: "layer_idx.module_name" -> [rank, in_features]
    pub lora_a: HashMap<String, Tensor>,
    /// LoRA B matrices: "layer_idx.module_name" -> [out_features, rank]
    pub lora_b: HashMap<String, Tensor>,
    /// VarStore owning all trainable parameters
    pub vs: VarStore,
    /// Per-parameter Muon optimizer state: VarStore path -> MuonState
    pub muon_states: HashMap<String, MuonState>,
    /// Muon optimizer configuration
    pub muon_config: MuonConfig,
    /// Per-key scaling: "layer_idx.module_name" -> alpha / layer_rank
    /// Replaces the single `scaling` field for non-uniform rank support (C7 fix).
    pub scaling_map: HashMap<String, f64>,
    /// Default scaling factor: alpha / rank (for uniform-rank deltas or backward compat)
    pub scaling: f64,
    /// LoRA rank
    pub rank: usize,
    /// Device for tensors
    pub device: Device,
    /// Target module names (without layer prefix)
    pub target_modules: Vec<String>,
    /// Number of model layers
    pub num_layers: usize,
    /// Learning rate for optimizer
    pub learning_rate: f64,

    // Accumulation tracking
    /// Total gradient steps accumulated
    pub accumulated_steps: u64,
    /// Maximum steps before requiring save/merge
    pub max_accumulated_steps: u64,
    /// Number of inference requests served
    pub request_count: u64,
    /// Running average of loss improvement per adaptation
    pub avg_loss_improvement: f64,
    /// Last time this delta was accessed
    pub last_access: Instant,
    /// When this delta was created
    pub created_at: Instant,
    /// Hash of last CAS snapshot (if any)
    pub last_snapshot_hash: Option<String>,
    /// Weight decay factor (used by Muon's decoupled weight decay)
    pub decay_lambda: f64,
    /// Per-key effective rank (for narrow-based rank adaptation).
    /// Key = "layer_idx.module_name", value = active rank (1..=allocated_rank).
    /// Defaults to the allocated rank for each key.
    pub effective_ranks: HashMap<String, usize>,
    /// Per-key maximum (allocated) rank. Set at creation time, never changes.
    max_ranks: HashMap<String, usize>,
    /// Alpha value (stored for scaling recalculation on rank change)
    alpha: f32,
    /// Optional per-tenant rank oracle for runtime rank adaptation
    pub rank_oracle: Option<super::ttt::RankOracle>,
}

impl TenantDelta {
    /// Create a new per-layer tenant delta with Kaiming init for A and zeros for B.
    ///
    /// Creates `num_layers × num_modules` A/B pairs keyed as `"layer_idx.module_name"`.
    ///
    /// # Arguments
    /// * `config` - Delta configuration (rank, alpha, target modules, layer_overrides, etc.)
    /// * `module_dims` - Map of module_name -> (in_features, out_features) (flat, for uniform layers)
    /// * `device` - Device for tensor allocation
    /// * `num_layers` - Number of model layers
    /// * `per_layer_dims` - Optional per-layer dim overrides: layer_idx -> module_name -> (in, out)
    ///   Required when different layer types have different `o_proj` dimensions (e.g. Qwen3.5).
    pub fn new(
        config: &TenantDeltaConfig,
        module_dims: &HashMap<String, (usize, usize)>,
        device: Device,
        num_layers: usize,
    ) -> Result<Self> {
        Self::new_with_per_layer_dims(config, module_dims, device, num_layers, None)
    }

    /// Create a new per-layer tenant delta with optional per-layer dimension overrides.
    ///
    /// Same as `new()` but accepts `per_layer_dims` for architectures where
    /// `o_proj` (or other modules) have different dimensions across layer types.
    pub fn new_with_per_layer_dims(
        config: &TenantDeltaConfig,
        module_dims: &HashMap<String, (usize, usize)>,
        device: Device,
        num_layers: usize,
        per_layer_dims: Option<&HashMap<usize, HashMap<String, (usize, usize)>>>,
    ) -> Result<Self> {
        let vs = VarStore::new(device);
        let root = vs.root();

        let mut lora_a = HashMap::new();
        let mut lora_b = HashMap::new();
        let mut scaling_map = HashMap::new();
        let mut effective_ranks = HashMap::new();
        let mut max_ranks = HashMap::new();

        for layer_idx in 0..num_layers {
            // Determine layer-specific rank and modules
            let (layer_rank, layer_modules): (usize, &Vec<String>) =
                if let Some(overrides) = &config.layer_overrides {
                    if let Some(lc) = overrides.get(&layer_idx) {
                        (lc.rank, &lc.target_modules)
                    } else {
                        (config.rank, &config.target_modules)
                    }
                } else {
                    (config.rank, &config.target_modules)
                };

            for module_name in layer_modules {
                // Per-layer dims take priority over flat module_dims (C1 fix)
                let (in_features, out_features) = per_layer_dims
                    .and_then(|pld| pld.get(&layer_idx))
                    .and_then(|m| m.get(module_name.as_str()))
                    .or_else(|| module_dims.get(module_name.as_str()))
                    .ok_or_else(|| {
                        anyhow!(
                            "Module '{}' not found in model dimensions (layer {})",
                            module_name,
                            layer_idx
                        )
                    })?;

                let key = format!("{}.{}", layer_idx, module_name);

                // Use Path::sub() for hierarchy — '.' is VarStore's path separator
                let layer_path = root.sub(format!("layer_{}", layer_idx)).sub(module_name);

                // A: Kaiming uniform initialization [rank, in_features]
                let a = layer_path.kaiming_uniform(
                    "lora_a",
                    &[layer_rank as i64, *in_features as i64],
                );

                // B: Zero initialization [out_features, rank]
                let b = layer_path.zeros(
                    "lora_b",
                    &[*out_features as i64, layer_rank as i64],
                );

                lora_a.insert(key.clone(), a);
                lora_b.insert(key.clone(), b);

                // Per-key scaling: alpha / layer_rank (C7 fix)
                scaling_map.insert(key.clone(), config.alpha as f64 / layer_rank as f64);

                // Track effective and max ranks for narrow-based adaptation
                effective_ranks.insert(key.clone(), layer_rank);
                max_ranks.insert(key, layer_rank);
            }
        }

        // Build Muon optimizer state (created lazily on first step)
        let muon_states = HashMap::new();
        let muon_config = MuonConfig {
            lr: config.learning_rate,
            momentum: config.muon_momentum,
            weight_decay: 0.0, // Muon orthogonalizes gradients; weight decay rarely needed
            ..MuonConfig::default()
        };

        let scaling = config.alpha as f64 / config.rank as f64;
        let now = Instant::now();

        // Collect unique target modules across all layers for iteration metadata
        let mut all_modules: Vec<String> = config.target_modules.clone();
        if let Some(overrides) = &config.layer_overrides {
            for lc in overrides.values() {
                for m in &lc.target_modules {
                    if !all_modules.contains(m) {
                        all_modules.push(m.clone());
                    }
                }
            }
        }

        Ok(Self {
            lora_a,
            lora_b,
            vs,
            muon_states,
            muon_config,
            scaling_map,
            scaling,
            rank: config.rank,
            device,
            target_modules: all_modules,
            num_layers,
            learning_rate: config.learning_rate,
            accumulated_steps: 0,
            max_accumulated_steps: config.max_accumulated_steps,
            request_count: 0,
            avg_loss_improvement: 0.0,
            last_access: now,
            created_at: now,
            last_snapshot_hash: None,
            decay_lambda: config.decay_lambda,
            effective_ranks,
            max_ranks,
            alpha: config.alpha,
            rank_oracle: None,
        })
    }

    /// Check if this delta has a module at the given layer
    pub fn has_module(&self, module_name: &str, layer_idx: usize) -> bool {
        self.lora_a.contains_key(&format!("{}.{}", layer_idx, module_name))
    }

    /// Compute LoRA correction for a given module at a specific layer: output += scaling * (x @ A^T) @ B^T
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, in_features]
    /// * `module_name` - Name of the target module
    /// * `layer_idx` - Layer index
    ///
    /// # Returns
    /// LoRA correction tensor [batch, seq_len, out_features]
    pub fn forward(&self, x: &Tensor, module_name: &str, layer_idx: usize) -> Result<Tensor> {
        let key = format!("{}.{}", layer_idx, module_name);
        let a = self.lora_a.get(&key)
            .ok_or_else(|| anyhow!("Module '{}' not found in delta", key))?;
        let b = self.lora_b.get(&key)
            .ok_or_else(|| anyhow!("Module '{}' B not found in delta", key))?;

        // Per-key scaling (C7 fix): use per-key alpha/rank, fall back to global scaling
        let scaling = self.scaling_map.get(&key).copied().unwrap_or(self.scaling);

        // Narrow to effective rank (view, no copy)
        let eff_rank = self.effective_ranks.get(&key).copied()
            .unwrap_or_else(|| a.size()[0] as usize) as i64;
        let a_eff = a.narrow(0, 0, eff_rank); // [eff_rank, in_features]
        let b_eff = b.narrow(1, 0, eff_rank); // [out_features, eff_rank]

        let x = x.to_kind(a.kind());
        let intermediate = x.f_matmul(&a_eff.tr())
            .map_err(|e| anyhow!("Delta forward matmul A failed for '{}': {}", key, e))?;
        let output = intermediate.f_matmul(&b_eff.tr())
            .map_err(|e| anyhow!("Delta forward matmul B failed for '{}': {}", key, e))?;

        Ok(output * scaling)
    }

    /// Compute LoRA correction for 2D tensors at a specific layer: output += scaling * (x @ A^T) @ B^T
    ///
    /// Handles shape [batch*seq_len, features] used inside attention layers.
    ///
    /// # Arguments
    /// * `x` - Input tensor [tokens, in_features]
    /// * `module_name` - Name of the target module (e.g., "q_proj", "v_proj")
    /// * `layer_idx` - Layer index
    ///
    /// # Returns
    /// LoRA correction tensor [tokens, out_features]
    pub fn forward_2d(&self, x: &Tensor, module_name: &str, layer_idx: usize) -> Result<Tensor> {
        let key = format!("{}.{}", layer_idx, module_name);
        let a = self.lora_a.get(&key)
            .ok_or_else(|| anyhow!("Module '{}' not found in delta", key))?;
        let b = self.lora_b.get(&key)
            .ok_or_else(|| anyhow!("Module '{}' B not found in delta", key))?;

        // Per-key scaling (C7 fix): use per-key alpha/rank, fall back to global scaling
        let scaling = self.scaling_map.get(&key).copied().unwrap_or(self.scaling);

        // Narrow to effective rank (view, no copy)
        let eff_rank = self.effective_ranks.get(&key).copied()
            .unwrap_or_else(|| a.size()[0] as usize) as i64;
        let a_eff = a.narrow(0, 0, eff_rank); // [eff_rank, in_features]
        let b_eff = b.narrow(1, 0, eff_rank); // [out_features, eff_rank]

        let x = x.to_kind(a.kind());
        let intermediate = x.f_matmul(&a_eff.tr())
            .map_err(|e| anyhow!("Delta forward_2d matmul A failed for '{}': {}", key, e))?;
        let output = intermediate.f_matmul(&b_eff.tr())
            .map_err(|e| anyhow!("Delta forward_2d matmul B failed for '{}': {}", key, e))?;

        Ok(output * scaling)
    }

    /// Compute the ratio of delta norm to a reference norm for drift monitoring
    ///
    /// Returns a map of "layer_idx.module_name" -> ||delta|| / ||base|| where delta is
    /// the effective weight change (B @ A) and base is the reference norm.
    pub fn delta_norm_ratio(&self, base_norms: &HashMap<String, f64>) -> HashMap<String, f64> {
        let _guard = tch::no_grad_guard();
        let mut ratios = HashMap::new();

        // Iterate lora_a.keys() to handle non-uniform per-layer module sets (I1 fix)
        for key in self.lora_a.keys() {
            if let (Some(a), Some(b)) = (self.lora_a.get(key), self.lora_b.get(key)) {
                let scaling = self.scaling_map.get(key).copied().unwrap_or(self.scaling);
                let delta = b.matmul(a) * scaling;
                let delta_norm: f64 = delta.norm().double_value(&[]);
                // Extract module name from key "layer_idx.module_name"
                let module_name = key.split_once('.').map_or(key.as_str(), |(_, m)| m);
                let base_norm = base_norms.get(module_name).copied().unwrap_or(1.0);
                ratios.insert(key.clone(), delta_norm / base_norm.max(1e-8));
            }
        }

        ratios
    }

    /// Extract state dict (A and B tensors) for serialization
    ///
    /// Keys use the format `"layer_idx.module_name.lora_a"` / `"layer_idx.module_name.lora_b"`.
    pub fn extract_state_dict(&self) -> HashMap<String, Tensor> {
        let _guard = tch::no_grad_guard();
        let mut state = HashMap::new();

        // Iterate lora_a.keys() to handle non-uniform per-layer module sets (I1 fix)
        for key in self.lora_a.keys() {
            if let Some(a) = self.lora_a.get(key) {
                state.insert(format!("{}.lora_a", key), a.copy());
            }
            if let Some(b) = self.lora_b.get(key) {
                state.insert(format!("{}.lora_b", key), b.copy());
            }
        }

        state
    }

    /// Load state dict (restore A and B tensors from a snapshot)
    ///
    /// Expects keys in format `"layer_idx.module_name.lora_a"` / `"layer_idx.module_name.lora_b"`.
    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> Result<()> {
        let _guard = tch::no_grad_guard();

        for key in self.lora_a.keys().cloned().collect::<Vec<_>>() {
            let a_key = format!("{}.lora_a", key);
            let b_key = format!("{}.lora_b", key);

            if let Some(a_src) = state.get(&a_key) {
                if let Some(a_dst) = self.lora_a.get_mut(&key) {
                    a_dst.copy_(a_src);
                }
            }

            if let Some(b_src) = state.get(&b_key) {
                if let Some(b_dst) = self.lora_b.get_mut(&key) {
                    b_dst.copy_(b_src);
                }
            }
        }

        Ok(())
    }

    /// Update last access time
    pub fn touch(&mut self) {
        self.last_access = Instant::now();
    }

    /// Estimate memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        let _guard = tch::no_grad_guard();
        let mut total = 0usize;

        for a in self.lora_a.values() {
            total += tensor_bytes(a);
        }
        for b in self.lora_b.values() {
            total += tensor_bytes(b);
        }

        total
    }

    /// Check if the delta has reached its accumulation limit
    pub fn is_at_capacity(&self) -> bool {
        self.accumulated_steps >= self.max_accumulated_steps
    }

    /// Reset the delta to zeros (re-initialize B matrices, keep A)
    pub fn reset(&mut self) {
        let _guard = tch::no_grad_guard();
        for b in self.lora_b.values_mut() {
            let _ = b.zero_();
        }
        self.accumulated_steps = 0;
        self.request_count = 0;
        self.avg_loss_improvement = 0.0;
    }

    /// Zero all gradients on trainable variables.
    pub fn zero_grad(&self) {
        for (_, var) in self.vs.variables() {
            if var.grad().defined() {
                let _ = var.grad().zero_();
            }
        }
    }

    /// Get the effective (active) rank for a key. Returns None if key doesn't exist.
    pub fn effective_rank(&self, key: &str) -> Option<usize> {
        self.effective_ranks.get(key).copied()
    }

    /// Set the effective rank for a key. Clamped to [1, max_rank].
    /// Also updates the scaling factor for this key (alpha / effective_rank).
    pub fn set_effective_rank(&mut self, key: &str, rank: usize) {
        if let Some(&max_r) = self.max_ranks.get(key) {
            let clamped = rank.clamp(1, max_r);
            self.effective_ranks.insert(key.to_owned(), clamped);
            // Recalculate scaling: alpha / effective_rank
            self.scaling_map.insert(key.to_owned(), self.alpha as f64 / clamped as f64);
        }
    }
}

impl TenantDelta {
    /// Compose two deltas: base + tenant
    ///
    /// Creates a new "view" delta whose forward pass sums corrections from both.
    /// The composed delta is non-trainable (for inference only).
    ///
    /// Both deltas' A/B matrices are summed element-wise for shared keys.
    /// For keys only in one delta, those corrections are used as-is.
    pub fn compose(
        base: &std::sync::Arc<parking_lot::Mutex<TenantDelta>>,
        tenant: &std::sync::Arc<parking_lot::Mutex<TenantDelta>>,
    ) -> std::sync::Arc<parking_lot::Mutex<TenantDelta>> {
        let base_guard = base.lock();
        let tenant_guard = tenant.lock();
        let _no_grad = tch::no_grad_guard();

        // Collect all per-layer keys from both deltas
        let mut all_keys: Vec<String> = base_guard.lora_a.keys().cloned().collect();
        for k in tenant_guard.lora_a.keys() {
            if !all_keys.contains(k) {
                all_keys.push(k.clone());
            }
        }

        let mut composed_a = HashMap::new();
        let mut composed_b = HashMap::new();

        for key in &all_keys {
            let base_a = base_guard.lora_a.get(key);
            let base_b = base_guard.lora_b.get(key);
            let tenant_a = tenant_guard.lora_a.get(key);
            let tenant_b = tenant_guard.lora_b.get(key);

            // Use per-key scaling if available (C7 fix), else fall back to global
            let base_scaling = base_guard.scaling_map.get(key).copied().unwrap_or(base_guard.scaling);
            let tenant_scaling = tenant_guard.scaling_map.get(key).copied().unwrap_or(tenant_guard.scaling);

            // Narrow to effective_rank before composition — matches forward_2d behavior.
            // Without this, compose operates on dormant rank dimensions (near-zero values
            // that waste FLOPs and may introduce noise from untrained parameters).
            let narrow_a = |a: &Tensor, guard: &TenantDelta| -> Tensor {
                let eff = guard.effective_ranks.get(key).copied()
                    .unwrap_or_else(|| a.size()[0] as usize) as i64;
                a.narrow(0, 0, eff)
            };
            let narrow_b = |b: &Tensor, guard: &TenantDelta| -> Tensor {
                let eff = guard.effective_ranks.get(key).copied()
                    .unwrap_or_else(|| b.size()[1] as usize) as i64;
                b.narrow(1, 0, eff)
            };

            // Check if effective ranks match for this specific key
            let base_eff_rank = base_a.map(|a| {
                base_guard.effective_ranks.get(key).copied()
                    .unwrap_or_else(|| a.size()[0] as usize)
            });
            let tenant_eff_rank = tenant_a.map(|a| {
                tenant_guard.effective_ranks.get(key).copied()
                    .unwrap_or_else(|| a.size()[0] as usize)
            });
            let ranks_match = base_eff_rank == tenant_eff_rank;

            match (base_a, tenant_a) {
                (Some(ba), Some(ta)) if ranks_match => {
                    // Same effective rank: narrow then add A matrices directly (pre-scaled)
                    let base_effective_a = narrow_a(ba, &base_guard) * base_scaling;
                    let tenant_effective_a = narrow_a(ta, &tenant_guard) * tenant_scaling;
                    composed_a.insert(key.clone(), base_effective_a + tenant_effective_a);
                }
                (Some(ba), Some(ta)) => {
                    // Different effective ranks: narrow each to its effective rank, compute
                    // W_eff = s1*(B1_eff @ A1_eff) + s2*(B2_eff @ A2_eff), store as A=W_eff, B=I.
                    #[allow(clippy::expect_used)] // invariant: B must exist when A exists
                    let bb = base_b.expect("base B must exist if base A exists");
                    #[allow(clippy::expect_used)]
                    let tb = tenant_b.expect("tenant B must exist if tenant A exists");
                    // Narrow to effective rank before matmul
                    let ba_eff = narrow_a(ba, &base_guard);
                    let bb_eff = narrow_b(bb, &base_guard);
                    let ta_eff = narrow_a(ta, &tenant_guard);
                    let tb_eff = narrow_b(tb, &tenant_guard);
                    // B_eff @ A_eff = [out_dim, eff_rank] @ [eff_rank, in_dim] = [out_dim, in_dim]
                    let base_w = bb_eff.matmul(&ba_eff) * base_scaling;
                    let tenant_w = tb_eff.matmul(&ta_eff) * tenant_scaling;
                    let w_eff = base_w + tenant_w;
                    // forward_2d computes: scaling * (x @ A^T) @ B^T
                    // We want: x @ W_eff^T  (result shape [tokens, out_dim])
                    // Store as: A = W_eff [out_dim, in_dim], B = I [out_dim, out_dim]
                    // Then: (x @ W_eff^T) @ I^T = x @ W_eff^T ✓
                    let out_dim = w_eff.size()[0];
                    let identity = Tensor::eye(out_dim, (w_eff.kind(), w_eff.device()));
                    composed_a.insert(key.clone(), w_eff);
                    composed_b.insert(key.clone(), identity);
                    continue; // Skip the B match below, we already handled it
                }
                (Some(ba), None) => {
                    composed_a.insert(key.clone(), narrow_a(ba, &base_guard) * base_scaling);
                }
                (None, Some(ta)) => {
                    composed_a.insert(key.clone(), narrow_a(ta, &tenant_guard) * tenant_scaling);
                }
                (None, None) => {}
            }

            // Only process B separately when we didn't already handle it above
            // (i.e., ranks matched or only one side had the key)
            match (base_b, tenant_b) {
                (Some(bb), Some(tb)) => {
                    let bb_eff = narrow_b(bb, &base_guard);
                    let tb_eff = narrow_b(tb, &tenant_guard);
                    composed_b.insert(key.clone(), &bb_eff + &tb_eff);
                }
                (Some(bb), None) => {
                    composed_b.insert(key.clone(), narrow_b(bb, &base_guard));
                }
                (None, Some(tb)) => {
                    composed_b.insert(key.clone(), narrow_b(tb, &tenant_guard));
                }
                (None, None) => {}
            }
        }

        // Merge target modules (without layer prefix)
        let mut all_modules: Vec<String> = base_guard.target_modules.clone();
        for m in &tenant_guard.target_modules {
            if !all_modules.contains(m) {
                all_modules.push(m.clone());
            }
        }

        let device = base_guard.device;
        let rank = base_guard.rank.max(tenant_guard.rank);
        let num_layers = base_guard.num_layers.max(tenant_guard.num_layers);
        let vs = VarStore::new(device);
        let now = Instant::now();

        std::sync::Arc::new(parking_lot::Mutex::new(TenantDelta {
            lora_a: composed_a,
            lora_b: composed_b,
            vs,
            muon_states: HashMap::new(),
            muon_config: MuonConfig::default(),
            scaling_map: HashMap::new(), // scaling=1.0 already baked in during composition
            scaling: 1.0, // Already pre-scaled during composition
            rank,
            device,
            target_modules: all_modules,
            num_layers,
            learning_rate: 0.0,
            accumulated_steps: 0,
            max_accumulated_steps: u64::MAX,
            request_count: 0,
            avg_loss_improvement: 0.0,
            last_access: now,
            created_at: now,
            last_snapshot_hash: None,
            decay_lambda: 0.0,
            effective_ranks: HashMap::new(),
            max_ranks: HashMap::new(),
            alpha: 1.0, // Already pre-scaled during composition
            rank_oracle: None,
        }))
    }
}


/// Extract raw bytes, dtype, and shape from a tch::Tensor for safetensors serialization.
///
/// Extracts raw bytes from a tensor for safetensors serialization.
/// Moves to CPU and makes contiguous if needed.
fn tensor_to_safetensors_data(tensor: &Tensor) -> Result<(safetensors::Dtype, Vec<usize>, Vec<u8>)> {
    let cpu_tensor = tensor.to(tch::Device::Cpu).contiguous();
    let kind = cpu_tensor.kind();
    let dtype = match kind {
        Kind::Float => safetensors::Dtype::F32,
        Kind::Half => safetensors::Dtype::F16,
        Kind::BFloat16 => safetensors::Dtype::BF16,
        Kind::Double => safetensors::Dtype::F64,
        Kind::Int => safetensors::Dtype::I32,
        Kind::Int64 => safetensors::Dtype::I64,
        Kind::Int16 => safetensors::Dtype::I16,
        Kind::Int8 => safetensors::Dtype::I8,
        Kind::Uint8 => safetensors::Dtype::U8,
        other => return Err(anyhow!("Unsupported tensor kind for safetensors: {:?}", other)),
    };
    let shape: Vec<usize> = cpu_tensor.size().iter().map(|&d| d as usize).collect();
    let numel: usize = shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| anyhow!("Tensor numel overflow: shape {:?}", shape))
    })?;
    let elem_size = kind.elt_size_in_bytes();
    let byte_len = numel.checked_mul(elem_size)
        .ok_or_else(|| anyhow!("Tensor byte length overflow: {} * {}", numel, elem_size))?;
    let mut bytes = vec![0u8; byte_len];
    cpu_tensor.copy_data_u8(&mut bytes, numel);
    Ok((dtype, shape, bytes))
}

/// Map module name to PEFT subpath
fn module_to_peft_subpath(module: &str) -> &str {
    match module {
        "gate_proj" | "up_proj" | "down_proj" => "mlp",
        _ => "self_attn", // q_proj, k_proj, v_proj, o_proj, and others
    }
}

impl TenantDelta {
    /// Serialize per-layer LoRA A/B matrices to safetensors bytes (in memory).
    ///
    /// Uses standard HuggingFace PEFT naming convention:
    /// `base_model.model.layers.{N}.{self_attn|mlp}.{module}.lora_{A|B}.weight`
    pub fn serialize_to_safetensors_bytes(&self) -> Result<Vec<u8>> {
        let mut pairs: Vec<(String, Tensor)> = Vec::new();

        // Iterate lora_a.keys() to handle non-uniform per-layer module sets (I1 fix)
        let mut keys: Vec<&String> = self.lora_a.keys().collect();
        keys.sort(); // deterministic output
        for key in keys {
            // key format: "layer_idx.module_name"
            let mut parts = key.splitn(2, '.');
            let layer_idx: usize = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let module = parts.next().unwrap_or(key.as_str());
            let subpath = module_to_peft_subpath(module);

            if let Some(a) = self.lora_a.get(key) {
                let peft_key = format!(
                    "base_model.model.layers.{}.{}.{}.lora_A.weight",
                    layer_idx, subpath, module
                );
                pairs.push((peft_key, a.shallow_clone()));
            }
            if let Some(b) = self.lora_b.get(key) {
                let peft_key = format!(
                    "base_model.model.layers.{}.{}.{}.lora_B.weight",
                    layer_idx, subpath, module
                );
                pairs.push((peft_key, b.shallow_clone()));
            }
        }

        serialize_tensor_pairs_to_bytes(pairs.into_iter())
    }

    /// Load per-layer LoRA from safetensors bytes (in memory).
    ///
    /// Parses standard HuggingFace PEFT naming conventions with layer indices.
    pub fn load_from_safetensors_bytes(data: &[u8], device: Device) -> Result<Self> {
        let tensors = SafeTensors::deserialize(data)
            .map_err(|e| anyhow!("safetensors deserialization failed: {}", e))?;

        let mut lora_a: HashMap<String, Tensor> = HashMap::new();
        let mut lora_b: HashMap<String, Tensor> = HashMap::new();
        let mut rank = 0usize;
        let mut max_layer = 0usize;
        let mut module_names: std::collections::HashSet<String> = std::collections::HashSet::new();

        for (name, view) in tensors.tensors() {
            if let Some((layer_idx, module, ab)) = extract_lora_components(&name) {
                let key = format!("{}.{}", layer_idx, module);
                let tensor = safetensors_view_to_tensor(&view, device)?;
                max_layer = max_layer.max(layer_idx);
                module_names.insert(module.clone());
                match ab.as_str() {
                    "lora_a" => {
                        if rank == 0 {
                            rank = tensor.size()[0] as usize;
                        }
                        lora_a.insert(key, tensor);
                    }
                    "lora_b" => {
                        lora_b.insert(key, tensor);
                    }
                    _ => {}
                }
            }
        }

        if lora_a.is_empty() {
            return Err(anyhow!("No LoRA A matrices found in safetensors data"));
        }

        // Validate that all A/B pairs have consistent shapes
        for (key, a_tensor) in &lora_a {
            let a_shape = a_tensor.size();
            if a_shape.len() != 2 {
                return Err(anyhow!("LoRA A tensor '{}' has invalid rank: expected 2, got {}", key, a_shape.len()));
            }
            if let Some(b_tensor) = lora_b.get(key) {
                let b_shape = b_tensor.size();
                if b_shape.len() != 2 {
                    return Err(anyhow!("LoRA B tensor '{}' has invalid rank: expected 2, got {}", key, b_shape.len()));
                }
                // A is [rank, in_dim], B is [out_dim, rank] — inner dims must match
                if a_shape[0] != b_shape[1] {
                    return Err(anyhow!(
                        "LoRA rank mismatch for '{}': A shape {:?} vs B shape {:?} (A[0]={} != B[1]={})",
                        key, a_shape, b_shape, a_shape[0], b_shape[1]
                    ));
                }
            }
        }

        let num_layers = max_layer + 1;
        let alpha = rank as f32;
        let target_modules: Vec<String> = module_names.into_iter().collect();
        let scaling = alpha as f64 / rank as f64;
        let vs = VarStore::new(device);
        let now = Instant::now();

        tracing::info!(
            "Loaded LoRA adapter from bytes: rank={}, layers={}, modules={:?}, alpha={}",
            rank,
            num_layers,
            target_modules,
            alpha
        );

        // Build per-key scaling map from loaded A matrix ranks
        let scaling_map: HashMap<String, f64> = lora_a
            .iter()
            .map(|(key, a)| {
                let layer_rank = a.size()[0] as usize;
                (key.clone(), alpha as f64 / layer_rank as f64)
            })
            .collect();

        // Build effective_ranks and max_ranks from loaded A matrices
        let effective_ranks: HashMap<String, usize> = lora_a
            .iter()
            .map(|(key, a)| (key.clone(), a.size()[0] as usize))
            .collect();
        let max_ranks = effective_ranks.clone();

        Ok(Self {
            lora_a,
            lora_b,
            vs,
            muon_states: HashMap::new(),
            muon_config: MuonConfig::default(),
            scaling_map,
            scaling,
            rank,
            device,
            target_modules,
            num_layers,
            learning_rate: 0.0,
            accumulated_steps: 0,
            max_accumulated_steps: u64::MAX,
            request_count: 0,
            avg_loss_improvement: 0.0,
            last_access: now,
            created_at: now,
            last_snapshot_hash: None,
            decay_lambda: 0.0,
            effective_ranks,
            max_ranks,
            alpha,
            rank_oracle: None,
        })
    }
}

/// Convert a safetensors TensorView to a tch::Tensor
fn safetensors_view_to_tensor(view: &safetensors::tensor::TensorView<'_>, device: Device) -> Result<Tensor> {
    let dtype = view.dtype();
    let kind = match dtype {
        safetensors::Dtype::F32 => Kind::Float,
        safetensors::Dtype::F16 => Kind::Half,
        safetensors::Dtype::BF16 => Kind::BFloat16,
        safetensors::Dtype::F64 => Kind::Double,
        safetensors::Dtype::I32 => Kind::Int,
        safetensors::Dtype::I64 => Kind::Int64,
        safetensors::Dtype::I16 => Kind::Int16,
        safetensors::Dtype::I8 => Kind::Int8,
        safetensors::Dtype::U8 => Kind::Uint8,
        other => return Err(anyhow!("Unsupported safetensors dtype: {:?}", other)),
    };
    let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
    let data = view.data();

    // Create tensor from raw bytes on CPU, then move to target device
    let tensor = Tensor::from_data_size(data, &shape, kind);
    Ok(tensor.to(device))
}

/// Shared safetensors serialization: converts (key, tensor) pairs to bytes.
///
/// Used by both `TenantDelta::serialize_to_safetensors_bytes()` (PEFT keys)
/// and `serialize_state_dict_to_bytes()` (raw keys).
fn serialize_tensor_pairs_to_bytes(
    pairs: impl Iterator<Item = (String, Tensor)>,
) -> Result<Vec<u8>> {
    let _guard = tch::no_grad_guard();
    let mut tensor_data: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    for (key, tensor) in pairs {
        let cpu = tensor.to_device(Device::Cpu).contiguous();
        let (dtype, shape, bytes) = tensor_to_safetensors_data(&cpu)?;
        tensor_data.push((key, dtype, shape, bytes));
    }

    if tensor_data.is_empty() {
        return Err(anyhow!("No tensors to serialize"));
    }

    let views: Vec<(String, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(key, dtype, shape, bytes)| {
            let view = TensorView::new(*dtype, shape.clone(), bytes)
                .unwrap_or_else(|_| unreachable!("TensorView construction with valid dtype/shape/bytes"));
            (key.clone(), view)
        })
        .collect();
    let data_refs: Vec<(&str, TensorView<'_>)> = views
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();

    safetensors::tensor::serialize(data_refs, &None)
        .map_err(|e| anyhow!("safetensors serialization failed: {}", e))
}

/// Serialize a state dict (HashMap<String, Tensor>) to safetensors bytes.
///
/// This replaces direct `Tensor::save_multi()` calls, allowing writes through FsOps.
pub fn serialize_state_dict_to_bytes(state_dict: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    serialize_tensor_pairs_to_bytes(
        state_dict.iter().map(|(k, v)| (k.clone(), v.shallow_clone())),
    )
}

/// Load a state dict from safetensors bytes.
///
/// This replaces `Tensor::load_multi()`, allowing reads through FsOps.
pub fn load_state_dict_from_bytes(data: &[u8]) -> Result<HashMap<String, Tensor>> {
    let tensors = SafeTensors::deserialize(data)
        .map_err(|e| anyhow!("safetensors deserialization failed: {}", e))?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let tensor = safetensors_view_to_tensor(&view, Device::Cpu)?;
        result.insert(name, tensor);
    }
    Ok(result)
}


/// Estimate tensor memory in bytes
fn tensor_bytes(t: &Tensor) -> usize {
    let numel = t.numel();
    let elem_size = match t.kind() {
        Kind::Half | Kind::BFloat16 => 2,
        Kind::Double => 8,
        _ => 4, // Float and others
    };
    numel * elem_size
}

// SAFETY: TenantDelta contains a VarStore backed by tch-rs raw pointers.
// VarStore itself is not Send/Sync, but all mutable access to TenantDelta
// goes through DeltaPool which wraps it in Arc<parking_lot::Mutex<TenantDelta>>.
// The Mutex ensures exclusive access, making cross-thread use safe.
//
// Invariant: TenantDelta must NEVER be shared across threads without a Mutex.
// DeltaPool enforces this — all public APIs return Arc<Mutex<TenantDelta>>.
unsafe impl Send for TenantDelta {}
unsafe impl Sync for TenantDelta {}

/// Type-safe wrapper that enforces Mutex protection for TenantDelta.
/// All cross-thread access must go through this wrapper.
pub type SharedTenantDelta = std::sync::Arc<parking_lot::Mutex<TenantDelta>>;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    const TEST_NUM_LAYERS: usize = 2;

    fn test_module_dims() -> HashMap<String, (usize, usize)> {
        let mut dims = HashMap::new();
        dims.insert("q_proj".to_owned(), (512, 512));
        dims.insert("v_proj".to_owned(), (512, 512));
        dims
    }

    #[test]
    fn test_tenant_delta_creation() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        assert_eq!(delta.rank, 8);
        assert_eq!(delta.target_modules.len(), 2);
        assert_eq!(delta.num_layers, TEST_NUM_LAYERS);
        // Per-layer: 2 layers * 2 modules = 4 entries
        assert_eq!(delta.lora_a.len(), 4);
        assert_eq!(delta.accumulated_steps, 0);
        assert_eq!(delta.request_count, 0);
    }

    #[test]
    fn test_tenant_delta_forward_shape() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        let x = Tensor::randn([1, 10, 512], (Kind::Float, Device::Cpu));
        let output = delta.forward(&x, "q_proj", 0).unwrap();
        assert_eq!(output.size(), vec![1, 10, 512]);

        // Layer 1 should also work
        let output1 = delta.forward(&x, "q_proj", 1).unwrap();
        assert_eq!(output1.size(), vec![1, 10, 512]);
    }

    #[test]
    fn test_state_dict_roundtrip() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        let state = delta.extract_state_dict();

        // Create a new delta and load the state
        let mut delta2 = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();
        delta2.load_state_dict(&state).unwrap();

        // Verify A matrices match for layer 0
        let diff: f64 = (&delta.lora_a["0.q_proj"] - &delta2.lora_a["0.q_proj"])
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert!(diff < 1e-6, "State dict roundtrip should preserve values");

        // Verify layer 1 too
        let diff1: f64 = (&delta.lora_a["1.q_proj"] - &delta2.lora_a["1.q_proj"])
            .abs()
            .sum(Kind::Float)
            .double_value(&[]);
        assert!(diff1 < 1e-6, "State dict roundtrip should preserve layer 1 values");
    }

    #[test]
    fn test_memory_bytes() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        let bytes = delta.memory_bytes();
        // 2 layers * 2 modules * (rank * in + out * rank) * 4 bytes
        let expected = TEST_NUM_LAYERS * 2 * (8 * 512 + 512 * 8) * 4;
        assert_eq!(bytes, expected);
    }

    #[test]
    fn test_reset() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let mut delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        delta.accumulated_steps = 100;
        delta.request_count = 50;
        delta.reset();

        assert_eq!(delta.accumulated_steps, 0);
        assert_eq!(delta.request_count, 0);

        // B should be zeros for all layers
        let b_norm: f64 = delta.lora_b["0.q_proj"].norm().double_value(&[]);
        assert!(b_norm < 1e-8, "B should be zeros after reset");
        let b_norm1: f64 = delta.lora_b["1.q_proj"].norm().double_value(&[]);
        assert!(b_norm1 < 1e-8, "B layer 1 should be zeros after reset");
    }

    #[test]
    fn test_forward_2d_shape() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        let x = Tensor::randn([10, 512], (Kind::Float, Device::Cpu));
        let output = delta.forward_2d(&x, "q_proj", 0).unwrap();
        assert_eq!(output.size(), vec![10, 512]);
    }

    #[test]
    fn test_forward_2d_gradient_flow() {
        let config = TenantDeltaConfig {
            rank: 4,
            alpha: 4.0,
            ..Default::default()
        };
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        let x = Tensor::randn([5, 512], (Kind::Float, Device::Cpu));

        let output = delta.forward_2d(&x, "q_proj", 0).unwrap();

        let loss = output.sum(Kind::Float);
        loss.backward();

        let variables = delta.vs.trainable_variables();
        let mut has_grad = false;
        for var in &variables {
            if var.grad().defined() {
                let grad_norm: f64 = var.grad().norm().double_value(&[]);
                if grad_norm > 0.0 {
                    has_grad = true;
                }
            }
        }
        assert!(has_grad, "Delta A/B matrices should have gradients after backward()");
    }

    #[test]
    fn test_has_module() {
        let config = TenantDeltaConfig::default();
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        assert!(delta.has_module("q_proj", 0));
        assert!(delta.has_module("q_proj", 1));
        assert!(delta.has_module("v_proj", 0));
        assert!(!delta.has_module("gate_proj", 0)); // not a target module
    }

    #[test]
    fn test_effective_rank_default_is_max() {
        let config = TenantDeltaConfig { rank: 8, ..Default::default() };
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        // Default effective rank should equal allocated rank
        assert_eq!(delta.effective_rank("0.q_proj"), Some(8));
        assert_eq!(delta.effective_rank("0.v_proj"), Some(8));
        assert_eq!(delta.effective_rank("99.nonexistent"), None);
    }

    #[test]
    fn test_effective_rank_forward() {
        let config = TenantDeltaConfig { rank: 8, ..Default::default() };
        let dims = test_module_dims();
        let mut delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        let x = Tensor::randn([2, 512], (Kind::Float, Device::Cpu));

        // Full rank forward
        let out_full = delta.forward_2d(&x, "q_proj", 0).unwrap();
        assert_eq!(out_full.size(), &[2, 512]);

        // Set effective rank to 4 (half)
        delta.set_effective_rank("0.q_proj", 4);
        let out_narrow = delta.forward_2d(&x, "q_proj", 0).unwrap();
        assert_eq!(out_narrow.size(), &[2, 512]); // output shape unchanged
    }

    #[test]
    fn test_set_effective_rank_clamped() {
        let config = TenantDeltaConfig { rank: 8, ..Default::default() };
        let dims = test_module_dims();
        let mut delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        // Setting eff_rank above max_rank should clamp to max
        delta.set_effective_rank("0.q_proj", 32);
        assert_eq!(delta.effective_rank("0.q_proj"), Some(8)); // clamped to allocated

        // Setting eff_rank to 0 should clamp to 1 (minimum)
        delta.set_effective_rank("0.q_proj", 0);
        assert_eq!(delta.effective_rank("0.q_proj"), Some(1));
    }

    #[test]
    fn test_effective_rank_scaling_adjusts() {
        let config = TenantDeltaConfig { rank: 8, alpha: 4.0, ..Default::default() };
        let dims = test_module_dims();
        let mut delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        // Original scaling: alpha / rank = 4.0 / 8 = 0.5
        let s0 = delta.scaling_map["0.q_proj"];
        assert!((s0 - 0.5).abs() < 1e-9);

        // Set effective rank to 4 -> scaling should be alpha / eff_rank = 4.0 / 4 = 1.0
        delta.set_effective_rank("0.q_proj", 4);
        let s1 = delta.scaling_map["0.q_proj"];
        assert!((s1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_effective_rank_with_layer_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert(0, LayerDeltaConfig {
            rank: 16,
            target_modules: vec!["q_proj".to_owned(), "v_proj".to_owned()],
        });
        let config = TenantDeltaConfig {
            rank: 8,
            layer_overrides: Some(overrides),
            ..Default::default()
        };
        let dims = test_module_dims();
        let delta = TenantDelta::new(&config, &dims, Device::Cpu, TEST_NUM_LAYERS).unwrap();

        // Layer 0 allocated at 16, layer 1 at default 8
        assert_eq!(delta.effective_rank("0.q_proj"), Some(16));
        assert_eq!(delta.effective_rank("1.q_proj"), Some(8));
    }
}
