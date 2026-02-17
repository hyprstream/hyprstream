//! Per-tenant LoRA delta for isolated Test-Time Training
//!
//! Each tenant gets their own LoRA delta (A/B weight matrices) that accumulates
//! adaptations across requests. Deltas are trained via SGD and can be persisted
//! to content-addressed storage or merged into permanent adapter files.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tch::nn::{OptimizerConfig, VarStore};
use tch::{Device, Kind, Tensor};
use safetensors::tensor::{TensorView, SafeTensors};

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

    /// Weight decay factor for AdamW (default: 0.02)
    #[serde(default = "default_decay_lambda")]
    pub decay_lambda: f64,
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
    /// AdamW optimizer over all trainable parameters
    pub optimizer: tch::nn::Optimizer,
    /// Scaling factor: alpha / rank
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
    /// Weight decay factor (used by AdamW)
    pub decay_lambda: f64,
}

impl TenantDelta {
    /// Create a new per-layer tenant delta with Kaiming init for A and zeros for B
    ///
    /// Creates `num_layers × num_modules` A/B pairs keyed as `"layer_idx.module_name"`.
    ///
    /// # Arguments
    /// * `config` - Delta configuration (rank, alpha, target modules, etc.)
    /// * `module_dims` - Map of module_name -> (in_features, out_features)
    /// * `device` - Device for tensor allocation
    /// * `num_layers` - Number of model layers
    pub fn new(
        config: &TenantDeltaConfig,
        module_dims: &HashMap<String, (usize, usize)>,
        device: Device,
        num_layers: usize,
    ) -> Result<Self> {
        let vs = VarStore::new(device);
        let root = vs.root();

        let mut lora_a = HashMap::new();
        let mut lora_b = HashMap::new();

        for layer_idx in 0..num_layers {
            for module_name in &config.target_modules {
                let (in_features, out_features) = module_dims
                    .get(module_name)
                    .ok_or_else(|| anyhow!("Module '{}' not found in model dimensions", module_name))?;

                let key = format!("{}.{}", layer_idx, module_name);

                // Use Path::sub() for hierarchy — '.' is VarStore's path separator
                let layer_path = root.sub(format!("layer_{}", layer_idx)).sub(module_name);

                // A: Kaiming uniform initialization [rank, in_features]
                let a = layer_path.kaiming_uniform(
                    "lora_a",
                    &[config.rank as i64, *in_features as i64],
                );

                // B: Zero initialization [out_features, rank]
                let b = layer_path.zeros(
                    "lora_b",
                    &[*out_features as i64, config.rank as i64],
                );

                lora_a.insert(key.clone(), a);
                lora_b.insert(key, b);
            }
        }

        // Build AdamW optimizer over all trainable parameters
        let optimizer = tch::nn::AdamW {
            beta1: 0.9,
            beta2: 0.999,
            wd: config.decay_lambda,
            eps: 1e-8,
            amsgrad: false,
        }
        .build(&vs, config.learning_rate)
        .map_err(|e| anyhow!("Failed to create AdamW optimizer: {}", e))?;

        let scaling = config.alpha as f64 / config.rank as f64;
        let now = Instant::now();

        Ok(Self {
            lora_a,
            lora_b,
            vs,
            optimizer,
            scaling,
            rank: config.rank,
            device,
            target_modules: config.target_modules.clone(),
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

        let x = x.to_kind(a.kind());
        let intermediate = x.f_matmul(&a.tr())
            .map_err(|e| anyhow!("Delta forward matmul A failed for '{}': {}", key, e))?;
        let output = intermediate.f_matmul(&b.tr())
            .map_err(|e| anyhow!("Delta forward matmul B failed for '{}': {}", key, e))?;

        Ok(output * self.scaling)
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

        let x = x.to_kind(a.kind());
        let intermediate = x.f_matmul(&a.tr())
            .map_err(|e| anyhow!("Delta forward_2d matmul A failed for '{}': {}", key, e))?;
        let output = intermediate.f_matmul(&b.tr())
            .map_err(|e| anyhow!("Delta forward_2d matmul B failed for '{}': {}", key, e))?;

        Ok(output * self.scaling)
    }

    /// Compute the ratio of delta norm to a reference norm for drift monitoring
    ///
    /// Returns a map of "layer_idx.module_name" -> ||delta|| / ||base|| where delta is
    /// the effective weight change (B @ A) and base is the reference norm.
    pub fn delta_norm_ratio(&self, base_norms: &HashMap<String, f64>) -> HashMap<String, f64> {
        let _guard = tch::no_grad_guard();
        let mut ratios = HashMap::new();

        for layer_idx in 0..self.num_layers {
            for module_name in &self.target_modules {
                let key = format!("{}.{}", layer_idx, module_name);
                if let (Some(a), Some(b)) = (self.lora_a.get(&key), self.lora_b.get(&key)) {
                    let delta = b.matmul(a) * self.scaling;
                    let delta_norm: f64 = delta.norm().double_value(&[]);
                    let base_norm = base_norms.get(module_name).copied().unwrap_or(1.0);
                    ratios.insert(key, delta_norm / base_norm.max(1e-8));
                }
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

        for layer_idx in 0..self.num_layers {
            for module_name in &self.target_modules {
                let key = format!("{}.{}", layer_idx, module_name);
                if let Some(a) = self.lora_a.get(&key) {
                    state.insert(format!("{}.lora_a", key), a.copy());
                }
                if let Some(b) = self.lora_b.get(&key) {
                    state.insert(format!("{}.lora_b", key), b.copy());
                }
            }
        }

        state
    }

    /// Load state dict (restore A and B tensors from a snapshot)
    ///
    /// Expects keys in format `"layer_idx.module_name.lora_a"` / `"layer_idx.module_name.lora_b"`.
    pub fn load_state_dict(&mut self, state: &HashMap<String, Tensor>) -> Result<()> {
        let _guard = tch::no_grad_guard();

        for layer_idx in 0..self.num_layers {
            for module_name in &self.target_modules {
                let key = format!("{}.{}", layer_idx, module_name);
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

        let ranks_match = base_guard.rank == tenant_guard.rank;

        for key in &all_keys {
            let base_a = base_guard.lora_a.get(key);
            let base_b = base_guard.lora_b.get(key);
            let tenant_a = tenant_guard.lora_a.get(key);
            let tenant_b = tenant_guard.lora_b.get(key);

            match (base_a, tenant_a) {
                (Some(ba), Some(ta)) if ranks_match => {
                    // Same rank: add A matrices directly (pre-scaled)
                    let base_effective_a = ba * base_guard.scaling;
                    let tenant_effective_a = ta * tenant_guard.scaling;
                    composed_a.insert(key.clone(), base_effective_a + tenant_effective_a);
                }
                (Some(ba), Some(ta)) => {
                    // Different ranks: compute effective weight W_eff = s1*(B1 @ A1) + s2*(B2 @ A2)
                    // then store as rank=out_dim identity-like decomposition: A=W_eff, B=I
                    // Instead, we compute the combined effective correction directly.
                    // We store the summed effective weight as A with B=identity,
                    // but that changes dimensions. Better approach: compute W_eff and
                    // store it with a thin SVD-like decomposition at the larger rank.
                    //
                    // Simplest correct approach: compute full effective weight per-key
                    // and store as A=[out_dim, in_dim], B=I[out_dim, out_dim] with scaling=1.
                    // But that's expensive. Instead, use the effective correction approach:
                    // store the pre-computed W_eff = s1*(B1@A1) + s2*(B2@A2) in composed_a
                    // with composed_b = I (identity) and scaling = 1.0.
                    //
                    // Actually, the cleanest approach is to just compute at forward time.
                    // Store separate effective weights: A = W_eff, B = I (identity of out_dim).
                    let bb = base_b.expect("base B must exist if base A exists");
                    let tb = tenant_b.expect("tenant B must exist if tenant A exists");
                    // W_eff = s1*(B1^T @ (A1^T))^T + s2*(B2^T @ (A2^T))^T
                    //       = s1*(B1 @ A1) + s2*(B2 @ A2)
                    // A is [rank, in_dim], B is [out_dim, rank]
                    // B @ A = [out_dim, in_dim]
                    // B @ A = [out_dim, rank] @ [rank, in_dim] = [out_dim, in_dim]
                    let base_w = bb.matmul(ba) * base_guard.scaling;
                    let tenant_w = tb.matmul(ta) * tenant_guard.scaling;
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
                    composed_a.insert(key.clone(), ba * base_guard.scaling);
                }
                (None, Some(ta)) => {
                    composed_a.insert(key.clone(), ta * tenant_guard.scaling);
                }
                (None, None) => {}
            }

            // Only process B separately when we didn't already handle it above
            // (i.e., ranks matched or only one side had the key)
            match (base_b, tenant_b) {
                (Some(bb), Some(tb)) => {
                    composed_b.insert(key.clone(), (bb + tb).shallow_clone());
                }
                (Some(bb), None) => {
                    composed_b.insert(key.clone(), bb.shallow_clone());
                }
                (None, Some(tb)) => {
                    composed_b.insert(key.clone(), tb.shallow_clone());
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
        // Composed delta is non-trainable — dummy optimizer
        let optimizer = tch::nn::AdamW {
            beta1: 0.9, beta2: 0.999, wd: 0.0, eps: 1e-8, amsgrad: false,
        }.build(&vs, 0.0).unwrap_or_else(|_| unreachable!("AdamW::build with valid VarStore"));
        let now = Instant::now();

        std::sync::Arc::new(parking_lot::Mutex::new(TenantDelta {
            lora_a: composed_a,
            lora_b: composed_b,
            vs,
            optimizer,
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

        for layer_idx in 0..self.num_layers {
            for module in &self.target_modules {
                let key = format!("{}.{}", layer_idx, module);
                let subpath = module_to_peft_subpath(module);

                if let Some(a) = self.lora_a.get(&key) {
                    let peft_key = format!(
                        "base_model.model.layers.{}.{}.{}.lora_A.weight",
                        layer_idx, subpath, module
                    );
                    pairs.push((peft_key, a.shallow_clone()));
                }
                if let Some(b) = self.lora_b.get(&key) {
                    let peft_key = format!(
                        "base_model.model.layers.{}.{}.{}.lora_B.weight",
                        layer_idx, subpath, module
                    );
                    pairs.push((peft_key, b.shallow_clone()));
                }
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
        let optimizer = tch::nn::AdamW {
            beta1: 0.9, beta2: 0.999, wd: 0.0, eps: 1e-8, amsgrad: false,
        }.build(&vs, 0.0).map_err(|e| anyhow!("Failed to create optimizer: {}", e))?;
        let now = Instant::now();

        tracing::info!(
            "Loaded LoRA adapter from bytes: rank={}, layers={}, modules={:?}, alpha={}",
            rank,
            num_layers,
            target_modules,
            alpha
        );

        Ok(Self {
            lora_a,
            lora_b,
            vs,
            optimizer,
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
}
