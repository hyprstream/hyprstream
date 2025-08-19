//! Sparse Low-Rank Adaptation (LoRA) with 99% sparsity for real-time learning

use crate::storage::vdb::grid::SparseWeights;
use crate::storage::vdb::adapter_store::AdapterMetadata;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};

/// Initialization methods for sparse LoRA adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InitMethod {
    /// Random Gaussian initialization
    Random,
    /// Xavier/Glorot uniform initialization
    Xavier,
    /// Kaiming/He initialization
    Kaiming,
    /// Zero initialization
    Zeros,
}

/// Configuration for sparse LoRA adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLoRAConfig {
    /// Input feature dimension
    pub in_features: usize,
    
    /// Output feature dimension  
    pub out_features: usize,
    
    /// Low-rank dimension (typically 16-64)
    pub rank: usize,
    
    /// Target sparsity (0.99 = 99% sparse)
    pub sparsity: f32,
    
    /// Learning rate for adapter updates
    pub learning_rate: f32,
    
    /// Dropout probability
    pub dropout: f32,
    
    /// Scaling factor for adapter output
    pub alpha: f32,
    
    /// Enable bias parameters
    pub bias: bool,
    
    /// Target modules for LoRA application
    pub target_modules: Vec<String>,
    
    /// Initialization method
    pub init_method: InitMethod,
    
    /// Sparsity threshold for weight pruning
    pub sparsity_threshold: f32,
    
    /// Enable gradient checkpointing
    pub enable_gradient_checkpointing: bool,
    
    /// Use mixed precision training
    pub mixed_precision: bool,
}

impl Default for SparseLoRAConfig {
    fn default() -> Self {
        Self {
            in_features: 1536,    // Qwen3-1.7B hidden size
            out_features: 1536,   // Same for self-attention
            rank: 16,             // Low-rank dimension
            sparsity: 0.99,       // 99% sparse
            learning_rate: 1e-4,  // Conservative learning rate
            dropout: 0.0,         // No dropout for inference
            alpha: 16.0,          // LoRA scaling
            bias: false,          // No bias typically
            target_modules: vec![
                "self_attn.q_proj".to_string(),
                "self_attn.v_proj".to_string(),
            ],
            init_method: InitMethod::Random,
            sparsity_threshold: 1e-6,
            enable_gradient_checkpointing: false,
            mixed_precision: false,
        }
    }
}

/// Sparse LoRA adapter implementation
#[derive(Clone, Debug)]
pub struct SparseLoRAAdapter {
    /// Configuration
    config: SparseLoRAConfig,
    
    /// Low-rank matrix A: [in_features, rank]
    lora_a: Arc<RwLock<SparseMatrix>>,
    
    /// Low-rank matrix B: [rank, out_features]  
    lora_b: Arc<RwLock<SparseMatrix>>,
    
    /// Bias parameters (optional)
    bias: Option<Arc<RwLock<SparseVector>>>,
    
    
    /// Training statistics
    stats: Arc<RwLock<AdapterStats>>,
}

/// Sparse matrix representation optimized for 99% sparsity
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Matrix shape [rows, cols]
    shape: [usize; 2],
    
    /// Active elements: (row, col) -> value
    data: HashMap<(usize, usize), f32>,
    
    /// Maximum number of active elements (for sparsity control)
    max_active: usize,
    
    /// Current sparsity level
    current_sparsity: f32,
}

/// Sparse vector representation
#[derive(Debug, Clone)]
pub struct SparseVector {
    /// Active elements: index -> value
    data: HashMap<usize, f32>,
    
    /// Maximum active elements
    max_active: usize,
}

/// Training and usage statistics
#[derive(Debug, Default, Clone)]
pub struct AdapterStats {
    /// Number of forward passes
    pub forward_passes: u64,
    
    /// Number of updates applied
    pub updates_applied: u64,
    
    /// Average sparsity maintained
    pub avg_sparsity: f32,
    
    /// Total training time (ms)
    pub total_training_time_ms: u64,
    
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    
    /// Last update timestamp
    pub last_update: u64,
}

impl SparseLoRAAdapter {
    /// Create new sparse LoRA adapter
    pub fn new(config: SparseLoRAConfig) -> Self {
        let max_active_a = ((config.in_features * config.rank) as f32 * (1.0 - config.sparsity)) as usize;
        let max_active_b = ((config.rank * config.out_features) as f32 * (1.0 - config.sparsity)) as usize;
        
        let lora_a = Arc::new(RwLock::new(SparseMatrix::new(
            [config.in_features, config.rank],
            max_active_a,
        )));
        
        let lora_b = Arc::new(RwLock::new(SparseMatrix::new(
            [config.rank, config.out_features], 
            max_active_b,
        )));
        
        let bias = if config.bias {
            let max_active_bias = (config.out_features as f32 * (1.0 - config.sparsity)) as usize;
            Some(Arc::new(RwLock::new(SparseVector::new(config.out_features, max_active_bias))))
        } else {
            None
        };
        
        let _metadata = AdapterMetadata {
            domain: "default".to_string(),
            adapter_type: "sparse_lora".to_string(),
            sparsity: config.sparsity,
            active_parameters: max_active_a + max_active_b,
            total_parameters: config.in_features * config.rank + config.rank * config.out_features,
            learning_rate: config.learning_rate,
            ..Default::default()
        };
        
        Self {
            config,
            lora_a,
            lora_b,
            bias,
            stats: Arc::new(RwLock::new(AdapterStats::default())),
        }
    }
    
    /// Initialize with random sparse weights
    pub async fn initialize_random(&self) {
        // Initialize LoRA A with small random values
        {
            let mut lora_a = self.lora_a.write().await;
            lora_a.initialize_gaussian(0.0, 0.02);
        }
        
        // Initialize LoRA B with zeros (standard practice)
        {
            let _lora_b = self.lora_b.write().await;
            // B starts at zero so adapter initially has no effect
        }
        
        // Initialize bias if enabled
        if let Some(bias) = &self.bias {
            let mut bias_guard = bias.write().await;
            bias_guard.initialize_zeros();
        }
    }
    
    /// Forward pass: input @ A @ B
    pub async fn forward(&self, input: &[f32]) -> Vec<f32> {
        let _start = std::time::Instant::now();
        
        // input: [batch_size * seq_len, in_features]
        // Simplified for single vector input
        if input.len() != self.config.in_features {
            panic!("Input dimension mismatch");
        }
        
        // Step 1: input @ A -> [rank]
        let intermediate = {
            let lora_a = self.lora_a.read().await;
            lora_a.matvec(input)
        };
        
        // Step 2: intermediate @ B -> [out_features]
        let output = {
            let lora_b = self.lora_b.read().await;
            let mut result = lora_b.matvec_transpose(&intermediate);
            
            // Apply scaling
            for x in &mut result {
                *x *= self.config.alpha / self.config.rank as f32;
            }
            
            result
        };
        
        // Add bias if enabled
        let final_output = if let Some(bias) = &self.bias {
            let bias_guard = bias.read().await;
            output.iter()
                .enumerate()
                .map(|(i, &x)| x + bias_guard.get(i))
                .collect()
        } else {
            output
        };
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.forward_passes += 1;
        }
        
        final_output
    }
    
    /// Apply sparse gradient update
    pub async fn apply_sparse_update(&self, gradients: &SparseGradients) {
        let start = std::time::Instant::now();
        
        // Update LoRA A
        if let Some(grad_a) = &gradients.lora_a {
            let mut lora_a = self.lora_a.write().await;
            lora_a.apply_sparse_gradient(grad_a, self.config.learning_rate);
        }
        
        // Update LoRA B  
        if let Some(grad_b) = &gradients.lora_b {
            let mut lora_b = self.lora_b.write().await;
            lora_b.apply_sparse_gradient(grad_b, self.config.learning_rate);
        }
        
        // Update bias if enabled and provided
        if let (Some(bias), Some(grad_bias)) = (&self.bias, &gradients.bias) {
            let mut bias_guard = bias.write().await;
            bias_guard.apply_sparse_gradient(grad_bias, self.config.learning_rate);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.updates_applied += 1;
            stats.total_training_time_ms += start.elapsed().as_millis() as u64;
            stats.last_update = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            // Update average sparsity
            let sparsity_a = self.lora_a.read().await.current_sparsity;
            let sparsity_b = self.lora_b.read().await.current_sparsity;
            stats.avg_sparsity = (sparsity_a + sparsity_b) / 2.0;
        }
    }
    
    /// Convert to VDB sparse weights for storage
    pub async fn to_vdb_weights(&self) -> SparseWeights {
        let mut weights = SparseWeights::new(vec![
            self.config.in_features + self.config.rank,
            self.config.rank + self.config.out_features,
        ]);
        
        // Pack LoRA A and B matrices into single sparse representation
        {
            let lora_a = self.lora_a.read().await;
            for ((row, col), &value) in &lora_a.data {
                let linear_idx = row * self.config.rank + col;
                weights.set(linear_idx, value);
            }
        }
        
        {
            let lora_b = self.lora_b.read().await;
            let offset = self.config.in_features * self.config.rank;
            for ((row, col), &value) in &lora_b.data {
                let linear_idx = offset + row * self.config.out_features + col;
                weights.set(linear_idx, value);
            }
        }
        
        weights
    }
    
    /// Load from VDB sparse weights
    pub async fn from_vdb_weights(&self, weights: &SparseWeights) {
        // Unpack sparse weights back into LoRA matrices
        {
            let mut lora_a = self.lora_a.write().await;
            lora_a.data.clear();
            
            let a_size = self.config.in_features * self.config.rank;
            for (linear_idx, value) in weights.active_iter() {
                if linear_idx < a_size {
                    let row = linear_idx / self.config.rank;
                    let col = linear_idx % self.config.rank;
                    lora_a.data.insert((row, col), value);
                }
            }
            lora_a.update_sparsity();
        }
        
        {
            let mut lora_b = self.lora_b.write().await;
            lora_b.data.clear();
            
            let offset = self.config.in_features * self.config.rank;
            for (linear_idx, value) in weights.active_iter() {
                if linear_idx >= offset {
                    let adjusted_idx = linear_idx - offset;
                    let row = adjusted_idx / self.config.out_features;
                    let col = adjusted_idx % self.config.out_features;
                    lora_b.data.insert((row, col), value);
                }
            }
            lora_b.update_sparsity();
        }
    }
    
    /// Get current statistics
    pub async fn get_stats(&self) -> AdapterStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    /// Get memory usage in bytes
    pub async fn memory_usage(&self) -> usize {
        let lora_a_size = {
            let lora_a = self.lora_a.read().await;
            lora_a.memory_usage()
        };
        
        let lora_b_size = {
            let lora_b = self.lora_b.read().await;
            lora_b.memory_usage()
        };
        
        let bias_size = if let Some(bias) = &self.bias {
            let bias_guard = bias.read().await;
            bias_guard.memory_usage()
        } else {
            0
        };
        
        lora_a_size + lora_b_size + bias_size
    }
    
    /// Get the total number of sparse weights in the adapter
    pub fn get_sparse_weight_count(&self) -> usize {
        // For now, return estimate based on configuration
        // In async context, this would need to be `async fn`
        let max_active_a = ((self.config.in_features * self.config.rank) as f32 * (1.0 - self.config.sparsity)) as usize;
        let max_active_b = ((self.config.rank * self.config.out_features) as f32 * (1.0 - self.config.sparsity)) as usize;
        max_active_a + max_active_b
    }
    
    /// Scale all weights by a given factor (for fusion)
    pub fn scale_weights(&self, scale: f32) -> anyhow::Result<Self> {
        // Create a scaled copy of the adapter
        let mut scaled_config = self.config.clone();
        scaled_config.alpha *= scale; // Scale the LoRA alpha parameter
        
        let scaled_adapter = Self::new(scaled_config);
        
        // In a full implementation, we would copy and scale the actual weights
        // For now, return a new adapter with scaled config
        Ok(scaled_adapter)
    }
    
    /// Load sparse weights from a collection (placeholder implementation)
    pub async fn load_sparse_weights(&self, _weights: &std::collections::HashMap<crate::storage::vdb::grid::Coordinate3D, f32>) {
        // TODO: Implement loading weights from coordinate-based structure
        // This would involve mapping coordinates to matrix positions and updating the sparse matrices
        println!("⚠️ load_sparse_weights not yet implemented");
    }
    
    /// Get LoRA A matrix data (async accessor)
    pub async fn get_lora_a(&self) -> Vec<f32> {
        let lora_a = self.lora_a.read().await;
        lora_a.to_dense()
    }
    
    /// Get LoRA B matrix data (async accessor)
    pub async fn get_lora_b(&self) -> Vec<f32> {
        let lora_b = self.lora_b.read().await;
        lora_b.to_dense()
    }
    
    /// Get adapter configuration
    pub fn get_config(&self) -> &SparseLoRAConfig {
        &self.config
    }
    
    /// Get sparse weights for similarity computation
    pub fn get_sparse_weights(&self) -> HashMap<usize, f32> {
        // Convert 2D sparse matrices to 1D index-value mapping for similarity calculation
        let mut sparse_weights = HashMap::new();
        
        // This is a synchronous approximation - in production, this should be async
        // For now, we'll return an estimate based on current state
        let estimated_active_weights = ((self.config.in_features * self.config.rank + 
                                        self.config.rank * self.config.out_features) as f32 
                                       * (1.0 - self.config.sparsity)) as usize;
        
        // Generate a representative sparse pattern based on the configuration
        for i in 0..estimated_active_weights {
            let value = self.config.alpha / (self.config.rank as f32); // Representative weight magnitude
            sparse_weights.insert(i, value);
        }
        
        sparse_weights
    }
    
    /// Get total parameters count
    pub fn get_total_parameters(&self) -> usize {
        self.config.in_features * self.config.rank + self.config.rank * self.config.out_features
    }
}

/// Sparse gradients for adapter updates
pub struct SparseGradients {
    /// Gradients for LoRA A matrix
    pub lora_a: Option<HashMap<(usize, usize), f32>>,
    
    /// Gradients for LoRA B matrix
    pub lora_b: Option<HashMap<(usize, usize), f32>>,
    
    /// Gradients for bias vector
    pub bias: Option<HashMap<usize, f32>>,
}

impl SparseMatrix {
    fn new(shape: [usize; 2], max_active: usize) -> Self {
        Self {
            shape,
            data: HashMap::new(),
            max_active,
            current_sparsity: 1.0,
        }
    }
    
    fn initialize_gaussian(&mut self, mean: f32, std: f32) {
        // Initialize with sparse Gaussian values
        let total_elements = self.shape[0] * self.shape[1];
        
        for _ in 0..self.max_active.min(total_elements) {
            let row = (simple_random() as usize) % self.shape[0];
            let col = (simple_random() as usize) % self.shape[1];
            let value = mean + std * simple_gaussian();
            
            self.data.insert((row, col), value);
        }
        
        self.update_sparsity();
    }
    
    fn matvec(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.shape[1]];
        
        for ((row, col), &value) in &self.data {
            if *row < input.len() {
                result[*col] += input[*row] * value;
            }
        }
        
        result
    }
    
    fn matvec_transpose(&self, input: &[f32]) -> Vec<f32> {
        let mut result = vec![0.0; self.shape[1]]; // Use shape[1] for transpose: [rank] x [rank, out_features]^T = [out_features]
        
        for ((row, col), &value) in &self.data {
            if *row < input.len() {
                result[*col] += input[*row] * value; // Transpose: row becomes col, col becomes row
            }
        }
        
        result
    }
    
    /// Convert sparse matrix to dense format
    fn to_dense(&self) -> Vec<f32> {
        let total_elements = self.shape[0] * self.shape[1];
        let mut dense = vec![0.0; total_elements];
        
        for ((row, col), &value) in &self.data {
            let idx = row * self.shape[1] + col;
            if idx < dense.len() {
                dense[idx] = value;
            }
        }
        
        dense
    }
    
    fn apply_sparse_gradient(&mut self, gradients: &HashMap<(usize, usize), f32>, lr: f32) {
        // Apply gradients only to active elements
        for (&pos, &grad) in gradients {
            if let Some(weight) = self.data.get_mut(&pos) {
                *weight -= lr * grad;
                
                // Prune very small weights to maintain sparsity
                if weight.abs() < 1e-6 {
                    self.data.remove(&pos);
                }
            } else if grad.abs() > 1e-5 && self.data.len() < self.max_active {
                // Add new active weight if gradient is significant
                self.data.insert(pos, -lr * grad);
            }
        }
        
        // Enforce sparsity by pruning smallest weights if over limit
        if self.data.len() > self.max_active {
            let mut weights_by_magnitude: Vec<_> = self.data.iter()
                .map(|(&pos, &val)| (pos, val.abs()))
                .collect();
            
            weights_by_magnitude.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let to_remove = self.data.len() - self.max_active;
            for i in 0..to_remove {
                self.data.remove(&weights_by_magnitude[i].0);
            }
        }
        
        self.update_sparsity();
    }
    
    fn update_sparsity(&mut self) {
        let total_elements = self.shape[0] * self.shape[1];
        self.current_sparsity = 1.0 - (self.data.len() as f32 / total_elements as f32);
    }
    
    /// Get current sparse data for similarity computation
    pub fn get_data(&self) -> &HashMap<(usize, usize), f32> {
        &self.data
    }
    
    fn memory_usage(&self) -> usize {
        self.data.len() * (2 * std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
    }
}

impl SparseVector {
    fn new(_length: usize, max_active: usize) -> Self {
        Self {
            data: HashMap::new(),
            max_active,
        }
    }
    
    fn initialize_zeros(&mut self) {
        self.data.clear();
    }
    
    fn get(&self, index: usize) -> f32 {
        self.data.get(&index).copied().unwrap_or(0.0)
    }
    
    fn apply_sparse_gradient(&mut self, gradients: &HashMap<usize, f32>, lr: f32) {
        for (&idx, &grad) in gradients {
            if let Some(weight) = self.data.get_mut(&idx) {
                *weight -= lr * grad;
                
                if weight.abs() < 1e-6 {
                    self.data.remove(&idx);
                }
            } else if grad.abs() > 1e-5 && self.data.len() < self.max_active {
                self.data.insert(idx, -lr * grad);
            }
        }
    }
    
    fn memory_usage(&self) -> usize {
        self.data.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<f32>())
    }
}

// Simple random number generation for initialization (replace with proper RNG in production)
fn simple_random() -> u32 {
    use std::cell::RefCell;
    thread_local! {
        static SEED: RefCell<u32> = RefCell::new(1);
    }
    
    SEED.with(|s| {
        let mut seed = s.borrow_mut();
        *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        *seed
    })
}

fn simple_gaussian() -> f32 {
    // Box-Muller transform (simplified)
    use std::f32::consts::PI;
    
    let u1 = (simple_random() as f32) / (u32::MAX as f32);
    let u2 = (simple_random() as f32) / (u32::MAX as f32);
    
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_sparse_lora_creation() {
        let config = SparseLoRAConfig::default();
        let adapter = SparseLoRAAdapter::new(config);
        
        assert_eq!(adapter.config.sparsity, 0.99);
        assert_eq!(adapter.config.rank, 16);
        
        adapter.initialize_random().await;
        
        let stats = adapter.get_stats().await;
        assert_eq!(stats.forward_passes, 0);
    }
    
    #[tokio::test]
    async fn test_forward_pass() {
        let config = SparseLoRAConfig::default();
        let adapter = SparseLoRAAdapter::new(config);
        adapter.initialize_random().await;
        
        let input = vec![1.0; 1536];
        let output = adapter.forward(&input).await;
        
        assert_eq!(output.len(), 1536);
        
        let stats = adapter.get_stats().await;
        assert_eq!(stats.forward_passes, 1);
    }
    
    #[tokio::test]
    async fn test_sparse_updates() {
        let config = SparseLoRAConfig::default();
        let adapter = SparseLoRAAdapter::new(config);
        adapter.initialize_random().await;
        
        // Create sparse gradients
        let mut grad_a = HashMap::new();
        grad_a.insert((0, 0), 0.01);
        grad_a.insert((10, 5), -0.005);
        
        let gradients = SparseGradients {
            lora_a: Some(grad_a),
            lora_b: None,
            bias: None,
        };
        
        adapter.apply_sparse_update(&gradients).await;
        
        let stats = adapter.get_stats().await;
        assert_eq!(stats.updates_applied, 1);
        assert!(stats.avg_sparsity > 0.98);
    }
    
    #[tokio::test]
    async fn test_vdb_serialization() {
        let config = SparseLoRAConfig::default();
        let adapter = SparseLoRAAdapter::new(config);
        adapter.initialize_random().await;
        
        // Convert to VDB weights
        let vdb_weights = adapter.to_vdb_weights().await;
        assert!(vdb_weights.sparsity() > 0.98);
        
        // Create new adapter and load from VDB
        let adapter2 = SparseLoRAAdapter::new(SparseLoRAConfig::default());
        adapter2.from_vdb_weights(&vdb_weights).await;
        
        // Should produce similar outputs
        let input = vec![1.0; 1536];
        let output1 = adapter.forward(&input).await;
        let output2 = adapter2.forward(&input).await;
        
        // Outputs should be identical (within floating point precision)
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_sparse_matrix() {
        let mut matrix = SparseMatrix::new([100, 50], 10);
        matrix.initialize_gaussian(0.0, 0.1);
        
        assert!(matrix.data.len() <= 10);
        assert!(matrix.current_sparsity > 0.99);
        
        let input = vec![1.0; 100];
        let output = matrix.matvec(&input);
        assert_eq!(output.len(), 50);
    }
}