//! LoRA adapter fusion for combining multiple adapters

use crate::adapters::sparse_lora::SparseLoRAAdapter;
use crate::inference::FusedAdapterWeights;

use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Strategy for fusing multiple LoRA adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple weighted average of adapter weights
    WeightedAverage,
    
    /// Learnable fusion with attention mechanism
    AttentionFusion,
    
    /// Sequential application of adapters
    Sequential,
    
    /// Task-specific routing between adapters
    TaskRouting,
    
    /// Sparse mixture of adapters
    SparseMixture { top_k: usize },
}

/// Dynamic fusion weights with attention-based computation
#[derive(Debug, Clone)]
pub struct DynamicFusionWeights {
    /// Per-adapter attention weights
    pub attention_weights: HashMap<String, f32>,
    /// Layer-wise attention patterns
    pub layer_weights: HashMap<String, Vec<f32>>,
    /// Adaptive threshold for weight allocation
    pub adaptive_threshold: f32,
    /// Similarity matrix between adapters
    pub similarity_matrix: HashMap<(String, String), f32>,
}

impl Default for DynamicFusionWeights {
    fn default() -> Self {
        Self {
            attention_weights: HashMap::new(),
            layer_weights: HashMap::new(),
            adaptive_threshold: 0.1,
            similarity_matrix: HashMap::new(),
        }
    }
}

/// LoRA fusion engine
pub struct LoRAFusion {
    strategy: FusionStrategy,
    
    /// Fusion statistics
    stats: FusionStats,
    
    /// Dynamic fusion weights computation
    dynamic_weights: DynamicFusionWeights,
}

/// Statistics about adapter fusion
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FusionStats {
    pub total_fusions: u64,
    pub avg_fusion_time_ms: f64,
    pub avg_adapters_per_fusion: f64,
    pub total_weights_fused: u64,
    pub compression_ratio: f32,
}

impl LoRAFusion {
    /// Create new fusion engine with specified strategy
    pub fn new(strategy: FusionStrategy) -> Self {
        Self {
            strategy,
            stats: FusionStats::default(),
            dynamic_weights: DynamicFusionWeights::default(),
        }
    }
    
    /// Create fusion engine with layer-wise attention
    pub fn with_layer_attention(strategy: FusionStrategy, num_layers: usize) -> Self {
        let mut fusion = Self::new(strategy);
        
        // Initialize layer-wise weights for attention patterns
        fusion.dynamic_weights.layer_weights = HashMap::new();
        for layer_idx in 0..num_layers {
            let layer_key = format!("layer_{}", layer_idx);
            fusion.dynamic_weights.layer_weights.insert(layer_key, vec![1.0; 8]); // 8 attention heads typical
        }
        
        fusion
    }
    
    /// Fuse multiple adapters with specified weights
    pub fn fuse_adapters(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        weights: &HashMap<String, f32>,
    ) -> Result<FusedAdapterWeights> {
        let start_time = std::time::Instant::now();
        
        if adapters.is_empty() {
            return Err(anyhow::anyhow!("No adapters provided for fusion"));
        }
        
        let adapters_len = adapters.len();
        
        let fused_weights = match &self.strategy {
            FusionStrategy::WeightedAverage => {
                self.fuse_weighted_average(adapters, weights)?
            }
            FusionStrategy::AttentionFusion => {
                self.fuse_with_attention(adapters, weights)?
            }
            FusionStrategy::Sequential => {
                self.fuse_sequential(adapters, weights)?
            }
            FusionStrategy::TaskRouting => {
                self.fuse_task_routing(adapters, weights)?
            }
            FusionStrategy::SparseMixture { top_k } => {
                self.fuse_sparse_mixture(adapters, weights, *top_k)?
            }
        };
        
        // Update statistics
        let fusion_time = start_time.elapsed().as_millis() as f64;
        self.stats.total_fusions += 1;
        self.stats.avg_fusion_time_ms = (self.stats.avg_fusion_time_ms * (self.stats.total_fusions - 1) as f64 + fusion_time) 
            / self.stats.total_fusions as f64;
        self.stats.avg_adapters_per_fusion = (self.stats.avg_adapters_per_fusion * (self.stats.total_fusions - 1) as f64 + adapters_len as f64)
            / self.stats.total_fusions as f64;
        
        Ok(fused_weights)
    }
    
    /// Fuse adapters using weighted average
    fn fuse_weighted_average(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        weights: &HashMap<String, f32>,
    ) -> Result<FusedAdapterWeights> {
        println!("🔀 Fusing {} adapters using weighted average", adapters.len());
        
        let mut fused_adapters = HashMap::new();
        let total_sparse_weights = adapters.iter()
            .map(|(_, adapter)| adapter.get_sparse_weight_count())
            .sum();
        
        // For weighted average, we combine the sparse weights
        for (adapter_id, adapter) in adapters {
            let weight = weights.get(&adapter_id).copied().unwrap_or(1.0);
            
            // Apply weight scaling to the adapter
            let scaled_adapter = adapter.scale_weights(weight)?;
            fused_adapters.insert(adapter_id, scaled_adapter);
        }
        
        let fusion_metadata = crate::inference::FusionMetadata {
            num_adapters: fused_adapters.len(),
            total_sparse_weights,
            fusion_strategy: "weighted_average".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        Ok(FusedAdapterWeights {
            weights: fused_adapters,
            fusion_metadata,
        })
    }
    
    /// Fuse adapters using attention mechanism
    fn fuse_with_attention(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        _weights: &HashMap<String, f32>,
    ) -> Result<FusedAdapterWeights> {
        println!("🎯 Fusing {} adapters using attention mechanism", adapters.len());
        
        // For attention fusion, we would use a learned attention mechanism
        // For now, implement a simplified version that uses adapter similarity
        let mut fused_adapters = HashMap::new();
        let mut total_sparse_weights = 0;
        
        // Calculate attention weights based on adapter similarity
        let attention_weights = self.calculate_attention_weights(&adapters)?;
        
        for (adapter_id, adapter) in adapters {
            let attention_weight = attention_weights.get(&adapter_id).copied().unwrap_or(1.0);
            let scaled_adapter = adapter.scale_weights(attention_weight)?;
            
            total_sparse_weights += scaled_adapter.get_sparse_weight_count();
            fused_adapters.insert(adapter_id, scaled_adapter);
        }
        
        let fusion_metadata = crate::inference::FusionMetadata {
            num_adapters: fused_adapters.len(),
            total_sparse_weights,
            fusion_strategy: "attention_fusion".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        Ok(FusedAdapterWeights {
            weights: fused_adapters,
            fusion_metadata,
        })
    }
    
    /// Fuse adapters sequentially
    fn fuse_sequential(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        weights: &HashMap<String, f32>,
    ) -> Result<FusedAdapterWeights> {
        println!("🔗 Fusing {} adapters sequentially", adapters.len());
        
        // Sort adapters by weight (highest weight applied last)
        let mut sorted_adapters = adapters;
        sorted_adapters.sort_by(|(id_a, _), (id_b, _)| {
            let weight_a = weights.get(id_a).copied().unwrap_or(0.0);
            let weight_b = weights.get(id_b).copied().unwrap_or(0.0);
            weight_a.partial_cmp(&weight_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let mut fused_adapters = HashMap::new();
        let mut total_sparse_weights = 0;
        
        for (adapter_id, adapter) in sorted_adapters {
            let weight = weights.get(&adapter_id).copied().unwrap_or(1.0);
            let scaled_adapter = adapter.scale_weights(weight)?;
            
            total_sparse_weights += scaled_adapter.get_sparse_weight_count();
            fused_adapters.insert(adapter_id, scaled_adapter);
        }
        
        let fusion_metadata = crate::inference::FusionMetadata {
            num_adapters: fused_adapters.len(),
            total_sparse_weights,
            fusion_strategy: "sequential".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        Ok(FusedAdapterWeights {
            weights: fused_adapters,
            fusion_metadata,
        })
    }
    
    /// Fuse adapters using task-based routing
    fn fuse_task_routing(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        weights: &HashMap<String, f32>,
    ) -> Result<FusedAdapterWeights> {
        println!("🎭 Fusing {} adapters using task routing", adapters.len());
        
        // For task routing, we would typically use a routing network
        // For now, implement a simplified version based on weights
        let mut fused_adapters = HashMap::new();
        let mut total_sparse_weights = 0;
        
        // Select top adapters based on weights
        let mut weighted_adapters: Vec<_> = adapters.into_iter().collect();
        weighted_adapters.sort_by(|(id_a, _), (id_b, _)| {
            let weight_a = weights.get(id_a).copied().unwrap_or(0.0);
            let weight_b = weights.get(id_b).copied().unwrap_or(0.0);
            weight_b.partial_cmp(&weight_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Use top 2 adapters for routing
        for (adapter_id, adapter) in weighted_adapters.into_iter().take(2) {
            let weight = weights.get(&adapter_id).copied().unwrap_or(1.0);
            let scaled_adapter = adapter.scale_weights(weight)?;
            
            total_sparse_weights += scaled_adapter.get_sparse_weight_count();
            fused_adapters.insert(adapter_id, scaled_adapter);
        }
        
        let fusion_metadata = crate::inference::FusionMetadata {
            num_adapters: fused_adapters.len(),
            total_sparse_weights,
            fusion_strategy: "task_routing".to_string(),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        Ok(FusedAdapterWeights {
            weights: fused_adapters,
            fusion_metadata,
        })
    }
    
    /// Fuse adapters using sparse mixture
    fn fuse_sparse_mixture(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        weights: &HashMap<String, f32>,
        top_k: usize,
    ) -> Result<FusedAdapterWeights> {
        println!("🎲 Fusing {} adapters using sparse mixture (top-{})", adapters.len(), top_k);
        
        // Select top-k adapters based on weights
        let mut weighted_adapters: Vec<_> = adapters.into_iter().collect();
        weighted_adapters.sort_by(|(id_a, _), (id_b, _)| {
            let weight_a = weights.get(id_a).copied().unwrap_or(0.0);
            let weight_b = weights.get(id_b).copied().unwrap_or(0.0);
            weight_b.partial_cmp(&weight_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let mut fused_adapters = HashMap::new();
        let mut total_sparse_weights = 0;
        
        // Use only top-k adapters
        for (adapter_id, adapter) in weighted_adapters.into_iter().take(top_k) {
            let weight = weights.get(&adapter_id).copied().unwrap_or(1.0);
            let scaled_adapter = adapter.scale_weights(weight)?;
            
            total_sparse_weights += scaled_adapter.get_sparse_weight_count();
            fused_adapters.insert(adapter_id, scaled_adapter);
        }
        
        let fusion_metadata = crate::inference::FusionMetadata {
            num_adapters: fused_adapters.len(),
            total_sparse_weights,
            fusion_strategy: format!("sparse_mixture_k{}", top_k),
            timestamp: chrono::Utc::now().timestamp(),
        };
        
        Ok(FusedAdapterWeights {
            weights: fused_adapters,
            fusion_metadata,
        })
    }
    
    /// Calculate attention weights based on adapter similarity using dynamic fusion
    fn calculate_attention_weights(
        &mut self,
        adapters: &[(String, SparseLoRAAdapter)],
    ) -> Result<HashMap<String, f32>> {
        let mut attention_weights = HashMap::new();
        
        if adapters.len() <= 1 {
            // Single adapter or no adapters
            for (adapter_id, _) in adapters {
                attention_weights.insert(adapter_id.clone(), 1.0);
            }
            return Ok(attention_weights);
        }
        
        println!("🧠 Computing dynamic attention weights for {} adapters", adapters.len());
        
        // Step 1: Calculate pairwise similarities between adapters
        let mut similarity_matrix = HashMap::new();
        for (i, (id1, adapter1)) in adapters.iter().enumerate() {
            for (id2, adapter2) in adapters.iter().skip(i + 1) {
                let similarity = self.compute_adapter_similarity(adapter1, adapter2)?;
                similarity_matrix.insert((id1.clone(), id2.clone()), similarity);
                similarity_matrix.insert((id2.clone(), id1.clone()), similarity);
            }
        }
        
        // Step 2: Compute attention weights using adaptive mechanism
        let mut raw_weights = HashMap::new();
        for (adapter_id, adapter) in adapters {
            // Base weight from adapter magnitude and sparsity
            let magnitude_weight = self.compute_magnitude_weight(adapter)?;
            let sparsity_weight = self.compute_sparsity_weight(adapter)?;
            
            // Similarity-based adjustment
            let similarity_adjustment = self.compute_similarity_adjustment(
                adapter_id, 
                adapters, 
                &similarity_matrix
            )?;
            
            // Combined weight with adaptive thresholding
            let raw_weight = magnitude_weight * 0.4 + sparsity_weight * 0.3 + similarity_adjustment * 0.3;
            raw_weights.insert(adapter_id.clone(), raw_weight);
        }
        
        // Step 3: Apply adaptive threshold and normalize
        let max_weight = raw_weights.values().cloned().fold(0.0f32, f32::max);
        let adaptive_threshold = self.dynamic_weights.adaptive_threshold * max_weight;
        
        // Filter weights below threshold and normalize
        let mut filtered_weights: Vec<(String, f32)> = raw_weights
            .into_iter()
            .filter(|(_, weight)| *weight >= adaptive_threshold)
            .collect();
            
        // Ensure at least one adapter is selected
        if filtered_weights.is_empty() {
            // Select the adapter with highest weight
            if let Some((best_id, best_weight)) = adapters.iter()
                .map(|(id, adapter)| (id.clone(), self.compute_magnitude_weight(adapter).unwrap_or(0.1)))
                .max_by(|(_, w1), (_, w2)| w1.partial_cmp(w2).unwrap_or(std::cmp::Ordering::Equal)) {
                filtered_weights.push((best_id, best_weight));
            }
        }
        
        // Normalize weights to sum to 1.0
        let total_weight: f32 = filtered_weights.iter().map(|(_, w)| w).sum();
        if total_weight > 0.0 {
            for (adapter_id, weight) in filtered_weights {
                attention_weights.insert(adapter_id, weight / total_weight);
            }
        } else {
            // Fallback to uniform weights
            let uniform_weight = 1.0 / adapters.len() as f32;
            for (adapter_id, _) in adapters {
                attention_weights.insert(adapter_id.clone(), uniform_weight);
            }
        }
        
        // Update internal state
        self.dynamic_weights.attention_weights = attention_weights.clone();
        self.dynamic_weights.similarity_matrix = similarity_matrix;
        
        println!("🎯 Dynamic weights computed: {:?}", attention_weights);
        Ok(attention_weights)
    }
    
    /// Update layer-wise attention patterns based on performance feedback
    pub fn update_layer_attention(&mut self, layer_name: &str, performance_scores: &[f32]) {
        if let Some(layer_weights) = self.dynamic_weights.layer_weights.get_mut(layer_name) {
            // Update layer attention weights with exponential moving average
            let alpha = 0.1; // Learning rate for attention updates
            
            for (i, &score) in performance_scores.iter().enumerate() {
                if i < layer_weights.len() {
                    layer_weights[i] = (1.0 - alpha) * layer_weights[i] + alpha * score;
                }
            }
            
            // Normalize layer weights to sum to 1.0
            let sum: f32 = layer_weights.iter().sum();
            if sum > 0.0 {
                for weight in layer_weights.iter_mut() {
                    *weight /= sum;
                }
            }
            
            println!("🎯 Updated layer attention for {}: {:?}", layer_name, layer_weights);
        }
    }
    
    /// Get layer-wise attention weights for a specific layer
    pub fn get_layer_attention(&self, layer_name: &str) -> Option<&Vec<f32>> {
        self.dynamic_weights.layer_weights.get(layer_name)
    }
    
    /// Compute layer-specific fusion weights
    pub fn compute_layer_fusion_weights(
        &self,
        layer_name: &str,
        base_weights: &HashMap<String, f32>
    ) -> HashMap<String, f32> {
        let mut layer_adjusted_weights = base_weights.clone();
        
        if let Some(layer_attention) = self.dynamic_weights.layer_weights.get(layer_name) {
            // Apply layer-specific attention modulation
            let layer_importance = layer_attention.iter().sum::<f32>() / layer_attention.len() as f32;
            
            for (_, weight) in layer_adjusted_weights.iter_mut() {
                *weight *= layer_importance;
            }
        }
        
        // Renormalize
        let total: f32 = layer_adjusted_weights.values().sum();
        if total > 0.0 {
            for weight in layer_adjusted_weights.values_mut() {
                *weight /= total;
            }
        }
        
        layer_adjusted_weights
    }
    
    /// Compute similarity between two LoRA adapters based on weight patterns
    fn compute_adapter_similarity(
        &self,
        adapter1: &SparseLoRAAdapter,
        adapter2: &SparseLoRAAdapter,
    ) -> Result<f32> {
        // Get sparse weight patterns for both adapters
        let weights1 = adapter1.get_sparse_weights();
        let weights2 = adapter2.get_sparse_weights();
        
        if weights1.is_empty() || weights2.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate cosine similarity between weight vectors
        let mut dot_product = 0.0f32;
        let mut norm1 = 0.0f32;
        let mut norm2 = 0.0f32;
        
        // Find common indices and compute similarity
        for (idx, &val1) in weights1.iter() {
            if let Some(&val2) = weights2.get(idx) {
                dot_product += val1 * val2;
            }
            norm1 += val1 * val1;
        }
        
        for &val2 in weights2.values() {
            norm2 += val2 * val2;
        }
        
        let norm_product = (norm1 * norm2).sqrt();
        if norm_product > 0.0 {
            Ok(dot_product / norm_product)
        } else {
            Ok(0.0)
        }
    }
    
    /// Compute weight based on adapter magnitude
    fn compute_magnitude_weight(&self, adapter: &SparseLoRAAdapter) -> Result<f32> {
        let weights = adapter.get_sparse_weights();
        if weights.is_empty() {
            return Ok(0.1); // Minimum weight
        }
        
        let magnitude: f32 = weights.values().map(|&w| w.abs()).sum();
        Ok((magnitude / weights.len() as f32).min(1.0))
    }
    
    /// Compute weight based on adapter sparsity pattern
    fn compute_sparsity_weight(&self, adapter: &SparseLoRAAdapter) -> Result<f32> {
        let sparse_count = adapter.get_sparse_weight_count();
        let total_params = adapter.get_total_parameters(); // Assuming this method exists
        
        if total_params == 0 {
            return Ok(0.1);
        }
        
        let sparsity_ratio = sparse_count as f32 / total_params as f32;
        // Higher sparsity gets higher weight (more efficient)
        Ok(sparsity_ratio.min(1.0))
    }
    
    /// Compute similarity-based adjustment for an adapter
    fn compute_similarity_adjustment(
        &self,
        adapter_id: &str,
        adapters: &[(String, SparseLoRAAdapter)],
        similarity_matrix: &HashMap<(String, String), f32>,
    ) -> Result<f32> {
        let mut total_similarity = 0.0f32;
        let mut count = 0;
        
        for (other_id, _) in adapters {
            if other_id != adapter_id {
                if let Some(&similarity) = similarity_matrix.get(&(adapter_id.to_string(), other_id.clone())) {
                    total_similarity += similarity;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            // Lower average similarity means more unique/important adapter
            let avg_similarity = total_similarity / count as f32;
            Ok(1.0 - avg_similarity.min(1.0))
        } else {
            Ok(0.5) // Neutral weight
        }
    }
    
    /// Get fusion statistics
    pub fn get_stats(&self) -> &FusionStats {
        &self.stats
    }
    
    /// Reset fusion statistics
    pub fn reset_stats(&mut self) {
        self.stats = FusionStats::default();
    }
    
    /// Get fusion strategy
    pub fn strategy(&self) -> &FusionStrategy {
        &self.strategy
    }
    
    /// Update fusion strategy
    pub fn set_strategy(&mut self, strategy: FusionStrategy) {
        self.strategy = strategy;
    }
    
    /// Benchmark different fusion strategies
    pub async fn benchmark_strategies(
        &mut self,
        adapters: Vec<(String, SparseLoRAAdapter)>,
        weights: &HashMap<String, f32>,
    ) -> Result<HashMap<String, f64>> {
        let mut benchmark_results = HashMap::new();
        
        let strategies = vec![
            FusionStrategy::WeightedAverage,
            FusionStrategy::AttentionFusion,
            FusionStrategy::Sequential,
            FusionStrategy::SparseMixture { top_k: 2 },
        ];
        
        for strategy in strategies {
            let original_strategy = std::mem::replace(&mut self.strategy, strategy.clone());
            
            let start_time = std::time::Instant::now();
            let _result = self.fuse_adapters(adapters.clone(), weights)?;
            let duration = start_time.elapsed().as_micros() as f64 / 1000.0; // Convert to ms
            
            let strategy_name = match strategy {
                FusionStrategy::WeightedAverage => "weighted_average",
                FusionStrategy::AttentionFusion => "attention_fusion",
                FusionStrategy::Sequential => "sequential",
                FusionStrategy::TaskRouting => "task_routing",
                FusionStrategy::SparseMixture { top_k } => {
                    benchmark_results.insert(format!("sparse_mixture_k{}", top_k), duration);
                    continue;
                }
            };
            
            benchmark_results.insert(strategy_name.to_string(), duration);
            self.strategy = original_strategy.clone();
        }
        
        Ok(benchmark_results)
    }
    
    /// Optimize fusion strategy based on performance metrics
    pub fn optimize_strategy(
        &mut self,
        performance_metrics: &HashMap<String, f32>,
    ) -> FusionStrategy {
        // Find the strategy with best performance
        let mut best_strategy = self.strategy.clone();
        let mut best_score = 0.0f32;
        
        for (strategy_name, score) in performance_metrics {
            if *score > best_score {
                best_score = *score;
                best_strategy = match strategy_name.as_str() {
                    "weighted_average" => FusionStrategy::WeightedAverage,
                    "attention_fusion" => FusionStrategy::AttentionFusion,
                    "sequential" => FusionStrategy::Sequential,
                    "task_routing" => FusionStrategy::TaskRouting,
                    s if s.starts_with("sparse_mixture_k") => {
                        let k = s.strip_prefix("sparse_mixture_k")
                            .and_then(|k_str| k_str.parse().ok())
                            .unwrap_or(2);
                        FusionStrategy::SparseMixture { top_k: k }
                    }
                    _ => continue,
                };
            }
        }
        
        self.strategy = best_strategy.clone();
        println!("🎯 Optimized fusion strategy to: {:?}", best_strategy);
        
        best_strategy
    }
}

use chrono;