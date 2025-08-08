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

/// LoRA fusion engine
pub struct LoRAFusion {
    strategy: FusionStrategy,
    
    /// Fusion statistics
    stats: FusionStats,
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
        }
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
        &self,
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
        &self,
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
        &self,
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
        &self,
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
        &self,
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
    
    /// Calculate attention weights based on adapter similarity
    fn calculate_attention_weights(
        &self,
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
        
        // Calculate pairwise similarities and derive attention weights
        // For now, use uniform weights as a placeholder
        let uniform_weight = 1.0 / adapters.len() as f32;
        for (adapter_id, _) in adapters {
            attention_weights.insert(adapter_id.clone(), uniform_weight);
        }
        
        // TODO: Implement actual similarity calculation based on:
        // - Weight magnitudes
        // - Sparsity patterns
        // - Historical performance
        
        Ok(attention_weights)
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