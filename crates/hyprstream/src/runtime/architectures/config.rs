//! Architecture configuration traits and implementations

use tch::Kind as DType;

/// Core configuration for model architectures
pub trait ArchitectureConfig: Send + Sync {
    /// Number of attention heads for queries
    fn num_attention_heads(&self) -> usize;

    /// Number of key-value heads (for MQA/GQA)
    fn num_key_value_heads(&self) -> usize;

    /// Hidden dimension size
    fn hidden_size(&self) -> usize;

    /// Intermediate/FFN dimension size
    fn intermediate_size(&self) -> usize;

    /// Head dimension (hidden_size / num_attention_heads)
    fn head_dim(&self) -> usize {
        self.hidden_size() / self.num_attention_heads()
    }

    /// Vocabulary size
    fn vocab_size(&self) -> usize;

    /// Maximum position embeddings
    fn max_position_embeddings(&self) -> usize;

    /// RoPE base frequency (if applicable)
    fn rope_theta(&self) -> Option<f32>;

    /// RoPE dimension count (if different from head_dim)
    fn rope_dim(&self) -> Option<usize>;

    /// Layer normalization epsilon
    fn layer_norm_eps(&self) -> f32;

    /// Whether to use RMSNorm instead of LayerNorm
    fn use_rms_norm(&self) -> bool;

    /// Whether this architecture uses parallel attention and FFN
    fn use_parallel_residual(&self) -> bool {
        false
    }

    /// Get attention type (MHA, MQA, GQA)
    fn attention_type(&self) -> AttentionType {
        let num_heads = self.num_attention_heads();
        let num_kv_heads = self.num_key_value_heads();

        if num_kv_heads == 1 {
            AttentionType::MQA
        } else if num_kv_heads < num_heads {
            AttentionType::GQA {
                num_groups: num_heads / num_kv_heads,
            }
        } else {
            AttentionType::MHA
        }
    }

    /// Default dtype for this architecture  
    fn default_dtype(&self) -> DType {
        DType::Half // F16 equivalent in tch
    }
}

/// Attention configuration
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub use_flash_attention: bool,
    pub use_sliding_window: bool,
    pub sliding_window_size: Option<usize>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            rotary_dim: None,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            use_flash_attention: false,
            use_sliding_window: false,
            sliding_window_size: None,
        }
    }
}

/// Attention mechanism types
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionType {
    /// Multi-Head Attention (all heads have separate K,V)
    MHA,
    /// Multi-Query Attention (single K,V shared across all Q heads)
    MQA,
    /// Grouped-Query Attention (groups of Q heads share K,V)
    GQA { num_groups: usize },
}

impl AttentionType {
    /// Get the expansion factor for key-value heads
    pub fn kv_expansion_factor(&self, num_heads: usize) -> usize {
        match self {
            AttentionType::MHA => 1,
            AttentionType::MQA => num_heads,
            AttentionType::GQA { num_groups } => num_heads / num_groups,
        }
    }

    /// Check if KV cache can be shared
    pub fn supports_kv_cache_sharing(&self) -> bool {
        !matches!(self, AttentionType::MHA)
    }
}

/// Normalization types
#[derive(Debug, Clone, PartialEq)]
pub enum NormalizationType {
    LayerNorm,
    RMSNorm,
    GroupNorm { num_groups: usize },
}

/// Feed-forward network configuration
#[derive(Debug, Clone)]
pub struct FFNConfig {
    pub intermediate_size: usize,
    pub hidden_size: usize,
    pub activation: ActivationType,
    pub use_gate: bool,
    pub dropout: f32,
}

/// Activation function types
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationType {
    GELU,
    SiLU, // Swish
    ReLU,
    GEGLU,
    Swish,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_type_detection() {
        struct TestConfig {
            heads: usize,
            kv_heads: usize,
        }

        impl ArchitectureConfig for TestConfig {
            fn num_attention_heads(&self) -> usize {
                self.heads
            }
            fn num_key_value_heads(&self) -> usize {
                self.kv_heads
            }
            fn hidden_size(&self) -> usize {
                4096
            }
            fn intermediate_size(&self) -> usize {
                11008
            }
            fn vocab_size(&self) -> usize {
                32000
            }
            fn max_position_embeddings(&self) -> usize {
                4096
            }
            fn rope_theta(&self) -> Option<f32> {
                Some(10000.0)
            }
            fn rope_dim(&self) -> Option<usize> {
                None
            }
            fn layer_norm_eps(&self) -> f32 {
                1e-6
            }
            fn use_rms_norm(&self) -> bool {
                true
            }
        }

        // MHA: equal heads
        let config = TestConfig {
            heads: 32,
            kv_heads: 32,
        };
        assert_eq!(config.attention_type(), AttentionType::MHA);

        // MQA: single KV head
        let config = TestConfig {
            heads: 32,
            kv_heads: 1,
        };
        assert_eq!(config.attention_type(), AttentionType::MQA);

        // GQA: grouped heads
        let config = TestConfig {
            heads: 32,
            kv_heads: 8,
        };
        assert_eq!(
            config.attention_type(),
            AttentionType::GQA { num_groups: 4 }
        );
    }

    #[test]
    fn test_kv_expansion_factor() {
        assert_eq!(AttentionType::MHA.kv_expansion_factor(32), 1);
        assert_eq!(AttentionType::MQA.kv_expansion_factor(32), 32);
        assert_eq!(
            AttentionType::GQA { num_groups: 4 }.kv_expansion_factor(32),
            8
        );
    }
}
