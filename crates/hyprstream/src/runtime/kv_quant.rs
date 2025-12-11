//! KV cache quantization types
//!
//! Defines quantization types for reducing GPU memory usage in KV cache.

use serde::{Deserialize, Serialize};

/// KV cache quantization type for inference
///
/// Quantization reduces GPU memory usage at a slight quality cost:
/// - `None`: Full precision (default)
/// - `Int8`: 50% memory savings, minimal quality loss
/// - `Nf4`: 75% memory savings, best quality for 4-bit (NormalFloat)
/// - `Fp4`: 75% memory savings, standard 4-bit quantization
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KVQuantType {
    /// No quantization (full precision FP16/BF16)
    #[default]
    None,
    /// 8-bit integer quantization (~50% memory savings)
    Int8,
    /// 4-bit NormalFloat quantization (~75% memory savings)
    Nf4,
    /// 4-bit FloatingPoint quantization (~75% memory savings)
    Fp4,
}
