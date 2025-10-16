//! Architecture detection from model files and metadata

use super::ModelArchitecture;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

/// Detects model architecture from various sources
pub struct ArchitectureDetector;

impl ArchitectureDetector {
    /// Detect architecture from SafeTensors path (looks for config.json)
    pub fn detect_from_safetensors(path: &Path) -> Result<ModelArchitecture> {
        let config_path = path
            .parent()
            .ok_or_else(|| anyhow!("Invalid model path"))?
            .join("config.json");

        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&config_str)?;

            // Check model_type field
            if let Some(model_type) = config.get("model_type").and_then(|v| v.as_str()) {
                return Ok(Self::parse_architecture_string(model_type));
            }

            // Check architectures array
            if let Some(architectures) = config.get("architectures").and_then(|v| v.as_array()) {
                if let Some(first_arch) = architectures.first().and_then(|v| v.as_str()) {
                    return Ok(Self::parse_architecture_string(first_arch));
                }
            }
        }

        // Default to Llama if we can't determine
        Ok(ModelArchitecture::Llama { version: 2 })
    }

    /// Parse architecture string into enum
    fn parse_architecture_string(arch_str: &str) -> ModelArchitecture {
        let arch_lower = arch_str.to_lowercase();

        if arch_lower.contains("llama") {
            let version = if arch_lower.contains("llama3") || arch_lower.contains("llama-3") {
                3
            } else if arch_lower.contains("llama2") || arch_lower.contains("llama-2") {
                2
            } else {
                1
            };
            ModelArchitecture::Llama { version }
        } else if arch_lower.contains("gemma") {
            ModelArchitecture::Gemma
        } else if arch_lower.contains("qwen") {
            let version = if arch_lower.contains("qwen3") || arch_lower.contains("qwen-3") {
                3
            } else if arch_lower.contains("qwen2") || arch_lower.contains("qwen-2") {
                2
            } else {
                1
            };

            // Detect MoE variant
            let is_moe = arch_lower.contains("moe") || 
                         arch_lower.contains("a3b") ||  // 30B-A3B
                         arch_lower.contains("a22b"); // 235B-A22B

            // Detect context length
            let context_length = if arch_lower.contains("262k") || arch_lower.contains("262144") {
                262144
            } else if arch_lower.contains("128k") || arch_lower.contains("128000") {
                128000
            } else if arch_lower.contains("32k") || arch_lower.contains("32000") {
                32000
            } else if version == 3 {
                // Default context for Qwen3 based on model size
                if is_moe {
                    262144 // MoE models have longer context
                } else {
                    128000 // Larger dense models default to 128K
                }
            } else {
                4096 // Legacy default
            };

            ModelArchitecture::Qwen {
                version,
                is_moe,
                context_length,
            }
        } else if arch_lower.contains("phi") {
            let version = if arch_lower.contains("phi3") || arch_lower.contains("phi-3") {
                3
            } else if arch_lower.contains("phi2") || arch_lower.contains("phi-2") {
                2
            } else {
                1
            };
            ModelArchitecture::Phi { version }
        } else if arch_lower.contains("mistral") {
            ModelArchitecture::Mistral
        } else if arch_lower.contains("starcoder") || arch_lower.contains("star-coder") {
            ModelArchitecture::Starcoder
        } else if arch_lower.contains("falcon") {
            ModelArchitecture::Falcon
        } else if arch_lower.contains("gpt-neox") || arch_lower.contains("gptneox") {
            ModelArchitecture::GPTNeoX
        } else if arch_lower.contains("gpt-oss") || arch_lower.contains("gptoss") {
            // Detect GPT-OSS variant
            let (total_params_b, active_params_b) = if arch_lower.contains("120b") {
                (120, 5.1)
            } else if arch_lower.contains("20b") {
                (20, 3.6)
            } else {
                (120, 5.1) // Default to larger model
            };

            // GPT-OSS uses 128 experts with 8 active per token
            ModelArchitecture::GPTOSS {
                total_params_b,
                active_params_b,
                num_experts: 128,
                experts_per_token: 8,
            }
        } else if arch_lower.contains("gpt-j") || arch_lower.contains("gptj") {
            ModelArchitecture::GPTJ
        } else {
            ModelArchitecture::Custom(arch_str.to_string())
        }
    }

    /// Get expected tensor shapes for an architecture
    pub fn get_expected_shapes(
        arch: &ModelArchitecture,
        batch_size: usize,
        seq_len: usize,
    ) -> HashMap<String, Vec<usize>> {
        let mut shapes = HashMap::new();

        match arch {
            ModelArchitecture::Gemma => {
                // Gemma uses 4 KV heads with 256 head dim
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 16, 256]);
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 4, 256]);
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 4, 256]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 3072]);
            }
            ModelArchitecture::Qwen {
                version: 3,
                is_moe: false,
                ..
            } => {
                // Qwen3 dense models use GQA
                // Sizes: 0.6B/1.7B/4B have 32K context, 8B/14B/32B have 128K
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 32, 128]);
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 8, 128]); // GQA with 8 KV heads
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 8, 128]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 4096]);
            }
            ModelArchitecture::Qwen {
                version: 3,
                is_moe: true,
                ..
            } => {
                // Qwen3 MoE models (30B-A3B, 235B-A22B)
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 32, 128]);
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 8, 128]);
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 8, 128]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 4096]);
                shapes.insert("moe_gate".to_string(), vec![batch_size, seq_len, 128]);
                // 128 experts
            }
            ModelArchitecture::GPTOSS { .. } => {
                // GPT-OSS uses MoE with GQA
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 48, 128]); // Estimated
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 8, 128]); // GQA
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 8, 128]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 6144]); // Estimated
                shapes.insert("moe_gate".to_string(), vec![batch_size, seq_len, 128]); // 128 experts
                shapes.insert(
                    "expert_ffn".to_string(),
                    vec![8, batch_size, seq_len, 16384],
                ); // 8 active experts
            }
            ModelArchitecture::GPTJ => {
                // GPT-J 6B standard shapes
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 16, 256]);
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 16, 256]);
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 16, 256]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 4096]);
            }
            ModelArchitecture::Llama { version: 3 } => {
                // Llama 3 8B uses GQA with 8 KV heads
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 32, 128]);
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 8, 128]);
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 8, 128]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 4096]);
            }
            ModelArchitecture::Llama { .. } => {
                // Llama 1/2 standard shapes
                shapes.insert("q_proj".to_string(), vec![batch_size, seq_len, 32, 128]);
                shapes.insert("k_proj".to_string(), vec![batch_size, seq_len, 32, 128]);
                shapes.insert("v_proj".to_string(), vec![batch_size, seq_len, 32, 128]);
                shapes.insert("hidden".to_string(), vec![batch_size, seq_len, 4096]);
            }
            _ => {
                // Generic transformer shapes
                shapes.insert("attention".to_string(), vec![batch_size, seq_len, 4096]);
            }
        }

        shapes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_architecture_strings() {
        assert_eq!(
            ArchitectureDetector::parse_architecture_string("llama"),
            ModelArchitecture::Llama { version: 1 }
        );
        assert_eq!(
            ArchitectureDetector::parse_architecture_string("llama2"),
            ModelArchitecture::Llama { version: 2 }
        );
        assert_eq!(
            ArchitectureDetector::parse_architecture_string("llama-3-8b"),
            ModelArchitecture::Llama { version: 3 }
        );
        assert_eq!(
            ArchitectureDetector::parse_architecture_string("gemma"),
            ModelArchitecture::Gemma
        );

        // Test Qwen3 detection
        let qwen3 = ArchitectureDetector::parse_architecture_string("qwen3-8b");
        match qwen3 {
            ModelArchitecture::Qwen {
                version,
                is_moe,
                context_length,
            } => {
                assert_eq!(version, 3);
                assert_eq!(is_moe, false);
                assert_eq!(context_length, 128000);
            }
            _ => panic!("Expected Qwen3"),
        }

        // Test Qwen3 MoE detection
        let qwen3_moe = ArchitectureDetector::parse_architecture_string("qwen3-30b-a3b");
        match qwen3_moe {
            ModelArchitecture::Qwen {
                version,
                is_moe,
                context_length,
            } => {
                assert_eq!(version, 3);
                assert_eq!(is_moe, true);
                assert_eq!(context_length, 262144);
            }
            _ => panic!("Expected Qwen3 MoE"),
        }

        // Test GPT-OSS detection
        let gpt_oss = ArchitectureDetector::parse_architecture_string("gpt-oss-120b");
        match gpt_oss {
            ModelArchitecture::GPTOSS {
                total_params_b,
                active_params_b,
                ..
            } => {
                assert_eq!(total_params_b, 120);
                assert_eq!(active_params_b, 5.1);
            }
            _ => panic!("Expected GPT-OSS"),
        }

        assert_eq!(
            ArchitectureDetector::parse_architecture_string("gpt-j"),
            ModelArchitecture::GPTJ
        );

        assert_eq!(
            ArchitectureDetector::parse_architecture_string("custom-model"),
            ModelArchitecture::Custom("custom-model".to_string())
        );
    }

    #[test]
    fn test_gemma_shapes() {
        let shapes = ArchitectureDetector::get_expected_shapes(&ModelArchitecture::Gemma, 1, 21);

        assert_eq!(shapes.get("k_proj"), Some(&vec![1, 21, 4, 256]));
        assert_eq!(shapes.get("q_proj"), Some(&vec![1, 21, 16, 256]));
    }

    #[test]
    fn test_qwen3_shapes() {
        // Test Qwen3 dense model shapes
        let shapes = ArchitectureDetector::get_expected_shapes(
            &ModelArchitecture::Qwen {
                version: 3,
                is_moe: false,
                context_length: 128000,
            },
            1,
            100,
        );

        assert_eq!(shapes.get("k_proj"), Some(&vec![1, 100, 8, 128])); // GQA
        assert_eq!(shapes.get("q_proj"), Some(&vec![1, 100, 32, 128]));

        // Test Qwen3 MoE model shapes
        let moe_shapes = ArchitectureDetector::get_expected_shapes(
            &ModelArchitecture::Qwen {
                version: 3,
                is_moe: true,
                context_length: 262144,
            },
            1,
            100,
        );

        assert_eq!(moe_shapes.get("moe_gate"), Some(&vec![1, 100, 128])); // 128 experts
    }

    #[test]
    fn test_gpt_oss_shapes() {
        let shapes = ArchitectureDetector::get_expected_shapes(
            &ModelArchitecture::GPTOSS {
                total_params_b: 120,
                active_params_b: 5.1,
                num_experts: 128,
                experts_per_token: 8,
            },
            2,
            50,
        );

        assert_eq!(shapes.get("moe_gate"), Some(&vec![2, 50, 128]));
        assert_eq!(shapes.get("expert_ffn"), Some(&vec![8, 2, 50, 16384])); // 8 active experts
        assert_eq!(shapes.get("k_proj"), Some(&vec![2, 50, 8, 128])); // GQA
    }
}
