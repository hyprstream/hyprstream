//! Training data loading and processing for LoRA fine-tuning

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A single training sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input text (prompt)
    pub input: String,
    /// Expected output (completion)
    pub output: String,
    /// Optional system message
    pub system: Option<String>,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Training dataset
#[derive(Clone)]
pub struct TrainingDataset {
    pub samples: Vec<TrainingSample>,
}

impl TrainingDataset {
    /// Load training data from JSONL file
    pub fn from_jsonl(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read training data from {:?}", path))?;

        let mut samples = Vec::new();
        for (line_num, line) in content.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let sample: TrainingSample = serde_json::from_str(line)
                .with_context(|| format!("Failed to parse line {} in {:?}", line_num + 1, path))?;

            samples.push(sample);
        }

        Ok(Self { samples })
    }

    /// Create dataset from conversation format
    /// Expects format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    pub fn from_conversations(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read conversation data from {:?}", path))?;

        let mut samples = Vec::new();

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let messages: Vec<serde_json::Value> = serde_json::from_str(line)?;

            // Convert conversation to training samples
            let mut current_input = String::new();

            for message in messages {
                let role = message["role"].as_str().unwrap_or("");
                let content = message["content"].as_str().unwrap_or("");

                match role {
                    "user" | "human" => {
                        if !current_input.is_empty() {
                            current_input.push('\n');
                        }
                        current_input.push_str(content);
                    }
                    "assistant" | "ai" => {
                        if !current_input.is_empty() {
                            samples.push(TrainingSample {
                                input: current_input.clone(),
                                output: content.to_string(),
                                system: None,
                                metadata: None,
                            });
                        }
                        current_input.clear();
                    }
                    _ => {} // ignore system messages for now
                }
            }
        }

        Ok(Self { samples })
    }

    /// Get number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Check if dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Get sample by index
    pub fn get(&self, index: usize) -> Option<&TrainingSample> {
        self.samples.get(index)
    }

    /// Iterator over samples
    pub fn iter(&self) -> impl Iterator<Item = &TrainingSample> {
        self.samples.iter()
    }

    /// Create batches of specified size
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = &[TrainingSample]> {
        self.samples.chunks(batch_size)
    }

    /// Split dataset into train/validation
    pub fn train_test_split(&self, train_ratio: f64) -> (TrainingDataset, TrainingDataset) {
        let split_point = ((self.samples.len() as f64) * train_ratio) as usize;

        let train_samples = self.samples[..split_point].to_vec();
        let test_samples = self.samples[split_point..].to_vec();

        (
            TrainingDataset {
                samples: train_samples,
            },
            TrainingDataset {
                samples: test_samples,
            },
        )
    }
}

/// Training data loader with chat template formatting
pub struct ChatTemplateDataLoader {
    dataset: TrainingDataset,
    template_engine: Option<crate::runtime::template_engine::TemplateEngine>,
}

impl ChatTemplateDataLoader {
    /// Create new data loader
    pub fn new(dataset: TrainingDataset) -> Self {
        Self {
            dataset,
            template_engine: None,
        }
    }

    /// Set chat template engine
    pub fn with_template(
        mut self,
        template_engine: crate::runtime::template_engine::TemplateEngine,
    ) -> Self {
        self.template_engine = Some(template_engine);
        self
    }

    /// Format a sample using chat template
    pub fn format_sample(&self, sample: &TrainingSample) -> Result<(String, String)> {
        use crate::runtime::template_engine::ChatMessage;

        if let Some(ref template) = self.template_engine {
            // Create chat messages
            let mut messages = Vec::new();

            if let Some(ref system) = sample.system {
                messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: system.clone(),
                });
            }

            messages.push(ChatMessage {
                role: "user".to_string(),
                content: sample.input.clone(),
            });

            // Format input (without assistant response)
            let formatted_input = template.apply_chat_template(&messages, Some(true))?;

            // Add assistant response for target
            messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: sample.output.clone(),
            });

            let full_formatted = template.apply_chat_template(&messages, Some(false))?;

            Ok((formatted_input, full_formatted))
        } else {
            // Simple concatenation without template
            let input = format!("Human: {}\n\nAssistant: ", sample.input);
            let target = format!("{}{}", input, sample.output);
            Ok((input, target))
        }
    }

    /// Get formatted batches for training
    pub fn formatted_batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = Result<Vec<(String, String)>>> + '_ {
        self.dataset.batches(batch_size).map(|batch| {
            batch
                .iter()
                .map(|sample| self.format_sample(sample))
                .collect::<Result<Vec<_>>>()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_jsonl_loading() {
        let temp_dir = tempdir().expect("test: create temp dir");
        let data_path = temp_dir.path().join("train.jsonl");

        let content = r#"{"input": "What is 2+2?", "output": "4"}
{"input": "Hello", "output": "Hi there!"}
"#;
        std::fs::write(&data_path, content).expect("test: write training data");

        let dataset = TrainingDataset::from_jsonl(&data_path).expect("test: load dataset");
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0).expect("test: get sample").input, "What is 2+2?");
        assert_eq!(dataset.get(0).expect("test: get sample").output, "4");
    }

    #[test]
    fn test_train_test_split() {
        let samples = vec![
            TrainingSample {
                input: "1".to_string(),
                output: "a".to_string(),
                system: None,
                metadata: None,
            },
            TrainingSample {
                input: "2".to_string(),
                output: "b".to_string(),
                system: None,
                metadata: None,
            },
            TrainingSample {
                input: "3".to_string(),
                output: "c".to_string(),
                system: None,
                metadata: None,
            },
            TrainingSample {
                input: "4".to_string(),
                output: "d".to_string(),
                system: None,
                metadata: None,
            },
        ];

        let dataset = TrainingDataset { samples };
        let (train, test) = dataset.train_test_split(0.75);

        assert_eq!(train.len(), 3);
        assert_eq!(test.len(), 1);
    }
}
