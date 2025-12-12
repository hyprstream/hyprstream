//! Self-supervised reinforcement learning trainer
//!
//! Uses generation quality metrics as reward signals for training without
//! explicit human feedback. High-quality generations are reinforced while
//! low-quality ones are down-weighted.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  1. INFERENCE (with metrics capture)                        │
//! │     → Generate response                                     │
//! │     → Capture quality metrics (perplexity, entropy, etc.)   │
//! │     → Store (prompt, response, metrics) in replay buffer    │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │  2. REPLAY (with gradients)                                 │
//! │     → Sample from replay buffer                             │
//! │     → Compute quality-weighted loss:                        │
//! │       loss = -quality_score * log_prob(response | prompt)   │
//! │     → Backprop to LoRA / base weights                      │
//! └─────────────────────────────────────────────────────────────┘
//! ```

use crate::runtime::generation_metrics::GenerationQualityMetrics;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A single training example from a generation
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// The input prompt (tokenized)
    pub prompt_tokens: Vec<i64>,
    /// The generated response (tokenized)
    pub response_tokens: Vec<i64>,
    /// Quality metrics captured during generation
    pub quality_metrics: GenerationQualityMetrics,
    /// Timestamp when this example was created
    pub created_at: std::time::Instant,
    /// Session ID (for session-based filtering)
    pub session_id: Option<String>,
}

impl TrainingExample {
    /// Create a new training example
    pub fn new(
        prompt_tokens: Vec<i64>,
        response_tokens: Vec<i64>,
        quality_metrics: GenerationQualityMetrics,
        session_id: Option<String>,
    ) -> Self {
        Self {
            prompt_tokens,
            response_tokens,
            quality_metrics,
            created_at: std::time::Instant::now(),
            session_id,
        }
    }

    /// Get the reward signal based on quality metrics
    ///
    /// Returns a value in [0, 1] where higher = better quality
    pub fn reward(&self) -> f32 {
        self.quality_metrics.quality_score()
    }

    /// Check if this example is high quality (worth reinforcing)
    pub fn is_high_quality(&self) -> bool {
        self.reward() > 0.5 && !self.quality_metrics.is_concerning()
    }

    /// Total sequence length
    pub fn total_length(&self) -> usize {
        self.prompt_tokens.len() + self.response_tokens.len()
    }
}

/// Configuration for the replay buffer
#[derive(Debug, Clone)]
pub struct ReplayBufferConfig {
    /// Maximum number of examples to store
    pub max_size: usize,
    /// Minimum quality score to keep in buffer
    pub min_quality_threshold: f32,
    /// Maximum age of examples in seconds (0 = no limit)
    pub max_age_secs: u64,
    /// Prioritize high-quality examples in sampling
    pub prioritized_sampling: bool,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            max_size: 10_000,
            min_quality_threshold: 0.3,
            max_age_secs: 3600, // 1 hour
            prioritized_sampling: true,
        }
    }
}

/// Thread-safe replay buffer for storing training examples
pub struct ReplayBuffer {
    examples: Arc<Mutex<VecDeque<TrainingExample>>>,
    config: ReplayBufferConfig,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the given configuration
    pub fn new(config: ReplayBufferConfig) -> Self {
        Self {
            examples: Arc::new(Mutex::new(VecDeque::with_capacity(config.max_size))),
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ReplayBufferConfig::default())
    }

    /// Add an example to the buffer
    ///
    /// Automatically filters out low-quality examples below threshold
    pub async fn add(&self, example: TrainingExample) {
        // Filter out low-quality examples
        if example.reward() < self.config.min_quality_threshold {
            tracing::debug!(
                "Skipping low-quality example: reward={:.3}",
                example.reward()
            );
            return;
        }

        let mut examples = self.examples.lock().await;

        // Remove old examples if max_age is set
        if self.config.max_age_secs > 0 {
            let now = std::time::Instant::now();
            examples.retain(|ex| {
                now.duration_since(ex.created_at).as_secs() < self.config.max_age_secs
            });
        }

        // Add new example
        examples.push_back(example);

        // Evict oldest if over capacity
        while examples.len() > self.config.max_size {
            examples.pop_front();
        }
    }

    /// Sample a batch of examples for training
    ///
    /// If prioritized_sampling is enabled, higher quality examples are
    /// more likely to be sampled.
    pub async fn sample(&self, batch_size: usize) -> Vec<TrainingExample> {
        let examples = self.examples.lock().await;

        if examples.is_empty() {
            return Vec::new();
        }

        let actual_batch_size = batch_size.min(examples.len());

        if self.config.prioritized_sampling {
            // Prioritized sampling: higher quality = higher probability
            self.prioritized_sample(&examples, actual_batch_size)
        } else {
            // Uniform random sampling
            self.uniform_sample(&examples, actual_batch_size)
        }
    }

    /// Uniform random sampling
    fn uniform_sample(
        &self,
        examples: &VecDeque<TrainingExample>,
        batch_size: usize,
    ) -> Vec<TrainingExample> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut indices: Vec<usize> = (0..examples.len()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(batch_size);

        indices.iter().map(|&i| examples[i].clone()).collect()
    }

    /// Prioritized sampling based on quality scores
    fn prioritized_sample(
        &self,
        examples: &VecDeque<TrainingExample>,
        batch_size: usize,
    ) -> Vec<TrainingExample> {
        use rand::distributions::WeightedIndex;
        use rand::prelude::*;

        // Compute weights based on quality scores
        let weights: Vec<f64> = examples
            .iter()
            .map(|ex| {
                // Square the quality score to emphasize high-quality examples
                let q = ex.reward() as f64;
                (q * q).max(0.01) // Minimum weight to avoid zero
            })
            .collect();

        let dist = match WeightedIndex::new(&weights) {
            Ok(d) => d,
            Err(_) => return self.uniform_sample(examples, batch_size),
        };

        let mut rng = rand::thread_rng();
        let mut sampled = Vec::with_capacity(batch_size);
        let mut sampled_indices = std::collections::HashSet::new();

        // Sample without replacement
        while sampled.len() < batch_size && sampled_indices.len() < examples.len() {
            let idx = dist.sample(&mut rng);
            if sampled_indices.insert(idx) {
                sampled.push(examples[idx].clone());
            }
        }

        sampled
    }

    /// Get the current number of examples in the buffer
    pub async fn len(&self) -> usize {
        self.examples.lock().await.len()
    }

    /// Check if the buffer is empty
    pub async fn is_empty(&self) -> bool {
        self.examples.lock().await.is_empty()
    }

    /// Get statistics about the buffer
    pub async fn stats(&self) -> ReplayBufferStats {
        let examples = self.examples.lock().await;

        if examples.is_empty() {
            return ReplayBufferStats::default();
        }

        let qualities: Vec<f32> = examples.iter().map(|ex| ex.reward()).collect();
        let sum: f32 = qualities.iter().sum();
        let mean = sum / qualities.len() as f32;

        let variance: f32 = qualities.iter().map(|q| (q - mean).powi(2)).sum::<f32>()
            / qualities.len() as f32;

        let high_quality_count = examples.iter().filter(|ex| ex.is_high_quality()).count();

        ReplayBufferStats {
            count: examples.len(),
            mean_quality: mean,
            std_quality: variance.sqrt(),
            min_quality: qualities.iter().cloned().fold(f32::INFINITY, f32::min),
            max_quality: qualities.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            high_quality_ratio: high_quality_count as f32 / examples.len() as f32,
        }
    }

    /// Clear all examples
    pub async fn clear(&self) {
        self.examples.lock().await.clear();
    }

    /// Get all high-quality examples (for targeted training)
    pub async fn get_high_quality(&self) -> Vec<TrainingExample> {
        let examples = self.examples.lock().await;
        examples
            .iter()
            .filter(|ex| ex.is_high_quality())
            .cloned()
            .collect()
    }
}

/// Statistics about the replay buffer
#[derive(Debug, Clone, Default)]
pub struct ReplayBufferStats {
    /// Number of examples in buffer
    pub count: usize,
    /// Mean quality score
    pub mean_quality: f32,
    /// Standard deviation of quality scores
    pub std_quality: f32,
    /// Minimum quality score
    pub min_quality: f32,
    /// Maximum quality score
    pub max_quality: f32,
    /// Ratio of high-quality examples
    pub high_quality_ratio: f32,
}

/// Configuration for the self-supervised trainer
#[derive(Debug, Clone)]
pub struct SelfSupervisedConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Minimum examples in buffer before training
    pub min_buffer_size: usize,
    /// Training steps per training cycle
    pub steps_per_cycle: usize,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Maximum gradient norm for clipping
    pub max_grad_norm: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: f64,
    /// Enable training on base model weights (vs LoRA only)
    pub train_base_model: bool,
    /// Quality score exponent for reward shaping (higher = more emphasis on high quality)
    pub quality_exponent: f32,
    /// Baseline for advantage computation (moving average of rewards)
    pub use_baseline: bool,
}

impl Default for SelfSupervisedConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-5,
            batch_size: 4,
            min_buffer_size: 100,
            steps_per_cycle: 10,
            gradient_accumulation_steps: 4,
            max_grad_norm: 1.0,
            weight_decay: 0.01,
            train_base_model: false, // Default to LoRA only
            quality_exponent: 2.0,
            use_baseline: true,
        }
    }
}

/// Self-supervised RL trainer using quality metrics as reward
pub struct SelfSupervisedTrainer {
    /// Replay buffer for storing training examples
    pub replay_buffer: ReplayBuffer,
    /// Training configuration
    config: SelfSupervisedConfig,
    /// Running average of rewards (for baseline)
    reward_baseline: Arc<Mutex<f32>>,
    /// Total training steps completed
    steps_completed: Arc<Mutex<usize>>,
}

impl SelfSupervisedTrainer {
    /// Create a new self-supervised trainer
    pub fn new(config: SelfSupervisedConfig, buffer_config: ReplayBufferConfig) -> Self {
        Self {
            replay_buffer: ReplayBuffer::new(buffer_config),
            config,
            reward_baseline: Arc::new(Mutex::new(0.5)), // Start at neutral
            steps_completed: Arc::new(Mutex::new(0)),
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(SelfSupervisedConfig::default(), ReplayBufferConfig::default())
    }

    /// Add a training example from a completed generation
    pub async fn add_example(
        &self,
        prompt_tokens: Vec<i64>,
        response_tokens: Vec<i64>,
        quality_metrics: GenerationQualityMetrics,
        session_id: Option<String>,
    ) {
        let example = TrainingExample::new(
            prompt_tokens,
            response_tokens,
            quality_metrics,
            session_id,
        );

        // Update reward baseline with exponential moving average
        if self.config.use_baseline {
            let mut baseline = self.reward_baseline.lock().await;
            let alpha = 0.01; // Smoothing factor
            *baseline = *baseline * (1.0 - alpha) + example.reward() * alpha;
        }

        self.replay_buffer.add(example).await;
    }

    /// Check if we have enough examples to train
    pub async fn ready_to_train(&self) -> bool {
        self.replay_buffer.len().await >= self.config.min_buffer_size
    }

    /// Compute the quality-weighted loss for a batch
    ///
    /// The loss is: -advantage * log_prob(response | prompt)
    /// where advantage = (quality_score^exponent - baseline)
    ///
    /// This encourages:
    /// - High quality → positive advantage → reinforce (lower loss)
    /// - Low quality → negative advantage → suppress (higher loss)
    pub async fn compute_loss(&self, examples: &[TrainingExample]) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }

        let baseline = *self.reward_baseline.lock().await;

        // Compute advantages
        let advantages: Vec<f32> = examples
            .iter()
            .map(|ex| {
                let shaped_reward = ex.reward().powf(self.config.quality_exponent);
                if self.config.use_baseline {
                    shaped_reward - baseline.powf(self.config.quality_exponent)
                } else {
                    shaped_reward
                }
            })
            .collect();

        // The actual loss computation would multiply advantages by log_probs
        // For now, return the mean advantage (for monitoring)
        let mean_advantage: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;

        // Negative because we want to MAXIMIZE the objective (minimize loss)
        -mean_advantage
    }

    /// Run a single training step
    ///
    /// Returns the average loss for this step
    pub async fn training_step(
        &self,
        _engine: &mut crate::runtime::TorchEngine,
    ) -> Result<TrainingStepResult> {
        // Sample batch from replay buffer
        let batch = self.replay_buffer.sample(self.config.batch_size).await;

        if batch.is_empty() {
            return Ok(TrainingStepResult {
                loss: 0.0,
                batch_size: 0,
                mean_reward: 0.0,
                mean_advantage: 0.0,
            });
        }

        let baseline = *self.reward_baseline.lock().await;

        // Compute rewards and advantages
        let rewards: Vec<f32> = batch.iter().map(|ex| ex.reward()).collect();
        let advantages: Vec<f32> = rewards
            .iter()
            .map(|&r| {
                let shaped = r.powf(self.config.quality_exponent);
                if self.config.use_baseline {
                    shaped - baseline.powf(self.config.quality_exponent)
                } else {
                    shaped
                }
            })
            .collect();

        let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
        let mean_advantage = advantages.iter().sum::<f32>() / advantages.len() as f32;

        // Perform forward pass with gradients for each example
        // In a full implementation, this would:
        // 1. Concatenate prompt + response tokens
        // 2. Forward pass through model
        // 3. Extract log_probs for response tokens
        // 4. Compute loss = -advantage * sum(log_probs)
        // 5. Backward pass and optimizer step

        let mut total_loss = 0.0;

        for (example, advantage) in batch.iter().zip(advantages.iter()) {
            // Simulate loss computation
            // Real implementation would compute:
            //   log_probs = model.forward(prompt + response).log_softmax(-1)
            //   response_log_probs = log_probs[prompt_len:].gather(response_tokens)
            //   loss = -advantage * response_log_probs.mean()

            let _sequence_tokens: Vec<i64> = example
                .prompt_tokens
                .iter()
                .chain(example.response_tokens.iter())
                .copied()
                .collect();

            // Placeholder: In real implementation, compute actual loss
            // For now, we use the advantage as a proxy
            let example_loss = -advantage;
            total_loss += example_loss;

            // Log for debugging
            tracing::trace!(
                "Example: prompt_len={}, response_len={}, reward={:.3}, advantage={:.3}",
                example.prompt_tokens.len(),
                example.response_tokens.len(),
                example.reward(),
                advantage
            );
        }

        let loss = total_loss / batch.len() as f32;

        // TODO: Actually apply gradients via engine
        // This would require exposing gradient computation in TorchEngine
        // For LoRA training: engine.apply_lora_gradients(loss)?;
        // For full training: engine.apply_gradients(loss)?;

        // Increment step counter
        {
            let mut steps = self.steps_completed.lock().await;
            *steps += 1;
        }

        // Trigger actual gradient computation if engine supports it
        if self.config.train_base_model {
            // Full model training (not yet supported)
            tracing::debug!("Full model training step (placeholder)");
        } else {
            // LoRA training
            tracing::debug!(
                "LoRA training step: loss={:.4}, mean_advantage={:.4}",
                loss,
                mean_advantage
            );
            // engine.train_lora_step(loss)?; // Would need to implement
        }

        // Log the loss from quality metrics for self-supervised signal
        // In practice, we'd compute actual language model loss here
        let _quality_loss = self.compute_loss(&batch).await;

        Ok(TrainingStepResult {
            loss,
            batch_size: batch.len(),
            mean_reward,
            mean_advantage,
        })
    }

    /// Run a full training cycle
    pub async fn train_cycle(
        &self,
        engine: &mut crate::runtime::TorchEngine,
    ) -> Result<TrainingCycleResult> {
        if !self.ready_to_train().await {
            let buffer_size = self.replay_buffer.len().await;
            tracing::info!(
                "Not enough examples to train. Buffer: {}/{} required",
                buffer_size,
                self.config.min_buffer_size
            );
            return Ok(TrainingCycleResult {
                steps: 0,
                total_loss: 0.0,
                mean_reward: 0.0,
            });
        }

        let mut total_loss = 0.0;
        let mut total_reward = 0.0;

        tracing::info!(
            "Starting training cycle: {} steps",
            self.config.steps_per_cycle
        );

        for step in 0..self.config.steps_per_cycle {
            let result = self.training_step(engine).await?;

            total_loss += result.loss;
            total_reward += result.mean_reward;

            if step % 5 == 0 {
                tracing::debug!(
                    "Step {}/{}: loss={:.4}, reward={:.3}, advantage={:.3}",
                    step + 1,
                    self.config.steps_per_cycle,
                    result.loss,
                    result.mean_reward,
                    result.mean_advantage
                );
            }
        }

        let mean_loss = total_loss / self.config.steps_per_cycle as f32;
        let mean_reward = total_reward / self.config.steps_per_cycle as f32;

        tracing::info!(
            "Training cycle complete: mean_loss={:.4}, mean_reward={:.3}",
            mean_loss,
            mean_reward
        );

        Ok(TrainingCycleResult {
            steps: self.config.steps_per_cycle,
            total_loss: mean_loss,
            mean_reward,
        })
    }

    /// Get training statistics
    pub async fn stats(&self) -> TrainerStats {
        let buffer_stats = self.replay_buffer.stats().await;
        let steps = *self.steps_completed.lock().await;
        let baseline = *self.reward_baseline.lock().await;

        TrainerStats {
            steps_completed: steps,
            reward_baseline: baseline,
            buffer_stats,
        }
    }

    /// Get the configuration
    pub fn config(&self) -> &SelfSupervisedConfig {
        &self.config
    }
}

/// Result of a single training step
#[derive(Debug, Clone)]
pub struct TrainingStepResult {
    /// Loss value
    pub loss: f32,
    /// Number of examples in batch
    pub batch_size: usize,
    /// Mean reward of batch
    pub mean_reward: f32,
    /// Mean advantage of batch
    pub mean_advantage: f32,
}

/// Result of a full training cycle
#[derive(Debug, Clone)]
pub struct TrainingCycleResult {
    /// Number of steps completed
    pub steps: usize,
    /// Mean loss across all steps
    pub total_loss: f32,
    /// Mean reward across all steps
    pub mean_reward: f32,
}

/// Statistics about the trainer
#[derive(Debug, Clone)]
pub struct TrainerStats {
    /// Total training steps completed
    pub steps_completed: usize,
    /// Current reward baseline
    pub reward_baseline: f32,
    /// Replay buffer statistics
    pub buffer_stats: ReplayBufferStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quality_metrics(quality: f32) -> GenerationQualityMetrics {
        // Create metrics that will produce approximately the desired quality score
        // quality_score = confidence * consistency * non_repetition
        // confidence = 1.0 / (1.0 + perplexity / 10.0)
        // So for quality=0.7, we want confidence~0.7, consistency~1.0, repetition~0

        // Solve: 0.7 = 1.0 / (1.0 + perplexity / 10.0)
        // 1.0 + perplexity/10.0 = 1.0/0.7
        // perplexity = (1.0/0.7 - 1.0) * 10.0 = ~4.3
        let target_confidence = quality.max(0.1);
        let perplexity = ((1.0 / target_confidence) - 1.0) * 10.0;

        GenerationQualityMetrics {
            perplexity: perplexity.max(1.0),
            avg_entropy: 2.0,
            entropy_variance: 0.1, // Low variance = high consistency
            repetition_ratio: 0.0, // No repetition
            token_count: 50,
        }
    }

    #[test]
    fn test_training_example() {
        let metrics = make_quality_metrics(0.7);
        let example = TrainingExample::new(
            vec![1, 2, 3],
            vec![4, 5, 6, 7],
            metrics,
            Some("session1".to_string()),
        );

        assert_eq!(example.prompt_tokens.len(), 3);
        assert_eq!(example.response_tokens.len(), 4);
        assert_eq!(example.total_length(), 7);
        assert!(example.reward() > 0.5);
        assert!(example.is_high_quality());
    }

    #[tokio::test]
    async fn test_replay_buffer_add_and_sample() {
        let config = ReplayBufferConfig {
            max_size: 100,
            min_quality_threshold: 0.0, // Accept all
            ..Default::default()
        };
        let buffer = ReplayBuffer::new(config);

        // Add some examples
        for i in 0..10 {
            let quality = 0.3 + (i as f32 * 0.05);
            let example = TrainingExample::new(
                vec![i as i64],
                vec![i as i64 + 1],
                make_quality_metrics(quality),
                None,
            );
            buffer.add(example).await;
        }

        assert_eq!(buffer.len().await, 10);

        // Sample
        let batch = buffer.sample(5).await;
        assert_eq!(batch.len(), 5);
    }

    #[tokio::test]
    async fn test_replay_buffer_quality_filtering() {
        let config = ReplayBufferConfig {
            max_size: 100,
            min_quality_threshold: 0.5, // Only accept quality > 0.5
            ..Default::default()
        };
        let buffer = ReplayBuffer::new(config);

        // Add low quality example (should be filtered)
        let low_quality = TrainingExample::new(
            vec![1],
            vec![2],
            make_quality_metrics(0.3),
            None,
        );
        buffer.add(low_quality).await;

        // Add high quality example (should be kept)
        let high_quality = TrainingExample::new(
            vec![3],
            vec![4],
            make_quality_metrics(0.7),
            None,
        );
        buffer.add(high_quality).await;

        // Only high quality should be in buffer
        assert_eq!(buffer.len().await, 1);
    }

    #[tokio::test]
    async fn test_trainer_ready_to_train() {
        let config = SelfSupervisedConfig {
            min_buffer_size: 5,
            ..Default::default()
        };
        let buffer_config = ReplayBufferConfig {
            min_quality_threshold: 0.0,
            ..Default::default()
        };
        let trainer = SelfSupervisedTrainer::new(config, buffer_config);

        // Not ready initially
        assert!(!trainer.ready_to_train().await);

        // Add examples
        for i in 0..5 {
            trainer
                .add_example(
                    vec![i as i64],
                    vec![i as i64 + 1],
                    make_quality_metrics(0.6),
                    None,
                )
                .await;
        }

        // Now ready
        assert!(trainer.ready_to_train().await);
    }

    #[tokio::test]
    async fn test_compute_loss() {
        let trainer = SelfSupervisedTrainer::with_defaults();

        // High quality examples should have negative loss (encouraging)
        let high_quality = vec![TrainingExample::new(
            vec![1],
            vec![2],
            make_quality_metrics(0.9),
            None,
        )];
        let loss = trainer.compute_loss(&high_quality).await;
        assert!(loss < 0.0, "High quality should have negative loss");

        // Low quality examples should have positive or near-zero loss
        let low_quality = vec![TrainingExample::new(
            vec![1],
            vec![2],
            make_quality_metrics(0.2),
            None,
        )];
        let loss = trainer.compute_loss(&low_quality).await;
        assert!(loss > -0.5, "Low quality should have higher loss");
    }
}
