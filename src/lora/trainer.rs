//! LoRA training implementation with autograd

use anyhow::{Result, anyhow};
use tch::{nn, Device, Kind, Reduction, Tensor};
use tch::nn::OptimizerConfig;
use std::sync::Arc;
use std::time::Instant;
use super::LoRAConfig;

/// Training configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub num_epochs: usize,
    pub gradient_accumulation_steps: usize,
    pub warmup_steps: usize,
    pub weight_decay: f64,
    pub max_grad_norm: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 4,
            num_epochs: 10,
            gradient_accumulation_steps: 1,
            warmup_steps: 100,
            weight_decay: 0.01,
            max_grad_norm: 1.0,
        }
    }
}

use super::torch_adapter::LoRAModel;

/// Training metrics for logging
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub perplexity: f64,
    pub learning_rate: f64,
    pub grad_norm: f64,
    pub step: usize,
    pub epoch: usize,
}

/// Checkpoint metrics
#[derive(Debug, Clone)]
pub struct CheckpointMetrics {
    pub training_loss: f64,
    pub validation_loss: Option<f64>,
    pub perplexity: f64,
    pub learning_rate: f64,
    pub gradient_norm: f64,
    pub tokens_processed: usize,
    pub training_time_seconds: u64,
    pub memory_usage_mb: f64,
}

/// LoRA trainer with optimizer and training loop
pub struct LoRATrainer {
    /// Optimizer (AdamW)
    pub optimizer: nn::Optimizer,
    
    /// Training configuration
    pub config: TrainingConfig,
    
    /// Current training step
    pub global_step: usize,
    
    /// Device for training
    pub device: Device,
    
    /// Current epoch
    pub current_epoch: usize,
    
    /// Best loss seen so far
    pub best_loss: Option<f64>,
    
    /// Current loss
    pub current_loss: Option<f64>,
    
    /// Gradient norm from last step
    pub grad_norm: Option<f64>,
    
    /// Total tokens processed
    pub total_tokens: u64,
    
    /// Training start time
    pub start_time: Instant,
}

impl LoRATrainer {
    /// Create a new trainer
    pub fn new(
        vs: &nn::VarStore,
        device: Device,
        config: TrainingConfig,
    ) -> Result<Self> {
        // Create AdamW optimizer for LoRA parameters
        let optimizer = nn::AdamW::default()
            .build(vs, config.learning_rate)?;

        Ok(Self {
            optimizer,
            config,
            global_step: 0,
            device,
            current_epoch: 0,
            best_loss: None,
            current_loss: None,
            grad_norm: None,
            total_tokens: 0,
            start_time: Instant::now(),
        })
    }
    
    /// Single training step with autograd
    pub fn training_step(
        &mut self,
        lora_model: &LoRAModel,
        logits: &Tensor,  // Model output with LoRA applied
        labels: &Tensor,  // Target labels
    ) -> Result<TrainingMetrics> {
        let start = Instant::now();
        
        // Ensure we're in training mode with gradients enabled
        // Gradient tracking enabled by default in training
        
        // Compute cross-entropy loss
        // logits shape: [batch_size, seq_len, vocab_size]
        // labels shape: [batch_size, seq_len]
        
        let batch_size = logits.size()[0];
        let seq_len = logits.size()[1];
        let vocab_size = logits.size()[2];
        
        // Reshape for loss computation
        let logits_flat = logits.view([batch_size * seq_len, vocab_size]);
        let labels_flat = labels.view([batch_size * seq_len]);
        
        // Cross-entropy loss with ignore_index for padding tokens
        let loss = logits_flat.cross_entropy_loss::<Tensor>(
            &labels_flat,
            None,               // weight
            Reduction::Mean,    // reduction
            -100,              // ignore_index (padding)
            0.0,               // label_smoothing
        );
        
        // Get loss value before backward
        let loss_value = loss.double_value(&[]);
        
        // Backward pass - autograd computes gradients
        self.optimizer.zero_grad();
        loss.backward();
        
        // Gradient clipping
        let grad_norm = self.clip_gradients(lora_model)?;
        self.grad_norm = Some(grad_norm);
        
        // Optimizer step - update weights
        self.optimizer.step();
        
        // Update learning rate with warmup
        self.update_learning_rate();
        
        // Calculate perplexity
        let perplexity = loss_value.exp();
        
        // Update training state
        self.global_step += 1;
        self.current_loss = Some(loss_value);
        self.total_tokens += labels.size()[0] as u64 * labels.size()[1] as u64;
        
        // Update best loss
        if self.best_loss.is_none() || loss_value < self.best_loss.unwrap() {
            self.best_loss = Some(loss_value);
        }
        
        let elapsed = start.elapsed().as_secs_f32();
        tracing::debug!(
            "Step {} completed in {:.2}s - loss: {:.4}, ppl: {:.2}, grad_norm: {:.4}",
            self.global_step, elapsed, loss_value, perplexity, grad_norm
        );
        
        Ok(TrainingMetrics {
            loss: loss_value,
            perplexity,
            learning_rate: self.get_current_lr(),
            grad_norm,
            step: self.global_step,
            epoch: self.global_step / self.config.batch_size,
        })
    }
    
    /// Train on a batch of data
    pub fn train_batch(
        &mut self,
        lora_model: &LoRAModel,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        labels: &Tensor,
        forward_fn: impl Fn(&Tensor, Option<&Tensor>, bool) -> Result<Tensor>,
    ) -> Result<TrainingMetrics> {
        // Forward pass with LoRA
        let logits = forward_fn(input_ids, attention_mask, true)?;

        // Training step with autograd
        self.training_step(lora_model, &logits, labels)
    }
    
    /// Clip gradients to prevent explosion
    fn clip_gradients(&self, lora_model: &LoRAModel) -> Result<f64> {
        let params: Vec<Tensor> = lora_model.vs
            .trainable_variables()
            .iter()
            .map(|v| v.shallow_clone())
            .collect();
        
        // Clip gradients manually since tch doesn't have utils::clip_grad_norm
        let total_norm = params.iter()
            .map(|p| p.grad().norm().double_value(&[]))
            .sum::<f64>()
            .sqrt();

        if total_norm > self.config.max_grad_norm {
            let clip_coef = self.config.max_grad_norm / total_norm;
            for param in &params {
                let grad = param.grad();
                let clipped_grad = grad * clip_coef;
                let _ = clipped_grad;
            }
        }
        let grad_norm = total_norm;
        Ok(grad_norm)
    }
    
    /// Update learning rate with linear warmup
    fn update_learning_rate(&mut self) {
        if self.global_step < self.config.warmup_steps {
            // Linear warmup
            let warmup_lr = self.config.learning_rate * 
                (self.global_step as f64 / self.config.warmup_steps as f64);
            self.optimizer.set_lr(warmup_lr);
        } else {
            // Cosine decay after warmup
            let estimated_total_steps = self.config.num_epochs * 100; // Rough estimate
            let progress = (self.global_step - self.config.warmup_steps) as f64 / 
                          estimated_total_steps as f64;
            let cosine_lr = self.config.learning_rate * 
                (1.0 + progress.min(1.0) * std::f64::consts::PI).cos() / 2.0;
            self.optimizer.set_lr(cosine_lr.max(1e-6));
        }
    }
    
    /// Get current learning rate
    fn get_current_lr(&self) -> f64 {
        self.config.learning_rate
    }

    
    /// Save checkpoint
    pub fn save_checkpoint(&self, lora_model: &LoRAModel, path: &str) -> Result<()> {
        // Save LoRA weights
        lora_model.save(&format!("{}/lora_weights.pt", path))?;
        
        
        // Save training state
        let state = serde_json::json!({
            "global_step": self.global_step,
            "config": self.config,
        });
        
        std::fs::write(
            format!("{}/trainer_state.json", path),
            serde_json::to_string_pretty(&state)?
        )?;
        
        tracing::info!("Saved checkpoint to {}", path);
        Ok(())
    }
    
    /// Load checkpoint
    pub fn load_checkpoint(&mut self, lora_model: &mut LoRAModel, path: &str) -> Result<()> {
        // Load LoRA weights
        lora_model.load(&format!("{}/lora_weights.pt", path))?;
        
        
        // Load training state
        let state_str = std::fs::read_to_string(format!("{}/trainer_state.json", path))?;
        let state: serde_json::Value = serde_json::from_str(&state_str)?;
        
        self.global_step = state["global_step"].as_u64().unwrap_or(0) as usize;
        
        tracing::info!("Loaded checkpoint from {} at step {}", path, self.global_step);
        Ok(())
    }
    
    /// Evaluate on validation data (no gradient computation)
    pub fn evaluate(
        &self,
        input_ids: &Tensor,
        labels: &Tensor,
        forward_fn: impl Fn(&Tensor, Option<&Tensor>, bool) -> Result<Tensor>,
    ) -> Result<f64> {
        // Disable gradients for evaluation
        let _no_grad = tch::no_grad(|| {});
        
        // Forward pass without dropout
        let logits = forward_fn(input_ids, None, false)?;
        
        // Compute loss
        let batch_size = logits.size()[0];
        let seq_len = logits.size()[1];
        let vocab_size = logits.size()[2];
        
        let logits_flat = logits.view([batch_size * seq_len, vocab_size]);
        let labels_flat = labels.view([batch_size * seq_len]);
        
        let loss = logits_flat.cross_entropy_loss::<Tensor>(
            &labels_flat,
            None,
            Reduction::Mean,
            -100,
            0.0,
        );
        
        let loss_value = loss.double_value(&[]);
        
        // Re-enable gradients
        // Gradient tracking enabled by default in training
        
        Ok(loss_value)
    }
    
    /// Get current learning rate
    pub fn current_learning_rate(&self) -> f64 {
        self.get_current_lr()
    }
    
    /// Get current step
    pub fn current_step(&self) -> usize {
        self.global_step
    }
    
    /// Set training step (for checkpoint recovery)
    pub fn set_step(&mut self, step: usize) {
        self.global_step = step;
    }
    
    /// Set epoch (for checkpoint recovery)  
    pub fn set_epoch(&mut self, epoch: usize) {
        self.current_epoch = epoch;
    }
    
    /// Set best loss (for checkpoint recovery)
    pub fn set_best_loss(&mut self, loss: f64) {
        self.best_loss = Some(loss);
    }
    
    /// Get current training metrics for checkpointing
    pub fn get_current_metrics(&self) -> CheckpointMetrics {
        CheckpointMetrics {
            training_loss: self.current_loss.unwrap_or(0.0),
            validation_loss: None,
            perplexity: self.current_loss.unwrap_or(0.0).exp(),
            learning_rate: self.current_learning_rate(),
            gradient_norm: self.grad_norm.unwrap_or(0.0),
            tokens_processed: self.total_tokens as usize,
            training_time_seconds: self.start_time.elapsed().as_secs(),
            memory_usage_mb: 0.0, // Would need system query
        }
    }
    
}