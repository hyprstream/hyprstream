//! Neural network implementations with GPU acceleration.
//! 
//! This module provides GPU-accelerated neural network operations using burn,
//! supporting both training and inference with automatic device selection
//! and dynamic batch sizing.

use std::sync::Arc;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::prelude::Backend;
use crate::error::Result;
use super::{GpuContext, GpuConfig, GpuMetrics};

/// Neural network model configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension
    pub output_dim: usize,
    /// Activation function
    pub activation: ActivationType,
    /// Learning rate
    pub learning_rate: f32,
}

/// Available activation functions
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
}

/// Neural network model with GPU acceleration
pub struct NeuralNetwork<B: Backend> {
    /// GPU context
    context: Arc<GpuContext<B>>,
    /// Network layers
    layers: Vec<Linear<B>>,
    /// Model configuration
    config: ModelConfig,
}

impl<B: Backend> NeuralNetwork<B> {
    /// Create a new neural network with the specified configuration
    pub fn new(context: Arc<GpuContext<B>>, config: ModelConfig) -> Result<Self> {
        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;
        
        // Create hidden layers
        for &hidden_dim in &config.hidden_dims {
            layers.push(Linear::builder()
                .with_input_size(input_dim)
                .with_output_size(hidden_dim)
                .build());
            input_dim = hidden_dim;
        }
        
        // Create output layer
        layers.push(Linear::builder()
            .with_input_size(input_dim)
            .with_output_size(config.output_dim)
            .build());
        
        Ok(Self {
            context,
            layers,
            config,
        })
    }

    /// Forward pass with automatic batching
    pub async fn forward(&self, input: Tensor<B>) -> Result<Tensor<B>> {
        let mut x = input;
        
        // Process through hidden layers
        for layer in &self.layers[..self.layers.len() - 1] {
            x = layer.forward(x);
            x = match self.config.activation {
                ActivationType::ReLU => x.relu(),
                ActivationType::Tanh => x.tanh(),
                ActivationType::Sigmoid => x.sigmoid(),
            };
        }
        
        // Output layer
        Ok(self.layers.last().unwrap().forward(x))
    }

    /// Train the model on batched data
    pub async fn train<D: AsRef<[f32]>>(&mut self, inputs: &[D], targets: &[D]) -> Result<f32> {
        let batch_size = self.determine_batch_size(inputs.len())?;
        let mut total_loss = 0.0;
        
        // Process in batches
        for (batch_inputs, batch_targets) in inputs.chunks(batch_size).zip(targets.chunks(batch_size)) {
            // Convert to tensors
            let x = self.context.tensor(batch_inputs.iter().flat_map(|x| x.as_ref().to_vec()).collect());
            let y = self.context.tensor(batch_targets.iter().flat_map(|x| x.as_ref().to_vec()).collect());
            
            // Forward pass
            let output = self.forward(x).await?;
            
            // Compute loss
            let loss = output.mse_loss(&y);
            total_loss += loss.to_scalar();
            
            // Backward pass and optimization
            let gradients = loss.backward();
            self.apply_gradients(&gradients)?;
        }
        
        Ok(total_loss / inputs.len() as f32)
    }

    /// Determine optimal batch size based on available memory
    async fn determine_batch_size(&self, dataset_size: usize) -> Result<usize> {
        let total_params: usize = self.layers.iter()
            .map(|layer| layer.parameters().weights().size().iter().product())
            .sum();
            
        let elem_size = std::mem::size_of::<f32>();
        let memory_per_item = total_params * elem_size;
        
        let metrics = self.context.get_metrics().await;
        let available_memory = metrics.available_memory;
        
        // Reserve 20% memory for gradients and temporary buffers
        let usable_memory = (available_memory as f32 * 0.8) as usize;
        let max_items = usable_memory / memory_per_item;
        
        Ok(max_items.min(dataset_size))
    }

    /// Apply gradients using configured optimizer
    fn apply_gradients(&mut self, gradients: &Tensor<B>) -> Result<()> {
        for layer in &mut self.layers {
            let mut module = layer.module_mut();
            let weights = module.weight_mut();
            let grads = gradients.narrow(0, weights.shape()[0], weights.shape()[1]);
            *weights -= &grads.mul_scalar(-self.config.learning_rate);
        }
        Ok(())
    }

    /// Get current GPU metrics
    pub async fn metrics(&self) -> GpuMetrics {
        self.context.get_metrics().await
    }

    /// Run performance benchmark
    pub async fn benchmark(&self, input_size: usize) -> Result<BenchmarkResults> {
        let mut results = BenchmarkResults::default();
        
        // Test different batch sizes
        for batch_size in [1, 8, 32, 128] {
            let input = self.context.tensor(vec![1.0f32; batch_size * input_size]);
            
            let start = std::time::Instant::now();
            self.forward(input).await?;
            let duration = start.elapsed();
            
            results.forward_times.push((batch_size, duration));
        }
        
        Ok(results)
    }
}

/// Neural network benchmark results
#[derive(Debug, Default)]
pub struct BenchmarkResults {
    /// Forward pass timings (batch_size, duration)
    pub forward_times: Vec<(usize, std::time::Duration)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[tokio::test]
    async fn test_neural_network() {
        let gpu_config = GpuConfig {
            backend: super::super::GpuBackend::Wgpu,
            max_batch_size: 128,
            memory_limit: None,
            enable_tensor_cores: false,
        };
        
        let context = Arc::new(GpuContext::<Wgpu>::new(gpu_config).unwrap());
        
        let model_config = ModelConfig {
            input_dim: 10,
            hidden_dims: vec![20, 20],
            output_dim: 1,
            activation: ActivationType::ReLU,
            learning_rate: 0.01,
        };
        
        let mut model = NeuralNetwork::new(context, model_config).unwrap();
        
        // Test forward pass
        let input = vec![1.0f32; 10];
        let x = model.context.tensor(input);
        let output = model.forward(x).await.unwrap();
        assert_eq!(output.size(), &[1]);
        
        // Test training
        let inputs: Vec<Vec<f32>> = vec![vec![1.0; 10]; 100];
        let targets: Vec<Vec<f32>> = vec![vec![1.0]; 100];
        let loss = model.train(&inputs, &targets).await.unwrap();
        assert!(loss >= 0.0);
        
        // Test benchmarking
        let results = model.benchmark(10).await.unwrap();
        assert!(!results.forward_times.is_empty());
    }
}