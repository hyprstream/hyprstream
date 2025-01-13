//! GPU benchmarking and performance measurement utilities.
//! 
//! This module provides tools for measuring GPU performance metrics including:
//! - Tensor operation throughput
//! - Memory bandwidth
//! - Device utilization
//! - Operation latency
//! - Memory usage patterns

use std::time::{Duration, Instant};
use burn::tensor::backend::Backend;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::error::Result;
use super::GpuContext;

/// Performance benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Tensor sizes to test
    pub sizes: Vec<usize>,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of measurement iterations
    pub measurement_iterations: usize,
    /// Enable detailed metrics collection
    pub detailed_metrics: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            sizes: vec![1024, 2048, 4096, 8192],
            warmup_iterations: 5,
            measurement_iterations: 10,
            detailed_metrics: true,
        }
    }
}

/// Benchmark measurement results
#[derive(Debug, Default)]
pub struct BenchResults {
    /// Operation timings by size
    pub timings: Vec<(usize, Duration)>,
    /// Memory throughput in GB/s
    pub memory_throughput: f64,
    /// Operations per second
    pub ops_per_second: f64,
    /// Peak GPU utilization
    pub peak_utilization: f32,
    /// Peak memory usage
    pub peak_memory: usize,
    /// Detailed metrics if enabled
    pub detailed: Option<DetailedMetrics>,
}

/// Detailed performance metrics
#[derive(Debug, Default)]
pub struct DetailedMetrics {
    /// Memory allocation patterns
    pub allocation_sizes: Vec<usize>,
    /// Kernel execution times
    pub kernel_times: Vec<Duration>,
    /// Memory transfer times
    pub transfer_times: Vec<Duration>,
    /// Cache hit rates
    pub cache_hit_rates: Vec<f32>,
}

/// GPU benchmark runner
pub struct BenchRunner<B: Backend> {
    /// GPU context
    context: Arc<GpuContext<B>>,
    /// Benchmark configuration
    config: BenchConfig,
    /// Current results
    results: Arc<RwLock<BenchResults>>,
}

impl<B: Backend> BenchRunner<B> {
    /// Create a new benchmark runner
    pub fn new(context: Arc<GpuContext<B>>, config: BenchConfig) -> Self {
        Self {
            context,
            config,
            results: Arc::new(RwLock::new(BenchResults::default())),
        }
    }

    /// Run matrix multiplication benchmark
    pub async fn bench_matmul(&self) -> Result<BenchResults> {
        let mut results = BenchResults::default();
        
        for &size in &self.config.sizes {
            // Create test matrices
            let a = self.context.tensor(vec![1.0f32; size * size]);
            let b = self.context.tensor(vec![1.0f32; size * size]);
            
            // Warmup runs
            for _ in 0..self.config.warmup_iterations {
                self.context.matmul(a.clone(), b.clone()).await?;
            }
            
            // Measurement runs
            let mut durations = Vec::new();
            let mut peak_util = 0.0f32;
            let mut peak_mem = 0;
            
            for _ in 0..self.config.measurement_iterations {
                let start = Instant::now();
                self.context.matmul(a.clone(), b.clone()).await?;
                durations.push(start.elapsed());
                
                // Update metrics
                let metrics = self.context.get_metrics().await;
                peak_util = peak_util.max(metrics.utilization);
                peak_mem = peak_mem.max(metrics.total_memory - metrics.available_memory);
            }
            
            // Calculate statistics
            let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            results.timings.push((size, avg_duration));
            results.peak_utilization = results.peak_utilization.max(peak_util);
            results.peak_memory = results.peak_memory.max(peak_mem);
            
            // Calculate throughput
            let elements = size * size * size;
            let bytes = elements * std::mem::size_of::<f32>();
            let seconds = avg_duration.as_secs_f64();
            results.memory_throughput = bytes as f64 / seconds / 1e9;
            results.ops_per_second = elements as f64 / seconds;
        }
        
        // Collect detailed metrics if enabled
        if self.config.detailed_metrics {
            results.detailed = Some(self.collect_detailed_metrics().await?);
        }
        
        Ok(results)
    }

    /// Collect detailed performance metrics
    async fn collect_detailed_metrics(&self) -> Result<DetailedMetrics> {
        let mut metrics = DetailedMetrics::default();
        
        for &size in &self.config.sizes {
            // Track memory allocations
            let before_mem = self.context.get_metrics().await.available_memory;
            let tensor = self.context.tensor(vec![1.0f32; size * size]);
            let after_mem = self.context.get_metrics().await.available_memory;
            metrics.allocation_sizes.push(before_mem - after_mem);
            
            // Measure transfer times
            let start = Instant::now();
            let _gpu_tensor = tensor.clone();
            metrics.transfer_times.push(start.elapsed());
            
            // Measure kernel times
            let start = Instant::now();
            self.context.matmul(tensor.clone(), tensor).await?;
            metrics.kernel_times.push(start.elapsed());
        }
        
        Ok(metrics)
    }

    /// Run all benchmarks
    pub async fn run_all(&self) -> Result<BenchResults> {
        // Matrix multiplication benchmark
        let matmul_results = self.bench_matmul().await?;
        
        // Update stored results
        let mut results = self.results.write().await;
        *results = matmul_results.clone();
        
        Ok(matmul_results)
    }

    /// Get current benchmark results
    pub async fn get_results(&self) -> BenchResults {
        self.results.read().await.clone()
    }

    /// Format results as a human-readable string
    pub fn format_results(results: &BenchResults) -> String {
        let mut output = String::new();
        
        output.push_str("GPU Benchmark Results:\n");
        output.push_str("=====================\n\n");
        
        // Matrix multiplication results
        output.push_str("Matrix Multiplication:\n");
        for (size, duration) in &results.timings {
            output.push_str(&format!(
                "{}x{}: {:.2}ms\n",
                size, size,
                duration.as_secs_f64() * 1000.0
            ));
        }
        
        output.push_str(&format!(
            "\nPeak Performance:\n\
             Memory Throughput: {:.2} GB/s\n\
             Operations/Second: {:.2e}\n\
             GPU Utilization: {:.1}%\n\
             Peak Memory: {:.2} GB\n",
            results.memory_throughput,
            results.ops_per_second,
            results.peak_utilization,
            results.peak_memory as f64 / 1e9
        ));
        
        // Detailed metrics if available
        if let Some(detailed) = &results.detailed {
            output.push_str("\nDetailed Metrics:\n");
            output.push_str("----------------\n");
            
            for (i, (alloc, kernel, transfer)) in detailed.allocation_sizes.iter()
                .zip(&detailed.kernel_times)
                .zip(&detailed.transfer_times)
                .enumerate()
            {
                output.push_str(&format!(
                    "Size {}: \n\
                     - Allocation: {:.2} MB\n\
                     - Kernel Time: {:.2}ms\n\
                     - Transfer Time: {:.2}ms\n",
                    results.timings[i].0,
                    *alloc as f64 / 1e6,
                    kernel.as_secs_f64() * 1000.0,
                    transfer.as_secs_f64() * 1000.0
                ));
            }
        }
        
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[tokio::test]
    async fn test_benchmarking() {
        let gpu_config = GpuConfig {
            backend: super::super::GpuBackend::Wgpu,
            max_batch_size: 1024,
            memory_limit: None,
            enable_tensor_cores: false,
        };
        
        let context = Arc::new(GpuContext::<Wgpu>::new(gpu_config).unwrap());
        
        let bench_config = BenchConfig {
            sizes: vec![128, 256],
            warmup_iterations: 2,
            measurement_iterations: 3,
            detailed_metrics: true,
        };
        
        let runner = BenchRunner::new(context, bench_config);
        let results = runner.run_all().await.unwrap();
        
        assert_eq!(results.timings.len(), 2);
        assert!(results.memory_throughput > 0.0);
        assert!(results.ops_per_second > 0.0);
        assert!(results.peak_utilization >= 0.0 && results.peak_utilization <= 100.0);
        assert!(results.peak_memory > 0);
        
        // Verify detailed metrics
        assert!(results.detailed.is_some());
        let detailed = results.detailed.unwrap();
        assert_eq!(detailed.allocation_sizes.len(), 2);
        assert_eq!(detailed.kernel_times.len(), 2);
        assert_eq!(detailed.transfer_times.len(), 2);
        
        // Test results formatting
        let formatted = BenchRunner::<Wgpu>::format_results(&results);
        assert!(formatted.contains("GPU Benchmark Results"));
        assert!(formatted.contains("Matrix Multiplication"));
        assert!(formatted.contains("Detailed Metrics"));
    }
}