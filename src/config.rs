//! Configuration management for Hyprstream service.
//!
//! This module provides configuration handling through multiple sources:
//! 1. Default configuration (embedded in binary)
//! 2. System-wide configuration file (`/etc/hyprstream/config.toml`)
//! 3. User-specified configuration file
//! 4. Environment variables (prefixed with `HYPRSTREAM_`)
//! 5. Command-line arguments
//!
//! Configuration options are loaded in order of precedence, with later sources
//! overriding earlier ones.
//!
//! # Environment Variables
//!
//! Backend-specific credentials should be provided via environment variables:
//! - `HYPRSTREAM_ENGINE_USERNAME` - Primary storage backend username
//! - `HYPRSTREAM_ENGINE_PASSWORD` - Primary storage backend password
//! - `HYPRSTREAM_CACHE_USERNAME` - Cache backend username (if needed)
//! - `HYPRSTREAM_CACHE_PASSWORD` - Cache backend password (if needed)
//! - `HYPRSTREAM_GPU_ENABLED` - Enable/disable GPU acceleration
//! - `HYPRSTREAM_GPU_MEMORY_LIMIT` - GPU memory limit in bytes

use clap::Parser;
use config::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use crate::error::Result;
use crate::gpu::{GpuBackend, GpuConfig};

/// Command-line arguments
#[derive(Debug, Parser)]
#[clap(version, about)]
pub struct Args {
    /// Configuration file path
    #[clap(short, long)]
    pub config: Option<PathBuf>,

    /// Storage engine type
    #[clap(long)]
    pub engine: Option<String>,

    /// Storage engine connection string
    #[clap(long)]
    pub engine_connection: Option<String>,

    /// Storage engine options
    #[clap(long = "engine-options")]
    pub engine_options: Vec<String>,

    /// Enable GPU acceleration
    #[clap(long)]
    pub gpu_enabled: Option<bool>,

    /// GPU backend selection (cuda, rocm, wgpu)
    #[clap(long)]
    pub gpu_backend: Option<String>,

    /// GPU memory limit in bytes
    #[clap(long)]
    pub gpu_memory_limit: Option<usize>,
}

/// Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Storage configuration
    pub storage: StorageConfig,
    /// Metrics configuration
    pub metrics: MetricsConfig,
    /// GPU settings
    #[serde(default)]
    pub gpu: GpuSettings,
}

/// Storage backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Storage engine type
    pub engine: String,
    /// Connection string
    pub connection: String,
    /// Engine-specific options
    #[serde(default)]
    pub options: HashMap<String, String>,
}

/// Metrics collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    #[serde(default = "default_metrics_enabled")]
    pub enabled: bool,
    /// Collection interval in seconds
    #[serde(default = "default_metrics_interval")]
    pub interval: u64,
    /// Maximum metrics retention period
    #[serde(default = "default_metrics_retention")]
    pub retention: u64,
}

/// GPU acceleration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSettings {
    /// Enable GPU acceleration
    #[serde(default)]
    pub enabled: bool,
    /// GPU backend selection
    #[serde(default = "default_gpu_backend")]
    pub backend: GpuBackend,
    /// Maximum batch size for operations
    #[serde(default = "default_batch_size")]
    pub max_batch_size: usize,
    /// Memory limit in bytes (None for no limit)
    #[serde(default)]
    pub memory_limit: Option<usize>,
    /// Enable tensor cores if available
    #[serde(default)]
    pub enable_tensor_cores: bool,
}

impl Default for GpuSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: default_gpu_backend(),
            max_batch_size: default_batch_size(),
            memory_limit: None,
            enable_tensor_cores: false,
        }
    }
}

/// Backend credentials
#[derive(Debug, Clone)]
pub struct Credentials {
    pub username: String,
    pub password: String,
}

impl ServiceConfig {
    /// Load configuration from all sources
    pub fn load(args: &Args) -> Result<Self> {
        let mut builder = config::Config::builder()
            .add_source(config::File::from_str(include_str!("../config/default.toml"), config::FileFormat::Toml))
            .add_source(config::File::with_name("/etc/hyprstream/config.toml").required(false));

        // Load user config if specified
        if let Some(path) = &args.config {
            builder = builder.add_source(config::File::from(path.as_path()));
        }

        // Add environment variables
        builder = builder.add_source(config::Environment::with_prefix("HYPRSTREAM"));

        // Build config
        let mut config: ServiceConfig = builder.build()?.try_deserialize()?;

        // Override with command line args
        if let Some(engine) = &args.engine {
            config.storage.engine = engine.clone();
        }
        if let Some(connection) = &args.engine_connection {
            config.storage.connection = connection.clone();
        }
        for opt in &args.engine_options {
            if let Some((key, value)) = opt.split_once('=') {
                config.storage.options.insert(key.to_string(), value.to_string());
            }
        }
        if let Some(enabled) = args.gpu_enabled {
            config.gpu.enabled = enabled;
        }
        if let Some(backend) = &args.gpu_backend {
            config.gpu.backend = match backend.as_str() {
                "cuda" => GpuBackend::Cuda,
                "rocm" => GpuBackend::Rocm,
                "wgpu" => GpuBackend::Wgpu,
                _ => GpuBackend::Cpu,
            };
        }
        if let Some(limit) = args.gpu_memory_limit {
            config.gpu.memory_limit = Some(limit);
        }

        Ok(config)
    }

    /// Get storage backend credentials from environment
    pub fn get_credentials(&self) -> Option<Credentials> {
        let username = env::var("HYPRSTREAM_ENGINE_USERNAME").ok()?;
        let password = env::var("HYPRSTREAM_ENGINE_PASSWORD").ok()?;
        Some(Credentials { username, password })
    }

    /// Get cache backend credentials from environment
    pub fn get_cache_credentials(&self) -> Option<Credentials> {
        let username = env::var("HYPRSTREAM_CACHE_USERNAME").ok()?;
        let password = env::var("HYPRSTREAM_CACHE_PASSWORD").ok()?;
        Some(Credentials { username, password })
    }

    /// Convert GPU settings to GpuConfig
    pub fn gpu_config(&self) -> GpuConfig {
        GpuConfig {
            backend: self.gpu.backend,
            max_batch_size: self.gpu.max_batch_size,
            memory_limit: self.gpu.memory_limit,
            enable_tensor_cores: self.gpu.enable_tensor_cores,
        }
    }
}

fn default_metrics_enabled() -> bool {
    true
}

fn default_metrics_interval() -> u64 {
    60
}

fn default_metrics_retention() -> u64 {
    86400
}

fn default_gpu_backend() -> GpuBackend {
    GpuBackend::Wgpu
}

fn default_batch_size() -> usize {
    1024
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let args = Args {
            config: None,
            engine: None,
            engine_connection: None,
            engine_options: vec![],
            gpu_enabled: None,
            gpu_backend: None,
            gpu_memory_limit: None,
        };

        let config = ServiceConfig::load(&args).unwrap();
        assert!(config.metrics.enabled);
        assert_eq!(config.metrics.interval, 60);
        assert_eq!(config.metrics.retention, 86400);
        assert!(config.gpu.enabled);
        assert_eq!(config.gpu.max_batch_size, 1024);
    }

    #[test]
    fn test_gpu_config_conversion() {
        let config = ServiceConfig {
            storage: StorageConfig {
                engine: "test".into(),
                connection: "test".into(),
                options: HashMap::new(),
            },
            metrics: MetricsConfig {
                enabled: true,
                interval: 60,
                retention: 86400,
            },
            gpu: GpuSettings {
                enabled: true,
                backend: GpuBackend::Cuda,
                max_batch_size: 2048,
                memory_limit: Some(1024 * 1024 * 1024),
                enable_tensor_cores: true,
            },
        };

        let gpu_config = config.gpu_config();
        assert_eq!(gpu_config.backend, GpuBackend::Cuda);
        assert_eq!(gpu_config.max_batch_size, 2048);
        assert_eq!(gpu_config.memory_limit, Some(1024 * 1024 * 1024));
        assert!(gpu_config.enable_tensor_cores);
    }
}
