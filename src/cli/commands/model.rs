//! Model management CLI commands

use clap::{Args, Subcommand};

/// Model management commands
#[derive(Args)]
pub struct ModelCommand {
    #[command(subcommand)]
    pub action: ModelAction,
}

/// Model management actions
#[derive(Subcommand)]
pub enum ModelAction {
    /// Pull a model from a registry
    Pull {
        /// Model URI (e.g., hf://microsoft/DialoGPT-medium)
        uri: String,
        
        /// Force re-download even if cached
        #[arg(long)]
        force: bool,
        
        /// Specific files to download
        #[arg(long, value_delimiter = ',')]
        files: Option<Vec<String>>,
        
        /// Preferred format (safetensors, gguf, pytorch)
        #[arg(long, default_value = "safetensors")]
        format: String,
        
        /// Auto-convert to SafeTensors if not available
        #[arg(long, default_value = "true")]
        auto_convert: bool,
        
        /// Show download progress
        #[arg(long, default_value = "true")]
        progress: bool,
    },
    
    /// List available models
    List {
        /// Filter by registry type (hf, ollama, custom)
        #[arg(long)]
        registry: Option<String>,
        
        /// Search query to filter models
        #[arg(long)]
        search: Option<String>,
        
        /// Include remote models (not just local cache)
        #[arg(long)]
        remote: bool,
        
        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
    },
    
    /// Get detailed information about a model
    Info {
        /// Model URI
        uri: String,
        
        /// Output format (json, yaml)
        #[arg(long, default_value = "json")]
        format: String,
    },
    
    /// Remove a model from local cache
    Remove {
        /// Model URI
        uri: String,
        
        /// Keep metadata but remove files
        #[arg(long)]
        keep_metadata: bool,
        
        /// Confirm removal without prompting
        #[arg(long)]
        yes: bool,
    },
    
    /// Search for models across registries
    Search {
        /// Search query
        query: String,
        
        /// Filter by registry type
        #[arg(long)]
        registry: Option<String>,
        
        /// Maximum number of results
        #[arg(long, default_value = "20")]
        limit: usize,
        
        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
    },
    
    /// Show cache status and statistics
    Cache {
        #[command(subcommand)]
        action: CacheAction,
    },
    
    /// Convert model between formats
    Convert {
        /// Source model path or URI
        source: String,
        
        /// Target format (safetensors, gguf)
        #[arg(long, default_value = "safetensors")]
        to: String,
        
        /// Output path (optional, defaults to same directory)
        #[arg(long)]
        output: Option<String>,
        
        /// Target precision (bf16, fp16, fp32)
        #[arg(long, default_value = "bf16")]
        precision: String,
        
        /// Verify conversion accuracy
        #[arg(long)]
        verify: bool,
    },
    
    /// List available registries
    Registries,
    
    /// Test a model with inference
    Test {
        /// Model path (local SafeTensors file)
        path: std::path::PathBuf,
        
        /// Test prompt
        #[arg(long, default_value = "Hello, how are you?")]
        prompt: String,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "50")]
        max_tokens: usize,
        
        /// Use X-LoRA if available
        #[arg(long)]
        xlora: bool,
        
        /// X-LoRA model ID (required if --xlora is used)
        #[arg(long)]
        xlora_model_id: Option<String>,
        
        /// Maximum adapters for X-LoRA
        #[arg(long, default_value = "4")]
        max_adapters: usize,
    },
}

/// Cache management actions
#[derive(Subcommand)]
pub enum CacheAction {
    /// Show cache status
    Status,
    
    /// Clean up old cached models
    Cleanup {
        /// Confirm cleanup without prompting
        #[arg(long)]
        yes: bool,
    },
    
    /// Verify integrity of cached models
    Verify {
        /// Model URI to verify (optional, verifies all if not specified)
        uri: Option<String>,
    },
}

/// Model pull configuration
#[derive(Debug, Clone)]
pub struct PullConfig {
    pub uri: String,
    pub force: bool,
    pub files: Option<Vec<String>>,
    pub show_progress: bool,
}

/// Model list configuration
#[derive(Debug, Clone)]
pub struct ListConfig {
    pub registry: Option<String>,
    pub search: Option<String>,
    pub include_remote: bool,
    pub format: OutputFormat,
}

/// Output format options
#[derive(Debug, Clone)]
pub enum OutputFormat {
    Table,
    Json,
    Yaml,
}

impl From<&str> for OutputFormat {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            _ => Self::Table,
        }
    }
}

/// Model information for display
#[derive(Debug, Clone)]
pub struct ModelDisplayInfo {
    pub uri: String,
    pub registry: String,
    pub org: String,
    pub name: String,
    pub size_mb: Option<f64>,
    pub cached: bool,
    pub files_count: usize,
    pub last_accessed: Option<String>,
    pub parameters: Option<String>,
    pub model_type: String,
}

impl ModelDisplayInfo {
    /// Format size for display
    pub fn format_size(&self) -> String {
        match self.size_mb {
            Some(size) => {
                if size < 1024.0 {
                    format!("{:.1} MB", size)
                } else {
                    format!("{:.1} GB", size / 1024.0)
                }
            }
            None => "Unknown".to_string(),
        }
    }
    
    /// Format parameters for display
    pub fn format_parameters(&self) -> String {
        match &self.parameters {
            Some(params) => params.clone(),
            None => "Unknown".to_string(),
        }
    }
    
    /// Format last accessed time
    pub fn format_last_accessed(&self) -> String {
        match &self.last_accessed {
            Some(time) => time.clone(),
            None => "Never".to_string(),
        }
    }
}