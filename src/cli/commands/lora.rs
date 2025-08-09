//! LoRA adapter management CLI commands

use clap::{Args, Subcommand};

/// LoRA adapter management commands
#[derive(Args)]
pub struct LoRACommand {
    #[command(subcommand)]
    pub action: LoRAAction,
}

/// LoRA management actions
#[derive(Subcommand)]
pub enum LoRAAction {
    /// Create a new LoRA adapter
    Create {
        /// Name for the LoRA adapter
        #[arg(long)]
        name: Option<String>,
        
        /// Base model URI (e.g., hf://microsoft/DialoGPT-medium) or UUID
        #[arg(long)]
        base_model: String,
        
        /// LoRA rank (lower = smaller adapter)
        #[arg(long, default_value = "8")]
        rank: usize,
        
        /// Alpha scaling parameter
        #[arg(long, default_value = "16.0")]
        alpha: f32,
        
        /// Dropout rate for training
        #[arg(long, default_value = "0.1")]
        dropout: f32,
        
        /// Target modules for LoRA adaptation
        #[arg(long, value_delimiter = ',', default_values = &["q_proj", "v_proj"])]
        target_modules: Vec<String>,
        
        /// Sparsity ratio (0.99 = 99% sparse)
        #[arg(long, default_value = "0.99")]
        sparsity: f32,
        
        /// Enable neural compression
        #[arg(long)]
        neural_compression: bool,
        
        /// Enable auto-regressive training
        #[arg(long)]
        auto_regressive: bool,
        
        /// Learning rate for training
        #[arg(long, default_value = "0.0001")]
        learning_rate: f32,
        
        /// Training batch size
        #[arg(long, default_value = "8")]
        batch_size: usize,
        
        /// Output format (json, table)
        #[arg(long, default_value = "json")]
        format: String,
    },
    
    /// List all LoRA adapters
    List {
        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
        
        /// Filter by base model
        #[arg(long)]
        base_model: Option<String>,
        
        /// Show only training-enabled adapters
        #[arg(long)]
        training_only: bool,
    },
    
    /// Get detailed information about a LoRA adapter
    Info {
        /// LoRA ID or name
        lora_id: String,
        
        /// Output format (json, yaml)
        #[arg(long, default_value = "json")]
        format: String,
        
        /// Include training statistics
        #[arg(long)]
        include_stats: bool,
    },
    
    /// Delete a LoRA adapter
    Delete {
        /// LoRA ID or name
        lora_id: String,
        
        /// Confirm deletion without prompting
        #[arg(long)]
        yes: bool,
    },
    
    /// Training management
    Train {
        #[command(subcommand)]
        action: TrainingAction,
    },
    
    /// Test inference with a LoRA adapter
    Infer {
        /// LoRA ID to use for inference
        lora_id: String,
        
        /// Input prompt
        #[arg(long)]
        prompt: Option<String>,
        
        /// Input file containing prompt
        #[arg(long)]
        input_file: Option<String>,
        
        /// Maximum tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,
        
        /// Temperature for sampling
        #[arg(long, default_value = "1.0")]
        temperature: f32,
        
        /// Top-p for nucleus sampling
        #[arg(long, default_value = "1.0")]
        top_p: f32,
        
        /// Enable streaming output
        #[arg(long)]
        stream: bool,
        
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
    
    /// Chat with a LoRA adapter
    Chat {
        /// LoRA ID to use for chat
        lora_id: String,
        
        /// Maximum tokens per response
        #[arg(long, default_value = "500")]
        max_tokens: usize,
        
        /// Temperature for sampling
        #[arg(long, default_value = "0.7")]
        temperature: f32,
        
        /// Load chat history from file
        #[arg(long)]
        history: Option<String>,
        
        /// Save chat history to file
        #[arg(long)]
        save_history: Option<String>,
    },
    
    /// Export LoRA adapter
    Export {
        /// LoRA ID to export
        lora_id: String,
        
        /// Output file path
        #[arg(long)]
        output: String,
        
        /// Export format (safetensors, pytorch, gguf)
        #[arg(long, default_value = "safetensors")]
        format: String,
        
        /// Include base model weights
        #[arg(long)]
        include_base: bool,
    },
    
    /// Import LoRA adapter
    Import {
        /// Input file path
        input: String,
        
        /// Name for imported adapter
        #[arg(long)]
        name: Option<String>,
        
        /// Auto-detect format
        #[arg(long)]
        auto_detect: bool,
    },
}

/// Training management actions
#[derive(Subcommand)]
pub enum TrainingAction {
    /// Start auto-regressive training
    Start {
        /// LoRA ID
        lora_id: String,
        
        /// Learning rate
        #[arg(long, default_value = "0.0001")]
        learning_rate: f32,
        
        /// Batch size
        #[arg(long, default_value = "8")]
        batch_size: usize,
        
        /// Training frequency (train every N samples)
        #[arg(long, default_value = "5")]
        frequency: usize,
        
        /// Enable mixed precision
        #[arg(long)]
        mixed_precision: bool,
    },
    
    /// Stop auto-regressive training
    Stop {
        /// LoRA ID
        lora_id: String,
    },
    
    /// Show training status
    Status {
        /// LoRA ID (optional, shows all if not specified)
        lora_id: Option<String>,
        
        /// Output format (table, json)
        #[arg(long, default_value = "table")]
        format: String,
        
        /// Continuously monitor training
        #[arg(long)]
        watch: bool,
        
        /// Update interval for watch mode (seconds)
        #[arg(long, default_value = "5")]
        interval: u64,
    },
    
    /// Manually add training samples
    Sample {
        /// LoRA ID
        lora_id: String,
        
        /// Input text
        #[arg(long)]
        input: Option<String>,
        
        /// Expected output
        #[arg(long)]
        output: Option<String>,
        
        /// Input file with training samples (JSON format)
        #[arg(long)]
        input_file: Option<String>,
    },
}

/// LoRA configuration for creation
#[derive(Debug, Clone)]
pub struct LoRACreateConfig {
    pub name: Option<String>,
    pub base_model: String,
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub sparsity_ratio: f32,
    pub neural_compression: bool,
    pub auto_regressive: bool,
    pub learning_rate: f32,
    pub batch_size: usize,
}

/// LoRA display information
#[derive(Debug, Clone)]
pub struct LoRADisplayInfo {
    pub id: String,
    pub name: String,
    pub base_model: String,
    pub rank: usize,
    pub sparsity_ratio: f32,
    pub training_enabled: bool,
    pub total_requests: u64,
    pub avg_latency_ms: f64,
    pub created_at: String,
    pub size_mb: f64,
    pub compression_ratio: f32,
    pub endpoint_url: String,
}

impl LoRADisplayInfo {
    /// Format creation time for display
    pub fn format_created_at(&self) -> String {
        // Convert timestamp to human-readable format
        self.created_at.clone()
    }
    
    /// Format size for display
    pub fn format_size(&self) -> String {
        if self.size_mb < 1.0 {
            format!("{:.2} KB", self.size_mb * 1024.0)
        } else if self.size_mb < 1024.0 {
            format!("{:.2} MB", self.size_mb)
        } else {
            format!("{:.2} GB", self.size_mb / 1024.0)
        }
    }
    
    /// Format sparsity as percentage
    pub fn format_sparsity(&self) -> String {
        format!("{:.1}%", self.sparsity_ratio * 100.0)
    }
    
    /// Format training status
    pub fn format_training_status(&self) -> String {
        if self.training_enabled {
            "🟢 Active"
        } else {
            "⚫ Inactive"
        }.to_string()
    }
}

/// Training status display information
#[derive(Debug, Clone)]
pub struct TrainingDisplayInfo {
    pub lora_id: String,
    pub name: String,
    pub is_training: bool,
    pub samples_processed: u64,
    pub current_loss: f32,
    pub learning_rate: f32,
    pub last_update: String,
    pub tokens_per_second: Option<f32>,
}

impl TrainingDisplayInfo {
    /// Format training status with emoji
    pub fn format_status(&self) -> String {
        if self.is_training {
            "🟢 Training"
        } else {
            "🔴 Stopped"
        }.to_string()
    }
    
    /// Format loss for display
    pub fn format_loss(&self) -> String {
        format!("{:.6}", self.current_loss)
    }
    
    /// Format tokens per second
    pub fn format_throughput(&self) -> String {
        match self.tokens_per_second {
            Some(tps) => format!("{:.1} tok/s", tps),
            None => "N/A".to_string(),
        }
    }
}

/// Chat message for interactive chat
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub timestamp: String,
}

/// Inference result
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub output: String,
    pub tokens_generated: usize,
    pub latency_ms: f64,
    pub finish_reason: String,
}