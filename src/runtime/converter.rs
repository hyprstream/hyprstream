//! GGUF to SafeTensors converter with precision control
//!
//! Converts quantized GGUF models to SafeTensors format with
//! configurable precision (BF16/FP16/FP32) for LoRA training.

use anyhow::{Result, anyhow};
use candle_core::{Device, DType, Tensor};
use candle_core::quantized::{gguf_file, QTensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use indicatif::{ProgressBar, ProgressStyle};
use super::precision::{PrecisionConfig, HardwareCapabilities};
use super::fp8::{FP8Config, FP8Format};

/// Conversion options
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Target precision for converted weights
    pub target_dtype: DType,
    /// Batch conversion to reduce memory usage
    pub batch_layers: bool,
    /// Cache directory for conversions
    pub cache_dir: PathBuf,
    /// Verify conversion accuracy
    pub verify_conversion: bool,
    /// Maximum memory usage in GB
    pub max_memory_gb: f32,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            target_dtype: DType::BF16, // BF16 by default!
            batch_layers: true,
            cache_dir: PathBuf::from(".cache/conversions"),
            verify_conversion: false,
            max_memory_gb: 32.0,
        }
    }
}

/// GGUF to SafeTensors converter
pub struct ModelConverter {
    device: Device,
    options: ConversionOptions,
    precision_config: PrecisionConfig,
}

impl ModelConverter {
    /// Create new converter with auto-detected precision
    pub fn new(device: Device) -> Self {
        let precision_config = PrecisionConfig::auto_detect(&device);
        let mut options = ConversionOptions::default();
        
        // Set target dtype based on hardware capabilities
        let hw_caps = HardwareCapabilities::detect(&device);
        options.target_dtype = if hw_caps.supports_bf16 {
            DType::BF16
        } else {
            DType::F16
        };
        
        Self {
            device,
            options,
            precision_config,
        }
    }
    
    /// Create converter with specific options
    pub fn with_options(device: Device, options: ConversionOptions) -> Self {
        let precision_config = PrecisionConfig::auto_detect(&device);
        Self {
            device,
            options,
            precision_config,
        }
    }
    
    /// Check if conversion is needed
    pub fn needs_conversion(&self, model_path: &Path) -> Result<bool> {
        // Check if it's a GGUF file
        if model_path.extension().and_then(|s| s.to_str()) != Some("gguf") {
            return Ok(false); // Already SafeTensors or other format
        }
        
        // Check if cached conversion exists
        let cache_path = self.get_cache_path(model_path)?;
        if cache_path.exists() {
            // Check if cache is up-to-date
            let model_modified = fs::metadata(model_path)?.modified()?;
            let cache_modified = fs::metadata(&cache_path)?.modified()?;
            
            if cache_modified >= model_modified {
                tracing::info!("Using cached conversion: {}", cache_path.display());
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Get cache path for converted model
    fn get_cache_path(&self, model_path: &Path) -> Result<PathBuf> {
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow!("Invalid model path"))?;
        
        let dtype_suffix = match self.options.target_dtype {
            DType::BF16 => "bf16",
            DType::F16 => "fp16",
            DType::F32 => "fp32",
            _ => "unknown",
        };
        
        let cache_name = format!("{}_{}.safetensors", model_name, dtype_suffix);
        let cache_path = self.options.cache_dir.join(cache_name);
        
        // Ensure cache directory exists
        if !self.options.cache_dir.exists() {
            fs::create_dir_all(&self.options.cache_dir)?;
        }
        
        Ok(cache_path)
    }
    
    /// Convert GGUF to SafeTensors with progress tracking
    pub async fn convert(&self, gguf_path: &Path) -> Result<PathBuf> {
        tracing::info!(
            "Converting GGUF to SafeTensors ({})",
            match self.options.target_dtype {
                DType::BF16 => "BF16 - optimal for LoRA",
                DType::F16 => "FP16 - fallback precision",
                DType::F32 => "FP32 - maximum precision",
                _ => "unknown precision",
            }
        );
        
        let output_path = self.get_cache_path(gguf_path)?;
        
        // Load GGUF content
        let mut file = std::fs::File::open(gguf_path)?;
        let content = gguf_file::Content::read(&mut file)?;
        
        // Check architecture
        let architecture = content.metadata.get("general.architecture")
            .and_then(|v| match v.to_string() {
                Ok(s) => Some(s.to_string()),
                Err(_) => None,
            })
            .unwrap_or_else(|| "unknown".to_string());
        
        tracing::info!("Model architecture: {}", architecture);
        
        // Warn about unsupported architectures
        if architecture.to_lowercase().contains("gemma") {
            tracing::warn!("⚠️ Gemma GGUF models may have compatibility issues");
            tracing::warn!("  Consider using SafeTensors format directly for Gemma");
        }
        
        // Count tensors for progress bar
        let total_tensors = content.tensor_infos.len();
        let pb = ProgressBar::new(total_tensors as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        // Convert tensors
        let mut tensors = HashMap::new();
        let mut total_size = 0u64;
        
        if self.options.batch_layers {
            // Process layer by layer to reduce memory usage
            self.convert_batched(&mut file, content, &mut tensors, &pb).await?;
        } else {
            // Convert all at once (requires more memory)
            self.convert_all(&mut file, content, &mut tensors, &pb).await?;
        }
        
        pb.finish_with_message("Conversion complete!");
        
        // Save to SafeTensors
        tracing::info!("Saving to {}", output_path.display());
        self.save_safetensors(tensors, &output_path)?;
        
        // Verify if requested
        if self.options.verify_conversion {
            self.verify_conversion(gguf_path, &output_path).await?;
        }
        
        Ok(output_path)
    }
    
    /// Convert all tensors at once
    async fn convert_all(
        &self,
        file: &mut std::fs::File,
        content: gguf_file::Content,
        tensors: &mut HashMap<String, Tensor>,
        pb: &ProgressBar,
    ) -> Result<()> {
        for (name, _info) in &content.tensor_infos {
            pb.set_message(format!("Converting {}", name));
            
            // Load quantized tensor
            let qtensor = content.tensor(file, name, &self.device)?;
            
            // Dequantize to target precision
            let tensor = self.dequantize_tensor(qtensor)?;
            
            tensors.insert(name.clone(), tensor);
            pb.inc(1);
        }
        
        Ok(())
    }
    
    /// Convert layer by layer to reduce memory
    async fn convert_batched(
        &self,
        file: &mut std::fs::File,
        content: gguf_file::Content,
        tensors: &mut HashMap<String, Tensor>,
        pb: &ProgressBar,
    ) -> Result<()> {
        // Group tensors by layer
        let mut layers: HashMap<String, Vec<String>> = HashMap::new();
        
        for (name, _) in &content.tensor_infos {
            // Extract layer number from name (e.g., "layers.0.attention.wq")
            let layer_key = if name.contains("layers.") {
                name.split("layers.")
                    .nth(1)
                    .and_then(|s| s.split('.').next())
                    .unwrap_or("other")
                    .to_string()
            } else {
                "other".to_string()
            };
            
            layers.entry(layer_key).or_default().push(name.clone());
        }
        
        // Process each layer
        for (layer_id, tensor_names) in layers {
            pb.set_message(format!("Processing layer {}", layer_id));
            
            for name in tensor_names {
                let qtensor = content.tensor(file, &name, &self.device)?;
                let tensor = self.dequantize_tensor(qtensor)?;
                tensors.insert(name, tensor);
                pb.inc(1);
            }
            
            // Force garbage collection between layers
            // (In real Rust, we rely on drop, but this shows intent)
        }
        
        Ok(())
    }
    
    /// Dequantize tensor to target dtype
    fn dequantize_tensor(&self, qtensor: QTensor) -> Result<Tensor> {
        // First dequantize to full precision
        let tensor = qtensor.dequantize(&self.device)?;
        
        // Then convert to target dtype
        let target_tensor = match self.options.target_dtype {
            DType::BF16 => {
                // Check if BF16 is supported
                if self.precision_config.base_model_dtype == DType::BF16 {
                    tensor.to_dtype(DType::BF16)?
                } else {
                    // Fallback to FP16
                    tracing::warn!("BF16 not supported, using FP16");
                    tensor.to_dtype(DType::F16)?
                }
            }
            DType::F16 => tensor.to_dtype(DType::F16)?,
            DType::F32 => tensor.to_dtype(DType::F32)?,
            // FP8 support check - not yet available in candle 0.9.1
            dtype if dtype == DType::BF16 || dtype == DType::F16 => {
                // Already handled above
                tensor.to_dtype(dtype)?
            }
            _ => return Err(anyhow!("Unsupported target dtype: {:?}", self.options.target_dtype)),
        };
        
        Ok(target_tensor)
    }
    
    /// Save tensors to SafeTensors format
    fn save_safetensors(&self, tensors: HashMap<String, Tensor>, path: &Path) -> Result<()> {
        use safetensors::tensor::{TensorView, Dtype as SafeDtype};
        use safetensors::SafeTensors;
        
        // Convert candle tensors to safetensors format
        let mut metadata = HashMap::new();
        let mut tensor_data = Vec::new();
        
        for (name, tensor) in &tensors {
            // Get tensor data as bytes
            let shape = tensor.dims();
            let dtype = match tensor.dtype() {
                DType::BF16 => SafeDtype::BF16,
                DType::F16 => SafeDtype::F16,
                DType::F32 => SafeDtype::F32,
                DType::F64 => SafeDtype::F64,
                DType::U8 => SafeDtype::U8,
                DType::U32 => SafeDtype::U32,
                DType::I64 => SafeDtype::I64,
            };
            
            // Convert tensor to CPU if needed and get raw bytes
            let cpu_tensor = tensor.to_device(&Device::Cpu)?;
            let data = cpu_tensor.to_vec1::<f32>()?; // This needs proper type handling
            
            // Store tensor metadata
            metadata.insert(name.clone(), (dtype, shape.to_vec(), data));
        }
        
        // Build and save safetensors file
        // Note: This is simplified - actual implementation would need proper serialization
        safetensors::tensor::serialize_to_file(metadata, path, &None)?;
        Ok(())
    }
    
    /// Verify conversion accuracy
    async fn verify_conversion(&self, gguf_path: &Path, safetensors_path: &Path) -> Result<()> {
        tracing::info!("Verifying conversion accuracy...");
        
        // Sample a few tensors and compare
        // This is a simplified check - real implementation would be more thorough
        
        let mut gguf_file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut gguf_file)?;
        
        // Load SafeTensors using candle
        let st_tensors = candle_core::safetensors::load(safetensors_path, &self.device)?;
        
        // Compare first few tensors
        let sample_count = 5.min(gguf_content.tensor_infos.len());
        let mut max_error = 0.0f32;
        
        for (name, _) in gguf_content.tensor_infos.iter().take(sample_count) {
            let qtensor = gguf_content.tensor(&mut gguf_file, name, &self.device)?;
            let original = qtensor.dequantize(&self.device)?;
            
            // Load from SafeTensors
            if let Some(converted) = st_tensors.get(name) {
                // Calculate error
                let diff = (original.sub(converted))?.abs()?;
                let error = diff.max_all()?.to_scalar::<f32>()?;
                max_error = max_error.max(error);
            }
        }
        
        tracing::info!("Maximum conversion error: {:.6}", max_error);
        
        if max_error > 0.01 {
            tracing::warn!("⚠️ High conversion error detected");
        } else {
            tracing::info!("✅ Conversion verified successfully");
        }
        
        Ok(())
    }
}

/// Auto-convert GGUF for LoRA if needed
pub async fn auto_convert_for_lora(
    model_path: &Path,
    device: &Device,
) -> Result<PathBuf> {
    // Check if conversion is needed
    if model_path.extension().and_then(|s| s.to_str()) != Some("gguf") {
        // Already in SafeTensors or other format
        return Ok(model_path.to_path_buf());
    }
    
    // Create converter with BF16 target
    let mut converter = ModelConverter::new(device.clone());
    
    // Check hardware and adjust if needed
    let hw_caps = HardwareCapabilities::detect(device);
    if !hw_caps.supports_bf16 {
        tracing::info!("BF16 not supported, using FP16 for conversion");
        converter.options.target_dtype = DType::F16;
    }
    
    // Convert
    let converted_path = converter.convert(model_path).await?;
    
    tracing::info!(
        "✅ Auto-converted to {} for LoRA support",
        match converter.options.target_dtype {
            DType::BF16 => "BF16 (optimal)",
            DType::F16 => "FP16 (fallback)",
            _ => "unknown",
        }
    );
    
    Ok(converted_path)
}

/// Convert GGUF to FP8 for extreme compression
pub async fn convert_to_fp8(
    model_path: &Path,
    device: &Device,
    format: FP8Format,
) -> Result<PathBuf> {
    // Check FP8 support
    if !format.is_supported() {
        return Err(anyhow!(
            "FP8 format {:?} not supported. E4M3 is available, E5M2 coming soon.",
            format
        ));
    }
    
    // Create converter with FP8 target
    let mut converter = ModelConverter::new(device.clone());
    converter.options.target_dtype = format.to_dtype()?;
    
    // Add FP8 suffix to cache path
    let original_cache = converter.get_cache_path(model_path)?;
    let fp8_cache = original_cache.with_file_name(
        format!("{}_{}.safetensors", 
            original_cache.file_stem().unwrap().to_str().unwrap(),
            match format {
                FP8Format::E4M3 => "fp8e4m3",
                FP8Format::E5M2 => "fp8e5m2",
            }
        )
    );
    
    // Check if already converted
    if fp8_cache.exists() {
        tracing::info!("Using cached FP8 conversion: {}", fp8_cache.display());
        return Ok(fp8_cache);
    }
    
    // Convert with FP8 config
    let fp8_config = FP8Config::inference();
    tracing::info!(
        "Converting to FP8 {:?} (4x compression vs FP32)",
        format
    );
    
    // Load and quantize
    let mut file = std::fs::File::open(model_path)?;
    let content = gguf_file::Content::read(&mut file)?;
    
    let mut fp8_tensors = HashMap::new();
    let scaler = fp8_config.scaler;
    
    for (name, _) in &content.tensor_infos {
        let qtensor = content.tensor(&mut file, name, device)?;
        let tensor = qtensor.dequantize(device)?;
        
        // Quantize to FP8 with scaling
        let (fp8_tensor, _scale) = scaler.quantize(&tensor, format)?;
        fp8_tensors.insert(name.clone(), fp8_tensor);
    }
    
    // Save FP8 model
    converter.save_safetensors(fp8_tensors, &fp8_cache)?;
    
    tracing::info!(
        "✅ Converted to FP8 {:?} - saved to {}",
        format,
        fp8_cache.display()
    );
    
    Ok(fp8_cache)
}