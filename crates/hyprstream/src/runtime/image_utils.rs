//! Image loading and preprocessing utilities for multimodal models
//!
//! This module provides utilities for:
//! - Loading images from disk (PNG, JPEG, etc.)
//! - Preprocessing images for vision encoders
//! - Creating fallback minimal images
//! - Batch image processing

use anyhow::{anyhow, Result};
use std::path::Path;
use tch::{Device, Kind, Tensor};
use tracing::{debug, warn};

/// Image preprocessing configuration
#[derive(Debug, Clone)]
pub struct ImagePreprocessConfig {
    /// Target image size (width and height)
    pub image_size: usize,
    /// Mean values for normalization [R, G, B]
    pub mean: [f32; 3],
    /// Std values for normalization [R, G, B]
    pub std: [f32; 3],
    /// Whether to convert to RGB (from RGBA, grayscale, etc.)
    pub convert_rgb: bool,
}

impl Default for ImagePreprocessConfig {
    fn default() -> Self {
        // SigLIP/CLIP default normalization
        Self {
            image_size: 384,
            mean: [0.48145466, 0.4578275, 0.40821073],
            std: [0.26862954, 0.261_302_6, 0.275_777_1],
            convert_rgb: true,
        }
    }
}

impl ImagePreprocessConfig {
    /// Configuration for SigLIP vision encoder
    pub fn siglip() -> Self {
        Self {
            image_size: 384,
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            convert_rgb: true,
        }
    }

    /// Configuration for CLIP vision encoder
    pub fn clip() -> Self {
        Self::default()
    }
}

/// Load an image from disk and preprocess it for vision encoder
///
/// # Arguments
/// * `path` - Path to the image file
/// * `config` - Preprocessing configuration
/// * `device` - Device to load tensor onto
///
/// # Returns
/// Tensor of shape [1, 3, H, W] ready for vision encoder
pub fn load_image(
    path: &Path,
    config: &ImagePreprocessConfig,
    device: Device,
) -> Result<Tensor> {
    debug!("Loading image from: {}", path.display());

    // Load image using image crate
    let img = image::open(path).map_err(|e| anyhow!("Failed to load image: {}", e))?;

    // Convert to RGB (always needed for model input)
    let img = img.to_rgb8();

    // Resize to target size (keeping aspect ratio, then center crop)
    let img = image::imageops::resize(
        &img,
        config.image_size as u32,
        config.image_size as u32,
        image::imageops::FilterType::Lanczos3,
    );

    // Convert to tensor [H, W, C]
    let (width, height) = img.dimensions();
    let img_vec: Vec<f32> = img
        .into_raw()
        .iter()
        .map(|&x| x as f32 / 255.0)
        .collect();

    // Reshape to [H, W, C]
    let tensor = Tensor::from_slice(&img_vec)
        .reshape([height as i64, width as i64, 3]);

    // Transpose to [C, H, W]
    let tensor = tensor.permute([2, 0, 1]);

    // Normalize
    let tensor = normalize_tensor(&tensor, &config.mean, &config.std)?;

    // Add batch dimension [1, C, H, W]
    let tensor = tensor.unsqueeze(0);

    // Move to target device
    Ok(tensor.to_device(device))
}

/// Load multiple images and batch them
pub fn load_images(
    paths: &[&Path],
    config: &ImagePreprocessConfig,
    device: Device,
) -> Result<Tensor> {
    if paths.is_empty() {
        return Err(anyhow!("No image paths provided"));
    }

    let images: Result<Vec<Tensor>> = paths
        .iter()
        .map(|path| load_image(path, config, device))
        .collect();

    let images = images?;

    // Stack into batch [N, C, H, W]
    Ok(Tensor::cat(&images, 0))
}

/// Create a minimal fallback image (1x1 random pixel expanded to full size)
///
/// This is used when no image is provided but the model requires vision input.
/// The image is essentially a solid color with a random RGB value.
///
/// # Arguments
/// * `config` - Preprocessing configuration (determines output size)
/// * `device` - Device to create tensor on
///
/// # Returns
/// Tensor of shape [1, 3, H, W] with random solid color
pub fn create_fallback_image(
    config: &ImagePreprocessConfig,
    device: Device,
) -> Result<Tensor> {
    warn!("Creating fallback minimal image (no image provided)");

    // Generate random RGB pixel values [0, 1]
    let r = fastrand::f32();
    let g = fastrand::f32();
    let b = fastrand::f32();

    debug!("Fallback image color: R={:.3}, G={:.3}, B={:.3}", r, g, b);

    // Create solid color image [C, H, W]
    let r_channel = Tensor::full(
        [1, config.image_size as i64, config.image_size as i64],
        r as f64,
        (Kind::Float, device),
    );
    let g_channel = Tensor::full(
        [1, config.image_size as i64, config.image_size as i64],
        g as f64,
        (Kind::Float, device),
    );
    let b_channel = Tensor::full(
        [1, config.image_size as i64, config.image_size as i64],
        b as f64,
        (Kind::Float, device),
    );

    // Stack channels [3, H, W]
    let tensor = Tensor::cat(&[r_channel, g_channel, b_channel], 0);

    // Normalize
    let tensor = normalize_tensor(&tensor, &config.mean, &config.std)?;

    // Add batch dimension [1, 3, H, W]
    Ok(tensor.unsqueeze(0))
}

/// Normalize a tensor using mean and std
///
/// Formula: (x - mean) / std
fn normalize_tensor(tensor: &Tensor, mean: &[f32; 3], std: &[f32; 3]) -> Result<Tensor> {
    let mean_tensor = Tensor::from_slice(mean)
        .to_device(tensor.device())
        .reshape([3, 1, 1]);
    let std_tensor = Tensor::from_slice(std)
        .to_device(tensor.device())
        .reshape([3, 1, 1]);

    Ok((tensor - mean_tensor) / std_tensor)
}

/// Image data structure for passing around
pub struct ImageInput {
    /// Preprocessed image tensor [1, 3, H, W]
    pub tensor: Tensor,
    /// Original path (if loaded from file)
    pub path: Option<String>,
    /// Whether this is a fallback minimal image
    pub is_fallback: bool,
}

impl ImageInput {
    /// Create from a file path
    pub fn from_path(
        path: &Path,
        config: &ImagePreprocessConfig,
        device: Device,
    ) -> Result<Self> {
        let tensor = load_image(path, config, device)?;
        Ok(Self {
            tensor,
            path: Some(path.to_string_lossy().to_string()),
            is_fallback: false,
        })
    }

    /// Create a fallback image
    pub fn fallback(config: &ImagePreprocessConfig, device: Device) -> Result<Self> {
        let tensor = create_fallback_image(config, device)?;
        Ok(Self {
            tensor,
            path: None,
            is_fallback: true,
        })
    }

    /// Get the tensor
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Check if this is a fallback image
    pub fn is_fallback(&self) -> bool {
        self.is_fallback
    }
}

/// Batch multiple images into a single tensor
///
/// Takes a vector of ImageInput instances and concatenates them along the batch dimension.
///
/// # Arguments
/// * `images` - Vector of ImageInput instances, each with shape [1, C, H, W]
///
/// # Returns
/// Batched tensor with shape [N, C, H, W] where N is the number of images
///
/// # Errors
/// Returns an error if the images have inconsistent shapes or if concatenation fails
pub fn batch_images(images: &[ImageInput]) -> Result<Tensor> {
    if images.is_empty() {
        return Err(anyhow::anyhow!("Cannot batch empty image list"));
    }

    if images.len() == 1 {
        // Single image - return as-is (already has batch dimension)
        return Ok(images[0].tensor.shallow_clone());
    }

    // Concatenate along batch dimension (dim 0)
    let tensors: Vec<&Tensor> = images.iter().map(|img| &img.tensor).collect();
    let batched = Tensor::cat(&tensors, 0);

    Ok(batched)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_image_creation() {
        let config = ImagePreprocessConfig::default();
        let device = Device::Cpu;

        let image = create_fallback_image(&config, device).expect("test: create fallback image");

        // Check shape
        assert_eq!(image.size(), &[1, 3, 384, 384]);

        // Check device
        assert_eq!(image.device(), device);
    }

    #[test]
    fn test_normalize_tensor() {
        let device = Device::Cpu;
        let tensor = Tensor::ones([3, 224, 224], (Kind::Float, device));

        let mean = [0.5, 0.5, 0.5];
        let std = [0.5, 0.5, 0.5];

        let normalized = normalize_tensor(&tensor, &mean, &std).expect("test: normalize tensor");

        // After normalization: (1.0 - 0.5) / 0.5 = 1.0
        let expected = Tensor::ones([3, 224, 224], (Kind::Float, device));

        assert!((normalized - expected).abs().mean(Kind::Float).double_value(&[]) < 1e-6);
    }

    #[test]
    fn test_image_input_fallback() {
        let config = ImagePreprocessConfig::default();
        let device = Device::Cpu;

        let image = ImageInput::fallback(&config, device).expect("test: create fallback image");

        assert!(image.is_fallback());
        assert!(image.path.is_none());
        assert_eq!(image.tensor().size(), &[1, 3, 384, 384]);
    }
}
