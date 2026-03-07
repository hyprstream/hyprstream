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

/// Load an image from raw bytes (PNG, JPEG, or other formats supported by `image` crate)
pub fn load_image_from_bytes(
    bytes: &[u8],
    config: &ImagePreprocessConfig,
    device: Device,
) -> Result<Tensor> {
    debug!("Loading image from {} bytes", bytes.len());

    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow!("Failed to decode image bytes: {}", e))?;

    let img = img.to_rgb8();

    let img = image::imageops::resize(
        &img,
        config.image_size as u32,
        config.image_size as u32,
        image::imageops::FilterType::Lanczos3,
    );

    let (width, height) = img.dimensions();
    let img_vec: Vec<f32> = img
        .into_raw()
        .iter()
        .map(|&x| x as f32 / 255.0)
        .collect();

    let tensor = Tensor::from_slice(&img_vec)
        .reshape([height as i64, width as i64, 3]);

    let tensor = tensor.permute([2, 0, 1]);
    let tensor = normalize_tensor(&tensor, &config.mean, &config.std)?;
    let tensor = tensor.unsqueeze(0);

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

/// Preprocess raw pixel bytes into a tensor ready for vision inference.
///
/// Handles pixel format conversion (RGB/BGR u8 or pre-normalized f32 CHW)
/// and optional SigLIP normalization (resize to 384x384, (x-0.5)/0.5).
pub fn preprocess_raw_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    channels: u32,
    pixel_format: &str,
    preprocess_mode: &str,
    batch_count: u32,
    row_stride: u32,
    config: &ImagePreprocessConfig,
    device: Device,
) -> Result<Tensor> {
    let batch = std::cmp::max(batch_count, 1) as i64;
    let h = height as i64;
    let w = width as i64;
    let c = channels as i64;

    let tensor = match pixel_format {
        "float32Chw" => {
            let expected = (batch * c * h * w * 4) as usize;
            anyhow::ensure!(
                pixels.len() >= expected,
                "float32Chw: expected {} bytes, got {}",
                expected, pixels.len()
            );
            let float_data: &[f32] = bytemuck::cast_slice(&pixels[..expected]);
            Tensor::from_slice(float_data)
                .reshape([batch, c, h, w])
                .to_device(device)
        }
        fmt @ ("rgb8" | "bgr8") => {
            let stride = if row_stride > 0 {
                row_stride as usize
            } else {
                (w as usize) * (c as usize)
            };
            let expected = (batch as usize) * (h as usize) * stride;
            anyhow::ensure!(
                pixels.len() >= expected,
                "{}: expected {} bytes (stride={}), got {}",
                fmt, expected, stride, pixels.len()
            );

            let tight_stride = (w as usize) * (c as usize);
            let pixel_data = if stride == tight_stride {
                pixels[..expected].to_vec()
            } else {
                let mut tight = Vec::with_capacity((batch as usize) * (h as usize) * tight_stride);
                for row in 0..((batch as usize) * (h as usize)) {
                    let start = row * stride;
                    tight.extend_from_slice(&pixels[start..start + tight_stride]);
                }
                tight
            };

            let t = Tensor::from_slice(&pixel_data)
                .to_kind(tch::Kind::Uint8)
                .reshape([batch, h, w, c]);

            let t = t.to_kind(tch::Kind::Float) / 255.0;

            let t = if fmt == "bgr8" {
                t.index_select(3, &Tensor::from_slice(&[2i64, 1, 0]).to_device(t.device()))
            } else {
                t
            };

            t.permute([0, 3, 1, 2]).to_device(device)
        }
        other => anyhow::bail!("unsupported pixel format: {}", other),
    };

    match preprocess_mode {
        "siglip" => {
            let target = config.image_size as i64;
            let tensor = if h != target || w != target {
                tensor.upsample_bilinear2d([target, target], false, None, None)
            } else {
                tensor
            };
            normalize_tensor_batched(&tensor, &config.mean, &config.std)
        }
        "none" => Ok(tensor),
        other => anyhow::bail!("unsupported preprocess mode: {}", other),
    }
}

/// Normalize a batched tensor [B, C, H, W] using per-channel mean and std.
fn normalize_tensor_batched(tensor: &Tensor, mean: &[f32; 3], std: &[f32; 3]) -> Result<Tensor> {
    let mean_tensor = Tensor::from_slice(mean)
        .to_device(tensor.device())
        .reshape([1, 3, 1, 1]);
    let std_tensor = Tensor::from_slice(std)
        .to_device(tensor.device())
        .reshape([1, 3, 1, 1]);
    Ok((tensor - mean_tensor) / std_tensor)
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
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
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

    // ── preprocess_raw_pixels tests ──────────────────────────────────

    #[test]
    fn preprocess_rgb8_shape() {
        let config = ImagePreprocessConfig::siglip();
        let w: u32 = 384;
        let h: u32 = 384;
        let pixels = vec![128u8; (w * h * 3) as usize];

        let t = preprocess_raw_pixels(
            &pixels, w, h, 3, "rgb8", "none", 1, 0, &config, Device::Cpu,
        )
        .expect("rgb8 preprocess");

        assert_eq!(t.size(), &[1, 3, 384, 384]);
        assert_eq!(t.kind(), Kind::Float);
    }

    #[test]
    fn preprocess_rgb8_values() {
        // Single pixel: R=255, G=0, B=128
        let config = ImagePreprocessConfig::siglip();
        let pixels: Vec<u8> = vec![255, 0, 128];

        let t = preprocess_raw_pixels(
            &pixels, 1, 1, 3, "rgb8", "none", 1, 0, &config, Device::Cpu,
        )
        .expect("rgb8 values");

        // [1, 3, 1, 1] — channel values should be /255.0
        assert_eq!(t.size(), &[1, 3, 1, 1]);
        let r = t.double_value(&[0, 0, 0, 0]);
        let g = t.double_value(&[0, 1, 0, 0]);
        let b = t.double_value(&[0, 2, 0, 0]);
        assert!((r - 1.0).abs() < 1e-5, "R should be 1.0, got {r}");
        assert!(g.abs() < 1e-5, "G should be 0.0, got {g}");
        assert!((b - 128.0 / 255.0).abs() < 1e-4, "B should be ~0.502, got {b}");
    }

    #[test]
    fn preprocess_bgr8_channel_swap() {
        // BGR pixel: B=10, G=20, R=30 → should become RGB: R=30, G=20, B=10
        let config = ImagePreprocessConfig::siglip();
        let pixels: Vec<u8> = vec![10, 20, 30];

        let t = preprocess_raw_pixels(
            &pixels, 1, 1, 3, "bgr8", "none", 1, 0, &config, Device::Cpu,
        )
        .expect("bgr8 swap");

        let r = t.double_value(&[0, 0, 0, 0]);
        let g = t.double_value(&[0, 1, 0, 0]);
        let b = t.double_value(&[0, 2, 0, 0]);
        assert!(
            (r - 30.0 / 255.0).abs() < 1e-4,
            "R channel should be 30/255, got {r}"
        );
        assert!(
            (g - 20.0 / 255.0).abs() < 1e-4,
            "G channel should be 20/255, got {g}"
        );
        assert!(
            (b - 10.0 / 255.0).abs() < 1e-4,
            "B channel should be 10/255, got {b}"
        );
    }

    #[test]
    fn preprocess_float32chw_passthrough() {
        let config = ImagePreprocessConfig::siglip();
        // 1x3x2x2 = 12 floats = 48 bytes
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let pixels: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

        let t = preprocess_raw_pixels(
            &pixels, 2, 2, 3, "float32Chw", "none", 1, 0, &config, Device::Cpu,
        )
        .expect("float32Chw passthrough");

        assert_eq!(t.size(), &[1, 3, 2, 2]);
        // First element should be 0.0
        assert!(t.double_value(&[0, 0, 0, 0]).abs() < 1e-5);
        // Second element (0,0,0,1) should be 0.1
        assert!((t.double_value(&[0, 0, 0, 1]) - 0.1).abs() < 1e-5);
    }

    #[test]
    fn preprocess_stride_padding_removal() {
        // 2x2 RGB with stride=8 (6 tight bytes + 2 padding per row)
        let config = ImagePreprocessConfig::siglip();
        let mut pixels = Vec::new();
        // Row 0: [R,G,B, R,G,B, pad, pad]
        pixels.extend_from_slice(&[10, 20, 30, 40, 50, 60, 0, 0]);
        // Row 1: [R,G,B, R,G,B, pad, pad]
        pixels.extend_from_slice(&[70, 80, 90, 100, 110, 120, 0, 0]);

        let t = preprocess_raw_pixels(
            &pixels, 2, 2, 3, "rgb8", "none", 1, 8, &config, Device::Cpu,
        )
        .expect("stride padding removal");

        assert_eq!(t.size(), &[1, 3, 2, 2]);
        // Top-left R channel should be 10/255
        let r00 = t.double_value(&[0, 0, 0, 0]);
        assert!(
            (r00 - 10.0 / 255.0).abs() < 1e-4,
            "top-left R should be 10/255, got {r00}"
        );
        // Bottom-right B channel should be 120/255
        let b11 = t.double_value(&[0, 2, 1, 1]);
        assert!(
            (b11 - 120.0 / 255.0).abs() < 1e-4,
            "bottom-right B should be 120/255, got {b11}"
        );
    }

    #[test]
    fn preprocess_siglip_normalization() {
        // With SigLIP: mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]
        // pixel 128 → 128/255 ≈ 0.502 → (0.502 - 0.5) / 0.5 ≈ 0.004
        let config = ImagePreprocessConfig::siglip();
        let pixels = vec![128u8; 3]; // 1x1 pixel

        let t = preprocess_raw_pixels(
            &pixels, 1, 1, 3, "rgb8", "siglip", 1, 0, &config, Device::Cpu,
        )
        .expect("siglip normalization");

        // SigLIP resizes to 384x384 then normalizes
        assert_eq!(t.size(), &[1, 3, 384, 384]);

        let val = t.double_value(&[0, 0, 0, 0]);
        let expected = (128.0 / 255.0 - 0.5) / 0.5;
        assert!(
            (val - expected).abs() < 1e-4,
            "SigLIP normalized value should be ~{expected:.4}, got {val:.4}"
        );
    }

    #[test]
    fn preprocess_rgb8_too_small() {
        let config = ImagePreprocessConfig::siglip();
        // 2x2 needs 12 bytes, provide only 6
        let pixels = vec![0u8; 6];

        let result = preprocess_raw_pixels(
            &pixels, 2, 2, 3, "rgb8", "none", 1, 0, &config, Device::Cpu,
        );
        assert!(result.is_err(), "should fail on undersized buffer");
    }

    #[test]
    fn preprocess_float32chw_too_small() {
        let config = ImagePreprocessConfig::siglip();
        // 1x3x2x2 needs 48 bytes, provide only 16
        let pixels = vec![0u8; 16];

        let result = preprocess_raw_pixels(
            &pixels, 2, 2, 3, "float32Chw", "none", 1, 0, &config, Device::Cpu,
        );
        assert!(result.is_err(), "should fail on undersized float32 buffer");
    }

    #[test]
    fn preprocess_unsupported_pixel_format() {
        let config = ImagePreprocessConfig::siglip();
        let pixels = vec![0u8; 12];

        let result = preprocess_raw_pixels(
            &pixels, 2, 2, 3, "yuv420", "none", 1, 0, &config, Device::Cpu,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unsupported pixel format"),
            "error should mention unsupported format: {err}"
        );
    }

    #[test]
    fn preprocess_unsupported_preprocess_mode() {
        let config = ImagePreprocessConfig::siglip();
        let pixels = vec![128u8; 3];

        let result = preprocess_raw_pixels(
            &pixels, 1, 1, 3, "rgb8", "clip", 1, 0, &config, Device::Cpu,
        );
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("unsupported preprocess mode"),
            "error should mention unsupported mode: {err}"
        );
    }

    #[test]
    fn preprocess_batch_count_zero_treated_as_one() {
        let config = ImagePreprocessConfig::siglip();
        let pixels = vec![128u8; 3];

        let t = preprocess_raw_pixels(
            &pixels, 1, 1, 3, "rgb8", "none", 0, 0, &config, Device::Cpu,
        )
        .expect("batch_count=0 should be treated as 1");

        assert_eq!(t.size()[0], 1, "batch dim should be 1");
    }
}
