//! Minimal Rust FFI wrapper for PyTorch C++ engine

use anyhow::Result;
use std::ffi::{c_char, c_void, CString};
use std::ptr;

/// Opaque handle to C++ engine
#[repr(C)]
pub struct NativeEngine {
    _private: [u8; 0],
}

#[link(name = "hyprstream_native")]
extern "C" {
    fn hyprstream_engine_create(model_path: *const c_char) -> *mut c_void;
    
    fn hyprstream_engine_forward(
        engine: *mut c_void,
        input_ids: *mut f32,
        batch_size: usize,
        sequence_length: usize,
    ) -> *mut f32;
    
    fn hyprstream_engine_update_lora(
        engine: *mut c_void,
        layer_name: *const c_char,
        lora_a: *mut f32,
        lora_b: *mut f32,
        rank: usize,
    );
    
    fn hyprstream_engine_destroy(engine: *mut c_void);
}

/// HyprStream engine with zero-copy PyTorch integration
pub struct HyprStreamEngine {
    engine: *mut c_void,
}

unsafe impl Send for HyprStreamEngine {}
unsafe impl Sync for HyprStreamEngine {}

impl HyprStreamEngine {
    /// Create new engine from model path
    pub fn new(model_path: &str) -> Result<Self> {
        let c_path = CString::new(model_path)?;
        let engine = unsafe { hyprstream_engine_create(c_path.as_ptr()) };
        
        if engine.is_null() {
            return Err(anyhow::anyhow!("Failed to create engine"));
        }
        
        Ok(Self { engine })
    }
    
    /// Forward pass with zero-copy
    /// 
    /// # Safety
    /// - `input_ids` must be valid GPU memory
    /// - Returned slice is valid until next forward call
    pub unsafe fn forward_raw(
        &self,
        input_ids: *mut f32,
        batch_size: usize,
        sequence_length: usize,
    ) -> *mut f32 {
        hyprstream_engine_forward(
            self.engine,
            input_ids,
            batch_size,
            sequence_length,
        )
    }
    
    /// Safe forward pass (allocates output buffer)
    pub fn forward(
        &self,
        input_ids: &[f32],
        batch_size: usize,
        sequence_length: usize,
        vocab_size: usize,
    ) -> Result<Vec<f32>> {
        // This version copies for safety
        let output_ptr = unsafe {
            self.forward_raw(
                input_ids.as_ptr() as *mut f32,
                batch_size,
                sequence_length,
            )
        };
        
        if output_ptr.is_null() {
            return Err(anyhow::anyhow!("Forward pass failed"));
        }
        
        // Copy output to Rust-owned memory
        let output_size = batch_size * sequence_length * vocab_size;
        let output = unsafe {
            std::slice::from_raw_parts(output_ptr, output_size).to_vec()
        };
        
        Ok(output)
    }
    
    /// Update LoRA weights in-place
    pub fn update_lora(
        &self,
        layer_name: &str,
        lora_a: &[f32],
        lora_b: &[f32],
        rank: usize,
    ) -> Result<()> {
        let c_name = CString::new(layer_name)?;
        
        unsafe {
            hyprstream_engine_update_lora(
                self.engine,
                c_name.as_ptr(),
                lora_a.as_ptr() as *mut f32,
                lora_b.as_ptr() as *mut f32,
                rank,
            );
        }
        
        Ok(())
    }
}

impl Drop for HyprStreamEngine {
    fn drop(&mut self) {
        unsafe {
            hyprstream_engine_destroy(self.engine);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_engine_creation() {
        // This would need a real model file to test
        // let engine = HyprStreamEngine::new("model.pt");
        // assert!(engine.is_ok());
    }
}