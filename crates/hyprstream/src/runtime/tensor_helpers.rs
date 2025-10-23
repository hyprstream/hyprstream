use anyhow::{anyhow, Result};
/// Helper functions for Tch tensor operations
use tch::{Device, Kind as DType, Tensor};

/// Helper trait to convert usize arrays to i64 for IntList compatibility
pub trait ToIntList {
    type Output;
    fn to_intlist(&self) -> Self::Output;
}

impl ToIntList for [usize; 2] {
    type Output = [i64; 2];
    fn to_intlist(&self) -> Self::Output {
        [self[0] as i64, self[1] as i64]
    }
}

impl ToIntList for [usize; 3] {
    type Output = [i64; 3];
    fn to_intlist(&self) -> Self::Output {
        [self[0] as i64, self[1] as i64, self[2] as i64]
    }
}

impl ToIntList for [usize; 4] {
    type Output = [i64; 4];
    fn to_intlist(&self) -> Self::Output {
        [
            self[0] as i64,
            self[1] as i64,
            self[2] as i64,
            self[3] as i64,
        ]
    }
}

impl ToIntList for Vec<usize> {
    type Output = Vec<i64>;
    fn to_intlist(&self) -> Self::Output {
        self.iter().map(|&x| x as i64).collect()
    }
}

/// Helper for tensor cloning (Tch requires output tensor)
pub fn clone_tensor(tensor: &Tensor) -> Tensor {
    let out = Tensor::zeros_like(tensor);
    tensor.clone(&out)
}

/// Helper for squared tensor (sqr -> square)
pub fn square_tensor(tensor: &Tensor) -> Result<Tensor> {
    Ok(tensor.square())
}

/// Helper for broadcast multiplication
pub fn broadcast_mul(tensor: &Tensor, other: &Tensor) -> Result<Tensor> {
    Ok(tensor * other)
}

/// Helper for broadcast addition
pub fn broadcast_add(tensor: &Tensor, other: &Tensor) -> Result<Tensor> {
    Ok(tensor + other)
}

/// Helper for broadcast subtraction
pub fn broadcast_sub(tensor: &Tensor, other: &Tensor) -> Result<Tensor> {
    Ok(tensor - other)
}

/// Helper for getting tensor dtype (Tch uses kind())
pub fn get_dtype(tensor: &Tensor) -> DType {
    tensor.kind()
}

/// Helper for converting to dtype with proper arguments
pub fn to_dtype(tensor: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(tensor.to_dtype(dtype, false, false))
}

/// Helper for concatenating tensors
pub fn cat_tensors(tensors: &[Tensor], dim: i64) -> Result<Tensor> {
    Ok(Tensor::cat(tensors, dim))
}

/// Helper for matmul operations
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    a.matmul(b)
}

/// Helper to convert tensor to vec
pub fn to_vec1<T: tch::kind::Element>(tensor: &Tensor) -> Result<Vec<T>> {
    // Convert to CPU and correct type
    let cpu_tensor = tensor.to_kind(T::KIND).to(Device::Cpu);

    // Get the data as a slice and convert to vec
    let numel = cpu_tensor.numel();
    let _result: Vec<T> = Vec::with_capacity(numel);

    // Use tensor's data pointer to extract values
    // This is a placeholder - actual implementation would use tensor.data_ptr() or similar
    // For now, we'll return an error since direct conversion is not available
    Err(anyhow!("Direct tensor to vec conversion not yet implemented for tch. Use tensor operations instead."))
}

/// Create a scalar tensor from a value
pub fn scalar_tensor(value: f32, device: Device) -> Tensor {
    Tensor::from_slice(&[value]).to(device)
}

/// Get dimensions as 3-tuple
pub fn dims3(tensor: &Tensor) -> Result<(i64, i64, i64)> {
    let size = tensor.size();
    if size.len() != 3 {
        return Err(anyhow!("Expected 3D tensor, got {}D", size.len()));
    }
    Ok((size[0], size[1], size[2]))
}

/// Get dimensions as 4-tuple  
pub fn dims4(tensor: &Tensor) -> Result<(i64, i64, i64, i64)> {
    let size = tensor.size();
    if size.len() != 4 {
        return Err(anyhow!("Expected 4D tensor, got {}D", size.len()));
    }
    Ok((size[0], size[1], size[2], size[3]))
}
