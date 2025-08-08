//! Example demonstrating OpenVDB-based sparse LoRA storage
//! 
//! This shows how to use OpenVDB for efficient storage and manipulation
//! of 99% sparse LoRA adapter weights with real-time updates.

use hyprstream::storage::vdb::{OpenVDBLoRAAdapter, OpenVDBBatchOps};
use std::collections::HashMap;
use anyhow::Result;
use rand;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 OpenVDB Sparse LoRA Storage Example");
    
    // Create a sparse LoRA adapter for a transformer layer
    // Simulating Qwen3-1.7B dimensions: 2048x2048
    let mut adapter = OpenVDBLoRAAdapter::new(2048, 2048)?;
    
    println!("📊 Initial adapter stats:");
    println!("  Active weights: {}", adapter.active_count());
    println!("  Memory usage: {} bytes", adapter.memory_usage());
    println!("  Sparsity ratio: {:.2}%", adapter.sparsity_ratio() * 100.0);
    
    // Simulate sparse LoRA weight updates (99% sparse pattern)
    println!("\n🔥 Setting sparse weights...");
    
    let sparse_updates = generate_sparse_weights(2048, 2048, 0.01); // 1% density
    adapter.batch_update(&sparse_updates)?;
    
    println!("📊 After adding {} sparse weights:", sparse_updates.len());
    println!("  Active weights: {}", adapter.active_count());
    println!("  Memory usage: {} bytes", adapter.memory_usage());
    println!("  Sparsity ratio: {:.2}%", adapter.sparsity_ratio() * 100.0);
    
    // Simulate training updates
    println!("\n🎯 Simulating training updates...");
    
    let gradient_updates = generate_gradient_updates(&sparse_updates, 0.001); // Small learning rate
    adapter.batch_update(&gradient_updates)?;
    
    // Demonstrate iteration over active weights
    println!("\n🔍 First 10 active weights:");
    for (i, (row, col, weight)) in adapter.active_weights().enumerate() {
        if i >= 10 { break; }
        println!("  [{}, {}] = {:.6}", row, col, weight);
    }
    
    // Demonstrate adapter fusion
    println!("\n🔗 Creating and fusing multiple adapters...");
    
    let mut adapter2 = OpenVDBLoRAAdapter::new(2048, 2048)?;
    let sparse_updates2 = generate_sparse_weights(2048, 2048, 0.005); // Different sparsity pattern
    adapter2.batch_update(&sparse_updates2)?;
    
    let mut adapter3 = OpenVDBLoRAAdapter::new(2048, 2048)?;
    let sparse_updates3 = generate_sparse_weights(2048, 2048, 0.008);
    adapter3.batch_update(&sparse_updates3)?;
    
    // Fuse multiple adapters with different scales
    let adapters = vec![&adapter, &adapter2, &adapter3];
    let scales = vec![1.0, 0.5, 0.3]; // Different importance weights
    
    let fused_adapter = OpenVDBBatchOps::fuse_adapters(&adapters, &scales)?;
    
    println!("📊 Fused adapter stats:");
    println!("  Active weights: {}", fused_adapter.active_count());
    println!("  Memory usage: {} bytes", fused_adapter.memory_usage());
    println!("  Sparsity ratio: {:.2}%", fused_adapter.sparsity_ratio() * 100.0);
    
    // Demonstrate I/O operations
    println!("\n💾 Testing file I/O...");
    
    let filename = "/tmp/test_lora_adapter.vdb";
    fused_adapter.save(filename)?;
    println!("  Saved adapter to {}", filename);
    
    let mut loaded_adapter = OpenVDBLoRAAdapter::new(2048, 2048)?;
    loaded_adapter.load(filename)?;
    
    println!("  Loaded adapter:");
    println!("    Active weights: {}", loaded_adapter.active_count());
    println!("    Memory usage: {} bytes", loaded_adapter.memory_usage());
    
    // Demonstrate compatibility with HashMap
    println!("\n🔄 Converting to/from HashMap for compatibility...");
    
    let hashmap = fused_adapter.to_hashmap();
    println!("  Converted to HashMap with {} entries", hashmap.len());
    
    let from_hashmap = OpenVDBLoRAAdapter::from_hashmap(&hashmap, (2048, 2048))?;
    println!("  Recreated from HashMap:");
    println!("    Active weights: {}", from_hashmap.active_count());
    
    // Performance comparison
    println!("\n⚡ Performance comparison:");
    demonstrate_performance_comparison().await?;
    
    println!("\n✅ OpenVDB LoRA example completed successfully!");
    
    Ok(())
}

/// Generate sparse weight pattern for LoRA adapter
fn generate_sparse_weights(rows: usize, cols: usize, density: f32) -> Vec<(i32, i32, f32)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let total_elements = (rows * cols) as f32;
    let num_active = (total_elements * density) as usize;
    
    let mut weights = Vec::with_capacity(num_active);
    
    for _ in 0..num_active {
        let row = rng.gen_range(0..rows) as i32;
        let col = rng.gen_range(0..cols) as i32;
        let weight = rng.gen_range(-1.0..1.0);
        
        // Only add non-zero weights
        if weight.abs() > 1e-6 {
            weights.push((row, col, weight));
        }
    }
    
    weights
}

/// Generate gradient updates for training simulation
fn generate_gradient_updates(existing_weights: &[(i32, i32, f32)], learning_rate: f32) -> Vec<(i32, i32, f32)> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    existing_weights.iter()
        .map(|(row, col, weight)| {
            let gradient = rng.gen_range(-0.1..0.1);
            let updated_weight = weight + learning_rate * gradient;
            (*row, *col, updated_weight)
        })
        .collect()
}

/// Demonstrate performance comparison between approaches
async fn demonstrate_performance_comparison() -> Result<()> {
    use std::time::Instant;
    
    let num_weights = 10000;
    let updates = generate_sparse_weights(2048, 2048, 0.002);
    
    // OpenVDB performance
    println!("  🔬 Testing OpenVDB performance...");
    let start = Instant::now();
    
    let mut openvdb_adapter = OpenVDBLoRAAdapter::new(2048, 2048)?;
    openvdb_adapter.batch_update(&updates)?;
    
    // Simulate some lookups
    for i in 0..1000 {
        let row = (i * 13) % 2048;
        let col = (i * 17) % 2048;
        let _ = openvdb_adapter.get_weight(row as i32, col as i32);
    }
    
    let openvdb_time = start.elapsed();
    let openvdb_memory = openvdb_adapter.memory_usage();
    
    // HashMap performance (for comparison)
    println!("  🔬 Testing HashMap performance...");
    let start = Instant::now();
    
    let mut hashmap: HashMap<(i32, i32), f32> = HashMap::new();
    for (row, col, weight) in &updates {
        if weight.abs() > 1e-8 {
            hashmap.insert((*row, *col), *weight);
        }
    }
    
    // Simulate same lookups
    for i in 0..1000 {
        let row = (i * 13) % 2048;
        let col = (i * 17) % 2048;
        let _ = hashmap.get(&(row, col)).unwrap_or(&0.0);
    }
    
    let hashmap_time = start.elapsed();
    let hashmap_memory = hashmap.len() * (std::mem::size_of::<(i32, i32)>() + std::mem::size_of::<f32>());
    
    println!("  📊 Performance Results:");
    println!("    OpenVDB: {:?}, {} bytes memory", openvdb_time, openvdb_memory);
    println!("    HashMap: {:?}, {} bytes memory", hashmap_time, hashmap_memory);
    println!("    Memory ratio: {:.2}x improvement", 
             hashmap_memory as f32 / openvdb_memory as f32);
    
    Ok(())
}