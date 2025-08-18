//! Benchmarks for sparse operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyprstream_core::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig, InitMethod};
use hyprstream_core::storage::vdb::{Coordinate3D, SparseWeightUpdate};
use std::collections::HashMap;

fn create_sparse_adapter(size: usize, sparsity: f32) -> SparseLoRAAdapter {
    let config = SparseLoRAConfig {
        in_features: size,
        out_features: size,
        rank: 16,
        alpha: 32.0,
        dropout: 0.0,
        target_modules: vec!["bench_module".to_string()],
        sparsity,
        sparsity_threshold: 0.01,
        learning_rate: 0.001,
        bias: false,
        enable_gradient_checkpointing: false,
        init_method: InitMethod::Random,
        mixed_precision: false,
    };
    SparseLoRAAdapter::new(config)
}

fn bench_sparse_adapter_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_adapter_creation");
    
    for size in [256, 512, 1024, 2048].iter() {
        for sparsity in [0.90, 0.95, 0.99].iter() {
            let id = format!("size_{}_sparsity_{}", size, sparsity);
            group.bench_with_input(
                BenchmarkId::from_parameter(&id),
                &(*size, *sparsity),
                |b, &(size, sparsity)| {
                    b.iter(|| {
                        create_sparse_adapter(size, sparsity)
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_sparse_weight_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_weight_access");
    
    let adapter = create_sparse_adapter(1024, 0.95);
    
    group.bench_function("get_single_weight", |b| {
        b.iter(|| {
            adapter.get_weight(black_box(100), black_box(200))
        });
    });
    
    group.bench_function("get_100_weights", |b| {
        b.iter(|| {
            for i in 0..100 {
                adapter.get_weight(black_box(i), black_box(i * 2));
            }
        });
    });
    
    group.finish();
}

fn bench_sparse_update_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_update_creation");
    
    for num_updates in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_updates),
            num_updates,
            |b, &num_updates| {
                b.iter(|| {
                    let mut updates = HashMap::new();
                    for i in 0..num_updates {
                        updates.insert(
                            Coordinate3D::new(i as i32, (i * 2) as i32, 0),
                            0.01 * i as f32
                        );
                    }
                    SparseWeightUpdate {
                        adapter_id: "bench_adapter".to_string(),
                        updates,
                        timestamp: chrono::Utc::now(),
                        gradient_norm: Some(0.1),
                        learning_rate: 0.001,
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_coordinate_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("coordinate_operations");
    
    group.bench_function("create_coordinate", |b| {
        b.iter(|| {
            Coordinate3D::new(black_box(100), black_box(200), black_box(1))
        });
    });
    
    group.bench_function("hash_coordinate", |b| {
        let coord = Coordinate3D::new(100, 200, 1);
        b.iter(|| {
            use std::hash::{Hash, Hasher};
            use std::collections::hash_map::DefaultHasher;
            let mut hasher = DefaultHasher::new();
            coord.hash(&mut hasher);
            hasher.finish()
        });
    });
    
    group.bench_function("compare_coordinates", |b| {
        let coord1 = Coordinate3D::new(100, 200, 1);
        let coord2 = Coordinate3D::new(101, 201, 1);
        b.iter(|| {
            black_box(coord1 == coord2)
        });
    });
    
    group.finish();
}

fn bench_sparsity_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparsity_patterns");
    
    // Benchmark different sparsity patterns
    let sizes = vec![
        ("small", 256),
        ("medium", 1024),
        ("large", 4096),
    ];
    
    for (name, size) in sizes {
        let adapter = create_sparse_adapter(size, 0.99);
        
        group.bench_function(format!("{}_count_nonzero", name), |b| {
            b.iter(|| {
                let mut count = 0;
                for i in 0..size {
                    for j in 0..size {
                        if adapter.get_weight(i as i32, j as i32).unwrap_or(0.0).abs() > 1e-6 {
                            count += 1;
                        }
                    }
                }
                count
            });
        });
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Compare memory usage of different sparsity levels
    group.bench_function("dense_equivalent", |b| {
        b.iter(|| {
            let size = 1024;
            let mut data = vec![vec![0.0f32; size]; size];
            for i in 0..size {
                for j in 0..size {
                    data[i][j] = 0.01 * (i + j) as f32;
                }
            }
            black_box(data)
        });
    });
    
    group.bench_function("sparse_95", |b| {
        b.iter(|| {
            create_sparse_adapter(1024, 0.95)
        });
    });
    
    group.bench_function("sparse_99", |b| {
        b.iter(|| {
            create_sparse_adapter(1024, 0.99)
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_sparse_adapter_creation,
    bench_sparse_weight_access,
    bench_sparse_update_creation,
    bench_coordinate_operations,
    bench_sparsity_patterns,
    bench_memory_usage
);
criterion_main!(benches);