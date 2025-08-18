# VDB Temporal Z-Axis Design: Time-Based Sparse Deltas

## Core Concept: Z-Axis as Temporal Dimension

Instead of using Z for different layer types, we use it to represent **time**, with each Z-slice storing **sparse weight deltas** rather than complete weight copies.

```
Z-Axis = Time
├── Z[0]: Base weights (t=0)
├── Z[1]: Sparse delta mask (t=1) 
├── Z[2]: Sparse delta mask (t=2)
├── Z[3]: Sparse delta mask (t=3)
└── Z[n]: Sparse delta mask (t=n)
```

## Key Innovation: Sparse Temporal Masks

Each Z-layer stores only the **weights that changed** at that timestep:

```cpp
// Instead of full weight copy at each timestep:
weights_t1 = full_copy(weights_t0)  // Wasteful!

// We store sparse deltas:
Z[0] = base_weights        // Full sparse weights
Z[1] = {coord_5: +0.01, coord_42: -0.02}  // Only changes
Z[2] = {coord_5: +0.005, coord_99: +0.03} // Only changes
Z[3] = {coord_42: -0.01}                  // Only changes
```

## Benefits

### 1. **Extreme Compression**
- Base layer: ~1% active (99% sparse)
- Delta layers: ~0.01% active (99.99% sparse)
- Total storage: O(changes) not O(weights × time)

### 2. **Natural Temporal Queries**
```cpp
// Get weights at time t=2
weights_t2 = Z[0] + Z[1] + Z[2]  // Accumulate deltas

// Get recent changes
recent_deltas = Z[t-5:t]  // Last 5 timesteps

// Rollback to previous state
weights_t_minus_1 = current - Z[t]
```

### 3. **Perfect VDB Utilization**
- Hierarchical tree structure handles sparse deltas efficiently
- Tile compression for unchanged regions
- Native 3D operations for temporal accumulation

## Implementation Design

### Coordinate Mapping
```rust
pub struct TemporalCoordinate {
    x: i32,      // Weight matrix column
    y: i32,      // Weight matrix row  
    z: i32,      // Temporal index (time)
}

impl TemporalCoordinate {
    pub fn at_time(&self, t: usize) -> Coordinate3D {
        Coordinate3D::new(self.x, self.y, t as i32)
    }
    
    pub fn base_weight(&self) -> Coordinate3D {
        Coordinate3D::new(self.x, self.y, 0)  // Z=0 is base
    }
}
```

### Temporal Storage Interface
```rust
pub trait TemporalVDBStorage {
    /// Store base weights at Z=0
    async fn store_base(&mut self, weights: &SparseWeights) -> Result<()>;
    
    /// Store only changed weights at Z=timestamp
    async fn store_delta(&mut self, 
        timestamp: usize,
        changes: &HashMap<(i32, i32), f32>
    ) -> Result<()>;
    
    /// Reconstruct weights at specific time
    async fn weights_at_time(&self, timestamp: usize) -> Result<SparseWeights>;
    
    /// Get weight history for specific coordinate
    async fn weight_history(&self, x: i32, y: i32) -> Result<Vec<(usize, f32)>>;
    
    /// Prune old deltas (keep only last N timesteps)
    async fn prune_history(&mut self, keep_last: usize) -> Result<()>;
}
```

### Delta Detection & Storage
```rust
pub struct DeltaDetector {
    threshold: f32,  // Minimum change to store
    previous_state: HashMap<(i32, i32), f32>,
}

impl DeltaDetector {
    pub fn compute_deltas(&mut self, 
        current: &SparseWeights
    ) -> HashMap<(i32, i32), f32> {
        let mut deltas = HashMap::new();
        
        for ((x, y), &new_val) in current.iter() {
            if let Some(&old_val) = self.previous_state.get(&(x, y)) {
                let change = new_val - old_val;
                if change.abs() > self.threshold {
                    deltas.insert((x, y), change);
                }
            } else if new_val.abs() > self.threshold {
                // New activation
                deltas.insert((x, y), new_val);
            }
        }
        
        // Check for deactivations
        for ((x, y), &old_val) in &self.previous_state {
            if !current.contains(&(x, y)) && old_val.abs() > self.threshold {
                deltas.insert((x, y), -old_val);  // Deactivation
            }
        }
        
        self.previous_state = current.to_hashmap();
        deltas
    }
}
```

## Advanced Features

### 1. **Circular Buffer in Z**
```rust
// Use Z-axis as circular buffer for streaming
const MAX_HISTORY: i32 = 1000;

fn get_z_for_timestamp(timestamp: usize) -> i32 {
    if timestamp == 0 {
        0  // Base weights always at Z=0
    } else {
        1 + ((timestamp - 1) % (MAX_HISTORY - 1)) as i32
    }
}
```

### 2. **Hierarchical Temporal Resolution**
```
Z[0]:     Base (t=0)
Z[1-10]:  High-frequency updates (every inference)
Z[11-50]: Medium-frequency (every 10 inferences)  
Z[51-100]: Low-frequency (every 100 inferences)
```

### 3. **Gradient Accumulation in Z**
```cpp
// Accumulate gradients over time in VDB
void accumulate_temporal_gradient(int x, int y, float gradient, int timestep) {
    auto coord = openvdb::Coord(x, y, timestep);
    
    // VDB handles sparse accumulation efficiently
    if (grid->isValueOn(coord)) {
        grid->setValue(coord, grid->getValue(coord) + gradient);
    } else {
        grid->setValue(coord, gradient);
    }
}
```

### 4. **Temporal Attention Patterns**
```rust
// Query which weights changed together
pub async fn correlated_changes(&self, t: usize) -> Vec<(Coordinate3D, Coordinate3D)> {
    let changes_at_t = self.get_delta_layer(t).await?;
    
    // Find weights that consistently change together
    let mut correlations = Vec::new();
    for (coord1, _) in &changes_at_t {
        for (coord2, _) in &changes_at_t {
            if self.co_occurrence_score(coord1, coord2) > 0.8 {
                correlations.push((coord1.to_3d(), coord2.to_3d()));
            }
        }
    }
    correlations
}
```

## Benefits Over Traditional Approaches

### vs. Checkpointing
- **Checkpoints**: Store complete model every N steps → O(n_weights × n_checkpoints)
- **Temporal Z**: Store only deltas → O(n_changes)
- **Compression**: 100-1000x better

### vs. HashMap of Timesteps
- **HashMap**: Random memory access, no spatial locality
- **VDB Z-layers**: Spatial-temporal locality, hardware-optimized traversal

### vs. Separate Gradient Storage
- **Separate**: Gradients disconnected from weights
- **Unified Z**: Weights and temporal changes in same structure

## Integration with Existing Hyprstream

### Minimal Changes Required
```cpp
// Old: Everything at Z=0
openvdb::Coord LoRAGrid::to3D(int32_t row, int32_t col) const {
    return openvdb::Coord(row, col, 0);
}

// New: Add temporal awareness
openvdb::Coord LoRAGrid::to3D(int32_t row, int32_t col, int32_t timestep = 0) const {
    return openvdb::Coord(row, col, timestep);
}
```

### Backward Compatible
- Z=0 remains base weights
- Existing code works unchanged
- New temporal features are additive

## Performance Implications

### Memory Usage
```
Traditional: 1.5GB × 100 checkpoints = 150GB
Temporal Z:  1.5GB + (15MB × 1000 deltas) = 16.5GB
Savings:     ~90% reduction
```

### Access Patterns
```cpp
// Temporal reconstruction is sequential in Z
// VDB optimizes for this pattern naturally
for (int z = 0; z <= current_time; ++z) {
    accumulate_delta(grid->getValueAtZ(x, y, z));
}
```

### Cache Performance
- Temporal locality: Recent deltas likely in cache
- Spatial locality: Related weights in same VDB nodes
- Z-traversal: Optimized by VDB's tree structure

## Implementation Phases

### Phase 1: Basic Temporal Z Storage
- Store base at Z=0
- Store deltas at Z=t
- Implement weight reconstruction

### Phase 2: Delta Optimization
- Threshold-based delta detection
- Compression of similar deltas
- Pruning of old history

### Phase 3: Advanced Queries
- Temporal correlation analysis
- Gradient accumulation patterns
- Attention weight evolution

### Phase 4: Performance Optimization
- Circular buffer implementation
- Hierarchical temporal resolution
- Parallel reconstruction

## Conclusion

Using Z for time with sparse delta masks is a perfect fit for Hyprstream because:

1. **Aligns with mission**: Temporal adaptation is core to the project
2. **Maximizes VDB**: Uses 3D structure naturally and efficiently
3. **Extreme efficiency**: Stores only changes, not full copies
4. **Natural operations**: Time-based queries map directly to Z-slices
5. **Hardware friendly**: Sequential Z-access optimizes cache/memory

This approach transforms VDB from a "sparse weight storage" into a "temporal weight evolution engine" - exactly what Hyprstream needs for real-time adaptation during inference.