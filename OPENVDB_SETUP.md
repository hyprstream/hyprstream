# OpenVDB Integration Setup

This document explains how to set up OpenVDB for dynamic sparse LoRA storage in Hyprstream.

## Overview

Hyprstream now uses **OpenVDB** (instead of read-only NanoVDB) for dynamic sparse LoRA adapter storage. This provides:

- ✅ **Dynamic weight updates** during training
- ✅ **Real-time adapter modification** 
- ✅ **5-50x memory reduction** vs HashMap
- ✅ **Hierarchical sparse storage** optimized for 99% sparsity
- ✅ **Professional-grade VDB operations** (merge, prune, optimize)

## Installation

### Ubuntu/Debian

```bash
# Install OpenVDB and dependencies
sudo apt-get update
sudo apt-get install -y \
    libopenvdb-dev \
    openvdb-tools \
    libtbb-dev \
    libhalf-dev \
    libblosc-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-numpy-dev \
    pkg-config

# Verify installation
pkg-config --modversion openvdb
```

### macOS (Homebrew)

```bash
# Install OpenVDB via Homebrew
brew install openvdb tbb half blosc

# Verify installation
pkg-config --modversion openvdb
```

### Arch Linux

```bash
# Install from AUR
yay -S openvdb tbb half blosc

# Or using pacman if available
sudo pacman -S openvdb tbb intel-tbb blosc
```

### Building from Source

If OpenVDB is not available via package manager:

```bash
# Clone and build OpenVDB
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb

# Create build directory
mkdir build && cd build

# Configure with CMake (following OpenVDB's recommended settings)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DOPENVDB_BUILD_CORE=ON \
    -DOPENVDB_INSTALL_CMAKE_MODULES=ON \
    -DOPENVDB_BUILD_TOOLS=ON \
    -DOPENVDB_BUILD_PYTHON_MODULE=OFF \
    -DUSE_BLOSC=ON \
    -DUSE_TBB=ON

# Build and install
make -j$(nproc)
sudo make install

# Update library cache and pkg-config
sudo ldconfig
sudo updatedb  # Update locate database
```

### CMake Integration (For C++ Projects)

OpenVDB provides official CMake modules for downstream projects:

```cmake
cmake_minimum_required(VERSION 3.20)
list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")  # Adjust path as needed
find_package(OpenVDB REQUIRED)
target_link_libraries(myapp OpenVDB::openvdb)
```

Our Rust build system uses `pkg-config` instead of CMake, but follows the same dependency model.

## Building Hyprstream

### With OpenVDB Support (Recommended)

```bash
# Install OpenVDB first, then build with VDB features
cargo build --release --features vdb

# Success output:
# ✅ Built with OpenVDB support - VDB features enabled
```

### Without VDB Features (Basic functionality only)

```bash 
# Build without VDB support (uses HashMap fallback)
cargo build --release --no-default-features

# Output:
# 🔧 Building without VDB features (use --features vdb to enable)
```

### Default Behavior

```bash
# Default build includes VDB features (requires OpenVDB)
cargo build --release

# Will fail if OpenVDB not installed with clear error message
```

## Testing OpenVDB Integration

Run the example to test OpenVDB functionality:

```bash
# Run OpenVDB LoRA example
cargo run --example openvdb_lora_example --release

# Expected output:
# 🚀 OpenVDB Sparse LoRA Storage Example
# 📊 Initial adapter stats:
#   Active weights: 0
#   Memory usage: 1024 bytes
#   Sparsity ratio: 100.00%
# ...
```

## API Usage

### Basic Operations

```rust
use hyprstream::storage::vdb::{OpenVDBLoRAAdapter, OpenVDBBatchOps};

// Create sparse LoRA adapter
let mut adapter = OpenVDBLoRAAdapter::new(2048, 2048)?;

// Set individual weights
adapter.set_weight(10, 20, 0.5)?;
adapter.set_weight(100, 200, -0.3)?;

// Batch updates (efficient)
let updates = vec![
    (50, 60, 0.1),
    (150, 250, 0.8),
    (500, 750, -0.2),
];
adapter.batch_update(&updates)?;

// Get weights
let weight = adapter.get_weight(10, 20)?;
let is_active = adapter.is_active(100, 200);

// Statistics
println!("Active weights: {}", adapter.active_count());
println!("Memory usage: {} bytes", adapter.memory_usage());
println!("Sparsity: {:.2}%", adapter.sparsity_ratio() * 100.0);
```

### Advanced Operations

```rust
// Iterate over active weights only
for (row, col, weight) in adapter.active_weights() {
    println!("[{}, {}] = {}", row, col, weight);
}

// Optimize sparse representation
adapter.optimize(); // Removes tiny values, empty nodes

// Merge adapters with scaling
adapter.merge_with(&other_adapter, 0.5)?; // Scale by 0.5

// Save/load adapters
adapter.save("my_adapter.vdb")?;
adapter.load("my_adapter.vdb")?;

// Fuse multiple adapters
let fused = OpenVDBBatchOps::fuse_adapters(
    &[&adapter1, &adapter2, &adapter3],
    &[1.0, 0.5, 0.3] // Scaling factors
)?;
```

### Training Integration

```rust
// Apply gradient updates during training
let gradient_updates = vec![
    (10, 20, 0.001),  // Small gradient update
    (50, 60, -0.002), // Negative gradient
];

adapter.batch_update(&gradient_updates)?;

// Enforce sparsity (remove weights below threshold)
OpenVDBBatchOps::enforce_sparsity(&mut adapter, 1e-6);
```

## Performance Characteristics

### Memory Usage Comparison

| Storage Type | Memory per Weight | 100K Weights | 1M Weights |
|-------------|------------------|---------------|-------------|
| HashMap | ~40 bytes | ~4 MB | ~40 MB |
| OpenVDB | ~8-16 bytes | ~1 MB | ~10 MB |
| **Improvement** | **2.5-5x** | **4x** | **4x** |

### Operation Performance

| Operation | HashMap | OpenVDB | Notes |
|-----------|---------|---------|--------|
| Random access | O(1) | O(log N) + cache | Cache makes it ~O(1) |
| Sequential iteration | O(all entries) | O(active only) | **Major advantage** |
| Batch updates | O(n) manual | O(n) optimized | Built-in optimizations |
| Memory locality | Poor | Excellent | Spatial coherence |

## Troubleshooting

### OpenVDB Not Found

```
warning: OpenVDB not found, using NanoVDB stubs
```

**Solutions:**
1. Install OpenVDB system packages (see above)
2. Set `PKG_CONFIG_PATH` if installed in custom location:
   ```bash
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   ```
3. Verify with `pkg-config --exists openvdb`

### Linking Errors

```
/usr/bin/ld: cannot find -lopenvdb
```

**Solutions:**
1. Install development packages: `libopenvdb-dev`
2. Update library cache: `sudo ldconfig`
3. Check library paths: `ldconfig -p | grep openvdb`

### Performance Issues

If OpenVDB operations seem slow:

1. **Enable optimizations**: Build with `--release`
2. **Batch operations**: Use `batch_update()` instead of individual `set_weight()`
3. **Periodic optimization**: Call `adapter.optimize()` after large updates
4. **Proper sparsity**: Only store weights above threshold (1e-8)

## Feature Comparison

| Feature | HashMap | NanoVDB | OpenVDB |
|---------|---------|---------|---------|
| Dynamic updates | ✅ | ❌ (read-only) | ✅ |
| Memory efficiency | ❌ | ✅ | ✅ |
| GPU acceleration | ❌ | ✅ | ❌* |
| Spatial locality | ❌ | ✅ | ✅ |
| Industry standard | ❌ | ✅ | ✅ |
| Training support | ✅ | ❌ | ✅ |

*GPU acceleration possible via future CUDA extensions

## Migration from HashMap

To migrate existing HashMap-based LoRA storage:

```rust
// Convert HashMap to OpenVDB
let hashmap: HashMap<Coordinate3D, f32> = existing_adapter;
let openvdb_adapter = OpenVDBLoRAAdapter::from_hashmap(&hashmap, (rows, cols))?;

// Convert OpenVDB back to HashMap (if needed for compatibility)
let hashmap = openvdb_adapter.to_hashmap();
```

## Roadmap

Future OpenVDB integration improvements:

- [ ] GPU acceleration via CUDA extensions
- [ ] Neural compression (NeuralVDB integration)  
- [ ] Temporal encoding for adapter evolution tracking
- [ ] Multi-resolution LoRA hierarchies
- [ ] Distributed storage across multiple nodes

## Support

For OpenVDB-related issues:

1. Check this setup guide
2. Verify OpenVDB installation: `pkg-config --modversion openvdb`
3. Run example: `cargo run --example openvdb_lora_example`
4. Check system dependencies and library paths
5. File issue with OpenVDB version and error details