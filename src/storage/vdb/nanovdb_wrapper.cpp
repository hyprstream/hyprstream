// NanoVDB C++ wrapper implementation for Rust FFI

#include "nanovdb_wrapper.h"
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/io/IO.h>

#ifdef NANOVDB_USE_CUDA
#include <nanovdb/cuda/DeviceBuffer.h>
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

using namespace nanovdb;

// Internal wrapper structs
struct NanoVDBGrid {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    const nanovdb::FloatGrid* grid;
    
    NanoVDBGrid(nanovdb::GridHandle<nanovdb::HostBuffer>&& h) : handle(std::move(h)) {
        grid = handle.grid<float>();
    }
};

struct NanoVDBBuilder {
    std::unique_ptr<nanovdb::tools::GridBuilder<float>> builder;
    
    NanoVDBBuilder(float background) {
        builder = std::make_unique<nanovdb::tools::GridBuilder<float>>(background);
    }
};

struct NanoVDBHandle {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    
    NanoVDBHandle(nanovdb::GridHandle<nanovdb::HostBuffer>&& h) : handle(std::move(h)) {}
};

#ifdef NANOVDB_USE_CUDA
struct CudaBuffer {
    nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer> handle;
    
    CudaBuffer(nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>&& h) : handle(std::move(h)) {}
};
#endif

extern "C" {

// Grid operations
NanoVDBGrid* nanovdb_create_float_grid(float background) {
    try {
        auto builder = nanovdb::tools::GridBuilder<float>(background);
        auto handle = builder.getHandle<nanovdb::HostBuffer>();
        return new NanoVDBGrid(std::move(handle));
    } catch (const std::exception& e) {
        std::cerr << "Error creating grid: " << e.what() << std::endl;
        return nullptr;
    }
}

void nanovdb_destroy_grid(NanoVDBGrid* grid) {
    delete grid;
}

// Builder operations
NanoVDBBuilder* nanovdb_create_builder(float background) {
    try {
        return new NanoVDBBuilder(background);
    } catch (const std::exception& e) {
        std::cerr << "Error creating builder: " << e.what() << std::endl;
        return nullptr;
    }
}

void nanovdb_destroy_builder(NanoVDBBuilder* builder) {
    delete builder;
}

void nanovdb_builder_set_value(NanoVDBBuilder* builder, NanoVDBCoord coord, float value) {
    if (!builder || !builder->builder) return;
    
    nanovdb::Coord c(coord.x, coord.y, coord.z);
    builder->builder->setValue(c, value);
}

void nanovdb_builder_set_value_on(NanoVDBBuilder* builder, NanoVDBCoord coord, float value) {
    if (!builder || !builder->builder) return;
    
    Coord c(coord.x, coord.y, coord.z);
    builder->builder->setValueOn(c, value);
}

void nanovdb_builder_set_value_off(NanoVDBBuilder* builder, NanoVDBCoord coord) {
    if (!builder || !builder->builder) return;
    
    Coord c(coord.x, coord.y, coord.z);
    builder->builder->setValueOff(c);
}

NanoVDBGrid* nanovdb_builder_get_grid(NanoVDBBuilder* builder) {
    if (!builder || !builder->builder) return nullptr;
    
    try {
        auto handle = builder->builder->getHandle<HostBuffer>();
        return new NanoVDBGrid(std::move(handle));
    } catch (const std::exception& e) {
        std::cerr << "Error getting grid from builder: " << e.what() << std::endl;
        return nullptr;
    }
}

// Grid queries  
float nanovdb_grid_get_value(const NanoVDBGrid* grid, NanoVDBCoord coord) {
    if (!grid || !grid->grid) return 0.0f;
    
    Coord c(coord.x, coord.y, coord.z);
    auto acc = grid->grid->getAccessor();
    return acc.getValue(c);
}

bool nanovdb_grid_is_active(const NanoVDBGrid* grid, NanoVDBCoord coord) {
    if (!grid || !grid->grid) return false;
    
    Coord c(coord.x, coord.y, coord.z);
    auto acc = grid->grid->getAccessor();
    return acc.isActive(c);
}

uint64_t nanovdb_grid_active_voxel_count(const NanoVDBGrid* grid) {
    if (!grid || !grid->grid) return 0;
    
    return grid->grid->activeVoxelCount();
}

uint64_t nanovdb_grid_memory_usage(const NanoVDBGrid* grid) {
    if (!grid || !grid->grid) return 0;
    
    return grid->grid->memUsage();
}

// Grid iteration
struct NanoVDBIterator {
    std::unique_ptr<FloatGrid::ValueIterator> iter;
    NanoVDBCoord coord;
    float value;
    bool valid;
};

NanoVDBIterator* nanovdb_grid_begin(const NanoVDBGrid* grid) {
    if (!grid || !grid->grid) return nullptr;
    
    auto iterator = std::make_unique<NanoVDBIterator>();
    iterator->iter = std::make_unique<FloatGrid::ValueIterator>(grid->grid->beginValueAll());
    
    if (iterator->iter && *iterator->iter) {
        auto coord = iterator->iter->getCoord();
        iterator->coord = {coord.x(), coord.y(), coord.z()};
        iterator->value = iterator->iter->getValue();
        iterator->valid = true;
    } else {
        iterator->valid = false;
    }
    
    return iterator.release();
}

void nanovdb_iterator_next(NanoVDBIterator* iter) {
    if (!iter || !iter->iter || !iter->valid) return;
    
    ++(*iter->iter);
    
    if (*iter->iter) {
        auto coord = iter->iter->getCoord();
        iter->coord = {coord.x(), coord.y(), coord.z()};
        iter->value = iter->iter->getValue();
        iter->valid = true;
    } else {
        iter->valid = false;
    }
}

void nanovdb_destroy_iterator(NanoVDBIterator* iter) {
    delete iter;
}

// I/O operations
NanoVDBHandle* nanovdb_create_handle(const NanoVDBGrid* grid) {
    if (!grid) return nullptr;
    
    // Create a copy of the handle
    auto handle_copy = grid->handle;
    return new NanoVDBHandle(std::move(handle_copy));
}

void nanovdb_destroy_handle(NanoVDBHandle* handle) {
    delete handle;
}

bool nanovdb_write_grid(const NanoVDBHandle* handle, const char* filename) {
    if (!handle || !filename) return false;
    
    try {
        io::writeGrid(filename, handle->handle);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing grid: " << e.what() << std::endl;
        return false;
    }
}

NanoVDBHandle* nanovdb_read_grid(const char* filename) {
    if (!filename) return nullptr;
    
    try {
        auto handle = io::readGrid<HostBuffer>(filename);
        return new NanoVDBHandle(std::move(handle));
    } catch (const std::exception& e) {
        std::cerr << "Error reading grid: " << e.what() << std::endl;
        return nullptr;
    }
}

#ifdef NANOVDB_USE_CUDA
// CUDA operations
CudaBuffer* nanovdb_grid_to_cuda(const NanoVDBGrid* grid) {
    if (!grid) return nullptr;
    
    try {
        auto cuda_handle = grid->handle.template deviceGrid<CudaDeviceBuffer>();
        return new CudaBuffer(std::move(cuda_handle));
    } catch (const std::exception& e) {
        std::cerr << "Error converting to CUDA: " << e.what() << std::endl;
        return nullptr;
    }
}

void nanovdb_destroy_cuda_buffer(CudaBuffer* buffer) {
    delete buffer;
}

// CUDA operations only available when CUDA is enabled
#ifdef NANOVDB_USE_CUDA

// CUDA kernel for sparse updates
__global__ void sparse_update_kernel(
    nanovdb::NanoGrid<float>* grid,
    const uint32_t* indices,
    const float* values,
    uint32_t count
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    
    // Convert linear index to 3D coordinate (assuming specific layout)
    uint32_t idx = indices[tid];
    int32_t z = 0; // 2D for now
    int32_t y = idx / 1536; // Qwen3 hidden dimension
    int32_t x = idx % 1536;
    
    nanovdb::Coord coord(x, y, z);
    auto acc = grid->getAccessor();
    acc.setValue(coord, values[tid]);
}

void nanovdb_cuda_sparse_update(
    CudaBuffer* buffer,
    const uint32_t* indices,
    const float* values, 
    uint32_t count
) {
    if (!buffer || count == 0) return;
    
    auto* d_grid = buffer->handle.deviceGrid<float>();
    if (!d_grid) return;
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);
    
    sparse_update_kernel<<<grid, block>>>(
        const_cast<nanovdb::NanoGrid<float>*>(d_grid),
        indices, 
        values,
        count
    );
    
    cudaDeviceSynchronize();
}

// CUDA kernel for sparse matrix multiplication
__global__ void sparse_multiply_kernel(
    const nanovdb::NanoGrid<float>* grid,
    const float* input,
    float* output,
    uint32_t input_size,
    uint32_t output_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= output_size) return;
    
    float sum = 0.0f;
    auto acc = grid->getAccessor();
    
    // Simplified sparse multiplication for demonstration
    for (uint32_t i = 0; i < input_size; ++i) {
        nanovdb::Coord coord(i, tid, 0);
        if (acc.isActive(coord)) {
            sum += acc.getValue(coord) * input[i];
        }
    }
    
    output[tid] = sum;
}

void nanovdb_cuda_sparse_multiply(
    const CudaBuffer* buffer,
    const float* input,
    float* output,
    uint32_t input_size, 
    uint32_t output_size
) {
    if (!buffer) return;
    
    auto* d_grid = buffer->handle.deviceGrid<float>();
    if (!d_grid) return;
    
    dim3 block(256);
    dim3 grid((output_size + block.x - 1) / block.x);
    
    sparse_multiply_kernel<<<grid, block>>>(
        d_grid,
        input,
        output,
        input_size,
        output_size
    );
    
    cudaDeviceSynchronize();
}

void nanovdb_cuda_batch_update(
    CudaBuffer* buffer,
    const NanoVDBCoord* coords,
    const float* values,
    uint32_t count
) {
    // Implementation for batch coordinate updates
    if (!buffer || count == 0) return;
    
    auto* d_grid = buffer->handle.deviceGrid<float>();
    if (!d_grid) return;
    
    // Convert coords to device memory and launch kernel
    // Simplified for demonstration - would need proper implementation
}

size_t nanovdb_cuda_buffer_size(const CudaBuffer* buffer) {
    if (!buffer) return 0;
    return buffer->handle.bufferSize();  // Use bufferSize instead of deprecated size()
}

bool nanovdb_cuda_is_available() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

#else

// Stub implementations for CPU-only builds
void nanovdb_cuda_sparse_update(
    CudaBuffer* buffer,
    const uint32_t* indices,
    const float* values, 
    uint32_t count
) {
    (void)buffer; (void)indices; (void)values; (void)count;
    // No-op in CPU-only build
}

void nanovdb_cuda_sparse_multiply(
    const CudaBuffer* buffer,
    const float* input,
    float* output,
    uint32_t input_size, 
    uint32_t output_size
) {
    (void)buffer; (void)input; (void)output; (void)input_size; (void)output_size;
    // No-op in CPU-only build
}

void nanovdb_cuda_batch_update(
    CudaBuffer* buffer,
    const NanoVDBCoord* coords,
    const float* values,
    uint32_t count
) {
    (void)buffer; (void)coords; (void)values; (void)count;
    // No-op in CPU-only build
}

size_t nanovdb_cuda_buffer_size(const CudaBuffer* buffer) {
    (void)buffer;
    return 0;
}

bool nanovdb_cuda_is_available() {
    return false;
}

#endif // NANOVDB_USE_CUDA
#endif

// Utility functions
NanoVDBCoord nanovdb_make_coord(int32_t x, int32_t y, int32_t z) {
    return {x, y, z};
}

void nanovdb_grid_get_stats(const NanoVDBGrid* grid, NanoVDBStats* stats) {
    if (!grid || !grid->grid || !stats) return;
    
    const auto& tree = grid->grid->tree();
    
    stats->active_voxels = grid->grid->activeVoxelCount();
    stats->memory_usage = grid->grid->memUsage();
    
    uint64_t total_voxels = stats->active_voxels; // Simplified
    if (total_voxels > 0) {
        stats->sparsity = 1.0f - (float)stats->active_voxels / (float)total_voxels;
    } else {
        stats->sparsity = 1.0f;
    }
    
    stats->tree_depth = 3; // NanoVDB standard depth
    stats->leaf_nodes = tree.nodeCount(0);
    stats->internal_nodes = tree.nodeCount(1) + tree.nodeCount(2);
}

} // extern "C"