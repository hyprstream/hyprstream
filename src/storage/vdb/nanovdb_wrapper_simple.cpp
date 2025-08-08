//! Simplified NanoVDB wrapper for CPU-only builds

#include "nanovdb_wrapper_simple.h"
#include <nanovdb/NanoVDB.h>
#include <iostream>
#include <memory>
#include <cstring>

using namespace nanovdb;

extern "C" {

// Simple stub implementations for CPU builds

NanoVDBGrid* nanovdb_create_float_grid(float background) {
    (void)background;
    std::cerr << "NanoVDB grid creation not yet implemented (using stub)" << std::endl;
    return nullptr;
}

void nanovdb_destroy_grid(NanoVDBGrid* grid) {
    (void)grid;
    // No-op
}

// Builder operations  
NanoVDBBuilder* nanovdb_create_builder(float background) {
    (void)background;
    std::cerr << "NanoVDB builder creation not yet implemented (using stub)" << std::endl;
    return nullptr;
}

void nanovdb_destroy_builder(NanoVDBBuilder* builder) {
    (void)builder;
    // No-op
}

void nanovdb_builder_set_value(NanoVDBBuilder* builder, NanoVDBCoord coord, float value) {
    (void)builder; (void)coord; (void)value;
    // No-op
}

void nanovdb_builder_set_value_on(NanoVDBBuilder* builder, NanoVDBCoord coord, float value) {
    (void)builder; (void)coord; (void)value;
    // No-op
}

void nanovdb_builder_set_value_off(NanoVDBBuilder* builder, NanoVDBCoord coord) {
    (void)builder; (void)coord;
    // No-op
}

NanoVDBGrid* nanovdb_builder_get_grid(NanoVDBBuilder* builder) {
    (void)builder;
    return nullptr;
}

// Grid queries
float nanovdb_grid_get_value(const NanoVDBGrid* grid, NanoVDBCoord coord) {
    (void)grid; (void)coord;
    return 0.0f;
}

bool nanovdb_grid_is_active(const NanoVDBGrid* grid, NanoVDBCoord coord) {
    (void)grid; (void)coord;
    return false;
}

uint64_t nanovdb_grid_active_voxel_count(const NanoVDBGrid* grid) {
    (void)grid;
    return 0;
}

uint64_t nanovdb_grid_memory_usage(const NanoVDBGrid* grid) {
    (void)grid;
    return 0;
}

// Grid iteration - return null iterator
NanoVDBIterator* nanovdb_grid_begin(const NanoVDBGrid* grid) {
    (void)grid;
    return nullptr;
}

void nanovdb_iterator_next(NanoVDBIterator* iter) {
    (void)iter;
}

void nanovdb_destroy_iterator(NanoVDBIterator* iter) {
    (void)iter;
}

// I/O operations
NanoVDBHandle* nanovdb_create_handle(const NanoVDBGrid* grid) {
    (void)grid;
    return nullptr;
}

void nanovdb_destroy_handle(NanoVDBHandle* handle) {
    (void)handle;
}

bool nanovdb_write_grid(const NanoVDBHandle* handle, const char* filename) {
    (void)handle; (void)filename;
    return false;
}

NanoVDBHandle* nanovdb_read_grid(const char* filename) {
    (void)filename;
    return nullptr;
}

// CUDA operations - stubs for CPU-only builds
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

CudaBuffer* nanovdb_grid_to_cuda(const NanoVDBGrid* grid) {
    (void)grid;
    return nullptr;
}

void nanovdb_destroy_cuda_buffer(CudaBuffer* buffer) {
    (void)buffer;
}

size_t nanovdb_cuda_buffer_size(const CudaBuffer* buffer) {
    (void)buffer;
    return 0;
}

bool nanovdb_cuda_is_available() {
    return false;
}

// Utility functions
NanoVDBCoord nanovdb_make_coord(int32_t x, int32_t y, int32_t z) {
    NanoVDBCoord coord;
    coord.x = x;
    coord.y = y;
    coord.z = z;
    return coord;
}

void nanovdb_grid_get_stats(const NanoVDBGrid* grid, NanoVDBStats* stats) {
    (void)grid;
    if (!stats) return;
    
    stats->active_voxels = 0;
    stats->memory_usage = 0;
    stats->sparsity = 1.0f;
    stats->tree_depth = 0;
    stats->leaf_nodes = 0;
    stats->internal_nodes = 0;
}

} // extern "C"