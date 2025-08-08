//! Simplified NanoVDB wrapper header for CPU-only builds
#ifndef NANOVDB_WRAPPER_SIMPLE_H
#define NANOVDB_WRAPPER_SIMPLE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Core types - simplified forward declarations
typedef struct NanoVDBGrid NanoVDBGrid;
typedef struct NanoVDBBuilder NanoVDBBuilder;
typedef struct NanoVDBCoord {
    int32_t x, y, z;
} NanoVDBCoord;

typedef struct NanoVDBHandle NanoVDBHandle;
typedef struct CudaBuffer CudaBuffer;  // Stub for compatibility

// Grid operations
NanoVDBGrid* nanovdb_create_float_grid(float background);
void nanovdb_destroy_grid(NanoVDBGrid* grid);

// Builder operations  
NanoVDBBuilder* nanovdb_create_builder(float background);
void nanovdb_destroy_builder(NanoVDBBuilder* builder);

void nanovdb_builder_set_value(NanoVDBBuilder* builder, NanoVDBCoord coord, float value);
void nanovdb_builder_set_value_on(NanoVDBBuilder* builder, NanoVDBCoord coord, float value);
void nanovdb_builder_set_value_off(NanoVDBBuilder* builder, NanoVDBCoord coord);

NanoVDBGrid* nanovdb_builder_get_grid(NanoVDBBuilder* builder);

// Grid queries
float nanovdb_grid_get_value(const NanoVDBGrid* grid, NanoVDBCoord coord);
bool nanovdb_grid_is_active(const NanoVDBGrid* grid, NanoVDBCoord coord);
uint64_t nanovdb_grid_active_voxel_count(const NanoVDBGrid* grid);
uint64_t nanovdb_grid_memory_usage(const NanoVDBGrid* grid);

// Grid iteration
typedef struct NanoVDBIterator {
    void* internal;
    NanoVDBCoord coord;
    float value;
    bool valid;
} NanoVDBIterator;

NanoVDBIterator* nanovdb_grid_begin(const NanoVDBGrid* grid);
void nanovdb_iterator_next(NanoVDBIterator* iter);
void nanovdb_destroy_iterator(NanoVDBIterator* iter);

// I/O operations
NanoVDBHandle* nanovdb_create_handle(const NanoVDBGrid* grid);
void nanovdb_destroy_handle(NanoVDBHandle* handle);

bool nanovdb_write_grid(const NanoVDBHandle* handle, const char* filename);
NanoVDBHandle* nanovdb_read_grid(const char* filename);

// CUDA operations - stubs for CPU-only
CudaBuffer* nanovdb_grid_to_cuda(const NanoVDBGrid* grid);
void nanovdb_destroy_cuda_buffer(CudaBuffer* buffer);

void nanovdb_cuda_sparse_update(
    CudaBuffer* buffer,
    const uint32_t* indices, 
    const float* values,
    uint32_t count
);

void nanovdb_cuda_sparse_multiply(
    const CudaBuffer* buffer,
    const float* input,
    float* output, 
    uint32_t input_size,
    uint32_t output_size
);

void nanovdb_cuda_batch_update(
    CudaBuffer* buffer,
    const NanoVDBCoord* coords,
    const float* values,
    uint32_t count
);

// GPU memory info
size_t nanovdb_cuda_buffer_size(const CudaBuffer* buffer);
bool nanovdb_cuda_is_available();

// Utility functions
NanoVDBCoord nanovdb_make_coord(int32_t x, int32_t y, int32_t z);

// Statistics
typedef struct NanoVDBStats {
    uint64_t active_voxels;
    uint64_t memory_usage;
    float sparsity;
    uint32_t tree_depth;
    uint64_t leaf_nodes;
    uint64_t internal_nodes;
} NanoVDBStats;

void nanovdb_grid_get_stats(const NanoVDBGrid* grid, NanoVDBStats* stats);

#ifdef __cplusplus
}
#endif

#endif // NANOVDB_WRAPPER_SIMPLE_H