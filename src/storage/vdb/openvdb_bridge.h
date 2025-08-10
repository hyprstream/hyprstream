//! OpenVDB C++ bridge header for Rust integration

#pragma once

#include <openvdb/openvdb.h>
#include <memory>
#include <vector>
#include "rust/cxx.h"

namespace hyprstream {

/// Sparse LoRA adapter using OpenVDB
class LoRAGrid {
public:
    using GridType = openvdb::FloatGrid;
    using AccessorType = GridType::Accessor;
    
    LoRAGrid();
    ~LoRAGrid();
    
    // Core sparse operations
    void setValue(int32_t row, int32_t col, float weight);
    float getValue(int32_t row, int32_t col) const;
    bool isActive(int32_t row, int32_t col) const;
    void setValueOff(int32_t row, int32_t col);
    
    // Batch operations
    void sparseFill(int32_t min_row, int32_t min_col, int32_t max_row, int32_t max_col, float value);
    
    // Iteration and analysis
    size_t activeVoxelCount() const;
    size_t memoryUsage() const;
    float sparsityRatio() const;
    
    // Optimization
    void prune(float tolerance = 0.0f);
    void merge(const LoRAGrid& other, float scale = 1.0f);
    
    // I/O operations
    bool writeToFile(rust::Str filename) const;
    bool readFromFile(rust::Str filename);
    
    // Internal grid access
    GridType::Ptr getGrid() const { return grid_; }
    
private:
    GridType::Ptr grid_;
    mutable std::unique_ptr<AccessorType> accessor_;
    
    // Convert 2D matrix coordinates to 3D OpenVDB coordinates
    openvdb::Coord to3D(int32_t row, int32_t col) const;
    
    // Performance tracking
    mutable size_t access_count_;
    mutable bool accessor_dirty_;
};

// /// Iterator for active (non-zero) weights (temporarily disabled)
// class ActiveWeightIterator {
// public:
//     explicit ActiveWeightIterator(const LoRAGrid& grid);
//     ~ActiveWeightIterator();
//     
//     bool hasNext() const;
//     ::hyprstream::WeightData next();
//     void reset();
//     
// private:
//     const LoRAGrid& grid_;
//     std::unique_ptr<openvdb::FloatGrid::ValueOnCIter> iter_;
// };

/// Batch sparse operations utilities
class SparseBatchOps {
public:
    // Apply sparse updates with automatic pruning
    static void applyDelta(LoRAGrid& base, const LoRAGrid& delta, float learning_rate);
    
    // Compress sparse representation
    static void optimizeSparsity(LoRAGrid& grid, float sparsity_threshold);
};

// Free functions for CXX bridge compatibility
std::unique_ptr<LoRAGrid> createLoRAGrid();
// std::unique_ptr<ActiveWeightIterator> createIterator(const LoRAGrid& grid);  // Temporarily disabled
void applyDelta(LoRAGrid& base, const LoRAGrid& delta, float learning_rate);
void optimizeSparsity(LoRAGrid& grid, float sparsity_threshold);

} // namespace hyprstream