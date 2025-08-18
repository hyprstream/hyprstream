//! OpenVDB C++ bridge implementation

#include "openvdb_bridge.h"
#include "rust/cxx.h"
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/io/File.h>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace hyprstream {

LoRAGrid::LoRAGrid() 
    : grid_(openvdb::FloatGrid::create(0.0f))  // Background value = 0
    , access_count_(0)
    , accessor_dirty_(true)
    , timestamp_ms_(0)
    , streaming_active_(false)
    , streaming_updates_(0)
{
    // Initialize OpenVDB
    openvdb::initialize();
    
    // Set grid metadata
    grid_->setName("LoRAAdapter");
    grid_->setGridClass(openvdb::GRID_FOG_VOLUME);  // Sparse data
    
    // Set transform (identity for matrix coordinates)
    grid_->setTransform(openvdb::math::Transform::createLinearTransform(1.0));
}

LoRAGrid::~LoRAGrid() = default;

openvdb::Coord LoRAGrid::to3D(int32_t row, int32_t col) const {
    // Map 2D matrix coordinates to 3D with Z=0
    // TODO: This constrains everything to a 2D plane, missing VDB's 3D advantages
    return openvdb::Coord(row, col, 0);
}

void LoRAGrid::setValue(int32_t row, int32_t col, float weight) {
    if (!accessor_ || accessor_dirty_) {
        accessor_ = std::make_unique<AccessorType>(grid_->getAccessor());
        accessor_dirty_ = false;
    }
    
    auto coord = to3D(row, col);
    
    if (std::abs(weight) > 1e-8f) {  // Sparsity threshold
        accessor_->setValue(coord, weight);
    } else {
        accessor_->setValueOff(coord);  // Remove from sparse storage
    }
    
    ++access_count_;
}

float LoRAGrid::getValue(int32_t row, int32_t col) const {
    if (!accessor_ || accessor_dirty_) {
        accessor_ = std::make_unique<AccessorType>(grid_->getAccessor());
        accessor_dirty_ = false;
    }
    
    auto coord = to3D(row, col);
    ++access_count_;
    
    return accessor_->getValue(coord);
}

bool LoRAGrid::isActive(int32_t row, int32_t col) const {
    if (!accessor_ || accessor_dirty_) {
        accessor_ = std::make_unique<AccessorType>(grid_->getAccessor());
        accessor_dirty_ = false;
    }
    
    auto coord = to3D(row, col);
    return accessor_->isValueOn(coord);
}

void LoRAGrid::setValueOff(int32_t row, int32_t col) {
    if (!accessor_ || accessor_dirty_) {
        accessor_ = std::make_unique<AccessorType>(grid_->getAccessor());
        accessor_dirty_ = false;
    }
    
    auto coord = to3D(row, col);
    accessor_->setValueOff(coord);
}


void LoRAGrid::sparseFill(int32_t min_row, int32_t min_col, int32_t max_row, int32_t max_col, float value) {
    openvdb::CoordBBox bbox(
        openvdb::Coord(min_row, min_col, 0),
        openvdb::Coord(max_row, max_col, 0)
    );
    
    grid_->sparseFill(bbox, value, /*active=*/true);
    accessor_dirty_ = true;
}

size_t LoRAGrid::activeVoxelCount() const {
    return grid_->activeVoxelCount();
}

size_t LoRAGrid::memoryUsage() const {
    return grid_->memUsage();
}

float LoRAGrid::sparsityRatio() const {
    auto active = activeVoxelCount();
    if (active == 0) return 1.0f;
    
    // Calculate total possible voxels from bounding box
    auto bbox = grid_->evalActiveVoxelBoundingBox();
    if (!bbox.empty()) {
        auto total = bbox.volume();
        return 1.0f - (static_cast<float>(active) / static_cast<float>(total));
    }
    
    return 0.99f;  // Default high sparsity for LoRA
}

void LoRAGrid::prune(float tolerance) {
    openvdb::tools::prune(grid_->tree(), tolerance);
    accessor_dirty_ = true;
}

void LoRAGrid::merge(const LoRAGrid& other, float scale) {
    if (scale == 1.0f) {
        // Simple union
        grid_->tree().merge(other.grid_->tree());
    } else {
        // Scaled merge - multiply other grid by scale first
        auto scaled_grid = other.grid_->deepCopy();
        openvdb::tools::foreach(scaled_grid->beginValueOn(), 
            [scale](const auto& iter) {
                auto value = iter.getValue();
                const_cast<decltype(iter)&>(iter).setValue(value * scale);
            });
        
        grid_->tree().merge(scaled_grid->tree());
    }
    
    accessor_dirty_ = true;
}

bool LoRAGrid::writeToFile(rust::Str filename) const {
    try {
        std::string filename_str(filename);
        openvdb::io::File file(filename_str);
        openvdb::GridPtrVec grids;
        grids.push_back(grid_);
        file.write(grids);
        file.close();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing LoRA grid: " << e.what() << std::endl;
        return false;
    }
}

bool LoRAGrid::readFromFile(rust::Str filename) {
    try {
        std::string filename_str(filename);
        openvdb::io::File file(filename_str);
        file.open();
        
        // Read first grid from file
        openvdb::GridBase::Ptr baseGrid = file.readGrid(file.beginName().gridName());
        if (baseGrid) {
            grid_ = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
            accessor_dirty_ = true;
            file.close();
            return true;
        }
        
        file.close();
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Error reading LoRA grid: " << e.what() << std::endl;
        return false;
    }
}

// ActiveWeightIterator implementation (temporarily disabled)
// ActiveWeightIterator::ActiveWeightIterator(const LoRAGrid& grid) 
//     : grid_(grid)
//     , iter_(std::make_unique<openvdb::FloatGrid::ValueOnCIter>(grid.getGrid()->cbeginValueOn())) 
// {
// }

// ActiveWeightIterator::~ActiveWeightIterator() = default;

// bool ActiveWeightIterator::hasNext() const {
//     return iter_ && *iter_;
// }

// ::hyprstream::WeightData ActiveWeightIterator::next() {
//     if (!iter_ || !*iter_) {
//         return {0, 0, 0.0f};
//     }
    
//     auto coord = iter_->getCoord();
//     auto weight = iter_->getValue();
    
//     ::hyprstream::WeightData data{coord.x(), coord.y(), weight};
    
//     ++(*iter_);
//     return data;
// }

// void ActiveWeightIterator::reset() {
//     iter_ = std::make_unique<openvdb::FloatGrid::ValueOnCIter>(grid_.getGrid()->cbeginValueOn());
// }

// SparseBatchOps implementation

void SparseBatchOps::applyDelta(LoRAGrid& base, const LoRAGrid& delta, float learning_rate) {
    // Apply scaled delta: base = base + learning_rate * delta
    base.merge(delta, learning_rate);
    base.prune(1e-8f);
}

void SparseBatchOps::optimizeSparsity(LoRAGrid& grid, float sparsity_threshold) {
    // Remove values below threshold
    auto grid_ptr = grid.getGrid();
    
    std::vector<openvdb::Coord> to_remove;
    for (auto iter = grid_ptr->beginValueOn(); iter; ++iter) {
        if (std::abs(iter.getValue()) < sparsity_threshold) {
            to_remove.push_back(iter.getCoord());
        }
    }
    
    auto accessor = grid_ptr->getAccessor();
    for (const auto& coord : to_remove) {
        accessor.setValueOff(coord);
    }
    
    grid.prune();
}

// NEW: Temporal streaming operations implementation

void LoRAGrid::setTimestamp(int64_t timestamp_ms) {
    timestamp_ms_ = timestamp_ms;
    
    // Store timestamp in grid metadata
    grid_->insertMeta("timestamp", openvdb::Int64Metadata(timestamp_ms));
}

int64_t LoRAGrid::getTimestamp() const {
    return timestamp_ms_;
}

std::unique_ptr<LoRAGrid> LoRAGrid::createTemporalSnapshot() const {
    auto snapshot = std::make_unique<LoRAGrid>();
    
    // Deep copy the grid
    snapshot->grid_ = grid_->deepCopy();
    snapshot->timestamp_ms_ = timestamp_ms_;
    
    return snapshot;
}

std::unique_ptr<LoRAGrid> LoRAGrid::interpolateWeights(const LoRAGrid& other, float alpha) const {
    auto result = std::make_unique<LoRAGrid>();
    
    // Linear interpolation: result = (1-alpha) * this + alpha * other
    // First copy this grid scaled by (1-alpha)
    result->grid_ = grid_->deepCopy();
    
    openvdb::tools::foreach(result->grid_->beginValueOn(), 
        [alpha](const auto& iter) {
            auto value = iter.getValue();
            const_cast<decltype(iter)&>(iter).setValue(value * (1.0f - alpha));
        });
    
    // Then add the other grid scaled by alpha
    auto scaled_other = other.grid_->deepCopy();
    openvdb::tools::foreach(scaled_other->beginValueOn(), 
        [alpha](const auto& iter) {
            auto value = iter.getValue();
            const_cast<decltype(iter)&>(iter).setValue(value * alpha);
        });
    
    result->grid_->tree().merge(scaled_other->tree());
    
    // Set interpolated timestamp
    result->timestamp_ms_ = static_cast<int64_t>(
        (1.0f - alpha) * timestamp_ms_ + alpha * other.timestamp_ms_
    );
    
    return result;
}

void LoRAGrid::beginStreamingUpdate() {
    streaming_active_ = true;
    streaming_updates_ = 0;
    
    // Pre-allocate accessor for performance
    if (!accessor_ || accessor_dirty_) {
        accessor_ = std::make_unique<AccessorType>(grid_->getAccessor());
        accessor_dirty_ = false;
    }
}

bool LoRAGrid::endStreamingUpdate() {
    streaming_active_ = false;
    
    // Prune small values after streaming updates
    if (streaming_updates_ > 0) {
        prune(1e-8f);
        accessor_dirty_ = true;
        
        std::cout << "Streaming update completed: " << streaming_updates_ 
                  << " updates, active voxels: " << activeVoxelCount() << std::endl;
    }
    
    return true;
}

void LoRAGrid::streamingSetValue(int32_t row, int32_t col, float weight, int64_t timestamp_ms) {
    // Update timestamp
    timestamp_ms_ = timestamp_ms;
    
    // Set value using cached accessor for performance
    if (!accessor_ || accessor_dirty_) {
        accessor_ = std::make_unique<AccessorType>(grid_->getAccessor());
        accessor_dirty_ = false;
    }
    
    auto coord = to3D(row, col);
    
    if (std::abs(weight) > 1e-8f) {
        accessor_->setValue(coord, weight);
    } else {
        accessor_->setValueOff(coord);
    }
    
    ++streaming_updates_;
    ++access_count_;
}

float LoRAGrid::computeGradientMagnitude() const {
    float magnitude_squared = 0.0f;
    size_t count = 0;
    
    // Compute L2 norm of all active values
    for (auto iter = grid_->cbeginValueOn(); iter; ++iter) {
        float value = iter.getValue();
        magnitude_squared += value * value;
        ++count;
    }
    
    return count > 0 ? std::sqrt(magnitude_squared) : 0.0f;
}

std::unique_ptr<LoRAGrid> LoRAGrid::computeGradientDifference(const LoRAGrid& other) const {
    auto gradient = std::make_unique<LoRAGrid>();
    
    // Create union of active coordinates from both grids
    auto result_grid = grid_->deepCopy();
    
    // Subtract other grid: gradient = this - other
    for (auto iter = other.grid_->cbeginValueOn(); iter; ++iter) {
        auto coord = iter.getCoord();
        auto other_value = iter.getValue();
        auto this_value = result_grid->tree().getValue(coord);
        
        float diff = this_value - other_value;
        if (std::abs(diff) > 1e-8f) {
            result_grid->tree().setValue(coord, diff);
        } else {
            result_grid->tree().setValueOff(coord);
        }
    }
    
    // Also handle coordinates only in this grid
    for (auto iter = grid_->cbeginValueOn(); iter; ++iter) {
        auto coord = iter.getCoord();
        if (!other.grid_->tree().isValueOn(coord)) {
            // This coordinate exists only in this grid
            result_grid->tree().setValue(coord, iter.getValue());
        }
    }
    
    gradient->grid_ = result_grid;
    gradient->prune(1e-8f);
    
    return gradient;
}

void LoRAGrid::applyGradientUpdate(const LoRAGrid& gradient, float learning_rate) {
    // Apply: this = this + learning_rate * gradient
    merge(gradient, learning_rate);
    
    // Update timestamp to current time
    auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    timestamp_ms_ = now;
}

// Free functions for CXX bridge compatibility
std::unique_ptr<LoRAGrid> createLoRAGrid() {
    return std::make_unique<LoRAGrid>();
}

// std::unique_ptr<ActiveWeightIterator> createIterator(const LoRAGrid& grid) {
//     return std::make_unique<ActiveWeightIterator>(grid);
// }

void applyDelta(LoRAGrid& base, const LoRAGrid& delta, float learning_rate) {
    SparseBatchOps::applyDelta(base, delta, learning_rate);
}

void optimizeSparsity(LoRAGrid& grid, float sparsity_threshold) {
    SparseBatchOps::optimizeSparsity(grid, sparsity_threshold);
}

} // namespace hyprstream