//! OpenVDB C++ bridge implementation

#include "openvdb_bridge.h"
#include <openvdb/tools/Prune.h>
#include <openvdb/tools/Composite.h>
#include <openvdb/tools/GridOperators.h>
#include <openvdb/io/File.h>
#include <iostream>
#include <algorithm>

namespace hyprstream {

LoRAGrid::LoRAGrid() 
    : grid_(openvdb::FloatGrid::create(0.0f))  // Background value = 0
    , access_count_(0)
    , accessor_dirty_(true) 
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

bool LoRAGrid::writeToFile(const std::string& filename) const {
    try {
        openvdb::io::File file(filename);
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

bool LoRAGrid::readFromFile(const std::string& filename) {
    try {
        openvdb::io::File file(filename);
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