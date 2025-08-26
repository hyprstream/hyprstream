#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <memory>
#include <string>

// Forward declare pybind11 types to avoid header dependency
namespace pybind11 {
    class scoped_interpreter;
    class module_;
}

namespace hyprstream {

class HyprStreamEngine {
public:
    explicit HyprStreamEngine(const std::string& model_path);
    ~HyprStreamEngine();
    
    // Zero-copy forward pass - returns pointer to PyTorch memory
    float* forward(
        float* input_ids,      // GPU memory pointer
        size_t batch_size,
        size_t sequence_length
    );
    
    // In-place LoRA weight update
    void update_lora_inplace(
        const std::string& layer_name,
        float* lora_a,
        float* lora_b,
        size_t rank
    );
    
    // OpenVDB integration (if needed)
    torch::Tensor vdb_to_tensor(void* vdb_grid);
    
    // Triton kernel integration
    void apply_triton_lora(
        torch::Tensor& weights,
        const torch::Tensor& lora_a,
        const torch::Tensor& lora_b
    );
    
private:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool has_cuda_;
    
    // Python interpreter for Triton
    std::unique_ptr<pybind11::scoped_interpreter> python_guard_;
    std::unique_ptr<pybind11::module_> triton_module_;
    bool triton_available_;
    
    // Internal methods
    void init_triton();
};

} // namespace hyprstream

// C API for Rust FFI
extern "C" {
    void* hyprstream_engine_create(const char* model_path);
    
    float* hyprstream_engine_forward(
        void* engine,
        float* input_ids,
        size_t batch_size,
        size_t sequence_length
    );
    
    void hyprstream_engine_update_lora(
        void* engine,
        const char* layer_name,
        float* lora_a,
        float* lora_b,
        size_t rank
    );
    
    void hyprstream_engine_destroy(void* engine);
}