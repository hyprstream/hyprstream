#include "engine.h"
#include <torch/cuda.h>
#include <iostream>

// Only include pybind11 if available
#ifdef HAS_PYTHON
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace hyprstream {

HyprStreamEngine::HyprStreamEngine(const std::string& model_path) 
    : device_(torch::kCPU), has_cuda_(false), triton_available_(false) {
    
    try {
        // Load model
        model_ = torch::jit::load(model_path);
        model_.eval();
        
        // Check for GPU availability (CUDA or ROCm)
        if (torch::cuda::is_available()) {
            device_ = torch::Device(torch::kCUDA);
            model_.to(device_);
            has_cuda_ = true;
            
            // Log GPU info
            std::cout << "🎮 GPU detected: " << (torch::cuda::device_count()) << " device(s)" << std::endl;
            
            // This works for both CUDA and ROCm
            auto props = torch::cuda::getDeviceProperties(0);
            std::cout << "   Device 0: " << props->name << std::endl;
            std::cout << "   Memory: " << (props->totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
        } else {
            std::cerr << "⚠️  No GPU detected, using CPU" << std::endl;
        }
        
        // Initialize Python for Triton (optional)
        try {
            init_triton();
        } catch (const std::exception& e) {
            std::cerr << "⚠️  Triton initialization failed: " << e.what() << std::endl;
            std::cerr << "   Dynamic LoRA updates will use fallback implementation" << std::endl;
        }
        
    } catch (const c10::Error& e) {
        std::cerr << "❌ Failed to load model: " << e.what() << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "❌ Engine initialization failed: " << e.what() << std::endl;
        throw;
    }
}
}

HyprStreamEngine::~HyprStreamEngine() {
    // Cleanup handled by smart pointers
}

float* HyprStreamEngine::forward(
    float* input_ids,
    size_t batch_size,
    size_t sequence_length
) {
    torch::NoGradGuard no_grad;
    
    // Wrap existing GPU memory - no copy
    auto input = torch::from_blob(
        input_ids,
        {static_cast<long>(batch_size), static_cast<long>(sequence_length)},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_)
    );
    
    // Forward pass
    std::vector<torch::jit::IValue> inputs{input};
    auto output = model_.forward(inputs).toTensor();
    
    // Return pointer to PyTorch's GPU memory - no copy
    return output.data_ptr<float>();
}

void HyprStreamEngine::update_lora_inplace(
    const std::string& layer_name,
    float* lora_a,
    float* lora_b,
    size_t rank
) {
    // Get the layer's weight tensor
    auto params = model_.named_parameters();
    
    for (auto& param : params) {
        if (param.name.find(layer_name) != std::string::npos) {
            auto weight = param.value;
            auto sizes = weight.sizes();
            
            if (sizes.size() != 2) {
                std::cerr << "⚠️  Expected 2D weight tensor, got " << sizes.size() << "D" << std::endl;
                continue;
            }
            
            auto out_features = sizes[0];
            auto in_features = sizes[1];
            
            // Wrap LoRA matrices - no copy
            auto a = torch::from_blob(
                lora_a,
                {static_cast<long>(rank), in_features},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_)
            );
            
            auto b = torch::from_blob(
                lora_b,
                {out_features, static_cast<long>(rank)},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_)
            );
            
            // Use Triton if available, otherwise fallback
            try {
                apply_triton_lora(weight, a, b);
                std::cout << "✅ LoRA update applied to " << layer_name << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "❌ LoRA update failed: " << e.what() << std::endl;
            }
            
            break;
        }
    }
}

torch::Tensor HyprStreamEngine::vdb_to_tensor(void* vdb_grid) {
    // Placeholder for OpenVDB integration
    // This would convert sparse VDB grid to dense tensor
    // This is the only necessary copy in the system
    return torch::zeros({1024, 1024}, torch::kFloat32).to(device_);
}

void HyprStreamEngine::init_triton() {
#ifdef HAS_PYTHON
    // Initialize Python interpreter
    python_guard_ = std::make_unique<py::scoped_interpreter>();
    
    try {
        // Import kernels module
        auto kernels = py::module_::import("kernels.lora");
        triton_module_ = std::make_unique<py::module_>(kernels);
        
        triton_available_ = true;
        std::cout << "✅ Triton kernels initialized" << std::endl;
        
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to import Triton kernels");
    }
#else
    throw std::runtime_error("Python support not compiled in");
#endif
}

void HyprStreamEngine::apply_triton_lora(
    torch::Tensor& weights,
    const torch::Tensor& lora_a,
    const torch::Tensor& lora_b
) {
    if (!triton_available_) {
        // Fallback to PyTorch implementation
        std::cout << "⚠️  Using PyTorch fallback for LoRA (Triton not available)" << std::endl;
        auto delta = torch::matmul(lora_b, lora_a);
        weights.add_(delta);
        return;
    }

#ifdef HAS_PYTHON
    try {
        py::gil_scoped_acquire acquire;
        
        // Get tensor pointers and shapes
        auto weight_ptr = reinterpret_cast<uintptr_t>(weights.data_ptr<float>());
        auto lora_a_ptr = reinterpret_cast<uintptr_t>(lora_a.data_ptr<float>());
        auto lora_b_ptr = reinterpret_cast<uintptr_t>(lora_b.data_ptr<float>());
        
        auto shape = weights.sizes().vec();
        int64_t m = shape[0];
        int64_t n = shape[1];
        int64_t rank = lora_a.size(0);
        
        // Call Triton kernel
        triton_module_->attr("apply_lora_from_cpp")(
            weight_ptr, lora_a_ptr, lora_b_ptr, m, n, rank
        );
        
    } catch (const py::error_already_set& e) {
        std::cerr << "Triton kernel error: " << e.what() << std::endl;
        // Fallback to PyTorch
        auto delta = torch::matmul(lora_b, lora_a);
        weights.add_(delta);
    }
#endif
}

} // namespace hyprstream

// C API implementation
extern "C" {

void* hyprstream_engine_create(const char* model_path) {
    try {
        return new hyprstream::HyprStreamEngine(model_path);
    } catch (...) {
        return nullptr;
    }
}

float* hyprstream_engine_forward(
    void* engine,
    float* input_ids,
    size_t batch_size,
    size_t sequence_length
) {
    if (!engine) return nullptr;
    
    auto* e = static_cast<hyprstream::HyprStreamEngine*>(engine);
    return e->forward(input_ids, batch_size, sequence_length);
}

void hyprstream_engine_update_lora(
    void* engine,
    const char* layer_name,
    float* lora_a,
    float* lora_b,
    size_t rank
) {
    if (!engine) return;
    
    auto* e = static_cast<hyprstream::HyprStreamEngine*>(engine);
    e->update_lora_inplace(layer_name, lora_a, lora_b, rank);
}

void hyprstream_engine_destroy(void* engine) {
    if (engine) {
        delete static_cast<hyprstream::HyprStreamEngine*>(engine);
    }
}

} // extern "C"