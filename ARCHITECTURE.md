# Hyprstream Architecture Diagram

## Executive Summary
This document outlines the target architecture for Hyprstream, identifying critical issues with the current implementation and providing a clear path to functional LLaMA.cpp inference integration.

## Current State Analysis

### ❌ Critical Issues Identified
1. **Mixed Concerns**: `src/inference/` contains FlightSQL transport code instead of ML logic
2. **Unused LLaMA.cpp**: Proper LLaMA.cpp integration exists in `src/runtime/` but isn't properly utilized
3. **API Confusion**: Transport protocols mixed with core inference logic
4. **Missing Integration**: No clear path from API requests to actual model inference

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  CLI Commands     │  REST Clients    │  FlightSQL Clients       │
│  hyprstream chat  │  curl/browser    │  DBeaver/BI tools        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        API TRANSPORT LAYER                      │
│                         src/api/                                │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────┐ │
│ │   REST API      │ │  FlightSQL      │ │    CLI Handlers     │ │
│ │   (Axum)        │ │  Service        │ │   (Clap)            │ │
│ │                 │ │                 │ │                     │ │
│ │ • Chat endpoint │ │ • SQL interface │ │ • model download    │ │
│ │ • OpenAI compat │ │ • Arrow data    │ │ • chat command      │ │
│ │ • Model mgmt    │ │ • Streaming     │ │ • server start      │ │
│ └─────────────────┘ └─────────────────┘ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE ORCHESTRATION                      │
│                        src/inference/                           │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                 InferenceEngine                             │ │
│ │                                                             │ │
│ │  • session_management()                                     │ │
│ │  • generate(prompt, params) -> InferenceResult             │ │
│ │  • load_model(path) -> ModelHandle                          │ │
│ │  • apply_lora_adapters(adapters)                            │ │
│ │  • stream_tokens() -> TokenStream                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │              LoRA Fusion Engine                             │ │
│ │                                                             │ │
│ │  • merge_adapters(base, adapters) -> FusedWeights          │ │
│ │  • dynamic_routing(input) -> AdapterSelection               │ │
│ │  • sparse_application(weights) -> OptimizedModel           │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RUNTIME EXECUTION                         │
│                        src/runtime/                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                 LlamaCppEngine                              │ │
│ │                    [KEEP - GOOD]                            │ │
│ │                                                             │ │
│ │  • llama_backend: LlamaBackend                              │ │
│ │  • llama_model: LlamaModel                                  │ │
│ │  • llama_context: LlamaContext                              │ │
│ │                                                             │ │
│ │  Methods:                                                   │ │
│ │  • load_model(path) -> Result<()>                           │ │
│ │  • tokenize(text) -> Vec<TokenId>                           │ │
│ │  • generate_tokens(prompt) -> TokenStream                   │ │
│ │  • apply_weights(lora_weights) -> Result<()>                │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       STORAGE BACKENDS                          │
│                        src/storage/                             │
├─────────────────────────────────────────────────────────────────┤
│ ┌───────────────┐ ┌───────────────┐ ┌─────────────────────────┐ │
│ │  Model Store  │ │  VDB Storage  │ │    LoRA Registry        │ │
│ │               │ │   [OPTIONAL]  │ │                         │ │
│ │ • GGUF files  │ │               │ │ • Adapter metadata      │ │
│ │ • SafeTensors │ │ • OpenVDB     │ │ • Sparse weights        │ │
│ │ • HF Download │ │ • Neural comp │ │ • Training history      │ │
│ └───────────────┘ └───────────────┘ └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow for Inference Request

```
1. CLIENT REQUEST
   │
   ▼
2. API LAYER (src/api/)
   │ Parse request (REST/FlightSQL/CLI)
   │ Validate parameters
   │ Extract session info
   ▼
3. INFERENCE ORCHESTRATION (src/inference/)
   │ Route to appropriate model
   │ Load LoRA adapters if specified
   │ Prepare generation parameters
   ▼
4. LORA FUSION (src/inference/)
   │ Merge base model + LoRA weights
   │ Apply sparse optimizations
   │ Create fused model state
   ▼
5. RUNTIME EXECUTION (src/runtime/)
   │ LlamaCppEngine.generate_tokens()
   │ Apply tokenization
   │ Run inference loop
   │ Stream tokens back
   ▼
6. RESPONSE STREAMING
   │ Format tokens (text/JSON/Arrow)
   │ Apply post-processing
   │ Return to client
```

## Required Changes by Module

### 🔥 URGENT FIXES NEEDED

#### src/inference/ - Complete Rewrite Required
**Current Problems:**
- `inference_service.rs` contains FlightSQL server code ❌
- Mixed transport concerns with ML logic ❌
- No actual LLaMA.cpp integration ❌

**Target Structure:**
```
src/inference/
├── mod.rs                    [REWRITE] - Clean module exports
├── inference_engine.rs       [REWRITE] - Core ML orchestration
├── lora_fusion.rs           [KEEP/ENHANCE] - LoRA weight merging
├── model_loader.rs          [ENHANCE] - Model loading utilities
├── session_manager.rs       [NEW] - Session state management
└── token_processor.rs       [NEW] - Tokenization utilities
```

**Key Methods Needed:**
```rust
impl InferenceEngine {
    async fn generate(&self, session_id: &str, prompt: &str, params: GenerationParams) -> Result<TokenStream>;
    async fn load_model(&mut self, model_path: &Path) -> Result<ModelHandle>;
    async fn create_session(&self, model_id: &str, lora_adapters: Vec<String>) -> Result<String>;
    async fn apply_lora_weights(&mut self, session_id: &str, adapters: &[LoRAAdapter]) -> Result<()>;
}
```

#### src/api/ - Move FlightSQL Here
**Required Actions:**
- Move `src/inference/inference_service.rs` → `src/api/flight_service.rs` ✅
- Keep all transport protocol code in API layer ✅
- API calls inference engine, not the other way around ✅

#### src/runtime/ - Good Foundation, Needs Enhancement
**Current State:** ✅ Good LLaMA.cpp integration exists
**Enhancements Needed:**
- Better error handling for model loading
- Support for LoRA weight application
- Context management improvements
- Memory optimization

### 🟡 MODERATE FIXES NEEDED

#### src/storage/ - Partial Cleanup
**VDB Features:**
- Keep VDB storage for advanced use cases ✅
- Make it optional and non-blocking ✅
- Fix any remaining compilation issues ✅

**Model Storage:**
- Enhance HuggingFace integration ✅
- Better model caching ✅
- GGUF format validation ✅

#### src/adapters/ - Enhancement
**Current State:** Basic structure exists ✅
**Needs:**
- Better LoRA loading from disk
- Runtime weight merging
- Memory-efficient sparse operations

### ✅ MODULES IN GOOD SHAPE

#### src/cli/ - Working Well
- Command structure is solid ✅
- Good integration with storage ✅
- Minor cleanup needed only ✅

#### src/models/ - Qwen3 Integration
- Good foundation for model-specific code ✅
- Can be extended for other model families ✅

## Implementation Priority

### Phase 1: Critical Path to Working Inference (Week 1)
1. **Rewrite `src/inference/inference_engine.rs`**
   - Create clean interface that uses `LlamaCppEngine`
   - Remove all FlightSQL/transport code
   - Focus on pure ML logic

2. **Move FlightSQL service to API layer**
   - `src/inference/inference_service.rs` → `src/api/flight_service.rs`
   - Update imports and dependencies
   - Ensure API calls inference engine

3. **Test basic inference pipeline**
   - Load GGUF model via LlamaCppEngine
   - Generate simple text responses
   - Verify end-to-end flow works

### Phase 2: LoRA Integration (Week 2)
1. **Enhance LoRA fusion engine**
   - Implement weight merging algorithms
   - Support multiple adapter loading
   - Memory-efficient operations

2. **Integrate with VDB storage**
   - Load LoRA weights from VDB
   - Cache merged weights
   - Optimize for real-time updates

### Phase 3: Production Ready (Week 3)
1. **Performance optimization**
   - Streaming responses
   - Context caching
   - Memory management

2. **API completeness**
   - OpenAI compatibility
   - Full FlightSQL support
   - Error handling

## Success Criteria

### ✅ Phase 1 Complete When:
- `hyprstream chat "Hello"` generates real LLaMA.cpp responses
- No FlightSQL code in `src/inference/`
- Clean separation between API and inference logic
- Basic model loading works end-to-end

### ✅ Phase 2 Complete When:
- LoRA adapters can be loaded and applied
- Multiple models can run simultaneously
- VDB storage integrates smoothly

### ✅ Phase 3 Complete When:
- Production-ready performance
- Full API compatibility
- Comprehensive error handling
- Documentation complete

## Key Architectural Principles

1. **Separation of Concerns**
   - API layer handles transport (REST, FlightSQL, CLI)
   - Inference layer handles ML logic only
   - Runtime layer handles model execution

2. **Clean Interfaces**
   - Each layer has well-defined public APIs
   - No leaking of internal implementation details
   - Easy to test and mock

3. **Scalability**
   - Support for multiple concurrent sessions
   - Efficient memory usage
   - Optional advanced features (VDB)

4. **Maintainability**
   - Clear module boundaries
   - Comprehensive error handling
   - Good documentation and testing

This architecture provides a clear path to functional LLaMA.cpp inference while maintaining the advanced VDB features that make Hyprstream unique.