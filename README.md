# HyprStream: LLM Inference Engine with Git-based Model Management

[![Rust](https://github.com/hyprstream/hyprstream/actions/workflows/rust.yml/badge.svg)](https://github.com/hyprstream/hyprstream/actions/workflows/rust.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE-AGPLV3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)

## Overview

HyprStream is a LLM inference and training engine built in Rust with PyTorch, featuring integrated training capabilities and Git-based model version control. It provides a unified platform for model inference, fine-tuning through LoRA adapters, and comprehensive model lifecycle management.

### Core Features

- **Inference API**: Providing compatibility with OpenAI's OpenAPI specification.
- **Management API**: Providing management APIs (WIP)
- **High-Performance Inference**: PyTorch-based engine with KV caching and optimized memory management
- **Hardware Acceleration**: CPU (default), NVIDIA GPU (CUDA), and AMD GPU (ROCm) support
- **LoRA Training & Adaptation**: Create, train, and deploy LoRA adapters for model customization
- **Git-based Model Management**: Version control for models using native Git repositories
- **Hugging Face Compatible**: Direct cloning and usage of models from Hugging Face Hub
- **Efficient Storage**: XET integration for lazy loading and deduplication of large model files
- **Multi-Model Support**: Qwen models (Qwen1/2/3 dense architectures), MoE support coming soon
- **Training Checkpoints**: Automatic checkpoint management with Git integration
- **Production Ready**: Built on stable PyTorch C++ API (libtorch) for reliability

## Installation

### Prerequisites

- **Operating System**: Linux (x86_64, ARM64)
  - Windows users: Use WSL2 (Windows Subsystem for Linux)
  - macOS: Not currently supported
- Rust 1.75+
- Git 2.0+
- libtorch (PyTorch C++ library)
- **Hardware Support:**
  - **CPU**: Full support (x86_64, ARM64)
  - **CUDA**: NVIDIA GPU support
  - **ROCm**: AMD GPU support (gfx90a, gfx1100+)
- 8GB+ RAM for inference, 16GB+ for training

### PyTorch Backend Selection

Hyprstream uses feature flags to select the PyTorch backend:

- **`tch-cpu`** (default): CPU-only inference
- **`tch-cuda`**: NVIDIA GPU acceleration via CUDA
- **`tch-rocm`**: AMD GPU acceleration via ROCm/HIP

### Run with Docker:

0. Set the tag

```
export TAG=latest-cuda-129 # or latest-rocm-6.4, latest-cpu, etc.
```

1. Setup policies:

```
# WARNING: this allows all local systems users administrative access to hyprstream.
$ sudo docker run --rm -it -v hyprstream-models:/root/.local/share/hyprstream hyprstreamv-rocm policy apply-template local
```

2. Pull model(s):

```
$ sudo docker run --rm -it -v hyprstream-models:/root/.local/share/hyprstream hyprstream:$TAG clone https://huggingface.co/qwen/qwen3-0.6b
```

3. Test inference and GPU initialization

```
$ sudo docker run --rm -it -v hyprstream-models:/root/.local/share/hyprstream hyprstream:$TAG infer --prompt "hello world" qwen3-0.6b:main
```


4. Deploy openai compatible server:

```
$ sudo docker run --rm -it -v hyprstream-models:/root/.local/share/hyprstream --device=/dev/kfd --device=/dev/dri hyprstream:$TAG server
```

### Building Docker images

#### ROCm:

$ docker build -t hyprstreamv-rocm --build-arg variant=rocm .

#### Nvidia:

$ docker build -t hyprstreamv-cuda --build-arg variant=cuda .

#### CPU:

$ docker build -t hyprstreamv-cpu .

### Building from Source

#### 1. Clone Repository

```bash
git clone https://github.com/hyprstream/hyprstream.git
cd hyprstream
```

#### 2. Install libtorch

You have three options for obtaining libtorch:

**Option A: Automatic Download (Recommended)**
```bash
# tch-rs will automatically download libtorch during build
# CPU version is downloaded by default
cargo build --release
```

**Option B: Download from PyTorch**
```bash
# CUDA 12.9 version
wget https://download.pytorch.org/libtorch/cu129/libtorch-cxx11-abi-shared-with-deps-2.8.0%2Bcu129.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.8.0+cu129.zip

# CUDA 13.0 Nightly
wget https://download.pytorch.org/libtorch/nightly/cu130/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip

# ROCm 6.4
wget https://download.pytorch.org/libtorch/rocm6.4/libtorch-shared-with-deps-2.8.0%2Brocm6.4.zip
unzip libtorch-shared-with-deps-2.8.0%2Brocm6.4.zip
```

**Option C: Use Existing PyTorch Installation**
```bash
# If you have PyTorch installed via pip/conda
export LIBTORCH_USE_PYTORCH=1
```

#### 3. Set Environment Variables

Configure libtorch location:

```bash
# Option 1: Set LIBTORCH to the directory containing 'lib' and 'include'
export LIBTORCH=/path/to/libtorch

# Option 2: Set individual paths
export LIBTORCH_INCLUDE=/path/to/libtorch
export LIBTORCH_LIB=/path/to/libtorch

# Option 3: Use system-wide installation
# libtorch installed at /usr/lib/libtorch.so is detected automatically

# Add to library path
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

#### 4. Build with Backend Selection

**CPU Backend (Default)**
```bash
# Automatic download
cargo build --release

# Or with manual libtorch
export LIBTORCH=/path/to/libtorch-cpu
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release
```

**CUDA Backend**
```bash
# Set CUDA version for automatic download
export TORCH_CUDA_VERSION=cu118  # or cu121, cu124
cargo build --release
```

**ROCm Backend (AMD GPUs)**
```bash
# Set environment variables
export ROCM_PATH=/opt/rocm
export LIBTORCH=/path/to/libtorch-rocm
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
export PYTORCH_ROCM_ARCH=gfx90a  # or gfx1100, gfx1101, etc.

# Build with ROCm feature
cargo build --release
```

#### 5. Run

```bash
# The binary will be at ./target/release/hyprstream
./target/release/hyprstream --help
```

### Additional Build Options

**Static Linking**
```bash
export LIBTORCH_STATIC=1
cargo build --release
```

**Combining Features**
```bash
# CUDA + OpenTelemetry
cargo build --release --no-default-features --features tch-cuda,otel

# ROCm + XET support
cargo build --release --no-default-features --features tch-rocm,xet
```

## Quick Start

### Model Management

#### Downloading Models

```bash
# Clone a model from Git repository (HuggingFace, GitHub, etc.)
hyprstream clone https://huggingface.co/Qwen/Qwen3-0.6B

# Clone with a custom name
hyprstream clone https://huggingface.co/Qwen/Qwen3-0.6B --name qwen3-small

# Clone from any Git repository (supports all Git transports including gittorrent://)
hyprstream clone https://github.com/user/custom-model.git --name my-custom-model
```

#### Managing Models

```bash
# List all cached models (shows names and UUIDs)
hyprstream list

# Get detailed model information using ModelRef syntax
# Note that git-ref branch and tag management is a work in progress.
# Simple 'by name' models are verified.
hyprstream inspect qwen3-small           # By name
hyprstream inspect qwen3-small:main      # Specific branch
hyprstream inspect qwen3-small:v1.0      # Specific tag
hyprstream inspect qwen3-small:abc123    # Specific commit

# Pull latest updates for a model
hyprstream pull qwen3-small           # Update to latest
hyprstream pull qwen3-small main      # Update specific branch

# Push changes to remote
hyprstream push qwen3-small origin main
```

### Running Inference

```bash
# Basic inference using ModelRef syntax
hyprstream infer qwen3-small \
    --prompt "Explain quantum computing in simple terms"

# Inference with specific model version
hyprstream infer qwen3-small:v1.0 \
    --prompt "Explain quantum computing"

# Inference with specific branch
hyprstream infer qwen3-small:main \
    --prompt "Write a Python function to sort a list" \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 1024
```

## Architecture

### System Components

```mermaid
graph TD
    CLI[CLI Interface] --> Engine[TorchEngine]
    Engine --> Models[Model Management]
    Engine --> Training[Training System]

    Models --> Git[Git Repository]
    Git --> Adapters[adapters/ Directory]
    Git --> Worktrees[Training Worktrees]

    Training --> Checkpoint[Checkpoint Manager]
    Checkpoint --> GitCommit[Git Commits]

    Engine --> Inference[Inference Pipeline]
    Inference --> KVCache[KV Cache]
    Inference --> Sampling[Sampling]
```

## Supported Models

| Architecture | Status | Models |
|-------------|--------|--------|
| Qwen Dense | ✅ Full Support | Qwen1, Qwen2, Qwen2.5, Qwen3 |
| Qwen MoE | 🚧 Coming Soon | Qwen2-MoE, Qwen2.5-MoE |
| Llama | 🚧 Planned | Llama2, Llama3 |
| Gemma | 🚧 Planned | Gemma 2B, 7B |
| Mistral | 🚧 Planned | Mistral 7B |

## API Usage

### OpenAI-Compatible REST API

HyprStream provides an OpenAI-compatible API endpoint for easy integration with existing tools and libraries:

```bash
# Start API server
hyprstream server --port 50051

# List available models (worktree-based)
curl http://localhost:50051/oai/v1/models

# Example response shows models as model:branch format
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "qwen3-small:main",
#       "object": "model",
#       "created": 1762974327,
#       "owned_by": "system driver:overlay2, saved:2.3GB, age:2h cached"
#     },
#     {
#       "id": "qwen3-small:experiment-1",
#       "object": "model",
#       "created": 1762975000,
#       "owned_by": "system driver:overlay2, saved:1.8GB, age:30m"
#     }
#   ]
# }

# Make chat completions request (OpenAI-compatible)
# NOTE: Models must be referenced with branch (model:branch format)
curl -X POST http://localhost:50051/oai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-small:main",
    "messages": [
      {"role": "user", "content": "Hello, world!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Or use with any OpenAI-compatible client
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:50051/oai/v1"
# Now use any OpenAI client library
# Note: Specify model as "qwen3-small:main" not just "qwen3-small"
```

#### Worktree-Based Model References

HyprStream uses Git worktrees for model management. The `/v1/models` endpoint lists **all worktrees** (not base models):

- **Format**: Models are always shown as `model:branch` (e.g., `qwen3-small:main`)
- **Multiple Versions**: Each worktree (branch) appears as a separate model
- **Metadata**: The `owned_by` field includes worktree metadata:
  - Storage driver (e.g., `driver:overlay2`)
  - Space saved via CoW (e.g., `saved:2.3GB`)
  - Worktree age (e.g., `age:2h`)
  - Cache status (`cached` if loaded in memory)

**Example**: If you have a model `qwen3-small` with branches `main`, `experiment-1`, and `training`, the API will list three separate entries:
- `qwen3-small:main`
- `qwen3-small:experiment-1`
- `qwen3-small:training`

This allows you to work with multiple versions of the same model simultaneously, each in its own worktree with isolated changes.

### Environment Configuration

HyprStream can be configured via environment variables with the `HYPRSTREAM_` prefix:

```bash
# Server configuration
export HYPRSTREAM_SERVER_HOST=0.0.0.0
export HYPRSTREAM_SERVER_PORT=8080
export HYPRSTREAM_API_KEY=your-api-key

# CORS settings
export HYPRSTREAM_CORS_ENABLED=true
export HYPRSTREAM_CORS_ORIGINS="*"

# Model management
export HYPRSTREAM_PRELOAD_MODELS=model1,model2,model3
export HYPRSTREAM_MAX_CACHED_MODELS=5
export HYPRSTREAM_MODELS_DIR=/custom/models/path

# Performance tuning
export HYPRSTREAM_USE_MMAP=true
export HYPRSTREAM_GENERATION_TIMEOUT=120
```

## Security & Authentication

Hyprstream implements layered security-in-depth:

### Security Layers

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Transport** | CURVE encryption (TCP) | End-to-end encryption for TCP connections |
| **Application** | Ed25519 signed envelopes | Request authentication and integrity |
| **Authorization** | Casbin policy engine | RBAC/ABAC access control |

### RPC Architecture

All inter-service communication uses ZeroMQ with Cap'n Proto serialization:

- **REQ/REP**: Synchronous RPC calls (policy checks, model queries)
- **PUB/SUB**: Event streaming (sandbox lifecycle, training progress)
- **XPUB/XSUB**: Steerable proxy for event distribution

Every request is wrapped in a `SignedEnvelope`:
- Ed25519 signature over the request payload
- Nonce for replay protection
- Timestamp for clock skew validation
- Request identity (Local user, API token, Peer, or Anonymous)

### Service Spawning

Services can run in multiple modes:
- **Tokio task**: In-process async execution
- **Dedicated thread**: For `!Send` types (e.g., tch-rs tensors)
- **Subprocess**: Isolated process with systemd or standalone backend

See [docs/rpc-architecture.md](docs/rpc-architecture.md) for detailed RPC infrastructure documentation.

## Policy Engine

**Quick Start:**
```bash
# View current policy
hyprstream policy show

# Check if a user has permission
hyprstream policy check alice model:qwen3-small infer

# Create an API token
hyprstream policy token create \
  --user alice \
  --name "dev-token" \
  --expires 30d \
  --scope "model:*"

# Apply a built-in template -- allow all local users access to all actions on all resources
hyprstream policy apply-template local
```

**Built-in Templates:**
- `local` - Full access for local users (default)
- `public-inference` - Anonymous inference access
- `public-read` - Anonymous read-only registry access

**REST API Authentication:**
```bash
# Create a token
hyprstream policy token create --user alice --name "my-token" --expires 1d

# Use with API requests
curl -H "Authorization: Bearer hypr_eyJ..." http://localhost:8080/v1/models
```

See [docs/rpc-architecture.md](docs/rpc-architecture.md) for detailed RPC and service infrastructure documentation.

## Telemetry & Observability

HyprStream supports OpenTelemetry for distributed tracing, enabled via the `otel` feature flag.

### Building with OpenTelemetry

```bash
# Build with otel support
cargo build --features otel --release

# Combine with other features
cargo build --no-default-features --features tch-cuda,otel --release
```

### OpenTelemetry Configuration

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `HYPRSTREAM_OTEL_ENABLE` | Enable/disable telemetry | `false` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP backend endpoint | `http://localhost:4317` |
| `OTEL_SERVICE_NAME` | Service name in traces | `hyprstream` |
| `HYPRSTREAM_LOG_DIR` | File logging directory | None (console only) |

### Usage Examples

**Local development (stdout exporter):**
```bash
export HYPRSTREAM_OTEL_ENABLE=true
export RUST_LOG=hyprstream=debug
hyprstream server --port 8080
# Spans printed to console
```

**Production (OTLP to Jaeger/Tempo):**
```bash
export HYPRSTREAM_OTEL_ENABLE=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
export OTEL_SERVICE_NAME=hyprstream-prod
hyprstream server --port 8080
```

**File logging (separate from OTEL):**
```bash
export HYPRSTREAM_LOG_DIR=/var/log/hyprstream
hyprstream server --port 8080
# Creates daily-rotated logs at /var/log/hyprstream/hyprstream.log
```

### Exporter Modes

- **OTLP**: Used automatically when running `server` command; sends traces to backends like Jaeger, Tempo, or Datadog
- **Stdout**: Used for CLI commands; prints spans to console for debugging

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project uses a dual-licensing model:

**AGPL-3.0** - The main application and crates providing APIs:
- `hyprstream` (main application)
- `hyprstream-metrics`
- `hyprstream-flight`

See [LICENSE-AGPLV3](LICENSE-AGPLV3) for details.

**MIT** - Library crates for broader reuse:
- `git2db` - Git repository management
- `gittorrent` - P2P git transport
- `git-xet-filter` - XET large file storage filter
- `cas-serve` - CAS server for XET over SSH
- `hyprstream-rpc` - RPC infrastructure
- `hyprstream-rpc-derive` - RPC derive macros

See [LICENSE-MIT](LICENSE-MIT) for details.

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [tch](https://github.com/LaurentMazare/tch-rs) - Rust bindings for PyTorch
- [SafeTensors](https://github.com/huggingface/safetensors) - Efficient tensor serialization
- [Git2](https://github.com/rust-lang/git2-rs) - Git operations in Rust
- [Tokio](https://tokio.rs/) - Async runtime
