# Running Hyprstream in Docker

Hyprstream supports containerized deployment using Docker Compose with a multi-service architecture.

Services communicate via ZeroMQ over a shared tmpfs IPC volume, enabling efficient inter-process communication without network overhead.

## Quick Start

### 1. Select your GPU profile and start services

```bash
# CPU inference
docker compose --profile cpu up -d

# NVIDIA CUDA GPU
docker compose --profile cuda up -d

# AMD ROCm GPU
docker compose --profile rocm up -d
```

### 2. Setup policies (required before first use)

```bash
# Allow local users administrative access
docker compose run --rm client policy apply-template local
```

### 3. Clone a model

```bash
docker compose run --rm client clone https://huggingface.co/Qwen/Qwen3-0.6B
```

### 4. Test inference

```bash
docker compose run --rm client infer qwen3-0.6b:main --prompt "Hello, world!"
```

### 5. Use the OpenAI-compatible API

The API is exposed on port 8010 by default:

```bash
curl http://localhost:8010/v1/models

curl -X POST http://localhost:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b:main",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Service Architecture

Docker Compose runs Hyprstream as separate microservices:

| Service | Description | Profile |
|---------|-------------|---------|
| `event` | Event bus (PUB/SUB) | all |
| `registry` | Model registry | all |
| `policy` | Authorization (Casbin) | all |
| `streams` | Token streaming | all |
| `model` | Model inference (CPU) | cpu |
| `model-cuda` | Model inference (NVIDIA) | cuda |
| `model-rocm` | Model inference (AMD) | rocm |
| `oai` | OpenAI-compatible API | all |
| `client` | CLI client | client |
| `flight` | Flight SQL service | flight |
| `worker` | Kata VM workers | workers |

### Optional Services

```bash
# Include Flight SQL service
docker compose --profile cpu --profile flight up -d

# Include worker service (requires privileged container)
docker compose --profile cpu --profile workers up -d

# Run CLI commands interactively
docker compose run --rm client
```

## Available Image Tags

Images are published to `ghcr.io/hyprstream/hyprstream`:

| Tag | Description |
|-----|-------------|
| `latest-cpu` | CPU-only inference |
| `latest-cuda128` | NVIDIA CUDA 12.8 |
| `latest-cuda130` | NVIDIA CUDA 13.0 |
| `latest-rocm` | AMD ROCm 7.1 |

Override the default tag:

```bash
TAG=latest-cuda130 docker compose --profile cuda up -d
```

## Volumes

| Volume | Purpose |
|--------|---------|
| `models` | Persistent model storage and signing keys |
| `ipc` | Tmpfs for inter-service ZeroMQ sockets |

## Building Docker Images

```bash
# CPU
docker build -t hyprstream:cpu --build-arg VARIANT=cpu .

# NVIDIA CUDA 12.8
docker build -t hyprstream:cuda128 --build-arg VARIANT=cuda128 .

# NVIDIA CUDA 13.0
docker build -t hyprstream:cuda130 --build-arg VARIANT=cuda130 .

# AMD ROCm 7.1
docker build -t hyprstream:rocm --build-arg VARIANT=rocm71 .
```

### Build arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `VARIANT` | `cpu` | Build variant: `cpu`, `cuda128`, `cuda130`, `rocm71` |
| `DEBIAN_VERSION` | `bookworm` | Debian base image version |
| `LIBTORCH_VERSION` | `2.10.0` | PyTorch/LibTorch version |

## Privileged Containers

Running Hyprstream in Docker does not require privileged containers, except for:

- **Workers service**: Requires privileged mode for Kata VM sandboxing (`--privileged`)
- **Device access**: GPU containers need device passthrough (handled automatically by profiles)

Multi-host deployments can run Workers outside Docker or in privileged containers while keeping inference services in unprivileged containers.
