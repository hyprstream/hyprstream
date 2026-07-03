# HyprStream: the open network for self-improving AI

[![Rust](https://github.com/hyprstream/hyprstream/actions/workflows/rust.yml/badge.svg)](https://github.com/hyprstream/hyprstream/actions/workflows/rust.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE-AGPLV3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-MIT)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-hyprstream%2Fhyprstream-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/hyprstream/hyprstream)

## Overview

**HyprStream is the runtime for AI that gets smarter the more you use it.**

Run open-weight or custom LLMs behind a drop-in OpenAI API — and watch it improve itself on your own traffic, every gain saved as a Git branch you can review, promote, or roll back. Models, tools, live inference streams, and sandboxed apps all live in **one composable, file-like namespace** that reaches from your laptop to your fleet to — federated by cryptographic identity — the open network.

It's a single abstraction that scales the whole way up: a distributed filesystem that is also the operating system, the agentic tool interface, and the substrate for a **federation of AIs that learn from each other.**

Built in Rust on PyTorch (`tch-rs`) — real inference *and* training, Git-native weights, microVM-sandboxed tool use, and a collaborative TUI for humans and agents. [Download](https://github.com/hyprstream/hyprstream/releases/) the AppImage (it auto-detects your NVIDIA or ROCm GPU) and you're running in minutes — see [docs/quickstart.md](docs/quickstart.md).

## Why HyprStream

### 🧠 Models that improve themselves

Most servers run a frozen model. **HyprStream's models get better while they serve** — they learn from your real traffic and keep the gains.

It's a closed loop you can audit and reverse at any step:

1. **Stage** — the model adapts to the task in front of it and can stage its own improvements through agentic (MCP) tools.
2. **Train** — it learns from your live traffic, or from a stronger **teacher model**. Each change is applied *speculatively*: in place, but instantly reversible — the way a CPU runs ahead of a branch it can undo.
3. **Evaluate** — agentic tools score every change automatically, so only adaptations that genuinely perform better survive.
4. **Promote** — merge a winning adaptation into the live (online) model and checkpoint it as a **Git branch of the weights** (`model:branch`). Checkpoint on your terms — every step, on demand via the API, or automatically through agentic tools — each one versioned, diffable, reviewable, and roll-back-able.

*Under the hood: test-time training (Muon optimizer + TTN profiling), with a runtime **rank oracle** that advises how much capacity each adaptation needs — a signal the teacher model can act on or overrule.*

### 📁 Everything is a file

Models, inference streams, MCP tools, repositories, and even **sandboxed applications** are all just files in a composable namespace. Want to run a model? Open it. Stream tokens? Read a file under `/stream/`. Drive a service? Write its `ctl` file. And you compose all of it from a **sandboxed, Unix-like shell** — `cat`, pipes, and a real scripting language — the same interface humans and agents use to drive models, tools, and apps. Give an agent a tool, or run an untrusted app, and it's a file you mount, scope, and revoke.

This Plan 9-inspired design (see [docs/vfs.md](docs/vfs.md)) gives one uniform interface for everything and per-process namespaces for true isolation. In the browser, HyprStream's WASM clients run as **[Wanix](https://github.com/tractordev/wanix) tasks**, and the VFS mounts Wanix's filesystem over 9P — so the *same* namespace spans your host and the browser. Composable, network-transparent, capability-scoped.

### 📦 Sandboxed tasks: containers, microVMs, and WASM

Untrusted code — agent tool-calls, user apps, CI workflows — runs in **workers**: isolated tasks projected into the namespace as a Plan 9-style process tree (`/exec/instances/<id>/ctl`) that you drive like any other file. One task API, four isolation backends, selected automatically and **fail-closed**:

- **Kata microVMs** (cloud-hypervisor) — hardware-virtualized isolation for the truly untrusted
- **OCI containers** (rootless Podman) — standard images, no daemon, no root
- **systemd-nspawn** — lightweight OS containers
- **WASM** (wasmtime) — capability-limited execution, no ambient network; the same sandbox browser apps get as Wanix tasks

Language engines run *inside* the sandbox family: a locked-down **Tcl shell** for the VFS, **Python** (RustPython over the wasmtime sandbox, mounted at `/lang/python`), and raw **wasmtime** capability profiles. A sandbox is reachable as files, but can't touch anything you didn't mount for it. See [docs/workers-architecture.md](docs/workers-architecture.md).

### 🌐 Share models, tools, and files across nodes

Federation turns isolated servers into one network. Use a model another node hosts, call an **MCP tool** a peer publishes, or mount a remote **filesystem** or live stream — addressed by its owner's cryptographic identity and used **as if it were local.** An agent on your node can reach tools, weights, and namespaces anywhere on the fabric, every call policy-gated. No brittle URLs or per-host API keys: resources are portable and verifiable (atproto `did:web`/`did:key` + `at://` handles), reached over QUIC/WebTransport or direct peer-to-peer (iroh). It's **off by default** — opt in with the setup wizard.

### 🔒 Security that fails closed

Zero-trust and **post-quantum-ready** by default: every RPC rides a COSE **hybrid-signed envelope** (Ed25519 + ML-DSA-65), streams and events are encrypted end to end so a forwarding relay never sees plaintext, and authorization is deny-by-default — Casbin policy plus **UCAN capability delegation**. Beneath all of it, a **label-based mandatory access control** layer — security labels compiled from capability grants into a hybrid-PQC-signed policy matrix, backed by a tamper-evident audit chain — is rolling out as the floor no policy grant can bypass. See [Security & Authentication](#security--authentication) and [docs/mac-architecture.md](docs/mac-architecture.md).

### ⚡ Runs on your hardware

One binary, any accelerator. The **Universal AppImage** auto-detects and loads the right backend — **NVIDIA CUDA**, **AMD ROCm** (including gfx1151 / Strix Halo), or CPU — with per-process GPU selection and no CUDA/ROCm toolchain juggling. Also packaged as a multi-variant **Nix flake** (CPU / CUDA / ROCm). Built on the stable PyTorch C++ API (libtorch).

Runs the **Llama 1/2/3, Gemma, Qwen3 (dense), Qwen3.5 (dense and MoE), and Janus (multimodal)** model families, cloned straight from HuggingFace or any Git remote.

### Also includes

- **Paged KV cache** — prefix caching and reuse, paged attention blocks, blockwise quantization, and CPU offload with transparent GPU restore.
- **Collaborative TUI** — a high-speed compositing multiplexer; share terminals live with teammates and agents. The same ratatui apps run natively and in the browser (WASI) via waxterm.
- **OpenAI-compatible API** — drop-in `/v1/chat/completions` for existing tools and client libraries, with multi-format tool calling (Qwen, Llama, Mistral) and parallel tool calls.
- **MCP server** — expose inference, model management, and repo ops as tools for Claude Code, Cursor, etc.
- **MoQ streaming fabric** — token streams, events, and container I/O ride Media-over-QUIC with QoS presets; the event bus is group-key encrypted (AES-256-GCM) end to end.
- **Git-native weights** — models and source tracked with Git; HuggingFace-compatible, with XET large-file storage.
- **Systemd integration** — optional user-level management for background workers and long-running services.

### Experimental Features

- **[Workflows]** — Git workflow files for local CI/CD and functions-as-a-service.
- **[Metrics]** — structured knowledge engine + time-series database powered by DuckDB, ADBC, and Flight.

## Installation

### Quick Install (AppImage, Linux)

Hyprstream requires `git` and `git-lfs` (available in all major Linux distros).

Download the [Universal AppImage](https://github.com/hyprstream/hyprstream/releases/). We publish AppImages for each CPU/GPU configuration; the Universal image is recommended for ease-of-use and GPU auto-detection.

```bash
# Download the latest release, then make it executable
chmod +x hyprstream-v0.4.0-x86_64.AppImage

# Guided setup (recommended): the wizard configures access policy, users,
# and API tokens. Add `-y` to accept defaults, `--start` to launch services,
# and `--enable-federation` to join the open network (see Federation below).
./hyprstream-v0.4.0-x86_64.AppImage wizard

# Add to PATH
export PATH="$HOME/.local/bin:$PATH"

# Start the services (or pass `--start` to the wizard above)
hyprstream service start
```

See [docs/quickstart.md](docs/quickstart.md) for prerequisites, source build, and first-time setup.

NOTE: For CUDA systems, make sure you have installed [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) and set `LD_PRELOAD`:

```bash
systemctl --user set-environment LD_PRELOAD=libtorch_cuda.so && systemctl --user restart hyprstream-model
```

The installed files will be located in `$HOME/.local/hyprstream/` and `$HOME/.local/bin/`.

### Install with Nix

A multi-variant flake builds CPU, CUDA, and ROCm packages:

```bash
nix build github:hyprstream/hyprstream#cpu     # or #cuda / #rocm
```

### Building from source

```bash
# Set LIBTORCH to your libtorch path, or use --features download-libtorch
cargo build --release
```

See [docs/quickstart.md](docs/quickstart.md) for prerequisites and [DEVELOP.md](DEVELOP.md) for detailed build instructions.

### Container deployment

Hyprstream can run inside containers. See [README-Docker.md](README-Docker.md) for Docker/Kubernetes deployment.

## Quick Start

### Clone a model

Hyprstream runs models cloned from Git repositories (HuggingFace, GitHub, etc.) — Llama, Gemma, Qwen3, Qwen3.5, and Janus families.

```bash
# Clone a model
hyprstream quick clone https://huggingface.co/Qwen/Qwen3-0.6B

# Clone with a custom name
hyprstream quick clone https://huggingface.co/Qwen/Qwen3-0.6B --name qwen3-small
```

### Managing models

Worktrees are automatically managed by hyprstream.

```bash
# List all cached models
hyprstream quick list

# Get detailed model information (model:branch format)
hyprstream quick info qwen3-small
hyprstream quick info qwen3-small:main
```

### Run inference

```bash
# Basic inference (streams by default; pass --sync for a single response)
hyprstream quick infer qwen3-small:main \
    --prompt "Explain quantum computing in simple terms"

# With options
hyprstream quick infer qwen3-small:main \
    --prompt "Write a Python function to sort a list" \
    --temperature 0.7 \
    --top-p 0.9 \
    --max-tokens 1024
```

## Everything is a file: the VFS

HyprStream is organized around a **Plan 9-inspired virtual filesystem** ([docs/vfs.md](docs/vfs.md)): resources are named files in a composable namespace rather than ad-hoc APIs. In the browser it runs on **[Wanix](https://github.com/tractordev/wanix)** — the same namespace, extended to the web.

- **Driven by a shell** — the whole namespace is scriptable from a **sandboxed, Unix-like shell**: `cat`, pipes, redirection, and a real scripting language (Tcl, via [molt](https://github.com/wduquette/molt)) — e.g. `set t [cat /config/temperature]; expr {$t + 0.1}`. The surface humans use is the surface agents use to call tools. It runs as a locked-down guest — **no direct host I/O**, dangerous commands removed, with instruction- and recursion-limits — so untrusted tool calls and scripts stay contained.
- **Models as branches** — every model is a Git repo; every adaptation is a worktree (`model:branch`). Versioned, diffable, reversible weights.
- **Streams as files** — token streams, events, and container I/O are read/written under `/stream/`-style mounts with backpressure and at-least-once / lossless QoS presets.
- **Control via `ctl`** — services are steered by writing their `ctl` file (a Plan 9 idiom), not bespoke management RPCs.
- **Tools as files** — MCP tools, repositories, and registries appear as files an agent can be granted, scoped to, and revoked from.
- **Tasks as files** — sandboxed workloads appear under a `/exec` process tree; write a task's `ctl` file to drive it ([docs/workers-architecture.md](docs/workers-architecture.md)).
- **Per-process namespaces + sandboxing** — each workload gets its own namespace. Untrusted *native* workloads run in **Kata microVMs** or rootless OCI containers; *browser* apps and agent tool-calls run as **Wanix WASM tasks** (WASI, no ambient network). Either way a sandbox is reachable as files but can't touch anything not mounted for it — capability-scoped by construction.
- **Wanix bridge** — Wanix's own filesystem mounts into the namespace at `/wanix/` over 9P, so browser and host resources compose into one tree.

The payoff is one abstraction that scales the whole way up: a **distributed filesystem** that is also the **operating system**, the **agentic tool interface**, and — federated across nodes — the substrate for a **network of AIs that learn from each other.** One namespace, all the way up.

## Federation: one network for models, tools, and files

Federation lets your node use — and offer — resources across the network instead of being a silo. Because everything is a file in a network-transparent namespace (see above), "remote" resources behave like local ones:

- **Federated models** — run a model another node hosts by its identity/name (`model:branch`); it loads and streams as if it were on your box.
- **Federated tools** — an agent on your node can call MCP tools a peer publishes (and you can offer yours), every call policy-checked.
- **Federated filesystems & streams** — mount a peer's namespace, read its event/token streams, or drive a shared service — composable across hosts.
- **Identity, not URLs** — everything is addressed by cryptographic identity (`did:web`/`did:key` + `at://` handle), so resources are portable, verifiable, and survive a host moving.

### Enable it

Federation is **off by default** — deny-by-default, your data stays yours. Turn on the open network (accept third-party apps and remote peers) via the setup wizard:

```bash
# Interactive — the wizard asks about federation during the Policy step
hyprstream wizard

# Non-interactive — opt in explicitly
hyprstream wizard -y --enable-federation
```

This applies the `federation-open` policy template. You can also scope it later with the [policy engine](#policy-engine) to allow only specific peer origins.

### Under the hood

Each node publishes an atproto-compatible identity document (`did:web`/`did:key`, `at://` handle, P-256 `#atproto` verification method, PDS-style service entry, and typed transport endpoints). Peers are admitted by a two-stage gate — an origin allowed by policy **and** a connecting key bound to a verification method in the peer's published DID (fail-closed otherwise) — and reached over QUIC/WebTransport or direct peer-to-peer via **iroh + pkarr** (dial by Ed25519 node id alone).

> Built for the open social web: HyprStream speaks atproto end-to-end — identity, admission, and transport — so your models, tools, and files federate with the broader Atmosphere, including atproto-native networks like [tiles.run](https://tiles.run).

## Architecture

![Architecture](architecture.png)

## Integrating Hyprstream into your business or workflow

### OpenAI-Compatible REST API

HyprStream provides an OpenAI-compatible API endpoint (port 6789 by default) for easy integration with existing tools and libraries:

```bash
# Start the services (includes the OpenAI-compatible API)
hyprstream service start

# List available models (worktree-based)
curl http://localhost:6789/oai/v1/models

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
curl -X POST http://localhost:6789/oai/v1/chat/completions \
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
export OPENAI_BASE_URL="http://localhost:6789/oai/v1"
# Now use any OpenAI client library
# Note: Specify model as "qwen3-small:main" not just "qwen3-small"
```

#### Worktree-Based Model References

HyprStream uses Git worktrees for model management. The `/oai/v1/models` endpoint lists **all worktrees** (not base models):

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

### MCP Integration (Claude Code, Cursor, etc.)

HyprStream includes a built-in [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes inference, model management, and repository operations as tools for AI coding assistants.

**1. Configure Claude Code:**

```
claude mcp add --transport http hyprstream http://localhost:6790/mcp
```

**2. Authenticate**

Use `/mcp`, select `hyprstream`, and select `Authenticate` or `Re-authenticate`.

**3. Available tools:**

Once connected, Claude Code can use hyprstream tools directly (tools are schema-generated as `service.method`):

| Tool | Description |
|------|-------------|
| `model.load` / `model.unload` / `model.status` | Load, unload, and inspect models |
| `infer.*` | Run (streaming) inference |
| `ttt.*` | Drive test-time training: init, train, status, export |
| `registry.list` / `registry.clone` | List and clone model repositories |
| `registry.repo.*` | Branch, worktree, merge, and tag operations |
| `policy.*` | Policy checks and token management |

**Configuration:**

The MCP server listens on port 6790 by default. To change it, set in your hyprstream config:

```toml
[mcp]
host = "127.0.0.1"
http_port = 6790
```

Or configure via the [OAI-compatible API](#openai-compatible-rest-api) on port 6789 for non-MCP clients.

### Advanced deployments

HyprStream can be configured via environment variables with the `HYPRSTREAM_` prefix:

```bash
# Server configuration
export HYPRSTREAM_SERVER_HOST=0.0.0.0
export HYPRSTREAM_SERVER_PORT=6789

# CORS settings
export HYPRSTREAM_CORS_ENABLED=true
export HYPRSTREAM_CORS_ORIGINS="*"

# Model management
export HYPRSTREAM_PRELOAD_MODELS=model1,model2,model3
export HYPRSTREAM_MAX_CACHED_MODELS=5

# Performance tuning
export HYPRSTREAM_USE_MMAP=true
```

## Security & Authentication

Hyprstream implements layered, zero-trust, **post-quantum-ready** security.

### Security Layers

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Transport** | QUIC / WebTransport + iroh (TLS / Noise) | Encrypted, NAT-traversing transport for browsers, hosts, and P2P peers |
| **Application** | COSE hybrid-signed envelopes (Ed25519 + ML-DSA-65) | Request authentication + integrity with post-quantum signatures |
| **Identity** | atproto `did:web`/`did:key` + federation admission | Identity-bound peering (origin policy + DID-key binding) |
| **Authorization** | Casbin policy engine + UCAN capability delegation | RBAC/ABAC access control, attenuated delegation chains, deny-by-default |
| **Mandatory floor** | Label-based MAC ([docs/mac-architecture.md](docs/mac-architecture.md)) | Security labels compiled from grants into a hybrid-PQC-signed policy matrix, tamper-evident audit chain; fails closed (enforcement wiring in rollout) |
| **Isolation** | Kata microVMs / rootless OCI / nspawn / WASM | Sandboxed workload execution for workers and tool calls |

### RPC & streaming architecture

Inter-service and inter-node communication runs over **moq-net** (moq-lite streaming) and **iroh**, with Cap'n Proto serialization:

- **Request/reply RPC** over QUIC/WebTransport (browsers, reachable hosts) and iroh (direct P2P).
- **Event bus** over a moq-lite "Live" stream — **group-key encrypted** (AES-256-GCM with membership-driven rekeying), so a forwarding relay never sees event plaintext.
- **Streaming plane** — moq groups carry token streams, events, and container I/O, with QoS presets (`Job` / `Log` / `Pipe`), per-frame chained-HMAC integrity, and AEAD payloads.

Every request is wrapped in a **COSE hybrid-signed envelope**:
- Ed25519 + ML-DSA-65 composite signature over the request (classical + post-quantum)
- Nonce for replay protection and timestamp for clock-skew validation
- Verified request identity (local user, API token, federated peer, or anonymous)

### Service Spawning

Services can run in multiple modes:
- **Tokio task**: In-process async execution
- **Dedicated thread**: For `!Send` types (e.g., tch-rs tensors)
- **Subprocess**: Isolated process with systemd or standalone backend

See [docs/rpc-architecture.md](docs/rpc-architecture.md) and [docs/streaming-service-architecture.md](docs/streaming-service-architecture.md) for details.

## Policy Engine

**Quick Start:**
```bash
# View current policy
hyprstream quick policy show

# Check if a user has permission
hyprstream quick policy check alice model:qwen3-small infer

# Create an API token (expiry defaults to 1y)
hyprstream quick policy token create alice \
  --name "dev-token" \
  --expires 30d \
  --scope "model:*"

# Apply a built-in template — e.g. allow anonymous inference
hyprstream quick policy apply-template public-inference
```

**Built-in Templates** (`hyprstream quick policy list-templates`):
- `federation-open` - Accept federated peers and third-party apps
- `public-inference` - Anonymous inference access
- `public-read` - Anonymous read-only registry access
- `ttt.user` / `ttt.agent` - Test-time-training roles

First-time setup is handled by `hyprstream wizard`, which configures users, access policy, and API tokens interactively.

**Worker Resources:**
| Resource | Description |
|----------|-------------|
| `sandbox:*`, `sandbox:{id}` | Pod sandbox (Kata/OCI/nspawn/WASM) operations |
| `container:*`, `container:{id}` | Container lifecycle within sandboxes |
| `image:*`, `image:{name}` | Image pull/push/list operations |
| `workflow:*`, `workflow:{path}` | Workflow execution (.github/workflows/*.yml) |
| `tool:*`, `tool:{name}` | MCP tool access (tool:bash, tool:read_file) |

**Policy History & Rollback:**
```bash
# View policy commit history
hyprstream quick policy history

# Compare draft vs running policy
hyprstream quick policy diff

# Rollback to previous version
hyprstream quick policy rollback HEAD~1
```

**REST API Authentication:**
```bash
# Create a token
hyprstream quick policy token create alice --name "my-token" --expires 1d

# Use with API requests
curl -H "Authorization: Bearer eyJ..." http://localhost:6789/oai/v1/models
```

See [docs/rpc-architecture.md](docs/rpc-architecture.md) for detailed RPC and service infrastructure documentation.

## Telemetry & Observability

HyprStream supports OpenTelemetry for distributed tracing, enabled via the `otel` feature flag.

### Building with OpenTelemetry

```bash
# Build with otel support
cargo build --features otel --release
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
hyprstream service start
# Spans printed to console
```

**Production (OTLP to Jaeger/Tempo):**
```bash
export HYPRSTREAM_OTEL_ENABLE=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
export OTEL_SERVICE_NAME=hyprstream-prod
hyprstream service start
```

**File logging (separate from OTEL):**
```bash
export HYPRSTREAM_LOG_DIR=/var/log/hyprstream
hyprstream service start
# Creates daily-rotated logs at /var/log/hyprstream/hyprstream.log
```

### Debugging GPU detection issues:

If the Universal AppImage is not detecting your GPU, you may override the settings:

```bash
# List all available backends
./hyprstream-x86_64.AppImage --list-backends

# Detect available backends
./hyprstream-x86_64.AppImage --detect-gpu

# Override backend selection for Universal AppImage:
HYPRSTREAM_BACKEND=cuda130 ./hyprstream-x86_64.AppImage service start
```

### System Requirements

- **Operating System**: Linux (x86_64, ARM64)
- **Inference Service Requirements (optional):**
  - **CPU**: Full support (x86_64, ARM64)
  - **CUDA**: NVIDIA host kernel modules (`nvidia-smi` works)
  - **ROCm**: AMDGPU kernel modules and userland (`rocm-smi` works)
- **Workers Service Requirements (optional):**
  - **Kata backend**: nested virtualization (a physical machine, bare-metal VM, or QEMU/KVM configured for it) and `cloud-hypervisor`
  - **OCI backend**: rootless Podman
  - **WASM / nspawn backends**: no special host requirements
- 8GB+ RAM for inference, 16GB+ for training
- **Optional Dependencies:**
  - `systemd` - For service management and worker process isolation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project uses a dual-licensing model:

**MIT OR AGPL-3.0 (your choice)** — the main `hyprstream` application.

**MIT** — all other crates, including the RPC/security stack (`hyprstream-rpc`, `hyprstream-rpc-derive`, `hyprstream-rpc-std`, `hyprstream-service`, `hyprstream-discovery`), the namespace stack (`hyprstream-vfs`, `hyprstream-9p`, `hyprstream-vfs-server`, `hyprstream-containedfs`), workers (`hyprstream-workers` + engine crates), the TUI stack (`hyprstream-tui`, `hyprstream-compositor`, `waxterm`, `chat-core`), `hyprstream-metrics`, `hyprstream-flight`, `hyprstream-pds`, and the Git/storage libraries (`git2db`, `gittorrent`, `git-xet-filter`, `cas-serve`).

See [LICENSE-MIT](LICENSE-MIT) and [LICENSE-AGPLV3](LICENSE-AGPLV3) for details.

## Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [tch](https://github.com/LaurentMazare/tch-rs) - Rust bindings for PyTorch
- [SafeTensors](https://github.com/huggingface/safetensors) - Efficient tensor serialization
- [Git2](https://github.com/rust-lang/git2-rs) - Git operations in Rust
- [Tokio](https://tokio.rs/) - Async runtime
- [iroh](https://www.iroh.computer/) - Peer-to-peer QUIC transport and discovery
- [moq](https://quic.video/) - Media-over-QUIC live streaming
- [Casbin](https://casbin.org/) - Authorization library for policy engine
- [Kata Containers](https://katacontainers.io/) - VM-based container isolation
- [cloud-hypervisor](https://www.cloudhypervisor.org/) - Virtual machine monitor
- [wasmtime](https://wasmtime.dev/) - WebAssembly runtime for sandboxed engines

## Star History

<a href="https://www.star-history.com/?repos=hyprstream%2Fhyprstream&type=date&logscale=&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=hyprstream/hyprstream&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=hyprstream/hyprstream&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=hyprstream/hyprstream&type=date&legend=top-left" />
 </picture>
</a>
