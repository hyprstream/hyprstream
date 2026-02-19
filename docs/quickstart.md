# Hyprstream Quickstart Guide

This guide walks you from prerequisites through your first inference. It is derived from the codebase, not from other documentation.

**Platform:** Linux only (AppImage, systemd, and all features).

---

## Prerequisites

### System packages

Hyprstream requires Git for model management. Install:

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install git git-lfs
```

**Linux (Fedora):**
```bash
sudo dnf install git git-lfs
```

**Linux (Arch):**
```bash
sudo pacman -S git git-lfs
```

### For building from source

- **Rust** 1.75 or later (`rustup` recommended)
- **LibTorch** (PyTorch C++ library) — see [Installation](#installation) for options
- **Cap'n Proto** compiler (`capnp`)
- **pkg-config** and **OpenSSL** dev headers

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install build-essential pkg-config libssl-dev libsystemd-dev capnproto ca-certificates
```

**Linux (Fedora):**
```bash
sudo dnf install gcc gcc-c++ pkg-config openssl-devel systemd-devel capnproto
```

---

## Installation

### Option A: AppImage

AppImages are pre-built for Linux x86_64. Download from [releases](https://github.com/hyprstream/hyprstream/releases).

1. Download the [Universal AppImage](https://github.com/hyprstream/hyprstream/releases) or a variant (cpu, cuda128, cuda130, rocm71).
2. Make it executable and install:

```bash
chmod +x hyprstream-v0.2.0-x86_64.AppImage
./hyprstream-v0.2.0-x86_64.AppImage service install
```

3. Add the binary to your PATH (if not already):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

The binary is installed at `~/.local/bin/hyprstream`.

### Option B: Build from source

1. Clone the repository:

```bash
git clone https://github.com/hyprstream/hyprstream.git
cd hyprstream
```

2. Install LibTorch (choose one):

**Automatic download** (simplest; CPU-only):

```bash
cargo build --release --features download-libtorch
```

**Manual download** — pick the variant for your hardware:

| Variant | URL |
|---------|-----|
| CPU | https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.10.0%2Bcpu.zip |
| CUDA 12.8 | https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.10.0%2Bcu128.zip |
| CUDA 13.0 | https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.10.0%2Bcu130.zip |
| ROCm 7.1 | https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-2.10.0%2Brocm7.1.zip |

Then:

```bash
unzip libtorch-*.zip
export LIBTORCH=$(pwd)/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build --release
```

3. Run the binary:

```bash
./target/release/hyprstream --help
```

### Option C: AppImage build script

For a reproducible build with the correct libtorch (CUDA, ROCm, etc.):

```bash
./appimage/build-appimage.sh build cuda130    # CUDA 13.0
./appimage/build-appimage.sh build cuda128    # CUDA 12.8
./appimage/build-appimage.sh build rocm71     # ROCm 7.1
./appimage/build-appimage.sh build cpu        # CPU only
```

Output: `appimage/output/hyprstream-dev-<variant>-x86_64.AppImage`

---

## First-time setup

### 1. Install the service (AppImage only)

If you used the AppImage, run:

```bash
hyprstream service install
```

This installs the `hyprstream` command and systemd units. For source builds, run the binary from `target/release/` or add it to your PATH manually.

### 2. Apply a policy template

Hyprstream uses a deny-by-default policy. Grant yourself access:

```bash
hyprstream quick policy apply-template local
```

This gives the current local user full access for CLI operations.

Other templates:

- `public-inference` — allow anonymous inference (for open API)
- `public-read` — allow anonymous registry browsing

### 3. Start services

Start all hyprstream services (registry, policy, model, OAI API, etc.):

```bash
hyprstream service start
```

Check status:

```bash
hyprstream service status
```

---

## Model management

### Clone a model

Models are Git repositories. Clone from HuggingFace or any Git host:

```bash
hyprstream quick clone https://huggingface.co/Qwen/Qwen3-0.6B
```

With a custom name:

```bash
hyprstream quick clone https://huggingface.co/Qwen/Qwen3-0.6B --name qwen3-small
```

Shallow clone (depth 1) is the default. Use `--full` for full history.

### List models

```bash
hyprstream quick list
```

### Inspect a model

```bash
hyprstream quick info qwen3-small
hyprstream quick info qwen3-small:main
```

### Optional: Preload a model

Preloading loads the model into memory for faster first inference:

```bash
hyprstream quick load qwen3-small:main
```

---

## Inference

### CLI inference

Run inference with a model reference (`model:branch`):

```bash
hyprstream quick infer qwen3-small:main --prompt "Explain quantum computing in simple terms"
```

With options:

```bash
hyprstream quick infer qwen3-small:main \
  --prompt "Write a Python function to sort a list" \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-tokens 1024 \
  --stream
```

Read prompt from stdin:

```bash
echo "What is Rust?" | hyprstream quick infer qwen3-small:main
```

### OpenAI-compatible API

The OAI API listens on port 6789 by default at `/oai/v1`.

List models:

```bash
curl http://localhost:6789/oai/v1/models
```

Chat completion:

```bash
curl -X POST http://localhost:6789/oai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-small:main",
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Use with OpenAI clients:

```bash
export OPENAI_API_KEY="dummy"
export OPENAI_BASE_URL="http://localhost:6789/oai/v1"
```

Always use the `model:branch` format (e.g. `qwen3-small:main`), not just the model name.

---

## Configuration

### Paths (XDG)

Hyprstream follows the XDG Base Directory specification:

| Purpose   | Path                                  |
|-----------|----------------------------------------|
| Models    | `~/.local/share/hyprstream/models`     |
| Config    | `~/.config/hyprstream`                |
| Cache     | `~/.cache/hyprstream`                 |

### Config file

Place a config file in the config directory. Supported names: `config`, `config.toml`, `config.json`, `config.yaml`.

Example `~/.config/hyprstream/config.toml`:

```toml
[oai]
host = "0.0.0.0"
port = 6789

[oai.cors]
enabled = true
```

### Environment variables

Override config with `HYPRSTREAM__`-prefixed variables (double underscore for nesting):

```bash
export HYPRSTREAM__OAI__PORT=8080
export HYPRSTREAM_CONFIG=/path/to/config.toml
```

Or pass a config file explicitly:

```bash
hyprstream --config /path/to/config.toml quick list
```

### Git2DB (model registry)

The registry uses `~/.config/git2db/config.toml` or `GIT2DB__*` env vars. For example:

```bash
export GIT2DB_WORKTREE__DRIVER=vfs
```

---

## Troubleshooting

### "Permission denied" or policy errors

Run:

```bash
hyprstream quick policy apply-template local
```

### "Commit failed" when applying policy template

If you see `Commit failed:` when running `policy apply-template`, ensure Git is configured:

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Then run the template again.

### CUDA / GPU not detected

Ensure `LD_PRELOAD` is set (systemd):

```bash
systemctl --user set-environment LD_PRELOAD=libtorch_cuda.so
systemctl --user restart hyprstream-model
```

Ensure libtorch CUDA/ROCm libraries are in your library path (see [Libtorch not found](#libtorch-not-found-source-build)).

### Services not starting

Check systemd:

```bash
systemctl --user status hyprstream-*
journalctl --user -u hyprstream-oai -f
```

### Libtorch not found (source build)

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
```

---

## Next steps

- **Branches and worktrees**: `hyprstream quick worktree add`, `hyprstream quick branch`
- **Training**: `hyprstream quick training` subcommands
- **MCP**: Configure Claude Code / Cursor to use `http://localhost:6790/mcp`
- **Workers**: See `docs/workers-architecture.md` for Kata-based workloads
