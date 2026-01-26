Hyprstream has limited support for running inside of Docker.

These images use the non-systemd release and are not built on the AppImage,
but install hyprstream atop a Debian base image.

Running Hyprstream in Docker does not require privileged containers, except to
support microvm based Workers, an optional feature.

Multi-host deployments of Hyprstream an support Workers running outside of Docker,
or in a privileged container, and non-worker services running in unprivileged user containers.

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

