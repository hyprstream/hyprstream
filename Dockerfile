# syntax=docker/dockerfile:1
# Multi-variant build for Hyprstream supporting CPU, CUDA, and ROCm

ARG VARIANT=cpu
ARG DEBIAN_VERSION=bookworm
ARG LIBTORCH_VERSION=2.10.0

# LibTorch download URLs for manual installation
ARG LIBTORCH_CUDA128_URL=https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu128.zip
ARG LIBTORCH_CUDA130_URL=https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcu130.zip
ARG LIBTORCH_ROCM_URL=https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Brocm7.1.zip
ARG LIBTORCH_CPU_URL=https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip

#############################################
# Base Builder - Common for all variants
#############################################

FROM debian:${DEBIAN_VERSION} AS builder-base

# Install build dependencies
# Note: binutils from backports required for OpenSSL AVX-512 assembly compatibility
RUN echo "deb http://deb.debian.org/debian bookworm-backports main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    build-essential \
    pkg-config \
    libssl-dev \
    libsystemd-dev \
    git \
    dialog \
    rsync \
    ca-certificates \
    capnproto \
    cmake \
    clang \
    libclang-dev \
    && apt-get install -y -t bookworm-backports binutils \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install sccache for compilation caching (works with BuildKit cache mounts)
RUN cargo install sccache --locked
ENV RUSTC_WRAPPER=sccache
ENV SCCACHE_DIR=/sccache

#############################################
# CUDA 12.8 Builder
#############################################

FROM builder-base AS builder-cuda128
ARG LIBTORCH_CUDA128_URL
ARG LIBTORCH_VERSION

ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Install CUDA repository and runtime libraries (needed for linking)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Download and extract LibTorch for CUDA 12.8 (cached across builds)
RUN --mount=type=cache,target=/tmp/libtorch-cache \
    CACHE_FILE="/tmp/libtorch-cache/libtorch-cuda128-${LIBTORCH_VERSION}.zip" && \
    if [ ! -f "$CACHE_FILE" ]; then \
        wget -q ${LIBTORCH_CUDA128_URL} -O "$CACHE_FILE"; \
    fi && \
    unzip -q "$CACHE_FILE" -d /opt

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

#############################################
# CUDA 13.0 Builder
#############################################

FROM builder-base AS builder-cuda130
ARG LIBTORCH_CUDA130_URL
ARG LIBTORCH_VERSION

ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Install CUDA repository and runtime libraries (needed for linking)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-13-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Download and extract LibTorch for CUDA 13.0 (cached across builds)
RUN --mount=type=cache,target=/tmp/libtorch-cache \
    CACHE_FILE="/tmp/libtorch-cache/libtorch-cuda130-${LIBTORCH_VERSION}.zip" && \
    if [ ! -f "$CACHE_FILE" ]; then \
        wget -q ${LIBTORCH_CUDA130_URL} -O "$CACHE_FILE"; \
    fi && \
    unzip -q "$CACHE_FILE" -d /opt

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

#############################################
# ROCm 7.1 Builder
#############################################

FROM builder-base AS builder-rocm71
ARG LIBTORCH_ROCM_URL
ARG LIBTORCH_VERSION

ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Download and extract LibTorch for ROCm 7.1 (cached across builds)
# Note: libtorch ROCm build bundles HIP/ROCm libraries
RUN --mount=type=cache,target=/tmp/libtorch-cache \
    CACHE_FILE="/tmp/libtorch-cache/libtorch-rocm71-${LIBTORCH_VERSION}.zip" && \
    if [ ! -f "$CACHE_FILE" ]; then \
        wget -q ${LIBTORCH_ROCM_URL} -O "$CACHE_FILE"; \
    fi && \
    unzip -q "$CACHE_FILE" -d /opt

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

#############################################
# CPU Builder
#############################################

FROM builder-base AS builder-cpu
ARG LIBTORCH_CPU_URL
ARG LIBTORCH_VERSION

# Download and extract LibTorch for CPU (cached across builds)
RUN --mount=type=cache,target=/tmp/libtorch-cache \
    CACHE_FILE="/tmp/libtorch-cache/libtorch-cpu-${LIBTORCH_VERSION}.zip" && \
    if [ ! -f "$CACHE_FILE" ]; then \
        wget -q ${LIBTORCH_CPU_URL} -O "$CACHE_FILE"; \
    fi && \
    unzip -q "$CACHE_FILE" -d /opt

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

#############################################
# CPU Builder (aarch64 / arm64)
#############################################
#
# There is NO aarch64 libtorch zip at download.pytorch.org/libtorch/cpu — that
# URL is x86_64-only. The PyTorch aarch64 manylinux_2_28 pip wheel, however,
# bundles a complete CPU libtorch (torch/lib/*.so + torch/include), so we install
# torch via pip and point LIBTORCH at the wheel's torch dir. bookworm ships
# glibc 2.36, satisfying the wheel's manylinux_2_28 (glibc 2.28+) floor.
#
# This stage is the toolchain+libtorch image only; the actual cargo build runs
# either in the `builder` stage below (VARIANT=cpu-arm64) or by mounting the
# workspace into this image (`container:`-style, as the merge-gate would).

FROM builder-base AS builder-cpu-arm64
ARG LIBTORCH_VERSION

# tch-rs's fork keys its ABI check to the exact libtorch version string; the pip
# wheel reports the same 2.10.0 but bypass keeps us resilient to wheel-suffix skew.
ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Python + pip to fetch the aarch64 torch wheel (bookworm python3 == 3.11 -> cp311).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install the CPU torch wheel (aarch64) and expose its bundled libtorch as
# /opt/libtorch so the shared `builder` stage's LIBTORCH=/opt/libtorch just works.
# --break-system-packages: bookworm marks the system python PEP-668 externally-managed.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages "torch==${LIBTORCH_VERSION}" \
    && TORCH_DIR="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')" \
    && ln -s "$TORCH_DIR" /opt/libtorch \
    && ls -la /opt/libtorch/lib/libtorch_cpu.so /opt/libtorch/include >/dev/null

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# CI tooling for the merge-gate job (rust.yml), baked here so the job does not
# fetch anything over the network at runtime (removes per-run round-trips and
# their flake/supply-chain risk; #1011 follow-up):
#   - git-lfs: git2db's LFS worktree tests shell out to the git-lfs binary.
#   - cargo-nextest: pinned + SHA-256 verified (never get.nexte.st/latest — a
#     mutable fetch that would otherwise run in an AWS-credentialed container).
ARG NEXTEST_VERSION=0.9.140
ARG NEXTEST_SHA256=8b3f4d4560b6b0f83774fecc6be07e47716dbad0eb0bb6c3890f478f4affe4b6
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && git lfs install --system --skip-repo \
    && rm -rf /var/lib/apt/lists/* && apt-get clean \
    && curl -fsSL "https://get.nexte.st/${NEXTEST_VERSION}/linux-arm" -o /tmp/nextest.tar.gz \
    && echo "${NEXTEST_SHA256}  /tmp/nextest.tar.gz" | sha256sum -c - \
    && tar zxf /tmp/nextest.tar.gz -C /root/.cargo/bin \
    && rm -f /tmp/nextest.tar.gz \
    && cargo-nextest --version && git-lfs --version

# Pre-provision the repo's pinned Rust toolchain (channel + components + targets)
# from rust-toolchain.toml so the merge gate does NOT rustup-download it at
# runtime. Channel is read from the file (no version duplication); the component/
# target adds operate on that active toolchain. build-image.yml rebuilds on
# rust-toolchain.toml changes so the baked toolchain stays in sync with the pin.
COPY rust-toolchain.toml /tmp/tc/rust-toolchain.toml
RUN cd /tmp/tc \
    && rustup show \
    && rustup component add clippy rustfmt \
    && rustup target add wasm32-unknown-unknown wasm32-wasip1 \
    && rustup target list --installed | grep -qx wasm32-wasip1 \
    && rm -rf /tmp/tc

#############################################
# Select Builder Based on Variant
#############################################

FROM builder-${VARIANT} AS builder

# Set working directory
WORKDIR /build

# Copy project files. Cargo.lock is REQUIRED: without it `cargo build` regenerates
# the lock from the registry and the `duckdb`/`datafusion` caret requirements can
# float across an arrow major boundary (duckdb 1.4.3 -> 1.10504.x pulls arrow 58,
# while datafusion 50 stays on arrow 56), producing E0308 at the MemTable/DFSchema
# boundaries. See crates/hyprstream-metrics/Cargo.toml and PR fixing #release-image.
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY crates ./crates

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# Build the project with BuildKit cache mounts for Cargo registry and sccache
# LIBTORCH is already set in the variant-specific builder stages
# We do NOT use LIBTORCH_USE_PYTORCH since we're using manual downloads
# Note: --no-default-features excludes systemd (not needed in containers)
# Cache mounts:
#   - /root/.cargo/registry: Cargo crate registry
#   - /root/.cargo/git: Git dependencies
#   - /sccache: Compiled artifacts (sccache)
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/sccache \
    OPENSSL_NO_VENDOR=1 cargo build --release --no-default-features --features otel,gittorrent,xet

#############################################
# Runtime Stage Selection (Distroless)
#############################################

#############################################
# CUDA 12.8 Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-cuda128

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcrypto.so* /usr/lib/x86_64-linux-gnu/

# Copy CUDA runtime libraries from builder (toolkit includes runtime)
COPY --from=builder /usr/local/cuda-12.8/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-12.8/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-12.8/lib64/libcublasLt.so* /usr/local/cuda/lib64/

# Copy entire LibTorch lib directory
COPY --from=builder /opt/libtorch/lib/ /opt/libtorch/lib/

#############################################
# CUDA 13.0 Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-cuda130

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcrypto.so* /usr/lib/x86_64-linux-gnu/

# Copy CUDA runtime libraries from builder (toolkit includes runtime)
COPY --from=builder /usr/local/cuda-13.0/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-13.0/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-13.0/lib64/libcublasLt.so* /usr/local/cuda/lib64/

# Copy entire LibTorch lib directory
COPY --from=builder /opt/libtorch/lib/ /opt/libtorch/lib/

#############################################
# ROCm 7.1 Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-rocm71

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcrypto.so* /usr/lib/x86_64-linux-gnu/

# Copy entire LibTorch lib directory (includes Tensile libraries for ROCm)
COPY --from=builder /opt/libtorch/lib/ /opt/libtorch/lib/

#############################################
# CPU Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-cpu

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcrypto.so* /usr/lib/x86_64-linux-gnu/

# Copy entire LibTorch lib directory
COPY --from=builder /opt/libtorch/lib/ /opt/libtorch/lib/

#############################################
# CPU Runtime (aarch64 / arm64)
#############################################
#
# Variant of runtime-cpu for the builder-cpu-arm64 toolchain (PyTorch aarch64
# pip wheel's libtorch). Debian multilib lives under /usr/lib/aarch64-linux-gnu
# on arm64, so the system-library COPYs cannot share the x86_64 runtime stage.
# Selected via `FROM runtime-${VARIANT}` when VARIANT=cpu-arm64. gcr.io/distroless
# cc-debian12 is multi-arch, so the base resolves to its arm64 variant natively.

FROM gcr.io/distroless/cc-debian12 AS runtime-cpu-arm64

# Copy required system libraries (arm64 multilib path)
COPY --from=builder /usr/lib/aarch64-linux-gnu/libgomp.so.1 /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libz.so.1 /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libssl.so* /usr/lib/aarch64-linux-gnu/
COPY --from=builder /usr/lib/aarch64-linux-gnu/libcrypto.so* /usr/lib/aarch64-linux-gnu/

# Copy entire LibTorch lib directory
COPY --from=builder /opt/libtorch/lib/ /opt/libtorch/lib/

#############################################
# Final Runtime
#############################################

FROM runtime-${VARIANT} AS runtime

# Copy binary from builder
COPY --from=builder /build/target/release/hyprstream /hyprstream

# Set library paths
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:/usr/local/cuda/lib64

# Expose default ports
EXPOSE 8080 50051

# Run hyprstream (distroless uses absolute paths)
ENTRYPOINT ["/hyprstream"]
