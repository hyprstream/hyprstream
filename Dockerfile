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
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

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
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

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
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

#############################################
# ROCm 7.1 Builder
#############################################

FROM builder-base AS builder-rocm71
ARG LIBTORCH_ROCM_URL
ARG LIBTORCH_VERSION

ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Download and extract LibTorch for ROCm 7.1 (cached across builds)
RUN --mount=type=cache,target=/tmp/libtorch-cache \
    CACHE_FILE="/tmp/libtorch-cache/libtorch-rocm71-${LIBTORCH_VERSION}.zip" && \
    if [ ! -f "$CACHE_FILE" ]; then \
        wget -q ${LIBTORCH_ROCM_URL} -O "$CACHE_FILE"; \
    fi && \
    unzip -q "$CACHE_FILE" -d /opt

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

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
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

#############################################
# Select Builder Based on Variant
#############################################

FROM builder-${VARIANT} AS builder

# Set working directory
WORKDIR /build

# Copy project files
COPY Cargo.toml ./
COPY crates ./crates

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Build the project with BuildKit cache mounts for Cargo registry
# LIBTORCH is already set in the variant-specific builder stages
# We do NOT use LIBTORCH_USE_PYTORCH since we're using manual downloads
# Note: --no-default-features excludes systemd (not needed in containers)
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo build --release --no-default-features --features otel,gittorrent,xet

#############################################
# Runtime Stage Selection
#############################################

# Base runtime stage
FROM debian:${DEBIAN_VERSION} AS runtime-base

# Install common runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

#############################################
# CUDA 12.8 Runtime
#############################################

FROM runtime-base AS runtime-cuda128

# Install CUDA repository and runtime libraries
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-runtime-12-8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy LibTorch libraries from builder
COPY --from=builder /opt/libtorch /opt/libtorch

#############################################
# CUDA 13.0 Runtime
#############################################

FROM runtime-base AS runtime-cuda130

# Install CUDA repository and runtime libraries
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-runtime-13-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy LibTorch libraries from builder
COPY --from=builder /opt/libtorch /opt/libtorch

#############################################
# ROCm 7.1 Runtime
#############################################

FROM runtime-base AS runtime-rocm71

RUN wget https://repo.radeon.com/amdgpu-install/7.1/ubuntu/jammy/amdgpu-install_7.1.70100-1_all.deb && \
    apt update && \
    apt install -y ./amdgpu-install_7.1.70100-1_all.deb && \
    apt install -y python3-setuptools python3-wheel rsync dialog && \
    apt update && \
    apt install -y rocm && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy LibTorch libraries from builder
COPY --from=builder /opt/libtorch /opt/libtorch

#############################################
# CPU Runtime
#############################################

FROM runtime-base AS runtime-cpu

# Copy LibTorch libraries from builder
COPY --from=builder /opt/libtorch /opt/libtorch

#############################################
# Final Runtime
#############################################

FROM runtime-${VARIANT} AS runtime

# Copy binary from builder
COPY --from=builder /build/target/release/hyprstream /usr/local/bin/hyprstream

# Set LD_LIBRARY_PATH for LibTorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

# Create directories for models and data
RUN mkdir -p /var/lib/hyprstream/models /var/lib/hyprstream/data
WORKDIR /var/lib/hyprstream

# Expose default ports
EXPOSE 8080 50051

# Run hyprstream
ENTRYPOINT ["hyprstream"]
CMD []
