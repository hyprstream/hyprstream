# syntax=docker/dockerfile:1
# Multi-variant build for Hyprstream supporting CPU, CUDA, and ROCm

ARG VARIANT=cpu
ARG DEBIAN_VERSION=bookworm

# LibTorch download URLs for manual installation
ARG LIBTORCH_CUDA_URL=https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.9.0%2Bcu130.zip
#ARG LIBTORCH_CUDA_URL=https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip
ARG LIBTORCH_ROCM_URL=https://download.pytorch.org/libtorch/nightly/rocm7.0/libtorch-shared-with-deps-latest.zip
ARG LIBTORCH_CPU_URL=https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip

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
    git \
    dialog \
    rsync \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

#############################################
# CUDA Builder
#############################################

FROM builder-base AS builder-cuda
ARG LIBTORCH_CUDA_URL

ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Install CUDA repository and runtime libraries (needed for linking)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y \
    cuda-toolkit-13-0 \
    && rm -rf /var/lib/apt/lists/*

# Download and extract LibTorch for CUDA
RUN wget -q ${LIBTORCH_CUDA_URL} -O libtorch.zip && \
    unzip -q libtorch.zip -d /opt && \
    rm libtorch.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

#############################################
# ROCm Builder
#############################################

FROM builder-base AS builder-rocm
ARG LIBTORCH_ROCM_URL

ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Download and extract LibTorch for ROCm
RUN wget -q ${LIBTORCH_ROCM_URL} -O libtorch.zip && \
    unzip -q libtorch.zip -d /opt && \
    rm libtorch.zip

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib:$LD_LIBRARY_PATH

#############################################
# CPU Builder
#############################################

FROM builder-base AS builder-cpu
ARG LIBTORCH_CPU_URL

# Download and extract LibTorch for CPU
RUN wget -q ${LIBTORCH_CPU_URL} -O libtorch.zip && \
    unzip -q libtorch.zip -d /opt && \
    rm libtorch.zip

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

# Build the project
# LIBTORCH is already set in the variant-specific builder stages
# We do NOT use LIBTORCH_USE_PYTORCH since we're using manual downloads
RUN cargo build --release --features otel

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
    && rm -rf /var/lib/apt/lists/*

#############################################
# CUDA Runtime
#############################################

FROM runtime-base AS runtime-cuda

# Install CUDA repository and runtime libraries
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y \
    cuda-runtime-13-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy LibTorch libraries from builder
COPY --from=builder /opt/libtorch /opt/libtorch

#############################################
# ROCm Runtime
#############################################

FROM runtime-base AS runtime-rocm

RUN wget https://repo.radeon.com/amdgpu-install/7.0.2/ubuntu/jammy/amdgpu-install_7.0.2.70002-1_all.deb && \
    apt update && \
    apt install -y ./amdgpu-install_7.0.2.70002-1_all.deb && \
    apt install -y python3-setuptools python3-wheel rsync dialog && \
    apt update && \
    apt install -y rocm && \
    rm -rf /var/lib/apt/lists/*;

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
CMD ["server"]
