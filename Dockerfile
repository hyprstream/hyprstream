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
    cmake \
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
# Select Builder Based on Variant
#############################################

FROM builder-${VARIANT} AS builder

# Set working directory
WORKDIR /build

# Copy project files
COPY Cargo.toml ./
COPY crates ./crates

ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

# Build the project with BuildKit cache mounts for Cargo registry
# LIBTORCH is already set in the variant-specific builder stages
# We do NOT use LIBTORCH_USE_PYTORCH since we're using manual downloads
# Note: --no-default-features excludes systemd (not needed in containers)
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    cargo build --release --no-default-features --features otel,gittorrent,xet

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

# Copy CUDA runtime libraries from builder (toolkit includes runtime)
COPY --from=builder /usr/local/cuda-12.8/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-12.8/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-12.8/lib64/libcublasLt.so* /usr/local/cuda/lib64/

# Copy only LibTorch shared libraries (skip static libs, headers, cmake)
COPY --from=builder /opt/libtorch/lib/*.so* /opt/libtorch/lib/

#############################################
# CUDA 13.0 Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-cuda130

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/

# Copy CUDA runtime libraries from builder (toolkit includes runtime)
COPY --from=builder /usr/local/cuda-13.0/lib64/libcudart.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-13.0/lib64/libcublas.so* /usr/local/cuda/lib64/
COPY --from=builder /usr/local/cuda-13.0/lib64/libcublasLt.so* /usr/local/cuda/lib64/

# Copy only LibTorch shared libraries (skip static libs, headers, cmake)
COPY --from=builder /opt/libtorch/lib/*.so* /opt/libtorch/lib/

#############################################
# ROCm 7.1 Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-rocm71

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/

# Copy only LibTorch shared libraries (bundles HIP/ROCm libs)
COPY --from=builder /opt/libtorch/lib/*.so* /opt/libtorch/lib/

#############################################
# CPU Runtime
#############################################

FROM gcr.io/distroless/cc-debian12 AS runtime-cpu

# Copy required system libraries
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libz.so.1 /usr/lib/x86_64-linux-gnu/

# Copy only LibTorch shared libraries (skip static libs, headers, cmake)
COPY --from=builder /opt/libtorch/lib/*.so* /opt/libtorch/lib/

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
