# Multi-stage build for Hyprstream
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install PyTorch (for building)
RUN pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Set working directory
WORKDIR /build

# Copy project files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build with system PyTorch
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
RUN cargo build --release

# Runtime stage
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    libgomp1 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch runtime
RUN pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Copy binary from builder
COPY --from=builder /build/target/release/hyprstream /usr/local/bin/hyprstream

# Set up PyTorch library path
ENV LIBTORCH_USE_PYTORCH=1
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# Create directories for models and data
RUN mkdir -p /var/lib/hyprstream/models /var/lib/hyprstream/data
WORKDIR /var/lib/hyprstream

# Expose default ports
EXPOSE 8080 50051

# Run hyprstream
ENTRYPOINT ["hyprstream"]
CMD ["server"]
