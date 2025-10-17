#!/bin/bash
# Build multi-variant Docker images for hyprstream
# Usage: ./build-docker.sh <cpu|cuda|rocm|all> [--push]

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="${IMAGE_NAME:-hyprstream}"
REGISTRY="${REGISTRY:-}"
VERSION=$(grep '^version = ' crates/hyprstream/Cargo.toml | sed 's/version = "\(.*\)"/\1/')

# Default versions - can be overridden via environment variables
PYTORCH_CUDA_TAG="${PYTORCH_CUDA_TAG:-2.8.0-cuda12.9-cudnn9-devel}"
PYTORCH_CUDA_RUNTIME_TAG="${PYTORCH_CUDA_RUNTIME_TAG:-2.8.0-cuda12.9-cudnn9-runtime}"
ROCM_PYTORCH_TAG="${ROCM_PYTORCH_TAG:-latest}"
DEBIAN_VERSION="${DEBIAN_VERSION:-bookworm}"
PYTORCH_CPU_VERSION="${PYTORCH_CPU_VERSION:-2.8.0}"

# Build options
DOCKER_BUILD_OPTS="${DOCKER_BUILD_OPTS:-}"

echo -e "${BLUE}===================================${NC}"
echo -e "${BLUE}Hyprstream Docker Build${NC}"
echo -e "${BLUE}===================================${NC}"
echo -e "Version: ${GREEN}${VERSION}${NC}"
echo -e "Image: ${GREEN}${IMAGE_NAME}${NC}"
echo -e "CUDA Tag: ${YELLOW}${PYTORCH_CUDA_TAG}${NC}"
echo -e "CUDA Runtime: ${YELLOW}${PYTORCH_CUDA_RUNTIME_TAG}${NC}"
echo -e "ROCm Tag: ${YELLOW}${ROCM_PYTORCH_TAG}${NC}"
echo -e "Ubuntu: ${YELLOW}${UBUNTU_VERSION}${NC}"
echo -e "PyTorch CPU: ${YELLOW}${PYTORCH_CPU_VERSION}${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

build_variant() {
    local variant=$1
    echo -e "${BLUE}Building ${variant} variant...${NC}"

    local full_image_name="${IMAGE_NAME}"
    if [ -n "${REGISTRY}" ]; then
        full_image_name="${REGISTRY}/${IMAGE_NAME}"
    fi

    local start_time=$(date +%s)

    docker build \
        --build-arg VARIANT=${variant} \
        --build-arg PYTORCH_CUDA_TAG=${PYTORCH_CUDA_TAG} \
        --build-arg PYTORCH_CUDA_RUNTIME_TAG=${PYTORCH_CUDA_RUNTIME_TAG} \
        --build-arg ROCM_PYTORCH_TAG=${ROCM_PYTORCH_TAG} \
        --build-arg DEBIAN_VERSION=${DEBIAN_VERSION} \
        --build-arg PYTORCH_CPU_VERSION=${PYTORCH_CPU_VERSION} \
        -t ${full_image_name}:${variant} \
        -t ${full_image_name}:${VERSION}-${variant} \
        -t ${full_image_name}:latest-${variant} \
        -f Dockerfile \
	--no-cache \
        ${DOCKER_BUILD_OPTS} \
        .

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo -e "${GREEN}✓ Built ${full_image_name}:${variant} in ${duration}s${NC}"
    echo ""
}

push_variant() {
    local variant=$1
    local full_image_name="${IMAGE_NAME}"
    if [ -n "${REGISTRY}" ]; then
        full_image_name="${REGISTRY}/${IMAGE_NAME}"
    fi

    echo -e "${BLUE}Pushing ${variant} variant...${NC}"
    docker push ${full_image_name}:${variant}
    docker push ${full_image_name}:${VERSION}-${variant}
    docker push ${full_image_name}:latest-${variant}
    echo -e "${GREEN}✓ Pushed ${variant} variant${NC}"
}

show_usage() {
    echo "Usage: $0 <cpu|cuda|rocm|all> [--push]"
    echo ""
    echo "Examples:"
    echo "  $0 cpu              # Build CPU variant only"
    echo "  $0 cuda             # Build CUDA variant only"
    echo "  $0 rocm             # Build ROCm variant only"
    echo "  $0 all              # Build all variants"
    echo "  $0 all --push       # Build all variants and push to registry"
    echo ""
    echo "Environment variables:"
    echo "  IMAGE_NAME                   Image name (default: hyprstream)"
    echo "  REGISTRY                     Docker registry (default: none)"
    echo "  PYTORCH_CUDA_TAG             PyTorch CUDA devel tag (default: ${PYTORCH_CUDA_TAG})"
    echo "  PYTORCH_CUDA_RUNTIME_TAG     PyTorch CUDA runtime tag (default: ${PYTORCH_CUDA_RUNTIME_TAG})"
    echo "  ROCM_PYTORCH_TAG             ROCm PyTorch tag (default: ${ROCM_PYTORCH_TAG})"
    echo "  UBUNTU_VERSION               Ubuntu version (default: ${UBUNTU_VERSION})"
    echo "  PYTORCH_CPU_VERSION          PyTorch CPU version (default: ${PYTORCH_CPU_VERSION})"
    echo "  DOCKER_BUILD_OPTS            Additional docker build options (default: none)"
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

VARIANT=$1
PUSH=${2:-}

case $VARIANT in
    cpu)
        build_variant cpu
        [ "$PUSH" = "--push" ] && push_variant cpu
        ;;
    cuda)
        build_variant cuda
        [ "$PUSH" = "--push" ] && push_variant cuda
        ;;
    rocm)
        build_variant rocm
        [ "$PUSH" = "--push" ] && push_variant rocm
        ;;
    all)
        build_variant cpu
        build_variant cuda
        build_variant rocm

        # Tag cpu as default latest
        echo -e "${BLUE}Tagging CPU variant as latest...${NC}"
        docker tag ${IMAGE_NAME}:cpu ${IMAGE_NAME}:latest
        echo -e "${GREEN}✓ Tagged ${IMAGE_NAME}:latest -> ${IMAGE_NAME}:cpu${NC}"
        echo ""

        if [ "$PUSH" = "--push" ]; then
            push_variant cpu
            push_variant cuda
            push_variant rocm

            echo -e "${BLUE}Pushing latest tag...${NC}"
            docker push ${IMAGE_NAME}:latest
            echo -e "${GREEN}✓ Pushed latest tag${NC}"
        fi
        ;;
    *)
        echo -e "${RED}❌ Invalid variant: $VARIANT${NC}"
        echo "Valid options: cpu, cuda, rocm, all"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}===================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${BLUE}===================================${NC}"
echo -e "Available images:"
docker images | grep ${IMAGE_NAME} | head -15
echo ""
echo -e "${YELLOW}Usage examples:${NC}"
echo -e "  docker run --rm ${IMAGE_NAME}:cpu --help"
echo -e "  docker run --rm --gpus all ${IMAGE_NAME}:cuda --help"
echo -e "  docker run --rm --device=/dev/kfd --device=/dev/dri ${IMAGE_NAME}:rocm --help"
