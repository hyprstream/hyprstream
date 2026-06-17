# libtorch.nix - Fetch and unpack libtorch variants from PyTorch CDN
#
# tch-rs needs LIBTORCH pointing to a directory containing lib/, include/, share/.
# We fetch the zip with fetchurl (known SHA256), then unpack with stdenv.mkDerivation.
#
# CUDA variants: autoAddDriverRunpath patches bundled libcudart.so et al so they can
# find libcuda.so.1 from the NVIDIA driver at /run/opengl-driver/lib (NixOS standard).
#
# ROCm variant: most HIP runtime libs are bundled in the zip. The AMD kernel driver
# interface (libdrm_amdgpu.so, /dev/kfd) must be provided by the host NixOS modules
# (hardware.amdgpu.enable). No autoAddDriverRunpath equivalent exists for AMD.
{ lib, stdenv, fetchurl, unzip, autoAddDriverRunpath }:

let
  version = "2.10.0";

  mkLibtorchVariant = { variant, url, sha256, nativeBuildInputs ? [] }:
    let
      zip = fetchurl {
        name = "libtorch-${variant}-${version}.zip";
        inherit url sha256;
      };
    in stdenv.mkDerivation {
      name = "libtorch-${variant}-${version}";
      src = zip;

      nativeBuildInputs = [ unzip ] ++ nativeBuildInputs;

      dontUnpack = true;
      dontConfigure = true;

      buildPhase = ''
        runHook preBuild
        mkdir -p "$TMPDIR/unpack"
        unzip -q "$src" -d "$TMPDIR/unpack"
        runHook postBuild
      '';

      installPhase = ''
        runHook preInstall
        mv "$TMPDIR/unpack/libtorch" "$out"
        runHook postInstall
      '';
    };

in {
  cpu = mkLibtorchVariant {
    variant = "cpu";
    url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${version}%2Bcpu.zip";
    sha256 = "c5bf8efda9224a2d971b19d1ef6cf3ba6fee8ab53e69c49427db003d1d300496";
  };

  # CUDA 12.8: autoAddDriverRunpath patches bundled .so files to find libcuda.so.1
  # from the NVIDIA driver at /run/opengl-driver/lib on NixOS.
  cuda128 = mkLibtorchVariant {
    variant = "cuda128";
    url = "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${version}%2Bcu128.zip";
    sha256 = "429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9";
    nativeBuildInputs = [ autoAddDriverRunpath ];
  };

  # CUDA 13.0: same driver runpath treatment as cuda128
  cuda130 = mkLibtorchVariant {
    variant = "cuda130";
    url = "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${version}%2Bcu130.zip";
    sha256 = "7ca9216c5eecc39d61ef550cd50988f651bbe3982a2dcf4fd5982dc5dfce4ca0";
    nativeBuildInputs = [ autoAddDriverRunpath ];
  };

  # ROCm 7.1: bundled HIP runtime is sufficient; AMD driver provided by host OS
  rocm71 = mkLibtorchVariant {
    variant = "rocm71";
    url = "https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-${version}%2Brocm7.1.zip";
    sha256 = "605532aeea2e22b639c2c4c239d2994f040457adff1a22cfb4c6d12b4b9641f7";
  };
}
