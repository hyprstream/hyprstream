# libtorch.nix - Fetch and unpack libtorch variants from PyTorch CDN
#
# tch-rs needs LIBTORCH pointing to a directory containing lib/, include/, share/.
# We fetch the zip with fetchurl (known SHA256), then unpack with stdenv.mkDerivation.
#
# CUDA variants: autoAddDriverRunpath patches bundled libcudart.so et al so they can
# find libcuda.so.1 from the NVIDIA driver at /run/opengl-driver/lib (NixOS standard).
#
# ROCm variant: most HIP runtime libs are bundled in the zip. libamd_comgr.so and
# libelf.so within the bundle need libz.so.1, which is not bundled. We patch their
# RPATH to point at the Nix zlib store path in postInstall.
{ lib, stdenv, fetchurl, unzip, autoAddDriverRunpath, patchelf, zlib }:

let
  version = "2.11.0";

  mkLibtorchVariant = { variant, url, sha256, nativeBuildInputs ? [], postInstall ? "" }:
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

      inherit postInstall;
    };

in {
  cpu = mkLibtorchVariant {
    variant = "cpu";
    url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${version}%2Bcpu.zip";
    sha256 = "ee3f453f6c3e3d934339875d76a2b04d6a1731554c016fc82be1f4b8d1d4ae62";
  };

  # CUDA 12.8: autoAddDriverRunpath patches bundled .so files to find libcuda.so.1
  # from the NVIDIA driver at /run/opengl-driver/lib on NixOS.
  cuda128 = mkLibtorchVariant {
    variant = "cuda128";
    url = "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${version}%2Bcu128.zip";
    sha256 = "e4732501926848c1fdec7411bf78952cd60b353ed990439eefacfebd34bc1351";
    nativeBuildInputs = [ autoAddDriverRunpath ];
  };

  # CUDA 13.0: same driver runpath treatment as cuda128
  cuda130 = mkLibtorchVariant {
    variant = "cuda130";
    url = "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${version}%2Bcu130.zip";
    sha256 = "a163eff74ffc1eaf3827e808c8bad3a88338ca68b5733d0974c1cbc9bc033295";
    nativeBuildInputs = [ autoAddDriverRunpath ];
  };

  # ROCm 7.1: libamd_comgr.so and libelf.so in the bundle need libz but don't bundle it.
  # Patch their DT_RPATH so the Nix dynamic linker can find libz.so.1 when they're loaded
  # as transitive deps (DT_RUNPATH on the final binary doesn't propagate to transitive deps).
  rocm71 = mkLibtorchVariant {
    variant = "rocm71";
    url = "https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-${version}%2Brocm7.1.zip";
    sha256 = "6d1bee01dea19ceb46c846c3fc7a2b71c816dabcfbc5da9b4266a41b1dc9f5eb";
    nativeBuildInputs = [ patchelf ];
    postInstall = ''
      # Match both unversioned (.so) and versioned (.so.1, .so.1.2, ...) sonames,
      # since the libtorch ROCm bundle may ship either form; an unversioned-only
      # match would silently skip versioned libraries.
      for so in "$out"/lib/libamd_comgr.so* "$out"/lib/libelf.so*; do
        [ -f "$so" ] && ${patchelf}/bin/patchelf --add-rpath "${zlib}/lib" "$so"
      done
    '';
  };
}
