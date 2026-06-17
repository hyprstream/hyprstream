# libtorch.nix - Fetch and unpack libtorch variants from PyTorch CDN
#
# tch-rs needs LIBTORCH pointing to a directory containing lib/, include/, share/.
# We fetch the zip with fetchurl (known SHA256), then unpack with runCommand.
{ lib, stdenv, fetchurl, unzip, runCommand }:

let
  version = "2.10.0";

  # Download zip -> unpack -> return derivation with libtorch/ contents
  mkLibtorchVariant = { variant, url, sha256 }:
    let
      zip = fetchurl {
        name = "libtorch-${variant}-${version}.zip";
        inherit url sha256;
      };
    in runCommand "libtorch-${variant}-${version}" { } ''
      mkdir -p $out $TMPDIR/unpack
      ${unzip}/bin/unzip -q ${zip} -d $TMPDIR/unpack
      mv $TMPDIR/unpack/libtorch/* $out/
    '';

in {
  cpu = mkLibtorchVariant {
    variant = "cpu";
    url = "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${version}%2Bcpu.zip";
    sha256 = "c5bf8efda9224a2d971b19d1ef6cf3ba6fee8ab53e69c49427db003d1d300496";
  };

  cuda128 = mkLibtorchVariant {
    variant = "cuda128";
    url = "https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-${version}%2Bcu128.zip";
    sha256 = "429aa9fead3cf3d557e7c310442a1fae3879cdc14a469ff452043b39b61666a9";
  };

  cuda130 = mkLibtorchVariant {
    variant = "cuda130";
    url = "https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-${version}%2Bcu130.zip";
    sha256 = "7ca9216c5eecc39d61ef550cd50988f651bbe3982a2dcf4fd5982dc5dfce4ca0";
  };

  rocm71 = mkLibtorchVariant {
    variant = "rocm71";
    url = "https://download.pytorch.org/libtorch/rocm7.1/libtorch-shared-with-deps-${version}%2Brocm7.1.zip";
    sha256 = "605532aeea2e22b639c2c4c239d2994f040457adff1a22cfb4c6d12b4b9641f7";
  };
}
