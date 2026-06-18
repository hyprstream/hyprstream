# HyprStream Nix Flake
#
# Multi-variant packaging: CPU, CUDA 12.8, CUDA 13.0, ROCm 7.1
#
# Uses crane for Rust builds with pre-vendored cargo deps.
# All registry and git deps are fetched in an isolated vendoring step.
#
# Usage:
#   nix build .#hyprstream-cpu
#   nix build .#hyprstream-cuda128
#   nix build .#hyprstream-cuda130
#   nix build .#hyprstream-rocm71
#   nix build .#hyprstream         # alias for cpu
#   nix develop                    # dev shell
{
  description = "HyprStream - Agentic infrastructure for continuously learning applications";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, crane, fenix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Fetch libtorch variants (GPU variants apply RPATH patches internally)
        libtorchVariants = import ./nix/libtorch.nix {
          inherit (pkgs) lib stdenv fetchurl unzip autoAddDriverRunpath patchelf zlib;
        };

        # Rust toolchain from fenix (stable channel)
        rustToolchain = fenix.packages.${system}.stable.toolchain;

        # Crane instance with our toolchain
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Source filtering: standard cargo sources + .capnp schemas (needed by hyprstream-rpc build.rs)
        src = pkgs.lib.cleanSourceWith {
          src = craneLib.path ./.;
          filter = path: type:
            (craneLib.filterCargoSources path type) ||
            (builtins.match ".*\\.capnp$" path != null);
        };

        # Pre-vendor cargo deps with fixes for broken git dependencies
        cargoVendorDir = craneLib.vendorCargoDeps {
          inherit src;
          # Fix: kata-containers shim-interface Cargo.toml references README.md
          # that doesn't exist. Override the git checkout derivation to create it.
          overrideVendorGitCheckout = _packages: drv:
            drv.overrideAttrs (old: {
              preInstall = (old.preInstall or "") + ''
                # Create missing README.md files referenced in Cargo.toml readme fields
                for toml in $(find . -name Cargo.toml); do
                  dir=$(dirname "$toml")
                  readme=$(grep -oP 'readme\s*=\s*"\K[^"]+' "$toml" 2>/dev/null || true)
                  if [ -n "$readme" ] && [ ! -f "$dir/$readme" ]; then
                    touch "$dir/$readme"
                  fi
                done

              '';
            });
        };

        # Common args shared by all variants
        commonArgs = {
          inherit src cargoVendorDir;
          pname = "hyprstream";
          version = "0.3.0-rc1";

          nativeBuildInputs = with pkgs; [
            pkg-config
            cmake
            perl
            capnproto
            protobuf
            llvmPackages.libclang
            llvmPackages.clang
            autoPatchelfHook
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

          buildInputs = with pkgs; [
            openssl
            zlib
            zeromq
            git
            systemd
            # C++ runtime needed by libtorch and kata-containers FFI
            stdenv.cc.cc.lib
          ];

          # Git deps require CLI fetch (Nix sandbox disables libgit2 SSH)
          CARGO_NET_GIT_FETCH_WITH_CLI = "true";

          # Tests require runtime libs (libstdc++, libtorch) not available in build sandbox
          doCheck = false;

          # Build only the main binary
          cargoExtraArgs = "-p hyprstream";


          # Fix: kata-containers protocols/build.rs generates protobuf .rs files
          # into src/ which is read-only in the Nix store vendor directory.
          # Copy the entire vendor dir to a writable tmpdir before building.
          preBuild = ''
            vendor_src="${builtins.toString cargoVendorDir}"
            echo "preBuild: making vendor dir writable (for kata-containers proto codegen)"
            cp -rL "$vendor_src" "$TMPDIR/writable-vendor"
            chmod -R u+w "$TMPDIR/writable-vendor"
            # Update cargo config to use the writable copy
            # crane puts cargo home at $CARGO_HOME (set by configureCargoVendoredDepsHook)
            if [ -n "''${CARGO_HOME:-}" ] && [ -f "$CARGO_HOME/config.toml" ]; then
              echo "preBuild: updating $CARGO_HOME/config.toml"
              sed -i "s|$vendor_src|$TMPDIR/writable-vendor|g" "$CARGO_HOME/config.toml"
            else
              echo "preBuild: CARGO_HOME not set or config missing, trying .cargo-home"
              for cfg in .cargo-home/config.toml .cargo/config.toml; do
                if [ -f "$cfg" ]; then
                  echo "preBuild: updating $cfg"
                  sed -i "s|$vendor_src|$TMPDIR/writable-vendor|g" "$cfg"
                fi
              done
            fi
          '';

        };

        # Build a variant by overlaying LIBTORCH env vars.
        # extraBuildInputs: GPU runtime libs (cudaPackages.*, rocmPackages.*) for RPATH
        mkHyprstream = variant: libtorch: extraBuildInputs:
          let
            craneArgs = commonArgs // {
              pname = "hyprstream-${variant}";

              # tch-rs needs LIBTORCH pointing to extracted libtorch directory
              LIBTORCH = "${libtorch}";
              LD_LIBRARY_PATH = "${libtorch}/lib";
              LIBTORCH_BYPASS_VERSION_CHECK = "1";

              # libtorch + GPU runtime libs so autoPatchelfHook sets correct RPATHs
              buildInputs = commonArgs.buildInputs ++ [ libtorch ] ++ extraBuildInputs;

              # Passthru for downstream use
              passthru = { inherit variant libtorch; };
            };
          in craneLib.buildPackage craneArgs;

        # CUDA runtime libs needed for RPATH hints (bundled CUDA toolkit in zip handles
        # the bulk; we add Nix packages so autoPatchelfHook can resolve any system deps)
        cuda12Libs = with pkgs.cudaPackages; [
          cuda_cudart libcublas libcufft libcurand libcusolver libcusparse
        ];

        # ROCm runtime libs — most HIP compute libs are bundled in the libtorch zip,
        # but rocm-runtime (libamdhip64.so) and supporting libs must come from Nix
        rocm71Libs = with pkgs.rocmPackages; [
          clr            # HIP runtime: libamdhip64.so
          rocblas
          hipblas
          miopen         # MIOpen DNN kernels
          rocrand
          hipfft
          hipsolver
          hipsparse
          rccl           # ROCm collective communications
        ];

      in {
        packages = {
          hyprstream     = mkHyprstream "cpu"     libtorchVariants.cpu     [];
          hyprstream-cpu = mkHyprstream "cpu"     libtorchVariants.cpu     [];
          hyprstream-cuda128 = mkHyprstream "cuda128" libtorchVariants.cuda128 cuda12Libs;
          hyprstream-cuda130 = mkHyprstream "cuda130" libtorchVariants.cuda130 cuda12Libs;
          hyprstream-rocm71  = mkHyprstream "rocm71"  libtorchVariants.rocm71  rocm71Libs;
        };

        devShells.default = let
          devLibtorch = libtorchVariants.cpu;
        in pkgs.mkShell {
          name = "hyprstream-dev";

          inputsFrom = [ (mkHyprstream "cpu" devLibtorch []) ];

          buildInputs = [ rustToolchain ];

          shellHook = ''
            export LIBTORCH="${devLibtorch}"
            export LD_LIBRARY_PATH="${devLibtorch}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export LIBTORCH_BYPASS_VERSION_CHECK=1
            export CARGO_NET_GIT_FETCH_WITH_CLI=true
            echo "HyprStream dev shell (CPU libtorch)"
            echo "  LIBTORCH=$LIBTORCH"
          '';
        };

        checks = {
          hyprstream-cpu = self.packages.${system}.hyprstream-cpu;
        };
      });
}
