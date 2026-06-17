# HyprStream Nix Flake
#
# Multi-variant packaging: CPU, CUDA 12.8, CUDA 13.0, ROCm 7.1
#
# Uses crane for Rust builds -- vendors ALL cargo deps under a single hash
# instead of 47 separate git dependency hashes.
#
# Usage:
#   nix build .#hyprstream-cpu
#   nix build .#hyprstream-cuda128
#   nix build .#hyprstream-cuda130
#   nix build .#hyprstream-rocm71
#   nix build .#hyprstream         # alias for cpu
#   nix develop                    # dev shell
#
# On first build, crane will error with the correct cargoHash.
# Replace lib.fakeHash in the hash line with the reported hash.
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

        # Fetch libtorch variants
        libtorchVariants = import ./nix/libtorch.nix {
          inherit (pkgs) lib stdenv fetchurl unzip runCommand;
        };

        # Rust toolchain from fenix (stable channel)
        rustToolchain = fenix.packages.${system}.stable.toolchain;

        # Crane instance with our toolchain
        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Common source filtering -- exclude build artifacts, git internals
        src = craneLib.cleanCargoSource ./.;

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
          ];

          buildInputs = with pkgs; [
            openssl
            zlib
            zeromq
            git
            systemd
          ];

          # Git deps require CLI fetch (Nix sandbox disables libgit2 SSH)
          CARGO_NET_GIT_FETCH_WITH_CLI = "true";

          # Build only the main binary
          cargoExtraArgs = "-p hyprstream";

          # Only hash of all cargo deps (registry + git) -- single FOD
          # On first build, replace lib.fakeHash with the real hash from the error
          cargoHash = pkgs.lib.fakeHash;
        };

        # Build a variant by overlaying LIBTORCH env vars
        mkHyprstream = variant: libtorch:
          let
            craneArgs = commonArgs // {
              pname = "hyprstream-${variant}";

              # tch-rs needs LIBTORCH pointing to extracted libtorch directory
              LIBTORCH = "${libtorch}";
              LD_LIBRARY_PATH = "${libtorch}/lib";
              LIBTORCH_BYPASS_VERSION_CHECK = "1";

              # Passthru for downstream use
              passthru = { inherit variant libtorch; };
            };
          in craneLib.buildPackage craneArgs;

      in {
        packages = {
          hyprstream = mkHyprstream "cpu" libtorchVariants.cpu;
          hyprstream-cpu = mkHyprstream "cpu" libtorchVariants.cpu;
          hyprstream-cuda128 = mkHyprstream "cuda128" libtorchVariants.cuda128;
          hyprstream-cuda130 = mkHyprstream "cuda130" libtorchVariants.cuda130;
          hyprstream-rocm71 = mkHyprstream "rocm71" libtorchVariants.rocm71;
        };

        devShells.default = let
          devLibtorch = libtorchVariants.cpu;
        in pkgs.mkShell {
          name = "hyprstream-dev";

          inputsFrom = [ (mkHyprstream "cpu" devLibtorch) ];

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
