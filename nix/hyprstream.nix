# hyprstream.nix - Main HyprStream derivation, parameterized by variant
#
# Usage (from flake.nix):
#   callPackage ./nix/hyprstream.nix { variant = "cpu"; libtorch = ...; }
{ lib
, stdenv
, rustPlatform
, pkg-config
, cmake
, perl
, capnproto
, openssl
, zlib
, libzmq
, git
, libtorch
, variant ? "cpu"
}:

rustPlatform.buildRustPackage rec {
  pname = "hyprstream-${variant}";
  version = "0.3.0-rc1";

  # Source is the repo root (parent of nix/)
  src = lib.cleanSourceWith {
    src = ./..;
    filter = path: type:
      let name = baseNameOf (toString path); in
      name != ".git"
      && name != ".worktrees"
      && name != "target"
      && name != "libtorch-cache"
      && name != "appimage-build";
  };

  cargoLock = {
    lockFile = ./../Cargo.lock;
    outputHashes = {
      # HyprStream fork of tch-rs (tch + torch-sys)
      "tch-0.23.0" = lib.fakeHash;
      "torch-sys-0.23.0" = lib.fakeHash;
      # HyprStream fork of tmq (ZeroMQ bindings)
      "tmq-0.5.0" = lib.fakeHash;
      # xet-core (HuggingFace large file storage)
      "cas_client-0.14.5" = lib.fakeHash;
      "cas_object-0.1.0" = lib.fakeHash;
      "cas_types-0.1.0" = lib.fakeHash;
      "chunk_cache-0.1.0" = lib.fakeHash;
      "data-0.14.5" = lib.fakeHash;
      "deduplication-0.14.5" = lib.fakeHash;
      "error_printer-0.14.5" = lib.fakeHash;
      "file_utils-0.14.2" = lib.fakeHash;
      "hub_client-0.1.0" = lib.fakeHash;
      "mdb_shard-0.14.5" = lib.fakeHash;
      "merklehash-0.14.5" = lib.fakeHash;
      "progress_tracking-0.1.0" = lib.fakeHash;
      "utils-0.14.5" = lib.fakeHash;
      "xet_config-0.14.5" = lib.fakeHash;
      "xet_runtime-0.1.0" = lib.fakeHash;
      # kata-containers (worker isolation)
      "api_client-0.1.0" = lib.fakeHash;
      "ch-config-0.1.0" = lib.fakeHash;
      "dbs-address-space-0.3.0" = lib.fakeHash;
      "dbs-allocator-0.1.1" = lib.fakeHash;
      "dbs-arch-0.2.3" = lib.fakeHash;
      "dbs-boot-0.4.0" = lib.fakeHash;
      "dbs-device-0.2.0" = lib.fakeHash;
      "dbs-interrupt-0.2.2" = lib.fakeHash;
      "dbs-legacy-devices-0.1.1" = lib.fakeHash;
      "dbs-pci-0.1.0" = lib.fakeHash;
      "dbs-upcall-0.3.0" = lib.fakeHash;
      "dbs-utils-0.2.1" = lib.fakeHash;
      "dbs-virtio-devices-0.3.1" = lib.fakeHash;
      "dragonball-0.1.0" = lib.fakeHash;
      "hypervisor-0.1.0" = lib.fakeHash;
      "kata-sys-util-0.1.0" = lib.fakeHash;
      "kata-types-0.1.0" = lib.fakeHash;
      "logging-0.1.0" = lib.fakeHash;
      "persist-0.1.0" = lib.fakeHash;
      "protocols-0.1.0" = lib.fakeHash;
      "runtime-spec-0.1.0" = lib.fakeHash;
      "safe-path-0.1.0" = lib.fakeHash;
      "shim-interface-0.1.0" = lib.fakeHash;
      "tests_utils-0.1.0" = lib.fakeHash;
      # nydus (container image service)
      "nydus-api-0.4.0" = lib.fakeHash;
      "nydus-rafs-0.4.0" = lib.fakeHash;
      "nydus-service-0.4.0" = lib.fakeHash;
      "nydus-storage-0.7.1" = lib.fakeHash;
      "nydus-upgrade-0.2.0" = lib.fakeHash;
      "nydus-utils-0.5.0" = lib.fakeHash;
    };
  };

  # Build-time tool dependencies
  nativeBuildInputs = [
    pkg-config
    cmake
    perl
    capnproto   # capnpc schema compiler
  ];

  # Runtime / link-time library dependencies
  buildInputs = [
    openssl
    zlib
    libzmq
    git
  ];

  # tch-rs build environment - points to the extracted libtorch directory
  LIBTORCH = "${libtorch}";
  LD_LIBRARY_PATH = "${libtorch}/lib";
  LIBTORCH_BYPASS_VERSION_CHECK = "1";

  # Git dependencies require CLI fetch (Nix sandbox disables libgit2 SSH)
  CARGO_NET_GIT_FETCH_WITH_CLI = "true";

  # Build only the main hyprstream binary and its workspace deps
  cargoBuildFlags = [ "-p" "hyprstream" ];
  cargoTestFlags = [ "-p" "hyprstream" ];

  # Passthru for downstream use (e.g. NixOS module)
  passthru = {
    inherit variant libtorch;
  };

  meta = with lib; {
    description = "Agentic infrastructure for continuously learning applications";
    homepage = "https://github.com/hyprstream/hyprstream";
    license = with licenses; [ mit agpl3Plus ];
    platforms = [ "x86_64-linux" ];
    mainProgram = "hyprstream";
  };
}
