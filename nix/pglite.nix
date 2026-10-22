# Provision the locked, non-ICU PGlite engine before Cargo enters the sandbox.
# pglite-rs-sys 0.2.7 overlooks compressed artifact crates during auto-discovery,
# then tries a HOME cache and network download. PGLITE_LIB_DIR bypasses that
# fallback; both upstream build scripts decompress into their writable OUT_DIR.
{ lib, stdenvNoCC, fetchurl, system, cargoLock }:

let
  targets = {
    x86_64-linux = "x86_64-unknown-linux-gnu";
    aarch64-linux = "aarch64-unknown-linux-gnu";
    x86_64-darwin = "x86_64-apple-darwin";
    aarch64-darwin = "aarch64-apple-darwin";
  };
  target = targets.${system} or (throw "Unsupported PGlite host platform: ${system}");
  lockedPackage = name:
    let matches = builtins.filter (package: package.name == name) cargoLock.package;
    in assert builtins.length matches == 1;
      builtins.head matches;
  artifact = lockedPackage "pglite-rs-lib-${target}";
  engine = lockedPackage "pglite-rs-sys";
in
assert artifact.version == engine.version;
assert artifact.source == "registry+https://github.com/rust-lang/crates.io-index";
stdenvNoCC.mkDerivation {
  pname = "pglite-native-${target}";
  inherit (artifact) version;
  src = fetchurl {
    name = "${artifact.name}-${artifact.version}.tar.gz";
    url = "https://static.crates.io/crates/${artifact.name}/${artifact.version}/download";
    sha256 = artifact.checksum;
  };

  dontConfigure = true;
  dontBuild = true;
  installPhase = ''
    runHook preInstall
    # Keep the full payload: pglite-rs also embeds the runtime and, when
    # selected, merges pgcrypto/pgvector archives without network access.
    for required in libpglite.a.zst pglite-runtime.tar.zst \
      pglite-ext-pgcrypto-${target}.tar.gz pglite-ext-pgvector-${target}.tar.gz; do
      test -s "lib/$required"
    done
    mkdir -p "$out"
    cp -r lib "$out/lib"
    runHook postInstall
  '';

  meta = {
    description = "Cargo.lock-pinned PGlite native engine and runtime payload";
    license = lib.licenses.mit;
    platforms = builtins.attrNames targets;
  };
}
