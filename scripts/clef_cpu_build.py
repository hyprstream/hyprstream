#!/usr/bin/env python3
"""Capture a real Release CPU test executable inside an existing BuildQ lease.

This builds and lists a test; it does not run inference or certify a review.
Run through hyprstream-build with HYPRSTREAM_LIBTORCH_VARIANT=cpu. The output
directory must be new, absolute and outside the source checkout.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


TEST = "services::model::tests::clef_cpu_acceptance::real_clef_cpu_authenticated_decide_acceptance"
SOURCE = "crates/hyprstream/src/services/model_clef_cpu_acceptance.rs"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def source_identity():
    if git("status", "--porcelain"):
        raise RuntimeError("acceptance build requires a clean source checkout")
    return {"commit": git("rev-parse", "HEAD"),
            "tree": git("rev-parse", "HEAD^{tree}"),
            "acceptance_source_sha256": sha(SOURCE)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("BUILDQ_LEASE_VERSION") != "1":
        parser.error("run this command through hyprstream-build/BuildQ")
    target = Path(os.environ["CARGO_TARGET_DIR"]).resolve()
    libtorch = Path(os.environ["LIBTORCH"]).resolve()
    if os.environ.get("HYPRSTREAM_LIBTORCH_VARIANT") != "cpu":
        parser.error("select HYPRSTREAM_LIBTORCH_VARIANT=cpu")
    root = Path(git("rev-parse", "--show-toplevel")).resolve()
    os.chdir(root)
    output = args.output
    if not output.is_absolute() or output.is_relative_to(root):
        parser.error("output must be absolute and outside the checkout")
    before = source_identity()
    # Existing build.rs input: derive it from the observed checkout, never a
    # coordinator-supplied candidate label. A new commit invalidates the build.
    os.environ["HYPRSTREAM_SOURCE_COMMIT"] = before["commit"]
    output.mkdir(mode=0o700)
    command = ["cargo", "test", "--release", "--locked", "-p", "hyprstream",
               "--no-default-features", "--features", "metrics,rocksdb,postgres",
               "--lib", TEST, "--no-run", "--message-format=json"]
    started = datetime.now(timezone.utc).isoformat()
    with (output / "build.jsonl").open("x") as stdout, (output / "build.stderr").open("x") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    if result.returncode:
        raise SystemExit(result.returncode)
    after = source_identity()
    if before != after:
        raise RuntimeError("source changed during compilation")
    artifacts = []
    for line in (output / "build.jsonl").read_text().splitlines():
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message["target"]["name"] == "hyprstream_core"
                and message["profile"]["test"] and message.get("executable")):
            artifacts.append(message)
    if len(artifacts) != 1:
        raise RuntimeError(f"expected one Cargo test executable, got {len(artifacts)}")
    artifact = artifacts[0]
    if Path(artifact["target"]["src_path"]).resolve() != root / "crates/hyprstream/src/lib.rs":
        raise RuntimeError("compiler artifact came from a different source checkout")
    profile = artifact["profile"]
    if str(profile["opt_level"]) != "3" or profile["debug_assertions"]:
        raise RuntimeError("Cargo did not produce the requested Release profile")
    executable = Path(artifact["executable"]).resolve()
    if not executable.is_relative_to(target / "release"):
        raise RuntimeError("compiler executable is outside the leased release target")
    saved = output / "hyprstream-clef-cpu-tests"
    shutil.copy2(executable, saved)
    digest = sha(saved)
    if digest != sha(executable):
        raise RuntimeError("executable changed during capture")
    listing = subprocess.run([str(saved), TEST, "--list", "--exact", "--format", "terse"],
                             capture_output=True, text=True, check=True)
    (output / "test-list.txt").write_text(listing.stdout)
    if listing.stdout.splitlines().count(f"{TEST}: test") != 1:
        raise RuntimeError("captured executable does not list the exact test")
    ldd = subprocess.run(["ldd", str(saved)], capture_output=True, text=True, check=True)
    (output / "ldd.txt").write_text(ldd.stdout)
    if "not found" in ldd.stdout:
        raise RuntimeError("captured executable has missing runtime dependencies")
    libraries = {}
    for line in ldd.stdout.splitlines():
        name, arrow, remainder = line.partition(" => ")
        if arrow and name.strip().startswith(("libtorch", "libc10", "libgomp")):
            path = Path(remainder.split()[0]).resolve()
            libraries[name.strip()] = {"path": str(path), "sha256": sha(path)}
    if not {"libtorch_cpu.so", "libc10.so"}.issubset(libraries):
        raise RuntimeError("captured executable does not resolve CPU libtorch")
    if any("cuda" in name or "hip" in name for name in libraries):
        raise RuntimeError("CPU executable unexpectedly links a GPU runtime")
    manifest = {"schema": "clef-cpu-build/2", "source": before,
                "source_after": after, "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "command": command, "build_exit": result.returncode,
                "compiler_artifact": artifact,
                "executable": str(saved), "executable_sha256": digest,
                "build_jsonl_sha256": sha(output / "build.jsonl"),
                "test_list_sha256": sha(output / "test-list.txt"),
                "environment": {key: os.environ.get(key) for key in (
                    "LIBTORCH", "HYPRSTREAM_LIBTORCH_VARIANT", "CARGO_TARGET_DIR",
                    "CARGO_BUILD_JOBS", "CARGO_INCREMENTAL", "RUSTFLAGS",
                    "HYPRSTREAM_SOURCE_COMMIT")},
                "cpu_libraries_sha256": {name: sha(libtorch / "lib" / name)
                    for name in ("libtorch_cpu.so", "libc10.so")},
                "resolved_runtime_libraries": libraries,
                "scope": "Build and test listing only; independent review and inference remain required"}
    (output / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "source": before, "executable_sha256": digest}))


if __name__ == "__main__":
    main()
