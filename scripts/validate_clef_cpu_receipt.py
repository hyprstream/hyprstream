#!/usr/bin/env python3
"""Bind a fresh Clef development receipt to a captured build and source.

The native harness performs numerical and lifecycle assertions. This verifier
checks that its actual output came from the candidate being reviewed. It does
not replace independent review, task-accuracy evaluation or lab acceptance.
"""

import argparse
import json
from pathlib import Path
import subprocess

from clef_cpu_build import SOURCE, TEST, sha


def validate_runtime_libraries(build):
    """Re-resolve and hash the CPU libraries that the captured executable loads."""
    cpu_names = ("libtorch_cpu.so", "libc10.so")
    libtorch = build.get("environment", {}).get("LIBTORCH")
    if not libtorch:
        raise ValueError("captured CPU libtorch directory is missing")
    try:
        cpu_hashes = {name: sha(Path(libtorch).resolve() / "lib" / name)
                      for name in cpu_names}
    except OSError as error:
        raise ValueError("captured CPU runtime library is unavailable") from error
    if build.get("cpu_libraries_sha256") != cpu_hashes:
        raise ValueError("captured CPU runtime library changed")

    try:
        ldd = subprocess.run(["ldd", build["executable"]], capture_output=True,
                             text=True, check=True)
        if "not found" in ldd.stdout:
            raise ValueError("captured runtime library is missing")
        resolved = {}
        for line in ldd.stdout.splitlines():
            name, arrow, remainder = line.partition(" => ")
            if arrow and name.strip().startswith(("libtorch", "libc10", "libgomp")):
                path = Path(remainder.split()[0]).resolve()
                resolved[name.strip()] = {"path": str(path), "sha256": sha(path)}
    except (OSError, subprocess.CalledProcessError, IndexError) as error:
        raise ValueError("captured runtime libraries cannot be resolved") from error
    if not set(cpu_names).issubset(resolved):
        raise ValueError("captured executable no longer resolves CPU libtorch")
    if build.get("resolved_runtime_libraries") != resolved:
        raise ValueError("captured runtime library path or contents changed")


def validate_runtime_library_identity(identity, build):
    """Bind file-backed DSOs observed in the executing process to the build."""
    expected = build.get("resolved_runtime_libraries")
    runtime = identity.get("runtime_libraries")
    if not isinstance(expected, dict) or not isinstance(runtime, dict):
        raise ValueError("actual process runtime library identity is missing")
    expected_by_path = {}
    for name, entry in expected.items():
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path:
            raise ValueError("captured runtime library manifest is malformed")
        if (path in expected_by_path
                and expected_by_path[path].get("sha256") != entry.get("sha256")):
            raise ValueError("captured runtime library manifest has conflicting identities")
        expected_by_path[path] = entry
    if set(runtime) != set(expected_by_path):
        raise ValueError("actual process runtime library set differs from the captured build")
    required = {expected[name]["path"] for name in ("libtorch_cpu.so", "libc10.so")
                if name in expected}
    if len(required) != 2 or not required.issubset(runtime):
        raise ValueError("actual process did not attest required CPU libtorch libraries")
    for path, captured in expected_by_path.items():
        observed = runtime[path]
        if not isinstance(observed, dict):
            raise ValueError("actual process runtime library identity is malformed")
        if observed.get("sha256") != captured.get("sha256"):
            raise ValueError("actual process loaded runtime library differs from the captured build")
        device = observed.get("device")
        inode = observed.get("inode")
        if (not isinstance(device, str)
                or len(device.split(":")) != 2
                or any(not part or any(char not in "0123456789abcdefABCDEF" for char in part)
                       for part in device.split(":"))
                or not isinstance(inode, int) or isinstance(inode, bool) or inode <= 0):
            raise ValueError("actual process runtime library mapping identity is malformed")


def validate(receipt, build, source, candidate, run_result, receipt_sha256):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if git("status", "--porcelain"):
        raise ValueError("source checkout is dirty")
    if candidate != git("rev-parse", "HEAD") or candidate != build["source"]["commit"]:
        raise ValueError("candidate does not match build/source commit")
    if git("rev-parse", "HEAD^{tree}") != build["source"]["tree"]:
        raise ValueError("source tree differs from build")
    if build["schema"] != "clef-cpu-build/2" or build["source"] != build["source_after"]:
        raise ValueError("build did not capture stable source")
    if build["build_exit"] != 0 or "--release" not in build["command"] or TEST not in build["command"]:
        raise ValueError("missing successful Release acceptance build")
    profile = build["compiler_artifact"]["profile"]
    if str(profile["opt_level"]) != "3" or profile["debug_assertions"] or not profile["test"]:
        raise ValueError("compiler artifact is not a Release test")
    if build["environment"]["HYPRSTREAM_LIBTORCH_VARIANT"] != "cpu":
        raise ValueError("build was not CPU")
    if sha(build["executable"]) != build["executable_sha256"]:
        raise ValueError("captured executable changed")
    validate_runtime_libraries(build)
    identity = receipt.get("source_identity", {})
    digest = sha(source / SOURCE)
    if digest != build["source"]["acceptance_source_sha256"] or digest != identity.get("acceptance_source_sha256"):
        raise ValueError("runtime did not emit the compiled candidate source digest")
    revision = identity.get("git_sha", "")
    # Existing build_info.rs deliberately shortens HYPRSTREAM_SOURCE_COMMIT.
    if revision != candidate[:7] or identity.get("git_dirty") != "false":
        raise ValueError("runtime build identity does not match the clean candidate")
    if identity.get("executable_sha256") != build["executable_sha256"]:
        raise ValueError("runtime executable does not match the captured Release artifact")
    validate_runtime_library_identity(identity, build)
    if (run_result.get("candidate") != candidate
            or run_result.get("executable_sha256") != build["executable_sha256"]
            or run_result.get("service_receipt_sha256") != receipt_sha256):
        raise ValueError("actual run result does not bind this candidate, executable and receipt")
    if run_result.get("exit") != 0 or run_result.get("controller_exit") != 0:
        raise ValueError("actual native/controller execution did not exit successfully")
    if (run_result.get("native_samples", 0) < 1
            or run_result.get("peak_native_swap_kib") != 0
            or run_result.get("peak_group_swap_kib") != 0):
        raise ValueError("CPU execution lacks native/group observations without swapping")
    if receipt.get("result") != "AUTHENTICATED_CPU_SERVICE_CASES_PASS" or receipt.get("t6_complete") is not True:
        raise ValueError("native acceptance did not complete")
    if receipt.get("model_service_stop_complete") is not True:
        raise ValueError("receipt does not report completed model service stop")
    orders = receipt.get("orders", [])
    # Eleven questions are served by ten externally recorded work orders:
    # 5 oracle + 3 repeat/isolation + oversized + timeout recovery. Premint
    # timeout-zero rejection and internally consumed midforward orders add none.
    if len(orders) != 10 or any(order.get("verified") is not True or
                              order.get("coordinate") != "inference:Decide/decide" for order in orders):
        raise ValueError("expected ten verified Decide/decide work orders")
    if receipt.get("corpus_records") != 5 or receipt.get("corpus_questions") != 11:
        raise ValueError("unexpected development corpus")
    return {"candidate": candidate, "executable_sha256": build["executable_sha256"],
            "source_digest": digest, "records": 5, "questions": 11,
            "scope": "Native CPU development evidence for this source and completed run; lab validation remains separate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--run-result", type=Path, required=True)
    args = parser.parse_args()
    report = validate(json.loads(args.receipt.read_text()), json.loads(args.build.read_text()),
                      args.source.resolve(), args.candidate,
                      json.loads(args.run_result.read_text()), sha(args.receipt))
    report["service_receipt_sha256"] = sha(args.receipt)
    report["build_manifest_sha256"] = sha(args.build)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
