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


def validate(receipt, build, source, candidate):
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
    identity = receipt.get("source_identity", {})
    digest = sha(source / SOURCE)
    if digest != build["source"]["acceptance_source_sha256"] or digest != identity.get("acceptance_source_sha256"):
        raise ValueError("runtime did not emit the compiled candidate source digest")
    revision = identity.get("git_sha", "")
    # Existing build_info.rs deliberately shortens HYPRSTREAM_SOURCE_COMMIT.
    if revision != candidate[:7] or identity.get("git_dirty") != "false":
        raise ValueError("runtime build identity does not match the clean candidate")
    if receipt.get("result") != "AUTHENTICATED_CPU_SERVICE_CASES_PASS" or receipt.get("t6_complete") is not True:
        raise ValueError("native acceptance did not complete")
    orders = receipt.get("orders", [])
    if len(orders) != 11 or any(order.get("verified") is not True or
                              order.get("coordinate") != "inference:Decide/decide" for order in orders):
        raise ValueError("expected eleven verified Decide/decide work orders")
    if receipt.get("corpus_records") != 5 or receipt.get("corpus_questions") != 11:
        raise ValueError("unexpected development corpus")
    return {"candidate": candidate, "executable_sha256": build["executable_sha256"],
            "source_digest": digest, "records": 5, "questions": 11,
            "scope": "Source-bound native CPU development gate; lab validation remains separate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    report = validate(json.loads(args.receipt.read_text()), json.loads(args.build.read_text()),
                      args.source.resolve(), args.candidate)
    report["service_receipt_sha256"] = sha(args.receipt)
    report["build_manifest_sha256"] = sha(args.build)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
