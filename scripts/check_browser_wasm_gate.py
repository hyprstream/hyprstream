#!/usr/bin/env python3
"""Fail-closed static regression for the browser-WASM CI coverage contract.

The browser gate used to check only hyprstream-rpc. www's real browser build
also compiles hyprstream-rpc-std and hyprstream-vfs, so VFS-only wasm32
regressions could pass both PR checks and the required merge-group job.

This zero-build check locks one shared script into both CI paths and proves that
the script retains the complete package set, target, and browser-only cfgs.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "rust.yml"
MERGE_DRIVER = ROOT / ".github" / "scripts" / "graviton-build-test.sh"
WASM_CHECK = ROOT / ".github" / "scripts" / "browser-wasm-check.sh"

EXPECTED_PACKAGES = {
    "hyprstream-rpc",
    "hyprstream-vfs",
    "hyprstream-rpc-std",
}
EXPECTED_CFGS = {
    "--cfg=web_sys_unstable_apis",
    '--cfg=getrandom_backend="wasm_js"',
}
WORKFLOW_INVOCATION = "bash /build/.github/scripts/browser-wasm-check.sh"
MERGE_DRIVER_WORKFLOW_INVOCATION = (
    "runuser -u ci -- bash -euo pipefail "
    "/build/.github/scripts/graviton-build-test.sh"
)
MERGE_INVOCATION = "bash .github/scripts/browser-wasm-check.sh"
EXPECTED_EXECUTION_TAIL = """\
append_rustflag '--cfg=web_sys_unstable_apis'
append_rustflag '--cfg=getrandom_backend="wasm_js"'
export RUSTFLAGS

readonly -a BROWSER_WASM_PACKAGES=(
  hyprstream-rpc
  hyprstream-vfs
  hyprstream-rpc-std
)

package_args=()
for package in "${BROWSER_WASM_PACKAGES[@]}"; do
  package_args+=( -p "$package" )
done

cargo check --locked \\
  --target wasm32-unknown-unknown \\
  "${package_args[@]}"
"""


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _package_set(script: str) -> set[str]:
    match = re.search(
        r"readonly -a BROWSER_WASM_PACKAGES=\(\n(?P<body>.*?)\n\)",
        script,
        flags=re.DOTALL,
    )
    _assert(match is not None, "browser check package array is missing or malformed")
    assert match is not None
    return {
        line.strip()
        for line in match.group("body").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _cfg_set(script: str) -> set[str]:
    return set(re.findall(r"^append_rustflag '([^']+)'$", script, flags=re.MULTILINE))


def check(workflow: str, merge_driver: str, wasm_check: str) -> None:
    _assert(
        workflow.count(WORKFLOW_INVOCATION) == 1,
        "rust.yml fast PR WASM job must invoke the shared browser check exactly once",
    )
    _assert(
        workflow.count(MERGE_DRIVER_WORKFLOW_INVOCATION) == 1,
        "rust.yml required build must invoke the merge driver exactly once",
    )
    _assert(
        merge_driver.count(MERGE_INVOCATION) == 1,
        "required merge driver must invoke the shared browser check exactly once",
    )
    packages = _package_set(wasm_check)
    _assert(
        packages == EXPECTED_PACKAGES,
        "browser check packages must be exactly "
        f"{sorted(EXPECTED_PACKAGES)}; got {sorted(packages)}",
    )
    cfgs = _cfg_set(wasm_check)
    _assert(
        cfgs == EXPECTED_CFGS,
        "browser check cfgs must be exactly "
        f"{sorted(EXPECTED_CFGS)}; got {sorted(cfgs)}",
    )
    _assert(
        wasm_check.count("--target wasm32-unknown-unknown") == 1,
        "browser check must target wasm32-unknown-unknown exactly once",
    )
    _assert(
        wasm_check.count("cargo check --locked") == 1,
        "browser check must use one locked cargo check",
    )
    _assert(
        wasm_check.count(EXPECTED_EXECUTION_TAIL) == 1,
        "browser cfgs, package array/loop, target, and Cargo arguments must remain "
        "one directly connected execution chain",
    )


def mutations(
    workflow: str, merge_driver: str, wasm_check: str
) -> list[tuple[str, str, str, str]]:
    cases = [
        (
            "remove PR invocation",
            workflow.replace(WORKFLOW_INVOCATION, "true", 1),
            merge_driver,
            wasm_check,
        ),
        (
            "remove merge invocation",
            workflow,
            merge_driver.replace(MERGE_INVOCATION, "true", 1),
            wasm_check,
        ),
        (
            "disconnect merge driver from required workflow",
            workflow.replace(MERGE_DRIVER_WORKFLOW_INVOCATION, "true", 1),
            merge_driver,
            wasm_check,
        ),
        (
            "change wasm target",
            workflow,
            merge_driver,
            wasm_check.replace(
                "--target wasm32-unknown-unknown",
                "--target wasm32-wasip1",
                1,
            ),
        ),
        (
            "drop web-sys cfg",
            workflow,
            merge_driver,
            wasm_check.replace(
                "append_rustflag '--cfg=web_sys_unstable_apis'\n",
                "",
                1,
            ),
        ),
        (
            "drop getrandom cfg",
            workflow,
            merge_driver,
            wasm_check.replace(
                "append_rustflag '--cfg=getrandom_backend=\"wasm_js\"'\n",
                "",
                1,
            ),
        ),
        (
            "disconnect package args from cargo",
            workflow,
            merge_driver,
            wasm_check.replace(
                '"${package_args[@]}"',
                "-p hyprstream-rpc",
                1,
            ),
        ),
        (
            "disconnect package array from args",
            workflow,
            merge_driver,
            wasm_check.replace(
                '  package_args+=( -p "$package" )',
                "  :",
                1,
            ),
        ),
        (
            "stop exporting browser cfgs",
            workflow,
            merge_driver,
            wasm_check.replace("export RUSTFLAGS", ": # RUSTFLAGS not exported", 1),
        ),
    ]
    for package in sorted(EXPECTED_PACKAGES):
        cases.append(
            (
                f"drop package {package}",
                workflow,
                merge_driver,
                wasm_check.replace(f"  {package}\n", "", 1),
            )
        )
    return cases


def main() -> int:
    workflow = WORKFLOW.read_text()
    merge_driver = MERGE_DRIVER.read_text()
    wasm_check = WASM_CHECK.read_text()

    try:
        check(workflow, merge_driver, wasm_check)
    except AssertionError as error:
        print(f"browser-WASM gate regression FAILED: {error}", file=sys.stderr)
        return 1

    cases = mutations(workflow, merge_driver, wasm_check)
    escaped: list[str] = []
    for label, mutated_workflow, mutated_driver, mutated_check in cases:
        try:
            check(mutated_workflow, mutated_driver, mutated_check)
        except AssertionError:
            continue
        escaped.append(label)

    if escaped:
        print(
            "browser-WASM gate self-test FAILED; mutations escaped: "
            + ", ".join(escaped),
            file=sys.stderr,
        )
        return 1

    print(f"browser-WASM gate regression: OK ({len(cases)} negative mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
