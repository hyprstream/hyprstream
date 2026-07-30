#!/usr/bin/env python3
"""Fail-closed static regression for the browser-WASM CI coverage contract.

The browser gate used to check only hyprstream-rpc. www's real browser build
also compiles hyprstream-rpc-std and hyprstream-vfs, so VFS-only wasm32
regressions could pass both PR checks and the required merge-group job.

This zero-build check locks one shared script into both CI paths and proves that
the script retains the complete package set, target, and browser-only cfgs.

#1425 r4: a `cargo check` compile pass cannot catch a regression in the actual
browser-fetch runtime behavior (JS callback, Request/Response, nonce retry,
response rejection) — only real execution can. The fast PR `WASM (browser
client)` job runs that real execution, but it is explicitly skipped on
`merge_group` (rust.yml `wasm` job `if:`), so a required merge-group `build`
could go green without ever launching a browser. This gate additionally locks
the required merge driver (graviton-build-test.sh) to invoking the real
browser-execution script (browser-wasm-test-ci.sh) — not just the compile-only
check — and locks the workflow to installing the browser dependency as root
*before* it drops to the non-root `ci` user that script runs as.
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
# #1425 r4: the required merge-group build path must invoke the REAL browser
# runner (not just the compile-only check above), and must install the
# browser as root before it creates/switches to the non-root `ci` user that
# runs graviton-build-test.sh (that user cannot apt-get/dnf install).
WORKFLOW_CHROMIUM_INSTALL_INVOCATION = (
    "bash /build/.github/scripts/install-chromium.sh"
)
WORKFLOW_USERADD = "useradd -m ci"
MERGE_REAL_EXECUTION_INVOCATION = "bash .github/scripts/browser-wasm-test-ci.sh"
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
    # #1425 r4: the required merge-group build must actually launch a browser,
    # not just compile-check — and the browser dependency must be installed
    # as root before the workflow drops to the non-root `ci` user that runs
    # graviton-build-test.sh, or the real-execution invocation below fails
    # closed at runtime with no browser available.
    _assert(
        merge_driver.count(MERGE_REAL_EXECUTION_INVOCATION) == 1,
        "required merge driver must invoke the real browser-execution script "
        "(browser-wasm-test-ci.sh) exactly once, not only the compile-only check",
    )
    _assert(
        workflow.count(WORKFLOW_CHROMIUM_INSTALL_INVOCATION) == 1,
        "rust.yml required build must install the browser (install-chromium.sh) "
        "exactly once, as root",
    )
    _assert(
        WORKFLOW_CHROMIUM_INSTALL_INVOCATION in workflow
        and WORKFLOW_USERADD in workflow
        and workflow.index(WORKFLOW_CHROMIUM_INSTALL_INVOCATION)
        < workflow.index(WORKFLOW_USERADD),
        "rust.yml required build must install the browser BEFORE switching to "
        "the non-root ci user (that user cannot apt-get/dnf install)",
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
            "remove real browser execution from merge driver",
            workflow,
            merge_driver.replace(MERGE_REAL_EXECUTION_INVOCATION, "true", 1),
            wasm_check,
        ),
        (
            "duplicate real browser execution invocation",
            workflow,
            merge_driver.replace(
                MERGE_REAL_EXECUTION_INVOCATION,
                MERGE_REAL_EXECUTION_INVOCATION + "\n" + MERGE_REAL_EXECUTION_INVOCATION,
                1,
            ),
            wasm_check,
        ),
        (
            "remove chromium install from required build job",
            workflow.replace(WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "true", 1),
            merge_driver,
            wasm_check,
        ),
        (
            "install chromium after switching to non-root ci user",
            (
                workflow.replace(WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "true", 1)
                .replace(
                    WORKFLOW_USERADD,
                    WORKFLOW_USERADD + "\n            " + WORKFLOW_CHROMIUM_INSTALL_INVOCATION,
                    1,
                )
            ),
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
