#!/usr/bin/env python3
"""Fail-closed static regression for the browser-WASM CI coverage contract.

The browser gate used to check only hyprstream-rpc. www's real browser build
also compiles hyprstream-rpc-std and hyprstream-vfs, so VFS-only wasm32
regressions could pass both PR checks and the required merge-group job.

This zero-build check locks one shared script into both CI paths and proves that
the script retains the complete package set, target, and browser-only cfgs.

#1425 r4/r5: a `cargo check` compile pass cannot catch a regression in the
actual browser-fetch runtime behavior (JS callback, Request/Response, nonce
retry, response rejection) — only real execution can. The fast PR `WASM
(browser client)` job runs that real execution, but it is explicitly skipped
on `merge_group` (rust.yml `wasm` job `if:`), so a required merge-group
`build` could go green without ever launching a browser.

r4 added raw-substring assertions locking the required merge driver
(graviton-build-test.sh) to invoking the real browser-execution script, and
the workflow to installing the browser as root before dropping to the
non-root `ci` user. An independent r5 review demonstrated seven adversarial
mutations that satisfied every r4 assertion while removing the actual
security-critical execution:

  1. commenting out the root install invocation in rust.yml
  2. commenting out the non-root `runuser` invocation in rust.yml
  3. commenting out the real-execution phase in graviton-build-test.sh
  4. hiding the root install behind an `if false; then ... fi` branch
  5. replacing install-chromium.sh's body with a bare `exit 0`
  6. replacing browser-wasm-test-ci.sh's body with a bare `exit 0`
  7. replacing browser-wasm-test.sh's real `cargo test` runner with a
     compile-only `cargo check`

Root causes: (a) raw substring counting cannot distinguish an executable
command from a comment or from code inside an always-false branch, and
(b) the gate never read install-chromium.sh, browser-wasm-test-ci.sh, or
browser-wasm-test.sh at all, so replacing any of their bodies with a no-op
was invisible to it.

r5 fixes both: every critical invocation is now checked for (i) appearing as
a literal, non-comment line and (ii) being *reachable* under a narrow
if/then/elif/else/fi model that recognizes a closed set of always-false
shell conditions (`false`, `[ 0 = 1 ]`, etc.) — deliberately not a general
shell interpreter, just enough to defeat the exact hiding pattern demonstrated
above. All three previously-uninspected scripts are now read and asserted to
contain their real command bodies, and the full six-hop chain is checked file
by file: rust.yml -> install-chromium.sh (root) -> `runuser -u ci` ->
graviton-build-test.sh -> browser-wasm-test-ci.sh -> browser-wasm-test.sh ->
named `wasm_browser_fetch` executed via `cargo test` (never `cargo check`).
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "rust.yml"
MERGE_DRIVER = ROOT / ".github" / "scripts" / "graviton-build-test.sh"
WASM_CHECK = ROOT / ".github" / "scripts" / "browser-wasm-check.sh"
INSTALL_CHROMIUM = ROOT / ".github" / "scripts" / "install-chromium.sh"
BROWSER_TEST_CI = ROOT / ".github" / "scripts" / "browser-wasm-test-ci.sh"
BROWSER_TEST = ROOT / ".github" / "scripts" / "browser-wasm-test.sh"

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
# The required merge-group build path must invoke the REAL browser runner
# (not just the compile-only check above), and must install the browser as
# root before it creates/switches to the non-root `ci` user that runs
# graviton-build-test.sh (that user cannot apt-get/dnf install).
WORKFLOW_CHROMIUM_INSTALL_INVOCATION = "bash /build/.github/scripts/install-chromium.sh"
WORKFLOW_USERADD = "useradd -m ci"
MERGE_REAL_EXECUTION_INVOCATION = "bash .github/scripts/browser-wasm-test-ci.sh"
# The CI wrapper must itself invoke the installer (defense in depth for the
# PR-context `wasm` job, which runs the wrapper as root with nothing
# pre-installed) and must delegate to the real named-artifact runner.
CI_WRAPPER_INSTALL_INVOCATION = 'bash "$SCRIPT_DIR/install-chromium.sh"'
CI_WRAPPER_DELEGATE_INVOCATION = 'exec bash "$SCRIPT_DIR/browser-wasm-test.sh"'
# install-chromium.sh must retain both real package-manager branches.
INSTALL_APT_INVOCATION = "apt-get install -y -qq chromium chromium-driver"
INSTALL_DNF_INVOCATION = "dnf install -y -q chromium chromedriver"
# browser-wasm-test.sh must run the REAL, named artifact through `cargo
# test` — never a compile-only `cargo check` substitute.
TEST_NAME_ASSIGNMENT = 'TEST_NAME="wasm_browser_fetch"'
CARGO_TEST_INVOCATION = "cargo test --locked"
CARGO_TEST_NAMED_ARG = '--test "$TEST_NAME"'
FORBIDDEN_CARGO_CHECK = "cargo check"

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


# ---------------------------------------------------------------------------
# Executable-command reachability: a narrow, auditable model that answers
# "is this line a real, non-comment command that is not sealed inside an
# always-false branch?" It is NOT a general shell interpreter — it recognizes
# exactly the always-false condition spellings a mutation would plausibly use
# to hide a command while keeping the surrounding script syntactically valid,
# and treats every other condition (including all real runtime checks this
# repository's scripts actually use, e.g. `command -v apt-get`, `[[ -z "$X" ]]`)
# as reachable. That asymmetry is deliberate: the goal is to reject the
# specific `if false; then ...; fi` hiding pattern demonstrated by the r5
# review, not to prove general shell reachability.
# ---------------------------------------------------------------------------

_ALWAYS_FALSE_CONDITIONS = {
    "false",
    "[ 0 = 1 ]",
    "[ 1 = 0 ]",
    "[ 0 -eq 1 ]",
    "[ 1 -eq 0 ]",
    "[[ 0 = 1 ]]",
    "[[ 1 = 0 ]]",
    "[[ 0 == 1 ]]",
    "[[ 1 == 0 ]]",
    "test 0 = 1",
    "test 1 = 0",
}

_IF_THEN_RE = re.compile(r"^if\s+(?P<cond>.+?)\s*;\s*then\s*$")
_IF_ONLY_RE = re.compile(r"^if\s+(?P<cond>.+)$")
_THEN_ONLY_RE = re.compile(r"^then\s*$")
_ELIF_THEN_RE = re.compile(r"^elif\s+(?P<cond>.+?)\s*;\s*then\s*$")
_ELSE_RE = re.compile(r"^else\s*$")
_FI_RE = re.compile(r"^fi\s*$")


def _norm_cond(cond: str) -> str:
    return " ".join(cond.split())


def _is_always_false(cond: str) -> bool:
    return _norm_cond(cond) in _ALWAYS_FALSE_CONDITIONS


def _is_comment(line: str) -> bool:
    return line.strip().startswith("#")


def _reachability(lines: list[str]) -> list[bool]:
    """Return, per line index, whether that line is reachable.

    `stack[i]` is True while inside a branch known to be unreachable (an
    always-false `if`/`elif` arm that has not yet hit its `else`).
    """
    stack: list[bool] = []
    pending_cond: str | None = None
    reachable: list[bool] = []

    for raw in lines:
        stripped = raw.strip()

        if pending_cond is not None:
            if _THEN_ONLY_RE.match(stripped):
                stack.append(_is_always_false(pending_cond))
                pending_cond = None
                reachable.append(not any(stack))
                continue
            # A bare `if <cond>` not immediately followed by `then` on its own
            # line: drop the pending condition (conservatively reachable) and
            # fall through to process this line normally below.
            pending_cond = None

        match = _IF_THEN_RE.match(stripped)
        if match:
            stack.append(_is_always_false(match.group("cond")))
            reachable.append(not any(stack))
            continue

        match = _ELIF_THEN_RE.match(stripped)
        if match and stack:
            stack[-1] = _is_always_false(match.group("cond"))
            reachable.append(not any(stack))
            continue

        if _ELSE_RE.match(stripped) and stack:
            stack[-1] = not stack[-1]
            reachable.append(not any(stack))
            continue

        if _FI_RE.match(stripped) and stack:
            stack.pop()
            reachable.append(not any(stack))
            continue

        match = _IF_ONLY_RE.match(stripped)
        if match and stripped.startswith("if "):
            pending_cond = match.group("cond")
            reachable.append(not any(stack))
            continue

        reachable.append(not any(stack))

    return reachable


def _assert_executable_once(
    label: str, script_name: str, script: str, needle: str
) -> None:
    """Assert `needle` appears exactly once in `script`, as a real
    (non-comment), reachable command line — not merely present as text
    anywhere (a comment, or inside an always-false branch, both count as
    absent for this purpose)."""
    lines = script.splitlines()
    matches = [i for i, line in enumerate(lines) if needle in line]
    _assert(
        len(matches) == 1,
        f"{script_name} must contain {label} exactly once (as literal text); found {len(matches)}",
    )
    idx = matches[0]
    _assert(
        not _is_comment(lines[idx]),
        f"{script_name}'s {label} line must be an executable command, not a comment",
    )
    reach = _reachability(lines)
    _assert(
        reach[idx],
        f"{script_name}'s {label} line must be reachable, not sealed inside an "
        "always-false branch",
    )


def _assert_absent(label: str, script_name: str, script: str, needle: str) -> None:
    _assert(
        needle not in script,
        f"{script_name} must not contain {label} — the real runner must never be "
        "replaced by this compile-only/no-op substitute",
    )


def check(
    workflow: str,
    merge_driver: str,
    wasm_check: str,
    install_chromium: str,
    browser_test_ci: str,
    browser_test: str,
) -> None:
    # ---- rust.yml: PR job still runs the compile-only check exactly once ----
    _assert_executable_once("the PR WASM compile-only check", "rust.yml", workflow, WORKFLOW_INVOCATION)

    # ---- rust.yml: required build job chain (hops 1-3) ----
    _assert_executable_once(
        "the required-build browser install", "rust.yml", workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION
    )
    _assert_executable_once(
        "the required-build non-root handoff", "rust.yml", workflow, MERGE_DRIVER_WORKFLOW_INVOCATION
    )
    _assert(
        workflow.index(WORKFLOW_CHROMIUM_INSTALL_INVOCATION) < workflow.index(WORKFLOW_USERADD),
        "rust.yml required build must install the browser BEFORE switching to "
        "the non-root ci user (that user cannot apt-get/dnf install)",
    )

    # ---- graviton-build-test.sh (hop 4): both phases, reachable + real ----
    _assert_executable_once(
        "the compile-only browser check", "graviton-build-test.sh", merge_driver, MERGE_INVOCATION
    )
    _assert_executable_once(
        "the real browser-execution phase",
        "graviton-build-test.sh",
        merge_driver,
        MERGE_REAL_EXECUTION_INVOCATION,
    )

    # ---- browser-wasm-test-ci.sh (hop 5): must couple to BOTH neighbors ----
    _assert_executable_once(
        "the install-chromium.sh call", "browser-wasm-test-ci.sh", browser_test_ci, CI_WRAPPER_INSTALL_INVOCATION
    )
    _assert_executable_once(
        "the delegation to browser-wasm-test.sh",
        "browser-wasm-test-ci.sh",
        browser_test_ci,
        CI_WRAPPER_DELEGATE_INVOCATION,
    )

    # ---- install-chromium.sh: both real package-manager branches present ----
    _assert_executable_once(
        "the apt-get chromium install", "install-chromium.sh", install_chromium, INSTALL_APT_INVOCATION
    )
    _assert_executable_once(
        "the dnf chromium install", "install-chromium.sh", install_chromium, INSTALL_DNF_INVOCATION
    )

    # ---- browser-wasm-test.sh (hop 6): named artifact, real cargo test ----
    _assert_executable_once(
        "the wasm_browser_fetch test-name assignment", "browser-wasm-test.sh", browser_test, TEST_NAME_ASSIGNMENT
    )
    _assert_executable_once(
        "the real cargo test invocation", "browser-wasm-test.sh", browser_test, CARGO_TEST_INVOCATION
    )
    _assert_executable_once(
        "the named --test argument", "browser-wasm-test.sh", browser_test, CARGO_TEST_NAMED_ARG
    )
    _assert_absent(
        "a compile-only cargo check substitute", "browser-wasm-test.sh", browser_test, FORBIDDEN_CARGO_CHECK
    )

    # ---- browser-wasm-check.sh: unchanged compile-only contract ----
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
    workflow: str,
    merge_driver: str,
    wasm_check: str,
    install_chromium: str,
    browser_test_ci: str,
    browser_test: str,
) -> list[tuple[str, str, str, str, str, str, str]]:
    def _comment_out(script: str, needle: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                lines[i] = f"{indent}# {line.lstrip()}"
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to comment out")

    def _wrap_if_false(script: str, needle: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\n" if line.endswith("\n") else ""
                wrapped = (
                    f"{indent}if false; then\n"
                    f"{line.rstrip(chr(10))}\n"
                    f"{indent}fi{newline}"
                )
                lines[i] = wrapped
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to wrap")

    cases: list[tuple[str, str, str, str, str, str, str]] = [
        (
            "remove PR invocation",
            workflow.replace(WORKFLOW_INVOCATION, "true", 1),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "remove merge invocation",
            workflow,
            merge_driver.replace(MERGE_INVOCATION, "true", 1),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "disconnect merge driver from required workflow",
            workflow.replace(MERGE_DRIVER_WORKFLOW_INVOCATION, "true", 1),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
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
            install_chromium,
            browser_test_ci,
            browser_test,
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
            install_chromium,
            browser_test_ci,
            browser_test,
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
            install_chromium,
            browser_test_ci,
            browser_test,
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
            install_chromium,
            browser_test_ci,
            browser_test,
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
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "stop exporting browser cfgs",
            workflow,
            merge_driver,
            wasm_check.replace("export RUSTFLAGS", ": # RUSTFLAGS not exported", 1),
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        # ---- the seven r5-demonstrated escapes, reproduced exactly ----
        (
            "r5#1: root install command commented out",
            _comment_out(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "r5#2: non-root runuser command commented out",
            _comment_out(workflow, MERGE_DRIVER_WORKFLOW_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "r5#3: real browser phase commented out",
            workflow,
            _comment_out(merge_driver, MERGE_REAL_EXECUTION_INVOCATION),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "r5#4: root install hidden behind false branch",
            _wrap_if_false(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "r5#5: install-chromium.sh replaced by exit 0",
            workflow,
            merge_driver,
            wasm_check,
            "#!/usr/bin/env bash\nexit 0\n",
            browser_test_ci,
            browser_test,
        ),
        (
            "r5#6: browser-wasm-test-ci.sh replaced by exit 0",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            "#!/usr/bin/env bash\nexit 0\n",
            browser_test,
        ),
        (
            "r5#7: browser-wasm-test.sh replaced by compile-only cargo check",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test.replace(
                'cargo test --locked \\\n  --manifest-path "$CRATE_MANIFEST" \\\n'
                '  --target wasm32-unknown-unknown \\\n  --test "$TEST_NAME"',
                'cargo check --locked --manifest-path "$CRATE_MANIFEST" '
                '--target wasm32-unknown-unknown',
                1,
            ),
        ),
        # ---- additional coupling cases the r5 finding requires ----
        (
            "runuser hidden behind false branch",
            _wrap_if_false(workflow, MERGE_DRIVER_WORKFLOW_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase hidden behind false branch",
            workflow,
            _wrap_if_false(merge_driver, MERGE_REAL_EXECUTION_INVOCATION),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "browser-wasm-test-ci.sh stops calling install-chromium.sh",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            _comment_out(browser_test_ci, CI_WRAPPER_INSTALL_INVOCATION),
            browser_test,
        ),
        (
            "browser-wasm-test-ci.sh stops delegating to browser-wasm-test.sh",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            _comment_out(browser_test_ci, CI_WRAPPER_DELEGATE_INVOCATION),
            browser_test,
        ),
        (
            "install-chromium.sh apt-get branch commented out",
            workflow,
            merge_driver,
            wasm_check,
            _comment_out(install_chromium, INSTALL_APT_INVOCATION),
            browser_test_ci,
            browser_test,
        ),
        (
            "install-chromium.sh dnf branch commented out",
            workflow,
            merge_driver,
            wasm_check,
            _comment_out(install_chromium, INSTALL_DNF_INVOCATION),
            browser_test_ci,
            browser_test,
        ),
        (
            "install-chromium.sh apt-get branch hidden behind false branch",
            workflow,
            merge_driver,
            wasm_check,
            _wrap_if_false(install_chromium, INSTALL_APT_INVOCATION),
            browser_test_ci,
            browser_test,
        ),
        (
            "browser-wasm-test.sh named --test arg commented out",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            _comment_out(browser_test, CARGO_TEST_NAMED_ARG),
        ),
    ]
    for package in sorted(EXPECTED_PACKAGES):
        cases.append(
            (
                f"drop package {package}",
                workflow,
                merge_driver,
                wasm_check.replace(f"  {package}\n", "", 1),
                install_chromium,
                browser_test_ci,
                browser_test,
            )
        )
    return cases


def main() -> int:
    workflow = WORKFLOW.read_text()
    merge_driver = MERGE_DRIVER.read_text()
    wasm_check = WASM_CHECK.read_text()
    install_chromium = INSTALL_CHROMIUM.read_text()
    browser_test_ci = BROWSER_TEST_CI.read_text()
    browser_test = BROWSER_TEST.read_text()

    try:
        check(workflow, merge_driver, wasm_check, install_chromium, browser_test_ci, browser_test)
    except AssertionError as error:
        print(f"browser-WASM gate regression FAILED: {error}", file=sys.stderr)
        return 1

    cases = mutations(
        workflow, merge_driver, wasm_check, install_chromium, browser_test_ci, browser_test
    )
    escaped: list[str] = []
    for label, m_workflow, m_driver, m_check, m_install, m_test_ci, m_test in cases:
        try:
            check(m_workflow, m_driver, m_check, m_install, m_test_ci, m_test)
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
