#!/usr/bin/env python3
"""Fail-closed static regression for the browser-WASM CI coverage contract.

The browser gate used to check only hyprstream-rpc. www's real browser build
also compiles hyprstream-rpc-std and hyprstream-vfs, so VFS-only wasm32
regressions could pass both PR checks and the required merge-group job.

This zero-build check locks one shared script into both CI paths and proves that
the script retains the complete package set, target, and browser-only cfgs.

A `cargo check` compile pass cannot catch a regression in the actual
browser-fetch runtime behavior (JS callback, Request/Response, nonce retry,
response rejection) — only real execution can. The fast PR `WASM (browser
client)` job runs that real execution, but it is explicitly skipped on
`merge_group` (rust.yml `wasm` job `if:`), so a required merge-group `build`
could go green without ever launching a browser.

An earlier revision of this gate locked the required merge driver
(graviton-build-test.sh) to invoking the real browser-execution script, and
the workflow to installing the browser as root before dropping to the
non-root `ci` user, using raw-substring assertions plus a *reachable-until-
proven-dead* control-flow model: a line counted unless the model could prove
the enclosing branch always-false. Independent review broke that model the
same way every round — anything the model could not parse (an unrecognized
construct, a non-literal condition it gave up on) defaulted to *reachable*,
so an escape only had to reach outside the model's small recognized
vocabulary, not defeat it.

This revision inverts the default: a critical invocation now counts as
executed only when every enclosing construct is one this model recognizes
AND proves reachable. Anything unrecognized — an unmodeled construct (`case`,
a `&&`/`||` guard on the invocation line itself), a condition shape the
evaluator cannot decide from a bare command name, a preceding unconditional
`exit` in the same linear scope, or a shell function shadowing the invoked
command — makes the gate fail closed. The model does not attempt to be a
general shell interpreter; it only has to be complete over the constructs the
six real hop files actually use for their own conditional logic
(`command -v foo`-style existence checks, `[[ -z "$VAR" ]]`-style dynamic
string tests), which continue to evaluate as legitimately reachable.

Every critical invocation is checked for (i) appearing as a literal,
non-comment line, (ii) being reachable under the fail-closed model above,
(iii) not sharing its line with an unevaluated `&&`/`||` guard, and (iv) not
naming a command that is shadowed by an earlier function definition in the
same script. The full six-hop chain is checked file by file: rust.yml ->
install-chromium.sh (root) -> `runuser -u ci` -> graviton-build-test.sh ->
browser-wasm-test-ci.sh -> browser-wasm-test.sh -> named `wasm_browser_fetch`
executed via `cargo test` (never `cargo check`).
"""

from __future__ import annotations

import pathlib
import re
import shlex
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
# Executable-command reachability: a conservative, fail-closed model that
# answers "is this line proven, by a construct this model recognizes, to be a
# real command that runs?" It is NOT a general shell interpreter. It is
# complete over exactly the constructs the six hop files legitimately use
# (if/elif/else/fi and while/until/do/done with either a literal or a
# dynamic-but-auditable condition, e.g. `command -v apt-get` or
# `[[ -z "$VAR" ]]`) and treats everything else — `case`, a `&&`/`||` guard
# on the invocation line, a bare variable used as the condition's command, an
# unparsable condition, a preceding unconditional `exit` in the same linear
# scope, or a shell function shadowing the invoked command — as NOT proven
# reachable. Default-reachable was the bug; default-unreachable is the fix.
# ---------------------------------------------------------------------------

_IF_THEN_RE = re.compile(r"^if\s+(?P<cond>.+?)\s*;\s*then\s*$")
_IF_ONLY_RE = re.compile(r"^if\s+(?P<cond>.+)$")
_THEN_ONLY_RE = re.compile(r"^then\s*$")
_ELIF_THEN_RE = re.compile(r"^elif\s+(?P<cond>.+?)\s*;\s*then\s*$")
_ELSE_RE = re.compile(r"^else\s*$")
_FI_RE = re.compile(r"^fi\s*$")
_WHILE_DO_RE = re.compile(r"^while\s+(?P<cond>.+?)\s*;\s*do\s*$")
_UNTIL_DO_RE = re.compile(r"^until\s+(?P<cond>.+?)\s*;\s*do\s*$")
_DONE_RE = re.compile(r"^done\s*$")
# `for`/`select` are loop headers this model does not evaluate at all (unlike
# `while`/`until`, which have a literal-decidable condition slot) — treated
# the same way as `case`: always unreachable, since none of the six hop
# files legitimately wrap a critical invocation in either.
_FOR_DO_RE = re.compile(r"^for\s+.+?\s*;\s*do\s*$")
_SELECT_DO_RE = re.compile(r"^select\s+.+?\s*;\s*do\s*$")
_CASE_IN_RE = re.compile(r"^case\s+.+\s+in\s*$")
_ESAC_RE = re.compile(r"^esac\s*$")
_EXIT_RE = re.compile(r"^exit(\s+[0-9]+)?\s*$")
_FUNC_DEF_RE = re.compile(
    r"^(?:function\s+)?(?P<name1>[A-Za-z_][A-Za-z0-9_-]*)\s*\(\)\s*\{?\s*$"
    r"|^function\s+(?P<name2>[A-Za-z_][A-Za-z0-9_-]*)\s*\{?\s*$"
)


def _norm_cond(cond: str) -> str:
    return " ".join(cond.split())


def _evaluate_test_body(tokens: list[str]) -> bool | None:
    """Literally evaluate a bracket/`test` body's token list, or ``None`` if
    this model cannot decide it (a dynamic operand, or an operator/arity this
    narrow evaluator does not cover) — ``None`` here still means "recognized
    bracket-test shape, undecidable value", never "unrecognized construct"."""
    if len(tokens) == 2 and tokens[0] in {"-n", "-z"} and "$" not in tokens[1]:
        truthy = bool(tokens[1])
        return truthy if tokens[0] == "-n" else not truthy
    if (
        len(tokens) == 3
        and tokens[0] == "!"
        and tokens[1] in {"-n", "-z"}
        and "$" not in tokens[2]
    ):
        truthy = bool(tokens[2])
        inner = truthy if tokens[1] == "-n" else not truthy
        return not inner
    if len(tokens) != 3 or any("$" in token for token in tokens):
        return None
    left, operator, right = tokens
    if operator in {"=", "==", "!="}:
        equal = left == right
        return equal if operator != "!=" else not equal
    if operator in {"-eq", "-ne", "-gt", "-lt", "-ge", "-le"}:
        try:
            left_i, right_i = int(left, 10), int(right, 10)
        except ValueError:
            return None
        if operator == "-eq":
            return left_i == right_i
        if operator == "-ne":
            return left_i != right_i
        if operator == "-gt":
            return left_i > right_i
        if operator == "-lt":
            return left_i < right_i
        if operator == "-ge":
            return left_i >= right_i
        return left_i <= right_i
    return None


def _condition_shape(cond: str) -> str:
    """Classify an `if`/`elif`/`while`/`until` condition string as one of:

    - ``"true"`` / ``"false"``: literally decidable from the text alone.
    - ``"dynamic"``: a recognized construct (bracket/`test` form, or a bare
      command invocation) whose truth this model cannot decide — a real
      runtime check like `command -v apt-get` or `[[ -z "$VAR" ]]`. Treated
      as reachable: these are exactly the legitimate conditionals the hop
      scripts use, and rejecting them would break the real, correct gate.
    - ``"unrecognized"``: a shape this model does not audit at all — a bare
      variable used as the condition's command (`if $F; then`), or text that
      does not even shell-tokenize. Treated as NOT reachable (fail-closed):
      the whole point of inverting this model is that an unauditable
      condition must not silently pass as proof of reachability.
    """
    normalized = _norm_cond(cond)
    if normalized.startswith("! "):
        inner = _condition_shape(normalized[2:])
        return {"true": "false", "false": "true"}.get(inner, inner)
    if normalized in {"true", "/bin/true", ":"}:
        return "true"
    if normalized in {"false", "/bin/false"}:
        return "false"

    try:
        tokens = shlex.split(normalized)
    except ValueError:
        return "unrecognized"
    if not tokens:
        return "unrecognized"

    if tokens[:1] == ["test"]:
        body = tokens[1:]
    elif tokens[0] in {"[", "[["}:
        closing = "]" if tokens[0] == "[" else "]]"
        if tokens[-1:] != [closing]:
            return "unrecognized"
        body = tokens[1:-1]
    else:
        # A bare-command condition: a legitimate, auditable runtime check
        # (`command -v apt-get`) UNLESS the command itself is indirected
        # through a variable, which this model cannot audit at all.
        if tokens[0].startswith("$"):
            return "unrecognized"
        return "dynamic"

    literal = _evaluate_test_body(body)
    if literal is True:
        return "true"
    if literal is False:
        return "false"
    return "dynamic"


def _is_comment(line: str) -> bool:
    return line.strip().startswith("#")


def _shadowed_commands(script: str) -> set[str]:
    """Names of shell functions defined anywhere in `script` — a function
    redefining one of the commands a critical invocation names (`bash`,
    `cargo`, `apt-get`, `dnf`, ...) would shadow the real command."""
    names: set[str] = set()
    for raw in script.splitlines():
        stripped = raw.strip()
        if _is_comment(stripped):
            continue
        match = _FUNC_DEF_RE.match(stripped)
        if match:
            name = match.group("name1") or match.group("name2")
            if name:
                names.add(name)
    return names


def _needle_tokens(needle: str) -> list[str]:
    try:
        return shlex.split(needle)
    except ValueError:
        return needle.split()


def _reachability(lines: list[str]) -> list[bool]:
    """Return, per line index, whether that line is proven reachable.

    A line is reachable only when (a) no enclosing `if`/`elif`/`while`/
    `until` frame is proven dead or unrecognized, (b) it is not inside a
    `case`/`esac` block (not modeled — always treated as unreachable), and
    (c) no unconditional `exit` earlier in the same linear scope precedes it.
    Each nested scope tracks its own linear-exit state independently: an
    `exit` inside an `if` body does not kill lines after that `if` closes.
    """
    stack: list[tuple[str, bool]] = []
    exit_dead_stack: list[bool] = []
    top_level_exit_dead = False
    pending_cond: str | None = None
    reachable: list[bool] = []

    # Prefix paren depth per line, so a bare `exit` embedded in a nested
    # command substitution or subshell (e.g. an awk script's own `exit`
    # statement inside `$(...)`) is never mistaken for a bash-level
    # unconditional exit in the enclosing script's linear flow. Net-counting
    # literal `(`/`)` characters is an approximation, not a shell parser —
    # sufficient here because the hop files only nest parens via command
    # substitution and array-literal syntax, both of which balance.
    paren_depth_at: list[int] = []
    _running_depth = 0
    for _raw in lines:
        paren_depth_at.append(_running_depth)
        _running_depth += _raw.count("(") - _raw.count(")")
        if _running_depth < 0:
            _running_depth = 0

    def _current_exit_dead() -> bool:
        return exit_dead_stack[-1] if exit_dead_stack else top_level_exit_dead

    def _mark_exit_dead() -> None:
        nonlocal top_level_exit_dead
        if exit_dead_stack:
            exit_dead_stack[-1] = True
        else:
            top_level_exit_dead = True

    def _line_reachable() -> bool:
        return not any(dead for _, dead in stack) and not _current_exit_dead()

    def _push(kind: str, dead: bool) -> None:
        stack.append((kind, dead))
        exit_dead_stack.append(False)

    def _pop() -> None:
        stack.pop()
        exit_dead_stack.pop()

    for _idx, raw in enumerate(lines):
        stripped = raw.strip()

        if pending_cond is not None:
            if _THEN_ONLY_RE.match(stripped):
                shape = _condition_shape(pending_cond)
                _push("if", shape in {"false", "unrecognized"})
                pending_cond = None
                reachable.append(_line_reachable())
                continue
            # A bare `if <cond>` not immediately followed by `then` on its
            # own line does not match this model's recognized shape at all.
            pending_cond = None

        match = _IF_THEN_RE.match(stripped)
        if match:
            shape = _condition_shape(match.group("cond"))
            _push("if", shape in {"false", "unrecognized"})
            reachable.append(_line_reachable())
            continue

        match = _ELIF_THEN_RE.match(stripped)
        if match and stack and stack[-1][0] == "if":
            shape = _condition_shape(match.group("cond"))
            stack[-1] = ("if", shape in {"false", "unrecognized"})
            reachable.append(_line_reachable())
            continue

        if _ELSE_RE.match(stripped) and stack and stack[-1][0] == "if":
            stack[-1] = ("if", not stack[-1][1])
            reachable.append(_line_reachable())
            continue

        if _FI_RE.match(stripped) and stack and stack[-1][0] == "if":
            _pop()
            reachable.append(_line_reachable())
            continue

        match = _IF_ONLY_RE.match(stripped)
        if match and stripped.startswith("if "):
            pending_cond = match.group("cond")
            reachable.append(_line_reachable())
            continue

        match = _WHILE_DO_RE.match(stripped)
        if match:
            shape = _condition_shape(match.group("cond"))
            _push("loop", shape in {"false", "unrecognized"})
            reachable.append(_line_reachable())
            continue

        match = _UNTIL_DO_RE.match(stripped)
        if match:
            shape = _condition_shape(match.group("cond"))
            _push("loop", shape in {"true", "unrecognized"})
            reachable.append(_line_reachable())
            continue

        if _FOR_DO_RE.match(stripped) or _SELECT_DO_RE.match(stripped):
            _push("loop", True)
            reachable.append(_line_reachable())
            continue

        if _DONE_RE.match(stripped) and stack and stack[-1][0] == "loop":
            _pop()
            reachable.append(_line_reachable())
            continue

        if _CASE_IN_RE.match(stripped):
            # `case`/`esac` pattern matching is not modeled at all: nothing
            # inside can be proven reachable, regardless of which arm a
            # critical invocation sits in.
            _push("case", True)
            reachable.append(_line_reachable())
            continue

        if _ESAC_RE.match(stripped) and stack and stack[-1][0] == "case":
            _pop()
            reachable.append(_line_reachable())
            continue

        line_reachable = _line_reachable()
        reachable.append(line_reachable)
        if line_reachable and paren_depth_at[_idx] == 0 and _EXIT_RE.match(stripped):
            _mark_exit_dead()

    return reachable


def _assert_executable_once(
    label: str, script_name: str, script: str, needle: str
) -> None:
    """Assert `needle` appears exactly once in `script`, as a real
    (non-comment), reachable command line, not sharing its line with an
    unevaluated `&&`/`||` guard, and not naming a command shadowed by an
    earlier function definition — anything else counts as absent."""
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
        f"{script_name}'s {label} line ({idx + 1}) is not proven reachable — an "
        "enclosing construct is dead, unrecognized (case/&&/||/exit/unaudited "
        "condition), or shadowed; extend the model deliberately if this is a "
        "legitimate new construct",
    )
    _assert(
        "&&" not in lines[idx] and "||" not in lines[idx],
        f"{script_name}'s {label} line ({idx + 1}) uses a compound && / || guard, "
        "which this model does not evaluate — express the guard as if/then/fi",
    )
    shadowed = _shadowed_commands(script) & set(_needle_tokens(needle))
    _assert(
        not shadowed,
        f"{script_name}'s {label} invokes {sorted(shadowed)}, which is redefined "
        "as a shell function in this script — the real command would never run",
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
        return _wrap_if(script, needle, "false")

    def _wrap_if(script: str, needle: str, condition: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\n" if line.endswith("\n") else ""
                wrapped = (
                    f"{indent}if {condition}; then\n"
                    f"{line.rstrip(chr(10))}\n"
                    f"{indent}fi{newline}"
                )
                lines[i] = wrapped
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to wrap")

    def _wrap_while(script: str, needle: str, condition: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\n" if line.endswith("\n") else ""
                lines[i] = (
                    f"{indent}while {condition}; do\n"
                    f"{line.rstrip(chr(10))}\n"
                    f"{indent}done{newline}"
                )
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to wrap")

    def _prepend_line(script: str, needle: str, new_line: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                lines.insert(i, f"{indent}{new_line}\n")
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to prepend before")

    def _wrap_case(script: str, needle: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\n" if line.endswith("\n") else ""
                wrapped = (
                    f'{indent}case "$RUNNER_OS" in\n'
                    f"{indent}  *)\n"
                    f"{line.rstrip(chr(10))}\n"
                    f"{indent}    ;;\n"
                    f"{indent}esac{newline}"
                )
                lines[i] = wrapped
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to wrap in case")

    def _wrap_for(script: str, needle: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                indent = line[: len(line) - len(line.lstrip())]
                newline = "\n" if line.endswith("\n") else ""
                wrapped = (
                    f"{indent}for _unused in 1; do\n"
                    f"{line.rstrip(chr(10))}\n"
                    f"{indent}done{newline}"
                )
                lines[i] = wrapped
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to wrap in for")

    def _guard_with_false_and(script: str, needle: str) -> str:
        lines = script.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if needle in line and not _is_comment(line):
                lines[i] = line.replace(needle, f"false && {needle}", 1)
                return "".join(lines)
        raise AssertionError(f"mutation setup: {needle!r} not found to && guard")

    def _shadow_command(script: str, command: str) -> str:
        lines = script.splitlines(keepends=True)
        insert_at = 1 if lines and lines[0].startswith("#!") else 0
        lines[insert_at:insert_at] = [f"{command}() {{\n", ":\n", "}\n"]
        return "".join(lines)

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
        # ---- the seven adversarially-demonstrated escapes, reproduced exactly ----
        (
            "escape#1: root install command commented out",
            _comment_out(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "escape#2: non-root runuser command commented out",
            _comment_out(workflow, MERGE_DRIVER_WORKFLOW_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "escape#3: real browser phase commented out",
            workflow,
            _comment_out(merge_driver, MERGE_REAL_EXECUTION_INVOCATION),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "escape#4: root install hidden behind false branch",
            _wrap_if_false(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        # Literal equivalents of `false` must not reopen escape#4: exercise
        # comparison evaluation, boolean command evaluation, and constant
        # loop-body reachability across the critical hops.
        (
            "root install hidden behind unequal literal comparison",
            _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "[ 1 = 2 ]"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install hidden behind /bin/false",
            _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "/bin/false"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "runuser hidden behind ! true",
            _wrap_if(workflow, MERGE_DRIVER_WORKFLOW_INVOCATION, "! true"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase hidden in while false",
            workflow,
            _wrap_while(merge_driver, MERGE_REAL_EXECUTION_INVOCATION, "false"),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase hidden in until true",
            workflow,
            _wrap_while(merge_driver, MERGE_REAL_EXECUTION_INVOCATION, "true").replace(
                "while true; do", "until true; do", 1
            ),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "escape#5: install-chromium.sh replaced by exit 0",
            workflow,
            merge_driver,
            wasm_check,
            "#!/usr/bin/env bash\nexit 0\n",
            browser_test_ci,
            browser_test,
        ),
        (
            "escape#6: browser-wasm-test-ci.sh replaced by exit 0",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            "#!/usr/bin/env bash\nexit 0\n",
            browser_test,
        ),
        (
            "escape#7: browser-wasm-test.sh replaced by compile-only cargo check",
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
        # ---- additional coupling cases ----
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
        # ---- fail-closed inversion: constructs not modeled at all ----
        (
            "root install hidden inside an unmodeled case statement",
            _wrap_case(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase hidden inside an unmodeled case statement",
            workflow,
            _wrap_case(merge_driver, MERGE_REAL_EXECUTION_INVOCATION),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install hidden inside an unmodeled for-loop header",
            _wrap_for(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase hidden inside an unmodeled for-loop header",
            workflow,
            _wrap_for(merge_driver, MERGE_REAL_EXECUTION_INVOCATION),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            'root install gated behind an always-false unary test ([ -n "" ])',
            _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, '[ -n "" ]'),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            'runuser gated behind [ ! -z "" ] (negated always-true -z)',
            _wrap_if(workflow, MERGE_DRIVER_WORKFLOW_INVOCATION, '[ ! -z "" ]'),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase gated behind an out-of-range integer comparison ([ 1 -gt 2 ])",
            workflow,
            _wrap_if(merge_driver, MERGE_REAL_EXECUTION_INVOCATION, "[ 1 -gt 2 ]"),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install gated behind [ 1 -lt 0 ]",
            _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "[ 1 -lt 0 ]"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install gated behind [ 0 -ge 1 ]",
            _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "[ 0 -ge 1 ]"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install gated behind [ 1 -le 0 ]",
            _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "[ 1 -le 0 ]"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install gated behind a bare variable used as the command (F=false; if $F)",
            _prepend_line(
                _wrap_if(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "$INSTALL_GATE"),
                "if $INSTALL_GATE; then",
                "F=false",
            ).replace("$INSTALL_GATE", "$F"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase gated behind a false && chain on the same line",
            workflow,
            _guard_with_false_and(merge_driver, MERGE_REAL_EXECUTION_INVOCATION),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install gated behind a false && chain on the same line",
            _guard_with_false_and(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "root install preceded by an unconditional exit 0 in the same scope",
            _prepend_line(workflow, WORKFLOW_CHROMIUM_INSTALL_INVOCATION, "exit 0"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "real browser phase preceded by an unconditional exit 0 in the same scope",
            workflow,
            _prepend_line(merge_driver, MERGE_REAL_EXECUTION_INVOCATION, "exit 0"),
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "rust.yml's bash invocations shadowed by a no-op bash() function",
            _shadow_command(workflow, "bash"),
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            browser_test,
        ),
        (
            "browser-wasm-test.sh's cargo invocation shadowed by a no-op function",
            workflow,
            merge_driver,
            wasm_check,
            install_chromium,
            browser_test_ci,
            _shadow_command(browser_test, "cargo"),
        ),
        (
            "install-chromium.sh's apt-get invocation shadowed by a no-op function",
            workflow,
            merge_driver,
            wasm_check,
            _shadow_command(install_chromium, "apt-get"),
            browser_test_ci,
            browser_test,
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
