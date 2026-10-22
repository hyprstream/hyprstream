#!/usr/bin/env python3
r"""Static gate for the versioned build matrix contract (native CI plan).

WHY THIS EXISTS
---------------
The native self-hosted CI/CD + ARM64 CUDA direction (2026-09-13) requires one
shared, reviewable contract for what we build, where it may execute, and how
its caches/artifacts are named — plus a guardrail so GitHub-hosted runners
cannot silently regain ground. `.github/build-matrix.json` is that contract:
architecture/backend rows (with runner class, builder-image pin, libtorch
identity, cache namespace, artifact names) and the `hosted_allowlist`, which
doubles as the migration checklist for every job still on `ubuntu-latest`.

Two decisions this gate freezes:

1. **Hosted-runner zero scan (plan N2.5, landed early):** every job in
   `.github/workflows/*.yml` or `*.yaml` that runs on a GitHub-hosted label must be
   explicitly allowlisted in the manifest, and every allowlist entry must
   still correspond to a hosted job. New hosted jobs fail; completed
   migrations must delete their entry (a stale entry fails too, so the
   checklist cannot rot). Only reviewed `external_service` entries (e.g. the
   Claude Code actions) may target `github-hosted` permanently.
2. **ARM CUDA fail-closed gate (plan N3):** any matrix row combining
   `arch=aarch64` with a GPU backend (cuda*/rocm*) must carry
   `status="gated"` with a `gate` block recording the blocker and build-only
   evidence. Marking such a row `active` requires `gate.hardware_proof` — a
   URL from a green `arm-cuda-enable-preflight` run on real NVIDIA hardware.
   The Graviton pool has no GPU (spike run 32897623715 recorded
   `torch.cuda.is_available() == False`), so a link-only "success" can never
   unlock ARM CUDA.

METHODOLOGY
-----------
Pure Python-stdlib inspection — no third-party YAML/JSON parser beyond
`json`, no network, no runner, no build. A focused fail-closed line parser
(the same technique as `check_rust_workflow_matrix.py`) extracts each job's
`runs-on:` labels; `runs-on` shapes it cannot classify (missing, multiline,
or fully dynamic with no static labels) are REJECTED, never guessed at. The
manifest is validated structurally: row ids unique, target triples matching
their architecture, runner classes declared, builder images digest-pinned
(directly or via `pin:` reference to `.github/builder-image.env`), cache
namespaces unique and containing their architecture/backend tokens, and
artifact templates carrying `{version}` + arch tokens. Failures exit
non-zero with concrete messages. Do NOT add `|| true`.

Row↔workflow binding is intentionally NOT enforced beyond the allowlist scan
in this version: rows describe build targets and the allowlist describes
migration debt; binding each job to a row lands with the Wave-2 migrations
that make it observable.

NON-VACUOUS
-----------
Unit tests in `scripts/tests/test_check_build_matrix.py` feed known-bad
mutations (unallowlisted hosted job, stale allowlist entry, ungated ARM CUDA
row activated without hardware proof, duplicate ids, unpinned image,
namespace collision, unclassifiable runs-on) through the same check
functions; each must be rejected. A clean synthetic repo must pass.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
MANIFEST_PATH = ROOT / ".github" / "build-matrix.json"
BUILDER_PIN_PATH = ROOT / ".github" / "builder-image.env"

SUPPORTED_SCHEMA_VERSIONS = {1}
ARCHES = {"x86_64", "aarch64"}
TARGET_TRIPLES = {
    "x86_64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
}
GPU_BACKENDS = {"cuda128", "cuda130", "rocm71"}
BACKENDS = GPU_BACKENDS | {"cpu", "universal"}
PRODUCTS = {"appimage", "appimage-universal", "rust-workspace"}
CACHE_KINDS = {"none", "s3-sccache", "s3-sccache+persistent-target"}
DIGEST_PIN_RE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
PIN_REF_RE = re.compile(r"^pin:([^@\s]+)$")
HOSTED_LABEL_RE = re.compile(
    r"^(ubuntu|macos|windows)-[a-z0-9.\-]+$"
)
VALID_ROW_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
URL_RE = re.compile(r"^https?://")


class Fail(Exception):
    """One concrete contract violation."""


def fail(msg: str) -> None:
    raise Fail(msg)


# --------------------------------------------------------------------------
# Minimal GitHub-Actions-YAML structural parser (stdlib only), following the
# constrained fail-closed technique of check_rust_workflow_matrix.py. We
# extract exactly one thing: each direct child of `jobs:` and its `runs-on:`
# value. Unsupported shapes are rejected rather than guessed at.


def _strip_key(token: str) -> str:
    """Strip surrounding YAML quotes and trailing comment from a token."""
    token = token.split("#", 1)[0].strip()
    if len(token) >= 2 and token[0] in "\"'" and token[-1] == token[0]:
        token = token[1:-1]
    return token


def _job_blocks(text: str) -> dict[str, list[str]]:
    """Return the raw line list for each direct child of the top-level
    `jobs:` map (2-space indent, this repo's uniform style)."""
    in_jobs = False
    blocks: dict[str, list[str]] = {}
    cur: str | None = None
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip())
            if indent == 0:
                cur = None
                in_jobs = _strip_key(line.split(":", 1)[0]) == "jobs"
                continue
            if in_jobs and indent == 2:
                stripped = line.strip()
                if stripped.endswith(":"):
                    cur = _strip_key(stripped[:-1])
                    blocks[cur] = []
                    continue
                # Every direct child of `jobs:` must be a block mapping.
                # Inline/flow mappings are deliberately rejected because this
                # parser cannot inspect their runs-on value safely.
                if ":" in stripped:
                    job = _strip_key(stripped.split(":", 1)[0])
                    fail(
                        f"job `{job}` uses an unsupported inline mapping; "
                        "write a block job with a single-line `runs-on:`"
                    )
        if in_jobs and cur is not None:
            blocks[cur].append(line)
    return blocks


def _job_runs_on(text: str) -> dict[str, list[str]]:
    """Return {job_name: static_runs_on_labels} for every job.

    Labels are normalized: brackets/quotes stripped, `${{ ... }}` expression
    tokens dropped. A job whose runs-on cannot be found on a single line, or
    that has no runs-on at all, is rejected — the gate must never silently
    skip a job it cannot read.
    """
    result: dict[str, list[str]] = {}
    for job, lines in _job_blocks(text).items():
        raw = None
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("runs-on:"):
                raw = stripped.split(":", 1)[1]
                break
            if stripped.startswith("- runs-on:") or stripped.startswith(
                "runs-on ="
            ):
                fail(
                    f"job `{job}` uses an unsupported runs-on form "
                    f"({stripped!r}); the gate refuses shapes it cannot parse"
                )
        if raw is None:
            fail(
                f"job `{job}` has no single-line `runs-on:`; reusable-workflow"
                " jobs and other unsupported shapes must be taught to this"
                " gate explicitly, never skipped"
            )
        value = _strip_key(raw)
        if not value:
            fail(f"job `{job}` has an empty runs-on value")
        if value.startswith("["):
            if not value.endswith("]"):
                fail(
                    f"job `{job}` has a multi-line/unclosed runs-on list;"
                    " unsupported shape"
                )
            value = value[1:-1]
        labels = []
        for token in value.split(","):
            token = token.strip()
            if not token or "${{" in token:
                continue  # dynamic expression tokens carry no static label
            labels.append(_strip_key(token))
        if not labels:
            fail(
                f"job `{job}` runs-on resolves to no static labels;"
                " cannot classify"
            )
        result[job] = labels
    return result


def _classify(labels: list[str]) -> str:
    """Classify a runs-on label list as hosted / self-hosted / unknown."""
    if "self-hosted" in labels:
        return "self-hosted"
    if any(HOSTED_LABEL_RE.match(label) for label in labels):
        return "hosted"
    return "unknown"


def scan_workflows(workflows_dir: pathlib.Path) -> dict[str, dict[str, str]]:
    """Return {workflow_filename: {job: classification}} for all workflows."""
    if not workflows_dir.is_dir():
        fail(f"workflow directory missing: {workflows_dir}")
    scanned: dict[str, dict[str, str]] = {}
    paths = sorted({*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml")})
    for path in paths:
        text = path.read_text(encoding="utf-8")
        jobs = _job_runs_on(text)
        scanned[path.name] = {job: _classify(labels) for job, labels in jobs.items()}
    if not scanned:
        fail(f"no workflow files found under {workflows_dir}")
    return scanned


# --------------------------------------------------------------------------
# Manifest validation


def _require(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


def _check_builder_image(row: dict, root: pathlib.Path) -> None:
    image = row.get("builder_image")
    if image is None:
        return
    _require(
        isinstance(image, str) and image,
        f"row `{row.get('id')}`: builder_image must be null or a string",
    )
    m = PIN_REF_RE.match(image)
    if m:
        pin_path = root / m.group(1)
        _require(
            pin_path.is_file(),
            f"row `{row['id']}`: builder_image pin file {m.group(1)!r} not found",
        )
        lines = [
            ln.strip()
            for ln in pin_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        _require(
            len(lines) == 1 and DIGEST_PIN_RE.match(lines[0]) is not None,
            f"row `{row['id']}`: pin file {m.group(1)!r} must hold exactly one"
            " digest-pinned image reference (<image>@sha256:<64 hex>)",
        )
    else:
        _require(
            DIGEST_PIN_RE.match(image) is not None,
            f"row `{row['id']}`: builder_image must be digest-pinned"
            f" (@sha256:<64 hex>) or a `pin:<path>` reference, got {image!r}",
        )


def check_manifest(manifest: dict, root: pathlib.Path) -> None:
    """Validate manifest structure, rows, and the ARM-CUDA gate."""
    version = manifest.get("schema_version")
    _require(
        version in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported schema_version {version!r}; supported:"
        f" {sorted(SUPPORTED_SCHEMA_VERSIONS)}",
    )

    classes = manifest.get("runner_classes")
    _require(
        isinstance(classes, dict) and classes,
        "runner_classes must be a non-empty object",
    )
    for name, cls in classes.items():
        _require(
            isinstance(cls, dict), f"runner class `{name}` must be an object"
        )
        labels = cls.get("labels")
        _require(
            isinstance(labels, list)
            and labels
            and all(isinstance(x, str) and x for x in labels),
            f"runner class `{name}`: labels must be a non-empty string list",
        )
        _require(
            cls.get("architecture") in ARCHES,
            f"runner class `{name}`: architecture must be one of {sorted(ARCHES)}",
        )
        _require(
            cls.get("status"),
            f"runner class `{name}`: status is required",
        )
        _require(
            isinstance(cls.get("substrate"), str) and cls.get("substrate"),
            f"runner class `{name}`: substrate description is required",
        )
    _require(
        "github-hosted" in classes,
        "runner_classes must declare the legacy `github-hosted` class so the"
        " allowlist semantics stay explicit",
    )

    rows = manifest.get("rows")
    _require(isinstance(rows, list) and rows, "rows must be a non-empty list")
    seen_ids: set[str] = set()
    seen_namespaces: dict[str, str] = {}
    seen_artifacts: dict[str, str] = {}
    for row in rows:
        _require(isinstance(row, dict), "each row must be an object")
        rid = row.get("id")
        _require(
            isinstance(rid, str) and VALID_ROW_ID_RE.match(rid) is not None,
            f"row id {rid!r} must match {VALID_ROW_ID_RE.pattern}",
        )
        _require(rid not in seen_ids, f"duplicate row id `{rid}`")
        seen_ids.add(rid)

        arch = row.get("arch")
        _require(arch in ARCHES, f"row `{rid}`: arch must be one of {sorted(ARCHES)}")
        _require(
            row.get("target_triple") == TARGET_TRIPLES[arch],
            f"row `{rid}`: target_triple must be {TARGET_TRIPLES[arch]!r}"
            f" for arch {arch}",
        )
        backend = row.get("backend")
        _require(
            backend in BACKENDS,
            f"row `{rid}`: backend must be one of {sorted(BACKENDS)}",
        )
        _require(
            row.get("product") in PRODUCTS,
            f"row `{rid}`: product must be one of {sorted(PRODUCTS)}",
        )
        status = row.get("status")
        _require(
            status in {"active", "gated"},
            f"row `{rid}`: status must be active or gated",
        )

        rclass = row.get("runner_class")
        _require(
            isinstance(rclass, str) and rclass in classes,
            f"row `{rid}`: runner_class {rclass!r} is not declared in"
            " runner_classes",
        )
        _require(
            classes[rclass]["architecture"] == arch,
            f"row `{rid}`: arch {arch} does not match runner class"
            f" `{rclass}` architecture {classes[rclass]['architecture']}",
        )

        _check_builder_image(row, root)

        libtorch = row.get("libtorch")
        _require(
            isinstance(libtorch, dict)
            and isinstance(libtorch.get("source"), str)
            and libtorch.get("source")
            and isinstance(libtorch.get("identity"), str)
            and libtorch.get("identity"),
            f"row `{rid}`: libtorch source+identity are required",
        )

        cache = row.get("cache")
        _require(
            isinstance(cache, dict) and cache.get("kind") in CACHE_KINDS,
            f"row `{rid}`: cache.kind must be one of {sorted(CACHE_KINDS)}",
        )
        namespace = cache.get("namespace")
        if namespace is not None:
            _require(
                isinstance(namespace, str) and namespace,
                f"row `{rid}`: cache.namespace must be a non-empty string"
                " or null",
            )
            _require(
                arch in namespace,
                f"row `{rid}`: cache namespace {namespace!r} must contain the"
                f" architecture token {arch!r} (target isolation law)",
            )
            if backend != "universal":
                _require(
                    backend in namespace,
                    f"row `{rid}`: cache namespace {namespace!r} must contain"
                    f" the backend token {backend!r}",
                )
            _require(
                namespace not in seen_namespaces,
                f"cache namespace {namespace!r} shared by rows"
                f" `{seen_namespaces.get(namespace, rid)}` and `{rid}`",
            )
            seen_namespaces[namespace] = rid

        artifacts = row.get("artifacts")
        _require(
            isinstance(artifacts, list),
            f"row `{rid}`: artifacts must be a list",
        )
        for tpl in artifacts:
            _require(
                isinstance(tpl, str) and "{version}" in tpl,
                f"row `{rid}`: artifact template {tpl!r} must contain"
                " {version}",
            )
            _require(
                arch in tpl,
                f"row `{rid}`: artifact template {tpl!r} must contain the"
                f" architecture token {arch!r}",
            )
            if backend != "universal":
                _require(
                    backend in tpl,
                    f"row `{rid}`: artifact template {tpl!r} must contain the"
                    f" backend token {backend!r}",
                )
            _require(
                tpl not in seen_artifacts,
                f"artifact template {tpl!r} shared by rows"
                f" `{seen_artifacts.get(tpl, rid)}` and `{rid}`",
            )
            seen_artifacts[tpl] = rid
        staging = row.get("staging_artifacts", [])
        _require(
            isinstance(staging, list),
            f"row `{rid}`: staging_artifacts must be a list",
        )
        for name in staging:
            _require(
                isinstance(name, str) and name and "{version}" not in name,
                f"row `{rid}`: staging artifact {name!r} must be a literal"
                " name (no {version})",
            )

        # ARM GPU fail-closed gate.
        if arch == "aarch64" and backend in GPU_BACKENDS:
            gate = row.get("gate")
            _require(
                isinstance(gate, dict),
                f"row `{rid}`: aarch64 GPU backend requires a `gate` object",
            )
            _require(
                isinstance(gate.get("blocker"), str) and gate.get("blocker"),
                f"row `{rid}`: gate.blocker is required",
            )
            boe = gate.get("build_only_evidence")
            _require(
                isinstance(boe, str) and URL_RE.match(boe) is not None,
                f"row `{rid}`: gate.build_only_evidence must be a URL",
            )
            if status == "active":
                proof = gate.get("hardware_proof")
                _require(
                    isinstance(proof, str) and URL_RE.match(proof) is not None,
                    f"row `{rid}`: marking an aarch64 {backend} row active"
                    " requires gate.hardware_proof (a green"
                    " arm-cuda-enable-preflight run URL on real NVIDIA"
                    " hardware); without it the row must stay gated",
                )
        # Gate-object consistency both ways. An ACTIVATED aarch64 GPU row keeps
        # its gate block — with hardware_proof filled in — as the permanent
        # evidence trail; for every other row a gate object on an active row
        # means a half-finished edit and is rejected.
        if status == "gated":
            _require(
                isinstance(row.get("gate"), dict),
                f"row `{rid}`: gated rows must carry a `gate` object",
            )
        else:
            gate = row.get("gate")
            if gate is not None:
                activated_gpu_row = (
                    arch == "aarch64"
                    and backend in GPU_BACKENDS
                    and isinstance(gate, dict)
                    and isinstance(gate.get("hardware_proof"), str)
                    and URL_RE.match(gate["hardware_proof"]) is not None
                )
                _require(
                    activated_gpu_row,
                    f"row `{rid}`: active rows must not carry a `gate` object"
                    " (only an activated aarch64 GPU row with"
                    " gate.hardware_proof may keep its evidence trail)",
                )


def check_allowlist(
    manifest: dict, scanned: dict[str, dict[str, str]]
) -> list[str]:
    """Enforce the hosted-runner allowlist in both directions.

    Returns a list of remaining hosted (workflow, job) pairs — the open
    migration debt — for the success summary.
    """
    classes = manifest["runner_classes"]
    allowlist = manifest.get("hosted_allowlist")
    if not isinstance(allowlist, dict):
        fail("hosted_allowlist must be an object")

    hosted_pairs: set[tuple[str, str]] = set()
    for wf_name, jobs in scanned.items():
        for job, cls in jobs.items():
            if cls == "hosted":
                hosted_pairs.add((wf_name, job))
            elif cls == "unknown":
                fail(
                    f"{wf_name} job `{job}` runs on labels this gate cannot"
                    " classify; teach the gate or use a declared class"
                )

    listed_pairs: set[tuple[str, str]] = set()
    for wf_name, jobs in allowlist.items():
        _require(
            isinstance(jobs, dict) and jobs,
            f"hosted_allowlist[{wf_name!r}] must be a non-empty job map",
        )
        for job, entry in jobs.items():
            listed_pairs.add((wf_name, job))
            _require(
                isinstance(entry, dict),
                f"hosted_allowlist[{wf_name}][{job}] must be an object",
            )
            reason = entry.get("reason")
            _require(
                isinstance(reason, str) and reason,
                f"hosted_allowlist[{wf_name}][{job}]: reason is required",
            )
            target = entry.get("target_class")
            _require(
                isinstance(target, str) and target in classes,
                f"hosted_allowlist[{wf_name}][{job}]: target_class"
                f" {target!r} is not a declared runner class",
            )
            if entry.get("external_service"):
                _require(
                    target == "github-hosted",
                    f"hosted_allowlist[{wf_name}][{job}]: external_service"
                    " entries must target the github-hosted class",
                )
            else:
                _require(
                    target != "github-hosted",
                    f"hosted_allowlist[{wf_name}][{job}]: only reviewed"
                    " external services may keep target_class github-hosted;"
                    " everything else must name a self-hosted migration"
                    " target",
                )

    missing = sorted(hosted_pairs - listed_pairs)
    if missing:
        detail = ", ".join(f"{wf}:{job}" for wf, job in missing)
        fail(
            "GitHub-hosted job(s) not in the build-matrix hosted_allowlist"
            f" (new hosted execution is rejected — migrate the job or"
            f" review it into the allowlist): {detail}"
        )
    stale = sorted(listed_pairs - hosted_pairs)
    if stale:
        detail = ", ".join(f"{wf}:{job}" for wf, job in stale)
        fail(
            "stale hosted_allowlist entr(ies) whose job no longer executes on"
            " a GitHub-hosted runner (or no longer exists) — delete the"
            f" entr(ies) as part of the completed migration: {detail}"
        )
    return [f"{wf}:{job}" for wf, job in sorted(hosted_pairs)]


def run_checks(root: pathlib.Path) -> list[str]:
    """Run the full gate against `root`. Returns remaining hosted jobs."""
    manifest_path = root / ".github" / "build-matrix.json"
    if not manifest_path.is_file():
        fail(f"build matrix manifest missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"build matrix manifest is not valid JSON: {exc}")
    _require(isinstance(manifest, dict), "manifest must be a JSON object")

    check_manifest(manifest, root)
    scanned = scan_workflows(root / ".github" / "workflows")
    return check_allowlist(manifest, scanned)


def main() -> int:
    try:
        remaining = run_checks(ROOT)
    except Fail as exc:
        print(f"check_build_matrix: FAIL: {exc}", file=sys.stderr)
        return 1
    print("check_build_matrix: OK")
    print(
        f"  hosted migration debt: {len(remaining)} job(s) still on"
        " GitHub-hosted runners (see hosted_allowlist in"
        " .github/build-matrix.json)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
