#!/usr/bin/env python3
"""Fail-closed checks for schema coverage and publication contracts."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SURFACES = ("cli", "mcp", "factory", "vfs", "typescript")
OWNER_DIRECTORIES = (
    "crates/hyprstream/schema",
    "crates/hyprstream-discovery/schema",
    "crates/hyprstream-pay/schema",
    "crates/hyprstream-rpc/schema",
    "crates/hyprstream-rpc-build/tests",
    "crates/hyprstream-rpc-std/schema",
    "crates/hyprstream-workers/schema",
)
PUBLIC_GLOBS = {
    "docs/*.md", "docs/adr/**/*.md", "docs/contracts/**/*.md",
    "docs/deployment/**/*.md", "docs/network/**/*.md", "docs/security/**/*.md", "docs/standards/*.md",
    "docs/standards/adr/**/*.md", "docs/standards/analysis/**/*.md",
    "docs/standards/runbooks/**/*.md", "docs/standards/v16/**/*.md",
}
EXCLUSIONS = {
    ".fleet-coord/**": "local coordination and handoff state is not publication input",
    "docs/plans/**": "unpublished planning artifacts are not corpus material",
    "docs/standards/rfc/**": "third-party or separately licensed RFC renderings require their own release review",
    "docs/.obsidian/**": "editor-local state",
    "**/.env*": "secrets and local credentials are never publication input",
    "codegen-out/**": "ignored generated artifacts are not source corpus",
    "dist/**": "release output is not source corpus",
}
CGR_ROOTS = {
    "crates/hyprstream/build.rs": ["crates/hyprstream/schema", "crates/hyprstream-rpc/schema"],
    "crates/hyprstream-discovery/build.rs": ["crates/hyprstream-discovery/schema", "crates/hyprstream-rpc/schema"],
    "crates/hyprstream-rpc/build.rs": ["crates/hyprstream-rpc/schema"],
    "crates/hyprstream-rpc-std/build.rs": ["crates/hyprstream-rpc-std/schema", "crates/hyprstream-rpc/schema"],
    "crates/hyprstream-workers/build.rs": ["crates/hyprstream-workers/schema", "crates/hyprstream-rpc/schema"],
}
EXPECTED_CGR_INVOCATIONS = {
    "crates/hyprstream/build.rs": {
        "invocations": [
            {"source_root": "crates/hyprstream/schema", "import_roots": ["crates/hyprstream-rpc/schema", "crates/hyprstream/schema"], "schemas": ["tui", "compositor_ipc"]},
            {"source_root": "crates/hyprstream-rpc/schema", "import_roots": ["crates/hyprstream-rpc/schema", "crates/hyprstream/schema"], "schemas": ["streaming", "nine"]},
        ],
    },
    "crates/hyprstream-discovery/build.rs": {"invocations": [{"source_root":"crates/hyprstream-discovery/schema","import_roots":["crates/hyprstream-rpc/schema"],"schemas":["discovery"]}]},
    "crates/hyprstream-rpc/build.rs": {"invocations": [{"source_root":"crates/hyprstream-rpc/schema","import_roots":[],"schemas":["common","streaming","events","annotations","optional","nine"]}]},
    "crates/hyprstream-rpc-std/build.rs": {"invocations": [{"source_root":"crates/hyprstream-rpc-std/schema","import_roots":["crates/hyprstream-rpc/schema","crates/hyprstream-rpc-std/schema"],"schemas":["inference","model","registry","policy","mcp","metrics","service_events","chat_core","oauth"]}]},
    "crates/hyprstream-workers/build.rs": {"invocations": [{"source_root":"crates/hyprstream-workers/schema","import_roots":["crates/hyprstream-rpc/schema"],"schemas":["worker","workflow"]}]},
}
PACKAGE_REQUIREMENTS = [
    "Publish only validated manifest entries.",
    "Preserve source license and provenance per document.",
    "Do not package excluded paths or generated output without a separate license review.",
    "Keep docs/system-ontology.md authoritative; projections may link to it but may not redefine canonical terms.",
]


class CatalogError(Exception):
    pass


def required(condition: bool, message: str) -> None:
    if not condition:
        raise CatalogError(message)


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], text=True, capture_output=True, check=False)
    required(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogError(f"{path}: {error}") from error


def tracked(repo: Path, *patterns: str) -> list[str]:
    output = git(repo, "ls-files", "--", *patterns)
    return [path for path in output.splitlines() if path]


def text(repo: Path, path: str, mutations: dict[str, str] | None) -> str:
    if mutations and path in mutations:
        return mutations[path]
    return (repo / path).read_text(encoding="utf-8")


def source_services(
    repo: Path, mutations: dict[str, str] | None = None, tracked_sources: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    cli = strip_rust_comments(text(repo, "crates/hyprstream/src/cli/schema_cli.rs", mutations))
    mcp = strip_rust_comments(text(repo, "crates/hyprstream/src/services/mcp_service.rs", mutations))
    factories = strip_rust_comments(text(repo, "crates/hyprstream/src/services/factories.rs", mutations))
    vfs = strip_rust_comments(text(repo, "crates/hyprstream-rpc-std/src/vfs_mount.rs", mutations))

    registrations: list[tuple[int, str, list[str] | None]] = [
        (match.start(), match.group(1), None)
        for match in re.finditer(r'build_service_command\(\s*"([a-z0-9-]+)"', cli)
    ]
    for match in re.finditer(
        r'let\s+(?P<binding>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*Command::new\("(?P<service>[a-z0-9-]+)"\)'
        r'(?P<body>.*?)\btool\s*=\s*tool\.subcommand\((?P=binding)\);',
        cli,
        re.DOTALL,
    ):
        methods = re.findall(r'Command::new\("([a-z0-9-]+)"\)', match.group("body"))
        registrations.append((match.start(), match.group("service"), methods))
    registrations.sort()
    cli_services = [service for _, service, _ in registrations]
    manual_services = {service: methods for _, service, methods in registrations if methods is not None}
    mcp_modules = re.findall(r"register_top_level!\(\s*reg,\s*([a-zA-Z0-9_:]+)::schema_metadata\(\)\s*\)", mcp)
    mcp_services = [module.rsplit("::", 1)[-1].removesuffix("_client") for module in mcp_modules]
    factory_matches = re.finditer(
        r'(?P<cfg>#\[cfg\(feature = "([^"]+)"\)\]\s*)?#\[service_factory\(\s*"(?P<name>[a-z0-9-]+)"',
        factories,
    )
    factory_services, features = [], {}
    for match in factory_matches:
        name = match.group("name")
        factory_services.append(name)
        if match.group(2):
            features[name] = f"feature={match.group(2)}"
    vfs_services = re.findall(r'impl_service_dispatch!\([^,]+,\s*"([a-z0-9-]+)"', vfs)
    ts_sources = tracked_sources if tracked_sources is not None else tracked(
        repo, "*.ts", "*.tsx", "*.js", "*.jsx", "package.json"
    )
    return {
        "cli": {
            "services": cli_services,
            "manual_services": manual_services,
            "method_policy": {
                "hidden": "excluded" if "if method.cli_hidden || method.is_streaming" in cli else "unknown",
                "streaming": "excluded" if "if method.cli_hidden || method.is_streaming" in cli else "unknown",
            },
        },
        "mcp": {
            "services": mcp_services,
            "method_policy": {
                "hidden": "excluded" if "if method.hidden {" in mcp else "unknown",
                "streaming": "included" if "if method.is_streaming {" in mcp else "unknown",
            },
        },
        "factory": {"services": factory_services, "feature_conditions": features},
        "vfs": {"services": vfs_services},
        "typescript": {"tracked_sources": ts_sources},
    }


def audited_input(repo: Path, event: str | None = None, revision: str | None = None) -> tuple[str, str]:
    event = event or os.environ.get("DOCS_CATALOG_EVENT", "local")
    revision = revision or os.environ.get("DOCS_CATALOG_AUDITED_COMMIT")
    if event == "pull_request":
        expected_head = os.environ.get("DOCS_CATALOG_AUDITED_HEAD")
        required(expected_head in {None, git(repo, "rev-parse", "HEAD")},
                 "pull-request checkout is not the workflow-supplied head")
        base = revision or git(repo, "merge-base", "HEAD", "refs/remotes/origin/main")
        git(repo, "cat-file", "-e", f"{base}^{{commit}}")
        required(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", base, "refs/remotes/origin/main"]).returncode == 0,
                 "pull-request audited input is not a remote main base ancestor")
        return event, base
    if event == "push":
        boundary = revision or git(repo, "rev-parse", "HEAD^")
        required(boundary != git(repo, "rev-parse", "HEAD"), "push audited input must precede pushed HEAD")
        required(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", boundary, "HEAD"]).returncode == 0,
                 "push audited input is not reachable from HEAD")
        return event, boundary
    if event == "local":
        return event, ""
    raise CatalogError(f"unsupported docs-catalog event {event}")


def corpus_paths(repo: Path, corpus: dict[str, Any]) -> list[str]:
    """Apply the manifest allow/deny rules to the exact tracked Markdown universe."""
    allow = [item["glob"] for item in corpus.get("public_prose", [])]
    deny = [item["glob"] for item in corpus.get("excluded", [])]
    candidates = tracked(repo, "docs/*.md", "docs/**/*.md")
    return sorted(path for path in candidates if any(path_matches(path, pattern) for pattern in allow)
                  and not any(path_matches(path, pattern) for pattern in deny))


def path_matches(path: str, pattern: str) -> bool:
    """Git-style path glob: `*` stays within one path component; `**` spans it."""
    pieces, index = ["^"], 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*" and pattern[index:index + 3] == "**/":
            pieces.append("(?:.*/)?"); index += 3
        elif char == "*" and pattern[index:index + 2] == "**":
            pieces.append(".*"); index += 2
        elif char == "*":
            pieces.append("[^/]*"); index += 1
        elif char == "?":
            pieces.append("[^/]"); index += 1
        else:
            pieces.append(re.escape(char)); index += 1
    return re.fullmatch("".join(pieces) + "$", path) is not None


def provenance_paths(repo: Path, corpus: dict[str, Any]) -> list[str]:
    manifests = [str(Path(directory).parent / "Cargo.toml") for directory in OWNER_DIRECTORIES]
    return sorted(set(tracked(repo, "*.capnp") + corpus_paths(repo, corpus) + tracked(repo, "build.rs", "**/build.rs") + [
        "crates/hyprstream/src/cli/schema_cli.rs", "crates/hyprstream/src/services/mcp_service.rs",
        "crates/hyprstream/src/services/factories.rs", "crates/hyprstream-rpc-std/src/vfs_mount.rs",
        ".github/license-boundary.toml", *manifests,
    ]))


def input_digest(repo: Path, paths: list[str], mutations: dict[str, str] | None = None,
                 tree: str | None = None) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if tree is None:
            content = text(repo, path, mutations).encode("utf-8")
        else:
            result = subprocess.run(["git", "-C", str(repo), "show", f"{tree}:{path}"], capture_output=True, check=False)
            required(result.returncode == 0, f"audited source tree omits {path}")
            content = result.stdout
        digest.update(path.encode("utf-8") + b"\0" + content + b"\0")
    return digest.hexdigest()


def check_provenance(record: dict[str, Any], repo: Path, label: str, corpus: dict[str, Any],
                     mutations: dict[str, str] | None = None, event: str | None = None,
                     revision: str | None = None) -> None:
    commit = record.get("source_commit")
    tree = record.get("source_tree")
    declared_digest = record.get("source_input_digest")
    required(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, f"{label} has invalid source_commit")
    required(isinstance(tree, str) and re.fullmatch(r"[0-9a-f]{40}", tree) is not None, f"{label} has invalid source_tree")
    required(isinstance(declared_digest, str) and re.fullmatch(r"[0-9a-f]{64}", declared_digest) is not None,
             f"{label} has invalid source_input_digest")
    topology, boundary = audited_input(repo, event, revision)
    # The Git pair is the durable event-boundary attestation.  Unlike a PR
    # intermediate, the base/push-before commit is present after merge, squash,
    # and rebase; the separately stored digest attests the selected source input.
    git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    required(git(repo, "rev-parse", f"{commit}^{{tree}}") == tree,
             f"{label} source_tree does not match source_commit")
    if topology != "local":
        required(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", boundary, commit]).returncode == 0,
                 f"{label} source_commit is not descended from the trusted {topology} boundary")
    staged_catalog = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet", "--",
                                     "docs/schema-catalog.json", "docs/corpus-sources.json"]).returncode != 0
    required(commit != git(repo, "rev-parse", "HEAD") or staged_catalog,
             f"{label} source_commit must not self-reference HEAD")
    required(tree != git(repo, "rev-parse", "HEAD^{tree}") or staged_catalog,
             f"{label} source_tree must not self-reference HEAD")
    paths = provenance_paths(repo, corpus)
    required(input_digest(repo, paths, mutations) == declared_digest,
             f"{label} current audited inputs differ from source_input_digest")
    if not mutations:
        required(input_digest(repo, paths, tree=tree) == declared_digest,
                 f"{label} source_tree does not reproduce the audited input digest")


def owner_manifest(repo: Path, schema_path: str, owner_directories: list[str]) -> tuple[str, str, str]:
    matches = [directory for directory in owner_directories if schema_path.startswith(f"{directory}/")]
    required(len(matches) == 1, f"{schema_path} does not map to exactly one owner directory")
    manifest = str(Path(matches[0]).parent / "Cargo.toml")
    source = text(repo, manifest, None)
    package = re.search(r'^name\s*=\s*"([^"]+)"', source, re.MULTILINE)
    license_match = re.search(r'^license\s*=\s*"([^"]+)"', source, re.MULTILINE)
    required(package is not None, f"{manifest} lacks package name")
    required(license_match is not None, f"{manifest} lacks package license")
    return matches[0], package.group(1), license_match.group(1)


def check_owner_directories(catalog: dict[str, Any], repo: Path) -> list[str]:
    owner_directories = catalog.get("owner_directories")
    required(owner_directories == list(OWNER_DIRECTORIES), "owner directory inventory drift")
    for directory in owner_directories:
        required((repo / directory).is_dir(), f"owner directory does not exist: {directory}")
        required((repo / directory).parent.joinpath("Cargo.toml").is_file(), f"owner manifest missing for {directory}")
    return owner_directories


def rust_lex(source: str, mask_literals: bool) -> str:
    """Offset-preserving Rust comment/literal lexer for the limited build.rs grammar."""
    out, index, block = [], 0, 0
    def blank(value: str) -> str:
        return "".join("\n" if char == "\n" else " " for char in value)
    while index < len(source):
        pair = source[index:index + 2]
        if block:
            if pair == "/*": block += 1; out.append("  "); index += 2; continue
            if pair == "*/": block -= 1; out.append("  "); index += 2; continue
            out.append("\n" if source[index] == "\n" else " "); index += 1; continue
        if pair == "//":
            end = source.find("\n", index); end = len(source) if end < 0 else end
            out.append(blank(source[index:end])); index = end; continue
        if pair == "/*": block += 1; out.append("  "); index += 2; continue
        raw = re.match(r"(?:br|r)(?P<hashes>#{0,255})\"", source[index:])
        if raw:
            close = '"' + raw.group("hashes")
            end = source.find(close, index + len(raw.group(0)))
            end = len(source) if end < 0 else end + len(close)
            value = source[index:end]; out.append(blank(value) if mask_literals else value); index = end; continue
        char = source[index]
        is_char = char == "'" and (index + 2 < len(source)) and (source[index + 1] == "\\" or source[index + 2] == "'")
        if char == '"' or is_char:
            quote, end = char, index + 1
            while end < len(source):
                if source[end] == "\\": end += 2; continue
                end += 1
                if source[end - 1] == quote: break
            value = source[index:end]; out.append(blank(value) if mask_literals else value); index = end; continue
        out.append(char); index += 1
    return "".join(out)


def strip_rust_comments(source: str) -> str:
    return rust_lex(source, False)


def strip_rust_noncode(source: str) -> str:
    return rust_lex(source, True)


def split_top_level(value: str) -> list[str]:
    items, start, depth = [], 0, 0
    for index, char in enumerate(value):
        if char in "([": depth += 1
        elif char in ")]": depth -= 1
        elif char == "," and depth == 0: items.append(value[start:index].strip()); start = index + 1
    items.append(value[start:].strip())
    return [item for item in items if item]


def cgr_inventory(build_file: str, source: str) -> list[dict[str, Any]]:
    source = strip_rust_comments(source)
    tokens = strip_rust_noncode(source)
    bindings: dict[str, list[str]] = {}
    for match in re.finditer(r"\blet\s+(?:mut\s+)?([A-Za-z_]\w*)\s*(?::[^=;]+)?=\s*([^;]+);", source):
        bindings.setdefault(match.group(1), []).append(match.group(2).strip())
    def binding(value: str) -> str:
        name = value.strip().lstrip("&").strip()
        required(name in bindings and len(bindings[name]) == 1, f"{build_file} has unresolved or shadowed binding {name}")
        required(not re.search(rf"\blet\s+mut\s+{re.escape(name)}\b", source), f"{build_file} has mutable persisted-CGR binding {name}")
        required(len(re.findall(rf"\b{re.escape(name)}(?:\s*:[^=;]+)?\s*=", source)) == 1, f"{build_file} mutates binding {name}")
        return bindings[name][0]
    def path(value: str) -> str:
        expr = binding(value)
        direct = re.fullmatch(r'Path::new\("([^"]+)"\)', expr)
        joined = re.fullmatch(r'Path::new\(&manifest_dir\)\.join\("([^"]+)"\)', expr)
        required(direct is not None or joined is not None, f"{build_file} has non-literal schema path binding")
        raw = (direct or joined).group(1)
        return os.path.normpath(str(Path(build_file).parent / raw)).replace("\\", "/")
    def paths(value: str) -> list[str]:
        expr = binding(value) if re.fullmatch(r"&?[A-Za-z_]\w*", value.strip()) else value.strip()
        required(expr.startswith("&[") and expr.endswith("]"), f"{build_file} has non-literal import roots")
        return [path(item) for item in split_top_level(expr[2:-1])]
    def schemas(value: str) -> list[str]:
        expr = binding(value) if re.fullmatch(r"&?[A-Za-z_]\w*", value.strip()) else value.strip()
        if expr.startswith("&"): expr = expr[1:]
        required(expr.startswith("[") and expr.endswith("]"), f"{build_file} has non-literal schema list")
        result = []
        for item in split_top_level(expr[1:-1]):
            found = re.fullmatch(r'"([^"]+)"', item.strip())
            required(found is not None, f"{build_file} has non-literal schema input")
            result.append(found.group(1))
        return result
    module_aliases = {"hyprstream_rpc_build"}
    function_aliases: set[str] = set()
    for match in re.finditer(r"\buse\s+hyprstream_rpc_build\s+as\s+([A-Za-z_]\w*)\s*;", tokens):
        module_aliases.add(match.group(1))
    for match in re.finditer(r"\buse\s+hyprstream_rpc_build\s*::\s*compile_schemas(?:\s+as\s+([A-Za-z_]\w*))?\s*;", tokens):
        function_aliases.add(match.group(1) or "compile_schemas")
    grouped_prefix = re.compile(r"\buse\s+hyprstream_rpc_build\s*::\s*\{")
    for group in grouped_prefix.finditer(tokens):
        start, index, depth = group.end(), group.end(), 1
        while index < len(tokens) and depth:
            depth += (tokens[index] == "{") - (tokens[index] == "}"); index += 1
        required(depth == 0, f"{build_file} has unterminated persisted-CGR grouped import")
        for match in re.finditer(r"(?:^|,)\s*compile_schemas(?:\s+as\s+([A-Za-z_]\w*))?\s*(?=,|$)", tokens[start:index - 1]):
            function_aliases.add(match.group(1) or "compile_schemas")
    for alias in module_aliases | function_aliases:
        if alias == "hyprstream_rpc_build":
            continue
        required(not re.search(rf"\blet\s+(?:mut\s+)?{re.escape(alias)}\b|\b{re.escape(alias)}\s*=", tokens),
                 f"{build_file} shadows or mutates persisted-CGR alias {alias}")

    call_patterns = [rf"\b{re.escape(alias)}\s*::\s*compile_schemas\s*\(" for alias in sorted(module_aliases)]
    call_patterns.extend(rf"\b{re.escape(alias)}\s*\(" for alias in sorted(function_aliases))
    recognized = re.compile("|".join(f"(?:{pattern})" for pattern in call_patterns))
    candidates = re.compile(r"\b(?:[A-Za-z_]\w*\s*::\s*)?compile_schemas\s*\(")
    calls, start, recognized_spans = [], 0, []
    while (match := recognized.search(tokens, start)) is not None:
        recognized_spans.append((match.start(), match.end()))
        found, index, depth = match.start(), match.end(), 1
        while index < len(source) and depth:
            depth += (source[index] == "(") - (source[index] == ")"); index += 1
        required(depth == 0, f"{build_file} has unterminated persisted-CGR call")
        args = split_top_level(source[match.end():index - 1])
        required(len(args) == 4, f"{build_file} persisted-CGR call has unexpected arguments")
        calls.append({"source_root": path(args[0]), "import_roots": paths(args[2]), "schemas": schemas(args[3])})
        start = index
    for candidate in candidates.finditer(tokens):
        required(any(begin == candidate.start() for begin, _ in recognized_spans),
                 f"{build_file} uses an unresolved persisted-CGR alias")
    return calls


def check_cgr(catalog: dict[str, Any], repo: Path, schemas: list[dict[str, Any]],
              mutations: dict[str, str] | None) -> dict[str, list[dict[str, Any]]]:
    roots = catalog.get("cgr_build_roots", {})
    required(roots == CGR_ROOTS, "persisted-CGR import roots drift")
    build_files = set(tracked(repo, "build.rs", "**/build.rs"))
    if mutations:
        build_files.update(path for path in mutations if path.endswith("/build.rs") or path == "build.rs")
    inventories = {path: cgr_inventory(path, text(repo, path, mutations)) for path in sorted(build_files)}
    producers = [path for path, calls in inventories.items() if calls]
    required(set(roots) == set(producers), "persisted-CGR producer inventory drift")
    for build_file, import_roots in roots.items():
        required(isinstance(import_roots, list) and import_roots, f"{build_file} lacks import roots")
        required(inventories[build_file] == EXPECTED_CGR_INVOCATIONS[build_file]["invocations"],
                 f"{build_file} persisted-CGR invocation inventory drift")
    for entry in schemas:
        producer = entry.get("cgr_producer")
        if producer is None:
            mode = entry.get("compiled_by")
            required(mode in {"capnp_only", "not_compiled"}, f"{entry['path']} must distinguish non-CGR compilation")
            if mode == "capnp_only":
                source = text(repo, "crates/hyprstream-rpc-build/build.rs", mutations)
                required("capnpc::CompilerCommand" in source and Path(entry["path"]).stem in source, f"{entry['path']} capnp-only claim drift")
            continue
        required(producer in roots, f"{entry['path']} producer is not an audited persisted-CGR root")
        matched = [call for call in inventories[producer] if entry["path"].startswith(f"{call['source_root']}/") and Path(entry["path"]).stem in call["schemas"]]
        required(len(matched) == 1,
                 f"{entry['path']} is absent from exact persisted-CGR inputs")
    return inventories


def schema_method_metadata(repo: Path, schemas: list[dict[str, Any]],
                           mutations: dict[str, str] | None) -> dict[str, list[str]]:
    def block(source: str, marker: str) -> str:
        start = source.find(marker)
        required(start >= 0, f"missing {marker}")
        start = source.find("{", start); depth = 1; index = start + 1
        while index < len(source) and depth:
            depth += (source[index] == "{") - (source[index] == "}"); index += 1
        required(depth == 0, f"unterminated {marker}")
        return source[start + 1:index - 1]
    def fields(source: str, struct: str) -> dict[str, str]:
        body = block(source, f"struct {struct}")
        union = block(body, "union")
        return {name: type_name for name, type_name in re.findall(r"(?m)^\s*(\w+)\s+@\d+\s*:\s*(\w+)", union)}
    hidden, streaming = [], []
    for entry in schemas:
        service = entry.get("service")
        if not service:
            continue
        source = text(repo, entry["path"], mutations)
        hidden.extend(
            f"{service}.{method}"
            for method in re.findall(
                r"(?ms)^\s*([A-Za-z][A-Za-z0-9_]*)\s+@\d+\s*:[^;]*?\$cliHidden\b[^;]*;", source
            )
        )
        pascal = "".join(part.capitalize() for part in service.split("-"))
        if f"struct {pascal}Request" not in source:
            continue
        request, response = fields(source, f"{pascal}Request"), fields(source, f"{pascal}Response")
        streaming.extend(f"{service}.{method}" for method in request if response.get(f"{method}Result") == "StreamInfo")
        def nested_streams(request_fields: dict[str, str], response_fields: dict[str, str], seen: set[tuple[str, str]]) -> list[str]:
            result: list[str] = []
            for scope, request_type in request_fields.items():
                response_type = response_fields.get(f"{scope}Result")
                if not (request_type.endswith("Request") and response_type and response_type.endswith("Response")) or (request_type, response_type) in seen:
                    continue
                seen.add((request_type, response_type))
                try:
                    nested_request, nested_response = fields(source, request_type), fields(source, response_type)
                except CatalogError:
                    continue
                result.extend(method for method in nested_request if nested_response.get(method) == "StreamInfo")
                result.extend(nested_streams(nested_request, nested_response, seen))
            return result
        streaming.extend(f"{service}.{method}" for method in nested_streams(request, response, set()))
    return {"cli_hidden": sorted(hidden), "streaming": sorted(streaming)}


def check_schema_catalog(catalog: dict[str, Any], repo: Path, schema_paths: list[str],
                         consumers: dict[str, dict[str, Any]], corpus: dict[str, Any], mutations: dict[str, str] | None,
                         event: str | None = None, revision: str | None = None) -> None:
    required(catalog.get("schema_version") == 1, "schema catalog must use schema_version 1")
    required(catalog.get("authority") == "docs/system-ontology.md", "system ontology must remain authoritative")
    check_provenance(catalog, repo, "schema catalog", corpus, mutations, event, revision)
    schemas = catalog.get("schemas")
    required(isinstance(schemas, list), "catalog.schemas must be a list")
    paths = [entry.get("path") for entry in schemas]
    required(len(paths) == len(set(paths)) and set(paths) == set(schema_paths), "catalog must exactly cover git-tracked .capnp files")
    ids = [entry.get("source_id") for entry in schemas]
    required(all(isinstance(value, str) and re.fullmatch(r"0x[0-9a-f]+", value) for value in ids), "each schema needs a source ID")
    required(len(ids) == len(set(ids)), "duplicate schema source identity")
    service_ids = [entry["service"] for entry in schemas if entry.get("service")]
    required(len(service_ids) == len(set(service_ids)), "duplicate schema service identity")
    by_service = {entry["service"]: entry for entry in schemas if entry.get("service")}
    owner_directories = check_owner_directories(catalog, repo)
    inventories = check_cgr(catalog, repo, schemas, mutations)
    capnp_build = text(repo, "crates/hyprstream-rpc-build/build.rs", mutations)
    for entry in schemas:
        for key in ("owner", "license", "kind", "exclusions"):
            required(bool(entry.get(key)), f"{entry['path']} lacks {key}")
        _, owner, license_id = owner_manifest(repo, entry["path"], owner_directories)
        required(entry["owner"] == owner, f"{entry['path']} owner differs from path-derived package")
        required(entry["license"] == license_id, f"{entry['path']} license differs from owner manifest")
        source = text(repo, entry["path"], mutations)
        found = re.search(r"^@(0x[0-9a-f]+);", source, re.MULTILINE)
        required(found is not None and found.group(1) == entry["source_id"], f"{entry['path']} source ID drift")
        stem = Path(entry["path"]).stem
        expected_service = stem if re.search(rf"(?m)^struct\s+{''.join(part.capitalize() for part in stem.split('_'))}Request\b", source) or re.search(r"(?m)^interface\s+", source) else None
        if expected_service is not None:
            required(entry["kind"] == "service" and entry.get("service") == expected_service,
                     f"{entry['path']} service identity/classification drift")
        elif entry["path"].endswith("wire_roundtrip_fixture.capnp"):
            required(entry["kind"] == "test-fixture" and entry.get("service") is None,
                     f"{entry['path']} fixture classification drift")
        elif "annotation " in source:
            required(entry["kind"] == "annotations" and entry.get("service") is None,
                     f"{entry['path']} annotation classification drift")
        elif re.search(r"\b9P\b|CompositorIpc", source):
            required(entry["kind"] == "protocol" and entry.get("service") is None,
                     f"{entry['path']} protocol classification drift")
        elif re.search(r"\b(?:ChatCoreIn|TypedEventEnvelope)\b", source):
            required(entry["kind"] == "type-only" and entry.get("service") is None,
                     f"{entry['path']} type-only classification drift")
        else:
            required(entry["kind"] == "shared-type" and entry.get("service") is None,
                     f"{entry['path']} shared-type classification drift")
        actual_cgr = [producer for producer, calls in inventories.items()
                      if any(entry["path"].startswith(f"{call['source_root']}/") and stem in call["schemas"] for call in calls)]
        if actual_cgr:
            required(entry.get("cgr_producer") == actual_cgr[0] and "compiled_by" not in entry,
                     f"{entry['path']} persisted-CGR compiler classification drift")
        elif entry["path"].endswith("wire_roundtrip_fixture.capnp"):
            required(entry.get("cgr_producer") is None and entry.get("compiled_by") == "capnp_only"
                     and re.search(r"capnpc::CompilerCommand::new\(\).*?\.file\(&schema\)", capnp_build, re.DOTALL) is not None,
                     f"{entry['path']} capnp-only compiler input drift")
        else:
            required(entry.get("cgr_producer") is None and entry.get("compiled_by") == "not_compiled",
                     f"{entry['path']} compiler classification drift")
        active = set(entry.get("surfaces", []))
        required(active <= set(SURFACES) | {"docs"}, f"{entry['path']} has unknown surface")
        for surface in SURFACES:
            required(surface in active or bool(entry["exclusions"].get(surface)), f"{entry['path']} lacks {surface} disposition")
            required(not (surface in active and surface in entry["exclusions"]),
                     f"{entry['path']} is both active and excluded on {surface}")
    required(catalog.get("method_metadata") == schema_method_metadata(repo, schemas, mutations),
             "schema hidden/streaming method metadata drift")

    declared = catalog.get("consumer_sets", {})
    required(set(declared) == set(SURFACES), "consumer sets must enumerate every surface")
    for surface in SURFACES:
        record, actual = declared[surface], consumers[surface]
        required(record.get("state") in {"active", "declared", "absent"}, f"invalid {surface} state")
        key = "tracked_sources" if surface == "typescript" else "services"
        required(record.get(key) == actual.get(key), f"{surface} source registration drift")
        expected_state = "active" if actual.get(key) else "absent"
        required(record.get("state") == expected_state, f"{surface} state does not match source registrations")
        if surface in {"cli", "mcp"}:
            required(record.get("method_policy") == actual.get("method_policy"), f"{surface} hidden/streaming policy drift")
        if surface == "cli":
            required(record.get("manual_services") == actual.get("manual_services"), "CLI manual registration drift")
        if surface == "factory":
            required(record.get("feature_conditions") == actual.get("feature_conditions"), "factory feature condition drift")
        if surface == "typescript":
            continue
        for service in record[key]:
            if service not in by_service:
                required(bool(record.get("non_schema_services", {}).get(service)), f"{surface} registers unclassified service {service}")
            else:
                required(surface in by_service[service].get("surfaces", []), f"{service} active on {surface} but omitted")
        for service, entry in by_service.items():
            if surface in entry.get("surfaces", []):
                required(service in record[key], f"{service} claims {surface} activity without source registration")


def check_corpus(corpus: dict[str, Any], repo: Path, mutations: dict[str, str] | None = None,
                 event: str | None = None, revision: str | None = None) -> None:
    required(corpus.get("schema_version") == 1 and corpus.get("authority") == "docs/system-ontology.md", "invalid corpus authority/version")
    check_provenance(corpus, repo, "corpus catalog", corpus, mutations, event, revision)
    for name in ("api_manifest", "corpus_manifest"):
        record = corpus.get(name, {})
        required(record.get("version") == 1 and isinstance(record.get("limit_bytes"), int) and record["limit_bytes"] > 0, f"invalid {name}")
        required(all(record.get(key) for key in ("path", "id", "integrity")), f"{name} lacks stable contract fields")
    required(corpus["api_manifest"] == {"version": 1, "path": "api/v1/{service}/{method}.json", "id": "api:{service}:{method}", "integrity": "sha256 of canonical UTF-8 JSON", "limit_bytes": 1048576},
             "API manifest contract drift")
    required(corpus["corpus_manifest"] == {"version": 1, "path": "corpus/v1/{document_id}.md", "id": "doc:{repository-relative-path}", "integrity": "sha256 of source bytes", "limit_bytes": 2097152},
             "corpus manifest contract drift")
    public, excluded = corpus.get("public_prose", []), corpus.get("excluded", [])
    required({item.get("glob") for item in public} == PUBLIC_GLOBS, "public prose allowlist widened or incomplete")
    for item in public:
        required(item.get("license") == "AGPL-3.0-only", f"{item.get('glob')} lacks exact SPDX license")
        required(isinstance(item.get("provenance"), str) and item["provenance"].startswith("tracked first-party repository"), "public prose provenance drift")
    required({item.get("glob"): item.get("reason") for item in excluded} == EXCLUSIONS, "corpus exclusions or reasons drift")
    package = corpus.get("package_contract", {})
    required(package.get("name") == "@hyprstream/docs" and package.get("state") == "declared-not-yet-published", "invalid docs package contract")
    required(package.get("requirements") == PACKAGE_REQUIREMENTS, "docs package requirements drift")
    required(package.get("exports") == {"./manifest": "./manifest.json", "./api/*": "./api/v1/*", "./corpus/*": "./corpus/v1/*"},
             "docs package exports drift")
    candidates = tracked(repo, "docs/*.md", "docs/**/*.md")
    for path in candidates:
        if not any(path_matches(path, item["glob"]) for item in excluded):
            required(any(path_matches(path, item["glob"]) for item in public), f"unallowlisted public prose: {path}")


def validate(repo: Path, catalog: dict[str, Any] | None = None, corpus: dict[str, Any] | None = None,
             schema_paths: list[str] | None = None, consumers: dict[str, dict[str, Any]] | None = None,
             mutations: dict[str, str] | None = None, event: str | None = None, revision: str | None = None) -> None:
    catalog = catalog or read_json(repo / "docs/schema-catalog.json")
    corpus = corpus or read_json(repo / "docs/corpus-sources.json")
    check_schema_catalog(
        catalog, repo, schema_paths or tracked(repo, "*.capnp"),
        consumers or source_services(repo, mutations), corpus, mutations, event, revision,
    )
    check_corpus(corpus, repo, mutations, event, revision)
    required("docs/system-ontology.md" in text(repo, "docs/contracts/docs-pipeline.md", None), "pipeline contract omits ontology authority")


def expect_failure(name: str, repo: Path, catalog: dict[str, Any], corpus: dict[str, Any],
                   schemas: list[str], consumers: dict[str, dict[str, Any]],
                   mutations: dict[str, str] | None = None, rebind_digest: bool = True) -> None:
    try:
        # Rebind the digest for source mutations so the intended extractor or
        # compiler assertion—not the outer provenance guard—must reject drift.
        trial_catalog, trial_corpus = copy.deepcopy(catalog), copy.deepcopy(corpus)
        if mutations and rebind_digest:
            digest = input_digest(repo, provenance_paths(repo, trial_corpus), mutations)
            trial_catalog["source_input_digest"] = digest
            trial_corpus["source_input_digest"] = digest
        validate(repo, trial_catalog, trial_corpus, schemas, consumers, mutations)
    except CatalogError:
        return
    raise AssertionError(f"mutation probe {name} unexpectedly passed")


def expect_cgr_failure(name: str, build_file: str, source: str) -> None:
    try:
        cgr_inventory(build_file, source)
    except CatalogError:
        return
    raise AssertionError(f"CGR mutation probe {name} unexpectedly passed")


def self_test(repo: Path) -> None:
    catalog, corpus = read_json(repo / "docs/schema-catalog.json"), read_json(repo / "docs/corpus-sources.json")
    schemas, consumers = tracked(repo, "*.capnp"), source_services(repo)
    validate(repo, catalog, corpus, schemas, consumers)
    # GitHub supplies the PR's immutable base SHA, which may be behind the
    # moving remote-main tip by the time the check runs.
    # A PR validates against its immutable remote-main base, not the catalog's
    # source snapshot: a source-plus-catalog PR legitimately pins a descendant
    # of that base that is not itself on main yet.
    validate(repo, catalog, corpus, schemas, consumers, event="pull_request",
             revision=git(repo, "merge-base", "HEAD", "refs/remotes/origin/main"))
    previous_head = os.environ.get("DOCS_CATALOG_AUDITED_HEAD")
    previous_event = os.environ.get("DOCS_CATALOG_EVENT")
    os.environ["DOCS_CATALOG_AUDITED_HEAD"] = "0" * 40
    os.environ["DOCS_CATALOG_EVENT"] = "pull_request"
    try:
        expect_failure("synthetic pull-request merge checkout", repo, catalog, corpus, schemas, consumers)
    finally:
        if previous_head is None: os.environ.pop("DOCS_CATALOG_AUDITED_HEAD", None)
        else: os.environ["DOCS_CATALOG_AUDITED_HEAD"] = previous_head
        if previous_event is None: os.environ.pop("DOCS_CATALOG_EVENT", None)
        else: os.environ["DOCS_CATALOG_EVENT"] = previous_event
    expect_failure("unlisted schema", repo, copy.deepcopy(catalog), corpus, schemas + ["new.capnp"], consumers)
    expect_failure("stale schema", repo, copy.deepcopy(catalog), corpus, schemas[1:], consumers)
    bad = copy.deepcopy(catalog); bad["source_commit"] = "not-a-git-revision"
    expect_failure("schema provenance", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["source_commit"] = "f" * 40
    expect_failure("fabricated provenance commit", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["source_commit"] = git(repo, "rev-parse", "HEAD"); bad["source_tree"] = git(repo, "rev-parse", "HEAD^{tree}")
    expect_failure("stale base provenance commit", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["source_tree"] = "0" * 40
    expect_failure("corpus provenance", repo, catalog, bad, schemas, consumers)
    stale = git(repo, "rev-parse", "HEAD")
    if stale == catalog["source_commit"]:
        stale = git(repo, "rev-parse", "HEAD~1")
    bad = copy.deepcopy(catalog)
    bad["source_commit"] = stale
    bad["source_tree"] = git(repo, "rev-parse", f"{stale}^{{tree}}")
    expect_failure("stale valid provenance", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(corpus)
    bad["source_commit"] = stale
    bad["source_tree"] = git(repo, "rev-parse", f"{stale}^{{tree}}")
    expect_failure("stale valid corpus provenance", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["schemas"][0]["license"] = "MIT"
    expect_failure("manifest license", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog)
    settlement = next(entry for entry in bad["schemas"] if entry["path"].endswith("settlement.capnp"))
    settlement["kind"], settlement["service"] = "type-only", None
    expect_failure("schema service reclassification", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog)
    model = next(entry for entry in bad["schemas"] if entry["path"].endswith("model.capnp"))
    model["cgr_producer"], model["compiled_by"] = None, "not_compiled"
    expect_failure("compiled schema not_compiled", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["cgr_build_roots"]["crates/hyprstream/build.rs"][0] = "made/up/schema"
    expect_failure("CGR import root", repo, bad, corpus, schemas, consumers)
    for name, path, before, after in [
        ("CLI registration", "crates/hyprstream/src/cli/schema_cli.rs", '"workflow",\n        &workflow_methods', '"settlement",\n        &workflow_methods'),
        ("MCP registration", "crates/hyprstream/src/services/mcp_service.rs", "tui_client::schema_metadata()", "mcp_client::schema_metadata()"),
        ("factory feature", "crates/hyprstream/src/services/factories.rs", '#[cfg(feature = "metrics")]', '#[cfg(feature = "other")]'),
        ("VFS registration", "crates/hyprstream-rpc-std/src/vfs_mount.rs", 'McpDispatch, "mcp"', 'McpDispatch, "oauth"'),
        ("CLI hidden policy", "crates/hyprstream/src/cli/schema_cli.rs", "if method.cli_hidden || method.is_streaming", "if method.cli_hidden"),
    ]:
        mutated = text(repo, path, None).replace(before, after)
        expect_failure(name, repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, {path: mutated}))
    cli_path = "crates/hyprstream/src/cli/schema_cli.rs"
    renamed = text(repo, cli_path, None).replace('Command::new("discovery")', 'Command::new("settlement")', 1)
    expect_failure("manual CLI rename", repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, {cli_path: renamed}))
    removed = text(repo, cli_path, None).replace("tool = tool.subcommand(discovery);", "// manual discovery registration removed", 1)
    expect_failure("manual CLI removal", repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, {cli_path: removed}))
    worker_path = "crates/hyprstream-workers/schema/worker.capnp"
    hidden_removed = text(repo, worker_path, None).replace("$cliHidden ", "", 1)
    expect_failure("schema hidden annotation", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {worker_path: hidden_removed})
    discovery_build = "crates/hyprstream-discovery/build.rs"
    discovery_source = text(repo, discovery_build, None)
    whitespace = discovery_source.replace("hyprstream_rpc_build::compile_schemas(", "hyprstream_rpc_build :: compile_schemas (", 1)
    required(cgr_inventory(discovery_build, whitespace) == EXPECTED_CGR_INVOCATIONS[discovery_build]["invocations"],
             "CGR whitespace normalization drift")
    aliased = ("use hyprstream_rpc_build::compile_schemas as compile;\n" + discovery_source
               .replace("hyprstream_rpc_build::compile_schemas(", "compile(", 1))
    required(cgr_inventory(discovery_build, aliased) == EXPECTED_CGR_INVOCATIONS[discovery_build]["invocations"],
             "CGR function-alias normalization drift")
    module_aliased = ("use hyprstream_rpc_build as rpc_build;\n" + discovery_source
                      .replace("hyprstream_rpc_build::compile_schemas(", "rpc_build::compile_schemas(", 1))
    required(cgr_inventory(discovery_build, module_aliased) == EXPECTED_CGR_INVOCATIONS[discovery_build]["invocations"],
             "CGR module-alias normalization drift")
    grouped_aliased = ("use hyprstream_rpc_build::{nested::{SchemaMetadata}, compile_schemas as compile, SchemaMetadata};\n" + discovery_source
                       .replace("hyprstream_rpc_build::compile_schemas(", "compile(", 1))
    required(cgr_inventory(discovery_build, grouped_aliased) == EXPECTED_CGR_INVOCATIONS[discovery_build]["invocations"],
             "CGR grouped function-alias normalization drift")
    lifetime_extra = discovery_source + "\nfn marker<'a>() {}\n" + discovery_source[discovery_source.find("hyprstream_rpc_build::compile_schemas("):]
    required(len(cgr_inventory(discovery_build, lifetime_extra)) == 2,
             "CGR lifetime tokenization hid an invocation")
    comment_extra = discovery_source + "\n/* // still block comment */\n" + discovery_source[discovery_source.find("hyprstream_rpc_build::compile_schemas("):]
    required(len(cgr_inventory(discovery_build, comment_extra)) == 2,
             "CGR block-comment tokenization hid an invocation")
    raw_string = discovery_source + '\nlet marker = br###"hyprstream_rpc_build::compile_schemas("###;\n'
    required(cgr_inventory(discovery_build, raw_string) == EXPECTED_CGR_INVOCATIONS[discovery_build]["invocations"],
             "CGR byte-raw string tokenization drift")
    expect_cgr_failure("CGR unresolved alias", discovery_build,
                       discovery_source.replace("hyprstream_rpc_build::compile_schemas(", "unknown::compile_schemas(", 1))
    expect_cgr_failure("CGR alias reassignment", discovery_build,
                       aliased + "\ncompile = other_compile;\n")
    expect_cgr_failure("CGR mutable binding", discovery_build,
                       discovery_source.replace("let rpc_schema_dir", "let mut rpc_schema_dir", 1))
    required(cgr_inventory("crates/unrelated/build.rs", 'fn main() { println!("compile_schemas"); }') == [],
             "CGR string-only producer false positive")
    root_changed = text(repo, discovery_build, None).replace("../hyprstream-rpc/schema", "../unrelated/schema", 1)
    expect_failure("CGR source import root", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {discovery_build: root_changed})
    fixture_build = "crates/hyprstream-rpc-build/build.rs"
    fixture_drift = text(repo, fixture_build, None).replace(".file(&schema)", '.file("tests/other.capnp")', 1)
    expect_failure("capnp-only compiler input", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {fixture_build: fixture_drift})
    shadowed = text(repo, discovery_build, None).replace(
        'let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");',
        'let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");\n    let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-workers/schema");',
    )
    expect_failure("CGR shadowed binding", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {discovery_build: shadowed})
    commented = text(repo, discovery_build, None).replace("../hyprstream-rpc/schema", "../hyprstream-workers/schema", 1) + '\n// let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");\n'
    expect_failure("CGR commented declaration", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {discovery_build: commented})
    extra = text(repo, discovery_build, None).replace("\n}", '\n    hyprstream_rpc_build::compile_schemas(schema_dir, Path::new(&out_dir), &[], &["extra"]);\n}', 1)
    expect_failure("CGR additional invocation", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {discovery_build: extra})
    extra_producer = {'crates/unrelated/build.rs': discovery_source}
    expect_failure("CGR additional producer", repo, copy.deepcopy(catalog), corpus, schemas, consumers, extra_producer)
    model_path = "crates/hyprstream-rpc-std/schema/model.capnp"
    model_drift = text(repo, model_path, None).replace("generateStream @1 :StreamInfo", "generateStream @1 :Text", 1)
    expect_failure("scoped model streaming response", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {model_path: model_drift})
    worker_drift = text(repo, worker_path, None).replace("attach @10 :StreamInfo", "attach @10 :Text", 1)
    expect_failure("scoped worker streaming response", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {worker_path: worker_drift})
    registry_path = "crates/hyprstream-rpc-std/schema/registry.capnp"
    registry_source = text(repo, registry_path, None)
    deep_stream = registry_source.replace(
        "struct WorktreeRequest {\n  name @0 :Text;\n  union {",
        "struct WorktreeRequest {\n  name @0 :Text;\n  union {\n    deepStream @99 :Void;",
        1,
    ).replace(
        "struct WorktreeResponse {\n  union {",
        "struct WorktreeResponse {\n  union {\n    deepStream @99 :StreamInfo;",
        1,
    )
    deep_metadata = schema_method_metadata(repo, catalog["schemas"], {registry_path: deep_stream})
    required("registry.deepStream" in deep_metadata["streaming"],
             "deeply scoped streaming extractor drift")
    expect_failure("deeply scoped streaming response", repo, copy.deepcopy(catalog), corpus, schemas,
                   consumers, {registry_path: deep_stream})
    expect_failure("TypeScript source", repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, tracked_sources=["package.json"]))
    bad = copy.deepcopy(catalog); bad.pop("owner_directories")
    expect_failure("owner directory inventory", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["schemas"][0]["owner"] = "hyprstream"
    expect_failure("path-derived owner", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["excluded"][0]["reason"] = ""
    expect_failure("exclusion reason", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["public_prose"][0]["glob"] = "docs/**/*.md"
    expect_failure("allowlist widening", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["public_prose"][0]["license"] = "MIT"
    expect_failure("public license", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["package_contract"]["requirements"] = []
    expect_failure("package requirements", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["api_manifest"]["integrity"] = "md5"
    expect_failure("API manifest contract", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["package_contract"]["exports"] = {"./corpus/*": "./dist/*"}
    expect_failure("package exports", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["consumer_sets"]["cli"]["state"] = "absent"
    expect_failure("consumer state contradiction", repo, bad, corpus, schemas, consumers)
    top_level = "docs/KV-CACHE-ARCHITECTURE.md"
    changed_top_level = text(repo, top_level, None) + "\nprovenance mutation\n"
    expect_failure("top-level corpus provenance", repo, catalog, corpus, schemas, consumers, {top_level: changed_top_level}, False)
    # Model a hosted push boundary with objects reachable from this checkout.
    push_commit = git(repo, "rev-parse", "HEAD~1")
    push_catalog, push_corpus = copy.deepcopy(catalog), copy.deepcopy(corpus)
    for record in (push_catalog, push_corpus):
        record["source_commit"] = push_commit
        record["source_tree"] = git(repo, "rev-parse", f"{push_commit}^{{tree}}")
    validate(repo, push_catalog, push_corpus, schemas, consumers, event="push", revision=push_commit)
    bad = copy.deepcopy(push_catalog); bad["source_commit"] = stale; bad["source_tree"] = git(repo, "rev-parse", f"{stale}^{{tree}}")
    try:
        validate(repo, bad, push_corpus, schemas, consumers, event="push", revision=push_commit)
    except CatalogError:
        pass
    else:
        raise AssertionError("mutation probe modeled main push provenance unexpectedly passed")
    print("docs catalog mutation probes: passed (33 expected failures plus modeled merge/squash/rebase main push)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        (self_test if args.self_test else validate)(args.repo.resolve())
        if not args.self_test:
            print("docs catalog: OK")
    except (CatalogError, AssertionError) as error:
        print(f"docs catalog: FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
