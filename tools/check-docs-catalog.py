#!/usr/bin/env python3
"""Fail-closed checks for schema coverage and publication contracts."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import stat
import tempfile
import sys
from pathlib import Path
from typing import Any, Callable

SURFACES = ("cli", "mcp", "factory", "vfs", "typescript")
CATALOG_SURFACES = (*SURFACES, "docs")
CLI_BUILDERS = ("build_service_command", "build_scoped_command_from_node")
CONSUMER_SOURCE_PATHS = {
    "cli": "crates/hyprstream/src/cli/schema_cli.rs",
    "mcp": "crates/hyprstream/src/services/mcp_service.rs",
    "factory": "crates/hyprstream/src/services/factories.rs",
    "vfs": "crates/hyprstream-rpc-std/src/vfs_mount.rs",
}
OWNER_DIRECTORIES = (
    "crates/hyprstream-pay/schema",
    "crates/hyprstream-rpc/schema",
    "crates/hyprstream-rpc-build/tests",
    "crates/hyprstream-rpc-std/schema",
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
    "crates/hyprstream/build.rs": ["crates/hyprstream-rpc/schema"],
    "crates/hyprstream-rpc/build.rs": ["crates/hyprstream-rpc/schema"],
    "crates/hyprstream-rpc-std/build.rs": ["crates/hyprstream-rpc-std/schema", "crates/hyprstream-rpc/schema"],
}
EXPECTED_CGR_INVOCATIONS = {
    "crates/hyprstream/build.rs": {"invocations": [{"source_root":"crates/hyprstream-rpc/schema","import_roots":["crates/hyprstream-rpc/schema"],"schemas":["streaming","nine"]}]},
    "crates/hyprstream-rpc/build.rs": {"invocations": [{"source_root":"crates/hyprstream-rpc/schema","import_roots":[],"schemas":["common","streaming","events","annotations","optional","nine"]}]},
    "crates/hyprstream-rpc-std/build.rs": {"invocations": [{"source_root":"crates/hyprstream-rpc-std/schema","import_roots":["crates/hyprstream-rpc/schema","crates/hyprstream-rpc-std/schema"],"schemas":["inference","model","registry","policy","mcp","metrics","service_events","chat_core","oauth","worker","workflow","discovery","tui","compositor_ipc"]}]},
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


def read_regular(repo: Path, path: str) -> bytes:
    """Read only regular files, without following a selected path's symlinks."""
    parts = path.split("/")
    required(parts and all(part not in {"", ".", ".."} for part in parts), f"invalid input path: {path}")
    descriptors = []
    try:
        directory = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(directory)
        for part in parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.append(directory)
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(descriptor, "rb") as source:
            required(stat.S_ISREG(os.fstat(source.fileno()).st_mode), f"selected input is not a regular file: {path}")
            return source.read()
    except OSError as error:
        raise CatalogError(f"cannot read regular input {path}: {error.strerror}") from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def read_json(path: Path, repo: Path | None = None) -> dict[str, Any]:
    root = repo or path.parent
    try:
        value = json.loads(read_regular(root, str(path.relative_to(root))))
        required(isinstance(value, dict), f"{path} must contain an object")
        return value
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CatalogError(f"{path}: {error}") from error


def tracked(repo: Path, *patterns: str) -> list[str]:
    output = git(repo, "ls-files", "--", *patterns)
    return [path for path in output.splitlines() if path]


def text(repo: Path, path: str, mutations: dict[str, str] | None) -> str:
    if mutations and path in mutations:
        return mutations[path]
    return read_regular(repo, path).decode("utf-8")


def unique_cli_bindings(cli: str, pattern: str) -> dict[str, tuple[str, int, tuple[int, ...]]]:
    """Supported subset: each referenced let name is unique in the file.

    Reject shadowing, even in an unused helper, instead of guessing Rust name
    resolution. Also retain declaration position/scope for registration checks.
    Count other let-pattern definitions too, not just recognized RHS forms.
    """
    bindings = {}
    declarations = list(re.finditer(r'\blet\s+([^;=]+)=', cli))
    for match in re.finditer(pattern, cli):
        name = match.group("binding")
        occurrences = sum(bool(re.search(rf'\b{re.escape(name)}\b', item.group(1))) for item in declarations)
        # Function/closure parameters are bindings too. Refuse these names
        # conservatively rather than treating an outer let as their authority.
        parameters = re.findall(r'\bfn\s+\w+[^{{;]*', cli)
        parameters += re.findall(r'\|([^|;{}]*)\|', cli)
        shadowed_parameter = any(re.search(rf'\b{re.escape(name)}\b', value)
                                 for value in parameters)
        required(name not in bindings and occurrences == 1 and not shadowed_parameter,
                 f"ambiguous CLI binding {name}")
        bindings[name] = (match.group("module"), match.start(), rust_lexical_scope(cli, match.start()))
    return bindings


def rust_lexical_scope(code: str, position: int) -> tuple[int, ...]:
    scopes = []
    for index, char in enumerate(code[:position]):
        if char == "{": scopes.append(index)
        elif char == "}":
            required(bool(scopes), "unbalanced CLI source scope")
            scopes.pop()
    return tuple(scopes)


def visible_cli_binding(cli: str, bindings: dict[str, tuple[str, int, tuple[int, ...]]],
                        name: str, position: int, service: str) -> str:
    required(name in bindings, f"CLI registration for {service} uses an unbound argument {name}")
    module, declaration, scope = bindings[name]
    call_scope = rust_lexical_scope(cli, position)
    required(declaration < position and call_scope[:len(scope)] == scope,
             f"CLI binding {name} is not visible at registration for {service}")
    return module


def mcp_registrations(source: str) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    """Inventory registration roots, excluding the recursive helper itself.

    The supported root grammar binds literal service/prefix arguments to a
    generated scoped_client_tree call. Unresolved root arguments fail closed.
    Keep both projections so a scoped change cannot hide behind a top-level
    registration of the same service.
    """
    code = strip_rust_noncode(source)
    definitions = list(re.finditer(r"\bfn register_schema_tools(?=\s*\()", code))
    required(len(definitions) == 1, "missing or ambiguous MCP registration root")
    definition = definitions[0].start()
    body = rust_fn_body(code[definition:], "register_schema_tools")
    start = code.find("{", definition) + 1
    raw = source[start:start + len(body)]
    roots: dict[str, list[dict[str, str]]] = {"top_level": [], "scoped": []}
    ordered = []
    for match in re.finditer(r"\bregister_top_level!\s*\(", body):
        args = re.match(r"\s*reg\s*,\s*([A-Za-z_][\w:]*)::schema_metadata\s*\(\s*\)\s*,?\s*\)", body[match.end():])
        required(args is not None, "unresolved MCP top-level registration")
        module = args.group(1)
        service = module.rsplit("::", 1)[-1].removesuffix("_client")
        roots["top_level"].append({"service": service, "module": module})
        ordered.append((match.start(), service))
    for match in re.finditer(r"\bregister_scoped_tools_recursive\s*\(", body):
        first = re.match(r"\s*reg\s*,\s*", body[match.end():])
        required(first is not None, "unresolved MCP scoped registration registry")
        service, end = rust_string(raw, match.end() + first.end(), "MCP scoped service")
        tree = re.match(r"\s*,\s*([A-Za-z_][\w:]*)::scoped_client_tree\s*\(\s*\)\s*,\s*", body[end:])
        required(tree is not None, f"MCP scoped registration for {service} lacks a generated tree binding")
        prefix, end = rust_literal(raw, end + tree.end(), "MCP scoped prefix")
        required(re.match(r"\s*,\s*&\[\s*\]\s*,?\s*\)", body[end:]) is not None,
                 f"MCP scoped registration for {service} has unresolved root scope arguments")
        roots["scoped"].append({"service": service, "module": tree.group(1), "prefix": prefix})
        ordered.append((match.start(), service))
    return list(dict.fromkeys(service for _, service in sorted(ordered))), roots


def rust_block_span(source: str, start: int, label: str) -> tuple[str, int]:
    """Return a brace-delimited code block and the index after its close."""
    opening = source.find("{", start)
    required(opening >= 0, f"{label} has no block")
    index, depth = opening, 1
    while depth:
        index += 1
        required(index < len(source), f"unterminated {label} block")
        depth += (source[index] == "{") - (source[index] == "}")
    return source[opening + 1:index], index + 1


def direct_matches(body: str, pattern: str) -> list[re.Match[str]]:
    """Matches at direct block scope; nested dead branches cannot satisfy policy."""
    matches = []
    for match in re.finditer(pattern, body):
        depth = 0
        for char in body[:match.start()]:
            depth += (char == "{") - (char == "}")
        if depth == 0:
            matches.append(match)
    return matches


def direct_call_only(body: str, pattern: str, label: str) -> bool:
    """Accept one direct call expression and no preceding/following branch control flow."""
    calls = direct_matches(body, pattern)
    if len(calls) != 1 or body[:calls[0].start()].strip():
        return False
    opening = body.find("(", calls[0].start())
    if opening < 0:
        return False
    index, depth = opening, 1
    while depth:
        index += 1
        if index >= len(body):
            return False
        depth += (body[index] == "(") - (body[index] == ")")
    tail = body[index + 1:]
    return tail.startswith(";") and not tail[1:].strip()


def mcp_loop_policy(body: str, label: str) -> tuple[bool, bool]:
    """Verify the effective hidden and streaming branches of one method loop."""
    loops = list(re.finditer(r"\bfor\s+method\s+in\s+methods\s*\{", body))
    required(len(loops) == 1, f"ambiguous MCP {label} method loop")
    loop, _ = rust_block_span(body, loops[0].start(), f"MCP {label} method loop")
    hidden = direct_matches(loop, r"\bif\s+method\.hidden\s*\{")
    required(len(hidden) == 1, f"MCP {label} hidden guard must be direct and unique")
    hidden_body, _ = rust_block_span(loop, hidden[0].start(), f"MCP {label} hidden guard")
    hidden_effective = hidden_body.strip() == "continue;"
    streaming = direct_matches(loop, r"\bif\s+method\.is_streaming\s*\{")
    required(len(streaming) == 1, f"MCP {label} streaming branch must be direct and unique")
    hidden_effective = hidden_effective and hidden[0].start() < streaming[0].start()
    streaming_body, end = rust_block_span(loop, streaming[0].start(), f"MCP {label} streaming branch")
    else_match = re.match(r"\s*else\s*\{", loop[end:])
    if else_match is None:
        return hidden_effective, False
    sync_body, _ = rust_block_span(loop, end + else_match.start(), f"MCP {label} synchronous branch")
    if label == "top-level":
        streaming_effective = (direct_call_only(streaming_body, r"\bregister_streaming_tool\s*\(", label)
                               and direct_call_only(sync_body, r"\bregister_sync_tool\s*\(", label))
    else:
        def scoped_registration(branch: str, flag: str) -> bool:
            registrations = direct_matches(branch, r"\breg\.register\s*\(\s*ToolEntry\s*\{")
            if not direct_call_only(branch, r"\breg\.register\s*\(", label) or len(registrations) != 1:
                return False
            entry, _ = rust_block_span(branch, registrations[0].start(), f"MCP {label} ToolEntry")
            return len(direct_matches(entry, rf"\bstreaming\s*:\s*{flag}\s*,")) == 1
        streaming_effective = scoped_registration(streaming_body, "true") and scoped_registration(sync_body, "false")
    return hidden_effective, streaming_effective


def mcp_method_policy(source: str) -> dict[str, str]:
    """Derive policy from the two runtime registration loops, not token counts."""
    code = strip_rust_noncode(source)
    roots = list(re.finditer(r"\bfn register_schema_tools(?=\s*\()", code))
    helpers = list(re.finditer(r"\bfn register_scoped_tools_recursive(?=\s*\()", code))
    required(len(roots) == 1, "missing or ambiguous MCP registration root")
    required(len(helpers) == 1, "missing or ambiguous MCP scoped registration helper")
    top, _ = rust_block_span(code, roots[0].start(), "MCP registration root")
    scoped, _ = rust_block_span(code, helpers[0].start(), "MCP scoped registration helper")
    top_hidden, top_streaming = mcp_loop_policy(top, "top-level")
    scoped_hidden, scoped_streaming = mcp_loop_policy(scoped, "scoped")
    return {
        "hidden": "excluded" if top_hidden and scoped_hidden else "unknown",
        "streaming": "included" if top_streaming and scoped_streaming else "unknown",
    }


def source_services(repo: Path, mutations: dict[str, str] | None = None) -> dict[str, dict[str, Any]]:
    cli_source = text(repo, "crates/hyprstream/src/cli/schema_cli.rs", mutations)
    mcp_source = text(repo, "crates/hyprstream/src/services/mcp_service.rs", mutations)
    factories_source = text(repo, "crates/hyprstream/src/services/factories.rs", mutations)
    vfs_source = text(repo, "crates/hyprstream-rpc-std/src/vfs_mount.rs", mutations)
    cli, mcp = strip_rust_noncode(cli_source), strip_rust_noncode(mcp_source)
    factories, vfs = strip_rust_noncode(factories_source), strip_rust_noncode(vfs_source)

    registrations: list[tuple[int, str, list[str] | None, dict[str, str] | None]] = []
    metadata_bindings = unique_cli_bindings(cli,
        r'\blet\s+(?P<binding>[a-zA-Z_]\w*)\s*=\s*extract_methods!\s*\(\s*'
        r'(?P<module>[A-Za-z_][\w:]*)::schema_metadata\s*\(\s*\)\s*\)\s*;')
    scoped_tree_bindings = unique_cli_bindings(cli,
        r'\blet\s+(?P<binding>[a-zA-Z_]\w*)\s*=\s*'
        r'(?P<module>[A-Za-z_][\w:]*)::scoped_client_tree\s*\(\s*\)\s*;')
    for match in re.finditer(r'\bbuild_service_command\s*\(\s*', cli):
        if re.match(r'"', cli_source[match.end():]) is None:
            continue
        service, end = rust_string(cli_source, match.end(), "CLI service registration")
        arguments = re.match(
            r'\s*,\s*&(?P<metadata>[a-zA-Z_]\w*)\s*,\s*(?P<tree>[a-zA-Z_]\w*)\s*,?\s*\)',
            cli[end:],
        )
        required(arguments is not None, f"CLI registration for {service} lacks metadata/tree bindings")
        metadata, tree = arguments.group("metadata"), arguments.group("tree")
        registrations.append((match.start(), service, None, {
            "metadata": visible_cli_binding(cli, metadata_bindings, metadata, match.start(), service),
            "scoped_tree": visible_cli_binding(cli, scoped_tree_bindings, tree, match.start(), service),
        }))
    for match in re.finditer(
        r'\blet\s+(?P<binding>[a-zA-Z_]\w*)\s*=\s*Command::new\s*\(\s*', cli
    ):
        if re.match(r'"', cli_source[match.end():]) is None:
            continue
        service, end = rust_string(cli_source, match.end(), "manual CLI registration")
        subcommand = re.search(
            rf'\btool\s*=\s*tool\.subcommand\(\s*{re.escape(match.group("binding"))}\s*\)\s*;', cli[end:]
        )
        if subcommand is None:
            continue
        body = cli[end:end + subcommand.start()]
        methods = [
            rust_string(cli_source, end + command.end(), "manual CLI method")[0]
            for command in re.finditer(r'\bCommand::new\s*\(\s*', body)
        ]
        registrations.append((match.start(), service, methods, None))
    registrations.sort()
    cli_services = [service for _, service, _, _ in registrations]
    guard = "if method.cli_hidden || method.is_streaming"
    cli_guarded = all(guard in rust_fn_body(cli, name) for name in CLI_BUILDERS)
    manual_services = {service: methods for _, service, methods, _ in registrations if methods is not None}
    cli_modules = {service: modules for _, service, _, modules in registrations if modules is not None}
    mcp_services, mcp_roots = mcp_registrations(mcp_source)
    factory_services, features, schema_attributes = [], {}, {}
    cfgs: list[tuple[int, int, str]] = []
    for match in re.finditer(r'#\s*\[\s*cfg\s*\(\s*feature\s*=\s*', factories):
        feature, end = rust_string(factories_source, match.end(), "factory feature condition")
        close = re.match(r'\s*\)\s*\]', factories[end:])
        if close is not None:
            cfgs.append((match.start(), end + close.end(), feature))
    for match in re.finditer(r'#\s*\[\s*service_factory\s*\(\s*', factories):
        depth, index = 1, match.end()
        while depth:
            required(index < len(factories), "unterminated service_factory attribute")
            depth += (factories[index] == "(") - (factories[index] == ")")
            index += 1
        attribute = factories[match.end():index - 1]
        name, _ = rust_string(factories_source, match.end(), "factory registration")
        factory_services.append(name)
        declared = re.search(r'\bschema\s*=\s*', attribute)
        schema_value = None
        if declared:
            schema_value, _ = rust_literal(factories_source, match.end() + declared.end(), "factory schema attribute")
        declared_meta = re.search(r'\bmetadata\s*=\s*', attribute)
        metadata_value = None
        if declared_meta:
            found = re.match(r'[A-Za-z_][\w:]*', attribute[declared_meta.end():])
            required(found is not None, f"factory {name} metadata attribute is not a module path")
            metadata_value = found.group(0)
        schema_attributes[name] = {"schema": schema_value, "metadata": metadata_value}
        prior = [feature for _, end, feature in cfgs
                 if end <= match.start() and factories[end:match.start()].strip() == ""]
        if prior:
            required(len(prior) == 1, f"ambiguous feature condition for factory {name}")
            features[name] = f"feature={prior[0]}"
    vfs_services, vfs_dispatches = [], {}
    for match in re.finditer(r'\bimpl_service_dispatch!\s*\(\s*([A-Za-z_]\w*)\s*,\s*', vfs):
        name, end = rust_string(vfs_source, match.end(), "VFS service registration")
        tail = re.match(r'\s*,\s*([A-Za-z_][\w:]*)\s*\)', vfs[end:])
        required(tail is not None, f"VFS registration for {name} lacks a generated module binding")
        vfs_dispatches[name] = {"dispatch": match.group(1), "module": tail.group(1)}
        vfs_services.append(name)
    ts_sources = typescript_schema_sources(repo, mutations)
    return {
        "cli": {
            "source": CONSUMER_SOURCE_PATHS["cli"],
            "services": cli_services,
            "manual_services": manual_services,
            "module_bindings": cli_modules,
            "method_policy": {
                "hidden": "excluded" if cli_guarded else "unknown",
                "streaming": "excluded" if cli_guarded else "unknown",
            },
        },
        "mcp": {
            "source": CONSUMER_SOURCE_PATHS["mcp"],
            "services": mcp_services,
            "registrations": mcp_roots,
            "method_policy": mcp_method_policy(mcp_source),
        },
        "factory": {"source": CONSUMER_SOURCE_PATHS["factory"], "services": factory_services, "feature_conditions": features, "schema_attributes": schema_attributes},
        "vfs": {"source": CONSUMER_SOURCE_PATHS["vfs"], "services": vfs_services, "dispatches": vfs_dispatches},
        "typescript": {"tracked_sources": ts_sources,
                       "schema_imports": typescript_schema_imports(repo, mutations)},
    }


def valid_object_id(value: Any, label: str) -> str:
    required(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value) is not None
             and value != "0" * 40, f"invalid {label}")
    return value


def audited_input(repo: Path, event: str | None = None, revision: str | None = None) -> tuple[str, str]:
    event = event or os.environ.get("DOCS_CATALOG_EVENT", "local")
    revision = revision or os.environ.get("DOCS_CATALOG_AUDITED_COMMIT")
    head = git(repo, "rev-parse", "HEAD")
    if event in {"pull_request", "push"}:
        expected_head = os.environ.get("DOCS_CATALOG_AUDITED_HEAD")
        required(expected_head in {None, head}, "checkout is not the workflow-supplied head")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            required(expected_head == head and revision is not None, "hosted event lacks exact head/base binding")
        if event == "pull_request":
            boundary = revision or git(repo, "merge-base", "HEAD", "refs/remotes/origin/main")
        else:
            boundary = revision or git(repo, "rev-parse", "HEAD^")
        valid_object_id(boundary, "audited boundary")
        git(repo, "cat-file", "-e", f"{boundary}^{{commit}}")
        if event == "pull_request":
            # The base branch can advance after the PR forks. Bind its exact
            # tree as the trust anchor without requiring it in head ancestry.
            git(repo, "merge-base", boundary, head)
        else:
            required(boundary != head, "push audited input must precede pushed HEAD")
            required(subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", boundary, head],
                                    capture_output=True).returncode == 0, "push audited input is not reachable from HEAD")
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


def deleted(path: str, mutations: dict[str, str] | None) -> bool:
    """A mutations entry of None marks the file as deleted in this probe."""
    return bool(mutations) and path in mutations and mutations[path] is None


def provenance_fixed_paths() -> set[str]:
    return {"crates/hyprstream/src/cli/schema_cli.rs", "crates/hyprstream/src/services/mcp_service.rs",
            "crates/hyprstream/src/services/factories.rs", "crates/hyprstream-rpc-std/src/vfs_mount.rs",
            ".github/license-boundary.toml", "tools/check-docs-catalog.py",
            *(str(Path(directory).parent / "Cargo.toml") for directory in OWNER_DIRECTORIES)}


def selected_input_paths(paths: list[str], read: Callable[[str], bytes], corpus: dict[str, Any]) -> list[str]:
    selected = set(provenance_fixed_paths())
    for path in paths:
        if path.endswith(".capnp") or path == "build.rs" or path.endswith("/build.rs"):
            selected.add(path)
        elif path.startswith("docs/") and path.endswith(".md") \
                and any(path_matches(path, item["glob"]) for item in corpus.get("public_prose", [])) \
                and not any(path_matches(path, item["glob"]) for item in corpus.get("excluded", [])):
            selected.add(path)
        elif path.endswith(TS_SOURCE_SUFFIXES) or is_package_manifest(path):
            if typescript_consumer(path, read(path).decode("utf-8")):
                selected.add(path)
    return sorted(selected)


def tree_reader(repo: Path, tree: str) -> tuple[list[str], Callable[[str], bytes]]:
    valid_object_id(tree, "source tree")
    entries = {}
    for entry in git(repo, "ls-tree", "-rz", "--full-tree", tree).split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            entries[path] = metadata.split()
    def read(path: str) -> bytes:
        required(path in entries, f"audited source tree omits {path}")
        mode, kind, oid = entries[path]
        required(mode in {"100644", "100755"} and kind == "blob", f"selected tree input is not a regular file: {path}")
        result = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", oid], capture_output=True)
        required(result.returncode == 0, f"cannot read audited blob {path}")
        return result.stdout
    return sorted(entries), read


def provenance_paths(repo: Path, corpus: dict[str, Any],
                     mutations: dict[str, str] | None = None) -> list[str]:
    read = lambda path: text(repo, path, mutations).encode("utf-8") if mutations else read_regular(repo, path)
    paths = [path for path in tracked(repo) if not deleted(path, mutations)]
    return [path for path in selected_input_paths(paths, read, corpus) if not deleted(path, mutations)]


def digest_inputs(paths: list[str], read: Callable[[str], bytes]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8") + b"\0" + read(path) + b"\0")
    return digest.hexdigest()


def input_digest(repo: Path, paths: list[str], mutations: dict[str, str] | None = None,
                 tree: str | None = None) -> str:
    if tree is not None:
        _, read = tree_reader(repo, tree)
    else:
        read = lambda path: text(repo, path, mutations).encode("utf-8") if mutations else read_regular(repo, path)
    return digest_inputs([path for path in paths if not deleted(path, mutations)], read)


def attested_tree_universe(repo: Path, tree: str, corpus: dict[str, Any]) -> set[str]:
    paths, read = tree_reader(repo, tree)
    return set(selected_input_paths(paths, read, corpus))


PROVENANCE_FIELDS = ("provenance_version", "authority", "source_commit", "source_tree", "source_input_digest")


def provenance_tuple(record: dict[str, Any]) -> tuple[Any, ...]:
    required(record.get("provenance_version") == 2, "unsupported provenance policy")
    required(record.get("authority") == "docs/system-ontology.md", "invalid provenance authority")
    valid_object_id(record.get("source_commit"), "source_commit")
    valid_object_id(record.get("source_tree"), "source_tree")
    required(isinstance(record.get("source_input_digest"), str)
             and re.fullmatch(r"[0-9a-f]{64}", record["source_input_digest"]) is not None,
             "invalid source_input_digest")
    return tuple(record[field] for field in PROVENANCE_FIELDS)


def object_exists(repo: Path, oid: str, kind: str) -> bool:
    # A wrong object type is contradictory evidence, not an unavailable object.
    result = subprocess.run(["git", "-C", str(repo), "cat-file", "-t", oid], capture_output=True, text=True)
    if result.returncode != 0:
        return False
    required(result.stdout.strip() == kind, f"declared {kind} has the wrong object type")
    return True


def inherited_provenance(repo: Path, boundary: str, record: dict[str, Any], paths: list[str]) -> bool:
    if not boundary:
        return False
    names, read = tree_reader(repo, git(repo, "rev-parse", f"{boundary}^{{tree}}"))
    manifests = ["docs/schema-catalog.json", "docs/corpus-sources.json"]
    if not all(path in names for path in manifests):
        return False
    try:
        prior_catalog, prior_corpus = [json.loads(read(path)) for path in manifests]
        if not all(isinstance(prior, dict) and tuple(prior.get(field) for field in PROVENANCE_FIELDS)
                   == provenance_tuple(record) for prior in (prior_catalog, prior_corpus)):
            return False
        check_publication_policy(prior_corpus)
        selected = selected_input_paths(names, read, prior_corpus)
        required(selected == paths, "trusted boundary input universe differs from inherited attestation")
        required(digest_inputs(selected, read) == record["source_input_digest"],
                 "trusted boundary inputs do not reproduce inherited digest")
        return True
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CatalogError(f"invalid trusted boundary manifest: {error}") from error


def recover_source_objects(repo: Path, commit: str) -> None:
    valid_object_id(commit, "recovery source_commit")
    origin = git(repo, "remote", "get-url", "origin")
    # The repository is fixed; a catalog can never choose a fetch URL/refspec.
    required(origin in {"https://github.com/hyprstream/hyprstream", "https://github.com/hyprstream/hyprstream.git"},
             "recovery origin is not the trusted repository")
    try:
        result = subprocess.run(["git", "-C", str(repo), "fetch", "--no-tags", "--no-write-fetch-head",
                                 "--no-recurse-submodules", "origin", commit], capture_output=True, text=True,
                                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, timeout=60)
    except subprocess.TimeoutExpired as error:
        raise CatalogError("source recovery timed out; restore the exact source object or re-attest current inputs") from error
    required(result.returncode == 0 and object_exists(repo, commit, "commit"),
             "source object unavailable from trusted origin; restore that exact object or re-attest current inputs")


def check_provenance(record: dict[str, Any], repo: Path, label: str, corpus: dict[str, Any],
                     mutations: dict[str, str] | None = None, event: str | None = None,
                     revision: str | None = None, recover: bool = False) -> None:
    provenance_tuple(record)
    commit, tree, declared_digest = (record[key] for key in ("source_commit", "source_tree", "source_input_digest"))
    topology, boundary = audited_input(repo, event, revision)
    staged_catalog = topology == "local" and subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached", "--quiet", "--", "docs/schema-catalog.json", "docs/corpus-sources.json"],
        capture_output=True).returncode != 0
    required(commit != git(repo, "rev-parse", "HEAD") or staged_catalog, f"{label} source_commit must not self-reference HEAD")
    required(tree != git(repo, "rev-parse", "HEAD^{tree}") or staged_catalog, f"{label} source_tree must not self-reference HEAD")
    paths = provenance_paths(repo, corpus, mutations)
    required(input_digest(repo, paths, mutations) == declared_digest, f"{label} current audited inputs differ from source_input_digest")

    def verify_available() -> bool:
        pair = object_exists(repo, commit, "commit")
        if pair:
            required(git(repo, "rev-parse", f"{commit}^{{tree}}") == tree, f"{label} source_tree does not match source_commit")
        tree_exists = object_exists(repo, tree, "tree")
        if tree_exists:
            selected = attested_tree_universe(repo, tree, corpus)
            required(not (selected - set(paths)), f"{label} declared tree attests inputs removed from the current checkout")
            # Source mutations are internal extractor-test inputs only. CLI
            # production validation never supplies this argument.
            if not mutations:
                required(selected == set(paths), f"{label} source tree input universe differs")
                required(input_digest(repo, paths, tree=tree) == declared_digest,
                         f"{label} source_tree does not reproduce the audited input digest")
        return pair and tree_exists

    if verify_available():
        return
    if not mutations and inherited_provenance(repo, boundary, record, paths):
        return
    required(recover and not mutations, f"{label} lacks verifiable source objects or inherited attestation; use --recover-source-objects")
    recover_source_objects(repo, commit)
    required(verify_available(), f"{label} recovered source objects are incomplete")


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
    def blank(value: str, literal: bool = False) -> str:
        fill = "\0" if literal else " "
        return "".join("\n" if char == "\n" else fill for char in value)
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
            value = source[index:end]; out.append(blank(value, literal=True) if mask_literals else value); index = end; continue
        char = source[index]
        is_char = char == "'" and (index + 2 < len(source)) and (source[index + 1] == "\\" or source[index + 2] == "'")
        if char == '"' or is_char:
            quote, end = char, index + 1
            while end < len(source):
                if source[end] == "\\": end += 2; continue
                end += 1
                if source[end - 1] == quote: break
            value = source[index:end]; out.append(blank(value, literal=True) if mask_literals else value); index = end; continue
        out.append(char); index += 1
    return "".join(out)


def strip_rust_comments(source: str) -> str:
    return rust_lex(source, False)


def strip_rust_noncode(source: str) -> str:
    return rust_lex(source, True)


def rust_literal(source: str, start: int, label: str) -> tuple[str, int]:
    """Read a normal Rust string after a token-validated prefix."""
    match = re.match(r'\s*("(?:\\.|[^"\\])*")', source[start:])
    required(match is not None, f"{label} must use a literal")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise CatalogError(f"{label} has invalid literal") from error
    return value, start + match.end()


def rust_string(source: str, start: int, label: str) -> tuple[str, int]:
    """Read a canonical service registration literal after a token-validated prefix."""
    value, end = rust_literal(source, start, label)
    required(re.fullmatch(r"[a-z0-9-]+", value) is not None,
             f"{label} has non-canonical service name")
    return value, end


def rust_fn_body(source: str, name: str) -> str:
    """Body of the top-level `fn <name>` in comment/literal-masked Rust source."""
    marker = source.find(f"fn {name}")
    required(marker >= 0, f"missing fn {name}")
    open_brace = source.find("{", marker)
    required(open_brace >= 0, f"fn {name} has no body")
    index, depth = open_brace, 1
    while depth:
        index += 1
        required(index < len(source), f"unterminated fn {name}")
        depth += (source[index] == "{") - (source[index] == "}")
    return source[open_brace + 1:index]


def strip_capnp_noncode(source: str, mask_literals: bool = True) -> str:
    """Offset-preserving Cap'n Proto comment lexer, optionally masking strings."""
    out, index = [], 0
    def blank(value: str) -> str:
        return "".join("\n" if char == "\n" else " " for char in value)
    while index < len(source):
        if source[index] == "#":
            end = source.find("\n", index)
            end = len(source) if end < 0 else end
            out.append(blank(source[index:end])); index = end; continue
        if source[index] == '"':
            end = index + 1
            while end < len(source):
                if source[end] == "\\": end += 2; continue
                end += 1
                if source[end - 1] == '"': break
            out.append(blank(source[index:end]) if mask_literals else source[index:end]); index = end; continue
        out.append(source[index]); index += 1
    return "".join(out)


def js_code_and_strings(source: str) -> tuple[str, dict[int, str]]:
    """Offset-preserving dependency lexer, including executable ${...} code.

    Template text stays inert. Balanced substitutions recursively lex code;
    only templates without substitutions are recorded as static specifiers.
    Comments, quoted braces and regex bodies cannot terminate substitutions.
    """
    out, strings, length = list(source), {}, len(source)
    expression_keywords = {"case", "delete", "do", "else", "in", "instanceof", "new", "return", "throw", "typeof", "void", "yield", "await"}

    def mask(start: int, end: int) -> None:
        out[start:end] = ["\n" if char == "\n" else " " for char in source[start:end]]

    def quoted(start: int) -> int:
        end, quote = start + 1, source[start]
        while end < length:
            if source[end] == "\\": end = min(end + 2, length); continue
            end += 1
            if source[end - 1] == quote: break
        closed = source[end - 1:end] == quote
        body_end = end - 1 if closed else end
        out[start + 1:body_end] = ["\0"] * (body_end - start - 1)
        if closed:
            strings[start + 1] = source[start + 1:body_end]
        return end

    def template(start: int) -> int:
        index, segment, substituted = start + 1, start, False
        while index < length:
            if source[index] == "\\": index = min(index + 2, length); continue
            if source[index] == "`":
                if not substituted:
                    return quoted(start)
                mask(segment, index + 1)
                return index + 1
            if source[index:index + 2] == "${":
                substituted = True
                mask(segment, index + 1)  # Keep the brace as an expression boundary.
                index = code(index + 2, substitution=True)
                segment = index
                continue
            index += 1
        mask(segment, length)
        return length

    def code(index: int, substitution: bool = False) -> int:
        can_start_regex, depth = True, 0
        while index < length:
            char, pair = source[index], source[index:index + 2]
            if pair in {"//", "/*"}:
                end = source.find("\n", index) if pair == "//" else source.find("*/", index + 2)
                end = length if end < 0 else end if pair == "//" else end + 2
                mask(index, end); index = end; continue
            if char == "/" and can_start_regex:
                end, character_class = index + 1, False
                while end < length:
                    current = source[end]
                    if current == "\\": end = min(end + 2, length); continue
                    if current == "[": character_class = True
                    elif current == "]": character_class = False
                    elif current == "/" and not character_class:
                        end += 1
                        while end < length and source[end].isalpha(): end += 1
                        break
                    elif current == "\n": break
                    end += 1
                mask(index, end); index = end; can_start_regex = False; continue
            if char in "'\"":
                index = quoted(index); can_start_regex = False; continue
            if char == "`":
                index = template(index); can_start_regex = False; continue
            if char == "}":
                if substitution and depth == 0:
                    return index + 1
                depth -= 1
            elif char == "{":
                depth += 1
            identifier = re.match(r"[A-Za-z_$][\w$]*", source[index:])
            if identifier:
                value = identifier.group(0)
                index += len(value); can_start_regex = value in expression_keywords; continue
            index += 1
            if char in ")]}.": can_start_regex = False
            elif not char.isspace(): can_start_regex = True
        return index

    code(0)
    return "".join(out), strings


TS_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
TS_PACKAGE_SECTIONS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")
TS_SPECIFIER_MARKER = re.compile(r"@hyprstream/docs|codegen-out|\.capnp$")
TS_DEPENDENCY = re.compile(r"(?:\brequire\s*\(\s*|\bimport\s*\(\s*|\bimport\s+|\bfrom\s+)(['\"`]\0+['\"`])")


def ts_dependency_specifiers(source: str) -> list[str]:
    code, strings = js_code_and_strings(source)
    specifiers = []
    for match in TS_DEPENDENCY.finditer(code):
        specifier = strings.get(match.start(1) + 1)
        if specifier is not None:
            specifiers.append(specifier)
    return specifiers


def ts_source_is_consumer(source: str) -> bool:
    return any(TS_SPECIFIER_MARKER.search(specifier) for specifier in ts_dependency_specifiers(source))


def is_package_manifest(path: str) -> bool:
    return Path(path).name == "package.json"


def typescript_consumer(path: str, source: str) -> bool:
    if is_package_manifest(path):
        try:
            package = json.loads(source)
        except json.JSONDecodeError as error:
            raise CatalogError("invalid package.json consumer input") from error
        required(isinstance(package, dict), "package.json must be an object")
        return any(isinstance(package.get(section), dict) and "@hyprstream/docs" in package[section]
                   for section in TS_PACKAGE_SECTIONS)
    return ts_source_is_consumer(source)


def typescript_schema_sources(repo: Path, mutations: dict[str, str] | None = None,
                              candidates: list[str] | None = None) -> list[str]:
    candidates = candidates if candidates is not None else tracked(repo, *("*" + suffix for suffix in TS_SOURCE_SUFFIXES), "package.json", "**/package.json")
    return sorted(path for path in candidates if not deleted(path, mutations)
                  and typescript_consumer(path, text(repo, path, mutations)))


def typescript_schema_imports(repo: Path, mutations: dict[str, str] | None = None) -> list[str]:
    """Resolve concrete Cap'n Proto imports to one tracked catalog source each."""
    schema_paths = tracked(repo, "*.capnp")
    by_name: dict[str, list[str]] = {}
    for path in schema_paths:
        by_name.setdefault(Path(path).name, []).append(path)
    resolved = []
    for path in typescript_schema_sources(repo, mutations):
        if is_package_manifest(path):
            continue
        for specifier in ts_dependency_specifiers(text(repo, path, mutations)):
            if not specifier.endswith(".capnp"):
                continue
            candidates = by_name.get(Path(specifier).name, [])
            required(len(candidates) == 1,
                     f"TypeScript consumer {path} has unresolved or ambiguous schema import {specifier}")
            resolved.append(candidates[0])
    return sorted(set(resolved))


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
    for match in re.finditer(r"\blet\s+(?:mut\s+)?([A-Za-z_]\w*)\s*(?::[^=;]+)?=\s*([^;]+);", tokens):
        bindings.setdefault(match.group(1), []).append(source[match.start(2):match.end(2)].strip())
    def binding(value: str) -> str:
        name = value.strip().lstrip("&").strip()
        required(name in bindings and len(bindings[name]) == 1, f"{build_file} has unresolved or shadowed binding {name}")
        required(not re.search(rf"\blet\s+mut\s+{re.escape(name)}\b", tokens), f"{build_file} has mutable persisted-CGR binding {name}")
        required(len(re.findall(rf"\b{re.escape(name)}(?:\s*:[^=;]+)?\s*=", tokens)) == 1, f"{build_file} mutates binding {name}")
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


def capnp_only_inputs(build_file: str, source: str) -> list[str]:
    """Resolve literal capnpc::CompilerCommand inputs without trusting comments/literals."""
    code = strip_rust_noncode(source)
    build_dir = str(Path(build_file).parent)
    # Bindings are (position, kind, value); kind is "format" (a template with an
    # optional {manifest} placeholder), "path" (a literal relative to this build
    # script), or "join" ("base\0literal" resolved against prior bindings).
    bindings: dict[str, list[tuple[int, str, str]]] = {}
    def literal(at: int, label: str) -> str | None:
        try:
            value, _ = rust_literal(source, at, label)
            return value
        except CatalogError:
            return None
    for match in re.finditer(r'\blet\s+([A-Za-z_]\w*)\s*=\s*', code):
        rest = code[match.end():]
        prefix = re.match(r'format!\s*\(\s*', rest)
        if prefix is not None:
            template = literal(match.end() + prefix.end(), "capnp-only compiler input")
            if template is not None:
                bindings.setdefault(match.group(1), []).append((match.start(), "format", template))
            continue
        path_new = re.match(r'(?:std::path::)?Path::new\s*\(\s*', rest)
        if path_new is not None:
            after = match.end() + path_new.end()
            try:
                value, end = rust_literal(source, after, "capnp-only path binding")
            except CatalogError:
                value = None
            if value is not None:
                tail = re.match(r'\s*\)\s*\.\s*join\s*\(\s*', source[end:])
                if tail is not None:
                    joined = literal(end + tail.end(), "capnp-only join input")
                    if joined is not None:
                        value = value + "/" + joined
                bindings.setdefault(match.group(1), []).append((match.start(), "path", value))
            continue
        join = re.match(r'&?([A-Za-z_]\w*)\s*\.\s*join\s*\(\s*', rest)
        if join is not None:
            value = literal(match.end() + join.end(), "capnp-only join input")
            if value is not None:
                bindings.setdefault(match.group(1), []).append((match.start(), "join", join.group(1) + "\0" + value))
    def resolve(entry: tuple[int, str, str], seen: frozenset[str] = frozenset()) -> str | None:
        position, kind, value = entry
        if kind == "format":
            return "format:" + value
        if kind == "path":
            return value
        base_name, joined = value.split("\0", 1)
        bases = [candidate for candidate in bindings.get(base_name, []) if candidate[0] < position]
        if not bases or base_name in seen:
            return None
        base = resolve(bases[-1], seen | {base_name})
        if base is None or base.startswith("format:"):
            return None
        return base + "/" + joined
    # Resolve `use capnpc::CompilerCommand` (optionally aliased, plain or grouped)
    # exactly as the persisted-CGR extractor resolves compile_schemas imports.
    constructors: set[str] = set()
    for match in re.finditer(r"\buse\s+capnpc\s*::\s*CompilerCommand(?:\s+as\s+([A-Za-z_]\w*))?\s*;", code):
        constructors.add(match.group(1) or "CompilerCommand")
    for group in re.finditer(r"\buse\s+capnpc\s*::\s*\{", code):
        start = index = group.end(); depth = 1
        while index < len(code) and depth:
            depth += (code[index] == "{") - (code[index] == "}"); index += 1
        required(depth == 0, f"{build_file} has unterminated capnpc grouped import")
        for found in re.finditer(r"(?:^|[,{]\s*)CompilerCommand(?:\s+as\s+([A-Za-z_]\w*))?\s*(?=,|}|$)", code[start:index - 1]):
            constructors.add(found.group(1) or "CompilerCommand")
    for name in constructors:
        required(not re.search(rf"\blet\s+(?:mut\s+)?{re.escape(name)}\b|\b{re.escape(name)}\s*=", code),
                 f"{build_file} shadows the imported CompilerCommand alias {name}")
    module_aliases: set[str] = set()
    for match in re.finditer(r"\buse\s+capnpc\s+as\s+([A-Za-z_]\w*)\s*;", code):
        module_aliases.add(match.group(1))
    for alias in module_aliases:
        required(not re.search(rf"\blet\s+(?:mut\s+)?{re.escape(alias)}\b|\b{re.escape(alias)}\s*=", code),
                 f"{build_file} shadows the capnpc module alias {alias}")
        foreign = [path for path in re.findall(rf"\buse\s+([A-Za-z_][\w:]*)\s+as\s+{re.escape(alias)}\s*;", code) if path != "capnpc"]
        required(not foreign, f"{build_file} has an ambiguous module alias {alias}")
    patterns = [r"capnpc\s*::\s*CompilerCommand\s*::\s*(?:new|default)\s*\(\s*\)"]
    patterns += [rf"(?<![:\w]){re.escape(name)}\s*::\s*(?:new|default)\s*\(\s*\)" for name in sorted(constructors)]
    patterns += [rf"(?<![:\w]){re.escape(alias)}\s*::\s*CompilerCommand\s*::\s*(?:new|default)\s*\(\s*\)" for alias in sorted(module_aliases)]
    recognized = re.compile("|".join(f"(?:{pattern})" for pattern in patterns))
    constructor = re.compile(r"\b(?:[A-Za-z_]\w*\s*::\s*)?CompilerCommand\s*::\s*(?:new|default)\s*\(")
    inputs: list[str] = []
    statement_starts: set[int] = set()
    for match in recognized.finditer(code):
        statement_starts.add(match.start())
        end = code.find(";", match.end())
        required(end >= 0, f"{build_file} has unterminated capnp-only compiler command")
        calls = list(re.finditer(r'\.file\s*\(\s*', code[match.start():end]))
        required(calls, f"{build_file} capnp-only compiler command lacks an input")
        for call in calls:
            at = match.start() + call.end()
            argument = re.match(r'&([A-Za-z_]\w*)\s*[,)]', source[at:])
            if argument is not None:
                position = match.start() + call.start()
                choices = [entry for entry in bindings.get(argument.group(1), []) if entry[0] < position]
                resolved = resolve(choices[-1]) if choices else None
                required(resolved is not None,
                         f"{build_file} capnp-only compiler input is unresolved")
                raw = resolved[7:].replace("{manifest}", build_dir) if resolved.startswith("format:") else build_dir + "/" + resolved
            else:
                try:
                    raw, end = rust_literal(source, at, "capnp-only compiler input")
                except CatalogError as error:
                    raise CatalogError(f"{build_file} capnp-only compiler input is unparseable") from error
                required(re.match(r'\s*[,)]', source[end:]) is not None,
                         f"{build_file} capnp-only compiler input is unparseable")
            inputs.append(os.path.normpath(raw.replace("{manifest}", build_dir)).replace("\\", "/"))
    for stray in constructor.finditer(code):
        required(stray.start() in statement_starts,
                 f"{build_file} has an unrecognized or ambiguous CompilerCommand constructor")
    return inputs


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
    direct_inputs = sorted({path for build_file in sorted(build_files)
                            for path in capnp_only_inputs(build_file, text(repo, build_file, mutations))})
    cgr_claimed = {entry["path"] for entry in schemas if entry.get("cgr_producer")}
    capnp_only = [path for path in direct_inputs if path not in cgr_claimed]
    expected_capnp_only = sorted(entry["path"] for entry in schemas if entry.get("compiled_by") == "capnp_only")
    required(sorted(capnp_only) == expected_capnp_only,
             "capnp-only compiler input inventory drift")
    for entry in schemas:
        producer = entry.get("cgr_producer")
        if producer is None:
            mode = entry.get("compiled_by")
            required(mode in {"capnp_only", "not_compiled"}, f"{entry['path']} must distinguish non-CGR compilation")
            if mode == "capnp_only":
                required(entry["path"] in direct_inputs, f"{entry['path']} capnp-only claim drift")
            else:
                required(entry["path"] not in direct_inputs,
                         f"{entry['path']} is compiled directly by a build script")
            required(entry.get("cgr_producers") is None, f"{entry['path']} must not claim CGR producers")
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
        return {name: type_name for name, type_name in re.findall(r"(?m)^\s*(\w+)\s+@\d+\s*:\s*(\w+(?:\.\w+)*)", union)}
    def hidden_fields(source: str, struct: str) -> list[str]:
        union = block(block(source, f"struct {struct}"), "union")
        return [name for name, annotations in re.findall(
            r"(?ms)^\s*(\w+)\s+@\d+\s*:\s*\w+(?:\.\w+)*"
            r"((?:(?!^\s*[A-Za-z][A-Za-z0-9_]*\s+@\d+)[^;])*);", union)
                if re.search(r"\$cliHidden\b", annotations)]
    hidden, streaming = [], []
    for entry in schemas:
        service = entry.get("service")
        if not service:
            continue
        raw_source = text(repo, entry["path"], mutations)
        source = strip_capnp_noncode(raw_source)
        import_source = strip_capnp_noncode(raw_source, mask_literals=False)
        stream_types = {"StreamInfo"}
        # Locate declarations with strings/comments masked, then read the path
        # from the same offsets with only comments masked. Valid comment gaps
        # before the path or semicolon must behave exactly like whitespace.
        for match in re.finditer(r"\busing\s+(\w+)\s*=\s*import\b", source):
            imported = re.match(r'\s*"(/?streaming\.capnp)"\s*;', import_source[match.end():])
            if imported:
                stream_types.add(f"{match.group(1)}.StreamInfo")
        pascal = "".join(part.capitalize() for part in service.split("-"))
        if f"struct {pascal}Request" not in source:
            continue
        request, response = fields(source, f"{pascal}Request"), fields(source, f"{pascal}Response")
        hidden.extend(f"{service}.{method}" for method in hidden_fields(source, f"{pascal}Request"))
        streaming.extend(f"{service}.{method}" for method in request if response.get(f"{method}Result") in stream_types)
        def nested_streams(request_fields: dict[str, str], response_fields: dict[str, str], seen: set[tuple[str, str]], prefix: str) -> list[str]:
            result: list[str] = []
            for scope, request_type in request_fields.items():
                response_type = response_fields.get(f"{scope}Result")
                if not (request_type.endswith("Request") and response_type and response_type.endswith("Response")) or (request_type, response_type) in seen:
                    continue
                path_seen = seen | {(request_type, response_type)}
                try:
                    nested_request, nested_response = fields(source, request_type), fields(source, response_type)
                except CatalogError:
                    continue
                hidden.extend(f"{prefix}.{scope}.{method}" for method in hidden_fields(source, request_type))
                result.extend(f"{scope}.{method}" for method in nested_request if nested_response.get(method) in stream_types)
                result.extend(f"{scope}.{method}" for method in nested_streams(nested_request, nested_response, path_seen, f"{prefix}.{scope}"))
            return result
        streaming.extend(f"{service}.{method}" for method in nested_streams(request, response, set(), service))
    return {"cli_hidden": sorted(hidden), "streaming": sorted(streaming)}


def check_schema_catalog(catalog: dict[str, Any], repo: Path, schema_paths: list[str],
                         consumers: dict[str, dict[str, Any]], corpus: dict[str, Any], mutations: dict[str, str] | None,
                         event: str | None = None, revision: str | None = None, recover: bool = False) -> None:
    required(catalog.get("schema_version") == 1, "schema catalog must use schema_version 1")
    required(catalog.get("authority") == "docs/system-ontology.md", "system ontology must remain authoritative")
    check_provenance(catalog, repo, "schema catalog", corpus, mutations, event, revision, recover)
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
    for entry in schemas:
        for key in ("owner", "license", "kind"):
            required(bool(entry.get(key)), f"{entry['path']} lacks {key}")
        required(isinstance(entry.get("exclusions"), dict), f"{entry['path']} exclusions must be an object")
        _, owner, license_id = owner_manifest(repo, entry["path"], owner_directories)
        required(entry["owner"] == owner, f"{entry['path']} owner differs from path-derived package")
        required(entry["license"] == license_id, f"{entry['path']} license differs from owner manifest")
        source = text(repo, entry["path"], mutations)
        code = strip_capnp_noncode(source)
        found = re.search(r"^@(0x[0-9a-f]+);", code, re.MULTILINE)
        required(found is not None and found.group(1) == entry["source_id"], f"{entry['path']} source ID drift")
        stem = Path(entry["path"]).stem
        expected_service = stem if re.search(rf"(?m)^struct\s+{''.join(part.capitalize() for part in stem.split('_'))}Request\b", code) or re.search(r"(?m)^interface\s+", code) else None
        if expected_service is not None:
            required(entry["kind"] == "service" and entry.get("service") == expected_service,
                     f"{entry['path']} service identity/classification drift")
        elif entry["path"].endswith("wire_roundtrip_fixture.capnp"):
            required(entry["kind"] == "test-fixture" and entry.get("service") is None,
                     f"{entry['path']} fixture classification drift")
        elif "annotation " in code:
            required(entry["kind"] == "annotations" and entry.get("service") is None,
                     f"{entry['path']} annotation classification drift")
        elif (re.search(r"(?m)^struct\s+NpRequest\b", code)
              and re.search(r"(?m)^struct\s+NpResponse\b", code)) or re.search(r"\bCompositorIpc", code):
            required(entry["kind"] == "protocol" and entry.get("service") is None,
                     f"{entry['path']} protocol classification drift")
        elif re.search(r"\b(?:ChatCoreIn|TypedEventEnvelope)\b", code):
            required(entry["kind"] == "type-only" and entry.get("service") is None,
                     f"{entry['path']} type-only classification drift")
        else:
            required(entry["kind"] == "shared-type" and entry.get("service") is None,
                     f"{entry['path']} shared-type classification drift")
        actual_cgr = [producer for producer, calls in inventories.items()
                      if any(entry["path"].startswith(f"{call['source_root']}/") and stem in call["schemas"] for call in calls)]
        if actual_cgr:
            required(entry.get("cgr_producer") == sorted(actual_cgr)[0] and "compiled_by" not in entry,
                     f"{entry['path']} persisted-CGR compiler classification drift")
            required(entry.get("cgr_producers") == sorted(actual_cgr),
                     f"{entry['path']} CGR producer inventory drift")
        elif entry["path"].endswith("wire_roundtrip_fixture.capnp"):
            required(entry.get("cgr_producer") is None and entry.get("compiled_by") == "capnp_only",
                     f"{entry['path']} capnp-only compiler input drift")
        else:
            required(entry.get("cgr_producer") is None and entry.get("compiled_by") == "not_compiled",
                     f"{entry['path']} compiler classification drift")
        declared_surfaces = entry.get("surfaces", [])
        required(isinstance(declared_surfaces, list) and len(declared_surfaces) == len(set(declared_surfaces)),
                 f"{entry['path']} has invalid surface inventory")
        active = set(declared_surfaces)
        required(active <= set(CATALOG_SURFACES), f"{entry['path']} has unknown surface")
        for surface in CATALOG_SURFACES:
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
        if surface != "typescript":
            required(record.get("source") == actual.get("source") == CONSUMER_SOURCE_PATHS[surface],
                     f"{surface} declared source path drift")
        key = "tracked_sources" if surface == "typescript" else "services"
        required(record.get(key) == actual.get(key), f"{surface} source registration drift")
        expected_state = "active" if actual.get(key) else "absent"
        required(record.get("state") == expected_state, f"{surface} state does not match source registrations")
        if surface in {"cli", "mcp"}:
            required(record.get("method_policy") == actual.get("method_policy"), f"{surface} hidden/streaming policy drift")
        if surface == "cli":
            required(record.get("manual_services") == actual.get("manual_services"), "CLI manual registration drift")
            required(record.get("module_bindings") == actual.get("module_bindings"), "CLI metadata/scoped-tree module binding drift")
        if surface == "mcp":
            required(record.get("registrations") == actual.get("registrations"), "MCP top-level/scoped registration binding drift")
        if surface == "factory":
            required(record.get("feature_conditions") == actual.get("feature_conditions"), "factory feature condition drift")
            required(record.get("schema_attributes") == actual.get("schema_attributes"), "factory schema/metadata attribute drift")
        if surface == "vfs":
            required(record.get("dispatches") == actual.get("dispatches"), "VFS generated-module binding drift")
        if surface == "typescript":
            required(record.get("schema_imports") == actual.get("schema_imports"),
                     "TypeScript concrete schema import inventory drift")
            by_path = {entry["path"]: entry for entry in schemas}
            for path in actual["schema_imports"]:
                required(path in by_path and "typescript" in by_path[path].get("surfaces", []),
                         f"{path} imported by TypeScript but excluded from TypeScript surface")
            continue
        for service in record[key]:
            if service not in by_service:
                required(bool(record.get("non_schema_services", {}).get(service)), f"{surface} registers unclassified service {service}")
            else:
                required(surface in by_service[service].get("surfaces", []), f"{service} active on {surface} but omitted")
        for service, entry in by_service.items():
            if surface in entry.get("surfaces", []):
                required(service in record[key], f"{service} claims {surface} activity without source registration")


def check_publication_policy(corpus: dict[str, Any]) -> None:
    public, excluded = corpus.get("public_prose", []), corpus.get("excluded", [])
    required({item.get("glob") for item in public} == PUBLIC_GLOBS, "public prose allowlist widened or incomplete")
    for item in public:
        required(item.get("license") == "AGPL-3.0-only", f"{item.get('glob')} lacks exact SPDX license")
        required(isinstance(item.get("provenance"), str) and item["provenance"].startswith("tracked first-party repository"), "public prose provenance drift")
    required(all(isinstance(item, dict) and isinstance(item.get("glob"), str)
                 and isinstance(item.get("reason"), str) and item["reason"].strip() for item in excluded),
             "every exclusion requires a nonempty reason")
    required(len({item["glob"] for item in excluded}) == len(excluded), "duplicate exclusion glob")
    required({item.get("glob"): item.get("reason") for item in excluded} == EXCLUSIONS, "corpus exclusions or reasons drift")


def check_corpus(corpus: dict[str, Any], repo: Path, mutations: dict[str, str] | None = None,
                 event: str | None = None, revision: str | None = None, recover: bool = False) -> None:
    required(corpus.get("schema_version") == 1 and corpus.get("authority") == "docs/system-ontology.md", "invalid corpus authority/version")
    check_provenance(corpus, repo, "corpus catalog", corpus, mutations, event, revision, recover)
    for name in ("api_manifest", "corpus_manifest"):
        record = corpus.get(name, {})
        required(record.get("version") == 1 and isinstance(record.get("limit_bytes"), int) and record["limit_bytes"] > 0, f"invalid {name}")
        required(all(record.get(key) for key in ("path", "id", "integrity")), f"{name} lacks stable contract fields")
    required(corpus["api_manifest"] == {"version": 1, "path": "api/v1/{service}/{scope}/{method}.json", "id": "api:{service}:{scope}:{method}", "integrity": "sha256 of canonical UTF-8 JSON", "limit_bytes": 1048576},
             "API manifest contract drift")
    required(corpus["corpus_manifest"] == {"version": 1, "path": "corpus/v1/{document_id}.md", "id": "doc:{repository-relative-path}", "integrity": "sha256 of source bytes", "limit_bytes": 2097152},
             "corpus manifest contract drift")
    check_publication_policy(corpus)
    public, excluded = corpus["public_prose"], corpus["excluded"]
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
             mutations: dict[str, str] | None = None, event: str | None = None, revision: str | None = None,
             recover: bool = False) -> None:
    catalog = catalog or read_json(repo / "docs/schema-catalog.json", repo)
    corpus = corpus or read_json(repo / "docs/corpus-sources.json", repo)
    required(provenance_tuple(catalog) == provenance_tuple(corpus), "catalog/corpus provenance tuples differ")
    derived_consumers = source_services(repo, mutations)
    if consumers is not None:
        required(consumers == derived_consumers, "consumer inventory is not derived from current source state")
    check_schema_catalog(
        catalog, repo, schema_paths or tracked(repo, "*.capnp"),
        derived_consumers, corpus, mutations, event, revision, recover,
    )
    check_corpus(corpus, repo, mutations, event, revision, recover)
    required("docs/system-ontology.md" in text(repo, "docs/contracts/docs-pipeline.md", None), "pipeline contract omits ontology authority")


def replace_nth(source: str, old: str, new: str, occurrence: int) -> str:
    index = -1
    for _ in range(occurrence):
        index = source.find(old, index + 1)
        required(index >= 0, f"mutation probe needle {old!r} occurrence {occurrence} is missing")
    return source[:index] + new + source[index + len(old):]


def expect_failure(name: str, repo: Path, catalog: dict[str, Any], corpus: dict[str, Any],
                   schemas: list[str], consumers: dict[str, dict[str, Any]],
                   mutations: dict[str, str] | None = None, rebind_digest: bool = True) -> None:
    try:
        # Rebind the digest for source mutations so the intended extractor or
        # compiler assertion—not the outer provenance guard—must reject drift.
        trial_catalog, trial_corpus = copy.deepcopy(catalog), copy.deepcopy(corpus)
        if mutations and rebind_digest:
            digest = input_digest(repo, provenance_paths(repo, trial_corpus, mutations), mutations)
            trial_catalog["source_input_digest"] = digest
            trial_corpus["source_input_digest"] = digest
        validate(repo, trial_catalog, trial_corpus, schemas, consumers, mutations)
    except CatalogError:
        return
    raise AssertionError(f"mutation probe {name} unexpectedly passed")


def expect_event_failure(name: str, repo: Path, catalog: dict[str, Any], corpus: dict[str, Any],
                         schemas: list[str], consumers: dict[str, dict[str, Any]], event: str, revision: str) -> None:
    try:
        validate(repo, catalog, corpus, schemas, consumers, event=event, revision=revision)
    except CatalogError:
        return
    raise AssertionError(f"event mutation probe {name} unexpectedly passed")


def expect_success(name: str, repo: Path, catalog: dict[str, Any], corpus: dict[str, Any],
                   schemas: list[str], mutations: dict[str, str]) -> None:
    """Prove a non-code decoy does not alter the source-derived inventory."""
    trial_catalog, trial_corpus = copy.deepcopy(catalog), copy.deepcopy(corpus)
    digest = input_digest(repo, provenance_paths(repo, trial_corpus, mutations), mutations)
    trial_catalog["source_input_digest"] = digest
    trial_corpus["source_input_digest"] = digest
    try:
        validate(repo, trial_catalog, trial_corpus, schemas, source_services(repo, mutations), mutations)
    except CatalogError as error:
        raise AssertionError(f"mutation probe {name} unexpectedly failed: {error}") from error


def expect_cgr_failure(name: str, build_file: str, source: str) -> None:
    try:
        cgr_inventory(build_file, source)
    except CatalogError:
        return
    raise AssertionError(f"CGR mutation probe {name} unexpectedly passed")


@contextmanager
def isolated_index(repo: Path):
    """Copy the exact caller index; all probe writes target only this copy.

    Preserve absent entries, unmerged stages and index extensions/flags without
    reconstructing entries. The environment is restored even if setup fails.
    A preexisting GIT_INDEX_FILE is the caller index and is never overwritten.
    """
    caller = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index"))
    previous = os.environ.get("GIT_INDEX_FILE")
    with tempfile.TemporaryDirectory(prefix="docs-catalog-index-") as temporary:
        index = Path(temporary) / "index"
        if caller.exists():
            index.write_bytes(caller.read_bytes())
        try:
            os.environ["GIT_INDEX_FILE"] = str(index)
            if not index.exists():
                git(repo, "read-tree", "--empty")
            yield
        finally:
            if previous is None: os.environ.pop("GIT_INDEX_FILE", None)
            else: os.environ["GIT_INDEX_FILE"] = previous


def probe_staged_removal(repo: Path, check_removed: Callable[[], None]) -> None:
    removed_schema = "crates/hyprstream-rpc/schema/optional.capnp"
    unrelated_schema = "crates/hyprstream-rpc/schema/common.capnp"
    def staged_entries(path: str) -> list[str]:
        return git(repo, "ls-files", "--stage", "--", path).splitlines()
    def stage_edit(path: str) -> str:
        original = read_regular(repo, path).decode("utf-8")
        blob = subprocess.run(["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
                              input=original + "\n# staged schema edit\n", text=True,
                              capture_output=True, check=True).stdout.strip()
        git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        return blob
    with isolated_index(repo):
        optional_blob = stage_edit(removed_schema)
        common_blob = stage_edit(unrelated_schema)
        git(repo, "rm", "--cached", "--force", "--quiet", removed_schema)
        check_removed()
        git(repo, "update-index", "--add", "--cacheinfo", f"100644,{optional_blob},{removed_schema}")
        for path, expected in [(removed_schema, optional_blob), (unrelated_schema, common_blob)]:
            entries = staged_entries(path)
            required(len(entries) == 1 and entries[0].split()[1] == expected,
                     f"removal probe changed its staged input {path}")


def self_test(repo: Path) -> None:
    catalog, corpus = read_json(repo / "docs/schema-catalog.json", repo), read_json(repo / "docs/corpus-sources.json", repo)
    schemas, consumers = tracked(repo, "*.capnp"), source_services(repo)
    validate(repo, catalog, corpus, schemas, consumers)
    # PR-topology probes need a derivable merge-base: hosted PR runs always
    # supply one, but source archives and non-origin remotes legitimately lack
    # the ref, so those probes are skipped there instead of failing the self-test.
    boundary = subprocess.run(["git", "-C", str(repo), "merge-base", "HEAD", "refs/remotes/origin/main"],
                              capture_output=True, text=True)
    has_pr_boundary = boundary.returncode == 0
    pr_base = boundary.stdout.strip() if has_pr_boundary else ""
    # A hosted push validates its own topology only: it must never invent a PR
    # merge-base model after an ordinary main push.
    if os.environ.get("DOCS_CATALOG_EVENT", "local") != "push" and has_pr_boundary:
        validate(repo, catalog, corpus, schemas, consumers, event="pull_request",
                 revision=pr_base)
        validate(repo, catalog, corpus, schemas, consumers, event="pull_request",
                 revision=catalog["source_commit"])
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
    if has_pr_boundary:
        bad = copy.deepcopy(catalog); bad["source_commit"] = "not-a-git-revision"
        expect_event_failure("schema provenance", repo, bad, corpus, schemas, consumers, "pull_request", pr_base)
        bad = copy.deepcopy(catalog); bad["source_commit"] = "f" * 40
        expect_event_failure("fabricated provenance commit", repo, bad, corpus, schemas, consumers, "pull_request", pr_base)
    # A committed catalog may not self-reference; during staged authoring the
    # transient HEAD-referencing pair is the sanctioned authoring state.
    # Removal direction: deleting an audited schema and its catalog entry must
    # collide with the declared tree, which still contains the deleted file.
    # The deletion is simulated in this worktree's own index (ls-files reads
    # it) and restored afterwards; worktree indexes are not shared.
    factories_path = "crates/hyprstream/src/services/factories.rs"
    attribute_swap = text(repo, factories_path, None).replace(
        'schema = "../../../hyprstream-rpc-std/schema/registry.capnp", metadata = hyprstream_rpc_std::registry_client::schema_metadata',
        'schema = "../../../hyprstream-rpc-std/schema/model.capnp", metadata = hyprstream_rpc_std::model_client::schema_metadata', 1)
    expect_failure("factory schema/metadata attribute swap", repo, copy.deepcopy(catalog), corpus, schemas,
                   source_services(repo, {factories_path: attribute_swap}), {factories_path: attribute_swap})
    vfs_mount = "crates/hyprstream-rpc-std/src/vfs_mount.rs"
    module_swap = text(repo, vfs_mount, None).replace(
        'impl_service_dispatch!(RegistryDispatch, "registry", crate::registry_client)',
        'impl_service_dispatch!(RegistryDispatch, "registry", crate::model_client)', 1)
    expect_failure("VFS generated-module swap", repo, copy.deepcopy(catalog), corpus, schemas,
                   source_services(repo, {vfs_mount: module_swap}), {vfs_mount: module_swap})
    cli_path = "crates/hyprstream/src/cli/schema_cli.rs"
    cli_module_swap = text(repo, cli_path, None).replace(
        "let registry_tree = registry_client::scoped_client_tree();",
        "let registry_tree = model_client::scoped_client_tree();", 1)
    expect_failure("CLI metadata/scoped-tree module swap", repo, copy.deepcopy(catalog), corpus, schemas,
                   source_services(repo, {cli_path: cli_module_swap}), {cli_path: cli_module_swap})
    removed_schema = "crates/hyprstream-rpc/schema/optional.capnp"
    probe_catalog = copy.deepcopy(catalog)
    probe_catalog["schemas"] = [entry for entry in probe_catalog["schemas"] if entry["path"] != removed_schema]
    pruned = [path for path in schemas if path != removed_schema]
    probe_staged_removal(repo, lambda: expect_failure(
        "declared tree attests removed input", repo, probe_catalog, corpus, pruned,
        consumers, {removed_schema: None}))
    committed = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet", "--",
                                "docs/schema-catalog.json", "docs/corpus-sources.json"]).returncode == 0
    bad = copy.deepcopy(catalog); bad["source_commit"] = git(repo, "rev-parse", "HEAD"); bad["source_tree"] = git(repo, "rev-parse", "HEAD^{tree}")
    if committed:
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
    stale_base = git(repo, "rev-parse", f"{catalog['source_commit']}~1")
    bad = copy.deepcopy(catalog)
    bad["source_commit"] = stale_base
    bad["source_tree"] = git(repo, "rev-parse", f"{stale_base}^{{tree}}")
    expect_failure("declared tree predates audited input", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["schemas"][0]["license"] = "AGPL-3.0-or-later"
    expect_failure("manifest license", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog)
    settlement = next(entry for entry in bad["schemas"] if entry["path"].endswith("settlement.capnp"))
    settlement["kind"], settlement["service"] = "type-only", None
    expect_failure("schema service reclassification", repo, bad, corpus, schemas, consumers)
    common_path = "crates/hyprstream-rpc/schema/common.capnp"
    common_comment_decoy = text(repo, common_path, None) + "\n# struct CommonRequest {}\n"
    expect_success("schema kind comment decoy", repo, catalog, corpus, schemas,
                   {common_path: common_comment_decoy})
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
    renamed = text(repo, cli_path, None).replace('Command::new("discovery")', 'Command::new("settlement")', 1)
    expect_failure("manual CLI rename", repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, {cli_path: renamed}))
    removed = text(repo, cli_path, None).replace("tool = tool.subcommand(discovery);", "// manual discovery registration removed", 1)
    expect_failure("manual CLI removal", repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, {cli_path: removed}))
    raw_cli = text(repo, cli_path, None)
    builder_guard = "if method.cli_hidden || method.is_streaming"
    guard_offsets = [item.start() for item in re.finditer(re.escape(builder_guard), raw_cli)]
    required(len(guard_offsets) == 2, "unexpected CLI builder guard count")
    for label, builder in [("CLI service builder policy", "fn build_service_command"),
                           ("CLI scoped builder policy", "fn build_scoped_command_from_node")]:
        fn_at = raw_cli.find(builder)
        following = raw_cli.find("\nfn ", fn_at + 1)
        span_end = len(raw_cli) if following < 0 else following
        inside = [offset for offset in guard_offsets if fn_at < offset < span_end]
        required(len(inside) == 1, f"{builder} guard not uniquely located")
        mutated = raw_cli[:inside[0]] + "if false" + raw_cli[inside[0] + len(builder_guard):]
        expect_failure(label, repo, copy.deepcopy(catalog), corpus, schemas,
                       source_services(repo, {cli_path: mutated}), {cli_path: mutated})
    worker_path = "crates/hyprstream-rpc-std/schema/worker.capnp"
    hidden_removed = text(repo, worker_path, None).replace("$cliHidden ", "", 1)
    expect_failure("schema hidden annotation", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {worker_path: hidden_removed})
    hidden_comment = text(repo, worker_path, None).replace("$cliHidden", "# $cliHidden", 1)
    required("worker.container.attach" not in schema_method_metadata(repo, catalog["schemas"], {worker_path: hidden_comment})["cli_hidden"],
             "comment-only hidden annotation drift")
    expect_failure("schema hidden comment decoy", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {worker_path: hidden_comment})
    hidden_literal = text(repo, worker_path, None).replace("$cliHidden", '$mutationSemantics("$cliHidden")', 1)
    required("worker.container.attach" not in schema_method_metadata(repo, catalog["schemas"], {worker_path: hidden_literal})["cli_hidden"],
             "literal-only hidden annotation drift")
    expect_failure("schema hidden literal decoy", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {worker_path: hidden_literal})
    canonical_build = "crates/hyprstream-rpc-std/build.rs"
    canonical_source = text(repo, canonical_build, None)
    whitespace = canonical_source.replace("hyprstream_rpc_build::compile_schemas(", "hyprstream_rpc_build :: compile_schemas (", 1)
    required(cgr_inventory(canonical_build, whitespace) == EXPECTED_CGR_INVOCATIONS[canonical_build]["invocations"],
             "CGR whitespace normalization drift")
    aliased = ("use hyprstream_rpc_build::compile_schemas as compile;\n" + canonical_source
               .replace("hyprstream_rpc_build::compile_schemas(", "compile(", 1))
    required(cgr_inventory(canonical_build, aliased) == EXPECTED_CGR_INVOCATIONS[canonical_build]["invocations"],
             "CGR function-alias normalization drift")
    module_aliased = ("use hyprstream_rpc_build as rpc_build;\n" + canonical_source
                      .replace("hyprstream_rpc_build::compile_schemas(", "rpc_build::compile_schemas(", 1))
    required(cgr_inventory(canonical_build, module_aliased) == EXPECTED_CGR_INVOCATIONS[canonical_build]["invocations"],
             "CGR module-alias normalization drift")
    grouped_aliased = ("use hyprstream_rpc_build::{nested::{SchemaMetadata}, compile_schemas as compile, SchemaMetadata};\n" + canonical_source
                       .replace("hyprstream_rpc_build::compile_schemas(", "compile(", 1))
    required(cgr_inventory(canonical_build, grouped_aliased) == EXPECTED_CGR_INVOCATIONS[canonical_build]["invocations"],
             "CGR grouped function-alias normalization drift")
    lifetime_extra = canonical_source + "\nfn marker<'a>() {}\n" + canonical_source[canonical_source.find("hyprstream_rpc_build::compile_schemas("):]
    required(len(cgr_inventory(canonical_build, lifetime_extra)) == 2,
             "CGR lifetime tokenization hid an invocation")
    comment_extra = canonical_source + "\n/* // still block comment */\n" + canonical_source[canonical_source.find("hyprstream_rpc_build::compile_schemas("):]
    required(len(cgr_inventory(canonical_build, comment_extra)) == 2,
             "CGR block-comment tokenization hid an invocation")
    raw_string = canonical_source + '\nlet marker = br###"hyprstream_rpc_build::compile_schemas("###;\n'
    required(cgr_inventory(canonical_build, raw_string) == EXPECTED_CGR_INVOCATIONS[canonical_build]["invocations"],
             "CGR byte-raw string tokenization drift")
    diagnostic_string = canonical_source + '\nlet diagnostic = r#"let schema_dir = Path::new(\\"elsewhere\\");"#;\n'
    required(cgr_inventory(canonical_build, diagnostic_string) == EXPECTED_CGR_INVOCATIONS[canonical_build]["invocations"],
             "CGR diagnostic literal binding drift")
    expect_cgr_failure("CGR unresolved alias", canonical_build,
                       canonical_source.replace("hyprstream_rpc_build::compile_schemas(", "unknown::compile_schemas(", 1))
    expect_cgr_failure("CGR alias reassignment", canonical_build,
                       aliased + "\ncompile = other_compile;\n")
    expect_cgr_failure("CGR mutable binding", canonical_build,
                       canonical_source.replace("let rpc_schema_dir", "let mut rpc_schema_dir", 1))
    required(cgr_inventory("crates/unrelated/build.rs", 'fn main() { println!("compile_schemas"); }') == [],
             "CGR string-only producer false positive")
    root_changed = text(repo, canonical_build, None).replace(
        '.join("../hyprstream-rpc/schema")', '.join("../unrelated/schema")', 1)
    expect_failure("CGR source import root", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {canonical_build: root_changed})
    fixture_build = "crates/hyprstream-rpc-build/build.rs"
    fixture_drift = text(repo, fixture_build, None).replace(".file(&schema)", '.file("tests/other.capnp")', 1)
    expect_failure("capnp-only compiler input", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {fixture_build: fixture_drift})
    fixture_binding_drift = text(repo, fixture_build, None).replace(
        "tests/wire_roundtrip_fixture.capnp", "tests/other.capnp", 1
    )
    expect_failure("capnp-only resolved compiler input", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {fixture_build: fixture_binding_drift})
    fixture_shadow = text(repo, fixture_build, None).replace(
        "\n\n    capnpc::CompilerCommand", '\n    let schema = format!("{manifest}/tests/unrelated.capnp");\n\n    capnpc::CompilerCommand', 1
    )
    expect_failure("capnp-only effective shadow binding", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {fixture_build: fixture_shadow})
    tui_build = "crates/hyprstream-tui/build.rs"
    required(capnp_only_inputs(tui_build, text(repo, tui_build, None)) == [],
             "TUI build unexpectedly compiles a canonical schema")
    join_fixture = '\n'.join([
        'let schema_dir = Path::new("../hyprstream-rpc-std/schema");',
        'let schema = schema_dir.join("compositor_ipc.capnp");',
        'capnpc::CompilerCommand::new().file(&schema).run();',
    ])
    required(capnp_only_inputs(tui_build, join_fixture) == ["crates/hyprstream-rpc-std/schema/compositor_ipc.capnp"],
             "capnp-only join-binding resolution drift")
    direct_drift = text(repo, canonical_build, None).replace(
        "\n}", '\n    capnpc::CompilerCommand::new().file("../hyprstream-pay/schema/settlement.capnp").run();\n}', 1
    )
    expect_failure("direct capnpc compiles uncompiled schema", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {canonical_build: direct_drift})
    chained_drift = text(repo, canonical_build, None).replace(
        "\n}",
        '\n    capnpc::CompilerCommand::new().file("{manifest}/../hyprstream-rpc/schema/nine.capnp").file("../hyprstream-pay/schema/settlement.capnp").run().expect("chained");\n}', 1
    )
    expect_failure("chained capnpc supplies uncompiled schema", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {canonical_build: chained_drift})
    aliased_drift = text(repo, canonical_build, None).replace(
        "\n}",
        '\n}\n\nuse capnpc::CompilerCommand as CapnpCmd;\n\nfn extra() {\n    CapnpCmd::new().file("../hyprstream-pay/schema/settlement.capnp").run().expect("aliased");\n}', 1
    )
    expect_failure("imported capnpc compiles uncompiled schema", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {canonical_build: aliased_drift})
    module_drift = text(repo, canonical_build, None).replace(
        "\n}",
        '\n}\n\nuse capnpc as cp;\n\nfn extra() {\n    cp::CompilerCommand::new().file("../hyprstream-pay/schema/settlement.capnp").run().expect("module-aliased");\n}', 1
    )
    expect_failure("module-aliased capnpc compiles uncompiled schema", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {canonical_build: module_drift})
    module_drift = text(repo, canonical_build, None).replace(
        "\n}",
        '\n}\n\nuse capnpc as cp;\n\nfn extra() {\n    cp::CompilerCommand::new().file("../hyprstream-pay/schema/settlement.capnp").run().expect("module-aliased");\n}', 1
    )
    expect_failure("module-aliased capnpc compiles uncompiled schema", repo, copy.deepcopy(catalog), corpus,
                   schemas, consumers, {canonical_build: module_drift})
    shadowed = text(repo, canonical_build, None).replace(
        'let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");',
        'let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");\n    let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-workers/schema");',
    )
    expect_failure("CGR shadowed binding", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {canonical_build: shadowed})
    commented = text(repo, canonical_build, None).replace('.join("../hyprstream-rpc/schema")', '.join("../hyprstream-workers/schema")', 1) + '\n// let rpc_schema_dir = Path::new(&manifest_dir).join("../hyprstream-rpc/schema");\n'
    expect_failure("CGR commented declaration", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {canonical_build: commented})
    extra = text(repo, canonical_build, None).replace("\n}", '\n    hyprstream_rpc_build::compile_schemas(schema_dir, Path::new(&out_dir), &[], &["extra"]);\n}', 1)
    expect_failure("CGR additional invocation", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {canonical_build: extra})
    extra_producer = {'crates/unrelated/build.rs': canonical_source}
    expect_failure("CGR additional producer", repo, copy.deepcopy(catalog), corpus, schemas, consumers, extra_producer)
    model_path = "crates/hyprstream-rpc-std/schema/model.capnp"
    model_drift = text(repo, model_path, None).replace("generateStream @1 :StreamInfo", "generateStream @1 :Text", 1)
    expect_failure("scoped model streaming response", repo, copy.deepcopy(catalog), corpus, schemas, consumers, {model_path: model_drift})
    qualified_model = text(repo, model_path, None).replace(
        'using import "/streaming.capnp".StreamInfo;',
        'using Streaming = import "/streaming.capnp";',
    ).replace(":StreamInfo", ":Streaming.StreamInfo")
    required(schema_method_metadata(repo, catalog["schemas"], {model_path: qualified_model})
             == schema_method_metadata(repo, catalog["schemas"], None),
             "qualified streaming type changes method identity")
    expect_success("qualified streaming import", repo, catalog, corpus, schemas,
                   {model_path: qualified_model})
    # A new qualified leaf must require a catalog update independently of
    # provenance, including comments on either side of the imported path.
    for name, declaration in [
        ("ordinary", 'using Streaming = import "/streaming.capnp";'),
        ("adjacent literal", 'using Streaming=import"/streaming.capnp";'),
        ("before path", 'using Streaming = import # ignored "decoy.capnp";\n  "/streaming.capnp";'),
        ("after path", 'using Streaming = import "/streaming.capnp" # ignored "decoy.capnp";\n  ;'),
        ("all gaps", 'using # declaration\n Streaming # alias\n = # assignment\n import # path\n "/streaming.capnp" # terminator\n ;'),
    ]:
        added_stream = text(repo, model_path, None).replace(
            'using import "/streaming.capnp".StreamInfo;',
            'using import "/streaming.capnp".StreamInfo;\n' + declaration, 1,
        ).replace(
            "struct InferRequest {\n  modelRef @0 :Text;\n  union {",
            'struct InferRequest {\n  modelRef @0 :Text;\n  union {\n'
            '    extraStream @5 :Void $scope(query) $dispatchMac("internal:pq-hybrid");', 1,
        ).replace(
            "struct InferResponse {\n  union {",
            "struct InferResponse {\n  union {\n    extraStream @5 :Streaming.StreamInfo;", 1,
        )
        added_metadata = schema_method_metadata(repo, catalog["schemas"], {model_path: added_stream})
        required("model.infer.extraStream" in added_metadata["streaming"],
                 f"qualified streaming leaf lost across {name} import gap")
        expect_failure(f"new qualified streaming leaf with {name} import", repo,
                       catalog, corpus, schemas, consumers, {model_path: added_stream})
        updated = copy.deepcopy(catalog); updated["method_metadata"] = added_metadata
        expect_success(f"cataloged qualified streaming leaf with {name} import", repo,
                       updated, corpus, schemas, {model_path: added_stream})
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
    required("registry.repo.worktree.deepStream" in deep_metadata["streaming"],
             "deeply scoped streaming extractor drift")
    shared_scope = deep_stream.replace(
        "repo @9 :RepositoryRequest", "mirror @98 :RepositoryRequest;\n    repo @9 :RepositoryRequest",
    ).replace(
        "repoResult @10 :RepositoryResponse;",
        "mirrorResult @98 :RepositoryResponse;\n    repoResult @10 :RepositoryResponse;",
    )
    shared_metadata = schema_method_metadata(repo, catalog["schemas"], {registry_path: shared_scope})
    required({"registry.repo.worktree.deepStream", "registry.mirror.worktree.deepStream"}
             <= set(shared_metadata["streaming"]),
             "shared request/response types lose distinct scope identities")
    expect_failure("deeply scoped streaming response", repo, copy.deepcopy(catalog), corpus, schemas,
                   consumers, {registry_path: deep_stream})
    for name, path, decoy in [
        ("CLI literal registration decoy", cli_path,
         'let _ = r#"build_service_command(\"ghost\", &ghost_methods, ghost_tree)"#;'),
        ("MCP literal registration decoy", "crates/hyprstream/src/services/mcp_service.rs",
         'let _ = "register_top_level!(reg, ghost_client::schema_metadata())";'),
        ("factory literal registration decoy", "crates/hyprstream/src/services/factories.rs",
         'let _ = r#"#[service_factory(\"ghost\")]"#;'),
        ("VFS literal registration decoy", "crates/hyprstream-rpc-std/src/vfs_mount.rs",
         'let _ = r#"impl_service_dispatch!(GhostDispatch, \"ghost\", crate::ghost_client)"#;'),
    ]:
        mutated = text(repo, path, None) + "\n" + decoy + "\n"
        required(source_services(repo, {path: mutated}) == consumers,
                 f"{name} altered source-derived registrations")
        expect_success(name, repo, catalog, corpus, schemas, {path: mutated})
    unrelated_js = {"website/unrelated.js": "export const unrelated = true;\n"}
    required(typescript_schema_sources(repo, unrelated_js, ["website/unrelated.js"]) == [],
             "unrelated tracked JavaScript is a schema consumer")
    required(set(typescript_schema_sources(repo)) <= set(provenance_paths(repo, corpus)),
             "TypeScript schema consumers are omitted from the provenance digest")
    backtick_only = {"web/ts_only.ts": 'import(`@hyprstream/docs`);\nexport const tree = require(`./generated/catalog.capnp`);\n'}
    required(typescript_schema_sources(repo, backtick_only, ["web/ts_only.ts"]) == ["web/ts_only.ts"],
             "backtick-only schema dependency missed")
    real_import = {"web/real.ts": 'import { manifest } from "@hyprstream/docs";\nexport const tree = require("./generated/catalog.capnp");\nexport const doc = import(`./generated/echo.capnp`);\n'}
    required(typescript_schema_sources(repo, real_import, ["web/real.ts"]) == ["web/real.ts"],
             "valid schema import or require missed")
    marked_comment = {"web/decoy.ts": '// import { manifest } from "@hyprstream/docs";\n/* const tree = require("./generated/echo.capnp"); */\nconst note = "loads codegen-out via @hyprstream/docs .capnp";\nconst tip = `see docs/x.capnp for details`;\n'}
    required(typescript_schema_sources(repo, marked_comment, ["web/decoy.ts"]) == [],
             "commented or string-only schema marker counted as a consumer")
    regex_literal = {"web/regex.ts": 'const dependency_decoy = /require("fixture.capnp")/;\n'}
    required(typescript_schema_sources(repo, regex_literal, ["web/regex.ts"]) == [],
             "regex-literal schema marker counted as a consumer")
    ts_path = ".github/scripts/verify-rpc-std-wasm-package.mjs"
    concrete_import = {ts_path: text(repo, ts_path, None) + '\nimport "./registry.capnp";\n'}
    concrete_consumers = source_services(repo, concrete_import)
    required(concrete_consumers["typescript"]["schema_imports"] == ["crates/hyprstream-rpc-std/schema/registry.capnp"],
             "concrete TypeScript schema import inventory drift")
    bad = copy.deepcopy(catalog)
    bad["consumer_sets"]["typescript"] = {
        "state": "active", "tracked_sources": [ts_path],
        "schema_imports": ["crates/hyprstream-rpc-std/schema/registry.capnp"],
    }
    expect_failure("TypeScript concrete schema surface", repo, bad, corpus, schemas,
                   concrete_consumers, concrete_import)
    positive = copy.deepcopy(catalog)
    positive["consumer_sets"]["typescript"] = {
        "state": "active", "tracked_sources": [ts_path],
        "schema_imports": ["crates/hyprstream-rpc-std/schema/registry.capnp"],
    }
    registry = next(entry for entry in positive["schemas"]
                    if entry["path"] == "crates/hyprstream-rpc-std/schema/registry.capnp")
    registry["surfaces"].append("typescript")
    registry["exclusions"].pop("typescript")
    expect_success("TypeScript concrete schema surface declared", repo, positive, corpus, schemas,
                   concrete_import)
    mcp_path = "crates/hyprstream/src/services/mcp_service.rs"
    for label, needle in [("hidden policy", "if method.hidden {"),
                          ("streaming policy", "if method.is_streaming {")]:
        for occurrence, path_name in [(1, "top-level"), (2, "scoped")]:
            mutated = replace_nth(text(repo, mcp_path, None), needle, "if false {", occurrence)
            expect_failure(f"MCP {path_name} {label}", repo, copy.deepcopy(catalog), corpus, schemas,
                           source_services(repo, {mcp_path: mutated}), {mcp_path: mutated})
    for label, before, after in [
        ("hidden guard is effective", "if method.hidden {\n                    continue;",
         "if method.hidden {}\n                if false { continue;"),
        ("streaming branch is effective", "if method.is_streaming {",
         "if method.is_streaming {}\n                if false {"),
    ]:
        mutated = replace_nth(text(repo, mcp_path, None), before, after, 1)
        expect_failure(f"MCP top-level {label}", repo, copy.deepcopy(catalog), corpus, schemas,
                       source_services(repo, {mcp_path: mutated}), {mcp_path: mutated})
    nested_stream = text(repo, mcp_path, None).replace(
        "register_streaming_tool(\n", "if false { register_streaming_tool(\n", 1
    ).replace("                } else {\n                    register_sync_tool(",
              "                }\n                } else {\n                    register_sync_tool(", 1)
    expect_failure("MCP nested streaming registration", repo, copy.deepcopy(catalog), corpus, schemas,
                   source_services(repo, {mcp_path: nested_stream}), {mcp_path: nested_stream})
    precontinued_stream = text(repo, mcp_path, None).replace(
        "if method.is_streaming {\n", "if method.is_streaming {\n                    continue;\n", 1)
    expect_failure("MCP precontinued streaming registration", repo, copy.deepcopy(catalog), corpus, schemas,
                   source_services(repo, {mcp_path: precontinued_stream}), {mcp_path: precontinued_stream})
    delayed_hidden = ("for method in methods {\n"
                      "if method.is_streaming { register_streaming_tool(); } else { register_sync_tool(); }\n"
                      "if method.hidden { continue; }\n}")
    required(mcp_loop_policy(delayed_hidden, "top-level") == (False, True),
             "MCP hidden guard after registration is accepted")
    # Scoped roots are a separate projection even when their service also
    # has top-level tools. Their removal or rebinding must update the catalog.
    scoped_source = text(repo, mcp_path, None)
    for name, before, after in [
        ("MCP scoped module", "registry_client::scoped_client_tree()", "model_client::scoped_client_tree()"),
        ("MCP scoped prefix", '        "registry",\n        &[],', '        "other",\n        &[],'),
        ("MCP scoped-only service", "    // Scoped tools:", '    register_scoped_tools_recursive(reg, "worker", worker_client::scoped_client_tree(), "worker", &[]);\n    // Scoped tools:'),
    ]:
        changed = replace_nth(scoped_source, before, after, 1)
        expect_failure(name, repo, copy.deepcopy(catalog), corpus, schemas,
                       source_services(repo, {mcp_path: changed}), {mcp_path: changed})
    default_build = "crates/hyprstream-discovery/build.rs"
    default_source = text(repo, default_build, None) + '\nfn extra() { capnpc::CompilerCommand::default().file("../hyprstream-pay/schema/settlement.capnp").run(); }'
    expect_failure("default constructor compiles unclassified input", repo, copy.deepcopy(catalog),
                   corpus, schemas, consumers, {default_build: default_source})
    bad = copy.deepcopy(catalog)
    fixture = next(entry for entry in bad["schemas"] if entry["path"].endswith("wire_roundtrip_fixture.capnp"))
    fixture["exclusions"].pop("docs")
    expect_failure("mandatory docs surface disposition", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["consumer_sets"]["cli"]["source"] = "crates/other/schema_cli.rs"
    expect_failure("declared consumer source path", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(catalog)
    bad["consumer_sets"]["typescript"] = {"state": "active", "tracked_sources": ["package.json"]}
    supplied = copy.deepcopy(consumers)
    supplied["typescript"] = {"tracked_sources": ["package.json"]}
    expect_failure("unrelated TypeScript schema consumer", repo, bad, corpus, schemas, supplied)
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
    bad = copy.deepcopy(corpus); bad["corpus_manifest"]["limit_bytes"] = 1
    expect_failure("corpus manifest contract", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["package_contract"]["exports"] = {"./corpus/*": "./dist/*"}
    expect_failure("package exports", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["consumer_sets"]["cli"]["state"] = "absent"
    expect_failure("consumer state contradiction", repo, bad, corpus, schemas, consumers)
    shared = next(entry for entry in catalog["schemas"] if entry["path"].endswith("streaming.capnp"))
    required(shared.get("cgr_producers") == ["crates/hyprstream-rpc/build.rs", "crates/hyprstream/build.rs"],
             "shared-schema CGR producer inventory drift")
    bad = copy.deepcopy(catalog)
    shared_entry = next(entry for entry in bad["schemas"] if entry["path"].endswith("streaming.capnp"))
    shared_entry["cgr_producers"] = shared_entry["cgr_producers"][:1]
    expect_failure("shared schema producer under-recorded", repo, bad, corpus, schemas, consumers)
    scoped = ["worker", "pod-sandbox", "pod-sandbox/container", "image"]
    identities = {corpus["api_manifest"]["id"].format(service="worker", scope=item, method="list") for item in scoped}
    required(len(identities) == len(scoped), "scoped API identities collide")
    top_level = "docs/KV-CACHE-ARCHITECTURE.md"
    changed_top_level = text(repo, top_level, None) + "\nprovenance mutation\n"
    expect_failure("top-level corpus provenance", repo, catalog, corpus, schemas, consumers, {top_level: changed_top_level}, False)
    # Model a real landing push: the pushed head carries every attested path at
    # the declared digest while the pre-feature boundary (the merge-base, whose
    # tree predates this PR's new docs files) supplies only ancestry. The
    # declared pair — not the boundary tree — carries the attestation, so the
    # landing validates even though the boundary omits attested paths.
    landing_boundary = pr_base if has_pr_boundary else git(repo, "rev-parse", "HEAD~1")
    validate(repo, catalog, corpus, schemas, consumers, event="push", revision=landing_boundary)
    # Missing/forged source objects are never accepted on digest alone. Real
    # recovery and trusted inheritance are exercised in the isolated fixtures.
    squashed = copy.deepcopy(catalog)
    squashed["source_commit"], squashed["source_tree"] = "f" * 40, "f" * 40
    expect_event_failure("unverified squash provenance", repo, squashed, corpus, schemas,
                         consumers, "push", landing_boundary)
    # A landing commit cannot attest itself.
    if committed:
        bad = copy.deepcopy(catalog)
        bad["source_commit"] = git(repo, "rev-parse", "HEAD")
        bad["source_tree"] = git(repo, "rev-parse", "HEAD^{tree}")
        try:
            validate(repo, bad, corpus, schemas, consumers, event="push", revision=landing_boundary)
        except CatalogError:
            pass
        else:
            raise AssertionError("mutation probe modeled landing push provenance unexpectedly passed")
    print("docs catalog mutation probes: passed (all expected failures plus modeled landing push)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--recover-source-objects", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        if args.self_test:
            required(not args.recover_source_objects, "self-tests never recover source objects")
            self_test(args.repo.resolve())
        else:
            validate(args.repo.resolve(), recover=args.recover_source_objects)
        if not args.self_test:
            print("docs catalog: OK")
    except (CatalogError, AssertionError, UnicodeError) as error:
        print(f"docs catalog: FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
