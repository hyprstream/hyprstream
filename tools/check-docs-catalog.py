#!/usr/bin/env python3
"""Fail-closed checks for schema coverage and publication contracts."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SURFACES = ("cli", "mcp", "factory", "vfs", "typescript")
OWNER_MANIFESTS = {
    "hyprstream": "crates/hyprstream/Cargo.toml",
    "hyprstream-discovery": "crates/hyprstream-discovery/Cargo.toml",
    "hyprstream-pay": "crates/hyprstream-pay/Cargo.toml",
    "hyprstream-rpc": "crates/hyprstream-rpc/Cargo.toml",
    "hyprstream-rpc-build": "crates/hyprstream-rpc-build/Cargo.toml",
    "hyprstream-rpc-std": "crates/hyprstream-rpc-std/Cargo.toml",
    "hyprstream-workers": "crates/hyprstream-workers/Cargo.toml",
}
PUBLIC_GLOBS = {
    "docs/*.md", "docs/adr/**/*.md", "docs/contracts/**/*.md",
    "docs/deployment/**/*.md", "docs/network/**/*.md", "docs/security/**/*.md",
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
    cli = text(repo, "crates/hyprstream/src/cli/schema_cli.rs", mutations)
    mcp = text(repo, "crates/hyprstream/src/services/mcp_service.rs", mutations)
    factories = text(repo, "crates/hyprstream/src/services/factories.rs", mutations)
    vfs = text(repo, "crates/hyprstream-rpc-std/src/vfs_mount.rs", mutations)

    cli_services = re.findall(r'build_service_command\(\s*"([a-z0-9-]+)"', cli)
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


def check_provenance(record: dict[str, Any], repo: Path, label: str) -> None:
    commit = record.get("source_commit")
    tree = record.get("source_tree")
    required(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, f"{label} has invalid source_commit")
    required(isinstance(tree, str) and re.fullmatch(r"[0-9a-f]{40}", tree) is not None, f"{label} has invalid source_tree")
    git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    required(git(repo, "rev-parse", f"{commit}^{{tree}}") == tree, f"{label} source_tree does not match source_commit")


def manifest_license(repo: Path, owner: str) -> str:
    manifest = OWNER_MANIFESTS.get(owner)
    required(manifest is not None, f"unknown schema owner {owner}")
    found = re.search(r'^license\s*=\s*"([^"]+)"', text(repo, manifest, None), re.MULTILINE)
    required(found is not None, f"{manifest} lacks package license")
    return found.group(1)


def check_cgr(catalog: dict[str, Any], repo: Path, schemas: list[dict[str, Any]]) -> None:
    roots = catalog.get("cgr_build_roots", {})
    required(roots == CGR_ROOTS, "persisted-CGR import roots drift")
    for build_file, import_roots in roots.items():
        source = text(repo, build_file, None)
        required("hyprstream_rpc_build::compile_schemas" in source, f"{build_file} is not a persisted-CGR producer")
        required(isinstance(import_roots, list) and import_roots, f"{build_file} lacks import roots")
        for root in import_roots:
            relative = os.path.relpath(root, str(Path(build_file).parent)).replace("\\", "/")
            required(relative in source or f'"{Path(root).name}"' in source, f"{build_file} does not reference declared import root {root}")
    for entry in schemas:
        producer = entry.get("cgr_producer")
        if producer is None:
            mode = entry.get("compiled_by")
            required(mode in {"capnp_only", "not_compiled"}, f"{entry['path']} must distinguish non-CGR compilation")
            if mode == "capnp_only":
                source = text(repo, "crates/hyprstream-rpc-build/build.rs", None)
                required("capnpc::CompilerCommand" in source and Path(entry["path"]).stem in source, f"{entry['path']} capnp-only claim drift")
            continue
        source = text(repo, producer, None)
        required(producer in roots, f"{entry['path']} producer is not an audited persisted-CGR root")
        required("hyprstream_rpc_build::compile_schemas" in source, f"{entry['path']} producer does not emit persisted CGR")
        required(re.search(rf'["\']{re.escape(Path(entry["path"]).stem)}["\']', source) is not None,
                 f"{entry['path']} is absent from declared persisted-CGR inputs")


def check_schema_catalog(catalog: dict[str, Any], repo: Path, schema_paths: list[str], consumers: dict[str, dict[str, Any]]) -> None:
    required(catalog.get("schema_version") == 1, "schema catalog must use schema_version 1")
    required(catalog.get("authority") == "docs/system-ontology.md", "system ontology must remain authoritative")
    check_provenance(catalog, repo, "schema catalog")
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
    for entry in schemas:
        for key in ("owner", "license", "kind", "exclusions"):
            required(bool(entry.get(key)), f"{entry['path']} lacks {key}")
        required(entry["license"] == manifest_license(repo, entry["owner"]), f"{entry['path']} license differs from owner manifest")
        source = text(repo, entry["path"], None)
        found = re.search(r"^@(0x[0-9a-f]+);", source, re.MULTILINE)
        required(found is not None and found.group(1) == entry["source_id"], f"{entry['path']} source ID drift")
        active = set(entry.get("surfaces", []))
        required(active <= set(SURFACES) | {"docs"}, f"{entry['path']} has unknown surface")
        for surface in SURFACES:
            required(surface in active or bool(entry["exclusions"].get(surface)), f"{entry['path']} lacks {surface} disposition")
    check_cgr(catalog, repo, schemas)

    declared = catalog.get("consumer_sets", {})
    required(set(declared) == set(SURFACES), "consumer sets must enumerate every surface")
    for surface in SURFACES:
        record, actual = declared[surface], consumers[surface]
        required(record.get("state") in {"active", "declared", "absent"}, f"invalid {surface} state")
        key = "tracked_sources" if surface == "typescript" else "services"
        required(record.get(key) == actual.get(key), f"{surface} source registration drift")
        if surface in {"cli", "mcp"}:
            required(record.get("method_policy") == actual.get("method_policy"), f"{surface} hidden/streaming policy drift")
        if surface == "factory":
            required(record.get("feature_conditions") == actual.get("feature_conditions"), "factory feature condition drift")
        if surface == "typescript":
            required(record.get("state") == "absent" and record[key] == [], "TypeScript state must reflect tracked sources")
            continue
        for service in record[key]:
            if service not in by_service:
                required(bool(record.get("non_schema_services", {}).get(service)), f"{surface} registers unclassified service {service}")
            else:
                required(surface in by_service[service].get("surfaces", []), f"{service} active on {surface} but omitted")
        for service, entry in by_service.items():
            if surface in entry.get("surfaces", []):
                required(service in record[key], f"{service} claims {surface} activity without source registration")


def check_corpus(corpus: dict[str, Any], repo: Path) -> None:
    required(corpus.get("schema_version") == 1 and corpus.get("authority") == "docs/system-ontology.md", "invalid corpus authority/version")
    check_provenance(corpus, repo, "corpus catalog")
    for name in ("api_manifest", "corpus_manifest"):
        record = corpus.get(name, {})
        required(record.get("version") == 1 and isinstance(record.get("limit_bytes"), int) and record["limit_bytes"] > 0, f"invalid {name}")
        required(all(record.get(key) for key in ("path", "id", "integrity")), f"{name} lacks stable contract fields")
    public, excluded = corpus.get("public_prose", []), corpus.get("excluded", [])
    required({item.get("glob") for item in public} == PUBLIC_GLOBS, "public prose allowlist widened or incomplete")
    for item in public:
        required(item.get("license") == "AGPL-3.0-only", f"{item.get('glob')} lacks exact SPDX license")
        required(isinstance(item.get("provenance"), str) and item["provenance"].startswith("tracked first-party repository"), "public prose provenance drift")
    required({item.get("glob"): item.get("reason") for item in excluded} == EXCLUSIONS, "corpus exclusions or reasons drift")
    package = corpus.get("package_contract", {})
    required(package.get("name") == "@hyprstream/docs" and package.get("state") == "declared-not-yet-published", "invalid docs package contract")
    required(package.get("requirements") == PACKAGE_REQUIREMENTS, "docs package requirements drift")
    allow, deny = [item["glob"] for item in public], [item["glob"] for item in excluded]
    for path in tracked(repo, "docs/**/*.md"):
        if not any(fnmatch.fnmatch(path, pattern) for pattern in deny):
            required(any(fnmatch.fnmatch(path, pattern) for pattern in allow), f"unallowlisted public prose: {path}")


def validate(repo: Path, catalog: dict[str, Any] | None = None, corpus: dict[str, Any] | None = None,
             schema_paths: list[str] | None = None, consumers: dict[str, dict[str, Any]] | None = None) -> None:
    catalog = catalog or read_json(repo / "docs/schema-catalog.json")
    corpus = corpus or read_json(repo / "docs/corpus-sources.json")
    check_schema_catalog(catalog, repo, schema_paths or tracked(repo, "*.capnp"), consumers or source_services(repo))
    check_corpus(corpus, repo)
    required("docs/system-ontology.md" in text(repo, "docs/contracts/docs-pipeline.md", None), "pipeline contract omits ontology authority")


def expect_failure(name: str, repo: Path, catalog: dict[str, Any], corpus: dict[str, Any],
                   schemas: list[str], consumers: dict[str, dict[str, Any]]) -> None:
    try:
        validate(repo, catalog, corpus, schemas, consumers)
    except CatalogError:
        return
    raise AssertionError(f"mutation probe {name} unexpectedly passed")


def self_test(repo: Path) -> None:
    catalog, corpus = read_json(repo / "docs/schema-catalog.json"), read_json(repo / "docs/corpus-sources.json")
    schemas, consumers = tracked(repo, "*.capnp"), source_services(repo)
    validate(repo, catalog, corpus, schemas, consumers)
    expect_failure("unlisted schema", repo, copy.deepcopy(catalog), corpus, schemas + ["new.capnp"], consumers)
    expect_failure("stale schema", repo, copy.deepcopy(catalog), corpus, schemas[1:], consumers)
    bad = copy.deepcopy(catalog); bad["source_commit"] = "0" * 40
    expect_failure("schema provenance", repo, bad, corpus, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["source_tree"] = "0" * 40
    expect_failure("corpus provenance", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(catalog); bad["schemas"][0]["license"] = "MIT"
    expect_failure("manifest license", repo, bad, corpus, schemas, consumers)
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
    expect_failure("TypeScript source", repo, copy.deepcopy(catalog), corpus, schemas, source_services(repo, tracked_sources=["package.json"]))
    bad = copy.deepcopy(corpus); bad["excluded"][0]["reason"] = ""
    expect_failure("exclusion reason", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["public_prose"][0]["glob"] = "docs/**/*.md"
    expect_failure("allowlist widening", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["public_prose"][0]["license"] = "MIT"
    expect_failure("public license", repo, catalog, bad, schemas, consumers)
    bad = copy.deepcopy(corpus); bad["package_contract"]["requirements"] = []
    expect_failure("package requirements", repo, catalog, bad, schemas, consumers)
    print("docs catalog mutation probes: passed (16 expected failures)")


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
