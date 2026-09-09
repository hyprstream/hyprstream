#!/usr/bin/env python3
"""Fail-closed contract check for the schema and documentation catalog."""

from __future__ import annotations

import argparse
import copy
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SURFACES = ("cli", "mcp", "factory", "vfs", "typescript")


class CatalogError(Exception):
    pass


def fail(message: str) -> None:
    raise CatalogError(message)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"{path}: {error}")


def tracked(repo: Path, *patterns: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", *patterns],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        fail(f"git ls-files failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line]


def source_services(repo: Path) -> dict[str, list[str]]:
    cli = (repo / "crates/hyprstream/src/cli/schema_cli.rs").read_text(encoding="utf-8")
    mcp = (repo / "crates/hyprstream/src/services/mcp_service.rs").read_text(encoding="utf-8")
    factories = (repo / "crates/hyprstream/src/services/factories.rs").read_text(encoding="utf-8")
    vfs = (repo / "crates/hyprstream-rpc-std/src/vfs_mount.rs").read_text(encoding="utf-8")

    cli_services = re.findall(r'build_service_command\(\s*"([a-z0-9-]+)"', cli)
    mcp_modules = re.findall(
        r"register_top_level!\(\s*reg,\s*([a-zA-Z0-9_:]+)::schema_metadata\(\)\s*\)", mcp
    )
    mcp_services = [module.rsplit("::", 1)[-1].removesuffix("_client") for module in mcp_modules]
    factory_services = re.findall(r'#\[service_factory\(\s*"([a-z0-9-]+)"', factories)
    vfs_services = re.findall(
        r'impl_service_dispatch!\([^,]+,\s*"([a-z0-9-]+)"', vfs
    )
    ts_sources = tracked(repo, "*.ts", "*.tsx", "*.js", "*.jsx", "package.json")
    return {
        "cli": cli_services,
        "mcp": mcp_services,
        "factory": factory_services,
        "vfs": vfs_services,
        "typescript": ts_sources,
    }


def required(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def check_schema_catalog(
    catalog: dict[str, Any], repo: Path, tracked_schemas: list[str], consumers: dict[str, list[str]]
) -> None:
    required(catalog.get("schema_version") == 1, "schema catalog must use schema_version 1")
    required(catalog.get("authority") == "docs/system-ontology.md", "system ontology must remain authoritative")
    required(
        catalog.get("owner_directories") == [
            "crates/hyprstream/schema",
            "crates/hyprstream-discovery/schema",
            "crates/hyprstream-pay/schema",
            "crates/hyprstream-rpc/schema",
            "crates/hyprstream-rpc-std/schema",
            "crates/hyprstream-workers/schema",
        ],
        "catalog must name all six production schema-owner directories",
    )
    schemas = catalog.get("schemas")
    required(isinstance(schemas, list), "catalog.schemas must be a list")
    paths = [entry.get("path") for entry in schemas]
    required(len(paths) == len(set(paths)), "duplicate schema path in catalog")
    required(set(paths) == set(tracked_schemas), "catalog schema paths must exactly equal git-tracked .capnp files")
    source_ids = [entry.get("source_id") for entry in schemas]
    required(all(isinstance(value, str) and value.startswith("0x") for value in source_ids), "each schema needs a source ID")
    required(len(source_ids) == len(set(source_ids)), "duplicate schema source identity")
    service_ids = [entry["service"] for entry in schemas if entry.get("service") is not None]
    required(len(service_ids) == len(set(service_ids)), "duplicate schema service identity")

    by_service = {entry["service"]: entry for entry in schemas if entry.get("service")}
    roots = catalog.get("cgr_build_roots", {})
    required(isinstance(roots, dict) and roots, "catalog must record CGR build/import roots")
    for build_file, import_roots in roots.items():
        required((repo / build_file).is_file(), f"missing CGR build root source {build_file}")
        required(isinstance(import_roots, list) and import_roots, f"{build_file} needs import roots")
        source = (repo / build_file).read_text(encoding="utf-8")
        for import_root in import_roots:
            required(Path(import_root).name in source, f"{build_file} no longer references import root {import_root}")
    for entry in schemas:
        for key in ("owner", "license", "kind", "exclusions"):
            required(bool(entry.get(key)), f"{entry['path']} lacks {key}")
        source = (repo / entry["path"]).read_text(encoding="utf-8")
        found = re.search(r"^@(0x[0-9a-f]+);", source, re.MULTILINE)
        required(found is not None and found.group(1) == entry["source_id"], f"{entry['path']} source ID drift")
        producer = entry.get("cgr_producer")
        if producer is not None:
            required((repo / producer).is_file(), f"{entry['path']} names missing CGR producer {producer}")
            producer_source = (repo / producer).read_text(encoding="utf-8")
            required(Path(entry["path"]).stem in producer_source, f"{entry['path']} is not represented by declared CGR producer")
        active = set(entry.get("surfaces", []))
        required(active <= set(SURFACES) | {"docs"}, f"{entry['path']} has unknown surface")
        exclusions = entry["exclusions"]
        for surface in SURFACES:
            required(
                surface in active or bool(exclusions.get(surface)),
                f"{entry['path']} lacks an active {surface} surface or an exclusion reason",
            )

    declared = catalog.get("consumer_sets", {})
    required(set(declared) == set(SURFACES), "consumer sets must enumerate CLI, MCP, factory, VFS, and TypeScript")
    for surface in SURFACES:
        record = declared[surface]
        required(record.get("state") in {"active", "declared", "absent"}, f"invalid {surface} state")
        expected = record.get("services", record.get("tracked_sources", []))
        required(expected == consumers[surface], f"{surface} consumer list drift: expected {expected}, source has {consumers[surface]}")
        if surface == "typescript":
            required(record.get("state") == "absent" and expected == [], "TypeScript state must describe the actual empty tracked set")
            continue
        for service in expected:
            if service not in by_service:
                reason = record.get("non_schema_services", {}).get(service)
                required(bool(reason), f"{surface} registers unclassified service {service}")
                continue
            required(surface in by_service[service].get("surfaces", []), f"{service} is active on {surface} but catalog omits it")
        for service, entry in by_service.items():
            if surface in entry.get("surfaces", []):
                required(service in expected, f"{service} is marked active on {surface} but source does not register it")


def check_corpus(corpus: dict[str, Any], repo: Path) -> None:
    required(corpus.get("schema_version") == 1, "corpus manifest must use schema_version 1")
    required(corpus.get("authority") == "docs/system-ontology.md", "corpus must retain ontology authority")
    for manifest in ("api_manifest", "corpus_manifest"):
        record = corpus.get(manifest, {})
        for key in ("version", "path", "id", "integrity", "limit_bytes"):
            required(key in record, f"{manifest} lacks {key}")
        required(record["version"] == 1 and record["limit_bytes"] > 0, f"invalid {manifest}")
    public = corpus.get("public_prose", [])
    excluded = corpus.get("excluded", [])
    required(public and excluded, "corpus needs explicit public and excluded source policies")
    for record in public:
        required(all(record.get(key) for key in ("glob", "license", "provenance")), "public source needs glob/license/provenance")
    excluded_globs = {record.get("glob") for record in excluded}
    required(".fleet-coord/**" in excluded_globs and "**/.env*" in excluded_globs, "corpus must exclude coordination and secrets")
    package = corpus.get("package_contract", {})
    required(package.get("name") == "@hyprstream/docs", "package contract must name @hyprstream/docs")
    required(package.get("state") == "declared-not-yet-published", "package contract must not claim publication")

    allow = [record["glob"] for record in public]
    deny = [record["glob"] for record in excluded]
    for path in tracked(repo, "docs/**/*.md"):
        if any(fnmatch.fnmatch(path, pattern) for pattern in deny):
            continue
        required(any(fnmatch.fnmatch(path, pattern) for pattern in allow), f"unallowlisted public prose: {path}")


def validate(repo: Path, catalog: dict[str, Any] | None = None, corpus: dict[str, Any] | None = None,
             tracked_schemas: list[str] | None = None, consumers: dict[str, list[str]] | None = None) -> None:
    catalog = catalog or read_json(repo / "docs/schema-catalog.json")
    corpus = corpus or read_json(repo / "docs/corpus-sources.json")
    tracked_schemas = tracked_schemas if tracked_schemas is not None else tracked(repo, "*.capnp")
    consumers = consumers or source_services(repo)
    check_schema_catalog(catalog, repo, tracked_schemas, consumers)
    check_corpus(corpus, repo)
    contract = (repo / "docs/contracts/docs-pipeline.md").read_text(encoding="utf-8")
    required("docs/system-ontology.md" in contract, "pipeline contract must name ontology authority")


def expect_mutation_failure(name: str, repo: Path, catalog: dict[str, Any], corpus: dict[str, Any],
                            schemas: list[str], consumers: dict[str, list[str]]) -> None:
    try:
        validate(repo, catalog, corpus, schemas, consumers)
    except CatalogError:
        return
    raise AssertionError(f"mutation probe {name} unexpectedly passed")


def self_test(repo: Path) -> None:
    catalog = read_json(repo / "docs/schema-catalog.json")
    corpus = read_json(repo / "docs/corpus-sources.json")
    schemas = tracked(repo, "*.capnp")
    consumers = source_services(repo)
    validate(repo, catalog, corpus, schemas, consumers)

    unlisted = schemas + ["crates/example/schema/new.capnp"]
    expect_mutation_failure("unlisted schema", repo, copy.deepcopy(catalog), corpus, unlisted, consumers)
    stale = schemas[1:]
    expect_mutation_failure("stale removed schema", repo, copy.deepcopy(catalog), corpus, stale, consumers)
    changed = copy.deepcopy(consumers)
    changed["cli"] = changed["cli"] + ["settlement"]
    expect_mutation_failure("changed CLI consumer list", repo, copy.deepcopy(catalog), corpus, schemas, changed)
    duplicate = copy.deepcopy(catalog)
    duplicate["schemas"][1]["source_id"] = duplicate["schemas"][0]["source_id"]
    expect_mutation_failure("duplicate identity", repo, duplicate, corpus, schemas, consumers)
    unexplained = copy.deepcopy(catalog)
    unexplained["schemas"][0]["exclusions"].pop("cli")
    expect_mutation_failure("unexplained exclusion", repo, unexplained, corpus, schemas, consumers)
    print("docs catalog mutation probes: passed (5 expected failures)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run mutation probes after normal validation")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(args.repo.resolve())
        else:
            validate(args.repo.resolve())
            print("docs catalog: OK")
    except (CatalogError, AssertionError) as error:
        print(f"docs catalog: FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
