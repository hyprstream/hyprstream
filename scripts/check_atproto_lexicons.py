#!/usr/bin/env python3
"""Vendor a pinned upstream Lexicon closure, or verify it entirely offline.

This checks source integrity and reference closure, not record validity or auth.
Only --refresh uses the network; normal CI verification never fetches a schema.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "https://raw.githubusercontent.com/bluesky-social/atproto/"
MAX_BYTES = 1024 * 1024
MAX_SCHEMAS = 512
LICENSES = ("LICENSE.txt", "LICENSE-MIT.txt", "LICENSE-APACHE.txt")


def schema_path(nsid):
    # Deliberately limited to the selected upstream namespaces. Never a URL.
    if not re.fullmatch(r"(?:com\.atproto|app\.bsky)(?:\.[A-Za-z][A-Za-z0-9]*)+", nsid):
        raise ValueError(f"unsupported upstream NSID: {nsid!r}")
    return nsid.replace(".", "/") + ".json"


def references(value):
    if isinstance(value, dict):
        if value.get("type") == "ref":
            yield value["ref"]
        if value.get("type") == "union":
            yield from value["refs"]
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def target(nsid, ref):
    if not isinstance(ref, str):
        raise ValueError("reference must be a string")
    parts = ref.split("#")
    if len(parts) > 2 or (len(parts) == 2 and not parts[1]):
        raise ValueError(f"invalid reference: {ref!r}")
    owner = parts[0] or nsid
    schema_path(owner)
    return owner, parts[1] if len(parts) == 2 else "main"


def decode(nsid, raw):
    if len(raw) > MAX_BYTES:
        raise ValueError(f"schema too large: {nsid}")
    doc = json.loads(raw)
    if doc.get("id") != nsid or doc.get("lexicon") != 1 or not doc.get("defs"):
        raise ValueError(f"invalid schema identity/version/definitions: {nsid}")
    return doc


def closure(roots, documents):
    pending = list(roots)
    seen = set()
    while pending:
        nsid = pending.pop()
        schema_path(nsid)
        if nsid in seen:
            continue
        if nsid not in documents:
            raise ValueError(f"missing schema: {nsid}")
        seen.add(nsid)
        for ref in references(documents[nsid]):
            owner, name = target(nsid, ref)
            if owner not in documents or name not in documents[owner]["defs"]:
                raise ValueError(f"unresolved reference: {nsid} -> {ref}")
            pending.append(owner)
    return seen


def fetch(commit, path):
    url = UPSTREAM + commit + "/" + path
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.url != url:
                raise ValueError("unexpected upstream redirect")
            raw = response.read(MAX_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"fetch failed for {path}: {exc}") from exc
    if len(raw) > MAX_BYTES:
        raise ValueError(f"upstream file too large: {path}")
    return raw


def refresh(selection, directory):
    commit = selection["upstream_commit"]
    raw_schemas = {}
    documents = {}
    pending = set(selection["roots"])
    while pending:
        if len(documents) + len(pending) > MAX_SCHEMAS:
            raise ValueError("schema closure exceeds limit")
        batch = sorted(pending)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda n: fetch(commit, "lexicons/" + schema_path(n)), batch))
        for nsid, raw in zip(batch, results):
            raw_schemas[nsid] = raw
            documents[nsid] = decode(nsid, raw)
        pending = {target(nsid, ref)[0] for nsid in batch
                   for ref in references(documents[nsid])} - documents.keys()
    closure(selection["roots"], documents)
    licenses = {name: fetch(commit, name) for name in LICENSES}
    manifest = {"upstream_commit": commit, "roots": selection["roots"],
                "files": {schema_path(n): hashlib.sha256(raw).hexdigest()
                          for n, raw in sorted(raw_schemas.items())},
                "licenses": {name: hashlib.sha256(raw).hexdigest()
                             for name, raw in licenses.items()}}
    # Fetch and validate everything before writing. Refuse stale/untracked files
    # instead of deleting anything when the selected closure shrinks.
    expected = set(manifest["files"]) | set(LICENSES) | {"manifest.json"}
    existing = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if existing - expected:
        raise ValueError(f"unexpected existing files: {sorted(existing - expected)}")
    for nsid, raw in raw_schemas.items():
        path = directory / schema_path(nsid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    for name, raw in licenses.items():
        (directory / name).write_bytes(raw)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def verify(selection, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    for key in ("upstream_commit", "roots"):
        if manifest[key] != selection[key]:
            raise ValueError(f"selection/manifest mismatch: {key}")
    files = manifest["files"]
    expected = set(files) | set(LICENSES) | {"manifest.json"}
    actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError(f"bundle file mismatch: {sorted(actual ^ expected)}")
    documents = {}
    for path, digest in files.items():
        nsid = path.removesuffix(".json").replace("/", ".")
        if schema_path(nsid) != path:
            raise ValueError(f"invalid manifest path: {path}")
        raw = (directory / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"schema digest mismatch: {path}")
        documents[nsid] = decode(nsid, raw)
    if set(manifest["licenses"]) != set(LICENSES):
        raise ValueError("license manifest mismatch")
    for name, digest in manifest["licenses"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"license digest mismatch: {name}")
    if closure(selection["roots"], documents) != documents.keys():
        raise ValueError("bundle contains schemas outside selected reference closure")
    return len(documents)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch the explicitly pinned upstream commit")
    args = parser.parse_args()
    selection = json.loads((ROOT / "lexicons/atproto-selection.json").read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", selection["upstream_commit"]):
        raise ValueError("upstream commit must be a full immutable SHA")
    if not selection["roots"] or len(set(selection["roots"])) != len(selection["roots"]):
        raise ValueError("roots must be nonempty and unique")
    directory = ROOT / "lexicons/upstream/atproto"
    if args.refresh:
        refresh(selection, directory)
    print(f"Verified {verify(selection, directory)} pinned upstream Lexicons (offline verification).")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f"ATProto Lexicon verification failed: {exc}", file=sys.stderr)
        sys.exit(1)
