#!/usr/bin/env python3
"""Validate #1059 local standards registries and generated RFCXML artifacts.

Only Python's standard library is used.  `--check-generated` invokes the exact
xml2rfc command supplied by the caller and compares temporary output bytewise.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STANDARDS = ROOT / "docs" / "standards"
REGISTRY = STANDARDS / "registry" / "domain-separation.json"
VOCABULARY = STANDARDS / "registry" / "profile-vocabulary.json"
MANIFEST = STANDARDS / "registry" / "obligations.json"
XML = STANDARDS / "rfc" / "draft-hyprstream-privacypass-pqhybrid-00.xml"
TEXT = XML.with_suffix(".txt")
HTML = XML.with_suffix(".html")
XML2RFC_VERSION = "3.34.0"


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be an object")
    return data


def validate_registry() -> list[str]:
    errors: list[str] = []
    data = load_json(REGISTRY)
    if data.get("registry_version") != 1:
        errors.append("registry_version must be 1")
    if "not an IANA" not in data.get("scope", ""):
        errors.append("registry scope must explicitly say it is not an IANA registry")
    canonical = data.get("canonicalization", {})
    for key in ("encoding", "text", "integers", "bytes", "digest"):
        if not canonical.get(key):
            errors.append(f"canonicalization.{key} is required")
    fields = data.get("fields", {})
    labels = data.get("labels", [])
    if not isinstance(fields, dict) or not fields:
        return errors + ["fields must be a nonempty object"]
    if not isinstance(labels, list) or not labels:
        return errors + ["labels must be a nonempty list"]
    seen: set[str] = set()
    allowed_types = {"text", "bytes", "uint"}
    for name, spec in fields.items():
        if not isinstance(spec, dict):
            errors.append(f"field {name}: definition must be an object")
            continue
        if spec.get("type") not in allowed_types:
            errors.append(f"field {name}: unsupported or missing type")
        if not spec.get("canonical"):
            errors.append(f"field {name}: canonical form is required")
        if spec.get("type") == "uint":
            minimum, maximum = spec.get("minimum"), spec.get("maximum")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in (minimum, maximum)):
                errors.append(f"field {name}: uint requires integer minimum/maximum")
            elif minimum < 0 or minimum > maximum:
                errors.append(f"field {name}: uint bounds must satisfy 0 <= minimum <= maximum")
        else:
            minimum, maximum = spec.get("min_bytes"), spec.get("max_bytes")
            if any(not isinstance(value, int) or isinstance(value, bool) for value in (minimum, maximum)):
                errors.append(f"field {name}: byte/text requires integer min_bytes/max_bytes")
            elif minimum < 0 or minimum > maximum:
                errors.append(f"field {name}: byte bounds must satisfy 0 <= min_bytes <= max_bytes")
    for label in labels:
        if not isinstance(label, dict):
            errors.append("each label definition must be an object")
            continue
        label_id = label.get("id")
        if not isinstance(label_id, str) or not re.fullmatch(r"hs-pp-pqhybrid/[a-z0-9-]+/v1", label_id):
            errors.append(f"invalid label id: {label_id!r}")
        elif label_id in seen:
            errors.append(f"duplicate label: {label_id}")
        else:
            seen.add(label_id)
        for key in ("purpose", "owner_issue", "status"):
            if not label.get(key):
                errors.append(f"label {label_id}: {key} is required")
        used = label.get("fields")
        if not isinstance(used, list) or not used:
            errors.append(f"label {label_id}: nonempty fields list is required")
        else:
            if len(used) != len(set(used)):
                errors.append(f"label {label_id}: fields must not contain duplicates")
            for field in used:
                if field not in fields:
                    errors.append(f"label {label_id}: unknown field {field}")
    return errors


def validate_vocabulary() -> list[str]:
    errors: list[str] = []
    data = load_json(VOCABULARY)
    if data.get("vocabulary_version") != 1:
        errors.append("vocabulary_version must be 1")
    scope = data.get("scope", "")
    for required in ("not a wire schema", "codepoint allocation"):
        if required not in scope:
            errors.append(f"vocabulary scope must state {required!r}")
    for key in ("roles", "principal_kinds", "carrier_profiles"):
        values = data.get(key)
        if (not isinstance(values, list) or not values or
                any(not isinstance(value, str) or not value for value in values) or
                len(values) != len(set(values))):
            errors.append(f"{key} must be a nonempty list of unique nonempty strings")
    control = data.get("inner_control", {})
    if not isinstance(control, dict):
        return errors + ["inner_control must be an object"]
    if not re.fullmatch(r"[a-z0-9-]+-v1", control.get("logical_namespace", "")):
        errors.append("inner_control.logical_namespace must be a local v1 label")
    if control.get("unknown_message_kind") != "reject":
        errors.append("inner_control.unknown_message_kind must be reject")
    kinds = control.get("message_kinds")
    if (not isinstance(kinds, list) or not kinds or
            any(not isinstance(value, str) or not value for value in kinds) or
            len(kinds) != len(set(kinds))):
        errors.append("inner_control.message_kinds must be a nonempty list of unique nonempty strings")
    for key in ("max_object_payload_bytes", "max_exchange_objects", "max_concurrent_exchanges_per_session"):
        value = control.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"inner_control.{key} must be a positive integer")
    states = data.get("states", {})
    if not isinstance(states, dict):
        return errors + ["states must be an object"]
    for role in ("client", "issuer", "origin"):
        values = states.get(role)
        if (not isinstance(values, list) or not values or
                any(not isinstance(value, str) or not value for value in values) or
                len(values) != len(set(values))):
            errors.append(f"states.{role} must be a nonempty list of unique nonempty strings")
    invariants = data.get("transition_invariants")
    if not isinstance(invariants, list) or not invariants:
        errors.append("transition_invariants must be a nonempty list")
    return errors


def normative_sentences(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if re.search(r"\bMUST(?: NOT)?\b", sentence)
    ]


def section_paragraphs(section: ET.Element):
    """Yield paragraphs in one section without descending into subsections."""
    for child in section:
        if child.tag == "section":
            continue
        if child.tag == "t":
            yield child
        yield from section_paragraphs(child)


def xml_normative_statements() -> list[tuple[str, str]]:
    try:
        root = ET.parse(XML).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{XML}: invalid XML: {exc}") from exc
    statements: list[tuple[str, str]] = []
    for section in root.findall("./middle//section"):
        anchor = section.get("anchor", "")
        if not anchor:
            continue
        for paragraph in section_paragraphs(section):
            if paragraph.get("anchor") == "requirements-language":
                continue
            text = " ".join(part.strip() for part in paragraph.itertext() if part.strip())
            statements.extend((statement, anchor) for statement in normative_sentences(text))
    return statements


def validate_manifest() -> list[str]:
    errors: list[str] = []
    data = load_json(MANIFEST)
    obligations = data.get("obligations", [])
    if not obligations:
        return ["obligation manifest must contain obligations"]
    statements: set[str] = set()
    statement_sources: dict[str, str] = {}
    ids: set[str] = set()
    for item in obligations:
        obligation_id = item.get("id")
        if not isinstance(obligation_id, str) or not re.fullmatch(r"PPQH-REQ-\d{3}", obligation_id):
            errors.append(f"invalid obligation id: {obligation_id!r}")
        elif obligation_id in ids:
            errors.append(f"duplicate obligation id: {obligation_id}")
        else:
            ids.add(obligation_id)
        statement = item.get("statement")
        if not isinstance(statement, str) or not re.search(r"\bMUST(?: NOT)?\b", statement):
            errors.append(f"{obligation_id}: exact normative statement is required")
        else:
            expected_keyword = "MUST NOT" if "MUST NOT" in statement else "MUST"
            if item.get("keyword") != expected_keyword:
                errors.append(f"{obligation_id}: keyword must be {expected_keyword}")
            normalized = re.sub(r"\s+", " ", statement).strip()
            if normalized in statements:
                errors.append(f"{obligation_id}: duplicate normative statement")
            statements.add(normalized)
            source = item.get("source")
            if not isinstance(source, str) or not source:
                errors.append(f"{obligation_id}: source section anchor is required")
            else:
                statement_sources[normalized] = source
        if item.get("kind") == "implemented-test":
            if not item.get("test"):
                errors.append(f"{obligation_id}: implemented-test requires test")
        elif item.get("kind") == "specification-only":
            if not item.get("owner") or not item.get("blocker") or not item.get("planned_test"):
                errors.append(f"{obligation_id}: specification-only requires owner, blocker, and planned_test")
        else:
            errors.append(f"{obligation_id}: kind must be implemented-test or specification-only")
    source_pairs = xml_normative_statements()
    source_statements = {statement for statement, _ in source_pairs}
    untracked = [statement for statement, _ in source_pairs if statement not in statements]
    stale = [statement for statement in statements if statement not in source_statements]
    for statement in untracked:
        errors.append(f"unmapped RFCXML normative statement: {statement}")
    for statement in stale:
        errors.append(f"manifest statement not found in RFCXML: {statement}")
    for statement, anchor in source_pairs:
        if statement_sources.get(statement) not in (None, anchor):
            errors.append(
                f"manifest source for {statement!r} must be {anchor!r}, "
                f"got {statement_sources[statement]!r}"
            )
    return errors


def check_generated(xml2rfc: str) -> list[str]:
    errors: list[str] = []
    if not TEXT.exists() or not HTML.exists():
        return ["generated RFCXML text and HTML outputs are missing"]
    version = subprocess.run([xml2rfc, "--version"], cwd=ROOT, text=True, capture_output=True)
    if version.returncode:
        return [f"xml2rfc --version failed: {version.stdout}{version.stderr}"]
    reported = f"{version.stdout}{version.stderr}".strip()
    if reported != f"xml2rfc {XML2RFC_VERSION}":
        return [f"xml2rfc must be exactly {XML2RFC_VERSION}; got {reported!r}"]
    with tempfile.TemporaryDirectory(prefix="hyprstream-rfcxml-") as tmp:
        tmp_path = Path(tmp)
        command = [xml2rfc, "--no-network", "--text", "--html", "--path", str(tmp_path), str(XML)]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            return [f"xml2rfc failed: {' '.join(command)}\n{result.stdout}{result.stderr}"]
        for suffix, expected in ((".txt", TEXT), (".html", HTML)):
            actual = tmp_path / f"{XML.stem}{suffix}"
            if not actual.exists():
                errors.append(f"xml2rfc did not create {actual.name}")
            elif actual.read_bytes() != expected.read_bytes():
                errors.append(f"stale generated output: {expected.relative_to(ROOT)}; run tools/generate_standards.py")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-generated", action="store_true", help="regenerate RFCXML in a temp directory and compare bytewise")
    parser.add_argument("--xml2rfc", default="xml2rfc", help="exact xml2rfc executable to invoke")
    args = parser.parse_args()
    errors = validate_registry() + validate_vocabulary() + validate_manifest()
    if args.check_generated:
        errors += check_generated(args.xml2rfc)
    if errors:
        print("standards validation failed:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("standards registries, vocabulary, and obligation manifest: OK")
    if args.check_generated:
        print("RFCXML generated text and HTML: current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
