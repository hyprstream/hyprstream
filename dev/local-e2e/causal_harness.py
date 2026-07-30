#!/usr/bin/env python3
"""Offline, disposable contract and response harness.

This module intentionally contains no service/container/cloud activation path.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import uuid
from typing import Any, NoReturn


CORE_SERVICES = (
    ("hyprstream.container", "event"),
    ("hyprstream-policy.container", "policy"),
    ("hyprstream-discovery.container", "discovery"),
    ("hyprstream-registry.container", "registry"),
    ("hyprstream-streams.container", "streams"),
    ("hyprstream-model.container", "model"),
    ("hyprstream-oai.container", "oai"),
    ("hyprstream-oauth.container", "oauth"),
)
IMMUTABLE_IMAGE = re.compile(r"^[^@\s]+@sha256:[0-9a-f]{64}$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class HarnessError(RuntimeError):
    """A fail-closed harness validation error."""


def _fail(message: str) -> NoReturn:
    raise HarnessError(message)


def _lstat_regular(path: Path, description: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        _fail(f"{description} is missing: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail(f"{description} must be a regular non-symlink file: {path}")
    return info


def _lstat_directory(path: Path, description: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError:
        _fail(f"{description} is missing: {path}")
    if not stat.S_ISDIR(info.st_mode):
        _fail(f"{description} must be a real non-symlink directory: {path}")
    return info


def _reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            _fail(f"path component is missing: {current}")
        if stat.S_ISLNK(info.st_mode):
            _fail(f"symlink path component rejected: {current}")


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        _fail(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


class OwnedRun:
    """One disposable run with held ports and symlink-safe secrets."""

    def __init__(self, task_root: Path):
        self.task_root = task_root.absolute()
        self.run_root: Path | None = None
        self.run_id = uuid.uuid4().hex
        self.marker_nonce = uuid.uuid4().hex
        self.sockets: list[socket.socket] = []
        self.context: dict[str, Any] = {}
        self.cleanup_armed = False
        self.closed = False

    def __enter__(self) -> "OwnedRun":
        root_info = _lstat_directory(self.task_root, "task root")
        _reject_symlink_components(self.task_root)
        if root_info.st_uid != os.getuid():
            _fail("task root must be owned by the current uid")
        if stat.S_IMODE(root_info.st_mode) & 0o077:
            _fail("task root must not grant group/other permissions")

        # Cleanup is armed before mkdtemp performs the first run mutation.
        self.cleanup_armed = True
        atexit.register(self.close)
        try:
            self.run_root = Path(
                tempfile.mkdtemp(prefix=f"run-{self.run_id}-", dir=self.task_root)
            )
            os.chmod(self.run_root, 0o700)
            self._write_owned_file("owner.marker", self.marker_nonce.encode() + b"\n")

            xdg: dict[str, str] = {}
            for env_name, relative in (
                ("XDG_CONFIG_HOME", "xdg/config"),
                ("XDG_STATE_HOME", "xdg/state"),
                ("XDG_DATA_HOME", "xdg/data"),
                ("XDG_CACHE_HOME", "xdg/cache"),
                ("XDG_RUNTIME_DIR", "xdg/runtime"),
            ):
                directory = self.run_root / relative
                directory.mkdir(parents=True, mode=0o700)
                os.chmod(directory, 0o700)
                xdg[env_name] = os.fspath(directory)
            (self.run_root / "secrets").mkdir(mode=0o700)

            ports: dict[str, int] = {}
            units: dict[str, str] = {}
            for _, service in CORE_SERVICES:
                reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                reservation.bind(("127.0.0.1", 0))
                reservation.listen(1)
                self.sockets.append(reservation)
                ports[service] = reservation.getsockname()[1]
                units[service] = (
                    f"hyprstream-causal-{self.run_id}-{service}.service"
                )

            self.context = {
                "contract_version": "owned-run-v1",
                "run_id": self.run_id,
                "run_root": os.fspath(self.run_root),
                "xdg": xdg,
                "units": units,
                "held_loopback_tcp_ports": ports,
            }
            context_path = self.run_root / "context.json"
            self._write_owned_file(
                "context.json",
                (json.dumps(self.context, sort_keys=True) + "\n").encode(),
            )
            self.context["context_path"] = os.fspath(context_path)
            return self
        except BaseException:
            self._cleanup_incomplete()
            raise

    def _write_owned_file(self, relative: str, content: bytes) -> Path:
        if self.run_root is None:
            _fail("run root has not been created")
        if not SAFE_NAME.fullmatch(relative):
            _fail(f"unsafe owned-file name: {relative}")
        destination = self.run_root / relative
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(destination, flags, 0o600)
        try:
            os.write(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)
        if stat.S_IMODE(destination.lstat().st_mode) != 0o600:
            _fail(f"owned file mode is not 0600: {destination}")
        return destination

    def write_secret(self, name: str, content: bytes) -> Path:
        if self.run_root is None:
            _fail("run root has not been created")
        if not SAFE_NAME.fullmatch(name):
            _fail(f"unsafe secret name: {name}")
        secrets = self.run_root / "secrets"
        info = _lstat_directory(secrets, "secret directory")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            _fail("secret directory ownership/mode changed")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        destination = secrets / name
        fd = os.open(destination, flags, 0o600)
        try:
            os.write(fd, content)
            os.fsync(fd)
        finally:
            os.close(fd)
        if stat.S_IMODE(destination.lstat().st_mode) != 0o600:
            _fail("secret file mode is not 0600")
        return destination

    def child_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(self.context["xdg"])
        environment["HYPRSTREAM_CAUSAL_CONTEXT"] = self.context["context_path"]
        environment["HYPRSTREAM_CAUSAL_UNIT_PREFIX"] = (
            f"hyprstream-causal-{self.run_id}"
        )
        return environment

    def _cleanup_incomplete(self) -> None:
        for reservation in self.sockets:
            reservation.close()
        self.sockets.clear()
        if self.run_root is None:
            self.closed = True
            return
        try:
            info = self.run_root.lstat()
        except FileNotFoundError:
            self.closed = True
            return
        if not stat.S_ISDIR(info.st_mode):
            _fail("incomplete run root was replaced; refusing cleanup")
        if (
            self.run_root.parent != self.task_root
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _fail("incomplete run ownership changed; refusing cleanup")
        shutil.rmtree(self.run_root)
        self.closed = True

    def close(self) -> None:
        if self.closed or not self.cleanup_armed:
            return
        for reservation in self.sockets:
            reservation.close()
        self.sockets.clear()
        if self.run_root is not None:
            try:
                run_info = self.run_root.lstat()
            except FileNotFoundError:
                self.closed = True
                return
            if not stat.S_ISDIR(run_info.st_mode):
                _fail("run root was replaced; refusing cleanup")
            _reject_symlink_components(self.run_root)
            marker = self.run_root / "owner.marker"
            info = _lstat_regular(marker, "run ownership marker")
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
                _fail("run ownership marker ownership/mode changed; refusing cleanup")
            if marker.read_text(encoding="utf-8") != self.marker_nonce + "\n":
                _fail("run ownership marker content changed; refusing cleanup")
            if self.run_root.parent != self.task_root:
                _fail("run root escaped its task root; refusing cleanup")
            shutil.rmtree(self.run_root)
        self.closed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def validate_response_contract(
    contract: dict[str, Any], case_name: str, observed: dict[str, Any]
) -> None:
    if contract.get("contract_version") != "v1":
        _fail("response contract_version must be v1")
    try:
        expected = contract["cases"][case_name]
    except (KeyError, TypeError):
        _fail(f"response case is not declared: {case_name}")
    if set(observed) != {"class", "status", "body"}:
        _fail("observed response must contain exactly class, status, and body")
    for field in ("class", "status"):
        if observed[field] != expected[field]:
            _fail(
                f"{case_name}: expected {field}={expected[field]!r}, "
                f"observed {observed[field]!r}"
            )
    body_rule = expected["body"]
    body = observed["body"]
    if body_rule is None:
        if body is not None:
            _fail(f"{case_name}: transport/startup failure must not have a body")
        return
    if not isinstance(body, dict):
        _fail(f"{case_name}: body must be a JSON object")
    if "exact" in body_rule:
        if body != body_rule["exact"]:
            _fail(f"{case_name}: response body is not exact")
        return
    exact_keys = set(body_rule["exact_keys"])
    if set(body) != exact_keys:
        _fail(f"{case_name}: response body keys are not exact")
    for field, rule in body_rule["fields"].items():
        if "literal" in rule and body[field] != rule["literal"]:
            _fail(f"{case_name}: {field} literal mismatch")
        if rule.get("type") == "nonempty_string" and (
            not isinstance(body[field], str) or not body[field]
        ):
            _fail(f"{case_name}: {field} must be a non-empty string")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _reject_symlink_components(path)
    _lstat_regular(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"{description} is not valid JSON: {error}")
    if not isinstance(value, dict):
        _fail(f"{description} must be a JSON object")
    return value


def validate_inference_contract(contract: dict[str, Any]) -> str:
    if contract.get("contract_version") != "v1":
        _fail("inference contract_version must be v1")
    if contract.get("implementation_merge") != (
        "6440b151dcc0238a0f92d877f2052ce2271395d8"
    ):
        _fail("inference implementation merge is not the reviewed v1 revision")
    if contract.get("deployment_source_revision") != (
        "995fc622ae08b2baa1983646419ecccf4fe6a386"
    ):
        _fail("inference deployment source is not the reviewed v1 revision")
    contract_evidence = contract.get("contract_evidence")
    if contract_evidence != {
        "document_id": "inference-service-to-metal-runtime-contract-2026-07-26",
        "producer_prs": {
            "implementation": 1356,
            "publisher": 1358,
            "immutable_index": 1360,
        },
    }:
        _fail("inference contract evidence is not the reviewed v1 evidence")
    image = contract.get("image")
    if not isinstance(image, dict):
        _fail("inference image contract is missing")
    digest = image.get("index_digest")
    image_reference = f"{image.get('repository')}@{digest}"
    if not IMMUTABLE_IMAGE.fullmatch(image_reference):
        _fail("inference image must be an immutable digest reference")
    evidence = image.get("publisher_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "verified"
        or not isinstance(evidence.get("reference"), str)
        or not evidence["reference"].startswith("https://")
    ):
        _fail("inference publisher evidence must be verified")
    services = contract.get("services")
    if not isinstance(services, list) or len(services) != 2:
        _fail("inference contract must contain exactly two services")
    seen_ports: set[int] = set()
    seen_sockets: set[str] = set()
    for replica, service in enumerate(services):
        if not isinstance(service, dict):
            _fail("inference service entry must be an object")
        expected_name = f"inference-cpu-{replica}"
        if service.get("replica") != replica:
            _fail("inference replicas must be ordered 0, 1")
        for field in ("inference_service_id", "unit_name", "container_name"):
            if service.get(field) != expected_name:
                _fail(f"inference {field} must be {expected_name}")
        endpoint = service.get("listen_endpoint", {})
        if endpoint != {
            "transport": "udp",
            "address": "0.0.0.0",
            "port": 7440 + replica,
        }:
            _fail(f"inference replica {replica} listen endpoint is not exact")
        port = endpoint["port"]
        if port in seen_ports:
            _fail("inference listen ports must be unique")
        seen_ports.add(port)
        health = service.get("health_endpoint", {})
        expected_socket = f"{expected_name}.sock"
        if health != {
            "transport": "authenticated-inference-ipc",
            "socket": expected_socket,
        }:
            _fail(f"inference replica {replica} health endpoint is not exact")
        if expected_socket in seen_sockets:
            _fail("inference health sockets must be unique")
        seen_sockets.add(expected_socket)
        probe = service.get("direct_probe")
        if probe != {
            "authentication_scope": "query",
            "readiness_expectation": "isReady=true",
            "health_expectation": "modelLoaded=true,status=ok",
        }:
            _fail(f"inference replica {replica} direct probe schema is not exact")
    return image_reference


def verify_ingest_contract(
    source_root: Path,
    expected_sha: str,
    review_records: list[Path],
    render_dir: Path,
    inference_contract_path: Path,
) -> dict[str, Any]:
    if not FULL_SHA.fullmatch(expected_sha):
        _fail("expected ingest SHA must be a full lowercase 40-hex SHA")
    for path, description in (
        (source_root, "ingest source root"),
        (render_dir, "rendered Quadlet directory"),
    ):
        _reject_symlink_components(path)
        _lstat_directory(path, description)
    if len(review_records) < 2 or len(set(review_records)) != len(review_records):
        _fail("at least two distinct independent review records are required")
    for review_record in review_records:
        _reject_symlink_components(review_record)
        _lstat_regular(review_record, "independent review record")

    head = _run_git(source_root, "rev-parse", "HEAD")
    if head != expected_sha:
        _fail(f"ingest checkout head {head} does not match {expected_sha}")
    if _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail("ingest checkout must be completely clean")
    for review_record in review_records:
        review_lines = review_record.read_text(encoding="utf-8").splitlines()
        if not review_lines:
            _fail("independent review record is empty")
        first_line = review_lines[0]
        if first_line != f"PASS {expected_sha}":
            _fail(
                "every independent review first line must be exactly "
                "PASS <expected-sha>"
            )

    image_references: set[str] = set()
    rendered_services: list[str] = []
    for filename, service in CORE_SERVICES:
        path = render_dir / filename
        _reject_symlink_components(path)
        _lstat_regular(path, f"rendered {service} Quadlet")
        text = path.read_text(encoding="utf-8")
        if re.search(r"\$\{[^}]+\}|@@[A-Z_]+@@", text):
            _fail(f"rendered {service} Quadlet contains an unresolved placeholder")
        exact_lines = set(text.splitlines())
        required = {
            f"ContainerName=hyprstream-{service}",
            "Volume=hyprstream-ipc.volume:/run/hyprstream",
            f"Exec=service start {service} --foreground --ipc",
        }
        missing = sorted(required - exact_lines)
        if missing:
            _fail(f"rendered {service} Quadlet is missing exact lines: {missing}")
        image_lines = [
            line.removeprefix("Image=")
            for line in text.splitlines()
            if line.startswith("Image=")
        ]
        if len(image_lines) != 1 or not IMMUTABLE_IMAGE.fullmatch(image_lines[0]):
            _fail(f"rendered {service} Quadlet must name one immutable image")
        image_references.add(image_lines[0])
        rendered_services.append(service)
    if len(image_references) != 1:
        _fail("all rendered core services must use one immutable image")

    volume = render_dir / "hyprstream-ipc.volume"
    _reject_symlink_components(volume)
    _lstat_regular(volume, "rendered IPC volume")
    if "[Volume]" not in volume.read_text(encoding="utf-8").splitlines():
        _fail("rendered IPC volume is not a Quadlet volume")

    inference_contract = _read_json(
        inference_contract_path, "versioned inference contract"
    )
    inference_image = validate_inference_contract(inference_contract)
    core_image = next(iter(image_references))
    if inference_image != core_image:
        _fail("core and inference contracts must pin the same immutable image")
    return {
        "contract_version": "ingest-render-v1",
        "source_sha": expected_sha,
        "core_services": rendered_services,
        "core_image": core_image,
        "inference_services": ["inference-cpu-0", "inference-cpu-1"],
    }


def _owned_run_command(args: argparse.Namespace) -> int:
    task_root = Path(args.task_root)
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        _fail("owned-run requires a command after --")
    with OwnedRun(task_root) as run:
        result = subprocess.run(command, env=run.child_environment(), check=False)
        return result.returncode


def _verify_ingest_command(args: argparse.Namespace) -> int:
    result = verify_ingest_contract(
        Path(args.source_root),
        args.expected_sha,
        [Path(path) for path in args.review_record],
        Path(args.render_dir),
        Path(args.inference_contract),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _assert_response_command(args: argparse.Namespace) -> int:
    contract = _read_json(Path(args.contract), "response contract")
    observed = _read_json(Path(args.observed), "observed response")
    validate_response_contract(contract, args.case, observed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    owned = subparsers.add_parser("owned-run")
    owned.add_argument("--task-root", required=True)
    owned.add_argument("command", nargs=argparse.REMAINDER)
    owned.set_defaults(function=_owned_run_command)

    ingest = subparsers.add_parser("verify-ingest")
    ingest.add_argument("--source-root", required=True)
    ingest.add_argument("--expected-sha", required=True)
    ingest.add_argument("--review-record", required=True, action="append")
    ingest.add_argument("--render-dir", required=True)
    ingest.add_argument("--inference-contract", required=True)
    ingest.set_defaults(function=_verify_ingest_command)

    response = subparsers.add_parser("assert-response")
    response.add_argument("--contract", required=True)
    response.add_argument("--case", required=True)
    response.add_argument("--observed", required=True)
    response.set_defaults(function=_assert_response_command)
    return parser


def main() -> int:
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    try:
        args = build_parser().parse_args()
        return args.function(args)
    except HarnessError as error:
        print(f"causal-harness: ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
