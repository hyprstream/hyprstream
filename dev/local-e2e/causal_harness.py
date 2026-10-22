#!/usr/bin/env python3
"""Offline, disposable contract and response harness.

This module intentionally contains no service/container/cloud activation path.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import re
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
REVIEW_IDENTITY = re.compile(r"^[a-z0-9][a-z0-9_.:/-]{0,127}$")
INFERENCE_CONTRACT_RELATIVE = Path(
    "modules/hyprstream-instance/fixtures/inference-contract/draft-v1.json"
)
QUADLET_TEMPLATE_RELATIVE = Path("stacks/hyprstream-staging/templates")
REPEATABLE_QUADLET_KEYS = frozenset({"Environment", "Volume"})
QUADLET_DESCRIPTIONS = {
    "event": "HyprStream event bus (PUB/SUB)",
    "policy": "HyprStream authorization (Casbin)",
    "discovery": "HyprStream discovery",
    "registry": "HyprStream model registry",
    "streams": "HyprStream token streaming",
    "model": "HyprStream model inference (CPU, synthetic staging fixture)",
    "oai": "HyprStream OpenAI-compatible API",
    "oauth": "HyprStream OAuth and DID document listener",
}
QUADLET_DEPENDENCIES = {
    "event": ("network.target", None),
    "policy": ("hyprstream-event.service", "hyprstream-event.service"),
    "discovery": ("hyprstream-policy.service", "hyprstream-policy.service"),
    "registry": ("hyprstream-event.service", "hyprstream-event.service"),
    "streams": ("hyprstream-event.service", "hyprstream-event.service"),
    "model": (
        "hyprstream-registry.service hyprstream-policy.service",
        "hyprstream-registry.service hyprstream-policy.service",
    ),
    "oai": ("hyprstream-policy.service", "hyprstream-policy.service"),
    "oauth": (
        "hyprstream-policy.service hyprstream-discovery.service",
        "hyprstream-policy.service hyprstream-discovery.service",
    ),
}


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
        self.task_fd: int | None = None
        self.run_fd: int | None = None
        self.task_identity: tuple[int, int] | None = None
        self.run_identity: tuple[int, int] | None = None
        self.marker_identity: tuple[int, int] | None = None

    def __enter__(self) -> "OwnedRun":
        root_info = _lstat_directory(self.task_root, "task root")
        _reject_symlink_components(self.task_root)
        if root_info.st_uid != os.getuid():
            _fail("task root must be owned by the current uid")
        if stat.S_IMODE(root_info.st_mode) & 0o077:
            _fail("task root must not grant group/other permissions")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        self.task_fd = os.open(self.task_root, directory_flags)
        task_fd_info = os.fstat(self.task_fd)
        self.task_identity = (task_fd_info.st_dev, task_fd_info.st_ino)
        if self.task_identity != (root_info.st_dev, root_info.st_ino):
            self._close_cleanup_fds()
            _fail("task root identity changed while opening it")

        # Cleanup is armed before mkdtemp performs the first run mutation.
        self.cleanup_armed = True
        atexit.register(self.close)
        try:
            self.run_root = Path(
                tempfile.mkdtemp(prefix=f"run-{self.run_id}-", dir=self.task_root)
            )
            os.chmod(self.run_root, 0o700)
            self.run_fd = os.open(
                self.run_root.name, directory_flags, dir_fd=self.task_fd
            )
            run_info = os.fstat(self.run_fd)
            self.run_identity = (run_info.st_dev, run_info.st_ino)
            self._write_owned_file("owner.marker", self.marker_nonce.encode() + b"\n")
            marker_info = os.stat(
                "owner.marker", dir_fd=self.run_fd, follow_symlinks=False
            )
            self.marker_identity = (marker_info.st_dev, marker_info.st_ino)

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
        if self.run_root is None or self.run_fd is None:
            _fail("run root has not been created")
        if not SAFE_NAME.fullmatch(relative):
            _fail(f"unsafe owned-file name: {relative}")
        destination = self.run_root / relative
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(relative, flags, 0o600, dir_fd=self.run_fd)
        try:
            os.write(fd, content)
            os.fsync(fd)
            info = os.fstat(fd)
        finally:
            os.close(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            _fail(f"owned file mode is not 0600: {destination}")
        return destination

    def write_secret(self, name: str, content: bytes) -> Path:
        if self.run_root is None or self.run_fd is None:
            _fail("run root has not been created")
        if not SAFE_NAME.fullmatch(name):
            _fail(f"unsafe secret name: {name}")
        secrets = self.run_root / "secrets"
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        secrets_fd = os.open("secrets", directory_flags, dir_fd=self.run_fd)
        info = os.fstat(secrets_fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            os.close(secrets_fd)
            _fail("secret directory ownership/mode changed")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        destination = secrets / name
        try:
            fd = os.open(name, flags, 0o600, dir_fd=secrets_fd)
            try:
                os.write(fd, content)
                os.fsync(fd)
                secret_info = os.fstat(fd)
            finally:
                os.close(fd)
        finally:
            os.close(secrets_fd)
        if (
            not stat.S_ISREG(secret_info.st_mode)
            or stat.S_IMODE(secret_info.st_mode) != 0o600
        ):
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
        self._cleanup_owned_run("incomplete")

    def close(self) -> None:
        if self.closed or not self.cleanup_armed:
            return
        self._cleanup_owned_run("completed")

    @staticmethod
    def _remove_tree_at(directory_fd: int) -> None:
        """Remove only entries reachable through an already verified directory fd."""
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        for name in os.listdir(directory_fd):
            info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                try:
                    child_info = os.fstat(child_fd)
                    if (child_info.st_dev, child_info.st_ino) != (
                        info.st_dev,
                        info.st_ino,
                    ):
                        _fail("run child identity changed; refusing cleanup")
                    OwnedRun._remove_tree_at(child_fd)
                    current = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    if (current.st_dev, current.st_ino) != (
                        info.st_dev,
                        info.st_ino,
                    ):
                        _fail("run child was substituted; refusing cleanup")
                    os.rmdir(name, dir_fd=directory_fd)
                finally:
                    os.close(child_fd)
            else:
                os.unlink(name, dir_fd=directory_fd)

    def _close_cleanup_fds(self) -> None:
        if self.run_fd is not None:
            os.close(self.run_fd)
            self.run_fd = None
        if self.task_fd is not None:
            os.close(self.task_fd)
            self.task_fd = None

    def _cleanup_owned_run(self, phase: str) -> None:
        for reservation in self.sockets:
            reservation.close()
        self.sockets.clear()
        if self.run_root is None:
            self._close_cleanup_fds()
            self.closed = True
            return
        try:
            if (
                self.task_fd is None
                or self.run_fd is None
                or self.task_identity is None
                or self.run_identity is None
                or self.marker_identity is None
            ):
                _fail(f"{phase} cleanup lacks captured ownership identity")
            _reject_symlink_components(self.run_root)
            try:
                current_task = self.task_root.lstat()
            except FileNotFoundError:
                _fail(f"{phase} task root is missing; refusing cleanup")
            task_fd_info = os.fstat(self.task_fd)
            if (
                (current_task.st_dev, current_task.st_ino) != self.task_identity
                or (task_fd_info.st_dev, task_fd_info.st_ino)
                != self.task_identity
            ):
                _fail(f"{phase} task-root identity changed; refusing cleanup")
            if self.run_root.parent != self.task_root:
                _fail(f"{phase} run root escaped its task root; refusing cleanup")
            try:
                run_info = os.stat(
                    self.run_root.name,
                    dir_fd=self.task_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                _fail(f"{phase} run root is missing; refusing cleanup")
            run_fd_info = os.fstat(self.run_fd)
            if not stat.S_ISDIR(run_info.st_mode) or (
                run_info.st_dev,
                run_info.st_ino,
            ) != self.run_identity:
                _fail(f"{phase} run root was substituted; refusing cleanup")
            if (run_fd_info.st_dev, run_fd_info.st_ino) != self.run_identity:
                _fail(f"{phase} run directory fd identity changed")
            if (
                run_info.st_uid != os.getuid()
                or stat.S_IMODE(run_info.st_mode) != 0o700
            ):
                _fail(f"{phase} run ownership changed; refusing cleanup")
            try:
                marker_info = os.stat(
                    "owner.marker", dir_fd=self.run_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                _fail(f"{phase} ownership marker is missing; refusing cleanup")
            if not stat.S_ISREG(marker_info.st_mode) or (
                marker_info.st_dev,
                marker_info.st_ino,
            ) != self.marker_identity:
                _fail(f"{phase} ownership marker changed; refusing cleanup")
            if (
                marker_info.st_uid != os.getuid()
                or stat.S_IMODE(marker_info.st_mode) != 0o600
            ):
                _fail("run ownership marker ownership/mode changed; refusing cleanup")
            marker_fd = os.open(
                "owner.marker",
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self.run_fd,
            )
            try:
                marker_content = os.read(marker_fd, 4097)
                if os.read(marker_fd, 1):
                    _fail("run ownership marker is too large; refusing cleanup")
            finally:
                os.close(marker_fd)
            if marker_content != (self.marker_nonce + "\n").encode():
                _fail("run ownership marker content changed; refusing cleanup")
            self._remove_tree_at(self.run_fd)
            current_run = os.stat(
                self.run_root.name,
                dir_fd=self.task_fd,
                follow_symlinks=False,
            )
            if (current_run.st_dev, current_run.st_ino) != self.run_identity:
                _fail(f"{phase} run root changed during cleanup; refusing removal")
            os.rmdir(self.run_root.name, dir_fd=self.task_fd)
            self.closed = True
        finally:
            self._close_cleanup_fds()
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


def _canonical_inference_contract() -> dict[str, Any]:
    return _read_json(
        Path(__file__).with_name("fixtures") / "ingest-inference-contract-v1.json",
        "bundled reviewed ingest inference contract",
    )


def validate_inference_contract(contract: dict[str, Any]) -> str:
    canonical = _canonical_inference_contract()
    if set(contract) != set(canonical):
        _fail("inference top-level key set is not the exact reviewed v1 schema")
    if contract != canonical:
        _fail("inference contract values are not the exact reviewed v1 artifact")
    services = contract["services"]
    if not isinstance(services, list) or len(services) != 2:
        _fail("inference contract must contain exactly two services")
    expected_service_keys = set(canonical["services"][0])
    for replica, service in enumerate(services):
        if not isinstance(service, dict) or set(service) != expected_service_keys:
            _fail(
                f"inference replica {replica} key set is not the exact reviewed schema"
            )
        if service != canonical["services"][replica]:
            _fail(f"inference replica {replica} values are not exact")
    image = contract["image"]
    image_reference = f"{image['repository']}@{image['index_digest']}"
    if not IMMUTABLE_IMAGE.fullmatch(image_reference):
        _fail("inference image must be an immutable digest reference")
    return image_reference


def _read_bytes(path: Path, description: str) -> bytes:
    _reject_symlink_components(path)
    _lstat_regular(path, description)
    try:
        return path.read_bytes()
    except OSError as error:
        _fail(f"could not read {description}: {error}")


def _parse_review_record(path: Path, expected_sha: str) -> tuple[str, str, bytes]:
    content = _read_bytes(path, "independent review record")
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        _fail(f"independent review record is not UTF-8: {error}")
    expected_prefix = (
        f"PASS {expected_sha}",
        "Review-Schema: independent-review-v1",
    )
    if tuple(lines[:2]) != expected_prefix or len(lines) < 4:
        _fail(
            "every independent review must begin with exact PASS SHA and "
            "independent-review-v1 schema"
        )
    fields: dict[str, str] = {}
    for line in lines[2:4]:
        if ": " not in line:
            _fail("review identity header is malformed")
        key, value = line.split(": ", 1)
        if key in fields:
            _fail("review identity header is duplicated")
        fields[key] = value
    if set(fields) != {"Reviewer-Identity", "Review-Seat"}:
        _fail("review identity header keys are not exact")
    reviewer = fields["Reviewer-Identity"]
    seat = fields["Review-Seat"]
    if not REVIEW_IDENTITY.fullmatch(reviewer) or not REVIEW_IDENTITY.fullmatch(seat):
        _fail("reviewer and seat identities must be structured safe identifiers")
    return reviewer, seat, hashlib.sha256(content).digest()


def _parse_quadlet(
    text: str, description: str
) -> list[tuple[str, list[tuple[str, str]]]]:
    sections: list[tuple[str, list[tuple[str, str]]]] = []
    seen_sections: set[str] = set()
    current: list[tuple[str, str]] | None = None
    seen_keys: set[str] = set()
    repeated_values: dict[str, set[str]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        section_match = re.fullmatch(r"\[([A-Za-z][A-Za-z0-9]*)\]", line)
        if section_match:
            section = section_match.group(1)
            if section in seen_sections:
                _fail(f"{description} has duplicate [{section}] section")
            seen_sections.add(section)
            current = []
            sections.append((section, current))
            seen_keys = set()
            repeated_values = {}
            continue
        if current is None or "=" not in line:
            _fail(f"{description}:{line_number} is not a canonical directive")
        key, value = line.split("=", 1)
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key) or not value:
            _fail(f"{description}:{line_number} has an invalid key/value")
        if key in seen_keys and key not in REPEATABLE_QUADLET_KEYS:
            _fail(f"{description} has duplicate {key}= directive")
        if key in REPEATABLE_QUADLET_KEYS:
            values = repeated_values.setdefault(key, set())
            if value in values:
                _fail(f"{description} repeats an identical {key}= directive")
            values.add(value)
        seen_keys.add(key)
        current.append((key, value))
    return sections


def _quadlet_fields(
    parsed: list[tuple[str, list[tuple[str, str]]]], section_name: str
) -> list[tuple[str, str]]:
    for name, fields in parsed:
        if name == section_name:
            return fields
    _fail(f"Quadlet is missing [{section_name}]")


def _render_source_template(
    template: str, values: dict[str, str], description: str
) -> str:
    placeholders = set(re.findall(r"\$\{([a-z0-9_]+)\}", template))
    if placeholders != set(values):
        _fail(
            f"{description} source template placeholders are not exact: "
            f"{sorted(placeholders)}"
        )
    rendered = template
    for key, value in values.items():
        if "\n" in value or "\r" in value or "${" in value:
            _fail(f"{description} render value for {key} is unsafe")
        rendered = rendered.replace(f"${{{key}}}", value)
    return rendered


def _expected_quadlet_sections(
    service: str,
    image: str,
    extra_environment: list[str],
) -> list[tuple[str, list[tuple[str, str]]]]:
    after, requires = QUADLET_DEPENDENCIES[service]
    unit = [("Description", QUADLET_DESCRIPTIONS[service]), ("After", after)]
    if requires is not None:
        unit.append(("Requires", requires))
    container = [
        ("Image", image),
        ("ContainerName", f"hyprstream-{service}"),
        ("Network", "host"),
        ("Volume", "hyprstream-ipc.volume:/run/hyprstream"),
        ("Volume", "/var/lib/hyprstream:/var/lib/hyprstream:z"),
        ("EnvironmentFile", "/var/lib/hyprstream/hyprstream.env"),
        *(("Environment", value) for value in extra_environment),
        ("Exec", f"service start {service} --foreground --ipc"),
        ("AutoUpdate", "registry"),
    ]
    return [
        ("Unit", unit),
        ("Container", container),
        ("Service", [("Restart", "always")]),
        ("Install", [("WantedBy", "default.target")]),
    ]


def _validate_rendered_quadlets(
    source_root: Path, render_dir: Path, image: str
) -> list[str]:
    expected_files = {filename for filename, _ in CORE_SERVICES}
    expected_files.add("hyprstream-ipc.volume")
    actual_files = {entry.name for entry in render_dir.iterdir()}
    if actual_files != expected_files:
        _fail(
            "rendered Quadlet file set is not exact; "
            f"unexpected/missing={sorted(actual_files ^ expected_files)}"
        )
    template_root = source_root / QUADLET_TEMPLATE_RELATIVE
    _reject_symlink_components(template_root)
    _lstat_directory(template_root, "ingest Quadlet template directory")
    rendered_services: list[str] = []
    for filename, service in CORE_SERVICES:
        path = render_dir / filename
        text = _read_bytes(path, f"rendered {service} Quadlet").decode("utf-8")
        if re.search(r"\$\{[^}]+\}|@@[A-Z_]+@@", text):
            _fail(f"rendered {service} Quadlet contains an unresolved placeholder")
        parsed = _parse_quadlet(text, f"rendered {service} Quadlet")
        container = _quadlet_fields(parsed, "Container")
        environment = [value for key, value in container if key == "Environment"]
        template_values = {"hyprstream_image": image}
        if service == "model":
            if len(environment) != 2:
                _fail("rendered model Quadlet must have exactly two model inputs")
            expected_prefixes = (
                "HYPRSTREAM__MODELS__DEFAULT=",
                "HYPRSTREAM__MODELS__REPOSITORY=",
            )
            model_values: list[str] = []
            for entry, prefix in zip(environment, expected_prefixes, strict=True):
                if not entry.startswith(prefix):
                    _fail("rendered model environment inputs are not exact")
                model_values.append(entry.removeprefix(prefix))
            if (
                model_values[0] != model_values[1]
                or "synthetic" not in model_values[0]
                or not re.fullmatch(
                    r"model://[a-z0-9][a-z0-9._/-]*", model_values[0]
                )
            ):
                _fail("rendered model fixture must be one synthetic model reference")
            template_values["synthetic_model_fixture"] = model_values[0]
        elif service == "oauth":
            if len(environment) != 6:
                _fail("rendered OAuth Quadlet must have exactly six TLS inputs")
            environment_map: dict[str, str] = {}
            for entry in environment:
                if "=" not in entry:
                    _fail("rendered OAuth environment input is malformed")
                key, value = entry.split("=", 1)
                if key in environment_map:
                    _fail("rendered OAuth environment key is duplicated")
                environment_map[key] = value
            expected_environment_keys = {
                "HYPRSTREAM__TLS__ENABLED",
                "HYPRSTREAM__TLS__MODE",
                "HYPRSTREAM__TLS__SERVER_NAME",
                "HYPRSTREAM__TLS__ACME_DOMAIN",
                "HYPRSTREAM__TLS__ACME_CONTACT",
                "HYPRSTREAM__TLS__ACME_CACHE_DIR",
            }
            if set(environment_map) != expected_environment_keys:
                _fail("rendered OAuth TLS environment key set is not exact")
            hostname = environment_map["HYPRSTREAM__TLS__SERVER_NAME"]
            contact = environment_map["HYPRSTREAM__TLS__ACME_CONTACT"]
            if (
                environment_map["HYPRSTREAM__TLS__ENABLED"] != "true"
                or environment_map["HYPRSTREAM__TLS__MODE"] != "acme"
                or environment_map["HYPRSTREAM__TLS__ACME_DOMAIN"] != hostname
                or environment_map["HYPRSTREAM__TLS__ACME_CACHE_DIR"]
                != "/var/lib/hyprstream/acme"
                or not re.fullmatch(
                    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
                    r"staging\.lab\.hyprstream\.com",
                    hostname,
                )
                or not re.fullmatch(r"mailto:[^@\s]+@[^@\s]+\.[^@\s]+", contact)
            ):
                _fail("rendered OAuth TLS/trust inputs are not exact")
            template_values["hyprstream_discovery_hostname"] = hostname
            template_values["hyprstream_acme_email"] = contact.removeprefix("mailto:")
        elif environment:
            _fail(f"rendered {service} Quadlet has unexpected environment inputs")
        expected_sections = _expected_quadlet_sections(service, image, environment)
        if parsed != expected_sections:
            _fail(f"rendered {service} Quadlet directives are not exact")
        template_path = template_root / f"{filename}.tftpl"
        template = _read_bytes(
            template_path, f"ingest source template for {service}"
        ).decode("utf-8")
        expected_render = _render_source_template(
            template, template_values, f"ingest {service} template"
        )
        if text != expected_render:
            _fail(
                f"rendered {service} Quadlet is not byte-canonical to ingest source"
            )
        rendered_services.append(service)
    volume_path = render_dir / "hyprstream-ipc.volume"
    volume_text = _read_bytes(volume_path, "rendered IPC volume").decode("utf-8")
    if _parse_quadlet(volume_text, "rendered IPC volume") != [
        ("Volume", [("Driver", "local")])
    ]:
        _fail("rendered IPC volume directives are not exact")
    volume_template = _read_bytes(
        template_root / "hyprstream-ipc.volume.tftpl",
        "ingest source IPC volume template",
    ).decode("utf-8")
    if volume_text != volume_template:
        _fail("rendered IPC volume is not byte-canonical to ingest source")
    return rendered_services


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

    head = _run_git(source_root, "rev-parse", "HEAD")
    if head != expected_sha:
        _fail(f"ingest checkout head {head} does not match {expected_sha}")
    if _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=all"):
        _fail("ingest checkout must be completely clean")
    review_file_identities: set[tuple[int, int]] = set()
    review_digests: set[bytes] = set()
    reviewer_identities: set[str] = set()
    review_seats: set[str] = set()
    for review_record in review_records:
        _reject_symlink_components(review_record)
        info = _lstat_regular(review_record, "independent review record")
        reviewer, seat, digest = _parse_review_record(review_record, expected_sha)
        review_file_identities.add((info.st_dev, info.st_ino))
        review_digests.add(digest)
        reviewer_identities.add(reviewer)
        review_seats.add(seat)
    review_count = len(review_records)
    if len(review_file_identities) != review_count:
        _fail("independent review records must have distinct device/inode identity")
    if len(review_digests) != review_count:
        _fail("independent review records must have distinct content digests")
    if len(reviewer_identities) != review_count or len(review_seats) != review_count:
        _fail("independent reviews require distinct reviewer and seat identities")

    bundled_contract_path = (
        Path(__file__).with_name("fixtures") / "ingest-inference-contract-v1.json"
    )
    source_contract_path = source_root / INFERENCE_CONTRACT_RELATIVE
    bundled_bytes = _read_bytes(
        bundled_contract_path, "bundled reviewed ingest inference contract"
    )
    source_bytes = _read_bytes(
        source_contract_path, "ingest-owned versioned inference contract"
    )
    supplied_bytes = _read_bytes(
        inference_contract_path, "rendered versioned inference contract"
    )
    if source_bytes != bundled_bytes:
        _fail("ingest-owned inference artifact is not the reviewed v1 artifact")
    if supplied_bytes != source_bytes:
        _fail("rendered inference artifact is not byte-exact to ingest source")
    inference_contract = _read_json(
        inference_contract_path, "rendered versioned inference contract"
    )
    inference_image = validate_inference_contract(inference_contract)
    rendered_services = _validate_rendered_quadlets(
        source_root, render_dir, inference_image
    )
    return {
        "contract_version": "ingest-render-v1",
        "source_sha": expected_sha,
        "core_services": rendered_services,
        "core_image": inference_image,
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
