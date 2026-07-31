from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "causal_harness", ROOT / "causal_harness.py"
)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def _template_for(service: str) -> str:
    after, requires = HARNESS.QUADLET_DEPENDENCIES[service]
    lines = [
        "[Unit]",
        f"Description={HARNESS.QUADLET_DESCRIPTIONS[service]}",
        f"After={after}",
    ]
    if requires is not None:
        lines.append(f"Requires={requires}")
    lines.extend(
        [
            "",
            "[Container]",
            "Image=${hyprstream_image}",
            f"ContainerName=hyprstream-{service}",
            "Network=host",
            "Volume=hyprstream-ipc.volume:/run/hyprstream",
            "Volume=/var/lib/hyprstream:/var/lib/hyprstream:z",
            "EnvironmentFile=/var/lib/hyprstream/hyprstream.env",
        ]
    )
    if service == "model":
        lines.extend(
            [
                "Environment=HYPRSTREAM__MODELS__DEFAULT=${synthetic_model_fixture}",
                "Environment=HYPRSTREAM__MODELS__REPOSITORY=${synthetic_model_fixture}",
            ]
        )
    if service == "oauth":
        lines.extend(
            [
                "Environment=HYPRSTREAM__TLS__ENABLED=true",
                "Environment=HYPRSTREAM__TLS__MODE=acme",
                "Environment=HYPRSTREAM__TLS__SERVER_NAME=${hyprstream_discovery_hostname}",
                "Environment=HYPRSTREAM__TLS__ACME_DOMAIN=${hyprstream_discovery_hostname}",
                "Environment=HYPRSTREAM__TLS__ACME_CONTACT=mailto:${hyprstream_acme_email}",
                "Environment=HYPRSTREAM__TLS__ACME_CACHE_DIR=/var/lib/hyprstream/acme",
            ]
        )
    lines.extend(
        [
            f"Exec=service start {service} --foreground --ipc",
            "AutoUpdate=registry",
            "",
            "[Service]",
            "Restart=always",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )
    return "\n".join(lines)


def _render_template(template: str, image: str) -> str:
    values = {
        "hyprstream_image": image,
        "synthetic_model_fixture": "model://synthetic-staging-fixture",
        "hyprstream_discovery_hostname": "discovery.staging.lab.hyprstream.com",
        "hyprstream_acme_email": "reviewer@example.invalid",
    }
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"${{{key}}}", value)
    return rendered


def _dict_paths(value: object, prefix: tuple[object, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, key)
            yield path
            yield from _dict_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _dict_paths(child, (*prefix, index))


def _parent_at(value: object, path: tuple[object, ...]):
    current = value
    for component in path[:-1]:
        current = current[component]
    return current, path[-1]


class IngestContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.source = base / "source"
        self.render = base / "render"
        self.review = base / "review-a.md"
        self.review_b = base / "review-b.md"
        self.inference = base / "inference.json"
        self.source.mkdir(mode=0o700)
        self.render.mkdir(mode=0o700)
        subprocess.run(["git", "init", "-q", self.source], check=True)
        subprocess.run(
            ["git", "-C", self.source, "config", "user.name", "Harness Test"],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                self.source,
                "config",
                "user.email",
                "harness@example.invalid",
            ],
            check=True,
        )
        self.contract_bytes = (
            ROOT / "fixtures" / "ingest-inference-contract-v1.json"
        ).read_bytes()
        self.contract = json.loads(self.contract_bytes)
        self.image = (
            f"{self.contract['image']['repository']}@"
            f"{self.contract['image']['index_digest']}"
        )
        source_contract = self.source / HARNESS.INFERENCE_CONTRACT_RELATIVE
        source_contract.parent.mkdir(parents=True)
        source_contract.write_bytes(self.contract_bytes)
        template_root = self.source / HARNESS.QUADLET_TEMPLATE_RELATIVE
        template_root.mkdir(parents=True)
        for filename, service in HARNESS.CORE_SERVICES:
            template = _template_for(service)
            (template_root / f"{filename}.tftpl").write_text(
                template, encoding="utf-8"
            )
            (self.render / filename).write_text(
                _render_template(template, self.image), encoding="utf-8"
            )
        volume = "[Volume]\nDriver=local\n"
        (template_root / "hyprstream-ipc.volume.tftpl").write_text(
            volume, encoding="utf-8"
        )
        (self.render / "hyprstream-ipc.volume").write_text(
            volume, encoding="utf-8"
        )
        subprocess.run(["git", "-C", self.source, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", self.source, "commit", "-q", "-m", "fixture"],
            check=True,
        )
        self.sha = subprocess.run(
            ["git", "-C", self.source, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self._write_review(
            self.review, self.sha, "gpt-5.6-sol", "independent-first"
        )
        self._write_review(
            self.review_b, self.sha, "claude-fable-5", "independent-second"
        )
        self.inference.write_bytes(self.contract_bytes)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _write_review(
        path: Path, sha: str, reviewer: str, seat: str, body: str = "fixture review"
    ) -> None:
        path.write_text(
            "\n".join(
                (
                    f"PASS {sha}",
                    "Review-Schema: independent-review-v1",
                    f"Reviewer-Identity: {reviewer}",
                    f"Review-Seat: {seat}",
                    "",
                    body,
                    "",
                )
            ),
            encoding="utf-8",
        )

    def _verify(self) -> dict[str, object]:
        return HARNESS.verify_ingest_contract(
            self.source,
            self.sha,
            [self.review, self.review_b],
            self.render,
            self.inference,
        )

    def _restore_render(self) -> None:
        template_root = self.source / HARNESS.QUADLET_TEMPLATE_RELATIVE
        for filename, _service in HARNESS.CORE_SERVICES:
            template = (template_root / f"{filename}.tftpl").read_text(
                encoding="utf-8"
            )
            (self.render / filename).write_text(
                _render_template(template, self.image), encoding="utf-8"
            )
        (self.render / "hyprstream-ipc.volume").write_text(
            "[Volume]\nDriver=local\n", encoding="utf-8"
        )
        for extra in self.render.iterdir():
            if extra.name not in {
                *(filename for filename, _ in HARNESS.CORE_SERVICES),
                "hyprstream-ipc.volume",
            }:
                extra.unlink()

    def test_exact_reviewed_render_and_inference_contract_pass(self) -> None:
        result = self._verify()
        self.assertEqual(result["source_sha"], self.sha)
        self.assertEqual(
            result["core_services"], [item[1] for item in HARNESS.CORE_SERVICES]
        )
        self.assertEqual(result["core_image"], self.image)

    def test_reviews_require_distinct_inode_digest_reviewer_seat_and_head(self) -> None:
        original_b = self.review_b.read_bytes()

        self.review_b.unlink()
        os.link(self.review, self.review_b)
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()
        self.review_b.unlink()
        self.review_b.write_bytes(original_b)

        self.review_b.write_bytes(self.review.read_bytes())
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._write_review(
            self.review_b, self.sha, "gpt-5.6-sol", "independent-second"
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._write_review(
            self.review_b, self.sha, "claude-fable-5", "independent-first"
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        stale = "0" * 40
        self._write_review(
            self.review, stale, "gpt-5.6-sol", "independent-first"
        )
        self._write_review(
            self.review_b, stale, "claude-fable-5", "independent-second"
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._write_review(
            self.review, self.sha, "gpt-5.6-sol", "independent-first"
        )
        self._write_review(
            self.review_b, stale, "claude-fable-5", "independent-second"
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._write_review(
            self.review_b, self.sha, "claude-fable-5", "independent-second"
        )
        (self.source / "new-head.txt").write_text("new\n", encoding="utf-8")
        subprocess.run(["git", "-C", self.source, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", self.source, "commit", "-q", "-m", "new head"],
            check=True,
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

    def test_every_inference_field_deletion_and_mutation_fails(self) -> None:
        paths = list(_dict_paths(self.contract))
        self.assertGreater(len(paths), 60)
        for path in paths:
            with self.subTest(operation="delete", path=path):
                changed = copy.deepcopy(self.contract)
                parent, final = _parent_at(changed, path)
                del parent[final]
                with self.assertRaises(HARNESS.HarnessError):
                    HARNESS.validate_inference_contract(changed)
            with self.subTest(operation="mutate", path=path):
                changed = copy.deepcopy(self.contract)
                parent, final = _parent_at(changed, path)
                parent[final] = None
                with self.assertRaises(HARNESS.HarnessError):
                    HARNESS.validate_inference_contract(changed)

    def test_inference_must_match_ingest_owned_artifact_byte_for_byte(self) -> None:
        changed = copy.deepcopy(self.contract)
        changed["services"][0]["memory_budget_mib"] = 4096
        self.inference.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self.inference.write_bytes(self.contract_bytes)
        source_contract = self.source / HARNESS.INFERENCE_CONTRACT_RELATIVE
        source_contract.write_text(json.dumps(self.contract) + "\n", encoding="utf-8")
        subprocess.run(["git", "-C", self.source, "add", "."], check=True)
        subprocess.run(
            ["git", "-C", self.source, "commit", "-q", "-m", "reformatted"],
            check=True,
        )
        new_sha = subprocess.run(
            ["git", "-C", self.source, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self._write_review(
            self.review, new_sha, "gpt-5.6-sol", "independent-first"
        )
        self._write_review(
            self.review_b, new_sha, "claude-fable-5", "independent-second"
        )
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS.verify_ingest_contract(
                self.source,
                new_sha,
                [self.review, self.review_b],
                self.render,
                self.inference,
            )

    def test_quadlet_duplicate_command_volume_dependency_and_extras_fail(self) -> None:
        event = self.render / "hyprstream.container"
        mutations = (
            ("duplicate-key", "ContainerName=hyprstream-event", "ContainerName=other\n"),
            (
                "extra-command",
                "Exec=service start event --foreground --ipc",
                "Exec=wrong command\n",
            ),
            (
                "extra-volume",
                "Volume=/var/lib/hyprstream:/var/lib/hyprstream:z",
                "Volume=/tmp:/host\n",
            ),
            ("wrong-dependency", "After=network.target", "After=wrong.service"),
            ("wrong-network", "Network=host", "Network=private"),
            ("extra-directive", "AutoUpdate=registry", "PodmanArgs=--privileged\n"),
        )
        for name, needle, insertion in mutations:
            with self.subTest(name=name):
                self._restore_render()
                text = event.read_text(encoding="utf-8")
                if name == "wrong-dependency" or name == "wrong-network":
                    text = text.replace(needle, insertion)
                else:
                    text = text.replace(needle, f"{needle}\n{insertion}")
                event.write_text(text, encoding="utf-8")
                with self.assertRaises(HARNESS.HarnessError):
                    self._verify()

        self._restore_render()
        oauth = self.render / "hyprstream-oauth.container"
        oauth.write_text(
            oauth.read_text(encoding="utf-8").replace(
                "HYPRSTREAM__TLS__MODE=acme", "HYPRSTREAM__TLS__MODE=disabled"
            ),
            encoding="utf-8",
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._restore_render()
        (self.render / "unexpected.container").write_text(
            "[Container]\nImage=unexpected\n", encoding="utf-8"
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

    def test_dirty_source_missing_service_mutable_image_and_symlink_fail(self) -> None:
        (self.source / "dirty.txt").write_text("dirty\n", encoding="utf-8")
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()
        (self.source / "dirty.txt").unlink()

        missing = self.render / "hyprstream-oai.container"
        saved = missing.read_text(encoding="utf-8")
        missing.unlink()
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()
        missing.write_text(saved, encoding="utf-8")

        event = self.render / "hyprstream.container"
        event.write_text(
            event.read_text(encoding="utf-8").replace(self.image, "image:latest"),
            encoding="utf-8",
        )
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._restore_render()
        real = self.inference.with_suffix(".real.json")
        self.inference.rename(real)
        self.inference.symlink_to(real)
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()


if __name__ == "__main__":
    unittest.main()
