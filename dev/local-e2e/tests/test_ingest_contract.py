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
        (self.source / "source.txt").write_text("locked\n", encoding="utf-8")
        subprocess.run(["git", "-C", self.source, "add", "source.txt"], check=True)
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
        self.review.write_text(f"PASS {self.sha}\n\nfixture review\n", encoding="utf-8")
        self.review_b.write_text(
            f"PASS {self.sha}\n\nsecond fixture review\n", encoding="utf-8"
        )
        self.image = "registry.example.invalid/hyprstream@sha256:" + "a" * 64
        for filename, service in HARNESS.CORE_SERVICES:
            (self.render / filename).write_text(
                "\n".join(
                    (
                        "[Container]",
                        f"Image={self.image}",
                        f"ContainerName=hyprstream-{service}",
                        "Volume=hyprstream-ipc.volume:/run/hyprstream",
                        f"Exec=service start {service} --foreground --ipc",
                        "",
                    )
                ),
                encoding="utf-8",
            )
        (self.render / "hyprstream-ipc.volume").write_text(
            "[Volume]\nDriver=local\n", encoding="utf-8"
        )
        self.contract = {
            "contract_version": "v1",
            "implementation_merge": "6440b151dcc0238a0f92d877f2052ce2271395d8",
            "deployment_source_revision": "995fc622ae08b2baa1983646419ecccf4fe6a386",
            "contract_evidence": {
                "document_id": "inference-service-to-metal-runtime-contract-2026-07-26",
                "producer_prs": {
                    "implementation": 1356,
                    "publisher": 1358,
                    "immutable_index": 1360,
                },
            },
            "image": {
                "repository": "registry.example.invalid/hyprstream",
                "index_digest": "sha256:" + "a" * 64,
                "publisher_evidence": {
                    "status": "verified",
                    "reference": "https://example.invalid/fixture",
                },
            },
            "services": [self._inference_service(0), self._inference_service(1)],
        }
        self._write_contract(self.contract)

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _inference_service(replica: int) -> dict[str, object]:
        name = f"inference-cpu-{replica}"
        return {
            "inference_service_id": name,
            "unit_name": name,
            "container_name": name,
            "replica": replica,
            "listen_endpoint": {
                "transport": "udp",
                "address": "0.0.0.0",
                "port": 7440 + replica,
            },
            "health_endpoint": {
                "transport": "authenticated-inference-ipc",
                "socket": f"{name}.sock",
            },
            "direct_probe": {
                "authentication_scope": "query",
                "readiness_expectation": "isReady=true",
                "health_expectation": "modelLoaded=true,status=ok",
            },
        }

    def _write_contract(self, contract: dict[str, object]) -> None:
        self.inference.write_text(json.dumps(contract) + "\n", encoding="utf-8")

    def _verify(self) -> dict[str, object]:
        return HARNESS.verify_ingest_contract(
            self.source,
            self.sha,
            [self.review, self.review_b],
            self.render,
            self.inference,
        )

    def test_exact_reviewed_render_and_inference_contract_pass(self) -> None:
        result = self._verify()
        self.assertEqual(result["source_sha"], self.sha)
        self.assertEqual(result["core_services"], [item[1] for item in HARNESS.CORE_SERVICES])
        self.assertEqual(result["core_image"], self.image)

    def test_review_dirty_source_missing_service_and_mutable_image_fail(self) -> None:
        self.review.write_text(f"REVISE {self.sha}\n", encoding="utf-8")
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()
        self.review.write_text(f"PASS {self.sha}\n", encoding="utf-8")

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

    def test_wrong_port_probe_and_symlinked_contract_fail(self) -> None:
        wrong = copy.deepcopy(self.contract)
        wrong["services"][1]["listen_endpoint"]["port"] = 7440
        self._write_contract(wrong)
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        wrong = copy.deepcopy(self.contract)
        wrong["services"][0]["direct_probe"]["health_expectation"] = "status=maybe"
        self._write_contract(wrong)
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()

        self._write_contract(self.contract)
        real = self.inference.with_suffix(".real.json")
        self.inference.rename(real)
        self.inference.symlink_to(real)
        with self.assertRaises(HARNESS.HarnessError):
            self._verify()


if __name__ == "__main__":
    unittest.main()
