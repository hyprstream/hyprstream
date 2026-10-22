from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "causal_harness", ROOT / "causal_harness.py"
)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
CONTRACT = json.loads(
    (ROOT / "fixtures" / "response-contract-v1.json").read_text(encoding="utf-8")
)


class ResponseContractTests(unittest.TestCase):
    def test_all_exact_positive_classes(self) -> None:
        observations = {
            "wrong_ca": {"class": "tls_failure", "status": None, "body": None},
            "wrong_trust_directory": {
                "class": "startup_failure",
                "status": None,
                "body": None,
            },
            "service_auth_bad_request": {
                "class": "http_json",
                "status": 400,
                "body": {"error": "InvalidRequest", "message": "aud is required"},
            },
            "session_exchange_missing_bearer": {
                "class": "http_json",
                "status": 401,
                "body": {
                    "error": "invalid_token",
                    "error_description": "Bearer token required",
                },
            },
            "session_exchange_missing_dpop": {
                "class": "http_json",
                "status": 400,
                "body": {
                    "error": "invalid_dpop_proof",
                    "error_description": "DPoP proof required",
                },
            },
            "whoami_unauthenticated": {
                "class": "http_json",
                "status": 200,
                "body": {
                    "did": None,
                    "kind": "unauthenticated",
                    "tenant": None,
                    "canActLocally": False,
                },
            },
            "inference_ready": {
                "class": "authenticated_rpc",
                "status": None,
                "body": {"isReady": True},
            },
            "inference_healthy": {
                "class": "authenticated_rpc",
                "status": None,
                "body": {"modelLoaded": True, "status": "ok"},
            },
            "inference_application_output": {
                "class": "authenticated_rpc",
                "status": None,
                "body": {"output": "hello"},
            },
        }
        self.assertEqual(set(observations), set(CONTRACT["cases"]))
        for name, observed in observations.items():
            HARNESS.validate_response_contract(CONTRACT, name, observed)

    def test_wrong_status_extra_key_and_empty_output_fail(self) -> None:
        bad = {
            "class": "http_json",
            "status": 404,
            "body": {"error": "InvalidRequest", "message": "bad"},
        }
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS.validate_response_contract(CONTRACT, "service_auth_bad_request", bad)

        bad["status"] = 400
        bad["body"]["extra"] = True
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS.validate_response_contract(CONTRACT, "service_auth_bad_request", bad)

        empty = {
            "class": "authenticated_rpc",
            "status": None,
            "body": {"output": ""},
        }
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS.validate_response_contract(
                CONTRACT, "inference_application_output", empty
            )


if __name__ == "__main__":
    unittest.main()
