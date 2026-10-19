#!/usr/bin/env python3
"""Regression tests for hakari's workspace-hack wasm gate normalization."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import hack_wasm_gate  # noqa: E402


class HackWasmGateTests(unittest.TestCase):
    def test_gate_restores_final_newline_after_hakari_rewrite(self):
        source = (
            '[target.\'cfg(not(target_arch = "wasm32"))\'.dependencies]\n'
            'workspace-hack = { version = "0.1", path = "../../workspace-hack" }'
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = pathlib.Path(directory) / "Cargo.toml"
            manifest.write_text(source)

            with mock.patch.object(hack_wasm_gate, "tracked_manifest_has_final_newline", return_value=True):
                self.assertTrue(hack_wasm_gate.gate_manifest(manifest))
            self.assertTrue(manifest.read_bytes().endswith(b"\n"))
            with mock.patch.object(hack_wasm_gate, "tracked_manifest_has_final_newline", return_value=True):
                self.assertFalse(hack_wasm_gate.gate_manifest(manifest))


if __name__ == "__main__":
    unittest.main()
