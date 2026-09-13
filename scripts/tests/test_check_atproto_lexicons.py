"""Offline regression tests for vendored source integrity and reference closure."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "lexicons", Path(__file__).resolve().parents[1] / "check_atproto_lexicons.py"
)
lexicons = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lexicons)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.selection = {"upstream_commit": "a" * 40, "roots": ["app.bsky.feed.post"]}
        self.docs = {
            "app.bsky.feed.post": {"lexicon": 1, "id": "app.bsky.feed.post", "defs": {
                "main": {"type": "record", "record": {"type": "ref", "ref": "#body"}},
                "body": {"type": "union", "refs": ["com.atproto.repo.strongRef"]}}},
            "com.atproto.repo.strongRef": {"lexicon": 1, "id": "com.atproto.repo.strongRef",
                                          "defs": {"main": {"type": "object"}}},
        }
        self.write_bundle()

    def write_bundle(self):
        files = {}
        for nsid, doc in self.docs.items():
            raw = json.dumps(doc).encode()
            relative = lexicons.schema_path(nsid)
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            files[relative] = hashlib.sha256(raw).hexdigest()
        licenses = {}
        for name in lexicons.LICENSES:
            (self.directory / name).write_bytes(b"fixture license")
            licenses[name] = hashlib.sha256(b"fixture license").hexdigest()
        (self.directory / "manifest.json").write_text(json.dumps(
            dict(self.selection, files=files, licenses=licenses)))

    def test_complete_closure_never_uses_network(self):
        with patch.object(lexicons.urllib.request, "urlopen", side_effect=AssertionError("network")):
            self.assertEqual(lexicons.verify(self.selection, self.directory), 2)

    def test_modified_bytes_fail_even_if_json_is_equivalent(self):
        path = self.directory / "app/bsky/feed/post.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            lexicons.verify(self.selection, self.directory)

    def test_missing_file_is_not_fetched(self):
        (self.directory / "com/atproto/repo/strongRef.json").unlink()
        with self.assertRaisesRegex(ValueError, "file mismatch"):
            lexicons.verify(self.selection, self.directory)

    def test_missing_definition_fails_with_updated_digest(self):
        self.docs["com.atproto.repo.strongRef"]["defs"] = {"other": {"type": "object"}}
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "unresolved reference"):
            lexicons.verify(self.selection, self.directory)

    def test_unused_schema_fails(self):
        self.docs["app.bsky.feed.like"] = {"id": "app.bsky.feed.like", "lexicon": 1,
                                           "defs": {"main": {"type": "object"}}}
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "outside selected reference closure"):
            lexicons.verify(self.selection, self.directory)

    def test_unexpected_file_fails(self):
        (self.directory / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "file mismatch"):
            lexicons.verify(self.selection, self.directory)

    def test_pin_change_requires_refresh(self):
        self.selection["upstream_commit"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "selection/manifest mismatch"):
            lexicons.verify(self.selection, self.directory)

    def test_reference_cycles_terminate(self):
        self.docs["com.atproto.repo.strongRef"]["defs"]["main"] = {
            "type": "ref", "ref": "app.bsky.feed.post"}
        self.write_bundle()
        self.assertEqual(lexicons.verify(self.selection, self.directory), 2)

    def test_urls_and_path_traversal_are_rejected(self):
        for ref in ("../../etc/passwd", "https://example.com/schema", "com.atproto/../../x"):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                lexicons.target("app.bsky.feed.post", ref)


if __name__ == "__main__":
    unittest.main()
