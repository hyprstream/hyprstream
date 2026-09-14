#!/usr/bin/env python3
"""Small real-Git provenance and extractor regressions; never edit the caller's index."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("catalog", ROOT / "tools/check-docs-catalog.py")
catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(catalog)
CORPUS = catalog.read_json(ROOT / "docs/corpus-sources.json", ROOT)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="docs-catalog-fixture-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "author"
        self.repo.mkdir()
        self.env = {**os.environ, "GIT_AUTHOR_NAME": "Catalog fixture", "GIT_COMMITTER_NAME": "Catalog fixture",
                    "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}
        # Production selection still scans the complete fixture tree. Only the
        # real workspace's fixed Rust consumer paths are irrelevant here.
        self.fixed = patch.object(catalog, "provenance_fixed_paths", return_value=set())
        self.fixed.start(); self.addCleanup(self.fixed.stop)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start(); self.addCleanup(self.environment.stop)
        self.git(self.repo, "init", "-q", "-b", "author")
        self.write(self.repo, "schema.capnp", "base\n")
        self.base = self.commit(self.repo)
        self.write(self.repo, "schema.capnp", "audited\n")
        self.source = self.commit(self.repo)
        self.tree = self.git(self.repo, "rev-parse", "HEAD^{tree}")
        self.corpus = copy.deepcopy(CORPUS)
        self.corpus.update(provenance_version=2, source_commit=self.source, source_tree=self.tree,
                           source_input_digest=catalog.input_digest(self.repo, ["schema.capnp"]))
        self.record = {key: self.corpus[key] for key in catalog.PROVENANCE_FIELDS}
        self.write_manifests(self.repo)
        self.head = self.commit(self.repo)

    def git(self, repo, *args, check=True):
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=self.env)
        if check and p.returncode:
            self.fail(p.stderr)
        return p.stdout.strip() if check else p

    def write(self, repo, path, data):
        dest = repo / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(data)

    def write_manifests(self, repo):
        self.write(repo, "docs/schema-catalog.json", json.dumps(self.record))
        self.write(repo, "docs/corpus-sources.json", json.dumps(self.corpus))

    def commit(self, repo):
        self.git(repo, "add", ".")
        self.git(repo, "commit", "-qm", "fixture")
        return self.git(repo, "rev-parse", "HEAD")

    def verify(self, repo=None, record=None, base=None, event="pull_request", recover=False):
        catalog.check_provenance(record or self.record, repo or self.repo, "fixture", self.corpus,
                                 event=event, revision=base or self.base, recover=recover)

    def squash_clone(self):
        self.git(self.repo, "checkout", "-q", "-b", "main", self.base)
        self.write(self.repo, "schema.capnp", "audited\n")
        self.write_manifests(self.repo)
        landing = self.commit(self.repo)
        self.git(self.repo, "branch", "-D", "author")
        client = self.root / "client"
        self.git(self.repo, "clone", "-q", "--no-local", "--single-branch", "--branch", "main", str(self.repo), str(client))
        self.assertFalse(catalog.object_exists(client, self.source, "commit"))
        return client, landing

    def test_direct_and_newer_unrelated_base(self):
        self.verify()
        self.git(self.repo, "checkout", "-q", "-b", "advanced", self.base)
        self.write(self.repo, "unrelated.txt", "main advanced\n")
        advanced = self.commit(self.repo)
        self.git(self.repo, "checkout", "-q", "author")
        self.assertNotEqual(self.git(self.repo, "merge-base", advanced, self.head), advanced)
        self.verify(base=advanced)  # Latest PR base is NOT an ancestor of head.

    def test_newer_boundary_and_push(self):
        self.write(self.repo, "unrelated.txt", "later\n")
        self.commit(self.repo)
        self.verify(base=self.head)
        self.verify(base=self.head, event="push")
        with self.assertRaises(catalog.CatalogError):
            self.verify(base=self.git(self.repo, "rev-parse", "HEAD"), event="push")

    def test_missing_squash_requires_evidence_then_recovers(self):
        client, landing = self.squash_clone()
        with self.assertRaisesRegex(catalog.CatalogError, "lacks verifiable"):
            self.verify(client, base=self.base, event="push")
        # Invoke the actual recovery implementation. Redirect only its argv's
        # pinned repository transport to this isolated origin; no network.
        original_run = subprocess.run
        def transport(args, **kwargs):
            if args[3:6] == ["remote", "get-url", "origin"]:
                return subprocess.CompletedProcess(args, 0, "https://github.com/hyprstream/hyprstream.git\n", "")
            return original_run(args, **kwargs)
        with patch.object(catalog.subprocess, "run", side_effect=transport):
            self.verify(client, base=self.base, event="push", recover=True)
            with self.assertRaisesRegex(catalog.CatalogError, "source object unavailable"):
                self.verify(client, dict(self.record, source_commit="f" * 40), base=self.base, event="push", recover=True)
        self.assertTrue(catalog.object_exists(client, self.source, "commit"))
        self.assertEqual(self.git(client, "rev-parse", "HEAD"), landing)
        self.assertFalse((client / ".git/FETCH_HEAD").exists())

    def test_exact_inheritance_after_squash_and_base_divergence(self):
        client, landing = self.squash_clone()
        self.write(client, "feature.txt", "branch change\n")
        feature = self.commit(client)
        self.git(client, "checkout", "-q", "-b", "advanced", landing)
        self.write(client, "main.txt", "main change\n")
        advanced = self.commit(client)
        self.git(client, "checkout", "-q", "--detach", feature)
        self.verify(client, base=advanced)
        self.assertFalse(catalog.object_exists(client, self.source, "commit"))
        for field, value in [("source_commit", "f" * 40), ("source_tree", "e" * 40),
                             ("source_input_digest", "f" * 64), ("provenance_version", 1)]:
            with self.subTest(field=field), self.assertRaises(catalog.CatalogError):
                self.verify(client, dict(self.record, **{field: value}), base=advanced)
        self.write(client, "schema.capnp", "changed input\n")
        with self.assertRaisesRegex(catalog.CatalogError, "current audited inputs"):
            self.verify(client, base=advanced)

    def test_corrupt_trusted_base_cannot_inherit(self):
        client, landing = self.squash_clone()
        self.write(client, "schema.capnp", "base corruption\n")
        bad_base = self.commit(client)
        self.git(client, "checkout", "-q", "--detach", landing)
        with self.assertRaisesRegex(catalog.CatalogError, "trusted boundary inputs"):
            self.verify(client, base=bad_base)

    def test_available_contradiction_never_inherits(self):
        for values in [{"source_tree": "0" * 40}, {"source_tree": self.base},
                       {"source_commit": self.base}, {"source_commit": "--upload-pack=evil"}]:
            with self.subTest(values=values), self.assertRaises(catalog.CatalogError):
                self.verify(record=dict(self.record, **values))
        original = copy.deepcopy(self.record)
        self.record["source_tree"] = self.git(self.repo, "rev-parse", self.base + "^{tree}")
        self.corpus["source_tree"] = self.record["source_tree"]
        self.write_manifests(self.repo)
        trusted = self.commit(self.repo)
        self.write(self.repo, "another.txt", "later\n"); self.commit(self.repo)
        with self.assertRaisesRegex(catalog.CatalogError, "does not match"):
            self.verify(base=trusted)
        self.record = original

    def test_invalid_recovery_claims_and_origin(self):
        for sha in ["f" * 40, "--upload-pack=evil"]:
            with self.subTest(sha=sha), self.assertRaises(catalog.CatalogError):
                catalog.recover_source_objects(self.repo, sha)
        self.git(self.repo, "remote", "add", "origin", "https://example.invalid/repo.git")
        with self.assertRaisesRegex(catalog.CatalogError, "trusted repository"):
            catalog.recover_source_objects(self.repo, self.source)

    def test_checkout_and_git_suffix_origins(self):
        self.git(self.repo, "remote", "add", "origin", str(self.repo))
        real_git = catalog.git
        for url in ["https://github.com/hyprstream/hyprstream", "https://github.com/hyprstream/hyprstream.git",
                    "https://user@github.com/hyprstream/hyprstream", "https://github.com/hyprstream/hyprstream?x=1",
                    "https://github.com/hyprstream/hyprstream#ref", "https://github.com/other/hyprstream"]:
            def configured_origin(repo, *args):
                return url if args == ("remote", "get-url", "origin") else real_git(repo, *args)
            with self.subTest(url=url), patch.object(catalog, "git", side_effect=configured_origin):
                if url in {"https://github.com/hyprstream/hyprstream", "https://github.com/hyprstream/hyprstream.git"}:
                    catalog.recover_source_objects(self.repo, self.source)
                else:
                    with self.assertRaisesRegex(catalog.CatalogError, "trusted repository"):
                        catalog.recover_source_objects(self.repo, self.source)

    def test_wrong_head_and_unrelated_boundary(self):
        with patch.dict(os.environ, {"DOCS_CATALOG_AUDITED_HEAD": "f" * 40}), self.assertRaises(catalog.CatalogError):
            self.verify()
        self.git(self.repo, "checkout", "-q", "--orphan", "unrelated")
        unrelated = self.commit(self.repo)
        self.git(self.repo, "checkout", "-q", "author")
        with self.assertRaises(catalog.CatalogError):
            self.verify(base=unrelated)

    def test_package_selection_parity_and_removal(self):
        for section in ["dependencies", "devDependencies", "peerDependencies"]:
            with self.subTest(section=section):
                self.write(self.repo, "package.json", json.dumps({section: {"@hyprstream/docs": "1"}}))
                source = self.commit(self.repo)
                tree = self.git(self.repo, "rev-parse", source + "^{tree}")
                paths = catalog.provenance_paths(self.repo, self.corpus)
                self.assertIn("package.json", paths)
                self.assertEqual(set(paths), catalog.attested_tree_universe(self.repo, tree, self.corpus))
                self.write(self.repo, "package.json", "{}")
                self.commit(self.repo)
                paths = catalog.provenance_paths(self.repo, self.corpus)
                record = dict(self.record, source_commit=source, source_tree=tree,
                              source_input_digest=catalog.input_digest(self.repo, paths))
                with self.assertRaisesRegex(catalog.CatalogError, "removed"):
                    self.verify(record=record)
                (self.repo / "package.json").unlink()
                self.git(self.repo, "add", "-u")
                with self.assertRaises(catalog.CatalogError):
                    self.verify(record=record)

    def test_cross_manifest_tuple_mismatch(self):
        bad = dict(self.corpus, source_tree="e" * 40)
        with self.assertRaisesRegex(catalog.CatalogError, "tuples differ"):
            catalog.validate(self.repo, self.record, bad, event="local")

    def test_old_policy_is_not_an_inheritance_anchor(self):
        client, landing = self.squash_clone()
        for path in ["docs/schema-catalog.json", "docs/corpus-sources.json"]:
            value = json.loads((client / path).read_text())
            value["provenance_version"] = 1
            self.write(client, path, json.dumps(value))
        old_policy = self.commit(client)
        self.git(client, "checkout", "-q", "--detach", landing)
        with self.assertRaisesRegex(catalog.CatalogError, "lacks verifiable"):
            self.verify(client, base=old_policy)

    def test_live_and_tree_nonfiles_are_rejected(self):
        (self.repo / "schema.capnp").unlink()
        (self.repo / "schema.capnp").symlink_to("/etc/passwd")
        with self.assertRaises(catalog.CatalogError):
            catalog.input_digest(self.repo, ["schema.capnp"])
        source = self.commit(self.repo)
        with self.assertRaisesRegex(catalog.CatalogError, "not a regular file"):
            catalog.input_digest(self.repo, ["schema.capnp"], tree=self.git(self.repo, "rev-parse", source + "^{tree}"))
        (self.repo / "schema.capnp").unlink()
        os.mkfifo(self.repo / "schema.capnp")
        with self.assertRaises(catalog.CatalogError):
            catalog.input_digest(self.repo, ["schema.capnp"])
        (self.repo / "nested").symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(catalog.CatalogError):
            catalog.read_regular(self.repo, "nested/outside")

    def test_nested_package_inventory_and_provenance(self):
        for section in ["dependencies", "devDependencies", "peerDependencies"]:
            with self.subTest(section=section):
                manifests = ["package.json", "packages/client/package.json", "web/deep/client/package.json"]
                for path in manifests:
                    self.write(self.repo, path, json.dumps({section: {"@hyprstream/docs": "1"}}))
                self.write(self.repo, "packages/unrelated/package.json", '{"dependencies":{"other":"1"}}')
                self.commit(self.repo)
                paths = catalog.provenance_paths(self.repo, self.corpus)
                self.assertEqual(catalog.typescript_schema_sources(self.repo), sorted(manifests))
                self.assertTrue(set(manifests) <= set(paths))
                tree = self.git(self.repo, "rev-parse", "HEAD^{tree}")
                self.assertEqual(set(paths), catalog.attested_tree_universe(self.repo, tree, self.corpus))
                before = catalog.input_digest(self.repo, paths)
                self.write(self.repo, manifests[1], json.dumps({section: {"@hyprstream/docs": "2"}}))
                self.assertNotEqual(before, catalog.input_digest(self.repo, paths))
                self.assertEqual(before, catalog.input_digest(self.repo, paths, tree=tree))

    def test_nested_manifest_symlink_rejected_live_and_tree(self):
        path = "packages/client/package.json"
        target = self.repo / path
        target.parent.mkdir(parents=True)
        target.symlink_to("../../schema.capnp")
        self.commit(self.repo)
        with self.assertRaises(catalog.CatalogError):
            catalog.provenance_paths(self.repo, self.corpus)
        tree = self.git(self.repo, "rev-parse", "HEAD^{tree}")
        with self.assertRaisesRegex(catalog.CatalogError, "not a regular file"):
            catalog.attested_tree_universe(self.repo, tree, self.corpus)

    def test_template_dependency_enters_live_and_tree_provenance(self):
        path = "packages/client/src/render.ts"
        self.write(self.repo, path, 'const x = `${import("@hyprstream/docs")}`;')
        source = self.commit(self.repo)
        tree = self.git(self.repo, "rev-parse", source + "^{tree}")
        paths = catalog.provenance_paths(self.repo, self.corpus)
        self.assertIn(path, paths)
        self.assertIn(path, catalog.typescript_schema_sources(self.repo))
        self.assertEqual(set(paths), catalog.attested_tree_universe(self.repo, tree, self.corpus))
        self.write(self.repo, path, 'const x = `inert @hyprstream/docs`;')
        self.assertNotIn(path, catalog.provenance_paths(self.repo, self.corpus))

    def test_index_probes_preserve_caller_bytes(self):
        optional = "crates/hyprstream-rpc/schema/optional.capnp"
        common = "crates/hyprstream-rpc/schema/common.capnp"
        for path in [optional, common]:
            self.write(self.repo, path, "# original\n")
        self.commit(self.repo)
        index = self.repo / ".git/index"
        for deletion in [None, optional, common]:
            for failure in [None, "second stage", "probe"]:
                with self.subTest(deletion=deletion, failure=failure):
                    self.git(self.repo, "reset", "--mixed", "HEAD")
                    if deletion:
                        self.git(self.repo, "rm", "--cached", "--quiet", deletion)
                    self.write(self.repo, "staged.txt", "unrelated staged edit\n")
                    self.git(self.repo, "add", "staged.txt")
                    self.git(self.repo, "update-index", "--assume-unchanged", "schema.capnp")
                    # Preserve unmerged stages too, not just stage0 rows/flags.
                    oid = self.git(self.repo, "rev-parse", "HEAD:schema.capnp")
                    subprocess.run(["git", "-C", str(self.repo), "update-index", "--index-info"],
                                   input=f"100644 {oid} 1\tconflicted.capnp\n100644 {oid} 2\tconflicted.capnp\n",
                                   text=True, env=self.env, check=True, capture_output=True)
                    before = index.read_bytes()
                    original_run = subprocess.run
                    stage_count = 0
                    def injected(args, **kwargs):
                        nonlocal stage_count
                        if "update-index" in args and "--cacheinfo" in args:
                            stage_count += 1
                            if failure == "second stage" and stage_count == 2:
                                raise RuntimeError("injected second stage failure")
                        return original_run(args, **kwargs)
                    def probe():
                        if failure == "probe":
                            raise RuntimeError("injected probe failure")
                        self.assertNotIn(optional, catalog.tracked(self.repo))
                    try:
                        with patch.object(catalog.subprocess, "run", side_effect=injected):
                            if failure:
                                with self.assertRaisesRegex(RuntimeError, "injected"):
                                    catalog.probe_staged_removal(self.repo, probe)
                            else:
                                catalog.probe_staged_removal(self.repo, probe)
                    finally:
                        self.assertEqual(index.read_bytes(), before, "caller index changed")
                        self.assertNotIn("GIT_INDEX_FILE", os.environ)


    def test_explicit_split_index_is_untouched(self):
        for path in ["crates/hyprstream-rpc/schema/optional.capnp", "crates/hyprstream-rpc/schema/common.capnp"]:
            self.write(self.repo, path, "# source\n")
        self.commit(self.repo)
        original = (self.repo / ".git/index").read_bytes()
        alternate = self.root / "caller.index"
        alternate.write_bytes(original)
        with patch.dict(os.environ, {"GIT_INDEX_FILE": str(alternate)}):
            catalog.git(self.repo, "update-index", "--split-index")
            catalog.git(self.repo, "rm", "--cached", "--quiet", "crates/hyprstream-rpc/schema/common.capnp")
            before = alternate.read_bytes()
            catalog.probe_staged_removal(self.repo, lambda: None)
            self.assertEqual(alternate.read_bytes(), before)
            self.assertEqual(os.environ["GIT_INDEX_FILE"], str(alternate))
        self.assertEqual((self.repo / ".git/index").read_bytes(), original)



class ExtractorTests(unittest.TestCase):
    def test_template_interpolation_dependencies(self):
        positives = [
            'const x = `${import("@hyprstream/docs")}`;',
            'const x = `${require("./generated/foo.capnp")}`;',
            'const x = `${`nested ${require("./generated/foo.capnp")}`}`;',
            'const x = `${({a: "}", b: /[{}]/, c: /* } */ import("@hyprstream/docs")})}`;',
            'const x = `${1 + 2} text ${import("@hyprstream/docs")}`;',
            'import(`@hyprstream/docs`);',
        ]
        negatives = [
            'const x = `inert import("@hyprstream/docs")`;',
            r'const x = `\${import("@hyprstream/docs")}`;',
            'const x = `${/require("fixture.capnp")/.test(x)}`;',
            'const x = `${ /* import("@hyprstream/docs") */ 1}`;',
            '''const x = `${'import("@hyprstream/docs")'}`;''',
            'const x = `${`nested inert require("fixture.capnp")`}`;',
        ]
        for expected, cases in [(True, positives), (False, negatives)]:
            for source in cases:
                with self.subTest(source=source):
                    self.assertEqual(catalog.ts_source_is_consumer(source), expected)
                    code, strings = catalog.js_code_and_strings(source)
                    self.assertEqual(len(code), len(source))
                    for offset, body in strings.items():
                        self.assertEqual(source[offset:offset + len(body)], body)

    def test_cli_ambiguous_bindings_fail_closed(self):
        path = "crates/hyprstream/src/cli/schema_cli.rs"
        original = catalog.text(ROOT, path, None)
        catalog.source_services(ROOT)  # Existing supported declarations stay valid.
        for name, expression in [
            ("registry_methods", "extract_methods!(model_client::schema_metadata())"),
            ("registry_tree", "model_client::scoped_client_tree()"),
        ]:
            declaration = f"let {name} = {expression};"
            for label, source in [
                ("unused later helper", original + f"\nfn unused() {{ {declaration} }}\n"),
                ("later same-scope shadow", original),
                ("nested-block shadow", original.replace("    let registry_methods", f"    {{ {declaration} }}\n    let registry_methods", 1)),
                ("unsupported shadow expression", original + f"\nfn unused() {{ let {name} = other(); }}\n"),
                ("function parameter", original + f"\nfn unused({name}: Fake) {{}}\n"),
                ("closure parameter", original + f"\nfn unused() {{ let f = |{name}| 1; }}\n"),
            ]:
                if label == "later same-scope shadow":
                    # Insert after the earlier registry registration, regardless
                    # of neighboring comments/registration formatting.
                    insertion = original.index("    let model_methods")
                    source = original[:insertion] + f"    {declaration}\n" + original[insertion:]
                with self.subTest(binding=name, case=label):
                    self.assertNotEqual(source, original)
                    with self.assertRaisesRegex(catalog.CatalogError, "ambiguous CLI binding"):
                        catalog.source_services(ROOT, {path: source})

    def test_cli_binding_must_precede_and_share_lexical_scope(self):
        path = "crates/hyprstream/src/cli/schema_cli.rs"
        original = catalog.text(ROOT, path, None)
        declaration = "let registry_methods = extract_methods!(registry_client::schema_metadata());"
        for label, source in [
            ("later", original.replace(declaration, "", 1).replace("    let model_methods", f"    {declaration}\n    let model_methods", 1)),
            ("unrelated scope", original.replace(declaration, "", 1) + f"\nfn unused() {{ {declaration} }}\n"),
        ]:
            with self.subTest(case=label), self.assertRaisesRegex(catalog.CatalogError, "not visible"):
                catalog.source_services(ROOT, {path: source})

    def test_exclusion_records(self):
        catalog.check_publication_policy(CORPUS)
        for reason in [None, "", "  ", 1]:
            for duplicate in [False, True]:
                for before in [False, True]:
                    with self.subTest(reason=reason, duplicate=duplicate, before=before):
                        trial = copy.deepcopy(CORPUS)
                        item = dict(trial["excluded"][0], reason=reason)
                        if duplicate:
                            trial["excluded"].insert(0 if before else len(trial["excluded"]), item)
                        else:
                            trial["excluded"][0] = item
                        with self.assertRaises(catalog.CatalogError):
                            catalog.check_publication_policy(trial)
        trial = copy.deepcopy(CORPUS); trial["excluded"].append(copy.deepcopy(trial["excluded"][0]))
        with self.assertRaises(catalog.CatalogError):
            catalog.check_publication_policy(trial)

    def test_scoped_hidden_and_decoys(self):
        schema = '''struct DemoRequest { union {
  root @0 :Void $cliHidden;
  first @1 :InnerRequest;
  second @2 :InnerRequest;
} }
struct DemoResponse { union {
  rootResult @0 :Void;
  firstResult @1 :InnerResponse;
  secondResult @2 :InnerResponse;
} }
struct InnerRequest { union {
  same @0 :Void $cliHidden;
  deeper @1 :DeepRequest;
  decoy @2 :Void $note("$cliHidden");
  commented @3 :Void; # $cliHidden
} }
struct InnerResponse { union {
  same @0 :Void;
  deeperResult @1 :DeepResponse;
} }
struct DeepRequest { union {
  same @0 :Void $cliHidden;
} }
struct DeepResponse { union {
  same @0 :Void;
} }
'''
        result = catalog.schema_method_metadata(ROOT, [{"service": "demo", "path": "fake.capnp"}], {"fake.capnp": schema})
        self.assertEqual(result["cli_hidden"], ["demo.first.deeper.same", "demo.first.same", "demo.root",
                                                 "demo.second.deeper.same", "demo.second.same"])

    def test_commented_hidden_field_stops_before_next_field(self):
        actual = catalog.read_json(ROOT / "docs/schema-catalog.json", ROOT)
        path = "crates/hyprstream-workers/schema/worker.capnp"
        source = catalog.text(ROOT, path, None)
        baseline = catalog.schema_method_metadata(ROOT, actual["schemas"], None)
        self.assertIn("worker.container.attach", baseline["cli_hidden"])
        for replacement in ["# $cliHidden", '$mutationSemantics("$cliHidden")']:
            with self.subTest(replacement=replacement):
                changed = source.replace("$cliHidden", replacement, 1)
                methods = catalog.schema_method_metadata(ROOT, actual["schemas"], {path: changed})
                self.assertNotIn("worker.container.attach", methods["cli_hidden"])
                self.assertIn("worker.container.detach", methods["cli_hidden"])

    def test_workflow_contract(self):
        source = (ROOT / ".github/workflows/docs-catalog.yml").read_text()
        for required in ["[self-hosted, linux, arm64, graviton, hyprstream-merge-gate]", "load-builder-image.sh",
                         'podman run --rm', "--recover-source-objects", "--self-test", "test-docs-catalog.py",
                         "persist-credentials: false", "fetch-depth: 0", "DOCS_CATALOG_AUDITED_HEAD",
                         "contents: read"]:
            self.assertIn(required, source)
        for forbidden in ["ubuntu-latest", "id-token:", "configure-aws-credentials", "continue-on-error", " || true", "cargo "]:
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
