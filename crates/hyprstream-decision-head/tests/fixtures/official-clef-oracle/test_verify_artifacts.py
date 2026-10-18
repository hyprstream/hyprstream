"""Tiny fresh artifact/oracle admission regressions; stdlib and synthetic files."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import py_compile
import shutil
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('clef_verify_artifacts', HERE / 'verify_artifacts.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
SHARDS = [f'model-{index:05d}-of-00004.safetensors' for index in range(1, 5)]
NAMES = ['.gitattributes', 'LICENSE', 'README.md', 'chat_template.jinja', 'config.json',
         'generation_config.json', 'joint_head.safetensors', 'joint_head_config.json',
         'joint_schema_model.py', *SHARDS, 'model.safetensors.index.json',
         'processor_config.json', 'tokenizer.json', 'tokenizer_config.json']


def fixture(root):
    manifest = {'revision': V.REVISION, 'files': []}
    for name in NAMES:
        data = ('tiny public fixture: ' + name).encode()
        if name == 'joint_schema_model.py':
            data = b'raise RuntimeError("UPSTREAM_FIXTURE_REACHED_AFTER_VERIFICATION")\n'
        elif name == 'model.safetensors.index.json':
            data = json.dumps({'weight_map': {'lm_head.weight': SHARDS[0],
                              **{f'layer{i}': shard for i, shard in enumerate(SHARDS)}}}).encode()
        (root / name).write_bytes(data)
        manifest['files'].append({'path': name, 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(), 'verified': True})
    return manifest


class FreshArtifactTests(unittest.TestCase):
    def test_all_seventeen_inputs_and_four_shards_rehashed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            checked = V.verify_artifacts(root, manifest)
            self.assertEqual(set(checked), set(NAMES))
            for entry in manifest['files']:
                self.assertEqual(checked[entry['path']], entry['sha256'])
            self.assertTrue(set(SHARDS) <= set(checked))

    def test_same_size_corruption_of_every_backbone_shard_rejected(self):
        for shard in SHARDS:
            with self.subTest(shard=shard), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root)
                path = root / shard
                path.write_bytes(b'X' + path.read_bytes()[1:])
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch: ' + shard):
                    V.verify_artifacts(root, manifest)

    def test_complete_manifest_geometry_and_revision_fail_closed(self):
        for mutation in ['missing', 'duplicate', 'revision', 'verified']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root)
                if mutation == 'missing': manifest['files'].pop()
                if mutation == 'duplicate': manifest['files'][-1] = copy.deepcopy(manifest['files'][0])
                if mutation == 'revision': manifest['revision'] = '0' * 40
                if mutation == 'verified': manifest['files'][-1]['verified'] = False
                with self.assertRaises(ValueError): V.verify_artifacts(root, manifest)

    def generator(self, root, manifest, mode, script_dir=HERE):
        manifest_path = root / 'artifact-manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        torch = types.ModuleType('torch')
        torch.set_num_threads = lambda _: None
        torch.set_num_interop_threads = lambda _: None
        torch.manual_seed = lambda _: None
        torch.float32 = 'fixture-float32'
        safe = types.ModuleType('safetensors')
        safe.safe_open = mock.Mock(side_effect=AssertionError('TENSOR_IO_NOT_ADMITTED'))
        tensor = types.ModuleType('safetensors.torch')
        tensor.load_file = mock.Mock(side_effect=AssertionError('TENSOR_IO_NOT_ADMITTED'))
        tensor.save_file = mock.Mock(side_effect=AssertionError('TENSOR_IO_NOT_ADMITTED'))
        transformers = types.ModuleType('transformers')
        transformers.AutoTokenizer = mock.Mock(side_effect=AssertionError('TOKENIZER_NOT_ADMITTED'))
        argv = ['generate.py', '--model-dir', str(root), '--artifact-manifest', str(manifest_path),
                '--output', str(root / 'output'), '--mode', mode]
        if mode == 'full': argv.append('--allow-full-model')
        with mock.patch.dict(sys.modules, {'torch': torch, 'safetensors': safe,
                'safetensors.torch': tensor, 'transformers': transformers}), \
             mock.patch.object(sys, 'argv', argv), mock.patch.object(sys, 'path', [str(script_dir), *sys.path]), \
             contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(str(script_dir / 'generate.py'), run_name='__main__')

    def test_generator_rejects_every_corrupt_shard_before_importing_model_in_both_modes(self):
        for mode in ['head', 'full']:
            for shard in SHARDS:
                with self.subTest(mode=mode, shard=shard), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = fixture(root)
                    path = root / shard
                    path.write_bytes(b'X' + path.read_bytes()[1:])
                    with self.assertRaisesRegex(ValueError, 'artifact hash mismatch: ' + shard):
                        self.generator(root, manifest, mode)
                    self.assertFalse((root / 'output').exists())

    def test_valid_generator_inputs_reach_only_explicit_fixture_import_boundary(self):
        for mode in ['head', 'full']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root)
                with self.assertRaisesRegex(RuntimeError, 'UPSTREAM_FIXTURE_REACHED_AFTER_VERIFICATION'):
                    self.generator(root, manifest, mode)
                self.assertFalse((root / 'output').exists())

    def test_generator_rejects_index_shard_absent_from_verified_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            index = root / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'lm_head.weight': 'unverified.safetensors'}}))
            entry = next(entry for entry in manifest['files'] if entry['path'] == index.name)
            entry.update(bytes=index.stat().st_size, sha256=hashlib.sha256(index.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'model index references an unverified shard'):
                self.generator(root, manifest, 'head')

    def test_same_cardinality_manifest_omissions_fail_before_model_execution(self):
        for mode in ['head', 'full']:
            for omitted in NAMES:
                with self.subTest(mode=mode, omitted=omitted), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    manifest = fixture(root)
                    decoy = root / 'valid-decoy.txt'
                    decoy.write_bytes(b'valid decoy')
                    entry = next(item for item in manifest['files'] if item['path'] == omitted)
                    entry.update(path=decoy.name, bytes=decoy.stat().st_size,
                                 sha256=hashlib.sha256(decoy.read_bytes()).hexdigest())
                    self.assertEqual(len(manifest['files']), 17)
                    self.assertEqual(len({item['path'] for item in manifest['files']}), 17)
                    with self.assertRaisesRegex(ValueError, 'exactly the pinned seventeen paths'):
                        self.generator(root, manifest, mode)

    @staticmethod
    def divergent_cache(path, original):
        # Produce an ordinary timestamp/size-valid pyc for different source.
        poison = b'raise RuntimeError("DIVERGENT_CACHED_BYTECODE_EXECUTED")\n'
        size = max(len(original), len(poison)) + 8
        padded = original + b'#' + b' ' * (size - len(original) - 1)
        path.write_bytes(poison + b'#' + b' ' * (size - len(poison) - 1))
        stamp = path.stat()
        py_compile.compile(str(path), doraise=True)
        path.write_bytes(padded)
        os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        return padded

    def test_divergent_model_and_verifier_pyc_never_execute(self):
        for mode in ['head', 'full']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root)
                source = root / 'joint_schema_model.py'
                data = self.divergent_cache(source, source.read_bytes())
                entry = next(item for item in manifest['files'] if item['path'] == source.name)
                entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
                scripts = root / 'scripts'; scripts.mkdir()
                for name in ['generate.py', 'verify_artifacts.py']:
                    shutil.copyfile(HERE / name, scripts / name)
                helper = scripts / 'verify_artifacts.py'
                self.divergent_cache(helper, helper.read_bytes())
                for cached in [source, helper]:
                    spec = importlib.util.spec_from_file_location('divergent_cache_control', cached)
                    module = importlib.util.module_from_spec(spec)
                    with self.assertRaisesRegex(RuntimeError, 'DIVERGENT_CACHED_BYTECODE_EXECUTED'):
                        spec.loader.exec_module(module)
                # Both files have valid divergent caches: actual generator must
                # execute helper source and the exact verified model buffer.
                with self.assertRaisesRegex(RuntimeError, 'UPSTREAM_FIXTURE_REACHED_AFTER_VERIFICATION'):
                    self.generator(root, manifest, mode, scripts)
                self.assertFalse((root / 'output').exists())

    def test_corrupt_source_rejected_even_with_valid_original_bytecode(self):
        for mode in ['head', 'full']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = fixture(root)
                source = root / 'joint_schema_model.py'
                stamp = source.stat()
                py_compile.compile(str(source), doraise=True)
                source.write_bytes(b'X' + source.read_bytes()[1:])
                os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch: joint_schema_model.py'):
                    self.generator(root, manifest, mode)

    def test_source_buffer_executed_without_reopening_after_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            original_compile = compile
            original_source = (root / 'joint_schema_model.py').read_bytes()
            def observed_compile(data, filename, mode, *args, **kwargs):
                if str(filename) == str(root / 'joint_schema_model.py'):
                    self.assertEqual(data, original_source)
                    (root / 'joint_schema_model.py').write_bytes(b'raise AssertionError("LATE_SOURCE_SUBSTITUTION")')
                return original_compile(data, filename, mode, *args, **kwargs)
            with mock.patch('builtins.compile', side_effect=observed_compile), \
                 self.assertRaisesRegex(RuntimeError, 'UPSTREAM_FIXTURE_REACHED_AFTER_VERIFICATION'):
                self.generator(root, manifest, 'head')

    def test_verifier_cli_works_without_tensor_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            path = root / 'artifact-manifest.json'; path.write_text(json.dumps(manifest))
            result = subprocess.run([sys.executable, '-B', '-S', str(HERE / 'verify_artifacts.py'),
                    '--model-dir', str(root), '--manifest', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('VERIFIED 17 pinned files', result.stdout)
            shard = root / SHARDS[-1]; shard.write_bytes(b'X' + shard.read_bytes()[1:])
            result = subprocess.run([sys.executable, '-B', '-S', str(HERE / 'verify_artifacts.py'),
                    '--model-dir', str(root), '--manifest', str(path)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('artifact hash mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
