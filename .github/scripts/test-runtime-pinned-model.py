#!/usr/bin/env python3
"""Offline disposable tests for the exact final-image prerequisite probe."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / '.github/scripts/check-runtime-pinned-model.sh'


class RuntimePrerequisiteTests(unittest.TestCase):
    def test_real_build_context_includes_only_the_runtime_probe(self):
        builder = shutil.which('podman') or shutil.which('docker')
        if builder is None:
            self.skipTest('Docker or Podman required for the build-context integration fixture')
        with tempfile.TemporaryDirectory(prefix='runtime-context-', dir=Path.home()) as directory:
            context = Path(directory) / 'context'
            context.mkdir()
            probe = '.github/scripts/check-runtime-pinned-model.sh'
            excluded = ('.github/workflows/private.yml', '.github/scripts/other.sh',
                        '.github/scripts/other.py', '.env', 'private.key', 'other.sh')
            for name in (probe, *excluded):
                path = context / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(CHECK.read_bytes() if name == probe else b'excluded fixture\n')
            shutil.copyfile(ROOT / '.dockerignore', context / '.dockerignore')
            (context / 'Dockerfile').write_text(
                'FROM scratch\nCOPY ' + probe + ' /probe\nCOPY . /context\n')
            output = Path(directory) / 'output'
            pull = '--pull=never' if Path(builder).name == 'podman' else '--pull=false'
            result = subprocess.run(
                [builder, 'build', pull, '--network=none', '--no-cache',
                 '--output', f'type=local,dest={output}', str(context)],
                capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual((output / 'probe').read_bytes(), CHECK.read_bytes())
            self.assertEqual((output / 'context' / probe).read_bytes(), CHECK.read_bytes())
            for name in excluded:
                self.assertFalse((output / 'context' / name).exists(), name)

    def run_check(self, git, cache):
        return subprocess.run(['/bin/sh', str(CHECK), str(git), str(cache)],
                              capture_output=True, timeout=15)

    def test_real_git_batch_and_private_cache(self):
        # Persistent home-backed fixture, never /tmp tmpfs or a Cargo target.
        with tempfile.TemporaryDirectory(prefix='runtime-prereq-', dir=Path.home()) as directory:
            cache = Path(directory) / 'cache'
            cache.mkdir(mode=0o700)
            result = self.run_check(shutil.which('git'), cache)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertIn(b'runtime-pinned-model: PASS', result.stdout)
            self.assertEqual(list(cache.iterdir()), [])
            cache.chmod(0o755)
            self.assertNotEqual(self.run_check(shutil.which('git'), cache).returncode, 0)
            cache.chmod(0o700)
            self.assertNotEqual(self.run_check(Path(directory) / 'missing-git', cache).returncode, 0)
            self.assertNotEqual(self.run_check(shutil.which('git'), Path(directory) / 'missing-cache').returncode, 0)
            link = Path(directory) / 'linked-cache'
            link.symlink_to(cache)
            self.assertNotEqual(self.run_check(shutil.which('git'), link).returncode, 0)

    def test_tmpfs_cache_is_rejected_when_available(self):
        if not Path('/dev/shm').is_dir():
            self.skipTest('no tmpfs fixture mount')
        with tempfile.TemporaryDirectory(prefix='runtime-prereq-', dir='/dev/shm') as directory:
            self.assertNotEqual(self.run_check(shutil.which('git'), directory).returncode, 0)

    def test_production_probe_follows_nonroot_user_and_real_closure_copy(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        runtime = dockerfile.split('FROM runtime-${VARIANT} AS runtime', 1)[1]
        self.assertIn('COPY --from=builder /out/runtime/ /', runtime)
        cache_copy = ('COPY --from=builder --chown=${RUNTIME_UID}:${RUNTIME_GID} '
                      '/out/runtime/var/cache/hyprstream-pinned /var/cache/hyprstream-pinned')
        self.assertIn(cache_copy, runtime)
        self.assertLess(runtime.index('ARG RUNTIME_UID'), runtime.index(cache_copy))
        self.assertLess(runtime.index('ARG RUNTIME_GID'), runtime.index(cache_copy))
        self.assertEqual(runtime.count('--chown='), 1)
        self.assertLess(runtime.index('USER ${RUNTIME_UID}:${RUNTIME_GID}'),
                        runtime.index('RUN /bin/sh /usr/local/libexec/check-runtime-pinned-model'))
        self.assertIn('ldd /usr/bin/git > /out/git.ldd', dockerfile)
        self.assertIn('libc.so.*|ld-linux*) continue', dockerfile)
        self.assertIn('install -d -m 0700 -o "$RUNTIME_UID" -g "$RUNTIME_GID"', dockerfile)


if __name__ == '__main__':
    unittest.main()
