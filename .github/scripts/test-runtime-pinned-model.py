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
        self.assertLess(runtime.index('USER ${RUNTIME_UID}:${RUNTIME_GID}'),
                        runtime.index('RUN /bin/sh /usr/local/libexec/check-runtime-pinned-model'))
        self.assertIn('ldd /usr/bin/git > /out/git.ldd', dockerfile)
        self.assertIn('libc.so.*|ld-linux*) continue', dockerfile)
        self.assertIn('install -d -m 0700 -o "$RUNTIME_UID" -g "$RUNTIME_GID"', dockerfile)


if __name__ == '__main__':
    unittest.main()
