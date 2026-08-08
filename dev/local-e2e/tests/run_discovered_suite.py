#!/usr/bin/env python3
"""Run the dev/local-e2e Python contract suite with a discovery floor.

`unittest discover` exits 0 when it collects zero tests. Asserting a pinned
per-module floor here turns a vanished test file (rename, deletion, a
pattern that stops matching) into a hard failure instead of a silent PASS.
"""
import sys
import unittest
from pathlib import Path

# Pinned minimum test count per test_*.py module. Bump the floor when adding
# tests to a module; a drop below it means discovery lost coverage silently.
MIN_TESTS_PER_MODULE = {
    "test_ingest_contract": 6,
    "test_owned_run": 6,
    "test_response_contract": 2,
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main(tests_dir: str) -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern="test_*.py")
    tests = list(flatten(suite))

    by_module = {}
    for test in tests:
        module = type(test).__module__
        by_module[module] = by_module.get(module, 0) + 1

    violations = [
        f"{module}: discovered {by_module.get(module, 0)}, expected >= {minimum}"
        for module, minimum in MIN_TESTS_PER_MODULE.items()
        if by_module.get(module, 0) < minimum
    ]
    if not tests:
        violations.append("discovery collected zero tests")

    if violations:
        print("run_discovered_suite: DISCOVERY FLOOR VIOLATION", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(
            f"  collected {len(tests)} total, by module: {by_module}",
            file=sys.stderr,
        )
        return 1

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        f"run_discovered_suite: discovery floor satisfied "
        f"({len(tests)} tests, by module: {by_module})"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    default_dir = str(Path(__file__).parent)
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else default_dir))
