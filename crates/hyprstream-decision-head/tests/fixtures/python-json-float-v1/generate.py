#!/usr/bin/env python3
"""Offline CPython json.dumps float oracle; standard library only, never serving."""
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import struct
import sys

root = Path(__file__).resolve().parent
rng = random.Random(0x434C45465F463634)
bits = {0, 1, 2, 0x000fffffffffffff, 0x0010000000000000,
        0x0010000000000001, 0x7fefffffffffffff, 0x7feffffffffffffe}
for exponent in range(0, 2047, 7):
    for mantissa in (0, 1, (1 << 52) - 1, rng.getrandbits(52)):
        bits.add((exponent << 52) | mantissa)
for _ in range(2048):
    value = rng.getrandbits(63)
    if value >> 52 != 2047:
        bits.add(value)
for center in (0x430fa36fd398d412, 0x3f1a36e2eb1c432d,
               0x430c6bf526340000, 0x4341c37937e08000):
    bits.update(range(center - 24, center + 25))
for exponent in range(-323, 309):
    value = float('1e' + str(exponent))
    raw = struct.unpack('>Q', struct.pack('>d', value))[0]
    bits.update((raw - 1, raw, raw + 1))
bits |= {raw | (1 << 63) for raw in tuple(bits)}
rows = []
for raw in sorted(bits):
    value = struct.unpack('>d', raw.to_bytes(8, 'big'))[0]
    assert math.isfinite(value)
    expected = json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True, allow_nan=False)
    rows.append(f'{raw:016x}\t{expected}\n')
asset = ''.join(rows).encode()
(root / 'expected.tsv').write_bytes(asset)
manifest = {
    'version': 'cpython-json-float/1',
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'runtime': sys.version,
    'implementation': platform.python_implementation(),
    'runtime_executable_sha256': hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
    'finite_cases': len(rows),
    'expected_sha256': hashlib.sha256(asset).hexdigest(),
    'seed': '0x434c45465f463634',
    'scope': 'IEEE bits -> exact CPython json.dumps; same strings also test native JSON parse -> bits/render. No nonfinite acceptance, model, tokenizer, or tensor oracle.',
    'command': 'python3 crates/hyprstream-decision-head/tests/fixtures/python-json-float-v1/generate.py',
}
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
