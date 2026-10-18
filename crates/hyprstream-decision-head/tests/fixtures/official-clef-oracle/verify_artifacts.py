#!/usr/bin/env python3
"""Fresh, bounded-memory verification of every pinned official oracle input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

REVISION = '8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3'

REQUIRED_PATHS = frozenset([
    '.gitattributes', 'LICENSE', 'README.md', 'chat_template.jinja', 'config.json',
    'generation_config.json', 'joint_head.safetensors', 'joint_head_config.json',
    'joint_schema_model.py', 'model.safetensors.index.json', 'processor_config.json',
    'tokenizer.json', 'tokenizer_config.json',
    *[f'model-{i:05d}-of-00004.safetensors' for i in range(1, 5)],
])


def verify_artifacts(model_dir, manifest, *, source_bytes=None):
    if manifest.get('revision') != REVISION:
        raise ValueError('artifact manifest source revision mismatch')
    items = manifest.get('files')
    if not isinstance(items, list) or len(items) != 17:
        raise ValueError('expected complete seventeen-file artifact manifest')
    if {item.get('path') for item in items} != REQUIRED_PATHS:
        raise ValueError('artifact manifest must contain exactly the pinned seventeen paths')
    checked = {}
    for item in items:
        name = item['path']
        if not isinstance(name, str):
            raise ValueError('invalid artifact manifest path')
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or str(relative) != name or name in checked
                or type(item['bytes']) is not int or item['bytes'] < 0
                or not isinstance(item['sha256'], str)
                or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
                or item.get('verified') is not True):
            raise ValueError('invalid artifact manifest entry')
        path = model_dir / relative
        with path.open('rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size != item['bytes']:
                raise ValueError('artifact size/type mismatch: ' + name)
            identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            capture = source_bytes is not None and name == 'joint_schema_model.py'
            if capture and info.st_size > 4 * 1024 * 1024:
                raise ValueError('pinned source exceeds capture limit')
            pieces = []
            digest = hashlib.sha256()
            left = item['bytes']
            while left:
                chunk = stream.read(min(left, 1024 * 1024))
                if not chunk:
                    raise ValueError('artifact truncated: ' + name)
                digest.update(chunk)
                if capture:
                    pieces.append(chunk)
                left -= len(chunk)
            if stream.read(1):
                raise ValueError('artifact grew: ' + name)
            after = os.fstat(stream.fileno())
        current = path.stat()
        if any((value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
               != identity for value in (after, current)):
            raise ValueError('artifact changed during verification: ' + name)
        actual = digest.hexdigest()
        if actual != item['sha256']:
            raise ValueError('artifact hash mismatch: ' + name)
        checked[name] = actual
        if capture:
            source_bytes[name] = b''.join(pieces)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    checked = verify_artifacts(args.model_dir, json.loads(args.manifest.read_text()))
    for name, digest in checked.items():
        print(name, digest, flush=True)
    print('VERIFIED', len(checked), 'pinned files')


if __name__ == '__main__':
    main()
