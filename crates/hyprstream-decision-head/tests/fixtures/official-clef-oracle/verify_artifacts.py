#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text());assert m['revision']=='8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3'
for item in m['files']:
 f=a.model_dir/item['path'];assert f.stat().st_size==item['bytes'],str(f)
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 assert h.hexdigest()==item['sha256'],str(f)
 print(item['path'],h.hexdigest(),flush=True)
print('VERIFIED',len(m['files']),'pinned files')
