#!/usr/bin/env node
// Write the docs-publish manifest. Same schema as cyberdione-corp's
// ci/ipfs-write-manifest.mjs (cyberdione.ipfs-release.v1) so the promoter
// can consume both uniformly; adds generatedAt. `pinned` is always false:
// this repo never pins — the promoter pins after re-deriving the CID from
// the CAR bytes, so this field only records that no pin was attempted here.

import { writeFileSync } from 'node:fs';

const [outPath, cid, sha256, packerVersion, sourceDir, dnslinkValue,
  commit, ref, runId, runAttempt] = process.argv.slice(2);
if (!runAttempt || !outPath) process.exit(2);

const manifest = {
  schema: 'cyberdione.ipfs-release.v1',
  site: 'hyprstream-docs',
  cid,
  cidVersion: 1,
  car: { path: 'site.car', sha256 },
  packer: `ipfs-car@${packerVersion}`,
  sourceDir,
  pinned: false,
  dnslink: { value: dnslinkValue, applied: false },
  ci: { commit, ref, runId, runAttempt, project: 'hyprstream/hyprstream' },
  generatedAt: new Date().toISOString(),
  attestation: null,
};
const json = `${JSON.stringify(manifest, null, 2)}\n`;
JSON.parse(json);
writeFileSync(outPath, json, { mode: 0o644 });
