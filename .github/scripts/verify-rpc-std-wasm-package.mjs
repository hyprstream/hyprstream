#!/usr/bin/env node
// Verify the downloaded build artifact before a publishing job hands it to npm.
// This intentionally performs no network access and accepts no credentials.
//
// Optional operator pre-statement (used by the manual staging gate): if the
// EXPECTED_VERSION and/or EXPECTED_SHA256 environment variables are set, the
// artifact's evidence must match them exactly. This lets a reviewer pre-state
// the identity out-of-band and refuse a build that diverges.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const PACKAGE_NAME = '@hyprstream/rpc';
const RELEASE_TAG_PREFIX = 'hyprstream-rpc';
const [distDir, expectedChannel] = process.argv.slice(2);
if (!distDir || !expectedChannel) {
  throw new Error('usage: verify-rpc-std-wasm-package.mjs <dist-dir> <staging|production>');
}
if (!['staging', 'production'].includes(expectedChannel)) {
  throw new Error(`publishing is forbidden for channel ${expectedChannel}`);
}

const entries = fs.readdirSync(distDir);
const tarballs = entries.filter((entry) => entry.endsWith('.tgz'));
if (tarballs.length !== 1 || !entries.includes('integrity.json') || entries.length !== 2) {
  throw new Error(`artifact must contain exactly one tarball and integrity.json: ${entries.join(', ')}`);
}
const evidence = JSON.parse(fs.readFileSync(path.join(distDir, 'integrity.json'), 'utf8'));
if (evidence.schema !== 1 || evidence.package !== PACKAGE_NAME) {
  throw new Error('unexpected package evidence identity');
}
if (evidence.channel !== expectedChannel) {
  throw new Error(`artifact channel ${evidence.channel} != ${expectedChannel}`);
}
if (evidence.registry !== 'https://registry.npmjs.org') {
  throw new Error(`unapproved npm registry ${evidence.registry}`);
}
if (evidence.tarball !== tarballs[0]) {
  throw new Error('evidence names a different tarball');
}
const bytes = fs.readFileSync(path.join(distDir, tarballs[0]));
const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
const integrity = `sha512-${crypto.createHash('sha512').update(bytes).digest('base64')}`;
if (sha256 !== evidence.sha256 || integrity !== evidence.integrity) {
  throw new Error('downloaded tarball failed SHA-256/SRI verification');
}

if (expectedChannel === 'production') {
  const expectedRef = `refs/tags/${RELEASE_TAG_PREFIX}-v${evidence.version}`;
  if (process.env.GITHUB_REF !== expectedRef) {
    throw new Error(`production ref must be exactly ${expectedRef}`);
  }
  if (evidence.version.includes('-')) {
    throw new Error('production package must use the crate release semver, not a prerelease/floating version');
  }
} else {
  if (process.env.GITHUB_REF !== 'refs/heads/main') {
    throw new Error('staging publication is restricted to refs/heads/main');
  }
  // Commit-deterministic dev prerelease: <crate-version>-dev.<commit-count>.<short-sha>
  if (!/^\d+\.\d+\.\d+-dev\.\d+\.[0-9a-f]{7,}$/.test(evidence.version)) {
    throw new Error(`staging version must be a commit-deterministic dev prerelease, got ${evidence.version}`);
  }
}

// Operator pre-statement gate. When the publishing job is driven by a manual
// dispatch that pre-stated the exact identity, refuse any divergence before
// handing the tarball to npm.
const expectedVersion = process.env.EXPECTED_VERSION;
const expectedSha256 = process.env.EXPECTED_SHA256;
if (expectedVersion !== undefined && expectedVersion !== evidence.version) {
  throw new Error(`artifact version ${evidence.version} != operator pre-stated ${expectedVersion}`);
}
if (expectedSha256 !== undefined && expectedSha256 !== evidence.sha256) {
  throw new Error(`artifact sha256 ${evidence.sha256} != operator pre-stated ${expectedSha256}`);
}
if ((expectedVersion === undefined) !== (expectedSha256 === undefined)) {
  throw new Error('EXPECTED_VERSION and EXPECTED_SHA256 must be set together');
}

console.log(path.join(distDir, tarballs[0]));
