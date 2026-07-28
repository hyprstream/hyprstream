#!/usr/bin/env node
// Verify the downloaded build artifact before a publishing job hands it to npm.
// This intentionally performs no network access and accepts no credentials.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

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
if (evidence.schema !== 1 || evidence.package !== 'hyprstream-rpc-std') {
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
  const expectedRef = `refs/tags/hyprstream-rpc-std-v${evidence.version}`;
  if (process.env.GITHUB_REF !== expectedRef) {
    throw new Error(`production ref must be exactly ${expectedRef}`);
  }
  if (evidence.version.includes('-')) {
    throw new Error('production package must use the crate release semver, not a prerelease/floating version');
  }
} else {
  if (process.env.GITHUB_REF !== 'refs/heads/main' || !evidence.version.includes('-dev.')) {
    throw new Error('staging package must be a unique main-branch dev prerelease');
  }
}

console.log(path.join(distDir, tarballs[0]));
