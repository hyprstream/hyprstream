#!/usr/bin/env node
// Adversarial probe for verify-rpc-std-wasm-package.mjs's tarball-manifest
// hardening. Hand-crafts a tarball containing TWO `package/package.json`
// entries (first a valid staging dev version, second a production semver) —
// exactly the shape npm pack itself can never emit, but that a tampered or
// hand-built archive could present. Runs the REAL verifier CLI as a
// subprocess (not a reimplementation) against it and asserts rejection.
//
// This is causal, not just "does it fail": a CONTROL case packages the exact
// same first manifest entry alone (no duplicate) through the same verifier
// invocation and asserts SUCCESS. That isolates the failure to the duplicate
// entry specifically — ruling out an unrelated setup mistake (bad hash,
// wrong channel regex, etc.) silently making both cases fail for the wrong
// reason.
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import zlib from 'node:zlib';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VERIFIER = path.join(HERE, 'verify-rpc-std-wasm-package.mjs');
const PACKAGE_NAME = '@hyprstream/rpc';
const STAGING_VERSION = '0.1.0-dev.1.abc1234';
const PRODUCTION_VERSION = '9.9.9';

// Read the same authoritative source the verifier itself now checks against,
// rather than hardcoding a license string here — a hardcoded value would
// silently stop testing anything real the next time the crate's declared
// license changes (the exact staleness class this whole check exists to
// catch). A tarball manifest with any OTHER value is guaranteed wrong.
function readCrateLicense() {
  const cargoTomlPath = path.join(HERE, '..', '..', 'crates', 'hyprstream-rpc-std', 'Cargo.toml');
  const lines = fs.readFileSync(cargoTomlPath, 'utf8').split(/\r?\n/);
  let inPackageTable = false;
  for (const line of lines) {
    const tableHeader = line.match(/^\[([^\]]*)\]\s*$/);
    if (tableHeader) {
      inPackageTable = tableHeader[1] === 'package';
      continue;
    }
    if (!inPackageTable) continue;
    const licenseLine = line.match(/^license\s*=\s*"([^"]+)"/);
    if (licenseLine) return licenseLine[1];
  }
  throw new Error(`no license field in [package] table of ${cargoTomlPath}`);
}
const CRATE_LICENSE = readCrateLicense();
const WRONG_LICENSE = CRATE_LICENSE === 'MIT' ? 'Apache-2.0' : 'MIT';

function tarHeader(name, size) {
  const buf = Buffer.alloc(512);
  buf.write(name, 0, 100, 'utf8');
  buf.write(`${size.toString(8).padStart(11, '0')}\0`, 124, 12, 'ascii');
  buf.write('0', 156, 1, 'ascii'); // typeflag '0' == regular file
  return buf;
}

function tarEntry(name, contentBuf) {
  const header = tarHeader(name, contentBuf.length);
  const pad = (512 - (contentBuf.length % 512)) % 512;
  return Buffer.concat([header, contentBuf, Buffer.alloc(pad)]);
}

function buildTarGz(entries) {
  const body = Buffer.concat(entries);
  const end = Buffer.alloc(1024); // two all-zero end-of-archive blocks
  return zlib.gzipSync(Buffer.concat([body, end]));
}

function manifestBuf(version, license = CRATE_LICENSE) {
  return Buffer.from(JSON.stringify({ name: PACKAGE_NAME, version, license }), 'utf8');
}

function writeCase(dir, entries, version, manifestSize = manifestBuf(version).length) {
  fs.mkdirSync(dir, { recursive: true });
  const gz = buildTarGz(entries);
  const tarballName = `hyprstream-rpc-${version}.tgz`;
  const tarballPath = path.join(dir, tarballName);
  fs.writeFileSync(tarballPath, gz);
  const sha256 = crypto.createHash('sha256').update(gz).digest('hex');
  const integrity = `sha512-${crypto.createHash('sha512').update(gz).digest('base64')}`;
  const evidence = {
    schema: 1,
    package: PACKAGE_NAME,
    version,
    channel: 'staging',
    registry: 'https://registry.npmjs.org',
    tarball: tarballName,
    sha256,
    integrity,
    npmShasum: crypto.createHash('sha1').update(gz).digest('hex'),
    source: { repository: 'https://github.com/hyprstream/hyprstream', commit: '0'.repeat(40), ref: 'refs/heads/main', workflow: null, crate: 'crates/hyprstream-rpc-std' },
    build: { target: 'web', profile: 'release', rustToolchain: '1.97.0', wasmPack: '0.13.1' },
    files: [{ path: 'package.json', size: manifestSize }],
  };
  fs.writeFileSync(path.join(dir, 'integrity.json'), `${JSON.stringify(evidence, null, 2)}\n`);
  return tarballPath;
}

function runVerifier(distDir) {
  return spawnSync(process.execPath, [VERIFIER, distDir, 'staging'], {
    encoding: 'utf8',
    env: { ...process.env, GITHUB_REF: 'refs/heads/main' },
  });
}

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'rpc-wasm-verify-probe-'));
let failures = [];

// --- Adversarial case: two package/package.json entries -----------------
const adversarialDir = path.join(root, 'adversarial');
writeCase(
  adversarialDir,
  [tarEntry('package/package.json', manifestBuf(STAGING_VERSION)), tarEntry('package/package.json', manifestBuf(PRODUCTION_VERSION))],
  STAGING_VERSION,
);
const adversarialResult = runVerifier(adversarialDir);
if (adversarialResult.status === 0) {
  failures.push('adversarial case: verifier ACCEPTED a tarball with two package/package.json entries (expected rejection)');
} else if (!/more than one .*package\/package\.json/i.test(adversarialResult.stderr)) {
  failures.push(`adversarial case: verifier rejected, but not for the expected duplicate-entry reason. stderr:\n${adversarialResult.stderr}`);
}

// --- Control case: the same first manifest entry, alone ------------------
const controlDir = path.join(root, 'control');
writeCase(controlDir, [tarEntry('package/package.json', manifestBuf(STAGING_VERSION))], STAGING_VERSION);
const controlResult = runVerifier(controlDir);
if (controlResult.status !== 0) {
  failures.push(`control case: verifier REJECTED a single-entry tarball that should have passed. stderr:\n${controlResult.stderr}`);
}

// --- Adversarial case: manifest license differs from the crate's own -----
// Proves a package whose declared license diverges from the authoritative
// crates/hyprstream-rpc-std/Cargo.toml license (e.g. a hardcoded/stale
// value) is refused, even with an otherwise-valid single manifest entry and
// matching hashes. This is the causal negative for the regression that
// motivated the check: a manifest hardcoding one license while the crate's
// Cargo.toml declares another.
const licenseMismatchDir = path.join(root, 'license-mismatch');
const wrongManifest = manifestBuf(STAGING_VERSION, WRONG_LICENSE);
writeCase(licenseMismatchDir, [tarEntry('package/package.json', wrongManifest)], STAGING_VERSION, wrongManifest.length);
const licenseMismatchResult = runVerifier(licenseMismatchDir);
if (licenseMismatchResult.status === 0) {
  failures.push(`license-mismatch case: verifier ACCEPTED a manifest declaring "${WRONG_LICENSE}" against a crate declaring "${CRATE_LICENSE}" (expected rejection)`);
} else if (!/manifest license .* != authoritative crate license/i.test(licenseMismatchResult.stderr)) {
  failures.push(`license-mismatch case: verifier rejected, but not for the expected license-mismatch reason. stderr:\n${licenseMismatchResult.stderr}`);
}

fs.rmSync(root, { recursive: true, force: true });

if (failures.length > 0) {
  for (const f of failures) console.error(`FAIL: ${f}`);
  throw new Error(`verify-rpc-std-wasm-package.probe.mjs: ${failures.length} assertion(s) failed`);
}
console.log('PASS: verifier rejects duplicate package/package.json entries, rejects a manifest license diverging from the crate\'s own, and still accepts a correct single-entry tarball');
