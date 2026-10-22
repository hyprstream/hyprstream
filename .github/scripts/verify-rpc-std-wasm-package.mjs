#!/usr/bin/env node
// Verify the downloaded build artifact before a publishing job hands it to npm.
// This intentionally performs no network access and accepts no credentials.
//
// The authoritative name/version is the one carried by the IMMUTABLE tarball's
// own package/package.json — npm publishes the tarball manifest, not the
// evidence file. So in addition to recomputing the tarball hashes and comparing
// them to integrity.json, this script extracts the tarball manifest and requires
// its name/version to exactly match the evidence and any operator pre-statement.
// That closes the forged-evidence gap where consistent hashes + a staging dev
// evidence version could front a tarball whose manifest is a production semver.
//
// Optional operator pre-statement (used by the manual staging gate): if the
// EXPECTED_VERSION and/or EXPECTED_SHA256 environment variables are set, the
// tarball manifest's version and the recomputed SHA-256 must match them exactly.
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const PACKAGE_NAME = '@hyprstream/rpc';
const RELEASE_TAG_PREFIX = 'hyprstream-rpc';
const MANIFEST_ENTRY = 'package/package.json';
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
const tarballPath = path.join(distDir, tarballs[0]);
const bytes = fs.readFileSync(tarballPath);
const sha256 = crypto.createHash('sha256').update(bytes).digest('hex');
const integrity = `sha512-${crypto.createHash('sha512').update(bytes).digest('base64')}`;
if (sha256 !== evidence.sha256 || integrity !== evidence.integrity) {
  throw new Error('downloaded tarball failed SHA-256/SRI verification');
}

// --- Authoritative tarball manifest inspection ---
// npm publishes the name/version from package/package.json inside the tarball.
// A build whose evidence claims one version but whose tarball manifest carries
// another must be refused here, regardless of matching hashes.
const manifest = JSON.parse(extractTarEntry(tarballPath, MANIFEST_ENTRY));
if (manifest.name !== evidence.package || manifest.name !== PACKAGE_NAME) {
  throw new Error(`tarball manifest name ${manifest.name} != evidence/evidence-expected ${PACKAGE_NAME}`);
}
if (manifest.version !== evidence.version) {
  throw new Error(`tarball manifest version ${manifest.version} != evidence version ${evidence.version}`);
}
// The published license must be the crate's own, read independently from the
// checked-out source rather than trusted from the tarball or the packaging
// step that produced it — a hardcoded/stale manifest license (e.g. shipped
// after the crate's Cargo.toml moved to a different license) must be refused
// here even if every hash matches.
const crateLicense = readCrateLicense();
if (manifest.license !== crateLicense) {
  throw new Error(`tarball manifest license ${manifest.license} != authoritative crate license ${crateLicense}`);
}
// A publishable package must not declare scripts or runtime dependencies that
// could run on install; the browser bundle is self-contained.
if (manifest.scripts && Object.keys(manifest.scripts).length > 0) {
  throw new Error('tarball manifest must not carry lifecycle scripts');
}
if (manifest.dependencies && Object.keys(manifest.dependencies).length > 0) {
  throw new Error('tarball manifest must not declare runtime dependencies');
}

const stagingVersionRe = /^\d+\.\d+\.\d+-dev\.\d+\.[0-9a-f]{7,}$/;
if (expectedChannel === 'production') {
  const expectedRef = `refs/tags/${RELEASE_TAG_PREFIX}-v${evidence.version}`;
  if (process.env.GITHUB_REF !== expectedRef) {
    throw new Error(`production ref must be exactly ${expectedRef}`);
  }
  if (evidence.version.includes('-')) {
    throw new Error('production package must use the crate release semver, not a prerelease/floating version');
  }
  if (manifest.version !== evidence.version) {
    throw new Error(`production tarball manifest version ${manifest.version} diverges from evidence ${evidence.version}`);
  }
} else {
  if (process.env.GITHUB_REF !== 'refs/heads/main') {
    throw new Error('staging publication is restricted to refs/heads/main');
  }
  // Commit-deterministic dev prerelease on the tarball's OWN version:
  // <crate-version>-dev.<commit-count>.<short-sha>. This refuses a tarball whose
  // manifest is a production semver even if its hashes match forged evidence.
  if (!stagingVersionRe.test(manifest.version)) {
    throw new Error(`staging tarball manifest version must be a commit-deterministic dev prerelease, got ${manifest.version}`);
  }
  if (manifest.version !== evidence.version) {
    throw new Error(`staging tarball manifest version ${manifest.version} diverges from evidence ${evidence.version}`);
  }
}

// Operator pre-statement gate. When the publishing job is driven by a manual
// dispatch that pre-stated the exact identity, refuse any divergence before
// handing the tarball to npm. The pre-statement binds the tarball manifest
// version (what npm actually publishes), not just the evidence metadata.
const expectedVersion = process.env.EXPECTED_VERSION;
const expectedSha256 = process.env.EXPECTED_SHA256;
if ((expectedVersion === undefined) !== (expectedSha256 === undefined)) {
  throw new Error('EXPECTED_VERSION and EXPECTED_SHA256 must be set together');
}
if (expectedVersion !== undefined) {
  if (manifest.version !== expectedVersion) {
    throw new Error(`tarball manifest version ${manifest.version} != operator pre-stated ${expectedVersion}`);
  }
  if (evidence.version !== expectedVersion) {
    throw new Error(`evidence version ${evidence.version} != operator pre-stated ${expectedVersion}`);
  }
}
if (expectedSha256 !== undefined && expectedSha256 !== sha256) {
  throw new Error(`tarball sha256 ${sha256} != operator pre-stated ${expectedSha256}`);
}

console.log(path.join(distDir, tarballs[0]));

// --- Authoritative crate license ------------------------------------------
// Dependency-free: this is one scalar field, not a reason to add a TOML
// parser dependency to a script whose whole point is verifying without one.
// Scoped to the `[package]` table specifically (not a bare whole-file regex)
// so a `license` key belonging to some other table can never be picked up.
function readCrateLicense() {
  const cargoTomlPath = path.join('crates', 'hyprstream-rpc-std', 'Cargo.toml');
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

// --- Minimal dependency-free USTAR/tar reader ---------------------------------
// Reads a gzip-compressed POSIX/USTAR tar and returns the bytes of a single
// regular-file entry. npm pack emits entries under `package/`. Only enough of
// USTAR is implemented to extract a small JSON manifest: header name/size/type,
// the prefix field, and 512-byte block padding. No PAX/GNU long-name support is
// needed for an npm pack of a short path.
//
// Deliberately scans the WHOLE archive rather than returning on the first
// match. npm's own extraction is last-entry-wins for a duplicated path, so a
// tarball carrying two `package/package.json` entries is ambiguous: which one
// is "the" manifest depends on which reader you ask. Rather than pick a
// side (and risk this verifier approving a manifest npm would not actually
// materialize), refuse any tarball with more than one entry at the target
// path outright — an npm pack of a single source directory can never produce
// that, so its presence only indicates a hand-crafted or tampered archive.
function extractTarEntry(gzPath, entryName) {
  const tar = zlib.gunzipSync(fs.readFileSync(gzPath));
  const decode = (buf) => buf.toString('utf8').replace(/\0+$/, '');
  let found;
  let matchCount = 0;
  for (let off = 0; off + 512 <= tar.length; ) {
    const header = tar.subarray(off, off + 512);
    const name = decode(header.subarray(0, 100));
    if (!name) break; // two consecutive zero blocks = end of archive
    const sizeField = decode(header.subarray(124, 136));
    const size = parseInt(sizeField || '0', 8);
    const typeflag = header.subarray(156, 157).toString('utf8');
    const prefix = decode(header.subarray(345, 500));
    off += 512;
    const isRegular = typeflag === '0' || typeflag === '' || typeflag === '\x00';
    if (isRegular) {
      const fullName = prefix ? `${prefix}/${name}` : name;
      if (fullName === entryName) {
        if (Number.isNaN(size) || size < 0 || off + size > tar.length) {
          throw new Error(`corrupt tar entry ${entryName}: bad size ${size}`);
        }
        matchCount += 1;
        if (matchCount > 1) {
          throw new Error(
            `tarball contains more than one ${entryName} entry; refusing an ambiguous manifest ` +
            '(npm extraction is last-entry-wins and could diverge from this check)'
          );
        }
        found = tar.subarray(off, off + size);
      }
    }
    off += Math.ceil(size / 512) * 512; // skip data + padding
  }
  if (!found) {
    throw new Error(`tar entry ${entryName} not found`);
  }
  return found;
}
