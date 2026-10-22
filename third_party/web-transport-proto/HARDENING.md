# Hyprstream local patch: decoded field-section budget

This is a vendored copy of `web-transport-proto` 0.6.0 from crates.io
(tarball sha256 0225d295c8ac00a2e9a498aefeaf3f3c6186da12a251c938189b15b82ea22808,
upstream git sha1 e36bf2cc6de5cf46621fb7dc3884be86f79c1336, path
web-transport-proto in github.com/moq-dev/web-transport), patched by the
h3-qpack-hardening lane on 2026-10-21.

## Delta vs upstream 0.6.0

`Cargo.toml`: appended an empty `[workspace]` table (vendoring isolation;
lets the copy be tested standalone). Everything else in it is upstream.

`src/qpack.rs`:
- new `DecodeError::FieldSectionTooLarge` variant;
- constants `FIELD_OVERHEAD_BYTES` (32), `MAX_DECODED_FIELD_SECTION_SIZE`
  (256 KiB), `MAX_DECODED_FIELDS` (4096);
- `Headers::decode` now accounts every decoded field line per RFC 9114
  4.2.2 (name + value bytes + 32 octets overhead) and cancels decoding at
  the first field that would exceed either cap — before assembling the
  field into the map. Previously there was no decoded-size or field-count
  limit: a 64 KiB wire HEADERS frame could materialize ~3.2 MB of decoded
  header data (static-table / huffman amplification) with unbounded CPU,
  and `ConnectRequest::decode` on a non-wire-capped buffer could push
  `http::HeaderMap` past its 32768-entry capacity (a panic).
- regression tests (`budget_tests` modules in `src/qpack.rs` and
  `src/connect.rs`): >32768 unique tiny fields, static-table accordion,
  entry-count and decoded-size incremental caps, dynamic-table rejection,
  large-but-legal acceptance.

Rationale and evidence: fleet checkpoint
`/home/birdetta/fleet-coord/execution/resume-20261021/h3-qpack-hardening.md`.

Drop this copy (and the matching `[patch.crates-io]` entry in the workspace
root Cargo.toml) once an upstream release ships equivalent limits.
