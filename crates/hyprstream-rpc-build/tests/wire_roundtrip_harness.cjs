// Node driver for the ts_codegen cross-implementation round-trip test (#807).
//
// This is plain CommonJS (no type-check) and is driven by the Rust integration
// test `wire_roundtrip.rs`. It imports the *tsc-compiled* output of the
// generated `capnp.ts` runtime + `wire_roundtrip_fixture.ts` service file, then
// exercises both wire directions against the Rust `capnp` reference impl:
//
//   node wire_roundtrip_harness.cjs rust2ts  <bytesFile>
//     Read a WireRoundtrip message the Rust side serialized into <bytesFile>,
//     parse it with the generated parser, print canonical JSON to stdout.
//
//   node wire_roundtrip_harness.cjs ts2rust <bytesFile>
//     Build the canonical WireRoundtrip with the generated builder (including
//     the growth-stress lists), write the wire bytes to <bytesFile>, and print
//     the TS self-parse (what TS thinks it just built) as canonical JSON.
//
//   node wire_roundtrip_harness.cjs choice-ts2rust <bytesFile> <arm>
//   node wire_roundtrip_harness.cjs choice-rust2ts <bytesFile>
//     Same two directions for the WireChoice union (arm ∈ none|count|label|values|ranged).
//
// Canonical JSON: bigints as decimal strings (JSON has no bigint), Uint8Array as
// number[], nested arrays as-is — the Rust side builds the identical shape from
// the capnp-rust reader and compares with serde_json::Value equality.
'use strict';

const fs = require('fs');
const {
  buildWireRoundtrip,
  parseWireRoundtrip,
  buildWireChoice_none,
  buildWireChoice_count,
  buildWireChoice_label,
  buildWireChoice_values,
  buildWireChoice_ranged,
  parseWireChoice,
} = require('./wire_roundtrip_fixture.js');

const BIG_F32 = 2048; // 8 KB of Float32 — well past the ~1 KB arena slack (#725/#772 growth path)
const BIG_EMBED_OUTER = 4;
const BIG_EMBED_INNER = 512; // 4 × 512 Float64 = 16 KB nested — growth + realloc of the outer shell

/** Convert any parsed value to a JSON-safe canonical form (bigints → string). */
function canon(v) {
  if (typeof v === 'bigint') return v.toString();
  if (v instanceof Uint8Array) return Array.from(v);
  if (Array.isArray(v)) return v.map(canon);
  if (v !== null && typeof v === 'object') {
    const o = {};
    for (const k of Object.keys(v)) o[k] = canon(v[k]);
    return o;
  }
  return v;
}

function out(obj) {
  process.stdout.write(JSON.stringify(canon(obj)));
  process.stdout.write('\n');
}

function readBytes(file) {
  return new Uint8Array(fs.readFileSync(file));
}

// --- canonical WireRoundtrip payload (used for ts2rust) ---------------------

function canonicalRoundtrip() {
  const data = Uint8Array.from({ length: 256 }, (_, i) => i);
  const bigF32s = new Array(BIG_F32);
  for (let i = 0; i < BIG_F32; i++) bigF32s[i] = i * 2; // f32-exact (integers)
  const bigEmbeds = new Array(BIG_EMBED_OUTER);
  for (let i = 0; i < BIG_EMBED_OUTER; i++) {
    const inner = new Array(BIG_EMBED_INNER);
    for (let j = 0; j < BIG_EMBED_INNER; j++) inner[j] = i * 1000 + j * 0.5; // f64-exact
    bigEmbeds[i] = inner;
  }
  return {
    flag: true,
    u8: 200,
    u16: 60000,
    u32: 4000000000,
    u64: 4294967297n, // 0x1_0000_0001 — not exactly representable as JS number
    i8: -5,
    i16: -1000,
    i32: -70000,
    i64: -9007199254740993n, // outside f64's integer range
    f32: 3.5,
    f64: 2.718281828459045,
    text: 'héllo 🦀 wörld',
    data,
    bools: [true, false, true, true, false],
    u8s: [0, 7, 14, 21, 28, 35],
    u16s: [10, 20, 30],
    u32s: [1, 100, 10000],
    u64s: [1n, 100n, 4294967297n],
    i8s: [-1, -2, 127],
    i16s: [], // empty primitive list — exercises the 0-count path
    i32s: [-70000, 0, 70000],
    i64s: [-9007199254740993n, 0n, 9007199254740991n],
    f32s: [1.5, 2.5, 3.5],
    f64s: [1.5, 2.25, 3.125],
    texts: ['a', 'b🦀c', ''], // includes an empty string
    datas: [Uint8Array.from([1, 2, 3]), new Uint8Array(0), Uint8Array.from([255, 254])],
    embeds: [[1.5, 2.5, 3.5], [], [10.0, 20.0]],
    bigF32s,
    bigEmbeds,
  };
}

function main() {
  const [, , mode, arg1, arg2] = process.argv;

  if (mode === 'rust2ts') {
    out(parseWireRoundtrip(readBytes(arg1)));
    return;
  }
  if (mode === 'ts2rust') {
    const p = canonicalRoundtrip();
    const bytes = buildWireRoundtrip(p);
    fs.writeFileSync(arg1, Buffer.from(bytes));
    out(parseWireRoundtrip(bytes)); // TS self-parse — catches build/parse asymmetry (#772 class)
    return;
  }
  if (mode === 'choice-rust2ts') {
    out(parseWireChoice(readBytes(arg1)));
    return;
  }
  if (mode === 'choice-ts2rust') {
    const arm = arg2;
    let bytes;
    switch (arm) {
      case 'none':   bytes = buildWireChoice_none(); break;
      case 'count':  bytes = buildWireChoice_count(42); break;
      case 'label':  bytes = buildWireChoice_label('rënamed 🦀'); break;
      case 'values': bytes = buildWireChoice_values([1.5, 2.5, 3.5, 4.5]); break;
      case 'ranged': bytes = buildWireChoice_ranged({ lo: -1.5, hi: 99.5 }); break;
      default: throw new Error(`unknown arm: ${arm}`);
    }
    fs.writeFileSync(arg1, Buffer.from(bytes));
    out(parseWireChoice(bytes));
    return;
  }
  throw new Error(`unknown mode: ${mode}`);
}

main();
