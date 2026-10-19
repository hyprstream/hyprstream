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
const { CapnpReader, StructReader } = require('./capnp.js');
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

function nestedDataReader(kind) {
  const far = kind === 'far' || kind === 'double-far';
  const segmented = kind.startsWith('segment-') || kind === 'far-pad-overrun';
  const raw = new Uint8Array(far ? 56 : segmented ? 48 : kind === 'truncated' ? 32 : 40);
  const view = new DataView(raw.buffer);
  const word = (index, lo, hi) => {
    view.setUint32(index * 8, lo >>> 0, true);
    view.setUint32(index * 8 + 4, hi >>> 0, true);
  };
  if (kind.startsWith('composite')) {
    word(0, 1, (1 << 3) | 7);
    word(1, kind === 'composite-kind' ? 5 : (1 << 2) >>> 0, kind === 'composite-shape' ? 0 : 1 << 16);
    word(2, 5, (3 << 3) | 2);
    raw.set([1, 2, 3], 32);
    if (kind === 'composite-body') word(0, 1, (2 << 3) | 7);
  } else if (segmented) {
    if (kind === 'segment-payload-before-start') {
      word(3, 1, (1 << 3) | 6);
      word(4, (1 + ((-3 << 2) >>> 0)) >>> 0, (3 << 3) | 2);
      raw.set([1, 2, 3], 16);
    } else {
      word(0, 1, ((kind === 'segment-list-overrun' ? 3 : 1) << 3) | 6);
      if (kind === 'far-pad-overrun') word(1, 26, 1);
      else word(1, 5, ((kind === 'segment-payload-overrun' ? 8 : 3) << 3) | 2);
      raw.set([1, 2, 3], 24);
    }
  } else {
    word(0, 1, (1 << 3) | 6);
    if (kind === 'empty') return new StructReader(raw, view, 0, 0, 1, [0]);
    if (kind === 'far') {
      word(1, 2, 1);
      word(4, 5, (3 << 3) | 2);
      raw.set([1, 2, 3], 48);
    } else if (kind === 'double-far') word(1, 6, 1);
    else if (kind === 'invalid') word(1, 0, 1);
    else {
      word(1, 5, ((kind === 'truncated' ? 16 : 3) << 3) | 2);
      raw.set([1, 2, 3], 24);
    }
  }
  if (kind === 'segment-declared-end') {
    word(0, 1, (1 << 3) | 6);
    word(1, 5, (3 << 3) | 2);
    raw.set([1, 2, 3], 24);
    return new StructReader(raw, view, 0, 0, 1, [0], [24]);
  }
  return new StructReader(raw, view, kind === 'segment-payload-before-start' ? 24 : 0, 0, 1, far ? [0, 32] : segmented ? [0, 24] : [0]);
}

function legacyStructReader() {
  const raw = new Uint8Array(40);
  const view = new DataView(raw.buffer);
  view.setUint32(0, 1, true);
  view.setUint32(4, (1 << 3) | 6, true);
  view.setUint32(8, 5, true);
  view.setUint32(12, (3 << 3) | 2, true);
  raw.set([1, 2, 3], 24);
  // segmentOffsets and segmentEnds are optional constructor arguments.
  return new StructReader(raw, view, 0, 0, 1);
}

function rootDataReader(kind) {
  if (kind === 'far-outer-child' || kind === 'far-child-payload-overrun') {
    const raw = new Uint8Array(64);
    const view = new DataView(raw.buffer);
    const word = (offset, lo, hi) => {
      view.setUint32(offset, lo >>> 0, true);
      view.setUint32(offset + 4, hi >>> 0, true);
    };
    // Three segments: root struct, far landing pad + list body, and far child
    // landing pad + payload. The table occupies 16 bytes.
    view.setUint32(0, 2, true);
    view.setUint32(4, 2, true);
    view.setUint32(8, 2, true);
    view.setUint32(12, 2, true);
    word(16, 0, 1 << 16); // root struct: zero data words, one pointer word
    word(24, 2, 1); // far pointer to segment 1's landing pad at byte 32
    word(32, 1, (1 << 3) | 6); // near List(Data) pointer, body at byte 40
    word(40, 2, 2); // far child pointer to segment 2's landing pad at byte 48
    word(48, 1, ((kind === 'far-child-payload-overrun' ? 9 : 3) << 3) | 2);
    raw.set([1, 2, 3], 56);
    return new CapnpReader(raw, 0, 1);
  }
  const raw = new Uint8Array(64);
  const view = new DataView(raw.buffer);
  const word = (offset, lo, hi) => {
    view.setUint32(offset, lo >>> 0, true);
    view.setUint32(offset + 4, hi >>> 0, true);
  };
  // One segment begins after the 8-byte segment table. Its declared end can
  // be shorter than the backing buffer to exercise segment-bound checks.
  view.setUint32(0, 0, true);
  view.setUint32(4, kind === 'composite-tag-overrun' ? 2 : kind === 'segment-list-overrun' ? 3 : kind === 'segment-payload-overrun' ? 4 : 7, true);
  word(8, 0, 1 << 16); // root struct: zero data words, one pointer word
  if (kind === 'invalid-far-pad') {
    word(16, (7 << 3) | 2, 0); // landing pad starts at the declared segment end
  } else if (kind === 'empty') {
    word(16, 0, 0);
  } else if (kind === 'composite' || kind === 'composite-shape' || kind === 'composite-body' || kind === 'composite-tag-kind' || kind === 'composite-tag-overrun') {
    const bodyWords = kind === 'composite-body' ? 2 : 1;
    word(16, 1, (bodyWords << 3) | 7);
    word(24, kind === 'composite-tag-kind' ? 5 : 1 << 2, kind === 'composite-shape' ? 0 : 1 << 16);
    word(32, 5, (3 << 3) | 2);
    raw.set([1, 2, 3], 48);
  } else if (kind === 'list-before-start') {
    word(16, ((-3 << 2) | 1) >>> 0, (1 << 3) | 6);
  } else {
    const count = kind === 'segment-list-overrun' ? 2 : 1;
    word(16, 1, (count << 3) | 6);
    const childOffsetWords = kind === 'payload-before-start' ? -4 : kind === 'invalid-child-kind' ? 1 : 1;
    word(24, ((childOffsetWords << 2) | (kind === 'invalid-child-kind' ? 0 : 1)) >>> 0, (3 << 3) | 2);
    raw.set([1, 2, 3], 40);
  }
  return new CapnpReader(raw, 0, 1);
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
  if (mode === 'struct-data-lists') {
    const valid = {};
    for (const kind of ['pointer', 'composite', 'far', 'empty']) {
      valid[kind] = nestedDataReader(kind).getDataList(0).map((bytes) => Array.from(bytes));
    }
    valid['legacy-constructor'] = legacyStructReader().getDataList(0).map((bytes) => Array.from(bytes));
    const rejected = [];
    for (const kind of ['truncated', 'invalid', 'double-far', 'composite-shape', 'composite-body', 'composite-kind', 'segment-list-overrun', 'segment-payload-overrun', 'segment-payload-before-start', 'far-pad-overrun', 'segment-declared-end']) {
      try { nestedDataReader(kind).getDataList(0); }
      catch { rejected.push(kind); }
    }
    out({ valid, rejected });
    return;
  }
  if (mode === 'root-data-lists') {
    const valid = {};
    for (const kind of ['pointer', 'composite', 'far-outer-child', 'empty']) {
      valid[kind] = rootDataReader(kind).getDataList(0).map((bytes) => Array.from(bytes));
    }
    const rejected = [];
    for (const kind of ['invalid-far-pad', 'invalid-child-kind', 'segment-list-overrun', 'segment-payload-overrun', 'payload-before-start', 'list-before-start', 'composite-shape', 'composite-body', 'composite-tag-kind', 'composite-tag-overrun', 'far-child-payload-overrun']) {
      try { rootDataReader(kind).getDataList(0); }
      catch { rejected.push(kind); }
    }
    out({ valid, rejected });
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
