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
  const far = kind === 'far' || kind === 'malformed-double-far' || kind === 'invalid-far-segment';
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
    if (kind === 'composite-body') word(0, 1, 7); // insufficient body
    if (kind === 'composite-padding') word(0, 1, (2 << 3) | 7);
    if (kind === 'composite-stride') word(1, 1 << 2, (2 << 16) | 1);
    if (kind === 'composite-body-overrun') {
      word(0, 1, (8 << 3) | 7);
      word(1, 1 << 2, (2 << 16) | 1);
    }
    if (kind === 'composite-count-overflow') word(1, 0x80000000, 1 << 16);
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
    if (kind === 'wrong-element-kind') word(0, 1, (3 << 3) | 2);
    if (kind === 'empty') return new StructReader(raw, view, 0, 0, 1, [0]);
    if (kind === 'far') {
      word(1, 2, 1);
      word(4, 5, (3 << 3) | 2);
      raw.set([1, 2, 3], 48);
    } else if (kind === 'invalid-far-segment') {
      word(1, 2, 99);
    } else if (kind === 'malformed-double-far') word(1, 6, 1);
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
  if (kind === 'segment-pointer-slot-overrun') {
    return new StructReader(raw, view, 0, 3, 1, [0], [24]);
  }
  return new StructReader(raw, view, kind === 'segment-payload-before-start' ? 24 : 0, 0, 1, far ? [0, 32] : segmented ? [0, 24] : [0]);
}

function doubleFarDataReader(kind, nested) {
  const container = kind === 'container';
  const rootWords = nested ? 4 : 3;
  const zeroWordObject = kind === 'zero-word-object' || kind === 'zero-word-nonempty-list';
  const objectWords = zeroWordObject ? 0 : container ? 2 : 1;
  const bytes = new Uint8Array(16 + (rootWords + 2 + objectWords) * 8);
  const view = new DataView(bytes.buffer);
  const word = (offset, lo, hi) => { view.setUint32(offset, lo >>> 0, true); view.setUint32(offset + 4, hi >>> 0, true); };
  view.setUint32(0, 2, true);
  view.setUint32(4, rootWords, true);
  view.setUint32(8, kind === 'pad-overrun' ? 1 : 2, true);
  view.setUint32(12, kind === 'object-declared-end' ? 0 : objectWords, true);
  word(16, 0, 1 << 16);
  let field = 24;
  if (nested) { word(field, 0, 1 << 16); field += 8; }
  if (container) word(field, 6, 1);
  else { word(field, 1, (1 << 3) | 6); word(field + 8, 6, 1); }
  const pad = 16 + rootWords * 8;
  const object = pad + 16;
  word(pad, kind === 'pad-kind' ? 0 : kind === 'pad-double' ? 6 : (kind === 'object-overrun' || kind === 'empty-object') ? 10 : 2, kind === 'object-segment' ? 99 : 2);
  word(pad + 8, kind === 'tag-kind' ? 2 : 1, container ? (1 << 3) | 6 : ((kind === 'empty-object' || kind === 'zero-word-object' ? 0 : kind === 'zero-word-nonempty-list' ? 1 : kind === 'payload-overrun' ? 9 : 3) << 3) | 2);
  if (container) { word(object, 1, (3 << 3) | 2); bytes.set([1, 2, 3], object + 8); }
  else if (!zeroWordObject) bytes.set([1, 2, 3], object);
  const root = new CapnpReader(bytes, 0, 1);
  return nested ? root.getStruct(0, 0, 1) : root;
}

function doubleFarStructReader(nested, nonempty, overrun = false, undersized = false) {
  const rootWords = nested ? 3 : 2;
  const objectWords = overrun || undersized ? 1 : 0;
  const bytes = new Uint8Array(16 + (rootWords + objectWords + 2) * 8);
  const view = new DataView(bytes.buffer);
  const word = (offset, lo, hi) => { view.setUint32(offset, lo >>> 0, true); view.setUint32(offset + 4, hi >>> 0, true); };
  view.setUint32(0, 2, true);
  view.setUint32(4, rootWords, true);
  view.setUint32(8, objectWords, true);
  view.setUint32(12, 2, true);
  word(16, 0, 1 << 16);
  if (nested) {
    word(24, 0, 1 << 16);
    word(32, 6, 2);
  } else {
    word(24, 6, 2);
  }
  const pad = 16 + (rootWords + objectWords) * 8;
  word(pad, 2, 1);
  word(pad + 8, 0, overrun ? 2 : undersized || nonempty ? 1 : 0);
  const root = new CapnpReader(bytes, 0, 1);
  return nested ? root.getStruct(0, 0, 1) : root;
}

function doubleFarCompositeReader(kind, nested) {
  const rootWords = nested ? 3 : 2;
  const objectWords = kind === 'tag-overrun' ? 1 : 2;
  const bytes = new Uint8Array(16 + (rootWords + objectWords + 2) * 8);
  const view = new DataView(bytes.buffer);
  const word = (offset, lo, hi) => { view.setUint32(offset, lo >>> 0, true); view.setUint32(offset + 4, hi >>> 0, true); };
  view.setUint32(0, 2, true);
  view.setUint32(4, rootWords, true);
  view.setUint32(8, objectWords, true);
  view.setUint32(12, 2, true);
  word(16, 0, 1 << 16);
  if (nested) {
    word(24, 0, 1 << 16);
    word(32, 6, 2);
  } else {
    word(24, 6, 2);
  }
  const object = 16 + rootWords * 8;
  const pad = object + objectWords * 8;
  word(pad, 2, 1);
  word(pad + 8, 1, kind === 'oversized-zero-word-count' ? 7 : (1 << 3) | 7); // composite list body words
  const structCount = kind === 'body-mismatch' ? 2 : 1;
  word(object, structCount << 2, 1); // one data word per element
  if (kind === 'oversized-zero-word-count') {
    // A zero-word struct body can otherwise claim an enormous logical count
    // while consuming no segment words.
    word(object, (65_537 << 2) >>> 0, 0);
  }
  const root = new CapnpReader(bytes, 0, 1);
  return nested ? root.getStruct(0, 0, 1) : root;
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
  } else if (kind === 'invalid-outer-kind') {
    word(16, 0, 1); // non-null struct pointer in a List(Data) slot
  } else if (kind === 'empty') {
    word(16, 0, 0);
  } else if (kind.startsWith('composite')) {
    const bodyWords = kind === 'composite-body' ? 0 : kind === 'composite-padding' ? 2 : kind === 'composite-body-overrun' ? 8 : 1;
    word(16, 1, (bodyWords << 3) | 7);
    word(24, kind === 'composite-tag-kind' ? 5 : 1 << 2, kind === 'composite-shape' ? 0 : 1 << 16);
    word(32, 5, (3 << 3) | 2);
    raw.set([1, 2, 3], 48);
    if (kind === 'composite-stride' || kind === 'composite-body-overrun') word(24, 1 << 2, (2 << 16) | 1);
    if (kind === 'composite-count-overflow') word(24, 0x80000000, 1 << 16);
  } else if (kind === 'list-before-start') {
    word(16, ((-3 << 2) | 1) >>> 0, (1 << 3) | 6);
  } else if (kind === 'null-child') {
    word(16, 1, (1 << 3) | 6);
    // A null List(Data) element is preserved as an empty byte vector.
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
  if (mode === 'evolved-root-data-rust2ts' || mode === 'evolved-nested-data-rust2ts') {
    const reader = new CapnpReader(readBytes(arg1), 0, 1);
    const oldReader = mode === 'evolved-root-data-rust2ts' ? reader : reader.getStruct(0, 0, 1);
    out(oldReader.getDataList(0));
    return;
  }
  if (mode === 'double-far-data-lists') {
    const output = {};
    for (const nested of [false, true]) {
      const valid = {};
      for (const kind of ['child', 'container', 'empty-object', 'zero-word-object']) valid[kind] = canon(doubleFarDataReader(kind, nested).getDataList(0));
      const rejected = [];
      for (const kind of ['pad-overrun', 'pad-kind', 'pad-double', 'object-segment', 'tag-kind', 'object-overrun', 'payload-overrun', 'object-declared-end', 'zero-word-nonempty-list']) {
        try { doubleFarDataReader(kind, nested).getDataList(0); } catch { rejected.push(kind); }
      }
      output[nested ? 'nested' : 'root'] = { valid, rejected };
    }
    out(output);
    return;
  }
  if (mode === 'double-far-zero-word-structs') {
    const output = {};
    for (const nested of [false, true]) {
      const empty = doubleFarStructReader(nested, false).getStruct(0, 1, 1);
      const nonempty = doubleFarStructReader(nested, true).getStruct(0, 1, 0);
      const undersized = doubleFarStructReader(nested, false, false, true).getStruct(0, 2, 0);
      output[nested ? 'nested' : 'root'] = {
        validEmpty: empty !== null,
        absentDataDefaults: empty?.getUint32(0) === 0,
        absentPointerDefaults: empty?.getDataList(0).length === 0,
        undersizedDataDefaults: undersized?.getUint32(8) === 0,
        rejectsNonempty: nonempty === null,
      };
    }
    out(output);
    return;
  }
  if (mode === 'double-far-object-shape') {
    const output = {};
    for (const nested of [false, true]) {
      const reader = doubleFarStructReader(nested, false, true);
      output[nested ? 'nested' : 'root'] = reader.getStruct(0, 2, 0) === null;
    }
    out(output);
    return;
  }
  if (mode === 'double-far-composite-lists') {
    const output = {};
    for (const nested of [false, true]) {
      output[nested ? 'nested' : 'root'] = {
        valid: doubleFarCompositeReader('valid', nested).getStructList(0, 1, 0).length,
        tagOverrun: doubleFarCompositeReader('tag-overrun', nested).getStructList(0, 1, 0).length,
        bodyMismatch: doubleFarCompositeReader('body-mismatch', nested).getStructList(0, 1, 0).length,
        oversizedZeroWordCount: doubleFarCompositeReader('oversized-zero-word-count', nested).getStructList(0, 1, 0).length,
      };
    }
    out(output);
    return;
  }
  if (mode === 'zero-word-struct-list-limit') {
    const bytes = new Uint8Array(32);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 0, true); // one segment
    view.setUint32(4, 3, true); // root struct, list pointer, composite tag
    view.setUint32(8, 0, true);
    view.setUint32(12, 1 << 16, true);
    view.setUint32(16, 1, true); // list pointer, zero body words
    view.setUint32(20, 7, true);
    view.setUint32(24, (65_537 << 2) >>> 0, true); // oversized zero-word count
    view.setUint32(28, 0, true);
    let rejected = false;
    try { new CapnpReader(bytes, 0, 1).getStructList(0, 0, 0); } catch { rejected = true; }
    out({ rejected });
    return;
  }
  if (mode === 'far-zero-word-struct-list-limit') {
    // Two segments: root struct + far pointer in segment 0, landing pointer +
    // oversized zero-word composite tag in segment 1.
    const bytes = new Uint8Array(48);
    const view = new DataView(bytes.buffer);
    view.setUint32(0, 1, true); // two segments
    view.setUint32(4, 2, true); // segment 0 words
    view.setUint32(8, 2, true); // segment 1 words
    view.setUint32(12, 0, true); // table padding
    view.setUint32(16, 0, true); // root struct, one pointer
    view.setUint32(20, 1 << 16, true);
    view.setUint32(24, 2, true); // single-far pointer to segment 1 word 0
    view.setUint32(28, 1, true);
    view.setUint32(32, 1, true); // landing pointer to composite tag at word 1
    view.setUint32(36, 7, true); // zero body words
    view.setUint32(40, (65_537 << 2) >>> 0, true);
    view.setUint32(44, 0, true);
    let rejected = false;
    try { new CapnpReader(bytes, 0, 1).getStructList(0, 0, 0); } catch { rejected = true; }
    out({ rejected });
    return;
  }
  if (mode === 'zero-word-composite-reader-limits') {
    const rootBytes = new Uint8Array(40);
    const rootView = new DataView(rootBytes.buffer);
    rootView.setUint32(0, 0, true);
    rootView.setUint32(4, 4, true);
    rootView.setUint32(8, 0, true);
    rootView.setUint32(12, 1 << 16, true);
    rootView.setUint32(16, 1, true);
    rootView.setUint32(20, 7, true);
    rootView.setUint32(24, (65_537 << 2) >>> 0, true);
    rootView.setUint32(28, 0, true);
    const structBytes = new Uint8Array(16);
    const structView = new DataView(structBytes.buffer);
    structView.setUint32(0, 1, true);
    structView.setUint32(4, 7, true);
    structView.setUint32(8, (65_537 << 2) >>> 0, true);
    const root = new CapnpReader(rootBytes, 0, 1);
    const nested = new StructReader(structBytes, structView, 0, 0, 1);
    const makeRootListReader = (elementSize) => {
      const bytes = new Uint8Array(24);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, 0, true);
      view.setUint32(4, 2, true);
      view.setUint32(8, 0, true);
      view.setUint32(12, 1 << 16, true);
      view.setUint32(16, 1, true);
      view.setUint32(20, (65_537 << 3) | elementSize, true);
      return new CapnpReader(bytes, 0, 1);
    };
    const makeStructListReader = (elementSize) => {
      const bytes = new Uint8Array(16);
      const view = new DataView(bytes.buffer);
      view.setUint32(0, 1, true);
      view.setUint32(4, (65_537 << 3) | elementSize, true);
      return new StructReader(bytes, view, 0, 0, 1);
    };
    const rejected = (reader, method) => {
      try { reader[method](0); return false; } catch { return true; }
    };
    out({
      rootNestedPrimitive: rejected(root, 'getUint8ListList'),
      structNestedPrimitive: rejected(nested, 'getUint8ListList'),
      rootText: rejected(root, 'getTextList'),
      structText: rejected(nested, 'getTextList'),
      rootNestedPointer: rejected(makeRootListReader(6), 'getUint8ListList'),
      structNestedPointer: rejected(makeStructListReader(6), 'getUint8ListList'),
      rootBoolean: rejected(makeRootListReader(1), 'getBoolList'),
      structBoolean: rejected(makeStructListReader(1), 'getBoolList'),
    });
    return;
  }
  if (mode === 'strided-composite-readers') {
    const makeRoot = (text) => {
      const bytes = new Uint8Array(96);
      const view = new DataView(bytes.buffer);
      const word = (offset, lo, hi) => { view.setUint32(offset, lo >>> 0, true); view.setUint32(offset + 4, hi >>> 0, true); };
      view.setUint32(0, 0, true);
      view.setUint32(4, 10, true);
      word(8, 0, 1 << 16);
      word(16, 1, (4 << 3) | 7);
      word(24, 2 << 2, (1 << 16) | 1);
      word(32, 0x12345678, 0);
      word(40, (2 << 2) | 1, ((text ? 4 : 3) << 3) | 2);
      word(48, 0x87654321, 0);
      word(56, (1 << 2) | 1, ((text ? 4 : 3) << 3) | 2);
      bytes.set(text ? Uint8Array.from(Buffer.from('one\0')) : Uint8Array.from([7, 8, 9]), 64);
      bytes.set(text ? Uint8Array.from(Buffer.from('two\0')) : Uint8Array.from([4, 5, 6]), 72);
      return new CapnpReader(bytes, 0, 1);
    };
    const makeStruct = (text) => {
      const bytes = new Uint8Array(88);
      const view = new DataView(bytes.buffer);
      const word = (offset, lo, hi) => { view.setUint32(offset, lo >>> 0, true); view.setUint32(offset + 4, hi >>> 0, true); };
      word(0, 0x12345678, 0);
      word(8, 1, (4 << 3) | 7);
      word(16, 2 << 2, (1 << 16) | 1);
      word(24, 0x12345678, 0);
      word(32, (2 << 2) | 1, ((text ? 4 : 3) << 3) | 2);
      word(40, 0x87654321, 0);
      word(48, (1 << 2) | 1, ((text ? 4 : 3) << 3) | 2);
      bytes.set(text ? Uint8Array.from(Buffer.from('one\0')) : Uint8Array.from([7, 8, 9]), 56);
      bytes.set(text ? Uint8Array.from(Buffer.from('two\0')) : Uint8Array.from([4, 5, 6]), 64);
      return new StructReader(bytes, view, 0, 1, 1);
    };
    const rootText = makeRoot(true).getTextList(0);
    const structText = makeStruct(true).getTextList(0);
    const rootNested = makeRoot(false).getUint8ListList(0);
    const structNested = makeStruct(false).getUint8ListList(0);
    out({ rootText, structText, rootNested, structNested });
    return;
  }
  if (mode === 'struct-data-lists') {
    const valid = {};
    for (const kind of ['pointer', 'composite', 'composite-padding', 'far', 'empty', 'wrong-element-kind']) {
      valid[kind] = nestedDataReader(kind).getDataList(0).map((bytes) => Array.from(bytes));
    }
    valid['legacy-constructor'] = legacyStructReader().getDataList(0).map((bytes) => Array.from(bytes));
    const rejected = [];
    for (const kind of ['truncated', 'invalid', 'malformed-double-far', 'invalid-far-segment', 'composite-shape', 'composite-body', 'composite-kind', 'composite-stride', 'composite-body-overrun', 'composite-count-overflow', 'segment-list-overrun', 'segment-payload-overrun', 'segment-payload-before-start', 'far-pad-overrun', 'segment-declared-end', 'segment-pointer-slot-overrun']) {
      try { nestedDataReader(kind).getDataList(0); }
      catch { rejected.push(kind); }
    }
    out({ valid, rejected });
    return;
  }
  if (mode === 'root-data-lists') {
    const valid = {};
    for (const kind of ['pointer', 'composite', 'composite-padding', 'far-outer-child', 'null-child', 'empty']) {
      valid[kind] = rootDataReader(kind).getDataList(0).map((bytes) => Array.from(bytes));
    }
    const rejected = [];
    for (const kind of ['invalid-far-pad', 'invalid-outer-kind', 'invalid-child-kind', 'segment-list-overrun', 'segment-payload-overrun', 'payload-before-start', 'list-before-start', 'composite-shape', 'composite-body', 'composite-tag-kind', 'composite-tag-overrun', 'composite-stride', 'composite-body-overrun', 'composite-count-overflow', 'far-child-payload-overrun']) {
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
