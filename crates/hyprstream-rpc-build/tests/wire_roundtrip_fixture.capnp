# Fixture schema for the ts_codegen cross-implementation round-trip test (#807).
#
# Every field here is a shape the generated `capnp.ts` runtime + codegen
# *support* — primitives, Text/Data, all flat primitive lists, Text/Data lists,
# doubly-nested primitive lists (`List(List(Float32))` / `List(List(Float64))`),
# and a union with a Void / scalar / Text / list / group arm. Wire shapes the
# codegen deliberately rejects (struct lists, nested `List(List(Text))`) are
# intentionally absent — #725's hard-fail-at-generation invariant covers those;
# this fixture covers the *supported* half (#807): that supported shapes are
# encoded correctly across the Rust `capnp` reference and the generated TS.
#
# `bigF32s` / `bigEmbeds` exist to push the TS arena past its ~1KB initial slack
# so growth/reallocation paths (the stale-buffer-capture + pointer-offset bug
# classes from #725/#772) actually execute.

@0xb1175a70b1175a70;

struct WireRoundtrip {
  flag    @0  :Bool;
  u8      @1  :UInt8;
  u16     @2  :UInt16;
  u32     @3  :UInt32;
  u64     @4  :UInt64;
  i8      @5  :Int8;
  i16     @6  :Int16;
  i32     @7  :Int32;
  i64     @8  :Int64;
  f32     @9  :Float32;
  f64     @10 :Float64;
  text    @11 :Text;
  data    @12 :Data;
  bools   @13 :List(Bool);
  u8s     @14 :List(UInt8);
  u16s    @15 :List(UInt16);
  u32s    @16 :List(UInt32);
  u64s    @17 :List(UInt64);
  i8s     @18 :List(Int8);
  i16s    @19 :List(Int16);
  i32s    @20 :List(Int32);
  i64s    @21 :List(Int64);
  f32s    @22 :List(Float32);
  f64s    @23 :List(Float64);
  texts   @24 :List(Text);
  datas   @25 :List(Data);
  embeds  @26 :List(List(Float32));
  bigF32s    @27 :List(Float32);
  bigEmbeds  @28 :List(List(Float64));
}

struct WireChoice {
  union {
    none   @0 :Void;
    count  @1 :UInt32;
    label  @2 :Text;
    values @3 :List(Float32);
    ranged :group {
      lo @4 :Float32;
      hi @5 :Float32;
    }
  }
}
