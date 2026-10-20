@0xf4a271b090dd7691;

# Keep this producer fixture separate from the TypeScript service schema:
# generated services do not expose struct lists, but old List(Data) readers
# must accept their representation after a producer adds fields.
struct Item {
  value @0 :Data;
  extra @1 :UInt64;
  extraPointer @2 :Text;
}
struct NewRoot { values @0 :List(Item); }
struct OldRoot { values @0 :List(Data); }
struct NewOuter { nested @0 :NewRoot; }
struct OldOuter { nested @0 :OldRoot; }
