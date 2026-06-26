# Optional wrapper types for Cap'n Proto schemas.
#
# Cap'n Proto has no built-in optional for scalar types. These union structs
# provide explicit None/Some semantics without sentinel-value zero-detection.
# Codegen detects these by name prefix + structure validation (option_inner_type()).
#
# Usage: import "/optional.capnp"; then use OptionFloat32, OptionUint32, etc.
# as field types. The codegen generates Option<T> Rust fields automatically.
#
# DO NOT migrate Text $optional fields to OptionText unless empty-string is a
# meaningful distinct value — the existing empty-string convention is simpler.

@0xf2172cd64075b94a;

struct OptionFloat32 { union { none @0 :Void; some @1 :Float32; } }
struct OptionFloat64 { union { none @0 :Void; some @1 :Float64; } }
struct OptionUint32  { union { none @0 :Void; some @1 :UInt32;  } }
struct OptionUint64  { union { none @0 :Void; some @1 :UInt64;  } }
struct OptionInt32   { union { none @0 :Void; some @1 :Int32;   } }
struct OptionInt64   { union { none @0 :Void; some @1 :Int64;   } }
struct OptionText    { union { none @0 :Void; some @1 :Text;    } }
struct OptionData    { union { none @0 :Void; some @1 :Data;    } }
struct OptionBool    { union { none @0 :Void; some @1 :Bool;    } }
