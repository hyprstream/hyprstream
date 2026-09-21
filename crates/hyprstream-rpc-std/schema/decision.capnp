@0xfe1d220e97eb195f;

using Opt = import "/optional.capnp";

# Cap'n Proto schema for the System One decision surface (P0.1b).
#
# jev-1 question-spec + answer wire types, mirroring the IR in
# hyprstream-decision (crates/hyprstream-decision/src/{spec,answer,entry}.rs).
# This is the pure data contract: the InferenceService registration points
# that carry these structs are added by P3.1.
#
# Contract pins inherited from the IR (normative there; restated here):
#   - null = abstained (AnswerValue.abstained).
#   - Choice: 2-255 options, insertion order is canonical (D2/D6).
#   - Score: >= 2 ordered levels, index = level, 0-based (D1).
#   - Entry null vs absent are distinct (D3/D4): OptEntry.none means the key
#     was omitted; OptEntry.some(Entry.null) is an explicit null entry.
#   - Label lists are NOT carried here: they are derivable from the spec
#     (noul ["false","true"], choice option names, score decimal indices).

# Question type tags. span/derived are reserved for profile v2: decoders must
# fail loudly on them rather than misreading the payload.
enum QuestionKind {
  noul    @0;
  choice  @1;
  score   @2;
  span    @3;
  derived @4;
}

# The jev-1 EntryType: string | object | array | null, plus numbers and
# booleans inside structured entries. Maps preserve insertion order (and any
# duplicate keys) exactly as authored.
struct Entry {
  union {
    null   @0 :Void;
    bool   @1 :Bool;
    number @2 :Float64;
    str    @3 :Text;
    seq    @4 :List(Entry);
    map    @5 :List(MapPair);
  }
}

struct MapPair {
  key   @0 :Text;
  value @1 :Entry;
}

# Optionality wrapper: `none` = key absent; `some` carrying Entry.null =
# explicitly null entry (D3/D4 keep those distinct).
struct OptEntry {
  union {
    none @0 :Void;
    some @1 :Entry;
  }
}

struct NoulCriteria {
  onTrue  @0 :OptEntry;
  onFalse @1 :OptEntry;
}

struct OptNoulCriteria {
  union {
    none @0 :Void;
    some @1 :NoulCriteria;
  }
}

struct ChoiceOption {
  name   @0 :Text;
  rubric @1 :OptEntry;
}

struct QuestionSpec {
  id           @0 :Text;
  # Denormalized tag equal to the body union discriminant, kept so routers can
  # dispatch on kind without decoding the body. Decoders must treat a mismatch
  # as a malformed message.
  kind         @1 :QuestionKind;
  instructions @2 :OptEntry;
  body :union {
    noul    @3 :OptNoulCriteria;
    choice  @4 :List(ChoiceOption);
    score   @5 :List(OptEntry);
    span    @6 :Void;
    derived @7 :Void;
  }
}

struct QuestionSet {
  state     @0 :OptEntry;
  questions @1 :List(QuestionSpec);
}

# Answer side. Raw distributions are the primary payload; labels, confidence,
# and expected values are derived downstream (hyprstream-decision confidence
# module), never stored.
struct AnswerValue {
  union {
    abstained @0 :Void;
    noul      @1 :Float32;        # P(true); the 2-wide distribution is [1-p, p]
    choice    @2 :List(Float32);  # one per option, canonical option order
    score     @3 :List(Float32);  # one per level, ascending level order
  }
}

struct QuestionAnswer {
  questionId   @0 :Text;
  value        @1 :AnswerValue;
  # Null pointer = no conformal set on this row (the common v1 case); an
  # abstained answer must not carry a set.
  conformalSet @2 :List(Text);
}

# Batch-level version triple: question-set schema version, resolved model id,
# calibration-fit version. calib uses the explicit OptionText wrapper (not a
# nullable Text pointer): pointer-null collapses to "" in consumers without
# pointer-presence tracking, which would silently flip "uncalibrated" to a
# calibration version of "" on a round-trip.
struct VersionTriple {
  schema @0 :Text;
  model  @1 :Text;
  calib  @2 :Opt.OptionText;   # none = uncalibrated raw distribution
}

struct AnswerRow {
  answers @0 :List(QuestionAnswer);
}

struct DecisionBatch {
  version @0 :VersionTriple;
  rows    @1 :List(AnswerRow);
}
