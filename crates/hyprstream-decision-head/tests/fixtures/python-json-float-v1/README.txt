CPYTHON FLOAT JSON ENCODING ORACLE (offline, no model assets)

Generate expected.tsv using CPython standard-library json.dumps only:
  python3 crates/hyprstream-decision-head/tests/fixtures/python-json-float-v1/generate.py
manifest.json binds generator/runtime version and SHA256, deterministic seed,
finite case count and expected TSV SHA256. Each line is hexadecimal IEEE754
binary64 bits, TAB, exact expected JSON float spelling. The 10608 finite cases
cover both signs, random mantissas, exponent/precision boundaries, subnormal and
maximum finite extremes, negativezero, and adjacent values around decimal ties
and Python fixed/scientific cutoffs. Python is never used by native serving.

Two default Rust unit tests consume the SAME committed TSV independently:
- f64::from_bits -> Clef render must equal the CPython string exactly.
- raw Clef wire JSON -> parsed binary64 bits AND render must match exactly.
Neither oracle mirrors the Rust formatter/parser. The wire test also rejects
NaN, Infinity, negativeInfinity and overflow; existing integer range/-0/quoted
number controls remain. The expected TSV is immutable after generation; test
failures are not fixed by rebaselining expected strings from Rust output.

The implementation uses Ryu1.0.20 shortest/ties-even digits with separate Python
presentation (fixed exponent[-4,16), integral.0, signedzero, signed2digitmin
scientificexponents), plus serde_json float_roundtrip for correct input bits.
Cargo features are additive: enabling float_roundtrip at this leaf also enables
correctlyrounded parsing in consumers sharing the same serde_json target build.
It changes prior approximate float bits for affected decimals, intentionally;
raw_value, integer syntax/range guards, object ordering and nonfinite rejection
are retained. No arbitrary_precision/global root manifest change is involved.

This establishes finite numeric encoding parity on the declared corpus, not
model accuracy/calibration, serving/lab acceptance, or exhaustive binary64 proof.
