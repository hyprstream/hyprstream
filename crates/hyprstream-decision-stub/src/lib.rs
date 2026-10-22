//! # hyprstream-decision-stub — System One P0.7 stub decision service + wire smoke test
//!
//! A deterministic mock decision service returning **jev-1-shaped, schema-valid answers**
//! behind the two contracts the real serving paths will implement:
//!
//! 1. **The jev-1 JSON/HTTPS facade** ([`facade`]): `POST /v1/systemone` and
//!    `GET /v1/models` with the envelope, answer derivations, and error shapes of the
//!    jev-1 compatibility profile pinned by `hyprstream-decision` (P0.1a). This is the
//!    portability-prove surface: the stock TypeSafe SDKs (`typesafe-sdk-python`,
//!    `@typesafe-ai/sdk`, both MIT) run against it unmodified via `TYPESAFE_BASE_URL`
//!    (S6a §4 — the adapter never speaks the `/v1/systemone` wire, so stock-SDK
//!    repointing is the proof).
//! 2. **The P3.5 Flight SQL/ADBC `decide` contract** ([`flight`]): a prepared statement
//!    pins a question set plus the (schema, model, calib) version triple; bound parameter
//!    batches carry one `state` per row; the result stream is a
//!    `hyprstream-decision` [`arrow`](https://docs.rs) `DecisionSchema` batch. Handles
//!    are content-derived and unknown handles fail loudly (`not_found`) — the drift
//!    semantics P3.5 requires of prepared statements.
//!
//! The mock ([`mock`]) is **deterministic**: distributions are a hash of
//! (model id, canonical question serialization, state, row index), so every byte on the
//! wire is reproducible and the wire-overhead floor (`tests/wire_floor.rs`) can pin exact
//! sizes. It is not a model — it exists so contract conformance and overhead are
//! measurable before P1.x lands.
//!
//! ## Facade decision — bare bool/number entries (deferred from P0.1a, owned by P0.7)
//!
//! The upstream jev-1 `EntryType` is `string | object | array | null`; bare booleans and
//! numbers in entry positions are outside it. `hyprstream-decision`'s IR admits
//! [`Entry::Bool`](hyprstream_decision::entry::Entry) / `Entry::Number` as a documented
//! superset-permissive extension (structured criteria routinely carry numeric
//! thresholds). **Decision: the facade ACCEPTS bare booleans and finite numbers in every
//! entry position** (`state`, `instructions`, criteria values) and treats them as the
//! IR's native scalars — canonical serialization renders them as their JSON literals
//! (`true`, `42.5`). It does **not** stringify them (that would change rubric text the
//! encoder trains on) and does **not** reject them (the IR, the Arrow contract, and the
//! scorer all handle them). Consequence, documented for P3.6 migration recipes: a
//! request using bare scalars is valid against this superset profile but is **not
//! portable to the upstream service**, whose schema admits only string/object/array/null.
//!
//! ## What the stub deliberately does not do
//!
//! No auth (any Bearer token is accepted; a missing one is a 401), no rate limiting, no
//! persistence, no calibration. `usage.input_tokens` is a documented stub approximation
//! (`ceil(serialized_bytes / 4)` over the canonical serialization — S6a A11 leaves the
//! tokenizer unspecified; consumers must not compare counts cross-service), and
//! `output_tokens` is the sum of answered cardinalities.

pub mod facade;
pub mod flight;
pub mod mock;
pub mod render;
pub mod wire;
