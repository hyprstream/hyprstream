//! MIR — Machine Intelligence Resource — model-naming/classification schema.
//!
//! MIR is the model-identity vocabulary defined by Tiles (tiles.run). It gives a
//! cross-system, reproducible way to classify a model independent of where it
//! lives (HuggingFace repo id, Modelfile `FROM`, local registry name).
//!
//! # Grammar
//!
//! ```text
//! mir:<domain>.<arch>.<variant>:<series>
//! ```
//!
//! Example: `mir:model.transformer.clip-l:stable-diffusion-xl`
//!
//! - `domain`  — one of `dev`, `model`, `ops`, `info`, `arch` (the issue's
//!   authoritative grammar uses `dev/model/ops/info`; Tiles also documents an
//!   `arch`/"architecture" domain, which we accept as a superset).
//! - `arch`    — an architecture taxonomy label (e.g. `resnet`, `lstm`, `vae`,
//!   `transformer`, `moe`, `lora`, ...). Unknown labels are accepted as an
//!   [`Arch::Extension`] (superset behaviour) rather than rejected, so a label
//!   Tiles adds later does not hard-fail us.
//! - `variant` — a free-form, normalized sub-classification (e.g. `clip-l`).
//! - `series`  — the normalized model series (e.g. `stable-diffusion-xl`),
//!   subject to the series-normalization rules below.
//!
//! # Source / fidelity note
//!
//! The taxonomy + normalization rules here are transcribed from the MIR section
//! of <https://www.tiles.run/llms-full.txt> (fetched 2026-06) cross-checked
//! against the issue (#283). The Tiles prose is loose about whether the third
//! dot-field is the "variant" or "series"; the issue's authoritative grammar
//! names the third dot-field `variant` and the post-colon field `series`, which
//! is what we implement. The canonical example
//! `mir:model.transformer.clip-l:stable-diffusion-xl` parses under this grammar.
//!
//! See `docs/interop/tiles-alignment.md` for the internal-id <-> MIR mapping.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The MIR scheme prefix (`mir:`).
pub const MIR_SCHEME: &str = "mir:";

/// Errors produced while parsing or validating a [`Mir`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MirError {
    /// String does not start with the `mir:` scheme.
    #[error("not a MIR identifier (missing `mir:` scheme): {0:?}")]
    MissingScheme(String),
    /// The `domain.arch.variant` body or `series` segment is missing.
    #[error("malformed MIR `{0:?}`: expected `mir:<domain>.<arch>.<variant>:<series>`")]
    Malformed(String),
    /// The domain segment is empty or not a recognized domain.
    #[error("invalid MIR domain {0:?}: expected one of dev/model/ops/info/arch")]
    InvalidDomain(String),
    /// The architecture segment is empty or syntactically invalid.
    #[error("invalid MIR architecture {0:?}")]
    InvalidArch(String),
    /// The variant segment is empty or syntactically invalid.
    #[error("invalid MIR variant {0:?}")]
    InvalidVariant(String),
    /// The series segment is empty or syntactically invalid.
    #[error("invalid MIR series {0:?}")]
    InvalidSeries(String),
}

/// MIR domain — the broadest classification axis, ordered most-specific to
/// most-general per the Tiles spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Domain {
    /// Pre-release items under evaluation.
    Dev,
    /// Publicly released ML models with identifiers.
    Model,
    /// Optimization / manipulation techniques.
    Ops,
    /// Metadata about layer names and settings.
    Info,
    /// Broad, general system-architecture terms (Tiles "Architecture" domain).
    Arch,
}

impl Domain {
    /// Canonical lowercase token for this domain.
    pub fn as_str(self) -> &'static str {
        match self {
            Domain::Dev => "dev",
            Domain::Model => "model",
            Domain::Ops => "ops",
            Domain::Info => "info",
            Domain::Arch => "arch",
        }
    }

    fn parse(s: &str) -> Result<Self, MirError> {
        match s.to_ascii_lowercase().as_str() {
            "dev" => Ok(Domain::Dev),
            "model" => Ok(Domain::Model),
            "ops" => Ok(Domain::Ops),
            "info" => Ok(Domain::Info),
            // Tiles documents the broad-architecture domain as "Architecture";
            // accept both `arch` and `architecture` as a superset.
            "arch" | "architecture" => Ok(Domain::Arch),
            _ => Err(MirError::InvalidDomain(s.to_owned())),
        }
    }
}

impl fmt::Display for Domain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The MIR architecture taxonomy.
///
/// Known labels are transcribed from the Tiles MIR taxonomy. Unknown labels are
/// preserved as [`Arch::Extension`] so MIR remains a superset: a label Tiles
/// adds later (or a project-local extension) parses rather than hard-failing,
/// while canonical emission stays stable (always lowercase).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Arch {
    /// A known taxonomy label, stored canonically (lowercase).
    Known(KnownArch),
    /// An out-of-taxonomy label, preserved verbatim (lowercased).
    Extension(String),
}

/// Architectures enumerated in the Tiles MIR taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KnownArch {
    /// Gated Recurrent Unit.
    Gru,
    /// Restricted Boltzmann Machine.
    Rbm,
    /// Tiny Autoencoder.
    Tae,
    /// Variational Autoencoder.
    Vae,
    /// Long Short-Term Memory.
    Lstm,
    /// Residual Network.
    Resnet,
    /// Convolutional Neural Network.
    Cnn,
    /// Region-based Convolutional Neural Network.
    Rcnn,
    /// Recurrent Neural Network.
    Rnn,
    /// Bi-directional Recurrent Neural Network.
    Brnn,
    /// Generative Adversarial Network.
    Gan,
    /// State-Space Model.
    Ssm,
    /// Detection Transformer.
    Detr,
    /// Vision Transformer.
    Vit,
    /// Mixture of Experts.
    Moe,
    /// Autoencoding Transformer.
    Aet,
    /// Sequence-to-Sequence Transformer.
    Stst,
    /// Autoregressive Transformer.
    Art,
    /// Low-Rank Adaptation.
    Lora,
    /// ControlNet.
    Controlnet,
    /// Unknown / unclassified architecture.
    Unclassified,
    /// Generic transformer (used by the canonical Tiles example; not in the
    /// abbreviation table but a first-class label in the spec's examples).
    Transformer,
}

impl KnownArch {
    /// Canonical lowercase token for this architecture.
    pub fn as_str(self) -> &'static str {
        match self {
            KnownArch::Gru => "gru",
            KnownArch::Rbm => "rbm",
            KnownArch::Tae => "tae",
            KnownArch::Vae => "vae",
            KnownArch::Lstm => "lstm",
            KnownArch::Resnet => "resnet",
            KnownArch::Cnn => "cnn",
            KnownArch::Rcnn => "rcnn",
            KnownArch::Rnn => "rnn",
            KnownArch::Brnn => "brnn",
            KnownArch::Gan => "gan",
            KnownArch::Ssm => "ssm",
            KnownArch::Detr => "detr",
            KnownArch::Vit => "vit",
            KnownArch::Moe => "moe",
            KnownArch::Aet => "aet",
            KnownArch::Stst => "stst",
            KnownArch::Art => "art",
            KnownArch::Lora => "lora",
            KnownArch::Controlnet => "controlnet",
            KnownArch::Unclassified => "unclassified",
            KnownArch::Transformer => "transformer",
        }
    }

    fn from_token(s: &str) -> Option<Self> {
        Some(match s {
            "gru" => KnownArch::Gru,
            "rbm" => KnownArch::Rbm,
            "tae" => KnownArch::Tae,
            "vae" => KnownArch::Vae,
            "lstm" => KnownArch::Lstm,
            "resnet" => KnownArch::Resnet,
            "cnn" => KnownArch::Cnn,
            "rcnn" => KnownArch::Rcnn,
            "rnn" => KnownArch::Rnn,
            "brnn" => KnownArch::Brnn,
            "gan" => KnownArch::Gan,
            "ssm" => KnownArch::Ssm,
            "detr" => KnownArch::Detr,
            "vit" => KnownArch::Vit,
            "moe" => KnownArch::Moe,
            "aet" => KnownArch::Aet,
            "stst" => KnownArch::Stst,
            "art" => KnownArch::Art,
            "lora" => KnownArch::Lora,
            "controlnet" => KnownArch::Controlnet,
            "unclassified" => KnownArch::Unclassified,
            "transformer" => KnownArch::Transformer,
            _ => return None,
        })
    }
}

impl Arch {
    /// Parse an architecture token. Known taxonomy labels become
    /// [`Arch::Known`]; any other syntactically-valid token becomes
    /// [`Arch::Extension`] (superset behaviour).
    fn parse(s: &str) -> Result<Self, MirError> {
        let lower = s.to_ascii_lowercase();
        if lower.is_empty() || !is_valid_token(&lower) {
            return Err(MirError::InvalidArch(s.to_owned()));
        }
        Ok(match KnownArch::from_token(&lower) {
            Some(known) => Arch::Known(known),
            None => Arch::Extension(lower),
        })
    }

    /// Canonical lowercase token for this architecture.
    pub fn as_str(&self) -> &str {
        match self {
            Arch::Known(k) => k.as_str(),
            Arch::Extension(s) => s.as_str(),
        }
    }

    /// Whether this architecture is part of the standard Tiles taxonomy.
    pub fn is_known(&self) -> bool {
        matches!(self, Arch::Known(_))
    }
}

impl fmt::Display for Arch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A parsed, canonical MIR identifier.
///
/// Round-trip guarantee: for any valid `s`, `Mir::parse(s)?.to_string()` equals
/// the canonical form of `s` (lowercase, normalized series). Parsing the
/// canonical form again yields an identical `Mir` (idempotent).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Mir {
    /// Classification domain.
    pub domain: Domain,
    /// Architecture taxonomy label.
    pub arch: Arch,
    /// Free-form normalized variant (e.g. `clip-l`).
    pub variant: String,
    /// Normalized model series (e.g. `stable-diffusion-xl`).
    pub series: String,
}

impl Mir {
    /// Construct from already-parsed parts, validating + normalizing each field.
    pub fn new(
        domain: Domain,
        arch: Arch,
        variant: impl Into<String>,
        series: impl Into<String>,
    ) -> Result<Self, MirError> {
        let variant = normalize_token(&variant.into());
        if variant.is_empty() || !is_valid_token(&variant) {
            return Err(MirError::InvalidVariant(variant));
        }
        let series = normalize_series(&series.into());
        if series.is_empty() || !is_valid_token(&series) {
            return Err(MirError::InvalidSeries(series));
        }
        Ok(Self { domain, arch, variant, series })
    }

    /// Parse a `mir:domain.arch.variant:series` string into a canonical [`Mir`].
    ///
    /// Rejects: a missing/incorrect scheme, fewer than three dot-fields in the
    /// body, a missing series, an unknown domain, or any empty/invalid segment.
    /// Accepts unknown architecture labels as extensions.
    pub fn parse(s: &str) -> Result<Self, MirError> {
        let s = s.trim();
        // Scheme match is case-insensitive (MIR: / mir:); the body is normalized
        // to lowercase per-field below, so canonical emission is always lowercase.
        let scheme_len = MIR_SCHEME.len();
        if s.len() < scheme_len || !s[..scheme_len].eq_ignore_ascii_case(MIR_SCHEME) {
            return Err(MirError::MissingScheme(s.to_owned()));
        }
        let rest = &s[scheme_len..];

        // Split body (domain.arch.variant) from series at the FIRST remaining
        // colon. The series itself must not contain a colon.
        let (body, series_raw) = rest
            .split_once(':')
            .ok_or_else(|| MirError::Malformed(s.to_owned()))?;
        if series_raw.contains(':') {
            return Err(MirError::Malformed(s.to_owned()));
        }

        // Body must be exactly domain.arch.variant. The variant may itself
        // contain hyphens but not dots (dots are field separators), so splitn(3)
        // would wrongly fold a 4th dot-field into the variant — require exactly 3.
        let parts: Vec<&str> = body.split('.').collect();
        if parts.len() != 3 {
            return Err(MirError::Malformed(s.to_owned()));
        }
        let domain = Domain::parse(parts[0])?;
        let arch = Arch::parse(parts[1])?;

        let variant = normalize_token(parts[2]);
        if variant.is_empty() || !is_valid_token(&variant) {
            return Err(MirError::InvalidVariant(parts[2].to_owned()));
        }

        let series = normalize_series(series_raw);
        if series.is_empty() || !is_valid_token(&series) {
            return Err(MirError::InvalidSeries(series_raw.to_owned()));
        }

        Ok(Self { domain, arch, variant, series })
    }

    /// Canonical string form (`mir:domain.arch.variant:series`).
    pub fn to_canonical_string(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Mir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}.{}.{}:{}",
            MIR_SCHEME, self.domain, self.arch, self.variant, self.series
        )
    }
}

impl FromStr for Mir {
    type Err = MirError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Mir::parse(s)
    }
}

/// Whether a string is a `mir:` identifier (cheap, case-insensitive prefix
/// check; does not validate).
pub fn is_mir(s: &str) -> bool {
    let s = s.trim_start();
    s.len() >= MIR_SCHEME.len() && s[..MIR_SCHEME.len()].eq_ignore_ascii_case(MIR_SCHEME)
}

/// Normalize a single token: lowercase, collapse whitespace/underscores/slashes
/// to hyphens, strip leading/trailing hyphens. Used for variant + as the base of
/// series normalization.
fn normalize_token(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            'A'..='Z' => out.push(ch.to_ascii_lowercase()),
            'a'..='z' | '0'..='9' | '-' | '.' => out.push(ch),
            ' ' | '\t' | '_' | '/' => out.push('-'),
            _ => {} // drop anything else
        }
    }
    // collapse consecutive hyphens
    let mut collapsed = String::with_capacity(out.len());
    let mut prev_hyphen = false;
    for ch in out.chars() {
        if ch == '-' {
            if !prev_hyphen {
                collapsed.push(ch);
            }
            prev_hyphen = true;
        } else {
            collapsed.push(ch);
            prev_hyphen = false;
        }
    }
    collapsed.trim_matches('-').to_owned()
}

/// Apply MIR series-normalization rules:
///
/// - Lowercase, hyphen-only separators.
/// - Strip any library/org prefix (drop everything before the last `/`).
/// - Remove parameter-size tokens (e.g. `7b`, `0.6b`, `13B`).
/// - Remove non-breaking semantic version suffixes, keeping only the breaking
///   (major) version (e.g. `v1.2` -> `v1`).
/// - Strip known library-name suffixes (e.g. `-diffusers`).
///
/// Examples (from the Tiles spec):
/// - `tencent-hunyuan/hunyuandiT-v1.2-diffusers` -> `hunyuandit-v1`
/// - `black-forest-labs/FLUX.1-dev` -> `flux1dev`
///
/// # Spec ambiguity (documented)
///
/// The two Tiles examples differ in separator handling: hunyuan keeps the `-`
/// before its `v1` version (`hunyuandit-v1`); flux drops all separators
/// (`flux1dev`). The distinguishing feature is that flux contains a *dotted
/// bare-number* token (`FLUX.1`) that merges a digit into the name, whereas
/// hunyuan/SDXL use clean word/`vN` boundaries. We reconcile both with one
/// deterministic rule:
///
/// - If any surviving token is a **dotted bare-number** (e.g. `flux.1`):
///   flatten the whole series to a single lowercase alnum run. => `flux1dev`.
/// - Otherwise: keep `-` separators between tokens, collapse `vN.M` -> `vN`,
///   and drop any stray `.`. => `hunyuandit-v1`, `stable-diffusion-xl`,
///   `qwen3`, `llama-2`.
///
/// This is a superset choice; see `docs/interop/tiles-alignment.md`.
fn normalize_series(s: &str) -> String {
    // Strip org/library prefix: keep the segment after the last '/'.
    let base = s.rsplit('/').next().unwrap_or(s);

    // Lowercase + hyphenate separators, but keep '.' for now so we can reason
    // about version numbers before flattening.
    let lowered = normalize_token(base);

    // Tokenize on hyphens; drop noise tokens; collapse minor versions.
    let mut tokens: Vec<String> = Vec::new();
    let mut has_dotted_bare_number = false;
    for tok in lowered.split('-') {
        if tok.is_empty() {
            continue;
        }
        // Drop known library-name suffix tokens.
        if is_library_token(tok) {
            continue;
        }
        // Drop parameter-size tokens like 7b / 0.6b / 13b.
        if is_param_size_token(tok) {
            continue;
        }
        if is_dotted_bare_number(tok) {
            has_dotted_bare_number = true;
        }
        // Collapse non-breaking semver: v1.2.3 -> v1, 1.2 -> 1.
        tokens.push(strip_minor_version(tok));
    }

    if has_dotted_bare_number {
        // Flatten everything to a single alnum run (flux1dev).
        tokens
            .into_iter()
            .flat_map(|t| t.chars().collect::<Vec<_>>())
            .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
            .collect()
    } else {
        // Keep hyphen separators; drop any '.' left in tokens.
        let kept: Vec<String> = tokens
            .into_iter()
            .map(|t| t.chars().filter(|&c| c != '.').collect::<String>())
            .filter(|t| !t.is_empty())
            .collect();
        kept.join("-").trim_matches('-').to_owned()
    }
}

/// Whether a token mixes a *name* with a dotted number (e.g. `flux.1`), the
/// flux-style marker that triggers full series flattening. Excludes explicit
/// version tokens like `v1.2` (which are handled as versions, keeping hyphens).
fn is_dotted_bare_number(tok: &str) -> bool {
    if !tok.contains('.') {
        return false;
    }
    // Exclude version tokens: `v` + dotted digits (e.g. `v1.2`).
    if let Some(rest) = tok.strip_prefix('v') {
        if rest.split('.').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())) {
            return false;
        }
    }
    let has_alpha = tok.chars().any(|c| c.is_ascii_alphabetic());
    let tail = tok.rsplit('.').next().unwrap_or("");
    has_alpha && !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit())
}

/// Known library-name tokens that should be stripped from a series.
fn is_library_token(tok: &str) -> bool {
    matches!(
        tok,
        "diffusers" | "transformers" | "gguf" | "safetensors" | "onnx" | "ggml"
    )
}

/// Whether a token is a parameter-size marker (e.g. `7b`, `0.6b`, `13b`, `70m`).
fn is_param_size_token(tok: &str) -> bool {
    let suffix = tok.chars().last();
    if !matches!(suffix, Some('b') | Some('m') | Some('k')) {
        return false;
    }
    let num = &tok[..tok.len() - 1];
    !num.is_empty()
        && num.chars().all(|c| c.is_ascii_digit() || c == '.')
        && num.chars().any(|c| c.is_ascii_digit())
}

/// Collapse a non-breaking semantic version to its breaking (major) component:
/// `v1.2.3` -> `v1`, `1.2` -> `1`. Non-version tokens are returned unchanged.
fn strip_minor_version(tok: &str) -> String {
    let (prefix, ver) = if let Some(rest) = tok.strip_prefix('v') {
        ("v", rest)
    } else {
        ("", tok)
    };
    // Only treat as a version if it looks like a dotted number.
    if ver.contains('.') && ver.split('.').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit())) {
        let major = ver.split('.').next().unwrap_or(ver);
        return format!("{prefix}{major}");
    }
    tok.to_owned()
}

/// Validate a normalized token: lowercase alnum, hyphens, dots; non-empty; no
/// leading/trailing hyphen or dot; no consecutive dots.
fn is_valid_token(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.starts_with('-') || s.ends_with('-') || s.starts_with('.') || s.ends_with('.') {
        return false;
    }
    if s.contains("..") {
        return false;
    }
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
}

// ============================================================================
// Internal model-id  <->  MIR mapping
// ============================================================================
//
// Our internal model identity is `ModelRef { model, git_ref }`, where `model`
// is a registry name that is typically a HuggingFace-style id (`org/repo`,
// e.g. `Qwen/Qwen3-0.6B`) or a bare local name (e.g. `llama3`). The MIR
// `series` field is the normalized model series. The git_ref/branch is NOT
// represented in MIR (MIR classifies the model, not a specific revision).
//
// Mapping directions and lossiness:
//
//   internal-id  ->  MIR     (LOSSY): the domain defaults to `model` and the
//       architecture defaults to `unclassified` because neither can be derived
//       from a HF repo id alone. The variant defaults to `base`. Only `series`
//       is derived (via series normalization). Callers that know the arch/
//       domain/variant should supply them via [`Mir::new`] instead.
//
//   MIR  ->  internal-id     (LOSSY, partial inverse): we recover the registry
//       name candidate from the `series` (the org/library prefix and the exact
//       casing/param-size of the original HF id are NOT recoverable). Use this
//       to *match* against registered models rather than to reconstruct the
//       canonical HF id.
//
// Because of the lossy directions there is no exact bijection. The round-trip
// that IS defined and tested: `mir_from_series(mir.series) == mir.series` for
// any already-normalized series (series normalization is idempotent), so
// `internal -> MIR -> series` is stable once normalized.

/// Default variant used when mapping an internal id with no known sub-variant.
pub const DEFAULT_VARIANT: &str = "base";

/// Derive a MIR from an internal model name (HF repo id or local name).
///
/// LOSSY: defaults domain=`model`, arch=`unclassified`, variant=`base`; only
/// the series is derived from the name. Prefer [`Mir::new`] when arch/variant
/// are known. Returns an error only if the name yields an empty series.
pub fn mir_from_internal_name(name: &str) -> Result<Mir, MirError> {
    Mir::new(Domain::Model, Arch::Known(KnownArch::Unclassified), DEFAULT_VARIANT, name)
}

/// Derive a MIR from an internal name with a known architecture and variant.
///
/// Still lossy on domain (defaults to `model`); series is derived from `name`.
pub fn mir_from_internal_name_with(
    name: &str,
    arch: Arch,
    variant: &str,
) -> Result<Mir, MirError> {
    Mir::new(Domain::Model, arch, variant, name)
}

/// Recover the candidate internal series name from a MIR.
///
/// This is the normalized series only — it does NOT recover the org/library
/// prefix or original casing of a HF repo id. Use it to match registered model
/// names (after applying the same normalization to them).
pub fn internal_series_from_mir(mir: &Mir) -> String {
    mir.series.clone()
}

/// Normalize an internal model name to the MIR series form, for matching.
///
/// Two internal names map to the same model iff their normalized series match.
pub fn series_key_for_name(name: &str) -> String {
    normalize_series(name)
}

/// Resolve a MIR against a set of registered model names, returning the unique
/// matching name (the core of `ModelService::resolve_model_ref`, factored out
/// for testability without a live registry).
///
/// Matches a registered name iff its series-key equals the MIR's series. Returns
/// `Err` for no match or an ambiguous (>1) match.
pub fn resolve_mir_against_names<'a>(
    mir: &Mir,
    registered_names: impl IntoIterator<Item = &'a str>,
) -> Result<String, MirResolveError> {
    let target = internal_series_from_mir(mir);
    let mut matches: Vec<String> = registered_names
        .into_iter()
        .filter(|n| !n.is_empty())
        .filter(|n| series_key_for_name(n) == target)
        .map(ToOwned::to_owned)
        .collect();
    matches.sort();
    matches.dedup();
    match matches.len() {
        0 => Err(MirResolveError::NotFound(target)),
        1 => Ok(matches.remove(0)),
        _ => Err(MirResolveError::Ambiguous { series: target, candidates: matches }),
    }
}

/// Error from [`resolve_mir_against_names`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MirResolveError {
    /// No registered model has a matching series key.
    #[error("no registered model matches MIR series {0:?}")]
    NotFound(String),
    /// More than one registered model matches the series key.
    #[error("MIR series {series:?} is ambiguous; candidates: {candidates:?}")]
    Ambiguous {
        /// The normalized series that was searched for.
        series: String,
        /// The registered names that matched.
        candidates: Vec<String>,
    },
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_canonical_example() {
        let mir = Mir::parse("mir:model.transformer.clip-l:stable-diffusion-xl").unwrap();
        assert_eq!(mir.domain, Domain::Model);
        assert_eq!(mir.arch, Arch::Known(KnownArch::Transformer));
        assert_eq!(mir.variant, "clip-l");
        assert_eq!(mir.series, "stable-diffusion-xl");
    }

    #[test]
    fn display_roundtrip_is_canonical() {
        let canonical = "mir:model.transformer.clip-l:stable-diffusion-xl";
        let mir = Mir::parse(canonical).unwrap();
        assert_eq!(mir.to_string(), canonical);
        // idempotent re-parse
        assert_eq!(Mir::parse(&mir.to_string()).unwrap(), mir);
    }

    #[test]
    fn parse_normalizes_case() {
        let mir = Mir::parse("MIR:MODEL.TRANSFORMER.CLIP-L:Stable-Diffusion-XL").unwrap();
        assert_eq!(mir.to_string(), "mir:model.transformer.clip-l:stable-diffusion-xl");
    }

    #[test]
    fn known_arch_parses() {
        for label in [
            "gru", "rbm", "tae", "vae", "lstm", "resnet", "cnn", "rcnn", "rnn", "brnn", "gan",
            "ssm", "detr", "vit", "moe", "aet", "stst", "art", "lora", "controlnet",
            "unclassified", "transformer",
        ] {
            let s = format!("mir:model.{label}.base:demo");
            let mir = Mir::parse(&s).unwrap();
            assert!(mir.arch.is_known(), "{label} should be known");
            assert_eq!(mir.arch.as_str(), label);
        }
    }

    #[test]
    fn unknown_arch_is_extension_superset() {
        let mir = Mir::parse("mir:model.mamba2.base:falcon").unwrap();
        assert_eq!(mir.arch, Arch::Extension("mamba2".to_owned()));
        assert!(!mir.arch.is_known());
        // canonical emission stays stable
        assert_eq!(mir.to_string(), "mir:model.mamba2.base:falcon");
    }

    #[test]
    fn all_domains_parse() {
        for (label, dom) in [
            ("dev", Domain::Dev),
            ("model", Domain::Model),
            ("ops", Domain::Ops),
            ("info", Domain::Info),
            ("arch", Domain::Arch),
            ("architecture", Domain::Arch),
        ] {
            let mir = Mir::parse(&format!("mir:{label}.vae.base:demo")).unwrap();
            assert_eq!(mir.domain, dom);
        }
    }

    #[test]
    fn reject_missing_scheme() {
        assert!(matches!(
            Mir::parse("model.transformer.clip-l:sdxl"),
            Err(MirError::MissingScheme(_))
        ));
    }

    #[test]
    fn reject_missing_series() {
        assert!(matches!(
            Mir::parse("mir:model.transformer.clip-l"),
            Err(MirError::Malformed(_))
        ));
    }

    #[test]
    fn reject_missing_domain_or_arch() {
        // only two dot-fields -> malformed (no variant)
        assert!(matches!(
            Mir::parse("mir:model.transformer:sdxl"),
            Err(MirError::Malformed(_))
        ));
        // empty domain
        assert!(matches!(
            Mir::parse("mir:.transformer.clip:sdxl"),
            Err(MirError::InvalidDomain(_))
        ));
    }

    #[test]
    fn reject_bad_domain() {
        assert!(matches!(
            Mir::parse("mir:bogus.vae.base:demo"),
            Err(MirError::InvalidDomain(_))
        ));
    }

    #[test]
    fn reject_extra_colon_in_series() {
        assert!(matches!(
            Mir::parse("mir:model.vae.base:demo:extra"),
            Err(MirError::Malformed(_))
        ));
    }

    #[test]
    fn reject_four_dot_fields() {
        assert!(matches!(
            Mir::parse("mir:model.vae.base.extra:demo"),
            Err(MirError::Malformed(_))
        ));
    }

    #[test]
    fn series_normalization_hunyuan() {
        // tencent-hunyuan/hunyuandiT-v1.2-diffusers -> hunyuandit-v1
        assert_eq!(normalize_series("tencent-hunyuan/hunyuandiT-v1.2-diffusers"), "hunyuandit-v1");
    }

    #[test]
    fn series_normalization_flux() {
        // black-forest-labs/FLUX.1-dev -> flux1dev
        assert_eq!(normalize_series("black-forest-labs/FLUX.1-dev"), "flux1dev");
    }

    #[test]
    fn series_normalization_strips_param_size() {
        assert_eq!(normalize_series("Qwen/Qwen3-0.6B"), "qwen3");
        // clean word boundaries -> hyphens kept (param-size 7b stripped)
        assert_eq!(normalize_series("meta-llama/Llama-2-7b"), "llama-2");
    }

    #[test]
    fn new_validates_and_normalizes() {
        let mir = Mir::new(Domain::Model, Arch::Known(KnownArch::Vae), "Clip_L", "Stable Diffusion XL").unwrap();
        assert_eq!(mir.variant, "clip-l");
        assert_eq!(mir.series, "stable-diffusion-xl");
        assert_eq!(mir.to_string(), "mir:model.vae.clip-l:stable-diffusion-xl");
    }

    #[test]
    fn is_mir_detects_scheme() {
        assert!(is_mir("mir:model.vae.base:demo"));
        assert!(is_mir("  mir:model.vae.base:demo"));
        assert!(!is_mir("Qwen/Qwen3-0.6B"));
        assert!(!is_mir("llama3:main"));
    }

    #[test]
    fn internal_name_to_mir_lossy() {
        let mir = mir_from_internal_name("Qwen/Qwen3-0.6B").unwrap();
        assert_eq!(mir.domain, Domain::Model);
        assert_eq!(mir.arch, Arch::Known(KnownArch::Unclassified));
        assert_eq!(mir.variant, "base");
        assert_eq!(mir.series, "qwen3");
        assert_eq!(mir.to_string(), "mir:model.unclassified.base:qwen3");
    }

    #[test]
    fn internal_name_to_mir_with_arch() {
        let mir = mir_from_internal_name_with(
            "stabilityai/stable-diffusion-xl",
            Arch::Known(KnownArch::Transformer),
            "clip-l",
        )
        .unwrap();
        assert_eq!(mir.to_string(), "mir:model.transformer.clip-l:stable-diffusion-xl");
    }

    #[test]
    fn mir_to_internal_series_roundtrip() {
        // internal name -> MIR -> series, stable once normalized
        let name = "Qwen/Qwen3-0.6B";
        let mir = mir_from_internal_name(name).unwrap();
        let series = internal_series_from_mir(&mir);
        assert_eq!(series, "qwen3");
        // series_key matching: the normalized key of the original name matches
        assert_eq!(series_key_for_name(name), series);
        // re-deriving from the recovered series is idempotent
        let mir2 = mir_from_internal_name(&series).unwrap();
        assert_eq!(mir2.series, mir.series);
    }

    #[test]
    fn series_key_matches_across_aliases() {
        // org-prefixed and bare names normalize to the same key
        assert_eq!(series_key_for_name("Qwen/Qwen3-0.6B"), series_key_for_name("Qwen3-0.6B"));
    }

    #[test]
    fn registry_resolves_model_by_mir() {
        // A MIR string resolves to a registered model name by series match.
        let names = ["Qwen3-0.6B", "Llama-2-7b", "stable-diffusion-xl"];
        let mir = Mir::parse("mir:model.unclassified.base:qwen3").unwrap();
        let resolved = resolve_mir_against_names(&mir, names).unwrap();
        assert_eq!(resolved, "Qwen3-0.6B");
    }

    #[test]
    fn registry_resolve_mir_not_found() {
        let names = ["Qwen3-0.6B"];
        let mir = Mir::parse("mir:model.vae.base:mistral").unwrap();
        assert!(matches!(
            resolve_mir_against_names(&mir, names),
            Err(MirResolveError::NotFound(_))
        ));
    }

    #[test]
    fn registry_resolve_mir_ambiguous() {
        // Two registered names normalizing to the same series -> ambiguous.
        let names = ["Qwen/Qwen3-0.6B", "qwen3"];
        let mir = Mir::parse("mir:model.unclassified.base:qwen3").unwrap();
        assert!(matches!(
            resolve_mir_against_names(&mir, names),
            Err(MirResolveError::Ambiguous { .. })
        ));
    }
}
