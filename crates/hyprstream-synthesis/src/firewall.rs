//! The contamination firewalls: the frozen **vob-1.1** benchmark manifest as
//! a consumed artifact.
//!
//! Two firewalls (plan P0.6/P1.3):
//!
//! - **Family-level**: designated gate families are never synthesized into
//!   training data. Synthesis families are structurally disjoint
//!   (`family.rs`), and this module re-checks every generated item's family
//!   against the manifest's `gate` designations at run time.
//! - **Item-level**: synthesized state/question content must not match the
//!   shared typed fingerprint of any completely attested frozen benchmark
//!   item. Source-specific artifact hashes remain distinct and unchanged.
//!
//! The loader **verifies the pins** before any check: the manifest file's
//! blake3, the embedded release id, and the embedded DISCLOSURE digest must
//! match the constants below, or loading fails loudly.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use hyprstream_bench::manifest::{blake3_hex, Manifest};

use crate::item::SynthItem;

/// Pinned frozen release id consumed by this pipeline.
pub const EXPECTED_RELEASE: &str = "vob-1.1";

/// Pinned blake3 of the frozen manifest file (`vob-1.1.manifest.json`).
pub const EXPECTED_MANIFEST_BLAKE3: &str =
    "a19596ff13fbf7c09c259d2ed2ec5c8aafdca49a23616099e5b17f84a0a5be8f";

/// Pinned blake3 of the benchmark's frozen DISCLOSURE.md (also embedded in
/// the manifest).
pub const EXPECTED_DISCLOSURE_BLAKE3: &str =
    "9bf26d86de584f07b1e4138dfcf197ff00c9563089365e861ffbabe084cb533d";

/// The default repo-relative path of the frozen manifest.
pub const DEFAULT_MANIFEST_PATH: &str = "crates/hyprstream-bench/manifest/vob-1.1.manifest.json";

/// Errors from loading/verifying the consumed manifest.
#[derive(Debug)]
pub enum FirewallError {
    /// Manifest file unreadable.
    Io(std::io::Error),
    /// Missing, incomplete or corrupt caller-supplied benchmark content index.
    ContentIndex(String),
    /// The pinned DISCLOSURE file unreadable (missing = drift).
    DisclosureIo(std::io::Error),
    /// Manifest JSON unparseable.
    Parse(serde_json::Error),
    /// The file's blake3 does not match [`EXPECTED_MANIFEST_BLAKE3`].
    ManifestDigestMismatch { found: String },
    /// The embedded release id does not match [`EXPECTED_RELEASE`].
    ReleaseMismatch { found: String },
    /// The embedded DISCLOSURE digest does not match
    /// [`EXPECTED_DISCLOSURE_BLAKE3`].
    DisclosureDigestMismatch { found: String },
}

impl std::fmt::Display for FirewallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContentIndex(err) => write!(f, "benchmark content index rejected: {err}"),
            Self::Io(err) => write!(f, "manifest unreadable: {err}"),
            Self::DisclosureIo(err) => {
                write!(f, "pinned DISCLOSURE file unreadable: {err}")
            }
            Self::Parse(err) => write!(f, "manifest unparseable: {err}"),
            Self::ManifestDigestMismatch { found } => write!(
                f,
                "manifest digest mismatch: expected {EXPECTED_MANIFEST_BLAKE3}, found {found} — the consumed artifact is not the frozen vob-1.1 manifest"
            ),
            Self::ReleaseMismatch { found } => {
                write!(f, "release mismatch: expected {EXPECTED_RELEASE}, found {found}")
            }
            Self::DisclosureDigestMismatch { found } => write!(
                f,
                "disclosure digest mismatch: expected {EXPECTED_DISCLOSURE_BLAKE3}, found {found}"
            ),
        }
    }
}

impl std::error::Error for FirewallError {}

/// Resolve the pinned DISCLOSURE file from the manifest's embedded
/// repo-relative path. Prefers the repo root four levels above the manifest
/// (`<root>/crates/hyprstream-bench/manifest/<file>`); falls back to the
/// manifest's parent's parent so a copied layout still resolves — and still
/// fails closed (`DisclosureIo`) if the file is absent.
fn disclosure_path_for(manifest_path: &Path, manifest: &Manifest) -> Option<std::path::PathBuf> {
    let embedded = Path::new(&manifest.disclosure.path);
    if let Some(root) = manifest_path.ancestors().nth(4) {
        let candidate = root.join(embedded);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    let file_name = embedded.file_name()?;
    manifest_path
        .parent()
        .and_then(Path::parent)
        .map(|dir| dir.join(file_name))
}

/// A synthesized item that hit a firewall.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FirewallViolation {
    /// The item's family is a designated gate family.
    GateFamily(String),
    /// Artifact hashes alone cannot authorize synthesis.
    MissingContentIndex,
    /// Invalid typed content cannot pass a fingerprint gate.
    InvalidContent(String),
    /// The shared typed content fingerprint matches a frozen benchmark item.
    ContaminatedItem {
        /// Synthesized item id.
        id: String,
        /// Colliding hash.
        hash: String,
    },
}

impl std::fmt::Display for FirewallViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingContentIndex => f.write_str(
                "attested complete benchmark content index is required before synthesis",
            ),
            Self::InvalidContent(error) => write!(f, "invalid typed content: {error}"),
            Self::GateFamily(family) => write!(
                f,
                "family {family} is a designated gate family — never synthesized"
            ),
            Self::ContaminatedItem { id, hash } => write!(
                f,
                "synthesized item {id} matches frozen content fingerprint {hash}"
            ),
        }
    }
}

impl std::error::Error for FirewallViolation {}

/// The source-qualified typed profile of the frozen vob1.1 generators.
/// Frozen artifact bytes render Entry values and omit some rubric fields;
/// admitting other types would not bind a typed fingerprint to those hashes.
pub const FROZEN_CONTENT_PROFILE: &str = "vob-1.1-text/1";

fn validate_frozen_profile(item: &hyprstream_bench::Item) -> Result<(), FirewallError> {
    use hyprstream_decision::{Entry, QuestionBody, QuestionKind};
    let typed = matches!(&item.state, Entry::Str(_))
        && matches!(&item.question.instructions, Some(Entry::Str(_)));
    let body = match (item.question.kind, &item.question.body) {
        (QuestionKind::Noul, QuestionBody::Noul { criteria: None }) => true,
        (QuestionKind::Choice, QuestionBody::Choice { options }) => {
            options.len() >= 2 && options.iter().all(|o| o.rubric.is_none())
        }
        (QuestionKind::Score, QuestionBody::Score { levels }) => {
            levels.len() >= 2 && levels.iter().all(Option::is_none)
        }
        _ => false,
    };
    if !typed || !body || item.truth >= item.question.cardinality() {
        return Err(FirewallError::ContentIndex(format!(
            "item {} violates source-qualified {FROZEN_CONTENT_PROFILE}",
            item.id
        )));
    }
    Ok(())
}

/// The loaded, pin-verified firewall.
#[derive(Debug, Clone)]
pub struct Firewall {
    release: String,
    manifest_blake3: String,
    item_hashes: HashSet<String>,
    expected_items: Vec<(String, String)>,
    content_hashes: Option<HashSet<String>>,
    gate_families: HashSet<String>,
}

impl Firewall {
    /// Load and pin-verify the frozen manifest from disk — and the frozen
    /// DISCLOSURE text it points at. Checking only the manifest's embedded
    /// disclosure digest would let a drifted or deleted DISCLOSURE.md pass
    /// unnoticed; the file itself is read, hashed, and compared against both
    /// the embedded pin and [`EXPECTED_DISCLOSURE_BLAKE3`].
    pub fn load(path: &Path) -> Result<Self, FirewallError> {
        let bytes = std::fs::read(path).map_err(FirewallError::Io)?;
        let text = String::from_utf8_lossy(&bytes);
        let manifest = Manifest::from_json(&text).map_err(FirewallError::Parse)?;
        // An unresolvable disclosure path (e.g. the manifest supplied as a
        // single-component relative path) must fail closed, not silently
        // skip the disclosure verification.
        let disclosure_path = disclosure_path_for(path, &manifest).ok_or_else(|| {
            FirewallError::DisclosureIo(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "cannot resolve pinned DISCLOSURE path {} relative to {}",
                    manifest.disclosure.path,
                    path.display()
                ),
            ))
        })?;
        let disclosure_bytes =
            std::fs::read(&disclosure_path).map_err(FirewallError::DisclosureIo)?;
        Self::from_verified_bytes(&bytes, &disclosure_bytes)
    }

    /// Build from the manifest's canonical bytes AND the actual DISCLOSURE
    /// text. Both digests are derived from the bytes themselves — never
    /// accepted as separate caller-supplied arguments — so a modified
    /// manifest (e.g. `items`/`families` cleared) cannot be paired with the
    /// frozen digest to disable the contamination checks, and a drifted or
    /// deleted disclosure cannot be papered over by the manifest's embedded
    /// pin. The pins are still verified: constructing a firewall that does
    /// not match the frozen vob-1.1 pins is an error, never a silent pass.
    pub fn from_verified_bytes(
        manifest_bytes: &[u8],
        disclosure_bytes: &[u8],
    ) -> Result<Self, FirewallError> {
        let manifest_blake3 = blake3_hex(manifest_bytes);
        if manifest_blake3 != EXPECTED_MANIFEST_BLAKE3 {
            return Err(FirewallError::ManifestDigestMismatch {
                found: manifest_blake3,
            });
        }
        let text = String::from_utf8_lossy(manifest_bytes);
        let manifest = Manifest::from_json(&text).map_err(FirewallError::Parse)?;
        if manifest.release != EXPECTED_RELEASE {
            return Err(FirewallError::ReleaseMismatch {
                found: manifest.release,
            });
        }
        let found_disclosure = blake3_hex(disclosure_bytes);
        if found_disclosure != EXPECTED_DISCLOSURE_BLAKE3
            || found_disclosure != manifest.disclosure.blake3
        {
            return Err(FirewallError::DisclosureDigestMismatch {
                found: found_disclosure,
            });
        }
        Ok(Self {
            release: manifest.release,
            manifest_blake3,
            expected_items: manifest
                .items
                .iter()
                .map(|i| (i.id.clone(), i.blake3.clone()))
                .collect(),
            content_hashes: None,
            item_hashes: manifest
                .items
                .iter()
                .map(|item| item.blake3.clone())
                .collect(),
            gate_families: manifest
                .families
                .iter()
                .filter(|row| row.designation == "gate")
                .map(|row| row.id.clone())
                .collect(),
        })
    }

    /// Attach COMPLETE typed benchmark content after verifying every artifact
    /// against the already pin-verified manifest. No generation or pin override.
    /// This consumes self so a failed replacement cannot be silently used.
    pub fn with_attested_items(
        mut self,
        items: &[hyprstream_bench::Item],
    ) -> Result<Self, FirewallError> {
        if items.len() != self.expected_items.len() {
            return Err(FirewallError::ContentIndex("incomplete item count".into()));
        }
        let expected: HashMap<_, _> = self
            .expected_items
            .iter()
            .map(|(id, hash)| (id.as_str(), hash.as_str()))
            .collect();
        if expected.len() != self.expected_items.len() {
            return Err(FirewallError::ContentIndex("duplicate manifest IDs".into()));
        }
        let mut seen = HashSet::new();
        let mut content = HashSet::new();
        for item in items {
            validate_frozen_profile(item)?;
            if !seen.insert(item.id.as_str())
                || item.question.id != item.id
                || expected.get(item.id.as_str()).copied() != Some(item.hash().as_str())
            {
                return Err(FirewallError::ContentIndex(format!(
                    "unknown, duplicate or corrupt item {}",
                    item.id
                )));
            }
            let fingerprint = item
                .content_fingerprint()
                .map_err(|e| FirewallError::ContentIndex(e.to_string()))?;
            content.insert(fingerprint);
        }
        self.content_hashes = Some(content);
        Ok(self)
    }

    /// Decode caller-supplied content; all trust still comes from exact
    /// frozen artifact verification in with_attested_items.
    pub fn with_content_index_json(self, text: &str) -> Result<Self, FirewallError> {
        let items = crate::content_index::decode(text).map_err(FirewallError::ContentIndex)?;
        self.with_attested_items(&items)
    }

    /// Pipeline preflight: even an empty manifest requires an explicitly
    /// attested empty index; absence is never an allow state.
    pub fn ensure_content_index(&self) -> Result<(), FirewallViolation> {
        self.content_hashes
            .as_ref()
            .map(|_| ())
            .ok_or(FirewallViolation::MissingContentIndex)
    }

    /// The consumed release id.
    pub fn release(&self) -> &str {
        &self.release
    }

    /// The consumed manifest's blake3 (recorded in row provenance).
    pub fn manifest_blake3(&self) -> &str {
        &self.manifest_blake3
    }

    /// The gate family ids.
    pub fn gate_families(&self) -> &HashSet<String> {
        &self.gate_families
    }

    /// Check a synthesized item against both firewalls.
    pub fn check_item(&self, item: &SynthItem) -> Result<(), FirewallViolation> {
        self.ensure_content_index()?;
        if self.gate_families.contains(item.family.as_str()) {
            return Err(FirewallViolation::GateFamily(
                item.family.as_str().to_owned(),
            ));
        }
        let hash = item
            .content_fingerprint()
            .map_err(|e| FirewallViolation::InvalidContent(e.to_string()))?;
        if self
            .content_hashes
            .as_ref()
            .is_some_and(|hashes| hashes.contains(&hash))
        {
            return Err(FirewallViolation::ContaminatedItem {
                id: item.id.clone(),
                hash,
            });
        }
        Ok(())
    }

    /// Whether a raw canonical hash collides with a frozen item.
    pub fn is_contaminated_hash(&self, hash: &str) -> bool {
        self.item_hashes.contains(hash)
    }
    #[cfg(test)]
    pub(crate) fn tiny_unattested(items: &[hyprstream_bench::Item]) -> Self {
        Self {
            release: "test-only".into(),
            manifest_blake3: "private-tiny-fixture".into(),
            item_hashes: items.iter().map(hyprstream_bench::Item::hash).collect(),
            expected_items: items.iter().map(|i| (i.id.clone(), i.hash())).collect(),
            content_hashes: None,
            gate_families: HashSet::from(["temporal".into(), "syllogism".into()]),
        }
    }
    #[cfg(test)]
    pub(crate) fn tiny_item() -> hyprstream_bench::Item {
        hyprstream_bench::Item::noul(
            hyprstream_bench::Family::Arith,
            hyprstream_bench::Stratum::Clean,
            7,
            hyprstream_decision::Entry::Str("private tiny benchmark".into()),
            "fixture predicate",
            true,
        )
    }
    #[cfg(test)]
    #[allow(clippy::expect_used)]
    pub(crate) fn tiny_fixture() -> Self {
        let item = Self::tiny_item();
        Self::tiny_unattested(std::slice::from_ref(&item))
            .with_attested_items(&[item])
            .expect("private tiny attestation")
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn committed_manifest_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../hyprstream-bench/manifest/vob-1.1.manifest.json")
    }

    #[test]
    fn committed_manifest_loads_and_matches_all_pins() {
        let firewall = Firewall::load(&committed_manifest_path()).unwrap();
        assert_eq!(firewall.release(), EXPECTED_RELEASE);
        assert_eq!(firewall.manifest_blake3(), EXPECTED_MANIFEST_BLAKE3);
        assert_eq!(
            firewall.gate_families(),
            &HashSet::from(["temporal".to_owned(), "syllogism".to_owned()])
        );
    }

    #[test]
    fn digest_mismatch_fails_closed() {
        assert!(matches!(
            Firewall::from_verified_bytes(b"tampered manifest bytes", b"disclosure"),
            Err(FirewallError::ManifestDigestMismatch { .. })
        ));
    }

    #[test]
    fn digest_is_derived_from_the_supplied_bytes() {
        // Regression (review): the constructor used to take the manifest and
        // its digest as separate arguments, so a caller could pair a cleared
        // manifest (no items, no gate families — both contamination checks
        // disabled) with the frozen digest. The digest is now derived from
        // the canonical bytes, so that pairing is unrepresentable.
        let text = std::fs::read_to_string(committed_manifest_path()).unwrap();
        let mut manifest = Manifest::from_json(&text).unwrap();
        manifest.items.clear();
        manifest.families.clear();
        let tampered = serde_json::to_vec(&manifest).unwrap();
        assert!(matches!(
            Firewall::from_verified_bytes(&tampered, b"disclosure"),
            Err(FirewallError::ManifestDigestMismatch { .. })
        ));
    }

    fn committed_disclosure_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../hyprstream-bench/DISCLOSURE.md")
    }

    #[test]
    fn known_manifest_hash_is_flagged() {
        let bytes = std::fs::read(committed_manifest_path()).unwrap();
        let disclosure = std::fs::read(committed_disclosure_path()).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        let manifest = Manifest::from_json(&text).unwrap();
        let first = manifest.items[0].blake3.clone();
        let firewall = Firewall::from_verified_bytes(&bytes, &disclosure).unwrap();
        assert!(firewall.is_contaminated_hash(&first));
        assert!(!firewall.is_contaminated_hash(&"e".repeat(64)));
    }

    #[test]
    fn constructor_requires_the_real_disclosure_bytes() {
        // Regression (review thread 2026-09-22): the public in-memory
        // constructor used to trust the manifest's embedded disclosure
        // digest without ever seeing the DISCLOSURE text, so a deleted or
        // drifted DISCLOSURE.md still produced a firewall that
        // `pipeline::run` accepted. The disclosure bytes are now an
        // argument and are hashed against both pins.
        let bytes = std::fs::read(committed_manifest_path()).unwrap();
        let error = match Firewall::from_verified_bytes(&bytes, b"drifted disclosure") {
            Err(error) => error,
            Ok(_) => panic!("drifted disclosure accepted"),
        };
        assert!(matches!(
            error,
            FirewallError::DisclosureDigestMismatch { .. }
        ));
    }

    /// Copy the frozen artifacts into a `crates/hyprstream-bench` layout
    /// under a temp root, optionally substituting the disclosure text.
    fn staged_layout(disclosure: Option<&[u8]>) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!("p13-fw-test-{}", std::process::id()));
        let bench = root.join("crates/hyprstream-bench");
        let manifest_dir = bench.join("manifest");
        std::fs::create_dir_all(&manifest_dir).unwrap();
        std::fs::copy(
            committed_manifest_path(),
            manifest_dir.join("vob-1.1.manifest.json"),
        )
        .unwrap();
        let _ = std::fs::remove_file(bench.join("DISCLOSURE.md"));
        if let Some(bytes) = disclosure {
            std::fs::write(bench.join("DISCLOSURE.md"), bytes).unwrap();
        }
        root
    }

    #[test]
    fn disclosure_file_is_read_and_verified_at_load() {
        // Regression (review): checking only the manifest's embedded
        // disclosure digest let a drifted or deleted DISCLOSURE.md pass.
        let committed = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../hyprstream-bench/DISCLOSURE.md"),
        )
        .unwrap();

        let good = staged_layout(Some(&committed));
        Firewall::load(&good.join("crates/hyprstream-bench/manifest/vob-1.1.manifest.json"))
            .unwrap();

        let drifted = staged_layout(Some(b"tampered disclosure text"));
        assert!(matches!(
            Firewall::load(&drifted.join("crates/hyprstream-bench/manifest/vob-1.1.manifest.json")),
            Err(FirewallError::DisclosureDigestMismatch { .. })
        ));

        let missing = staged_layout(None);
        assert!(matches!(
            Firewall::load(&missing.join("crates/hyprstream-bench/manifest/vob-1.1.manifest.json")),
            Err(FirewallError::DisclosureIo(_))
        ));
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod content_admission_tests {
    use super::*;
    #[test]
    fn only_complete_verified_content_can_enable_checking() {
        let item = Firewall::tiny_item();
        let missing = Firewall::tiny_unattested(std::slice::from_ref(&item));
        assert_eq!(
            missing.ensure_content_index(),
            Err(FirewallViolation::MissingContentIndex)
        );
        assert!(missing.clone().with_attested_items(&[]).is_err());
        let mut corrupt = item.clone();
        corrupt.state = hyprstream_decision::Entry::Str("changed".into());
        assert!(missing.clone().with_attested_items(&[corrupt]).is_err());
        let mut corrupt = item.clone();
        corrupt.id = "other".into();
        assert!(missing.clone().with_attested_items(&[corrupt]).is_err());
        let mut unbound = item.clone();
        unbound.question.id = "unbound".into();
        assert!(missing.clone().with_attested_items(&[unbound]).is_err());
        assert!(missing
            .with_attested_items(std::slice::from_ref(&item))
            .unwrap()
            .ensure_content_index()
            .is_ok());
        let empty = Firewall::tiny_unattested(&[]);
        assert!(empty.ensure_content_index().is_err());
        assert!(empty
            .with_attested_items(&[])
            .unwrap()
            .ensure_content_index()
            .is_ok());
        let mut second = item.clone();
        second.id = "second".into();
        second.question.id = "second".into();
        let expected = Firewall::tiny_unattested(&[item.clone(), second]);
        assert!(expected.with_attested_items(&[item.clone(), item]).is_err());
    }
    #[test]
    fn copied_content_is_blocked_even_when_every_source_identity_differs() {
        let benchmark = Firewall::tiny_item();
        let firewall = Firewall::tiny_unattested(std::slice::from_ref(&benchmark))
            .with_attested_items(std::slice::from_ref(&benchmark))
            .unwrap();
        let copied = crate::item::SynthItem::new(
            crate::family::SynthFamily::Triage,
            999,
            0,
            1,
            benchmark.state.clone(),
            benchmark
                .question
                .instructions
                .as_ref()
                .map(hyprstream_decision::Entry::canonical_text),
            benchmark.question.body.clone(),
        );
        assert_ne!(copied.hash(), benchmark.hash());
        assert_eq!(
            copied.content_fingerprint().unwrap(),
            benchmark.content_fingerprint().unwrap()
        );
        assert!(matches!(
            firewall.check_item(&copied),
            Err(FirewallViolation::ContaminatedItem { .. })
        ));
        let mut safe = copied.clone();
        safe.state = hyprstream_decision::Entry::Str("different meaningful content".into());
        assert!(firewall.check_item(&safe).is_ok());
    }
    #[test]
    fn actual_json_index_is_attested_not_trusted() {
        let item = Firewall::tiny_item();
        let json=serde_json::json!({"version":hyprstream_bench::content::CONTENT_FINGERPRINT_VERSION,"items":[{"id":item.id,"family":"arith","stratum":"clean","group":item.group,"seed":7,"truth":1,"state":"private tiny benchmark","question":{"instructions":"fixture predicate","body":{"type":"noul","criteria":null}}}]}).to_string();
        assert!(Firewall::tiny_unattested(std::slice::from_ref(&item))
            .with_content_index_json(&json)
            .unwrap()
            .ensure_content_index()
            .is_ok());
        for bad in [
            json.replace("private tiny benchmark", "corrupt content"),
            json.replace("decision-content/1", "unknown"),
        ] {
            assert!(Firewall::tiny_unattested(std::slice::from_ref(&item))
                .with_content_index_json(&bad)
                .is_err());
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod frozen_profile_tests {
    use super::*;
    use hyprstream_decision::{Entry, NoulCriteria, QuestionBody, QuestionKind};
    fn rejects_against(original: &hyprstream_bench::Item, substitution: hyprstream_bench::Item) {
        // Causal witness: the old artifact hash alone admits this substitution.
        assert_eq!(original.hash(), substitution.hash());
        assert!(Firewall::tiny_unattested(std::slice::from_ref(original))
            .with_attested_items(&[substitution])
            .is_err());
    }
    #[test]
    fn old_artifact_hash_aliases_cannot_change_typed_index_content() {
        let mut original = Firewall::tiny_item();
        original.state = Entry::Str("true".into());
        let mut alternative = original.clone();
        alternative.state = Entry::Bool(true);
        rejects_against(&original, alternative);
        let mut original = Firewall::tiny_item();
        original.question.instructions = Some(Entry::Str(String::new()));
        let mut alternative = original.clone();
        alternative.question.instructions = None;
        rejects_against(&original, alternative);
        let original = Firewall::tiny_item();
        let mut alternative = original.clone();
        alternative.question.body = QuestionBody::Noul {
            criteria: Some(NoulCriteria {
                on_true: None,
                on_false: None,
            }),
        };
        rejects_against(&original, alternative);
        let mut original = Firewall::tiny_item();
        original.question.kind = QuestionKind::Score;
        original.question.body = QuestionBody::Score {
            levels: vec![None, None],
        };
        let mut alternative = original.clone();
        if let QuestionBody::Score { levels } = &mut alternative.question.body {
            levels[0] = Some(Entry::Str("unhashed rubric".into()));
        }
        rejects_against(&original, alternative);
        let original = hyprstream_bench::Item::choice(
            hyprstream_bench::Family::Arith,
            hyprstream_bench::Stratum::Clean,
            9,
            Entry::Str("s".into()),
            "q",
            vec!["a".into(), "b".into()],
            0,
        );
        let mut alternative = original.clone();
        if let QuestionBody::Choice { options } = &mut alternative.question.body {
            options[0].rubric = Some(Entry::Str(String::new()));
        }
        rejects_against(&original, alternative);
        let mut alternative = original.clone();
        alternative.question.kind = QuestionKind::Noul;
        assert!(Firewall::tiny_unattested(std::slice::from_ref(&original))
            .with_attested_items(&[alternative])
            .is_err());
    }
}
