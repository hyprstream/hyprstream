//! The contamination firewalls: the frozen **vob-1.1** benchmark manifest as
//! a consumed artifact.
//!
//! Two firewalls (plan P0.6/P1.3):
//!
//! - **Family-level**: designated gate families are never synthesized into
//!   training data. Synthesis families are structurally disjoint
//!   (`family.rs`), and this module re-checks every generated item's family
//!   against the manifest's `gate` designations at run time.
//! - **Item-level**: no synthesized item may hash-collide with a frozen
//!   benchmark item (dedup and the firewall use the same canonical hash).
//!
//! The loader **verifies the pins** before any check: the manifest file's
//! blake3, the embedded release id, and the embedded DISCLOSURE digest must
//! match the constants below, or loading fails loudly.

use std::collections::HashSet;
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
    /// The item's canonical hash matches a frozen benchmark item.
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
            Self::GateFamily(family) => write!(
                f,
                "family {family} is a designated gate family — never synthesized"
            ),
            Self::ContaminatedItem { id, hash } => write!(
                f,
                "synthesized item {id} hash-collides with frozen benchmark item {hash}"
            ),
        }
    }
}

impl std::error::Error for FirewallViolation {}

/// The loaded, pin-verified firewall.
#[derive(Debug, Clone)]
pub struct Firewall {
    release: String,
    manifest_blake3: String,
    item_hashes: HashSet<String>,
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
        let found = blake3_hex(&disclosure_bytes);
        if found != EXPECTED_DISCLOSURE_BLAKE3 || found != manifest.disclosure.blake3 {
            return Err(FirewallError::DisclosureDigestMismatch { found });
        }
        Self::from_verified_bytes(&bytes)
    }

    /// Build from the manifest's canonical bytes. The digest is derived from
    /// the bytes themselves — never accepted as a separate caller-supplied
    /// argument, so a modified manifest (e.g. `items`/`families` cleared)
    /// cannot be paired with the frozen digest to disable the contamination
    /// checks. The pins are still verified: constructing a firewall that does
    /// not match the frozen vob-1.1 pins is an error, never a silent pass.
    pub fn from_verified_bytes(manifest_bytes: &[u8]) -> Result<Self, FirewallError> {
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
        if manifest.disclosure.blake3 != EXPECTED_DISCLOSURE_BLAKE3 {
            return Err(FirewallError::DisclosureDigestMismatch {
                found: manifest.disclosure.blake3,
            });
        }
        Ok(Self {
            release: manifest.release,
            manifest_blake3,
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
        if self.gate_families.contains(item.family.as_str()) {
            return Err(FirewallViolation::GateFamily(
                item.family.as_str().to_owned(),
            ));
        }
        let hash = item.hash();
        if self.item_hashes.contains(&hash) {
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
            Firewall::from_verified_bytes(b"tampered manifest bytes"),
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
            Firewall::from_verified_bytes(&tampered),
            Err(FirewallError::ManifestDigestMismatch { .. })
        ));
    }

    #[test]
    fn known_manifest_hash_is_flagged() {
        let bytes = std::fs::read(committed_manifest_path()).unwrap();
        let text = String::from_utf8_lossy(&bytes);
        let manifest = Manifest::from_json(&text).unwrap();
        let first = manifest.items[0].blake3.clone();
        let firewall = Firewall::from_verified_bytes(&bytes).unwrap();
        assert!(firewall.is_contaminated_hash(&first));
        assert!(!firewall.is_contaminated_hash(&"e".repeat(64)));
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
