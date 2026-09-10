//! Account-label grammar and never-reuse allocation policy (#1160 / A2).
//!
//! Account labels are permanent DNS names and are therefore treated as a
//! one-way door.  The policy is deliberately narrow: callers must provide a
//! canonical, lowercase ASCII LDH label.  Unicode input is rejected rather
//! than normalized; this makes NFKC, mixed-script, and confusable handling
//! fail closed without depending on a locale or a changing Unicode table.
//! Punycode A-labels (`xn--…`) are accepted as opaque ASCII labels, while the
//! corresponding Unicode spelling is never accepted at this boundary.
//!
//! [`AccountLabelRegistry`] reserves a label with an exclusive file create.
//! A reservation file is the durable tombstone: it is never removed, even if
//! a later mint fails or an operator retires the account.  A crash can leave an
//! empty reservation file, which is intentionally still reserved and therefore
//! cannot cause a later account to inherit the old label.

use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// The maximum DNS label length in octets (RFC 1035).
pub const MAX_ACCOUNT_LABEL_LEN: usize = 63;

/// Names owned by infrastructure or protocol endpoints rather than accounts.
/// Keep this list conservative: once a name is minted it cannot be reclaimed.
pub const RESERVED_ACCOUNT_LABELS: &[&str] = &[
    "admin",
    "administrator",
    "api",
    "atproto",
    "auth",
    "contact",
    "dashboard",
    "default",
    "discover",
    "dns",
    "ftp",
    "health",
    "help",
    "hostmaster",
    "info",
    "localhost",
    "mail",
    "ns1",
    "ns2",
    "postmaster",
    "root",
    "security",
    "support",
    "system",
    "webmaster",
    "www",
];

/// A validated, canonical account label.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AccountLabel(String);

impl AccountLabel {
    /// Parse a canonical account label.
    ///
    /// We reject non-ASCII rather than silently applying NFKC.  This also
    /// rejects mixed-script and confusable spellings at the only point where a
    /// permanent label can be introduced.  `xn--` A-labels are retained as
    /// opaque ASCII DNS labels; Unicode U-label input is not accepted.
    pub fn parse(input: &str) -> Result<Self> {
        anyhow::ensure!(!input.is_empty(), "account label is empty");
        anyhow::ensure!(
            input == input.trim(),
            "account label has surrounding whitespace"
        );
        anyhow::ensure!(
            input.len() <= MAX_ACCOUNT_LABEL_LEN,
            "account label exceeds 63 octets"
        );
        anyhow::ensure!(
            input.is_ascii(),
            "account label must be canonical ASCII; Unicode labels are rejected"
        );
        anyhow::ensure!(
            input
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'),
            "account label must be a lowercase LDH label"
        );
        anyhow::ensure!(
            !input.starts_with('-') && !input.ends_with('-'),
            "account label must not start or end with a hyphen"
        );
        anyhow::ensure!(
            !input.contains('.'),
            "account label must contain exactly one DNS label"
        );
        if input.starts_with("xn--") {
            anyhow::ensure!(input.len() > 4, "punycode account label has no payload");
        }
        anyhow::ensure!(
            !RESERVED_ACCOUNT_LABELS.contains(&input),
            "account label {input:?} is reserved"
        );
        Ok(Self(input.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AccountLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A label returned after an exclusive durable reservation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReservedAccountLabel {
    label: AccountLabel,
    path: PathBuf,
}

impl ReservedAccountLabel {
    pub fn label(&self) -> &AccountLabel {
        &self.label
    }

    /// The durable tombstone path.  This is useful for audit receipts and is
    /// never exposed as a mutable handle to the reservation file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Filesystem-backed, never-reuse account-label registry.
#[derive(Clone, Debug)]
pub struct AccountLabelRegistry {
    root: PathBuf,
}

impl AccountLabelRegistry {
    /// Open (and create) the directory that stores permanent reservations.
    pub fn new(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root)
            .with_context(|| format!("create account-label registry {}", root.display()))?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Reserve a label exactly once.  `create_new` is the atomic collision
    /// boundary; an existing or partially-written file is still a tombstone.
    pub fn reserve(&self, input: &str) -> Result<ReservedAccountLabel> {
        let label = AccountLabel::parse(input)?;
        let path = self.root.join(format!("{}.reserved", label.as_str()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("reserve account label {label}"))?;
        // The file is an audit record, not the source of exclusivity.  Write
        // and sync it before returning; if this fails, the empty file remains
        // a safe permanent tombstone.
        write_reservation_record(&mut file, &label)?;
        file.sync_all()
            .with_context(|| format!("sync account-label reservation {}", path.display()))?;
        sync_directory(&self.root)?;
        Ok(ReservedAccountLabel { label, path })
    }

    /// Return whether a label has ever been reserved.  Malformed or empty
    /// tombstones return `true` because reusing them would violate permanence.
    pub fn is_reserved(&self, input: &str) -> Result<bool> {
        let label = AccountLabel::parse(input)?;
        Ok(self.path_for(&label).is_file())
    }

    fn path_for(&self, label: &AccountLabel) -> PathBuf {
        self.root.join(format!("{}.reserved", label.as_str()))
    }
}

fn write_reservation_record(file: &mut File, label: &AccountLabel) -> Result<()> {
    writeln!(file, "version=1")?;
    writeln!(file, "label={}", label.as_str())?;
    writeln!(file, "state=reserved")?;
    Ok(())
}

fn sync_directory(path: &Path) -> Result<()> {
    // Directory fsync is available on the Unix deployment targets.  On
    // platforms where opening a directory is unsupported, the exclusive file
    // create is still the durable reservation primitive; return the error so
    // callers cannot mistake an unsynced allocation for a completed one.
    let directory = File::open(path)
        .with_context(|| format!("open account-label registry {} for sync", path.display()))?;
    directory
        .sync_all()
        .with_context(|| format!("sync account-label registry {}", path.display()))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn exhaustive_grammar_corpus_is_fail_closed() {
        let max_length = "a".repeat(MAX_ACCOUNT_LABEL_LEN);
        for valid in [
            "alice",
            "agent-7",
            "a1",
            "xn--bcher-kva",
            max_length.as_str(),
        ] {
            AccountLabel::parse(valid).expect(valid);
        }
        for invalid in [
            "",
            " Alice",
            "alice ",
            "Alice",
            "alice.example",
            "-alice",
            "alice-",
            "alice_1",
            "bücher",
            "аlice", // Cyrillic а, a confusable/mixed-script spelling.
            "xn--",
            "www",
            "admin",
        ] {
            assert!(
                AccountLabel::parse(invalid).is_err(),
                "accepted {invalid:?}"
            );
        }
    }

    #[test]
    fn reservation_is_atomic_and_never_reused() {
        let directory = tempfile::tempdir().expect("tempdir");
        let registry =
            AccountLabelRegistry::new(directory.path().join("labels")).expect("registry");
        let reservation = registry.reserve("alice").expect("first reservation");
        assert_eq!(reservation.label().as_str(), "alice");
        assert!(registry.is_reserved("alice").expect("reserved lookup"));
        assert!(
            registry.reserve("alice").is_err(),
            "a label must never be reused"
        );
        let bytes = fs::read(reservation.path()).expect("reservation record");
        let record = String::from_utf8(bytes).expect("utf8 record");
        assert!(record.contains("version=1"));
        assert!(record.contains("label=alice"));
        assert!(record.contains("state=reserved"));
    }

    #[test]
    fn an_empty_crash_tombstone_still_blocks_reuse() {
        let directory = tempfile::tempdir().expect("tempdir");
        let registry =
            AccountLabelRegistry::new(directory.path().join("labels")).expect("registry");
        let tombstone = registry.root().join("crashed.reserved");
        File::create(&tombstone).expect("crash tombstone");
        assert!(registry.is_reserved("crashed").expect("reserved lookup"));
        assert!(
            registry.reserve("crashed").is_err(),
            "crash tombstones are permanent"
        );
    }
}
