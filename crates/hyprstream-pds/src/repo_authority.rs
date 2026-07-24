//! Which DID methods *our* PDS accepts as a **repo authority** — the account DID
//! that owns a repository and signs its commits (#908, design #905 §6).
//!
//! Public atproto infrastructure allowlists only `did:plc` and `did:web` as repo
//! authorities. hyprstream additionally accepts **`did:at9p`** (the
//! self-certifying hybrid-PQC capsule identity, #879/#880) on *our own* side: a
//! did:at9p account can own and sign a repo hosted by our PDS. Public publication
//! of such a repo goes through the #896 `alsoKnownAs` bridge, which republishes
//! under a classical `did:web`/`did:plc` authority that public infra will accept —
//! so accepting did:at9p here never implies public infra must.
//!
//! This is a pure classification of the DID *method*; it is not a signature check
//! (that is [`crate::commit::Commit::verify`]) and not a full DID-grammar
//! validation (the resolver owns that). It answers exactly one question: *is this
//! DID method one our PDS will host a repo for?*

use anyhow::{bail, Result};

use hyprstream_rpc::identity::Did;

/// Whether `did` is a path-form `did:web` identifier.
///
/// The atproto DID profile permits only host-form `did:web`; any additional
/// colon in the method-specific identifier introduces a path segment. Encoded
/// host ports (`%3A`) do not contain a literal colon and are deliberately not
/// classified here — this helper answers only the path-form question.
pub fn is_path_form_did_web(did: &str) -> bool {
    let Some(method_id) = did.strip_prefix("did:web:") else {
        return false;
    };
    let method_id = method_id.split(['?', '#']).next().unwrap_or(method_id);
    method_id.contains(':')
}

/// A DID method our PDS accepts as a repo authority.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepoAuthority {
    /// `did:web` — an operated node whose keys live as DID-document verification
    /// methods. Accepted by public atproto infra.
    Web,
    /// `did:plc` — the atproto placeholder DID method. Accepted by public atproto
    /// infra.
    Plc,
    /// `did:at9p` — the self-certifying hybrid-PQC capsule identity (#879/#880).
    /// Accepted by our PDS; **not** by public atproto infra (bridged via #896).
    At9p,
}

impl RepoAuthority {
    /// Whether public atproto infrastructure will itself accept this authority as
    /// a repo owner.
    ///
    /// `true` for `did:web`/`did:plc`; `false` for `did:at9p`. A did:at9p repo is
    /// published to public infra only via the #896 `alsoKnownAs` bridge under a
    /// classical DID.
    pub fn is_publicly_publishable(self) -> bool {
        matches!(self, RepoAuthority::Web | RepoAuthority::Plc)
    }
}

/// Classify `did` as an accepted repo authority for our PDS, or reject it.
///
/// Accepts `did:web`, `did:plc`, and `did:at9p`; every other DID method is
/// rejected. The `did:at9p` arm is what this issue (#908) adds — the classical
/// methods were already implicitly accepted by the commit layer, which never
/// method-checked its `did`.
pub fn accept_repo_authority(did: &str) -> Result<RepoAuthority> {
    let d = Did::new(did.to_owned());
    if d.is_did_web() {
        if is_path_form_did_web(did) {
            bail!(
                "path-form did:web {did:?} is not an accepted repo authority; the atproto profile requires host-form did:web"
            );
        }
        Ok(RepoAuthority::Web)
    } else if did.starts_with("did:plc:") {
        Ok(RepoAuthority::Plc)
    } else if d.is_did_at9p() {
        Ok(RepoAuthority::At9p)
    } else {
        bail!("did {did:?} is not an accepted repo authority (want did:web, did:plc, or did:at9p)")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn accepts_did_at9p() {
        let a = accept_repo_authority("did:at9p:bafkrei1234567890abcdefghijklmnop").unwrap();
        assert_eq!(a, RepoAuthority::At9p);
        assert!(!a.is_publicly_publishable(), "did:at9p is bridged, not public");
    }

    #[test]
    fn accepts_classical_methods() {
        let web = accept_repo_authority("did:web:alice.example.com").unwrap();
        assert_eq!(web, RepoAuthority::Web);
        assert!(web.is_publicly_publishable());

        let plc = accept_repo_authority("did:plc:abc123").unwrap();
        assert_eq!(plc, RepoAuthority::Plc);
        assert!(plc.is_publicly_publishable());
    }

    #[test]
    fn rejects_path_form_did_web_repo_authority() {
        assert!(is_path_form_did_web("did:web:accounts.example:users:alice"));
        assert!(!is_path_form_did_web("did:web:alice.accounts.example"));
        assert!(!is_path_form_did_web("did:web:localhost%3A6791"));

        let err = accept_repo_authority("did:web:accounts.example:users:alice").unwrap_err();
        assert!(err.to_string().contains("path-form did:web"));
    }

    #[test]
    fn rejects_unknown_methods() {
        assert!(accept_repo_authority("did:key:z6Mkxyz").is_err());
        assert!(accept_repo_authority("did:example:123").is_err());
        assert!(accept_repo_authority("not-a-did").is_err());
        assert!(accept_repo_authority("").is_err());
    }
}
