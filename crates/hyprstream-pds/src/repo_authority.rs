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
//! Public authorities are validated as complete repository identifiers, not
//! DID URLs. Host-form web authorities use the hosted-account DNS rules plus
//! atproto's top-level-domain restrictions, with an explicit self-hosted OAuth
//! extension for canonical encoded ports (including .test hosts). This extension
//! does not imply interoperability with public AT Protocol services.
//! PLC authorities require their full base32 identifier. Native at9p capsule
//! validation remains the native resolver's responsibility. This is not a
//! signature or ownership check (see [`crate::commit::Commit::verify`]).

use anyhow::{bail, ensure, Result};

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
    ensure!(
        did.len() <= 2048 && !did.contains(['/', '?', '#', '\\', '\0']),
        "repo authority must be a bare DID, not a DID URL"
    );
    let d = Did::new(did.to_owned());
    if d.is_did_web() {
        if is_path_form_did_web(did) {
            bail!(
                "path-form did:web {did:?} is not an accepted repo authority; the atproto profile requires host-form did:web"
            );
        }
        // Repository-specific compatibility with the canonical service DID
        // produced from configured OAuth origins. The public AT Protocol DID
        // profile is stricter: non-localhost ports are a Hyprstream self-hosted
        // extension, not a public interoperability guarantee.
        let (host_did, has_port) = if let Some((host, port)) = did.split_once("%3A") {
            let number = port.parse::<u16>()?;
            ensure!(
                number != 0 && number != 443 && port == number.to_string(),
                "repo did:web port must be canonical, nonzero and non-default"
            );
            (host, true)
        } else {
            (did, false)
        };
        // Validating the undecoded host rejects userinfo, IPs, localhost,
        // other percent escapes and repeated encoded ports without rewriting
        // the accepted DID used in repository keys or signatures.
        crate::did_op::validate_host_form_did_web(host_did)?;
        let tld = host_did.rsplit('.').next().unwrap_or_default();
        ensure!(
            tld.as_bytes().first().is_some_and(u8::is_ascii_lowercase),
            "public did:web top-level domain must start with an ASCII letter"
        );
        ensure!(
            !matches!(
                tld,
                "alt"
                    | "arpa"
                    | "example"
                    | "internal"
                    | "invalid"
                    | "local"
                    | "localhost"
                    | "onion"
            ),
            "public did:web top-level domain is reserved or disallowed by atproto"
        );
        ensure!(
            tld != "test" || has_port,
            "repo did:web .test requires the canonical self-hosted OAuth port form"
        );
        Ok(RepoAuthority::Web)
    } else if let Some(identifier) = did.strip_prefix("did:plc:") {
        ensure!(
            identifier.len() == 24
                && identifier
                    .bytes()
                    .all(|b| matches!(b, b'a'..=b'z' | b'2'..=b'7')),
            "repo did:plc requires 24 lowercase base32 characters"
        );
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
        assert!(
            !a.is_publicly_publishable(),
            "did:at9p is bridged, not public"
        );
    }

    #[test]
    fn accepts_classical_methods() {
        let web = accept_repo_authority("did:web:alice.example.com").unwrap();
        assert_eq!(web, RepoAuthority::Web);
        assert!(web.is_publicly_publishable());

        let plc = accept_repo_authority("did:plc:ewvi7nxzyoun6zhxrhs64oiz").unwrap();
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

    #[test]
    fn rejects_malformed_public_identifiers_and_did_urls() {
        for did in [
            "did:plc:",
            "did:plc:abc",
            "did:plc:ewvi7nxzyoun6zhxrhs64oi1",
            "did:plc:EWVI7NXZYOUN6ZHX RHS64OIZ",
            "did:web:",
            "did:web:example..com",
            "did:web:-example.com",
            "did:web:example.com-",
            "did:web:example_com",
            "did:web:example.com%23fragment",
            "did:web:example.com#fragment",
            "did:web:example.com?query",
            "did:web:example.com/path",
            "did:web:example.com\0",
            "did:plc:ewvi7nxzyoun6zhxrhs64oiz#key",
            "did:plc:ewvi7nxzyoun6zhxrhs64oiz?query",
            "did:plc:ewvi7nxzyoun6zhxrhs64oiz/path",
        ] {
            assert!(accept_repo_authority(did).is_err(), "{did:?}");
        }
    }

    #[test]
    fn public_web_authority_enforces_tld_and_production_host_rules() {
        for host in [
            "example.arpa",
            "example.onion",
            "example.123",
            "example.1com",
            "127.0.0.1",
            "[::1]",
            "%5B%3A%3A1%5D",
            "localhost",
            "localhost%3A6791",
            "example.com%3A443",
            "example.alt",
            "example.example",
            "example.internal",
            "example.invalid",
            "example.local",
            "example.localhost",
            "example.test",
        ] {
            assert!(
                accept_repo_authority(&format!("did:web:{host}")).is_err(),
                "{host}"
            );
        }
        for host in [
            "example.com",
            "alice.example.com",
            "123.example.com",
            "arpa.example.com",
            "onion.example.com",
            "example.xn--fiqs8s",
        ] {
            assert_eq!(
                accept_repo_authority(&format!("did:web:{host}")).unwrap(),
                RepoAuthority::Web
            );
        }
    }
    #[test]
    fn repo_web_authority_preserves_canonical_oauth_port_extension() {
        for did in [
            "did:web:pds.example.test%3A8443",
            "did:web:pds.example.com%3A8443",
            "did:web:example.com%3A1",
            "did:web:example.com%3A65535",
        ] {
            assert_eq!(accept_repo_authority(did).unwrap(), RepoAuthority::Web);
        }
        for host in [
            "example.com%3A0",
            "example.com%3A443",
            "example.com%3A65536",
            "example.com%3A08443",
            "example.com%3A+8443",
            "example.com%3A",
            "example.com%3A8443%3A8444",
            "example.com%3a8443",
            "example.com%253A8443",
            "example.com%3A8443:users",
            "user@example.com%3A8443",
            "user%40example.com%3A8443",
            "127.0.0.1%3A8443",
            "192.0.2.1%3A8443",
            "[::1]%3A8443",
            "localhost%3A8443",
            "example.123%3A8443",
            "example.1com%3A8443",
            "example.arpa%3A8443",
            "example.onion%3A8443",
            "example.internal%3A8443",
            "example.invalid%3A8443",
            "example.local%3A8443",
            "example.localhost%3A8443",
            "example.alt%3A8443",
            "example.example%3A8443",
            "example.com%3A8443#key",
            "example.com%3A8443?query",
            "example.com%3A8443/path",
        ] {
            assert!(
                accept_repo_authority(&format!("did:web:{host}")).is_err(),
                "{host}"
            );
        }
    }
}
