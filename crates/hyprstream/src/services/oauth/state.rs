//! OAuth 2.1 server state management.
//!
//! Manages registered clients, pending authorization codes, refresh tokens,
//! and delegates token issuance to PolicyService via ZMQ.

use anyhow::Context as _;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::token_store::TokenStore;
use super::user_service::UserService;
use crate::auth::user_store::UserStore;
use crate::config::OAuthConfig;
use crate::services::{DiscoveryClient, PolicyClient};
use hyprstream_util::TtlCache;

/// Extract RSA public key components (n, e) from PKCS#8 DER and build a JWK.
///
/// Uses simple_asn1 (transitive dep of jsonwebtoken) for DER parsing.
fn extract_rsa_jwk_from_der(pkcs8_der: &[u8], kid: &str) -> Option<serde_json::Value> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

    // PKCS#8 wraps the RSA key. We need to extract n and e from the inner
    // RSA public key. Rather than parsing ASN.1 manually, use jsonwebtoken
    // to create an EncodingKey and then use openssl to extract components.
    //
    // Shell out to openssl for public key component extraction.
    let temp = tempfile::NamedTempFile::new().ok()?;
    std::fs::write(temp.path(), pkcs8_der).ok()?;

    let output = std::process::Command::new("openssl")
        .args([
            "pkey",
            "-inform",
            "DER",
            "-in",
            temp.path().to_str()?,
            "-noout",
            "-text",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;

    // Parse modulus and exponent from openssl text output.
    // This is fragile but avoids adding ASN.1 parsing dependencies.
    let text = String::from_utf8_lossy(&output.stdout);
    let mut n_hex = String::new();
    let mut in_modulus = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Modulus:") || trimmed == "modulus:" {
            in_modulus = true;
            continue;
        }
        if trimmed.starts_with("Exponent:") || trimmed.starts_with("publicExponent:") {
            in_modulus = false;
            // Extract exponent (usually 65537 = 0x10001)
            // We'll use the standard value
            continue;
        }
        if in_modulus {
            // Lines like "    00:ab:cd:..."
            let hex_part: String = trimmed.chars().filter(char::is_ascii_hexdigit).collect();
            n_hex.push_str(&hex_part);
        }
    }

    if n_hex.is_empty() {
        return None;
    }

    // Strip leading zero byte if present (ASN.1 unsigned integer padding)
    if n_hex.starts_with("00") && n_hex.len() > 2 {
        n_hex = n_hex[2..].to_owned();
    }

    let n_bytes = hex::decode(&n_hex).ok()?;
    let e_bytes = vec![0x01, 0x00, 0x01]; // 65537

    let n_b64 = URL_SAFE_NO_PAD.encode(&n_bytes);
    let e_b64 = URL_SAFE_NO_PAD.encode(&e_bytes);

    Some(serde_json::json!({
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": kid,
        "n": n_b64,
        "e": e_b64,
    }))
}

/// A dynamically registered OAuth client (RFC 7591) or Client ID Metadata Document client.
///
/// Field set tracks
/// [draft-ietf-oauth-client-id-metadata-document-00] §4 (Client Metadata
/// Document) and [RFC 7591] §2 (Client Metadata): only the fields hyprstream
/// actually consumes are kept; unknown fields in the source document are
/// dropped at parse time.
#[derive(Debug, Clone)]
pub struct RegisteredClient {
    pub client_id: String,
    pub redirect_uris: Vec<String>,
    pub client_name: Option<String>,
    /// Homepage / informational URI. Shown on the consent screen.
    pub client_uri: Option<String>,
    /// Logo URL. Shown on the consent screen.
    pub logo_uri: Option<String>,
    /// Grant types this client is permitted to use. Empty = AS default
    /// (`authorization_code` + `refresh_token`).
    pub grant_types: Vec<String>,
    /// Response types this client is permitted to request. Empty = `code`.
    pub response_types: Vec<String>,
    /// Token endpoint client auth method. `"none"` = public client (PKCE
    /// is mandatory). `"private_key_jwt"` requires a non-empty `jwks` /
    /// `jwks_uri`.
    pub token_endpoint_auth_method: Option<String>,
    /// Inlined JWKS (CIMD §4 / RFC 7591 §2). Mutually exclusive with
    /// `jwks_uri` per RFC 7591. Used for `private_key_jwt` client auth and
    /// future request-object signing.
    pub jwks: Option<serde_json::Value>,
    /// JWKS endpoint URL — alternative to inline `jwks`.
    pub jwks_uri: Option<String>,
    /// Optional host `did:key` supplied by a PDS attachment client. Validated
    /// during registration and retained for PDS identity association.
    pub hyprstream_node_did: Option<String>,
    /// Space-separated scope tokens the client declared at registration
    /// (RFC 7591 / RFC 6749 §3.3). Requested scopes at PAR/authorize are
    /// validated as a subset of this (#1113 rev2). `None` → treated as
    /// "no client-side restriction" (the client declared no ceiling).
    pub scope: Option<String>,
    /// atproto OAuth: confidential clients set `dpop_bound_access_tokens:
    /// true` in their metadata document. Parsed and retained here (#1146);
    /// enforcement (rejecting confidential clients that omit it on the
    /// atproto profile) is follow-up work.
    pub dpop_bound_access_tokens: Option<bool>,
    /// True if this client was registered via Client ID Metadata Document (HTTPS URL client_id)
    pub is_cimd: bool,
    pub registered_at: Instant,
}

impl RegisteredClient {
    /// The client's declared scope tokens, parsed from [`Self::scope`].
    /// Empty when the client declared no scopes (no client-side restriction).
    pub fn declared_scopes(&self) -> Vec<String> {
        self.scope
            .as_deref()
            .map(|s| s.split_whitespace().map(str::to_owned).collect())
            .unwrap_or_default()
    }
}

/// A Pushed Authorization Request (RFC 9126) awaiting consumption.
///
/// Holds the already-validated authorize parameters keyed by the `request_uri`
/// returned to the client. Single-use, short TTL.
#[derive(Debug, Clone)]
pub struct PushedAuthRequest {
    pub params: super::authorize::AuthorizeParams,
    pub expires_at: Instant,
}

impl PushedAuthRequest {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Server-side state stashed between `authorize_get` and `authorize_post`,
/// keyed by the consent nonce. Carries the complete resolved authorization
/// request so hidden form fields never become mutable authority on POST.
#[derive(Debug, Clone)]
pub struct AuthorizeBinding {
    pub request: super::authorize::AuthorizeParams,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::user_store::UserProfile;
    use async_trait::async_trait;
    use hyprstream_rpc::crypto::CryptoPolicy;
    use rand::rngs::OsRng;

    fn state_with_user_store(store: Arc<dyn UserStore>) -> OAuthState {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::signer::LocalSigner;
        use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x41; 32]);
        let remote_key = ed25519_dalek::SigningKey::from_bytes(&[0x42; 32]).verifying_key();
        let make_client = || {
            Arc::new(
                RpcClientImpl::new(
                    LocalSigner::new(signing_key.clone()),
                    LazyUdsTransport::new("/dev/null/oauth-eligibility-test.sock".into()),
                    Some(remote_key),
                )
                .with_response_verify_policy(CryptoPolicy::Classical),
            )
        };
        let mut config = OAuthConfig::default();
        config.external_url = Some("https://pds.example.test".to_owned());
        OAuthState::new(
            &config,
            PolicyClient::new(make_client()),
            DiscoveryClient::new(make_client()),
            signing_key.verifying_key().to_bytes(),
        )
        .with_user_store(store)
    }

    struct KeyReadErrorStore {
        profile: UserProfile,
    }

    #[async_trait]
    impl UserStore for KeyReadErrorStore {
        async fn get_profile(&self, _username: &str) -> anyhow::Result<Option<UserProfile>> {
            Ok(Some(self.profile.clone()))
        }

        async fn register(&self, _username: &str) -> anyhow::Result<String> {
            anyhow::bail!("not used")
        }

        async fn set_profile(&self, _username: &str, _profile: crate::auth::UserProfilePatch) -> anyhow::Result<()> {
            anyhow::bail!("not used")
        }

        async fn remove(&self, _username: &str) -> anyhow::Result<bool> {
            anyhow::bail!("not used")
        }

        async fn list_users(&self) -> Vec<String> {
            Vec::new()
        }

        async fn search(
            &self,
            _filter: &crate::auth::user_store::UserFilter,
        ) -> anyhow::Result<Vec<(String, UserProfile)>> {
            anyhow::bail!("not used")
        }

        async fn set_active(&self, _username: &str, _active: bool) -> anyhow::Result<()> {
            anyhow::bail!("not used")
        }

        async fn list_pubkeys(
            &self,
            _username: &str,
        ) -> anyhow::Result<Vec<crate::auth::user_store::PubkeyEntry>> {
            anyhow::bail!("synthetic key read failure")
        }

        async fn add_pubkey(
            &self,
            _username: &str,
            _pubkey: ed25519_dalek::VerifyingKey,
            _label: Option<String>,
        ) -> anyhow::Result<String> {
            anyhow::bail!("not used")
        }

        async fn add_pubkey_hybrid(
            &self,
            _username: &str,
            _pubkey: ed25519_dalek::VerifyingKey,
            _ml_dsa_vk: Vec<u8>,
            _label: Option<String>,
        ) -> anyhow::Result<String> {
            anyhow::bail!("not used")
        }

        async fn remove_pubkey(&self, _username: &str, _fingerprint: &str) -> anyhow::Result<bool> {
            anyhow::bail!("not used")
        }

        async fn get_pubkey_user(&self, _fingerprint: &str) -> anyhow::Result<Option<String>> {
            anyhow::bail!("not used")
        }

        async fn touch_pubkey(&self, _username: &str, _fingerprint: &str) -> anyhow::Result<()> {
            anyhow::bail!("not used")
        }
    }

    #[test]
    fn mesh_kem_derivation_failure_fails_closed_under_hybrid() {
        let key = ed25519_dalek::SigningKey::generate(&mut OsRng);
        let err = mesh_kem_public_for_policy(&key, CryptoPolicy::Hybrid, |_| {
            Err(anyhow::anyhow!("synthetic derivation failure"))
        })
        .unwrap_err();

        assert!(
            err.to_string().contains("Hybrid policy"),
            "unexpected error: {err:?}",
        );
    }

    #[test]
    fn mesh_kem_derivation_failure_is_empty_under_classical() {
        let key = ed25519_dalek::SigningKey::generate(&mut OsRng);
        let public = mesh_kem_public_for_policy(&key, CryptoPolicy::Classical, |_| {
            Err(anyhow::anyhow!("synthetic derivation failure"))
        })
        .unwrap();

        assert!(public.is_none());
    }

    /// #1113 rev2 / #1124: a local username has NO atproto identity →
    /// rejected (fail closed). Synthesizing a path-form did:web is
    /// spec-invalid and was removed; provisioning is tracked in #1124.
    #[test]
    fn subject_did_for_local_username_rejected() {
        let err = subject_did_for("https://pds.example.com", "alice").unwrap_err();
        assert_eq!(err, SubjectDidError::NoAtprotoIdentity);
    }

    /// #1113 rev2: a valid `did:plc` is accepted (real atproto account DID).
    #[test]
    fn subject_did_for_did_plc_accepted() {
        assert_eq!(
            subject_did_for("https://x", "did:plc:abcdefghijklmnqrstuvwx2p").unwrap(),
            "did:plc:abcdefghijklmnqrstuvwx2p"
        );
    }

    /// #1113 rev2: a host-form `did:web:host[:port]` (NO path) is accepted.
    #[test]
    fn subject_did_for_host_form_did_web_accepted() {
        assert_eq!(
            subject_did_for("https://x", "did:web:pds.example.com").unwrap(),
            "did:web:pds.example.com"
        );
        assert_eq!(
            subject_did_for("https://x", "did:web:127.0.0.1:6791").unwrap(),
            "did:web:127.0.0.1:6791"
        );
    }

    /// #1113 rev2: a path-form did:web (rejected by the atproto DID profile)
    /// is rejected — the atproto DID profile forbids path components.
    #[test]
    fn subject_did_for_path_form_did_web_rejected() {
        let err = subject_did_for("https://x", "did:web:pds.example.com:users:alice").unwrap_err();
        assert_eq!(err, SubjectDidError::PathFormDidWeb);
    }

    /// #1113 correction: `did:at9p` and other non-plc/non-web DID methods
    /// are NOT eligible for atproto OAuth tokens.
    #[test]
    fn subject_did_for_at9p_fails_closed() {
        let err = subject_did_for("https://pds.example.com", "did:at9p:abc").unwrap_err();
        assert_eq!(err, SubjectDidError::UnsupportedMethod);
    }

    /// #1113 rev2: a malformed did:plc (empty body) is rejected.
    #[test]
    fn subject_did_for_malformed_plc_rejected() {
        let err = subject_did_for("https://x", "did:plc:").unwrap_err();
        assert_eq!(err, SubjectDidError::MalformedDid);
    }

    /// #1113 r4: a SHORT did:plc (e.g. 9 chars) is rejected — @atproto/did
    /// requires exactly 24 base32 chars. The old fixture `did:plc:abc123xyz`
    /// passed validation but failed the real schema.
    #[test]
    fn subject_did_for_short_plc_rejected() {
        let err = subject_did_for("https://x", "did:plc:abc123xyz").unwrap_err();
        assert_eq!(err, SubjectDidError::MalformedDid);
    }

    /// #1113 r5: a classical account WITH a mapped did:plc succeeds —
    /// the mapped DID is returned (the positive success path).
    #[test]
    fn atproto_eligibility_mapped_did_plc_succeeds() {
        use crate::auth::user_store::{KeyAlgorithm, PubkeyEntry, UserProfile};
        let pubkeys = vec![PubkeyEntry {
            fingerprint: "key1".into(),
            pubkey: ed25519_dalek::VerifyingKey::from_bytes(&[1u8; 32]).unwrap(),
            label: None,
            created_at: 0,
            last_used_at: None,
            algorithm: KeyAlgorithm::Ed25519,
            pq_pubkey: None,
        }];
        let mut profile = UserProfile::default();
        profile.atproto_did = Some("did:plc:abcdefghijklmnqrstuvwx2p".into());
        let did = evaluate_atproto_eligibility(&pubkeys, Some(&profile), "https://x").unwrap();
        assert_eq!(did, "did:plc:abcdefghijklmnqrstuvwx2p");
    }

    /// #1113 r5: a classical account WITHOUT a mapping fails closed.
    #[test]
    fn atproto_eligibility_no_mapping_fails_closed() {
        use crate::auth::user_store::{KeyAlgorithm, PubkeyEntry, UserProfile};
        let pubkeys = vec![PubkeyEntry {
            fingerprint: "key1".into(),
            pubkey: ed25519_dalek::VerifyingKey::from_bytes(&[1u8; 32]).unwrap(),
            label: None,
            created_at: 0,
            last_used_at: None,
            algorithm: KeyAlgorithm::Ed25519,
            pq_pubkey: None,
        }];
        let profile = UserProfile::default();
        let err = evaluate_atproto_eligibility(&pubkeys, Some(&profile), "https://x").unwrap_err();
        assert_eq!(err, SubjectDidError::NoAtprotoIdentity);
    }

    /// #1113 r5: an at9p-backed account is rejected from the atproto profile
    /// — inspected via the key mapping, not the subject string.
    #[test]
    fn atproto_eligibility_at9p_account_rejected() {
        use crate::auth::user_store::{KeyAlgorithm, PubkeyEntry, UserProfile};
        let pubkeys = vec![PubkeyEntry {
            fingerprint: "pq-key".into(),
            pubkey: ed25519_dalek::VerifyingKey::from_bytes(&[1u8; 32]).unwrap(),
            label: None,
            created_at: 0,
            last_used_at: None,
            algorithm: KeyAlgorithm::HybridEd25519MlDsa65,
            pq_pubkey: Some(vec![0u8; 1952]), // ML-DSA-65 public key bytes
        }];
        let mut profile = UserProfile::default();
        // Even with a mapped DID, the at9p key algorithm rejects first.
        profile.atproto_did = Some("did:plc:abcdefghijklmnqrstuvwx2p".into());
        let err = evaluate_atproto_eligibility(&pubkeys, Some(&profile), "https://x").unwrap_err();
        assert_eq!(err, SubjectDidError::At9pBackedAccount);
    }

    /// #1113 r5: the default production backend must retain the mapped DID,
    /// and the real account lookup (keys + profile) must reach the success
    /// path after closing and reopening RocksDB.
    #[tokio::test]
    async fn atproto_eligibility_succeeds_after_rocksdb_roundtrip() {
        use crate::auth::rocksdb_store::RocksDbUserStore;
        use crate::auth::user_store::UserProfile;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&[0x33; 32]);
        {
            let store = RocksDbUserStore::open(dir.path()).unwrap();
            store.register("alice").await.unwrap();
            store
                .add_pubkey(
                    "alice",
                    signing_key.verifying_key(),
                    Some("login".to_owned()),
                )
                .await
                .unwrap();
            store
                .set_profile(
                    "alice",
                    UserProfile {
                        atproto_did: Some("did:plc:abcdefghijklmnqrstuvwx2p".to_owned()),
                        ..Default::default()
                    }.into(),
                )
                .await
                .unwrap();
        }

        let reopened: Arc<dyn UserStore> = Arc::new(RocksDbUserStore::open(dir.path()).unwrap());
        assert_eq!(
            reopened
                .get_profile("alice")
                .await
                .unwrap()
                .unwrap()
                .atproto_did
                .as_deref(),
            Some("did:plc:abcdefghijklmnqrstuvwx2p")
        );
        let state = state_with_user_store(reopened);
        assert_eq!(
            state
                .check_atproto_account_eligibility("alice")
                .await
                .unwrap(),
            "did:plc:abcdefghijklmnqrstuvwx2p"
        );
    }

    /// #1113 r5: a profile read that would otherwise succeed cannot mask a
    /// key-store failure. The account is rejected rather than downgraded to
    /// an apparently keyless identity.
    #[tokio::test]
    async fn atproto_eligibility_key_read_error_fails_closed() {
        let store = Arc::new(KeyReadErrorStore {
            profile: UserProfile {
                atproto_did: Some("did:plc:abcdefghijklmnqrstuvwx2p".to_owned()),
                ..Default::default()
            },
        });
        let state = state_with_user_store(store);
        assert_eq!(
            state
                .check_atproto_account_eligibility("alice")
                .await
                .unwrap_err(),
            SubjectDidError::NoAtprotoIdentity
        );
    }

    /// #1113 rev2 finding 7: atproto profile activates only when `atproto` is
    /// in the granted set — device/generic flows without it stay non-atproto.
    #[test]
    fn atproto_profile_detection() {
        assert!(atproto_profile_active(&["atproto".to_owned()]));
        assert!(atproto_profile_active(&[
            "atproto".to_owned(),
            "read:*:*".to_owned()
        ]));
        assert!(!atproto_profile_active(&["read:*:*".to_owned()]));
        assert!(!atproto_profile_active(&[]));
    }

    #[tokio::test]
    async fn authorize_binding_sweep_tracks_live_nonces() {
        let store = Arc::new(KeyReadErrorStore { profile: UserProfile::default() });
        let state = state_with_user_store(store);
        let binding = AuthorizeBinding {
            request: crate::services::oauth::authorize::AuthorizeParams {
                client_id: "client".to_owned(),
                redirect_uri: "https://client.example/callback".to_owned(),
                code_challenge: "challenge".to_owned(),
                code_challenge_method: "S256".to_owned(),
                response_type: "code".to_owned(),
                state: None,
                scope: Some("read:*:*".to_owned()),
                resource: None,
                nonce: None,
                dpop_jkt: None,
                client_assertion_jkt: None,
            },
        };
        let now = Instant::now();
        state.pending_nonces.write().await.insert("live".to_owned(), now + Duration::from_secs(60));
        state.pending_nonces.write().await.insert("expired".to_owned(), now - Duration::from_secs(1));
        let mut bindings = state.pending_authorize_bindings.write().await;
        bindings.insert("live".to_owned(), binding.clone());
        bindings.insert("expired".to_owned(), binding.clone());
        bindings.insert("orphan".to_owned(), binding);
        drop(bindings);

        state.sweep_authorize_sessions(now).await;

        let bindings = state.pending_authorize_bindings.read().await;
        assert_eq!(bindings.len(), 1);
        assert!(bindings.contains_key("live"));
    }

    /// #1113 rev2 finding 4: requested scopes are validated against the
    /// server-supported ∩ client-declared sets; garbage/undeclared tokens
    /// yield invalid_scope and the actual granted set is returned.
    #[test]
    fn validate_requested_scopes_rejects_garbage() {
        let server = [
            "atproto".to_owned(),
            "transition:generic".to_owned(),
            "read:*:*".to_owned(),
        ];
        let res = validate_requested_scopes(
            &["atproto".to_owned(), "bogus".to_owned()],
            &server,
            None,
            true,
        );
        assert_eq!(res.unwrap_err(), ScopeError::InvalidScope);
    }

    /// #1113 rev2 finding 4: the granted set is the intersection of requested
    /// with server-supported, intersected with the client's declared scopes.
    #[test]
    fn validate_requested_scopes_intersects_client_declared() {
        let server = [
            "atproto".to_owned(),
            "transition:generic".to_owned(),
            "read:*:*".to_owned(),
        ];
        let declared = ["atproto".to_owned(), "read:*:*".to_owned()];
        let granted = validate_requested_scopes(
            &["atproto".to_owned(), "read:*:*".to_owned()],
            &server,
            Some(&declared),
            true,
        )
        .unwrap();
        assert_eq!(granted, vec!["atproto".to_owned(), "read:*:*".to_owned()]);
        // Client did NOT declare transition:generic → requesting it is invalid.
        let res = validate_requested_scopes(
            &["atproto".to_owned(), "transition:generic".to_owned()],
            &server,
            Some(&declared),
            true,
        );
        assert_eq!(res.unwrap_err(), ScopeError::InvalidScope);
    }

    /// #1113 rev2 finding 4: when the atproto profile is requested, `atproto`
    /// MUST survive into the granted set, else AtprotoRequired.
    #[test]
    fn validate_requested_scopes_requires_atproto_when_profile_active() {
        let server = ["atproto".to_owned(), "read:*:*".to_owned()];
        let res = validate_requested_scopes(&["read:*:*".to_owned()], &server, None, true);
        assert_eq!(res.unwrap_err(), ScopeError::AtprotoRequired);
        // Non-atproto request (require_atproto=false) does not require it.
        let granted =
            validate_requested_scopes(&["read:*:*".to_owned()], &server, None, false).unwrap();
        assert_eq!(granted, vec!["read:*:*".to_owned()]);
    }

    /// #1113 rev2 finding 5: the issuer is canonicalized to an exact origin
    /// (scheme://host[:port]) — trailing slash and path are stripped so
    /// endpoint URLs and the token `iss` are well-formed.
    #[test]
    fn canonical_issuer_origin_strips_path_and_slash() {
        assert_eq!(
            canonical_issuer_origin("https://pds.example.com/").as_deref(),
            Some("https://pds.example.com")
        );
        assert_eq!(
            canonical_issuer_origin("https://pds.example.com/oauth/par").as_deref(),
            Some("https://pds.example.com")
        );
        assert_eq!(
            canonical_issuer_origin("http://127.0.0.1:6791").as_deref(),
            Some("http://127.0.0.1:6791")
        );
        assert!(canonical_issuer_origin("pds.example.com").is_none());
    }

    /// #1113 rev2 F4/F6: the default grant set (omitted-scope fallback) must
    /// NOT include `atproto` — those scopes are supported-but-explicit. A
    /// client that omits `scope` must NOT silently activate the strict profile.
    #[test]
    fn default_scopes_do_not_include_atproto() {
        let cfg = crate::config::OAuthConfig::default();
        assert!(
            !cfg.default_scopes.iter().any(|s| s == "atproto"),
            "default_scopes must NOT include atproto (supported-but-explicit): {:?}",
            cfg.default_scopes
        );
        // But server_supported_scopes DOES include it.
        let supported = advertised_scopes_for_test(&cfg.default_scopes);
        assert!(supported.iter().any(|s| s == "atproto"));
        assert!(supported.iter().any(|s| s == "transition:generic"));
    }

    /// Helper mirroring the advertised_scopes logic for test assertions.
    fn advertised_scopes_for_test(default_scopes: &[String]) -> Vec<String> {
        let mut scopes = default_scopes.to_vec();
        for s in &["atproto", "transition:generic"] {
            if !scopes.iter().any(|x| x == *s) {
                scopes.push((*s).to_owned());
            }
        }
        scopes
    }
}

/// A pending authorization code awaiting token exchange.
#[derive(Debug, Clone)]
pub struct PendingAuthCode {
    pub code: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub code_challenge: String,
    pub scopes: Vec<String>,
    /// RFC 8707 resource indicator (the audience for the token)
    pub resource: Option<String>,
    /// OIDC nonce — echoed into the id_token when scope includes "openid".
    pub oidc_nonce: Option<String>,
    pub created_at: Instant,
    pub expires_at: Instant,
    /// Authenticated username from Ed25519 challenge-response on the consent page.
    /// Used as the JWT `sub` claim for the issued token.
    pub username: String,
    /// Ed25519 verifying key verified during challenge-response.
    /// Included in the JWT `pub_key` claim to bind the user's key identity.
    pub verifying_key: Option<ed25519_dalek::VerifyingKey>,
    /// DPoP key thumbprint bound at PAR (#1113 rev2 finding 3). When the
    /// atproto profile is active, the token endpoint must receive a DPoP
    /// proof from the same key.
    pub dpop_jkt: Option<String>,
    /// RFC 7638 thumbprint of the client-assertion key verified at PAR
    /// (#1146 T3.3). When set, the token endpoint must receive a
    /// `client_assertion` signed by the same key, and the binding is
    /// carried into the issued refresh token.
    pub client_assertion_jkt: Option<String>,
}

impl PendingAuthCode {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Pending external OIDC authentication flow.
///
/// Stores the original hyprstream authorize request so it can be resumed
/// after the user authenticates with the external provider.
#[derive(Debug, Clone)]
pub struct PendingExternalAuth {
    pub provider_slug: String,
    pub external_state: String,
    pub external_nonce: String,
    /// Provider kind, carried through for dispatch in the callback handler.
    pub provider_kind: crate::config::ProviderKind,
    /// Whether PKCE was sent to the external provider.
    pub pkce_supported: bool,
    pub pkce_verifier: String,
    pub client_secret: Option<String>,
    pub token_endpoint: String,
    // Original hyprstream authorize request
    pub original_client_id: String,
    pub original_redirect_uri: String,
    pub original_code_challenge: String,
    pub original_scopes: String,
    pub original_state: Option<String>,
    pub original_resource: Option<String>,
    pub original_oidc_nonce: Option<String>,
    pub created_at: Instant,
    pub expires_at: Instant,
}

/// Status of a pending device authorization code (RFC 8628).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceCodeStatus {
    /// User has not yet approved or denied.
    Pending,
    /// User approved the authorization request.
    Approved,
    /// User denied the authorization request.
    Denied,
}

/// A pending device authorization code (RFC 8628).
#[derive(Debug, Clone)]
pub struct PendingDeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub client_id: String,
    pub scopes: Vec<String>,
    /// RFC 8707 resource indicator (the audience for the token)
    pub resource: Option<String>,
    pub status: DeviceCodeStatus,
    pub created_at: Instant,
    pub expires_at: Instant,
    /// Minimum polling interval in seconds
    pub interval: u64,
    /// Last time the client polled for this code
    pub last_polled: Option<Instant>,
    /// Random nonce for challenge-response auth (43 chars base64url of 32 bytes)
    pub nonce: String,
    /// Username of the person who approved this code (set on POST /verify success)
    pub approved_by: Option<String>,
    /// Ed25519 verifying key verified during device challenge-response.
    /// Included in the JWT `pub_key` claim to bind the user's key identity.
    pub verifying_key: Option<ed25519_dalek::VerifyingKey>,
}

impl PendingDeviceCode {
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// A stored refresh token entry (OAuth 2.1 rotation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshTokenEntry {
    pub client_id: String,
    /// JWT subject (username) of the token owner.
    pub username: String,
    pub scopes: Vec<String>,
    pub resource: Option<String>,
    /// Unix timestamp (seconds) at which this token expires.
    pub expires_at_unix: i64,
    /// Ed25519 verifying key bytes bound to this token (cnf key continuity on refresh).
    pub verifying_key_bytes: Option<[u8; 32]>,
    /// RFC 9449 JWK thumbprint. When set, refresh requires a DPoP proof from
    /// this exact key; it is carried forward during refresh-token rotation.
    #[serde(default)]
    pub dpop_jkt: Option<String>,
    /// RFC 7638 thumbprint of the client-assertion key this session was
    /// issued under (#1146 T3.3). When set, refresh requires a
    /// `client_assertion` verified by the same key — refresh cannot switch
    /// to another registered key, and removing the key from the client's
    /// JWKS revokes the session. Carried forward during rotation.
    #[serde(default)]
    pub client_assertion_jkt: Option<String>,
    /// Present only for UCAN-grant refresh tokens (`client_id` `ucan-grant:{sub}`).
    ///
    /// MAC #547 / B1 (#673): refresh of a UCAN grant is NOT a free re-mint — the
    /// refresh path must re-run the S6 gate chain (`evaluate_refresh`) against the
    /// grant, because the ceiling may have been amended or revoked since the
    /// access token was minted (ZSP: access is re-evaluated on refresh). That
    /// requires persisting the grant + the requested access so the refresh path
    /// can re-present them. `None` for every non-UCAN-grant refresh token, which
    /// keeps the generic OAuth 2.1 rotation path unchanged.
    #[serde(default)]
    pub ucan_grant: Option<UcanGrantRefresh>,
}

/// Re-evaluation context persisted alongside a UCAN-grant refresh token so the
/// refresh path can re-present the grant to `evaluate_refresh` (MAC #547 / B1
/// #673). Carrying the grant itself (not just an id) lets the gate chain
/// re-check the ceiling against current state on every refresh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UcanGrantRefresh {
    /// `base64url(CBOR)` of the UCAN grant, re-presented to `evaluate_refresh`
    /// verbatim (the same encoding the client sent as `subject_token`).
    pub grant_cbor_b64: String,
    /// BLAKE3 content id (hex) of the CBOR grant. Binds this refresh entry to
    /// exactly the grant it was minted from and is checked on every refresh so
    /// a corrupted/substituted stored blob fails closed.
    pub grant_cid: String,
    /// The RFC 8693 requested `scope` (the S3 `action:resource:identifier`
    /// triple) used to rebuild the `GrantRequest` on refresh.
    pub requested_scope: Option<String>,
    /// The RFC 8707 `audience` resource indicator, if any.
    pub audience: Option<String>,
}

impl RefreshTokenEntry {
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at_unix
    }
}

/// Shared OAuth server state.
pub struct OAuthState {
    /// Dynamically-registered (RFC 7591) clients keyed by issued UUID
    /// client_id. CIMD clients live in `cimd_cache` instead — DCR
    /// entries have no TTL and outlive cache evictions, so storage is
    /// separated.
    pub clients: RwLock<HashMap<String, RegisteredClient>>,
    /// CIMD metadata cache. CIMD documents (HTTPS-URL client_ids) are
    /// fetched + verified once, then cached respecting HTTP cache
    /// headers, bounded TTL and capacity. See `cimd_cache` module.
    pub cimd_cache: Arc<super::cimd_cache::CimdCache>,
    /// JWKS-URI fetch cache for `private_key_jwt` client auth. Keyed by
    /// the absolute URL; value is `(parsed_jwks_json, fetched_at)`.
    /// Entries expire after `client_jwks_uri_cache_ttl_secs` and are
    /// evicted lazily on read. Capacity is implicitly bounded by the
    /// number of registered clients with jwks_uri (typically small).
    pub jwks_uri_cache: RwLock<HashMap<String, (serde_json::Value, Instant)>>,
    /// TTL for the `jwks_uri_cache`. Copied from
    /// [`OAuthConfig::client_jwks_uri_cache_ttl_secs`] at construction.
    pub client_jwks_uri_cache_ttl: Duration,
    /// Pending authorization codes (single-use, 60s TTL)
    pub pending_codes: RwLock<HashMap<String, PendingAuthCode>>,
    /// Pending authorize nonces (single-use, 5-min TTL).
    /// Proves a nonce was issued by this server and hasn't been replayed.
    pub pending_nonces: RwLock<HashMap<String, Instant>>,
    /// Complete resolved authorization requests, keyed and consumed alongside
    /// the consent nonce so browser-carried fields are never authority.
    pub pending_authorize_bindings: RwLock<HashMap<String, AuthorizeBinding>>,
    /// Pending Pushed Authorization Requests (RFC 9126), keyed by `request_uri`.
    /// Single-use, 60s TTL. In-memory is correct here — no value to persistence.
    pub pending_par_requests: RwLock<HashMap<String, super::state::PushedAuthRequest>>,
    /// Pending device authorization codes (RFC 8628), keyed by device_code
    pub pending_device_codes: RwLock<HashMap<String, PendingDeviceCode>>,
    /// Reverse lookup: user_code -> device_code
    pub device_code_by_user_code: RwLock<HashMap<String, String>>,
    /// Persistent refresh token store. Keyed by opaque token string.
    /// None when no credentials path is configured (tokens silently lost on restart).
    pub token_db: Option<Arc<dyn TokenStore>>,
    /// PolicyClient for JWT token issuance via ZMQ
    pub policy_client: PolicyClient,
    /// DiscoveryClient for resolving service QUIC endpoints via ZMQ
    pub discovery_client: DiscoveryClient,
    /// Issuer URL (e.g., "http://localhost:6791")
    pub issuer_url: String,
    /// Default scopes for new clients
    pub default_scopes: Vec<String>,
    /// Access token TTL in seconds
    pub token_ttl: u32,
    /// Refresh token TTL in seconds
    pub refresh_token_ttl: u32,
    /// When true, `/oauth/authorize` rejects inline params and requires a `request_uri`
    /// from a prior `/oauth/par` call (RFC 9126). Advertised in server metadata.
    pub require_pushed_authorization_requests: bool,
    /// HTTP client for fetching Client ID Metadata Documents
    pub http_client: reqwest::Client,
    /// Raw Ed25519 verifying key bytes (32 bytes) for the JWKS endpoint.
    pub verifying_key_bytes: [u8; 32],
    /// User credential store for Ed25519 challenge-response device verification.
    /// Now backed by `user_service`. Legacy code uses `user_store_reader()`.
    /// Kept as Option for backward-compatible `is_none()` checks.
    pub user_service: Option<Arc<UserService>>,
    /// Ed25519 signing key for signing entity configurations (OpenID Federation 1.0).
    /// `None` when not configured.
    pub signing_key: Option<ed25519_dalek::SigningKey>,
    /// Named root-DID identity slots (drain/active/lead).
    ///
    /// This store is seeded from `signing_key` on first deployment, then rotated
    /// independently with bounded overlap. The root DID publisher snapshots all
    /// three roles; entity statements sign with the active role.
    pub root_identity_key_store: Option<Arc<crate::auth::SigningKeyStore>>,
    /// OpenID Federation 1.0 Trust Anchor URLs.
    /// Included as `authority_hints` in the entity configuration JWT.
    pub authority_hints: Vec<String>,
    /// Pending external OIDC authentication flows (keyed by external state).
    pub pending_external_auths: RwLock<HashMap<String, PendingExternalAuth>>,
    /// OIDC discovery cache for external providers.
    pub oidc_discovery: super::oidc_discovery::SharedDiscoveryCache,
    /// Server-side session store for browser login flow.
    pub sessions: super::session::SessionStore,
    /// In-memory `com.atproto.*` XRPC read slice repo store (#1112). Populated
    /// by the write path (#910) and tests; the read endpoints consult it
    /// directly. Default-empty so existing construction sites need no change.
    pub xrpc_repos: Arc<super::xrpc::XrpcRepoStore>,
    /// When `true`, the XRPC read-slice routes (`/xrpc/…`) are mounted on the
    /// OAuth router (#1112). Copied from `OAuthConfig::xrpc_read_slice`.
    pub xrpc_read_slice: bool,
    /// RSA encoding key for RS256 id_token signing (optional, loaded from secrets).
    pub rsa_encoding_key: Option<jsonwebtoken::EncodingKey>,
    /// RSA public key as JWK JSON (for the JWKS endpoint).
    pub rsa_jwk: Option<serde_json::Value>,
    /// RSA key kid (for the JWT header).
    pub rsa_kid: Option<String>,
    /// Seen DPoP JTIs for replay prevention (RFC 9449 §11.1). Backed by
    /// the shared `TtlCache` with atomic check-and-record — see
    /// `check_and_record_dpop_jti`. TTL = iat + 120s per entry.
    pub dpop_jti_seen: TtlCache<String, ()>,
    /// Seen client-assertion JTIs for replay prevention (RFC 7523 §3 /
    /// atproto OAuth; #1146 T3.3). Keyed `{client_id}\u{1f}{jti}`; TTL =
    /// the assertion's remaining `exp` lifetime. See
    /// `check_and_record_assertion_jti`.
    pub assertion_jti_seen: TtlCache<String, ()>,
    /// Server-issued DPoP nonces. Value = expiry unix timestamp.
    pub dpop_nonces: RwLock<HashMap<String, i64>>,
    /// Per-client (keyed by DPoP `jkt` thumbprint) nonce-issuance state.
    /// Once a jkt appears here, RFC 9449 §8 enforcement kicks in: subsequent
    /// proofs from the same key MUST carry a server-issued nonce. Value =
    /// expiry unix timestamp (matches nonce TTL; entry pruned when expired
    /// to allow re-bootstrap after silence).
    pub dpop_clients_seen: RwLock<HashMap<String, i64>>,
    /// Trusted external OIDC issuers for the JWT bearer grant (RFC 7523).
    pub trusted_issuers: std::collections::HashMap<String, crate::config::TrustedIssuerConfig>,
    /// CA JWT signing key for browser WIT issuance (POST /oauth/wit).
    /// Derived from the root CA key via derive_purpose_key("hyprstream-jwt-v1").
    /// None when credentials are unavailable (WIT endpoint returns 503).
    pub ca_jwt_key: Option<Arc<ed25519_dalek::SigningKey>>,
    /// Anonymous device identity store.
    pub device_store: Option<Arc<dyn crate::auth::DeviceStore>>,
    /// Unix timestamp of when the JWT signing key became active (nbf for JWKS entry).
    pub jwt_key_nbf: i64,
    /// Unix timestamp of when the JWT signing key expires (exp for JWKS entry).
    /// Default: jwt_key_nbf + 14 days.
    pub jwt_key_exp: i64,
    /// Multi-slot JWT signing key store for rotation (drain/active/lead lifecycle).
    /// When present, JWKS serves all slots and issuance uses the active key.
    pub signing_key_store: Option<Arc<crate::auth::SigningKeyStore>>,
    /// Shared JWT ID blocklist for access token revocation (shared with PolicyService).
    pub jti_blocklist: Option<Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>>,
    /// ES256 (P-256) signing key rotation store for JWKS and DPoP/atproto interop.
    pub es256_key_store: Option<Arc<crate::auth::Es256SigningKeyStore>>,
    /// ML-DSA-65 signing key rotation store for PQ-hybrid JWT issuance.
    pub ml_dsa_key_store: Option<Arc<crate::auth::MlDsaSigningKeyStore>>,
    /// MAC #547 / B2 (#674): the tamper-evident audit sink the S6 grant path
    /// (`mac::exchange::audited_evaluate_grant`/`audited_evaluate_refresh`)
    /// records every decision through. `None` means the grant-path audit
    /// trail is not configured — the grant path treats that as fail-closed
    /// (deny, not "audit best-effort") rather than minting unaudited tokens.
    /// Production wiring is [`crate::mac::audit::WalAuditStore`] + an
    /// [`crate::mac::audit::cose::OwnedCoseAuditSigner`] (see the `oauth`
    /// service factory).
    pub audit_sink: Option<Arc<dyn crate::mac::audit::AuditSink>>,
    /// QUIC/WebTransport cert hashes (SHA-256 of leaf DER, `sha2-256` multihash encoding).
    /// Published in the DID-doc `#quic` service entry so peers can pin the cert (#185).
    /// A set so cert rotation can publish old + new simultaneously.
    pub quic_cert_hashes: Vec<[u8; 32]>,
    /// Public QUIC URI (`https://host:port`) for the DID-doc service entry (#185).
    /// None until the QUIC server is started and the cert hash is known.
    pub quic_public_uri: Option<String>,
    /// Raw ML-DSA-65 verifying-key bytes (1952 bytes) for the node's mesh
    /// post-quantum signing key (#157). Published as the `#mesh-pq` Multikey
    /// verification method in the root DID document. Derived from the same
    /// Ed25519 key as [`Self::signing_key`] (via `derive_mesh_mldsa_key`), so
    /// the published VM equals the key the node signs mesh responses with.
    /// `None` when the entity signing key is not configured.
    pub mesh_pq_verifying_key: Option<Vec<u8>>,
    /// The node's `#mesh-kem` hybrid keyAgreement public material (S1 / #552):
    /// one ML-KEM-768-hybrid recipient public (X25519 + ML-KEM-768 encapsulation
    /// keys, suite `HyKemX25519MlKem768`). Published as the `keyAgreement`
    /// verification methods in the root DID document. Derived from the same
    /// Ed25519 key as [`Self::signing_key`] (via `derive_mesh_kem_recipient`),
    /// so the published keys equal what the node decapsulates `#mesh-kem`
    /// envelopes with. `None` when the entity signing key is not configured,
    /// or when derivation failed under Classical policy (Hybrid fails closed
    /// during `with_signing_key`).
    pub mesh_kem_public: Option<hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>,
    /// Self-certifying at9p capsule rendered from the same live signing key as
    /// the root did:web document, never from deployment-provisioned bytes.
    pub(crate) at9p_identity: Option<super::did_document::RenderedAt9pIdentity>,
    /// #282: the node's iroh endpoint id (its Ed25519 `node_id`, 32 bytes),
    /// published only as an `IrohTransport` service entry when bound. It is not
    /// a verification method or JWKS key (#1031).
    pub iroh_node_id: Option<[u8; 32]>,
    /// #282: iroh relay URLs to advertise in the `IrohTransport` entry's
    /// `relays`. Empty = rely on pkarr/DNS discovery for reachability (the
    /// peer resolves direct paths by node_id alone).
    pub iroh_relays: Vec<String>,
}

fn mesh_kem_public_for_policy(
    key: &ed25519_dalek::SigningKey,
    policy: hyprstream_rpc::crypto::CryptoPolicy,
    derive: impl FnOnce(
        &ed25519_dalek::SigningKey,
    ) -> anyhow::Result<hyprstream_rpc::crypto::hybrid_kem::RecipientKeypair>,
) -> anyhow::Result<Option<hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>> {
    match derive(key) {
        Ok(kem_kp) => Ok(Some(kem_kp.public())),
        Err(e) if policy.uses_pq() => Err(e)
            .context("failed to derive #mesh-kem hybrid keyAgreement identity under Hybrid policy"),
        Err(e) => {
            tracing::warn!(
                error = %e,
                "failed to derive #mesh-kem hybrid keyAgreement identity under Classical policy; \
                 root DID document will publish an empty keyAgreement relationship",
            );
            Ok(None)
        }
    }
}

impl OAuthState {
    /// DPoP jti replay-dedup cache: capacity bound (memory ceiling for the
    /// `TtlCache<String, ()>`). 120s TTL per entry.
    const DPOP_JTI_MAX_ENTRIES: usize = 10_000;
    /// Inline reap budget per access (heap pops). Bounds tail latency.
    const DPOP_JTI_REAP_BUDGET: usize = 64;
    /// Client-assertion jti replay-dedup cache: same bounds as the DPoP
    /// registry; per-entry TTL is the assertion's remaining lifetime.
    const ASSERTION_JTI_MAX_ENTRIES: usize = 10_000;
    /// Inline reap budget per access (heap pops). Bounds tail latency.
    const ASSERTION_JTI_REAP_BUDGET: usize = 64;

    pub fn new(
        config: &OAuthConfig,
        policy_client: PolicyClient,
        discovery_client: DiscoveryClient,
        verifying_key_bytes: [u8; 32],
    ) -> Self {
        Self {
            clients: RwLock::new(HashMap::new()),
            pending_authorize_bindings: RwLock::new(HashMap::new()),
            cimd_cache: Arc::new(super::cimd_cache::CimdCache::new(
                super::cimd_cache::CimdCacheConfig::default(),
            )),
            jwks_uri_cache: RwLock::new(HashMap::new()),
            client_jwks_uri_cache_ttl: Duration::from_secs(config.client_jwks_uri_cache_ttl_secs),
            pending_codes: RwLock::new(HashMap::new()),
            pending_nonces: RwLock::new(HashMap::new()),
            pending_par_requests: RwLock::new(HashMap::new()),
            pending_device_codes: RwLock::new(HashMap::new()),
            device_code_by_user_code: RwLock::new(HashMap::new()),
            token_db: None,
            policy_client,
            discovery_client,
            issuer_url: config.issuer_url(),
            default_scopes: config.default_scopes.clone(),
            token_ttl: config.token_ttl_seconds,
            refresh_token_ttl: config.refresh_token_ttl_seconds,
            require_pushed_authorization_requests: config.require_pushed_authorization_requests,
            http_client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            verifying_key_bytes,
            user_service: None,
            signing_key: None,
            root_identity_key_store: None,
            authority_hints: config.authority_hints.clone(),
            pending_external_auths: RwLock::new(HashMap::new()),
            oidc_discovery: std::sync::Arc::new(
                super::oidc_discovery::OidcDiscoveryCache::default(),
            ),
            sessions: super::session::SessionStore::default(),
            xrpc_repos: Arc::new(super::xrpc::XrpcRepoStore::new()),
            xrpc_read_slice: config.xrpc_read_slice,
            rsa_encoding_key: None,
            rsa_jwk: None,
            rsa_kid: None,
            dpop_jti_seen: TtlCache::new(Self::DPOP_JTI_MAX_ENTRIES, Self::DPOP_JTI_REAP_BUDGET),
            assertion_jti_seen: TtlCache::new(
                Self::ASSERTION_JTI_MAX_ENTRIES,
                Self::ASSERTION_JTI_REAP_BUDGET,
            ),
            dpop_nonces: RwLock::new(HashMap::new()),
            dpop_clients_seen: RwLock::new(HashMap::new()),
            trusted_issuers: config.trusted_issuers.clone(),
            ca_jwt_key: None,
            device_store: None,
            jwt_key_nbf: chrono::Utc::now().timestamp(),
            jwt_key_exp: chrono::Utc::now().timestamp() + 14 * 86400,
            signing_key_store: None,
            jti_blocklist: None,
            es256_key_store: None,
            ml_dsa_key_store: None,
            audit_sink: None,
            quic_cert_hashes: Vec::new(),
            quic_public_uri: None,
            mesh_pq_verifying_key: None,
            mesh_kem_public: None,
            at9p_identity: None,
            iroh_node_id: None,
            iroh_relays: Vec::new(),
        }
    }

    /// Attach the anonymous device store.
    pub fn with_device_store(mut self, store: Arc<dyn crate::auth::DeviceStore>) -> Self {
        self.device_store = Some(store);
        self
    }

    /// Set JWT signing key validity window for the JWKS endpoint.
    pub fn with_jwt_key_timestamps(mut self, nbf: i64, exp: i64) -> Self {
        self.jwt_key_nbf = nbf;
        self.jwt_key_exp = exp;
        self
    }

    /// Attach the CA JWT signing key for browser WIT issuance (`POST /oauth/wit`).
    pub fn with_ca_jwt_key(mut self, key: ed25519_dalek::SigningKey) -> Self {
        self.ca_jwt_key = Some(Arc::new(key));
        self
    }

    /// Attach the multi-slot signing key store (rotation).
    ///
    /// When set, JWKS serves all slots and WIT issuance uses the active key.
    pub fn with_signing_key_store(mut self, store: Arc<crate::auth::SigningKeyStore>) -> Self {
        self.signing_key_store = Some(store);
        self
    }

    /// Attach the independently persisted root-DID identity rotation set.
    pub fn with_root_identity_key_store(
        mut self,
        store: Arc<crate::auth::SigningKeyStore>,
    ) -> Self {
        self.root_identity_key_store = Some(store);
        self
    }

    /// Attach the ES256 (P-256) key rotation store.
    pub fn with_es256_key_store(mut self, store: Arc<crate::auth::Es256SigningKeyStore>) -> Self {
        self.es256_key_store = Some(store);
        self
    }

    /// Attach the ML-DSA-65 key rotation store.
    pub fn with_ml_dsa_key_store(mut self, store: Arc<crate::auth::MlDsaSigningKeyStore>) -> Self {
        self.ml_dsa_key_store = Some(store);
        self
    }

    /// Attach the S6 grant-path audit sink (B2, #674). Without this, the
    /// grant path fails closed on every request rather than minting tokens
    /// with no audit trail.
    pub fn with_audit_sink(mut self, sink: Arc<dyn crate::mac::audit::AuditSink>) -> Self {
        self.audit_sink = Some(sink);
        self
    }

    /// Set the node's QUIC transport info for DID-doc publication (#185).
    ///
    /// `cert_hashes` is the set of SHA-256 cert DER hashes currently in use
    /// (old + new during rotation). `public_uri` is `https://host:port` that
    /// external peers dial.
    pub fn with_quic_transport(mut self, public_uri: String, cert_hashes: Vec<[u8; 32]>) -> Self {
        self.quic_public_uri = Some(public_uri);
        self.quic_cert_hashes = cert_hashes;
        self
    }

    /// Set the node's iroh transport info for DID-doc publication (#282).
    ///
    /// Call this only when the iroh substrate is actually bound: it makes
    /// `root_did_document` advertise an `IrohTransport` service entry. The
    /// `node_id` is reachability metadata, never a DID verification method.
    /// `relays` may be empty — peers then resolve reachability by node_id alone
    /// via iroh's pkarr/DNS discovery.
    pub fn with_iroh_transport(mut self, node_id: [u8; 32], relays: Vec<String>) -> Self {
        self.iroh_node_id = Some(node_id);
        self.iroh_relays = relays;
        self
    }

    /// Attach the shared JWT ID blocklist (shared with PolicyService).
    ///
    /// When set, `POST /oauth/revoke` on access tokens writes the JTI into
    /// this blocklist so the PolicyService RPC enforcement path also rejects
    /// revoked tokens — closing the gap between HTTP revocation and RPC auth.
    pub fn with_jti_blocklist(
        mut self,
        bl: Arc<hyprstream_rpc::auth::InMemoryJtiBlocklist>,
    ) -> Self {
        self.jti_blocklist = Some(bl);
        self
    }

    /// Return the verifying key to use for JWT bearer token validation.
    ///
    /// Prefers the active slot from the signing key store; falls back to the
    /// legacy `verifying_key_bytes` field (policy-service-issued key).
    pub async fn jwt_bearer_verifying_key(&self) -> Option<ed25519_dalek::VerifyingKey> {
        if let Some(store) = &self.signing_key_store {
            if let Some(bytes) = store.active_verifying_key_bytes().await {
                return ed25519_dalek::VerifyingKey::from_bytes(&bytes).ok();
            }
        }
        ed25519_dalek::VerifyingKey::from_bytes(&self.verifying_key_bytes).ok()
    }

    /// Return the active JWT signing key for token issuance (WIT/ADT).
    ///
    /// Prefers the active slot from the signing key store; falls back to `ca_jwt_key`.
    pub async fn active_jwt_signing_key(&self) -> Option<Arc<ed25519_dalek::SigningKey>> {
        if let Some(store) = &self.signing_key_store {
            if let Some(key) = store.active_key().await {
                return Some(key);
            }
        }
        self.ca_jwt_key.clone()
    }

    /// Active root-DID signing key, falling back to the pre-rotation singleton.
    pub async fn active_root_identity_signing_key(&self) -> Option<Arc<ed25519_dalek::SigningKey>> {
        if let Some(store) = &self.root_identity_key_store {
            if let Some(key) = store.active_key().await {
                return Some(key);
            }
        }
        self.signing_key.clone().map(Arc::new)
    }

    /// Attach a user credential store. Creates a `UserService` backed by the store
    /// for SCIM/RPC access and legacy OAuth handler reads.
    pub fn with_user_store(mut self, store: Arc<dyn UserStore>) -> Self {
        self.user_service = Some(Arc::new(UserService::new(store)));
        self
    }

    /// Attach a pre-built `UserService`. Used when the service is constructed externally
    /// (e.g., for testing or when the store is shared across services).
    pub fn with_user_service(mut self, service: Arc<UserService>) -> Self {
        self.user_service = Some(service);
        self
    }

    /// Get read access to the user store via the UserService.
    /// Returns None if no user store is configured.
    pub fn user_store_reader(&self) -> Option<Arc<dyn UserStore>> {
        self.user_service.as_ref().map(|s| s.store())
    }

    /// Backward-compatible check: returns true if a user store is configured.
    pub fn has_user_store(&self) -> bool {
        self.user_service.is_some()
    }

    /// Attach the signing key for OpenID Federation 1.0 entity configuration signing.
    ///
    /// Also derives and stores the node's mesh ML-DSA-65 verifying key (#157)
    /// from this same Ed25519 key, so the root DID document's `#mesh-pq`
    /// Multikey verification method matches the post-quantum key the mesh signs
    /// with (`derive_mesh_mldsa_key`), and the node's `#mesh-kem` hybrid
    /// keyAgreement public material (S1 / #552, `derive_mesh_kem_recipient`) so
    /// peers can anchor a recipient key that matches what this node decapsulates
    /// with. A `#mesh-kem` derivation failure is fail-closed under Hybrid
    /// policy: startup returns an error rather than publishing a root DID
    /// document with an empty `keyAgreement` relationship. Under Classical
    /// policy, the failure is logged and `mesh_kem_public` stays unset.
    pub fn with_signing_key(
        mut self,
        key: ed25519_dalek::SigningKey,
        policy: hyprstream_rpc::crypto::CryptoPolicy,
    ) -> anyhow::Result<Self> {
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&key);
        self.mesh_pq_verifying_key =
            Some(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk));
        self.at9p_identity = Some(super::did_document::render_at9p_identity(
            &self.issuer_url,
            &key,
            &pq_sk,
        )?);
        self.mesh_kem_public = mesh_kem_public_for_policy(&key, policy, |key| {
            hyprstream_rpc::node_identity::derive_mesh_kem_recipient(key)
        })?;
        self.signing_key = Some(key);
        Ok(self)
    }

    /// Inject a pre-built token store implementation.
    pub fn with_token_store_impl(&mut self, store: Arc<dyn TokenStore>) {
        self.token_db = Some(store);
    }

    /// Persist a refresh token entry to the store.
    pub async fn put_refresh_token(
        &self,
        token: &str,
        entry: &RefreshTokenEntry,
        ttl_secs: u64,
    ) -> anyhow::Result<()> {
        let Some(ref store) = self.token_db else {
            tracing::warn!("Refresh token store not configured — token will not survive restart");
            return Ok(());
        };
        store.put(token, entry, ttl_secs).await
    }

    /// Look up a refresh token. Returns `None` if not found or expired (lazy expiry).
    pub async fn get_refresh_token(
        &self,
        token: &str,
    ) -> anyhow::Result<Option<RefreshTokenEntry>> {
        let Some(ref store) = self.token_db else {
            return Ok(None);
        };
        store.get(token).await
    }

    /// Atomically claim a refresh token for one-time rotation. Only one caller
    /// across all OAuth replicas can receive a given entry.
    pub async fn take_refresh_token(&self, token: &str) -> anyhow::Result<Option<RefreshTokenEntry>> {
        let Some(ref store) = self.token_db else {
            return Ok(None);
        };
        store.take(token).await
    }

    /// Remove a refresh token (revocation / rotation).
    pub async fn delete_refresh_token(&self, token: &str) -> anyhow::Result<()> {
        let Some(ref store) = self.token_db else {
            return Ok(());
        };
        store.delete(token).await
    }

    /// Check a DPoP JTI for replay and record it if new.
    ///
    /// Returns `true` when the JTI is fresh (caller should proceed).
    /// Returns `false` when the JTI has been seen within its validity window (replay).
    /// Expired entries are pruned opportunistically on each call.
    pub fn check_and_record_dpop_jti(&self, jti: &str, iat: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        // Window ends at iat + 120s (±60s skew + 60s buffer); TTL = remainder.
        let ttl_secs = ((iat + 120) - now).max(0) as u64;
        self.dpop_jti_seen
            .insert_if_absent(jti.to_owned(), (), Duration::from_secs(ttl_secs))
    }

    /// Check a client-assertion JTI for replay and record it if new
    /// (#1146 T3.3; RFC 7523 §3 — an AS MUST NOT accept the same `jti`
    /// more than once).
    ///
    /// Keyed by `{client_id}\u{1f}{jti}` so distinct clients cannot
    /// collide on the same identifier. The entry lives until the
    /// assertion's own `exp`, after which a replayed assertion would be
    /// rejected as expired anyway.
    ///
    /// Returns `true` when the JTI is fresh (caller should proceed);
    /// `false` when it was seen within its validity window (replay).
    pub fn check_and_record_assertion_jti(&self, client_id: &str, jti: &str, exp: i64) -> bool {
        let now = chrono::Utc::now().timestamp();
        let ttl_secs = (exp - now).max(0) as u64;
        self.assertion_jti_seen.insert_if_absent(
            format!("{client_id}\u{1f}{jti}"),
            (),
            Duration::from_secs(ttl_secs),
        )
    }

    /// Issue a fresh server-side DPoP nonce (RFC 9449 §8).
    ///
    /// Returns the base64url nonce string that should be placed in the
    /// `DPoP-Nonce` response header. Stored with a 5-minute TTL.
    pub async fn issue_dpop_nonce(&self) -> String {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        let nonce = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes);
        let expiry = chrono::Utc::now().timestamp() + 300; // 5-minute TTL
        self.dpop_nonces.write().await.insert(nonce.clone(), expiry);
        nonce
    }

    /// Validate a DPoP nonce issued by this server. Returns `true` if valid and unexpired.
    pub async fn verify_dpop_nonce(&self, nonce: &str) -> bool {
        let now = chrono::Utc::now().timestamp();
        let store = self.dpop_nonces.read().await;
        store.get(nonce).is_some_and(|&exp| exp > now)
    }

    /// RFC 9449 §8 nonce-enforcement bookkeeping: has this client (`jkt`)
    /// previously been issued a nonce that has not yet expired? If so the
    /// next DPoP proof from this key MUST include a valid nonce.
    pub async fn dpop_client_requires_nonce(&self, jkt: &str) -> bool {
        let now = chrono::Utc::now().timestamp();
        let mut store = self.dpop_clients_seen.write().await;
        store.retain(|_, exp| *exp > now);
        store.contains_key(jkt)
    }

    /// Record that this client (`jkt`) has been issued a server nonce.
    /// Future proofs from this key are required to carry one (sliding 5-min
    /// window per nonce TTL).
    pub async fn mark_dpop_client_nonced(&self, jkt: &str) {
        let expiry = chrono::Utc::now().timestamp() + 300;
        self.dpop_clients_seen
            .write()
            .await
            .insert(jkt.to_owned(), expiry);
    }

    /// Validate the JWT `sub` claim for an atproto-conformant access token
    /// (#1113 rev2 / #1124 split).
    ///
    /// atproto OAuth requires the access token's `sub` to be a real,
    /// resolvable atproto account DID: `did:plc:<opaque>` or host-form
    /// `did:web:<host>[:port]` (NO path component — the atproto DID profile
    /// rejects `did:web:host:users:alice`). A hyprstream-local account
    /// (enrolled by Ed25519 username) has no atproto DID yet; provisioning
    /// is tracked in #1124. Such accounts are REJECTED from the atproto
    /// profile (fail closed), never minted a spec-invalid path-form alias.
    pub fn subject_did(&self, subject: &str) -> Result<String, SubjectDidError> {
        subject_did_for(&self.issuer_url, subject)
    }

    /// Origin-only issuer required by the atproto OAuth profile.
    pub fn atproto_issuer_url(&self) -> String {
        canonical_issuer_origin(&self.issuer_url).unwrap_or_else(|| self.issuer_url.clone())
    }

    /// Select the issuer used at a profile-sensitive mint/redirect boundary.
    pub fn issuer_for_scopes(&self, scopes: &[String]) -> String {
        if atproto_profile_active(scopes) {
            self.atproto_issuer_url()
        } else {
            self.issuer_url.clone()
        }
    }

    /// Check whether an account is eligible for the atproto OAuth profile
    /// and return the account's mapped atproto DID (#1113 r5 / #1124 split).
    ///
    /// This inspects the ACCOUNT/KEY MAPPING, not the subject string:
    /// 1. Rejects at9p-backed accounts (any key with
    ///    [`KeyAlgorithm::HybridEd25519MlDsa65`] → [`SubjectDidError::At9pBackedAccount`]).
    /// 2. Looks up the account's MAPPED atproto DID from the profile's
    ///    `atproto_did` field. If present and valid → returns it (success).
    /// 3. An account without a mapped DID fails closed
    ///    ([`SubjectDidError::NoAtprotoIdentity`]).
    ///
    /// PROVISIONING mappings (creating did:plc, etc.) is #1124 — out of scope.
    /// The LOOKUP + success path exists here so the atproto OAuth profile can
    /// emit the mapped DID as `sub` when an account has one.
    pub async fn check_atproto_account_eligibility(
        &self,
        username: &str,
    ) -> Result<String, SubjectDidError> {
        let user_store = self
            .user_store_reader()
            .ok_or(SubjectDidError::NoAtprotoIdentity)?;

        // Fail closed: a malformed/transient key-store read must never be
        // reinterpreted as "this account has no keys". Doing so can hide a
        // HybridEd25519MlDsa65 (at9p) binding and incorrectly admit a mapped
        // profile to the atproto OAuth profile.
        let pubkeys = user_store.list_pubkeys(username).await.map_err(|error| {
            tracing::warn!(%username, %error, "atproto eligibility key lookup failed closed");
            SubjectDidError::NoAtprotoIdentity
        })?;
        let profile = user_store.get_profile(username).await.ok().flatten();

        evaluate_atproto_eligibility(&pubkeys, profile.as_ref(), &self.issuer_url)
    }

    /// The full set of scopes this AS supports (#1113 rev2 F4).
    /// Extends `default_scopes` (the omitted-scope grant set) with the
    /// atproto transition scopes, which are supported-but-explicit: a client
    /// must request them to activate the strict profile; omitting `scope`
    /// grants only `default_scopes`.
    pub fn server_supported_scopes(&self) -> Vec<String> {
        let mut scopes = self.default_scopes.clone();
        for atproto_scope in &["atproto", "transition:generic"] {
            if !scopes.iter().any(|s| s == *atproto_scope) {
                scopes.push((*atproto_scope).to_owned());
            }
        }
        scopes
    }

    /// Attach an RSA key for RS256 id_token signing (OIDC interop).
    ///
    /// `rsa_der` is the PKCS#8 DER-encoded RSA private key.
    pub fn with_rsa_key(mut self, rsa_der: &[u8]) -> Self {
        // Build encoding key
        self.rsa_encoding_key = Some(jsonwebtoken::EncodingKey::from_rsa_der(rsa_der));

        // Extract public key components for JWKS using jsonwebtoken's DecodingKey
        if let Some(mut jwk) = extract_rsa_jwk_from_der(rsa_der, "") {
            let n = jwk.get("n").and_then(|v| v.as_str()).unwrap_or_default();
            let e = jwk.get("e").and_then(|v| v.as_str()).unwrap_or_default();
            let kid = super::jwks::compute_rsa_kid(n, e);
            if let Some(obj) = jwk.as_object_mut() {
                obj.insert("kid".to_owned(), serde_json::Value::String(kid.clone()));
            }
            self.rsa_kid = Some(kid);
            self.rsa_jwk = Some(jwk);
        }

        self
    }

    /// Spawn a background task that sweeps expired codes every 30 seconds.
    pub fn spawn_code_sweeper(self: &Arc<Self>) {
        let state = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;

                // Sweep expired auth codes
                {
                    let mut codes = state.pending_codes.write().await;
                    codes.retain(|_, code| !code.is_expired());
                }

                // Sweep expired authorize nonces and their request snapshots.
                state.sweep_authorize_sessions(Instant::now()).await;

                // Sweep expired PAR requests (60s TTL)
                {
                    let mut par = state.pending_par_requests.write().await;
                    par.retain(|_, req| !req.is_expired());
                }

                // Sweep expired device codes
                {
                    let mut device_codes = state.pending_device_codes.write().await;
                    let mut user_code_map = state.device_code_by_user_code.write().await;
                    device_codes.retain(|_, dc| {
                        if dc.is_expired() {
                            user_code_map.remove(&dc.user_code);
                            false
                        } else {
                            true
                        }
                    });
                }

                // Sweep expired DPoP JTIs and nonces
                {
                    let now = chrono::Utc::now().timestamp();
                    // dpop_jti_seen is a TtlCache (self-evicting); sweep the
                    // nonce + client-dedup maps only.
                    state.dpop_nonces.write().await.retain(|_, exp| *exp > now);
                    state
                        .dpop_clients_seen
                        .write()
                        .await
                        .retain(|_, exp| *exp > now);
                }

                // Sweep expired external auth flows
                {
                    let now = Instant::now();
                    let mut auths = state.pending_external_auths.write().await;
                    auths.retain(|_, auth| auth.expires_at > now);
                }

                // Sweep expired sessions
                state.sessions.sweep().await;
            }
        });
    }

    async fn sweep_authorize_sessions(&self, now: Instant) {
        let mut nonces = self.pending_nonces.write().await;
        nonces.retain(|_, expiry| *expiry > now);
        let mut bindings = self.pending_authorize_bindings.write().await;
        bindings.retain(|nonce, _| nonces.contains_key(nonce));
    }
}

/// Evaluate atproto account eligibility from the account's stored keys and
/// profile (#1113 r5 / #1124 split). Free-function form so it is unit-testable
/// without constructing an OAuthState or UserStore.
///
/// - at9p-backed account (any key with `HybridEd25519MlDsa65`) → reject.
/// - account with a mapped `atproto_did` on its profile → validate and return.
/// - account without a mapping → `NoAtprotoIdentity` (fail closed, #1124).
pub fn evaluate_atproto_eligibility(
    pubkeys: &[crate::auth::user_store::PubkeyEntry],
    profile: Option<&crate::auth::user_store::UserProfile>,
    issuer_url: &str,
) -> Result<String, SubjectDidError> {
    use crate::auth::user_store::KeyAlgorithm;

    // 1. Reject at9p-backed accounts by inspecting the key algorithm.
    if pubkeys
        .iter()
        .any(|pk| pk.algorithm == KeyAlgorithm::HybridEd25519MlDsa65)
    {
        return Err(SubjectDidError::At9pBackedAccount);
    }

    // 2. Look up the account's MAPPED atproto DID from the profile.
    if let Some(did) = profile.and_then(|p| p.atproto_did.as_deref()) {
        return subject_did_for(issuer_url, did);
    }

    // 3. No mapped atproto DID — fail closed.
    Err(SubjectDidError::NoAtprotoIdentity)
}

/// Validate that a subject is a real, resolvable atproto account DID (#1113
/// rev2 → #1124 split).
///
/// The atproto DID profile (`@atproto/did`) accepts only:
/// - `did:plc:<opaque>` — the registered PLC-directory form.
/// - `did:web:<host>[:port]` — the **host-form only** (NO path component);
///   `did:web:host:users:alice` is explicitly rejected by the atproto DID
///   profile.
///
/// Hyprstream-local accounts (enrolled by Ed25519 username) do NOT yet carry
/// a real atproto DID — provisioning one is tracked in #1124. Such accounts
/// are REJECTED from the atproto profile (fail closed with
/// [`SubjectDidError::NoAtprotoIdentity`]) rather than being minted a
/// spec-invalid path-form did:web. This function performs DID-form
/// validation, not a string-prefix match — an at9p-backed account whose
/// enrolled identifier is a plain username is rejected just like any other
/// non-DID subject.
pub fn subject_did_for(_issuer_url: &str, subject: &str) -> Result<String, SubjectDidError> {
    // did:plc — the only non-`did:web` atproto method. A valid did:plc
    // identifier has exactly 24 chars of base32-lowercase (a-z, 2-7), per
    // the atproto PLC spec. The vendored @atproto/did schema rejects shorter
    // suffixes like `did:plc:abc123xyz` (9 chars; @atproto/did rejects
    // anything shorter than 24 base32 chars).
    if let Some(rest) = subject.strip_prefix("did:plc:") {
        if rest.len() == 24 && rest.chars().all(|c| matches!(c, 'a'..='z' | '2'..='7')) {
            return Ok(subject.to_owned());
        }
        return Err(SubjectDidError::MalformedDid);
    }
    // did:web — host-form only. The atproto DID profile rejects path
    // components (`/`-segments encoded as extra colon groups are allowed by
    // the base did:web spec, but atproto's narrower profile does not). A
    // host-form did:web has exactly one path segment after the method:
    // `did:web:<host>[:port]`. Reject `did:web:<host>:users:...` (path form).
    if let Some(rest) = subject.strip_prefix("did:web:") {
        if rest.is_empty() {
            return Err(SubjectDidError::MalformedDid);
        }
        // atproto host-form: no `/`, no `:`-delimited path beyond host[:port].
        // A path-form did:web (e.g. `did:web:host:users:alice`) carries
        // additional colon segments; reject it.
        if rest.contains('/') {
            return Err(SubjectDidError::PathFormDidWeb);
        }
        let segment_count = rest.split(':').count();
        // host[:port] → 1 or 2 segments. ≥3 implies a path form.
        if segment_count > 2 {
            return Err(SubjectDidError::PathFormDidWeb);
        }
        return Ok(subject.to_owned());
    }
    // Any other did: method (did:at9p, did:key, ...) is NOT eligible.
    if subject.starts_with("did:") {
        return Err(SubjectDidError::UnsupportedMethod);
    }
    // A non-DID subject (local username) has no atproto identity — fail
    // closed. Provisioning real DIDs for hosted accounts is tracked in #1124.
    Err(SubjectDidError::NoAtprotoIdentity)
}

/// Errors raised by [`subject_did_for`] / [`OAuthState::subject_did`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SubjectDidError {
    /// The subject carries a DID method that is not eligible for atproto
    /// OAuth tokens (e.g. `did:at9p`, `did:key`). The token path fails closed.
    #[error("DID method not eligible for atproto OAuth tokens")]
    UnsupportedMethod,
    /// A did:plc/did:web identifier is malformed (empty or illegal body).
    #[error("malformed atproto DID")]
    MalformedDid,
    /// A path-form did:web (e.g. `did:web:host:users:alice`) is rejected by
    /// the atproto DID profile — only host-form did:web is accepted.
    #[error("path-form did:web rejected by atproto DID profile")]
    PathFormDidWeb,
    /// A non-DID subject (local username) has no atproto identity. Real
    /// atproto DID provisioning for hosted accounts is tracked in #1124.
    #[error("account has no atproto identity; provisioning tracked in #1124")]
    NoAtprotoIdentity,
    /// The account is at9p-backed (enrolled with a hybrid PQ key) —
    /// inspected via the account/key mapping, not the subject string.
    /// at9p accounts are not eligible for atproto OAuth tokens.
    #[error("at9p-backed account not eligible for atproto OAuth")]
    At9pBackedAccount,
}

/// The atproto transition scope name. Its presence in a granted scope set
/// activates the strict atproto OAuth profile (#1113 rev2): DPoP-mandatory,
/// DID subject, `token_type: DPoP` response, scope enforcement.
pub const ATPROTO_SCOPE: &str = "atproto";

/// True when the granted scope set activates the atproto strict profile.
pub fn atproto_profile_active(scopes: &[String]) -> bool {
    scopes.iter().any(|s| s == ATPROTO_SCOPE)
}

/// Normalize a single scope token. RFC 6749 §3.3 scope tokens are `1*(%x21 /
/// %x23-5B / %x5D-7E)` — no space, no `"`, no control chars. Returns `None`
/// for tokens that are empty or contain characters outside the allowed set.
pub fn normalize_scope_token(tok: &str) -> Option<&str> {
    if tok.is_empty() {
        return None;
    }
    if tok.bytes().all(|b| b >= 0x21 && b != b'"' && b != 0x7f) {
        Some(tok)
    } else {
        None
    }
}

/// Validate a requested scope set against the server-supported and (optionally)
/// client-declared scopes (#1113 rev2 finding 4).
///
/// Returns the **actual granted set** (the intersection of requested with the
/// server-supported set, intersected with `client_declared` when provided),
/// preserving the server's canonical ordering of the supported scopes.
///
/// - Unknown / undeclared / malformed requested tokens yield
///   [`ScopeError::InvalidScope`] (RFC 6749 §4.1.2.1 / §3.3) so the caller
///   rejects with `invalid_scope`.
/// - When `require_atproto` is true (the atproto profile), the granted set
///   MUST contain [`ATPROTO_SCOPE`]; otherwise [`ScopeError::AtprotoRequired`].
pub fn validate_requested_scopes(
    requested: &[String],
    server_supported: &[String],
    client_declared: Option<&[String]>,
    require_atproto: bool,
) -> Result<Vec<String>, ScopeError> {
    // The set the client is permitted to ask for.
    let allowed: Vec<&str> = match client_declared {
        Some(declared) if !declared.is_empty() => server_supported
            .iter()
            .filter(|s| declared.iter().any(|d| d == *s))
            .map(String::as_str)
            .collect(),
        _ => server_supported.iter().map(String::as_str).collect(),
    };

    let mut granted: Vec<String> = Vec::new();
    for tok in requested {
        let tok = normalize_scope_token(tok).ok_or(ScopeError::InvalidScope)?;
        if !allowed.contains(&tok) {
            return Err(ScopeError::InvalidScope);
        }
        if !granted.iter().any(|g| g == tok) {
            granted.push(tok.to_owned());
        }
    }
    if require_atproto && !granted.iter().any(|s| s == ATPROTO_SCOPE) {
        return Err(ScopeError::AtprotoRequired);
    }
    Ok(granted)
}

/// Errors raised by [`validate_requested_scopes`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ScopeError {
    /// A requested scope token is unknown, undeclared by the client, or
    /// malformed → reject with OAuth `invalid_scope` (RFC 6749 §3.3).
    #[error("invalid or unsupported scope")]
    InvalidScope,
    /// The atproto profile is active but `atproto` is not in the granted set.
    #[error("atproto scope required for this profile")]
    AtprotoRequired,
}

/// Canonicalize an issuer/external URL to an exact origin (scheme://host[:port])
/// with no trailing slash and no path (#1113 rev2 finding 5). Returns `None`
/// when the URL has no scheme or authority.
pub fn canonical_issuer_origin(issuer_url: &str) -> Option<String> {
    let (scheme, rest) = issuer_url.split_once("://")?;
    if scheme.is_empty() {
        return None;
    }
    let authority = rest.split('/').next()?;
    if authority.is_empty() {
        return None;
    }
    Some(format!("{scheme}://{authority}"))
}
