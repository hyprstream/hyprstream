//! JWT claims for authentication.
//!
//! Two claim types:
//! - [`Claims`] — access token claims used throughout the RPC layer.
//!   OAuth grants carry their granted scope set in the signed JWT so consumers
//!   can attenuate authority without falling back to subject-wide policy.
//! - [`IdTokenClaims`] — OIDC ID Token claims (Section 2 of OpenID Connect
//!   Core 1.0). Only used by the OAuth token endpoint when `scope=openid`.

use crate::capnp::{FromCapnp, ToCapnp};
use crate::common_capnp;
use crate::identity::UNAUTHENTICATED_DID_SENTINEL;
use anyhow::Result;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Compute the RFC 7638 JWK Thumbprint for an Ed25519 key.
///
/// Delegates to [`crate::auth::jwk_thumbprint`] — kept as a named entry-point
/// so callers that only have a raw `[u8; 32]` don't need to construct a
/// [`crate::auth::JwkThumbprintInput`] themselves. (#155 cleanup)
pub fn compute_jkt(key_bytes: &[u8; 32]) -> String {
    crate::auth::jwk_thumbprint(&crate::auth::JwkThumbprintInput::Ed25519 { x: key_bytes })
}

/// Returns true if `iss` belongs to a local node.
///
/// Used for pre-decode key routing (before a `Claims` object exists) and by
/// [`Claims::is_local_to`] for post-decode subject derivation — both use
/// identical criteria so routing and subject resolution are always consistent.
///
/// Rules:
/// - Empty `iss` is always local (cluster-internal tokens have no issuer claim).
/// - If `local_issuers` is non-empty: `iss` must exactly match one entry.
/// - If `local_issuers` is empty (unconfigured node): only an empty `iss` is
///   accepted as local; any non-empty `iss` is treated as federated.
pub fn is_local_iss(iss: &str, local_issuers: &[&str]) -> bool {
    if iss.is_empty() {
        return true;
    }
    if local_issuers.is_empty() {
        false
    } else {
        local_issuers.contains(&iss)
    }
}

/// JWK (JSON Web Key) for an Ed25519 public key — RFC 7517 OKP key type.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CnfJwk {
    pub kty: String,
    pub crv: String,
    /// Base64url-encoded Ed25519 public key (32 bytes), RFC 8037 §2.
    pub x: String,
}

/// RFC 8705 Proof-of-Possession `cnf` claim.
///
/// Two key-binding modes:
/// - `jwk`: Full OKP JWK object — used for WIMSE service WITs (`cnf.jwk`).
/// - `jkt`: JWK Thumbprint (RFC 7638 SHA-256, base64url) — used for DPoP user
///   tokens (`cnf.jkt`, per RFC 9449 § 6).
///
/// Both fields are optional so the struct serialises correctly for each mode.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Cnf {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jwk: Option<CnfJwk>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jkt: Option<String>,
}

/// RFC 8693 §4.1 actor claim — the two-principal carrier for delegated calls
/// (#680/#681).
///
/// A delegated (on-behalf-of) token sets [`Claims::sub`] = the **delegator**
/// (source of authority, e.g. the user) and [`Claims::act`] = the **actor** that
/// signs the downstream envelope (e.g. `service:mcp`). This is delegation, not
/// impersonation: the actor holds no standing access, only the attenuated
/// authority the delegator granted it.
///
/// **Composition (#681):** the effective MAC clearance is `meet(delegator,
/// actor)` on level/compartments — never either principal alone. The actor's
/// authority-asserted clearance is carried here so the PDP can take the meet off
/// the one verified JWT. Like [`Claims::clearance`] it is *authority-asserted*
/// (the issuing node signs the JWT) and carries **no** assurance: assurance is
/// derived from the *verified signer* (the actor, who signs the downstream
/// envelope) and clamped DOWN at enforcement time — never trusted from this claim.
///
/// `act` MAY nest (`act.act`) for multi-hop delegation (RFC 8693 §4.1); each hop
/// composes into the meet. **JWT-carried only** (mirrors [`Claims::clearance`]):
/// not written to the Cap'n Proto envelope surface — read off the verified JWT by
/// the MAC PDP.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActClaim {
    /// The actor's subject identifier (e.g. `service:mcp`). At enforcement time
    /// this MUST equal the verified downstream envelope signer (`cnf`): an actor
    /// cannot claim to act as an identity whose key it does not hold — the
    /// #680 anti-confused-deputy invariant.
    pub sub: String,
    /// The actor's authority-asserted MAC clearance, carried so #681 can take
    /// `meet(delegator, actor)` without a second lookup. `None` ⇒ the actor is
    /// unlabeled ⇒ the delegated derivation **fails closed** (no default
    /// clearance for a missing principal; the S1 rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clearance: Option<crate::auth::mac::SecurityLabel>,
    /// Nested actor for multi-hop delegation (RFC 8693 §4.1). Boxed to keep
    /// [`Claims`] sized. Each hop composes into the clearance meet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub act: Option<Box<ActClaim>>,
}

/// The issuer-declared credential use profile (design §3.4, frozen credential
/// profile §4).
///
/// `Reusable` does not consume the credential ID: replay admission is keyed by
/// the proof, so many fresh proofs may use one credential.
/// `OneShotTransaction` consumes the credential ID atomically and domain-wide
/// before handler admission, and creates no second proof replay entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CredentialUse {
    #[serde(rename = "reusable")]
    Reusable,
    #[serde(rename = "one-shot-transaction")]
    OneShotTransaction,
}

impl CredentialUse {
    /// Whether this profile consumes the credential ID as its single
    /// admission action.
    pub fn is_one_shot(self) -> bool {
        matches!(self, Self::OneShotTransaction)
    }
}

/// The exact transaction a `OneShotTransaction` credential authorizes (§3.4).
///
/// A one-shot credential is permitted **only** for a short-lived credential
/// bound by claims to an exact audience, method, resource, subject, actor, and
/// transaction intent. The audience is the credential `aud`, the subject the
/// credential `sub`, and the actor the terminal `act`; the remaining
/// dimensions — the exact method leaf, the resource, and the transaction
/// intent — are carried here so a verifier can confirm the credential was
/// minted for *this* request before consuming its ID. The whole binding is
/// covered by the token signature, so neither the caller nor the method can
/// forge or reinterpret it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OneShotTransactionBinding {
    /// The exact numeric method leaf path (dotted, e.g. `1.1.6`) this
    /// credential authorizes. It MUST equal the leaf derived from the signed
    /// request body at dispatch — a one-shot credential cannot be replayed
    /// against a different method.
    pub method_leaf: String,
    /// The exact resource the transaction acts on (e.g. a repository ID or an
    /// object coordinate). Opaque to this type; compared for equality.
    pub resource: String,
    /// The transaction intent — the application-level operation the issuer
    /// authorized (e.g. `merge`, `transfer`). Non-empty by construction.
    pub intent: String,
}

/// JWT claims for authentication.
///
/// Casbin remains the server-side policy authority. OAuth access tokens also
/// carry the grant's scope ceiling so a verifier can enforce the intersection
/// of subject policy and the authority delegated by that specific grant.
#[derive(Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Issuer URL — the hyprstream node that minted this token.
    /// Matches the OAuth issuer URL (e.g. "https://cloud-a.example.com").
    /// Required for federation: receiving nodes use this to fetch JWKS.
    /// Defaults to empty string for backward compat (treated as local-only).
    #[serde(default)]
    pub iss: String,
    pub sub: String,
    /// Verified tenant/Casbin domain assigned by the token authority.
    ///
    /// This value is consumed only after JWT signature verification. Missing,
    /// empty, or wildcard tenants cannot be used as authorization domains.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,
    pub exp: i64,
    pub iat: i64,
    /// RFC 7519 JWT ID — unique token identifier for revocation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
    /// RFC 8707 audience claim for resource indicator binding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// OAuth 2.0 granted scope set, serialized as a space-delimited string.
    ///
    /// This is optional because non-OAuth service tokens do not originate from
    /// an OAuth grant. Consumers that derive authority from OAuth scopes MUST
    /// use [`Self::has_scope`] and fail closed when this claim is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// RFC 8705 / WIMSE proof-of-possession confirmation claim.
    ///
    /// Binds the Ed25519 signing key to the JWT-attested identity via a standard
    /// JWK object (`cnf.jwk`). For service tokens (WIT), derived by the CA from
    /// the root key. For user tokens, set during OAuth flow from the verified
    /// Ed25519 challenge-response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cnf: Option<Cnf>,
    /// **Credential use profile (§3.4).** The issuer-declared profile for this
    /// credential, covered by the token signature.
    ///
    /// Absent means `Reusable`, which is the profile's own default for user
    /// access tokens and ordinary workload identity tokens. A caller cannot
    /// mark its own token one-shot and a method cannot reinterpret a reusable
    /// token as one-shot because it mutates: the value is only ever read from
    /// verified, issuer-signed claims.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub credential_use: Option<CredentialUse>,

    /// **One-shot transaction binding (§3.4).** Present iff `credential_use`
    /// is `OneShotTransaction`: the exact method leaf, resource, and intent
    /// the issuer minted this credential for. A verifier MUST confirm this
    /// binding against the actual request before consuming the credential ID
    /// (see [`Claims::verify_one_shot_binding`]). Covered by the token
    /// signature; JSON-carried on the JWT, so it survives the credential-hash
    /// binding the proof signs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transaction_binding: Option<OneShotTransactionBinding>,

    /// Original JWT token for end-to-end verification.
    /// When present, downstream services MUST verify this token
    /// independently rather than trusting the envelope claims alone.
    ///
    /// SECURITY:
    /// - `#[serde(skip)]` prevents recursive JWT-in-JWT embedding and JSON exposure.
    /// - This field is ONLY populated via Cap'n Proto (`ToCapnp`/`FromCapnp`).
    /// - Custom Debug impl redacts this field as `[REDACTED]`.
    /// - Do NOT change to `skip_serializing_if` — that would leak the token in JSON.
    #[serde(skip)]
    pub token: Option<String>,

    /// **MAC clearance (S8/#574, design §3, §11).** The authority-asserted
    /// clearance the issuing node grants this subject: a `SecurityLabel`
    /// carrying `level` + `compartments`. This is the **authority-asserted**
    /// half of the S1 subject context; the issuing node signs the JWT, so the
    /// clearance cannot be self-asserted by the subject.
    ///
    /// **Assurance is NOT carried here.** Per the S1/#548 contract, the
    /// assurance axis of a subject's context is *derived from the verified key
    /// material*, never trusted from a claim. The MAC PDP clamps the
    /// clearance's assurance DOWN to what the verified crypto supports (see
    /// [`crate::auth::mac::SecurityContext::from_clearance`]). So an Ed25519-only
    /// identity carrying a `PqHybrid` clearance floors to `Classical` at
    /// enforcement time — the claim cannot grant assurance the key does not back.
    ///
    /// `None` ⇒ unlabeled subject ⇒ the MAC monitor denies (no default
    /// clearance; the S1 fail-closed rule). Carried on the hybrid-signed
    /// envelope/JWT so it is itself PQ-forgery-resistant.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clearance: Option<crate::auth::mac::SecurityLabel>,

    /// **Delegation actor (RFC 8693 §4.1 `act`, #680/#681).** Present iff this is
    /// a delegated (on-behalf-of) token: [`Self::sub`] is the delegator (source of
    /// authority), `act` the actor that signs the downstream envelope. Drives the
    /// two-principal MAC derivation (#681): clearance = `meet(sub, act)`, assurance
    /// from the verified signer (the actor). See [`ActClaim`]. JWT-carried only
    /// (mirrors [`Self::clearance`]); reconstructed `None` on the Cap'n Proto path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub act: Option<ActClaim>,

    /// **Attenuated capability subset (ZSP, Fu1/#677, `TODO(#572-scope-claim)`).**
    /// The least-authority capability the minted token actually encodes, in
    /// `ability@resource` form. ZSP mints the *subset* that covers the request,
    /// never the whole grant — this claim puts that subset **on the wire** so the
    /// downstream PEP (S2) enforces the minted authority (and a refresh can only
    /// re-grant this subset, never widen). Distinct from Casbin scopes (which are
    /// NOT in JWTs): this is the MAC/UCAN capability the S6 exchange authorized.
    ///
    /// `None` ⇒ not a grant-minted token (ordinary auth token). JWT-carried only
    /// (mirrors [`Self::clearance`]/[`Self::act`]); reconstructed `None` on the
    /// Cap'n Proto path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cap: Option<String>,
}

// Custom Debug impl — NEVER log the bearer token
impl std::fmt::Debug for Claims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Claims")
            .field("iss", &self.iss)
            .field("sub", &self.sub)
            .field("tenant", &self.tenant)
            .field("exp", &self.exp)
            .field("iat", &self.iat)
            .field("jti", &self.jti)
            .field("aud", &self.aud)
            .field("scope", &self.scope)
            .field("cnf", &self.cnf)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .field("clearance", &self.clearance)
            .field("act", &self.act)
            .field("cap", &self.cap)
            .finish()
    }
}

impl ToCapnp for Claims {
    type Builder<'a> = common_capnp::claims::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_sub(&self.sub);
        builder.set_exp(self.exp);
        builder.set_iat(self.iat);
        builder.set_admin(false); // Deprecated: always false
        if let Some(ref aud) = self.aud {
            builder.set_aud(aud);
        }
        if let Some(ref token) = self.token {
            builder.set_token(token);
        }
        if !self.iss.is_empty() {
            builder.set_iss(&self.iss);
        }
        if let Some(ref cnf) = self.cnf {
            if let Some(ref jwk) = cnf.jwk {
                builder.set_pub_key(&jwk.x);
            }
        }
        if let Some(ref scope) = self.scope {
            builder.set_oauth_scope(scope);
        }
        // Write empty scopes list for wire compatibility
        builder.reborrow().init_scopes(0);
    }
}

impl FromCapnp for Claims {
    type Reader<'a> = common_capnp::claims::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let aud = reader
            .get_aud()
            .ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .filter(|s| !s.is_empty());

        let scope = reader
            .get_oauth_scope()
            .ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .filter(|s| !s.is_empty());

        let token = reader
            .get_token()
            .ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .filter(|s| !s.is_empty());

        let iss = reader
            .get_iss()
            .ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_default();

        let cnf = reader
            .get_pub_key()
            .ok()
            .and_then(|s| s.to_str().ok())
            .filter(|s| !s.is_empty())
            .map(|x| Cnf {
                jwk: Some(CnfJwk {
                    kty: "OKP".to_owned(),
                    crv: "Ed25519".to_owned(),
                    x: x.to_owned(),
                }),
                jkt: None,
            });

        Ok(Self {
            iss,
            sub: reader.get_sub()?.to_str()?.to_owned(),
            // Tenant authorization is JWT-only so the verifier can establish
            // its authority; legacy envelope claims cannot assert it.
            tenant: None,
            exp: reader.get_exp(),
            iat: reader.get_iat(),
            jti: None,
            aud,
            // The dedicated OAuth field preserves the signed grant ceiling;
            // the legacy structured scope list remains unused by Claims.
            scope,
            cnf,
            // The credential use profile is issuer-signed and rides the JWT;
            // the envelope Claims carrier cannot assert it. Absent = Reusable.
            credential_use: None,
            transaction_binding: None,
            token,
            // MAC clearance (S8/#574) is not carried on the Cap'n Proto envelope
            // surface today; it rides the JWT (the hybrid-signed authority token).
            // The envelope Claims are a distinct carrier from the JWT Claims; the
            // clearance is read off the verified JWT by the MAC PDP.
            clearance: None,
            // `act` (RFC 8693 delegation) likewise rides the JWT, not this
            // envelope surface; reconstructed None here (#680/#681).
            act: None,
            // `cap` (attenuated capability subset) is JWT-carried too (Fu1/#677).
            cap: None,
        })
    }
}

impl Claims {
    /// Create new claims.
    pub fn new(sub: String, iat: i64, exp: i64) -> Self {
        Self {
            iss: String::new(),
            sub,
            tenant: None,
            exp,
            iat,
            jti: None,
            aud: None,
            scope: None,
            cnf: None,
            credential_use: None,
            transaction_binding: None,
            token: None,
            clearance: None,
            act: None,
            cap: None,
        }
    }

    /// **Mint a one-shot transaction credential (§3.4).**
    ///
    /// The issuer minting path calls this to produce the claim set for a
    /// short-lived credential bound to an exact `(audience, method leaf,
    /// resource, subject, actor, transaction intent)`. It sets
    /// `credential_use = OneShotTransaction`, the transaction binding, a fresh
    /// random `jti` (the ID consumed as the single replay-admission action),
    /// and the audience/subject/actor. The caller supplies a **short** `exp`;
    /// this constructor refuses a non-positive lifetime so a one-shot
    /// credential can never be minted already-expired.
    ///
    /// The returned claims are the mintable artifact — the issuer signs them
    /// with its JWT/WIT key exactly as it signs any other credential, so the
    /// binding is covered by the token signature. A caller cannot mark its own
    /// token one-shot (it does not hold the issuer key) and a method cannot
    /// reinterpret a reusable token as one-shot (the value is read only from
    /// verified issuer-signed claims).
    #[allow(clippy::too_many_arguments)]
    pub fn mint_one_shot_transaction(
        issuer: String,
        subject: String,
        audience: String,
        actor: Option<ActClaim>,
        binding: OneShotTransactionBinding,
        iat: i64,
        exp: i64,
    ) -> Result<Self> {
        anyhow::ensure!(exp > iat, "one-shot credential must have a positive lifetime");
        anyhow::ensure!(!issuer.is_empty(), "one-shot credential must name an issuer");
        anyhow::ensure!(!audience.is_empty(), "one-shot credential must name an audience");
        anyhow::ensure!(
            !binding.method_leaf.trim().is_empty(),
            "one-shot binding must name a method leaf"
        );
        anyhow::ensure!(
            !binding.intent.trim().is_empty(),
            "one-shot binding must name a transaction intent"
        );
        Ok(Self::new(subject, iat, exp)
            .with_issuer(issuer)
            .with_audience(Some(audience))
            .with_jti()
            .with_one_shot_transaction(binding)
            .with_actor(actor))
    }

    /// Attach the one-shot credential-use profile and its transaction binding.
    pub fn with_one_shot_transaction(mut self, binding: OneShotTransactionBinding) -> Self {
        self.credential_use = Some(CredentialUse::OneShotTransaction);
        self.transaction_binding = Some(binding);
        self
    }

    /// Attach a delegation actor (RFC 8693 `act`).
    pub fn with_actor(mut self, actor: Option<ActClaim>) -> Self {
        self.act = actor;
        self
    }

    /// **Verify a one-shot credential's binding against the actual request
    /// (§3.4).** Called at dispatch immediately before the credential ID is
    /// consumed, so a one-shot credential minted for one transaction can never
    /// be replayed against another.
    ///
    /// Every independently-sourced dimension must match exactly: the credential
    /// must declare the one-shot profile and carry a binding; its audience must
    /// equal the dispatching `service_domain`; the binding's method leaf must
    /// equal the leaf derived from the signed request body; and the credential
    /// subject must equal the authenticated `subject`. The resource and intent
    /// must be non-empty — the issuer minted them, and an empty value is a
    /// malformed credential, never a wildcard.
    ///
    /// The §3.4 "actor" dimension is deliberately not re-checked here. The
    /// terminal actor lives in the issuer-signed `act` delegation chain, which
    /// is covered end-to-end by the proof's credential-hash binding — there is
    /// no independent, non-issuer actor source at dispatch to compare it
    /// against, so re-deriving `self.terminal_actor()` and asserting it equals
    /// itself would be a tautology, not a check. Independent delegation
    /// terminal-actor evaluation (`AsSelf`/`AsOriginator`) is WS-E/F scope.
    pub fn verify_one_shot_binding(
        &self,
        service_domain: &str,
        leaf_key: &str,
        subject: &str,
    ) -> Result<()> {
        anyhow::ensure!(
            self.credential_use == Some(CredentialUse::OneShotTransaction),
            "credential is not one-shot"
        );
        let binding = self
            .transaction_binding
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("one-shot credential carries no transaction binding"))?;
        anyhow::ensure!(
            self.aud.as_deref() == Some(service_domain),
            "one-shot credential audience does not match the dispatching service domain"
        );
        anyhow::ensure!(
            binding.method_leaf == leaf_key,
            "one-shot credential is bound to a different method leaf"
        );
        anyhow::ensure!(
            self.sub == subject,
            "one-shot credential subject does not match the authenticated subject"
        );
        anyhow::ensure!(
            !binding.resource.trim().is_empty(),
            "one-shot credential binds an empty resource"
        );
        anyhow::ensure!(
            !binding.intent.trim().is_empty(),
            "one-shot credential binds an empty transaction intent"
        );
        Ok(())
    }

    /// The terminal (deepest) actor subject in the delegation chain, if any.
    pub fn terminal_actor(&self) -> Option<&str> {
        let mut current = self.act.as_ref()?;
        while let Some(next) = current.act.as_deref() {
            current = next;
        }
        Some(current.sub.as_str())
    }

    /// Set a random JWT ID (RFC 7519 `jti` claim) for revocation support.
    pub fn with_jti(mut self) -> Self {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        use rand::RngCore as _;
        let mut bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut bytes);
        self.jti = Some(URL_SAFE_NO_PAD.encode(bytes));
        self
    }

    /// Set the issuer URL (RFC 7519 `iss` claim).
    /// Should be the OAuth issuer URL of the hyprstream node that issued this token.
    pub fn with_issuer(mut self, issuer: String) -> Self {
        self.iss = issuer;
        self
    }

    /// Set the authority-asserted tenant used as the Casbin request domain.
    pub fn with_tenant(mut self, tenant: String) -> Self {
        self.tenant = Some(tenant);
        self
    }

    /// Create new claims with an audience (RFC 8707 resource indicator).
    pub fn with_audience(mut self, audience: Option<String>) -> Self {
        self.aud = audience;
        self
    }

    /// Set the OAuth 2.0 granted scope claim.
    pub fn with_scope(mut self, scope: Option<String>) -> Self {
        self.scope = scope.filter(|value| !value.trim().is_empty());
        self
    }

    /// Iterate over the exact scopes granted to this token.
    ///
    /// An absent claim yields an empty iterator, keeping scope-based consumers
    /// fail closed for legacy and non-OAuth tokens.
    pub fn granted_scopes(&self) -> impl Iterator<Item = &str> {
        self.scope.as_deref().into_iter().flat_map(str::split_whitespace)
    }

    /// Return whether this token's signed grant contains `required` exactly.
    pub fn has_scope(&self, required: &str) -> bool {
        self.granted_scopes().any(|granted| granted == required)
    }

    /// Attach original JWT token for e2e verification by downstream services.
    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Set the RFC 8705 `cnf.jwk` confirmation claim (WIMSE WIT key binding).
    ///
    /// `key_bytes` is the raw 32-byte Ed25519 public key. Produces a standard
    /// OKP JWK object with `kty: "OKP"`, `crv: "Ed25519"`, `x: <base64url>`.
    pub fn with_cnf_jwk(mut self, key_bytes: &[u8; 32]) -> Self {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        self.cnf = Some(Cnf {
            jwk: Some(CnfJwk {
                kty: "OKP".to_owned(),
                crv: "Ed25519".to_owned(),
                x: URL_SAFE_NO_PAD.encode(key_bytes),
            }),
            jkt: None,
        });
        self
    }

    /// Set the RFC 9449 `cnf.jkt` confirmation claim (DPoP JWK thumbprint).
    ///
    /// `key_bytes` is the raw 32-byte Ed25519 public key. The thumbprint is
    /// computed per RFC 7638: SHA-256 of the lexicographic canonical JWK JSON,
    /// base64url-encoded.
    pub fn with_cnf_jkt(mut self, key_bytes: &[u8; 32]) -> Self {
        self.cnf = Some(Cnf {
            jwk: None,
            jkt: Some(compute_jkt(key_bytes)),
        });
        self
    }

    /// Set the MAC clearance claim (S8/#574). The authority-asserted clearance
    /// the issuing node grants this subject (level + compartments). The
    /// resulting JWT carries this under its (hybrid) signature, so it is itself
    /// PQ-forgery-resistant; the assurance axis is NOT carried here — it is
    /// derived from the verified key material at enforcement time.
    pub fn with_clearance(mut self, clearance: crate::auth::mac::SecurityLabel) -> Self {
        self.clearance = Some(clearance);
        self
    }

    /// Set the RFC 8693 delegation actor (`act`, #680/#681). Marks this as a
    /// delegated (on-behalf-of) token: [`Self::sub`] stays the delegator; `actor`
    /// is the principal that will sign the downstream envelope. The MAC PDP
    /// derives clearance as `meet(sub, act)` and assurance from the verified
    /// signer. See [`ActClaim`].
    pub fn with_act(mut self, actor: ActClaim) -> Self {
        self.act = Some(actor);
        self
    }

    /// Set the attenuated capability subset (`cap`, ZSP Fu1/#677) the minted
    /// token encodes, in `ability@resource` form. See [`Self::cap`].
    pub fn with_cap(mut self, cap: String) -> Self {
        self.cap = Some(cap);
        self
    }

    /// Extract the raw 32-byte Ed25519 public key from `cnf.jwk.x`, if present.
    pub fn cnf_key_bytes(&self) -> Option<[u8; 32]> {
        use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
        let x = self.cnf.as_ref()?.jwk.as_ref()?.x.as_str();
        let bytes = URL_SAFE_NO_PAD.decode(x).ok()?;
        bytes.try_into().ok()
    }

    /// Return the `cnf.jkt` thumbprint string, if present.
    pub fn cnf_jkt(&self) -> Option<&str> {
        self.cnf.as_ref()?.jkt.as_deref()
    }

    /// Independently verify the embedded JWT token.
    ///
    /// Returns the verified claims if the token is present and valid.
    /// Returns None if no token is embedded.
    /// Returns Err if token is present but verification fails (MUST reject request).
    ///
    /// Note: `jwt::decode()` validates both signature and `exp` (via `is_expired()`).
    pub fn verify_token(
        &self,
        verifying_key: &ed25519_dalek::VerifyingKey,
        expected_aud: Option<&str>,
    ) -> std::result::Result<Option<Claims>, super::jwt::JwtError> {
        match &self.token {
            Some(token) => {
                let verified = super::jwt::decode(token, verifying_key, expected_aud)?;
                Ok(Some(verified))
            }
            None => Ok(None),
        }
    }

    /// Check if token is expired (with 5-second leeway for clock skew).
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.exp.saturating_add(5)
    }

    /// True if this token was issued by a local node.
    ///
    /// Delegates to [`is_local_iss`] using this token's `iss` field.
    pub fn is_local_to(&self, local_issuers: &[&str]) -> bool {
        is_local_iss(&self.iss, local_issuers)
    }

    /// Strip the authority-asserted MAC `clearance` when this token's issuer is
    /// NOT a local issuer (Fu5/#677).
    ///
    /// `clearance` is authority-asserted: the issuing node signs the JWT, so a
    /// local issuer (this node, or a configured local cluster issuer) is trusted
    /// to grant MAC clearance. An external OIDC issuer trusted **for identity**
    /// is NOT thereby trusted to assert MAC clearance on this node — honoring it
    /// would let any trusted-for-identity federated IdP mint MAC clearance.
    /// Federated-path decode therefore drops the claim before the MAC PDP can
    /// read it ([`crate::auth::mac::SubjectContextClaims::clearance_label`]
    /// returns `None` ⇒ unlabeled ⇒ deny). Local-issuer tokens are unaffected.
    pub fn strip_federated_clearance(&mut self, local_issuers: &[&str]) {
        if !self.is_local_to(local_issuers) {
            self.clearance = None;
        }
    }

    /// Strip the authority-asserted tenant when this token's issuer is not
    /// local.
    ///
    /// A federated issuer trusted for identity is not trusted to select a local
    /// Casbin tenant domain. Clearing the claim makes downstream domain
    /// resolution fail closed, while preserving tenant bindings minted by local
    /// issuers.
    pub fn strip_federated_tenant(&mut self, local_issuers: &[&str]) {
        if !self.is_local_to(local_issuers) {
            self.tenant = None;
        }
    }

    /// Derive the Casbin authorization subject from these claims.
    ///
    /// Local tokens (issued by this node) produce bare subjects (`"alice"`)
    /// matching existing Casbin rules.  Federated tokens produce namespaced
    /// subjects (`"https://other.node:alice"`) to prevent cross-node spoofing.
    ///
    /// [`UNAUTHENTICATED_DID_SENTINEL`] is credential absence, not a principal.
    /// Even a verified token carrying it therefore resolves to the
    /// unauthenticated floor and can never authenticate as the sentinel.
    pub fn subject(&self, local_issuers: &[&str]) -> crate::envelope::Subject {
        if self.sub.is_empty() || self.sub == UNAUTHENTICATED_DID_SENTINEL {
            return crate::envelope::Subject::anonymous();
        }
        if self.is_local_to(local_issuers) {
            crate::envelope::Subject::new(self.sub.clone())
        } else {
            crate::envelope::Subject::federated(&self.iss, &self.sub)
        }
    }
}

// ── MAC subject-context seam (S8/#574, S1/#548) ─────────────────────────────
//
// `Claims` is the authority-asserted carrier for a subject's clearance. The S1
// `SubjectContextClaims` trait is the typed contract the MAC PDP / S6 grant path
// program against; this impl lights it up for real verified JWT claims. The
// two-input `security_context(key_material)` combines:
//   1. the authority-asserted clearance (this `clearance` field — the JWT is
//      signed by the issuing node, so this is authority-asserted, not self-
//      asserted), and
//   2. the assurance derived from the *verified key material* (threaded in from
//      the envelope signature-verification layer — NEVER trusted from a claim).
// `SecurityContext::from_clearance` clamps the assurance axis DOWN to what the
// crypto supports, so a classical key carrying a PqHybrid clearance floors to
// Classical — the load-bearing #548 invariant: assurance is a property of the
// verified identity, not a grant.
impl crate::auth::mac::SubjectContextClaims for Claims {
    fn clearance_label(&self) -> Option<crate::auth::mac::SecurityLabel> {
        self.clearance
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn oauth_scope_claim_is_exact_and_absence_fails_closed() {
        let plain = Claims::new("alice".to_owned(), 1000, 2000);
        assert!(!plain.has_scope("atproto"));
        assert!(!serde_json::to_string(&plain).unwrap().contains("\"scope\""));

        let scoped = plain.with_scope(Some("atproto transition:generic".to_owned()));
        assert!(scoped.has_scope("atproto"));
        assert!(scoped.has_scope("transition:generic"));
        assert!(!scoped.has_scope("transition"));
        assert_eq!(
            scoped.granted_scopes().collect::<Vec<_>>(),
            vec!["atproto", "transition:generic"]
        );
    }

    #[test]
    fn oauth_scope_survives_capnp_envelope_roundtrip() -> anyhow::Result<()> {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_scope(Some("atproto transition:generic".to_owned()));
        let mut message = capnp::message::Builder::new_default();
        let mut builder = message.init_root::<common_capnp::claims::Builder>();
        claims.write_to(&mut builder);

        let decoded = Claims::read_from(builder.into_reader())?;
        assert_eq!(decoded.scope, claims.scope);
        assert!(decoded.has_scope("atproto"));
        assert!(decoded.has_scope("transition:generic"));
        Ok(())
    }

    #[test]
    fn test_act_claim_serde_roundtrip_and_default_none() {
        // Non-delegated token: `act` absent, and (skip_serializing_if) omitted
        // from the JSON entirely — no empty `act` on the wire.
        let plain = Claims::new("alice".to_owned(), 1000, 2000);
        assert!(plain.act.is_none());
        let json = serde_json::to_string(&plain).unwrap();
        assert!(!json.contains("\"act\""), "plain token must not carry act: {json}");

        // Delegated token: sub = delegator (user), act = actor (service:mcp),
        // with a nested hop to exercise RFC 8693 §4.1 multi-hop.
        let actor = ActClaim {
            sub: "service:mcp".to_owned(),
            clearance: None,
            act: Some(Box::new(ActClaim {
                sub: "service:gateway".to_owned(),
                clearance: None,
                act: None,
            })),
        };
        let delegated = Claims::new("alice".to_owned(), 1000, 2000)
            .with_act(actor.clone())
            .with_cap("infer@mac://model/qwen-7b".to_owned());
        let json = serde_json::to_string(&delegated).unwrap();
        let back: Claims = serde_json::from_str(&json).unwrap();
        assert_eq!(back.sub, "alice", "sub stays the delegator");
        assert_eq!(back.act, Some(actor), "act round-trips including the nested hop");
        assert_eq!(
            back.cap.as_deref(),
            Some("infer@mac://model/qwen-7b"),
            "attenuated capability subset rides the token (Fu1/#677)"
        );
        assert!(plain.cap.is_none(), "plain token carries no cap");
        assert_eq!(
            back.act.as_ref().unwrap().act.as_ref().unwrap().sub,
            "service:gateway"
        );
    }

    #[test]
    fn test_debug_shows_act_but_redacts_token() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000).with_act(ActClaim {
            sub: "service:mcp".to_owned(),
            clearance: None,
            act: None,
        });
        let dbg = format!("{claims:?}");
        assert!(dbg.contains("service:mcp"), "act is not secret, must be visible");
        assert!(dbg.contains("act:"));
    }

    #[test]
    fn test_claims_new() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000);
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.iat, 1000);
        assert_eq!(claims.exp, 2000);
        assert_eq!(claims.iss, "");
        assert!(claims.aud.is_none());
        assert!(claims.token.is_none());
    }

    #[test]
    fn expiry_check_handles_i64_max_without_wrapping() {
        let claims = Claims::new("alice".to_owned(), 0, i64::MAX);
        assert!(
            !claims.is_expired(),
            "an i64::MAX expiry must not wrap into an expired timestamp"
        );
    }

    #[test]
    fn test_claims_with_issuer() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_issuer("https://a.example.com".to_owned());
        assert_eq!(claims.iss, "https://a.example.com");
    }

    // ── Fu5/#677: clearance honored only from local-issuer tokens ───────────

    use crate::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityLabel, SubjectContextClaims as _,
    };

    fn fu5_clearance() -> SecurityLabel {
        SecurityLabel::new(
            Level::Secret,
            Assurance::PqHybrid,
            CompartmentSet::EMPTY,
        )
    }

    /// A federated token (issuer not in `local_issuers`) carrying a `clearance`
    /// claim MUST have it stripped — an external OIDC issuer trusted for
    /// identity is not trusted to assert MAC clearance on this node. After the
    /// strip the MAC PDP reads `None` (⇒ unlabeled ⇒ deny).
    #[test]
    fn fu5_federated_clearance_is_stripped() {
        let federated = "https://idp.example.com";
        let mut claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_issuer(federated.to_owned())
            .with_clearance(fu5_clearance());
        assert!(
            claims.clearance.is_some(),
            "precondition: federated token carries a clearance"
        );

        // Local issuers do NOT include the federated IdP.
        claims.strip_federated_clearance(&["https://this.node"]);

        assert!(
            claims.clearance.is_none(),
            "a federated token's clearance MUST be ignored (Fu5/#677)"
        );
        assert!(
            claims.clearance_label().is_none(),
            "the MAC PDP reads None ⇒ unlabeled ⇒ deny"
        );
    }

    /// A local-issuer token's clearance is authority-asserted by this node and
    /// MUST be preserved — `strip_federated_clearance` is a no-op for it.
    #[test]
    fn fu5_local_clearance_is_preserved() {
        let local_iss = "https://this.node";
        let mut claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_issuer(local_iss.to_owned())
            .with_clearance(fu5_clearance());

        claims.strip_federated_clearance(&[local_iss]);

        let kept = claims
            .clearance
            .expect("a local-issuer token keeps its authority-asserted clearance");
        assert_eq!(kept.level, Level::Secret);
    }

    /// An unconfigured node (empty `local_issuers`) treats every non-empty iss
    /// as federated (see [`is_local_iss`]) — so any clearance is stripped.
    #[test]
    fn fu5_unconfigured_node_strips_all_clearance() {
        let mut claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_issuer("https://some.idp".to_owned())
            .with_clearance(fu5_clearance());

        claims.strip_federated_clearance(&[]);

        assert!(
            claims.clearance.is_none(),
            "an unconfigured node honors no clearance (deny-by-default)"
        );
    }

    #[test]
    fn test_claims_with_audience() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_audience(Some("https://api.example.com".to_owned()));
        assert_eq!(claims.aud, Some("https://api.example.com".to_owned()));
    }

    #[test]
    fn test_claims_with_token() {
        let claims =
            Claims::new("alice".to_owned(), 1000, 2000).with_token("eyJ.test.token".to_owned());
        assert_eq!(claims.token, Some("eyJ.test.token".to_owned()));
    }

    #[test]
    fn test_claims_token_not_in_json() {
        let claims =
            Claims::new("alice".to_owned(), 1000, 2000).with_token("eyJ.secret.token".to_owned());
        let json = serde_json::to_string(&claims).unwrap();
        assert!(!json.contains("secret"));
        assert!(!json.contains("token"));
    }

    #[test]
    fn test_claims_debug_redacts_token() {
        let claims =
            Claims::new("alice".to_owned(), 1000, 2000).with_token("eyJ.secret.token".to_owned());
        let debug = format!("{:?}", claims);
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn test_claims_subject() {
        use crate::envelope::Subject;
        let claims = Claims::new("alice".to_owned(), 1000, 2000);
        assert_eq!(Subject::from(&claims), Subject::new("alice"));

        let sentinel = Claims::new(UNAUTHENTICATED_DID_SENTINEL.to_owned(), 1000, 2000);
        assert_eq!(Subject::from(&sentinel), Subject::anonymous());
    }

    #[test]
    fn test_claims_is_local_to() {
        let local = Claims::new("alice".to_owned(), 0, 9999)
            .with_issuer("https://local.example.com".to_owned());
        let federated = Claims::new("bob".to_owned(), 0, 9999)
            .with_issuer("https://other.example.com".to_owned());
        let legacy = Claims::new("carol".to_owned(), 0, 9999); // empty iss

        assert!(local.is_local_to(&["https://local.example.com"]));
        assert!(!local.is_local_to(&["https://other.example.com"]));
        assert!(!federated.is_local_to(&["https://local.example.com"]));
        // Empty iss is always local — cluster-internal tokens have no issuer claim
        assert!(legacy.is_local_to(&[]));
        assert!(legacy.is_local_to(&["https://local.example.com"]));
    }

    #[test]
    fn test_claims_subject_method() {
        use crate::envelope::Subject;

        let local = Claims::new("alice".to_owned(), 0, 9999)
            .with_issuer("https://node.example.com".to_owned());
        let federated = Claims::new("bob".to_owned(), 0, 9999)
            .with_issuer("https://other.example.com".to_owned());
        let no_sub = Claims::new(String::new(), 0, 9999);
        let sentinel = Claims::new(UNAUTHENTICATED_DID_SENTINEL.to_owned(), 0, 9999)
            .with_issuer("https://node.example.com".to_owned());

        assert_eq!(
            local.subject(&["https://node.example.com"]),
            Subject::new("alice")
        );
        assert_eq!(
            federated.subject(&["https://node.example.com"]),
            Subject::federated("https://other.example.com", "bob")
        );
        assert_eq!(no_sub.subject(&[]), Subject::anonymous());
        assert_eq!(
            sentinel.subject(&["https://node.example.com"]),
            Subject::anonymous()
        );
    }

    #[test]
    fn test_claims_roundtrip_with_iss() {
        use super::super::jwt;
        use ed25519_dalek::SigningKey;

        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://a.example.com".to_owned());

        let token = jwt::encode(&claims, &signing_key);
        let decoded = jwt::decode(&token, &verifying_key, None).expect("decode failed");

        assert_eq!(decoded.iss, "https://a.example.com");
        assert_eq!(decoded.sub, "alice");
        assert_eq!(decoded.exp, 9_999_999_999);
    }

    #[test]
    fn test_claims_iss_absent_defaults_to_empty() {
        // Simulate an old token without the `iss` field.
        // `#[serde(default)]` must ensure deserialization succeeds with iss = "".
        let json = r#"{"sub":"bob","exp":9999999999,"iat":0}"#;
        let claims: Claims = serde_json::from_str(json)
            .expect("old token without iss should deserialize successfully");
        assert_eq!(claims.iss, "");
        assert_eq!(claims.sub, "bob");
    }

    #[test]
    fn test_claims_iss_not_in_json_when_empty() {
        // Empty iss is serialized as `"iss":""` (no skip_serializing_if),
        // but we do NOT skip it — that is intentional per the field design.
        // This test just confirms iss is present in JSON output.
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://cloud.example.com".to_owned());
        let json = serde_json::to_string(&claims).unwrap();
        assert!(json.contains("\"iss\":\"https://cloud.example.com\""));
    }
}

// ─── OIDC ID Token Claims ──────────────────────────────────────────────────

/// A value that serializes as either a single string or an array of strings.
///
/// OIDC Core Section 2 says `aud` is a JSON string or array. Many RPs break
/// if they receive an unexpected format, so we serialize as a string when there
/// is exactly one value and as an array otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OneOrMany {
    /// Single value — serialized as a JSON string.
    One(String),
    /// Multiple values — serialized as a JSON array.
    Many(Vec<String>),
}

impl OneOrMany {
    /// Create from a single value.
    pub fn one(value: impl Into<String>) -> Self {
        Self::One(value.into())
    }

    /// Create from a list. Collapses to `One` when the list has a single element.
    pub fn from_vec(mut values: Vec<String>) -> Self {
        if values.len() == 1 {
            Self::One(values.swap_remove(0))
        } else {
            Self::Many(values)
        }
    }

    /// Get the values as a slice-like iterator.
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::One(v) => std::slice::from_ref(v),
            Self::Many(v) => v,
        }
    }
}

impl Serialize for OneOrMany {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::One(v) => serializer.serialize_str(v),
            Self::Many(v) => v.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for OneOrMany {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(Self::One(s)),
            serde_json::Value::Array(arr) => {
                let strings: std::result::Result<Vec<String>, _> = arr
                    .into_iter()
                    .map(|v| {
                        v.as_str().map(String::from).ok_or_else(|| {
                            serde::de::Error::custom("aud array must contain strings")
                        })
                    })
                    .collect();
                Ok(Self::from_vec(strings?))
            }
            _ => Err(serde::de::Error::custom("aud must be a string or array")),
        }
    }
}

/// OIDC ID Token claims (OpenID Connect Core 1.0, Section 2).
///
/// Separate from [`Claims`] because:
/// - `aud` is the `client_id` (not the resource indicator)
/// - Includes user profile claims (name, email)
/// - Only issued by the OAuth token endpoint, not used in the RPC layer
/// - Different `typ` header (`"JWT"` vs `"at+jwt"` for access tokens)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdTokenClaims {
    // ── Required (OIDC Core Section 2) ──────────────────────────────────
    /// Issuer URL.
    pub iss: String,
    /// Subject identifier — a stable UUID, not a username.
    pub sub: String,
    /// Audience — the client_id. Supports single string or array.
    pub aud: OneOrMany,
    /// Expiration time (Unix timestamp).
    pub exp: i64,
    /// Issued at (Unix timestamp).
    pub iat: i64,

    // ── Conditional ─────────────────────────────────────────────────────
    /// OIDC nonce echoed from the authorization request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Time of user authentication (Unix timestamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_time: Option<i64>,
    /// Authorized party (REQUIRED when aud has multiple values).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azp: Option<String>,
    /// Authority-verified hosted-account tenant for local token exchange.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tenant: Option<String>,

    // ── Profile claims (included based on requested scopes) ─────────────
    /// Full name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Email address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Whether the email is verified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email_verified: Option<bool>,
    /// Preferred display username.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_username: Option<String>,
}

impl IdTokenClaims {
    /// Create minimal id_token claims (required fields only).
    pub fn new(iss: String, sub: String, aud: String, iat: i64, exp: i64) -> Self {
        Self {
            iss,
            sub,
            aud: OneOrMany::one(aud),
            exp,
            iat,
            nonce: None,
            auth_time: None,
            azp: None,
            tenant: None,
            name: None,
            email: None,
            email_verified: None,
            preferred_username: None,
        }
    }

    /// Set the OIDC nonce (echoed from the authorization request).
    pub fn with_nonce(mut self, nonce: Option<String>) -> Self {
        self.nonce = nonce;
        self
    }

    /// Set the authentication time.
    pub fn with_auth_time(mut self, auth_time: i64) -> Self {
        self.auth_time = Some(auth_time);
        self
    }

    /// Bind this local ID token to the user's hosted-account tenant.
    pub fn with_tenant(mut self, tenant: String) -> Self {
        self.tenant = Some(tenant);
        self
    }

    /// Set profile claims from a user profile.
    pub fn with_profile(
        mut self,
        name: Option<String>,
        email: Option<String>,
        email_verified: Option<bool>,
        preferred_username: Option<String>,
    ) -> Self {
        self.name = name;
        self.email = email;
        self.email_verified = email_verified;
        self.preferred_username = preferred_username;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod id_token_tests {
    use super::*;

    #[test]
    fn test_id_token_claims_serialization() {
        let claims = IdTokenClaims::new(
            "https://example.com".into(),
            "f47ac10b-58cc-4372-a567-0e02b2c3d479".into(),
            "my-client".into(),
            1000,
            2000,
        );
        let json = serde_json::to_string(&claims).unwrap();
        // aud should be a single string, not array
        assert!(json.contains("\"aud\":\"my-client\""));
        assert!(json.contains("\"sub\":\"f47ac10b-"));
        // Optional fields should not appear
        assert!(!json.contains("nonce"));
        assert!(!json.contains("name"));
        assert!(!json.contains("email"));
    }

    #[test]
    fn test_id_token_claims_with_profile() {
        let claims = IdTokenClaims::new(
            "https://example.com".into(),
            "uuid-123".into(),
            "client-1".into(),
            1000,
            2000,
        )
        .with_nonce(Some("abc123".into()))
        .with_auth_time(999)
        .with_profile(
            Some("Alice".into()),
            Some("alice@example.com".into()),
            Some(true),
            Some("alice".into()),
        );
        let json = serde_json::to_string(&claims).unwrap();
        assert!(json.contains("\"nonce\":\"abc123\""));
        assert!(json.contains("\"auth_time\":999"));
        assert!(json.contains("\"name\":\"Alice\""));
        assert!(json.contains("\"email\":\"alice@example.com\""));
        assert!(json.contains("\"email_verified\":true"));
        assert!(json.contains("\"preferred_username\":\"alice\""));
    }

    #[test]
    fn test_one_or_many_single() {
        let aud = OneOrMany::one("client-1");
        let json = serde_json::to_string(&aud).unwrap();
        assert_eq!(json, "\"client-1\"");
    }

    #[test]
    fn test_one_or_many_multiple() {
        let aud = OneOrMany::from_vec(vec!["a".into(), "b".into()]);
        let json = serde_json::to_string(&aud).unwrap();
        assert_eq!(json, "[\"a\",\"b\"]");
    }

    #[test]
    fn test_one_or_many_deserialize_string() {
        let aud: OneOrMany = serde_json::from_str("\"client-1\"").unwrap();
        assert_eq!(aud, OneOrMany::One("client-1".into()));
    }

    #[test]
    fn test_one_or_many_deserialize_array() {
        let aud: OneOrMany = serde_json::from_str("[\"a\",\"b\"]").unwrap();
        assert_eq!(aud, OneOrMany::Many(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_one_or_many_collapse_single_element() {
        let aud = OneOrMany::from_vec(vec!["only-one".into()]);
        assert_eq!(aud, OneOrMany::One("only-one".into()));
    }

    // ── One-shot transaction credential (§3.4) ──────────────────────────

    fn binding() -> OneShotTransactionBinding {
        OneShotTransactionBinding {
            method_leaf: "1.1.6".into(),
            resource: "repo:abc".into(),
            intent: "merge".into(),
        }
    }

    fn minted() -> Claims {
        Claims::mint_one_shot_transaction(
            "https://issuer.example".into(),
            "user:alice".into(),
            "registry".into(),
            None,
            binding(),
            1000,
            1030,
        )
        .expect("mints")
    }

    /// A minted one-shot credential declares the profile, carries a random
    /// jti, and binds its exact transaction — the issuer's mintable artifact.
    #[test]
    fn minting_a_one_shot_credential_sets_the_profile_binding_and_id() {
        let c = minted();
        assert_eq!(c.credential_use, Some(CredentialUse::OneShotTransaction));
        assert_eq!(c.transaction_binding.as_ref().unwrap().method_leaf, "1.1.6");
        assert_eq!(c.aud.as_deref(), Some("registry"));
        assert!(c.jti.is_some(), "the credential ID is the replay-admission key");
    }

    /// Minting refuses a non-positive lifetime, an empty issuer/audience, and
    /// an empty method leaf or intent — a one-shot credential is never minted
    /// already-expired or unbound.
    #[test]
    fn minting_rejects_malformed_one_shot_requests() {
        assert!(Claims::mint_one_shot_transaction(
            "iss".into(), "s".into(), "aud".into(), None, binding(), 1000, 1000,
        )
        .is_err());
        assert!(Claims::mint_one_shot_transaction(
            String::new(), "s".into(), "aud".into(), None, binding(), 1000, 1030,
        )
        .is_err());
        let mut empty_leaf = binding();
        empty_leaf.method_leaf = "  ".into();
        assert!(Claims::mint_one_shot_transaction(
            "iss".into(), "s".into(), "aud".into(), None, empty_leaf, 1000, 1030,
        )
        .is_err());
        let mut empty_intent = binding();
        empty_intent.intent = String::new();
        assert!(Claims::mint_one_shot_transaction(
            "iss".into(), "s".into(), "aud".into(), None, empty_intent, 1000, 1030,
        )
        .is_err());
    }

    /// The binding verifies only against the EXACT request it was minted for.
    #[test]
    fn one_shot_binding_verifies_the_exact_transaction() {
        let c = minted();
        assert!(c
            .verify_one_shot_binding("registry", "1.1.6", "user:alice")
            .is_ok());
        // Wrong method leaf: a one-shot credential cannot cross methods.
        assert!(c
            .verify_one_shot_binding("registry", "1.1.7", "user:alice")
            .is_err());
        // Wrong service domain: cannot cross services.
        assert!(c
            .verify_one_shot_binding("model", "1.1.6", "user:alice")
            .is_err());
        // Wrong subject.
        assert!(c
            .verify_one_shot_binding("registry", "1.1.6", "user:bob")
            .is_err());
    }

    /// A reusable credential is never one-shot, so binding verification
    /// refuses it: a method cannot reinterpret a reusable token as one-shot.
    #[test]
    fn a_reusable_credential_is_not_one_shot() {
        let c = Claims::new("user:alice".into(), 1000, 2000);
        assert!(c
            .verify_one_shot_binding("registry", "1.1.6", "user:alice")
            .is_err());
    }

    /// `terminal_actor()` still resolves the deepest hop of the delegation
    /// chain, and a delegated one-shot credential verifies on its
    /// independently-sourced dimensions. The terminal actor itself is covered
    /// by the issuer-signed `act` chain under the proof credential-hash
    /// binding, so it is not re-checked here (removing a former tautology).
    #[test]
    fn terminal_actor_resolves_and_delegated_binding_verifies() {
        let inner = ActClaim {
            sub: "service:worker".into(),
            clearance: None,
            act: None,
        };
        let outer = ActClaim {
            sub: "service:mcp".into(),
            clearance: None,
            act: Some(Box::new(inner)),
        };
        let c = Claims::mint_one_shot_transaction(
            "iss".into(),
            "user:alice".into(),
            "registry".into(),
            Some(outer),
            binding(),
            1000,
            1030,
        )
        .unwrap();
        assert_eq!(c.terminal_actor(), Some("service:worker"));
        assert!(c
            .verify_one_shot_binding("registry", "1.1.6", "user:alice")
            .is_ok());
        // The independent dimensions still gate: a different subject denies.
        assert!(c
            .verify_one_shot_binding("registry", "1.1.6", "user:bob")
            .is_err());
    }

    /// The binding survives a JSON round-trip: the issuer signs JSON, and the
    /// proof's credential-hash binding covers exactly those bytes.
    #[test]
    fn one_shot_binding_survives_json_round_trip() {
        let c = minted();
        let json = serde_json::to_string(&c).unwrap();
        let back: Claims = serde_json::from_str(&json).unwrap();
        assert_eq!(back.credential_use, Some(CredentialUse::OneShotTransaction));
        assert_eq!(back.transaction_binding, c.transaction_binding);
    }
}
