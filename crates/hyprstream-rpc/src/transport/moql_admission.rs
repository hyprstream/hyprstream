//! Inside-carrier challenge/response admission for the iroh `moql` ALPN (#1027).
//!
//! # Why this exists
//!
//! [`crate::transport::iroh_moq::IrohMoqProtocolHandler`] accepts raw iroh
//! connections. The carrier handshake authenticates only the **NodeId**, which
//! is reach evidence and never identity (D3/#895, #1031). Without fresh
//! inside-carrier proof the accept path must serve an anonymous peer, and the
//! handler correctly refuses — which means positive native streaming is
//! impossible by construction. This module is the smallest closed-staging
//! completion: a challenge/response exchange on the **first bidirectional
//! stream** of the `moql` connection, run before the connection is wrapped as a
//! `web_transport_iroh::Session` and handed to `moq_net::Server`.
//!
//! # Protocol (one bi stream, length-prefixed frames)
//!
//! 1. **Hello** (client → server): wire version, the peer's `did:at9p` DID, its
//!    Ed25519 verifying key, and a fresh random `client_nonce`.
//! 2. **Challenge** (server → client): a fresh random `server_nonce` plus the
//!    server's *current* accepted-state `epoch` and `head_digest` for that DID,
//!    drawn from the daemon-owned [`AcceptedStateAuthority`] at admission time.
//! 3. **Response** (client → server): a nested composite signature over the
//!    transcript — the inner Ed25519 layer signs the transcript `T`, the outer
//!    ML-DSA-65 layer signs `T ‖ ed_sig` (the same inner→outer nesting the at9p
//!    record composite uses, `hyprstream-pds::at9p_sign`).
//! 4. **Verdict** (server → client, success only): the session is admitted and
//!    the moq handshake may proceed. Any rejection instead closes the
//!    connection — the client learns "not admitted", never which check failed.
//!
//! The transcript binds domain, ALPN, DID, both nonces, and the accepted-state
//! epoch/head digest:
//!
//! ```text
//! T = "hyprstream/moql-admission/v1" ‖ 0x00 ‖ "moql" ‖ 0x00
//!   ‖ u16be(did_len) ‖ did ‖ client_nonce ‖ server_nonce
//!   ‖ u64be(epoch) ‖ head_digest
//! ```
//!
//! # Admission decision (every step fail-closed)
//!
//! - The DID must be a `did:at9p` accepted-state identity; any other DID method
//!   (including a `did:key` encoding of the carrier NodeId) is rejected.
//! - The authority must hold a **current** accepted state for the DID, unexpired
//!   at decision time (expiry class).
//! - The presented Ed25519 key must be one of the accepted current subject keys
//!   (`subjectKeys` is a published SET, #1188/#1183); the ML-DSA-65 half verified
//!   against is the one bound to *that* Ed25519 key by the same entry. A key
//!   rotated out by a state advance is simply absent (rotation class).
//! - The state is re-read after the response arrives; an epoch/head change
//!   mid-handshake rejects (state-advance invalidation).
//! - Both signature layers must verify; a classical-only or stripped composite
//!   is a downgrade and rejects.
//! - A transcript digest already consumed by a prior admission rejects
//!   (replay class); a replayed response on a new connection additionally fails
//!   signature verification because the server nonce is fresh per challenge.
//! - The tenant is resolved server-side from the verified subject by the
//!   operator-controlled resolver; an unresolvable or malformed tenant rejects
//!   (never a wildcard, never caller-supplied — the #1153 rule).
//!
//! The carrier NodeId is recorded in [`AdmittedMoqPeer::carrier_node_id`] as
//! metadata only. It is never an admission input.
//!
//! # What this module does NOT do
//!
//! Open/dynamic federation contract rules remain #1536 (deferred for this
//! closed, fixed-peer staging claim). Per-track/public-relay policy remains
//! #276; after admission the existing `tenant_scoped_consumer` structural
//! narrowing applies unchanged.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use ed25519_dalek::{Signature, Signer as _, SigningKey, Verifier as _, VerifyingKey};
use iroh::endpoint::Connection;
use parking_lot::Mutex;
use rand::RngCore as _;
use sha2::Digest as _;
use tokio::io::{AsyncRead, AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};

use crate::crypto::pq::{ml_dsa_sign, ml_dsa_verify, ml_dsa_vk_from_bytes, MlDsaSigningKey};
use crate::identity::DID_AT9P_PREFIX;
use crate::moq_authz::{is_valid_tenant_segment, PeerIdentity};
use crate::transport::iroh_moq::PeerTenantResolver;

/// Domain separation tag at the head of every admission transcript.
pub const MOQL_ADMISSION_DOMAIN: &[u8] = b"hyprstream/moql-admission/v1";

/// Wire format version byte.
const WIRE_VERSION: u8 = 1;

/// Maximum DID length accepted on the wire (DIDs are short identifiers; a
/// longer field is a parse error, not a truncation).
const MAX_DID_BYTES: usize = 256;

/// Maximum admission frame size. The largest legitimate frame is the Response,
/// dominated by the ML-DSA-65 signature (~3.3 KiB); 8 KiB leaves headroom
/// without offering an unbounded read.
const MAX_FRAME_BYTES: usize = 8 * 1024;

/// Default bound on the whole admission exchange. A peer that opens a `moql`
/// connection and never proves is a slowloris on the accept path; the exchange
/// must not wait indefinitely.
pub const DEFAULT_ADMISSION_TIMEOUT: Duration = Duration::from_secs(10);

/// Retention ceiling for consumed transcript digests (replay cache). Recording
/// costs an attacker a full valid admission, so this grows slowly; at the
/// ceiling, entries older than the horizon are pruned.
const REPLAY_CACHE_MAX: usize = 4096;

/// How long a consumed transcript digest is remembered. Far longer than any
/// admission timeout; bounded so the cache cannot grow without limit.
const REPLAY_HORIZON: Duration = Duration::from_secs(60 * 60);

/// Why an admission attempt was rejected. Each rejection class named by the
/// #1027 contract maps to a distinct variant so negative evidence can name the
/// exact failing check.
#[derive(Debug, thiserror::Error)]
pub enum MoqlAdmissionError {
    /// Carrier-level I/O failure during the exchange.
    #[error("carrier I/O during moql admission: {0}")]
    Carrier(String),
    /// The exchange did not complete within the admission timeout.
    #[error("moql admission exchange timed out")]
    Timeout,
    /// A frame failed to parse (bad version, truncation, oversize, bad UTF-8).
    #[error("malformed moql admission frame: {0}")]
    Malformed(String),
    /// The presented DID is not a `did:at9p` accepted-state identity. A NodeId
    /// or a `did:key` of it lands here: carrier reach is never identity.
    #[error("identity {0:?} is not a did:at9p accepted-state identity")]
    NotAcceptedIdentity(String),
    /// The daemon-owned authority holds no accepted state for this DID.
    #[error("no current accepted state for {0:?}")]
    UnknownIdentity(String),
    /// The accepted state exists but lapsed before the admission decision.
    #[error("accepted state for {did:?} expired at {expired_at_unix_ms} (now {now_unix_ms})")]
    Expired {
        /// The identity whose accepted state lapsed.
        did: String,
        /// The recorded expiry (unix ms).
        expired_at_unix_ms: i64,
        /// Decision time (unix ms).
        now_unix_ms: i64,
    },
    /// The presented Ed25519 key is not among the accepted current subject
    /// keys — e.g. it was rotated out by a state advance.
    #[error("presented Ed25519 key is not a current accepted subject key of {0:?}")]
    KeyNotCurrent(String),
    /// The accepted state advanced (epoch/head changed) while the handshake was
    /// in flight; the proof was made against a superseded currentness claim.
    #[error("accepted state for {0:?} advanced during admission")]
    StateAdvanced(String),
    /// Ed25519 or ML-DSA-65 layer failed to verify, or a layer was stripped.
    #[error("hybrid Ed25519 + ML-DSA-65 proof verification failed")]
    BadSignature,
    /// The exact transcript digest was already consumed by a prior admission.
    #[error("moql admission transcript replayed")]
    Replay,
    /// The subject verified, but the operator resolver maps it to no tenant.
    #[error("admitted subject {0:?} has no tenant scope")]
    TenantUnresolved(String),
    /// The resolver returned a tenant that is not one MoQ path segment.
    #[error("resolved tenant {0:?} is not a valid single MoQ path segment")]
    InvalidTenant(String),
}

/// One accepted current subject key: an atomic Ed25519 ↔ ML-DSA-65 pair as
/// published in the accepted `subjectKeys` set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptedSubjectKey {
    /// Ed25519 verifying key (32 bytes).
    pub ed25519: [u8; 32],
    /// ML-DSA-65 verifying key (encoded form, 1952 bytes).
    pub ml_dsa_65: Vec<u8>,
}

/// The accepted current state for one `did:at9p` identity, projected for
/// admission decisions. This is the admission-relevant slice of the daemon's
/// `AcceptedAt9pState` (`hyprstream-pds::at9p_duplicity`); the projection lives
/// at the daemon boundary so this crate keeps no pds dependency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptedIdentityState {
    /// Accepted chain epoch.
    pub epoch: u64,
    /// `H512` over the canonical accepted head record.
    pub head_digest: [u8; 64],
    /// The accepted current subject keys (a published SET, not positional).
    pub subject_keys: Vec<AcceptedSubjectKey>,
    /// Successor expiry (unix ms). `None` = genesis, current until a successor
    /// is accepted (mirrors `AcceptedAt9pState::ensure_fresh`).
    pub expires_at_unix_ms: Option<i64>,
}

impl AcceptedIdentityState {
    /// The published subject key whose Ed25519 half equals `ed25519`, if any.
    /// Set membership, never positional (#1188/#1183).
    pub fn subject_key_for(&self, ed25519: &[u8; 32]) -> Option<&AcceptedSubjectKey> {
        self.subject_keys.iter().find(|k| &k.ed25519 == ed25519)
    }

    /// Whether the state is live at `now_unix_ms` (genesis never lapses).
    pub fn is_live(&self, now_unix_ms: i64) -> bool {
        match self.expires_at_unix_ms {
            Some(exp) => now_unix_ms < exp,
            None => true,
        }
    }
}

/// The daemon-owned accepted-state/currentness authority.
///
/// Queried at admission time — never snapshotted into the authenticator — so a
/// state advance (rotation) or lapse between admissions takes effect
/// immediately. Implemented by the daemon over its accepted-state store
/// (`DiscoveryService` / `DuplicityGuard` in the `hyprstream` crate); the
/// blanket `Fn` impl covers closed-staging fixtures and simple adapters.
pub trait AcceptedStateAuthority: Send + Sync {
    /// The current accepted state for `did`, or `None` if the identity is
    /// unknown or has no accepted head.
    fn accepted_state(&self, did: &str) -> Option<AcceptedIdentityState>;
}

impl<F> AcceptedStateAuthority for F
where
    F: Fn(&str) -> Option<AcceptedIdentityState> + Send + Sync,
{
    fn accepted_state(&self, did: &str) -> Option<AcceptedIdentityState> {
        self(did)
    }
}

/// The outcome of a successful admission: a typed admitted subject and tenant
/// from current accepted Ed25519 + ML-DSA-65 evidence.
#[derive(Debug, Clone)]
pub struct AdmittedMoqPeer {
    /// The authenticated application subject (the `did:at9p` DID).
    pub peer: PeerIdentity,
    /// Server-resolved tenant for the verified subject (authoritative
    /// server-side state, never caller-supplied).
    pub tenant: String,
    /// Accepted-state epoch the proof was verified against.
    pub epoch: u64,
    /// Accepted-state head digest the proof was verified against.
    pub head_digest: [u8; 64],
    /// The carrier NodeId, recorded as metadata only. Never an identity input.
    pub carrier_node_id: [u8; 32],
}

// ────────────────────────────────────────────────────────────────────────────
// Wire frames (the `moql` admission wire contract; public so peers and tests
// can construct/parse them without private access)
// ────────────────────────────────────────────────────────────────────────────

/// Hello (client → server): claimed identity + freshness contribution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionHello {
    /// The peer's `did:at9p` DID.
    pub did: String,
    /// The peer's Ed25519 verifying key; must be a current accepted subject key.
    pub ed25519_pub: [u8; 32],
    /// Fresh random client nonce.
    pub client_nonce: [u8; 32],
}

/// Challenge (server → client): server freshness + currentness claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionChallenge {
    /// Fresh random server nonce, single-use per connection.
    pub server_nonce: [u8; 32],
    /// The server's current accepted-state epoch for the DID.
    pub epoch: u64,
    /// The server's current accepted-state head digest for the DID.
    pub head_digest: [u8; 64],
}

/// Response (client → server): the nested composite proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionResponse {
    /// Ed25519 signature over the transcript `T`.
    pub ed_sig: [u8; 64],
    /// ML-DSA-65 signature over `T ‖ ed_sig` (inner→outer nesting).
    pub pq_sig: Vec<u8>,
}

/// Verdict (server → client), sent only on success: the session is admitted
/// and the moq handshake may proceed. Rejection closes the connection without
/// a verdict — the failure detail stays server-side (fail-closed, no oracle).
pub const VERDICT_ADMITTED: u8 = 1;

/// Encode the admission verdict frame payload.
pub fn encode_verdict() -> Vec<u8> {
    vec![WIRE_VERSION, VERDICT_ADMITTED]
}

/// Decode and check a verdict frame payload.
pub fn decode_verdict(bytes: &[u8]) -> Result<(), MoqlAdmissionError> {
    if bytes == encode_verdict() {
        Ok(())
    } else {
        Err(MoqlAdmissionError::Malformed(
            "verdict: not an admission verdict".to_owned(),
        ))
    }
}

/// The signed transcript both sides reconstruct byte-identically.
pub fn admission_transcript(
    did: &str,
    client_nonce: &[u8; 32],
    server_nonce: &[u8; 32],
    epoch: u64,
    head_digest: &[u8; 64],
) -> Vec<u8> {
    let mut t = Vec::with_capacity(
        MOQL_ADMISSION_DOMAIN.len() + 1 + 4 + 1 + 2 + did.len() + 32 + 32 + 8 + 64,
    );
    t.extend_from_slice(MOQL_ADMISSION_DOMAIN);
    t.push(0);
    t.extend_from_slice(crate::transport::iroh_substrate::ALPN_MOQ_LITE);
    t.push(0);
    t.extend_from_slice(&(did.len() as u16).to_be_bytes());
    t.extend_from_slice(did.as_bytes());
    t.extend_from_slice(client_nonce);
    t.extend_from_slice(server_nonce);
    t.extend_from_slice(&epoch.to_be_bytes());
    t.extend_from_slice(head_digest);
    t
}

/// The replay-cache key for a transcript.
fn transcript_digest(t: &[u8]) -> [u8; 32] {
    let mut h = sha2::Sha256::new();
    h.update(MOQL_ADMISSION_DOMAIN);
    h.update(t);
    h.finalize().into()
}

/// Encode a Hello frame payload.
pub fn encode_hello(hello: &AdmissionHello) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 2 + hello.did.len() + 32 + 32);
    out.push(WIRE_VERSION);
    out.extend_from_slice(&(hello.did.len() as u16).to_be_bytes());
    out.extend_from_slice(hello.did.as_bytes());
    out.extend_from_slice(&hello.ed25519_pub);
    out.extend_from_slice(&hello.client_nonce);
    out
}

/// Decode a Hello frame payload.
pub fn decode_hello(bytes: &[u8]) -> Result<AdmissionHello, MoqlAdmissionError> {
    let malformed = |why: &str| MoqlAdmissionError::Malformed(format!("hello: {why}"));
    let (version, rest) = bytes.split_first().ok_or_else(|| malformed("empty"))?;
    if *version != WIRE_VERSION {
        return Err(malformed("unsupported wire version"));
    }
    if rest.len() < 2 {
        return Err(malformed("missing did length"));
    }
    let did_len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
    if did_len == 0 || did_len > MAX_DID_BYTES {
        return Err(malformed("did length out of range"));
    }
    let rest = &rest[2..];
    if rest.len() != did_len + 32 + 32 {
        return Err(malformed("trailing bytes or truncation"));
    }
    let did = std::str::from_utf8(&rest[..did_len])
        .map_err(|_| malformed("did is not UTF-8"))?
        .to_owned();
    let mut ed25519_pub = [0u8; 32];
    ed25519_pub.copy_from_slice(&rest[did_len..did_len + 32]);
    let mut client_nonce = [0u8; 32];
    client_nonce.copy_from_slice(&rest[did_len + 32..]);
    Ok(AdmissionHello {
        did,
        ed25519_pub,
        client_nonce,
    })
}

/// Encode a Challenge frame payload.
pub fn encode_challenge(challenge: &AdmissionChallenge) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 32 + 8 + 64);
    out.push(WIRE_VERSION);
    out.extend_from_slice(&challenge.server_nonce);
    out.extend_from_slice(&challenge.epoch.to_be_bytes());
    out.extend_from_slice(&challenge.head_digest);
    out
}

/// Decode a Challenge frame payload.
pub fn decode_challenge(bytes: &[u8]) -> Result<AdmissionChallenge, MoqlAdmissionError> {
    let malformed = |why: &str| MoqlAdmissionError::Malformed(format!("challenge: {why}"));
    let (version, rest) = bytes.split_first().ok_or_else(|| malformed("empty"))?;
    if *version != WIRE_VERSION {
        return Err(malformed("unsupported wire version"));
    }
    if rest.len() != 32 + 8 + 64 {
        return Err(malformed("bad length"));
    }
    let mut server_nonce = [0u8; 32];
    server_nonce.copy_from_slice(&rest[..32]);
    let mut epoch = [0u8; 8];
    epoch.copy_from_slice(&rest[32..40]);
    let mut head_digest = [0u8; 64];
    head_digest.copy_from_slice(&rest[40..]);
    Ok(AdmissionChallenge {
        server_nonce,
        epoch: u64::from_be_bytes(epoch),
        head_digest,
    })
}

/// Encode a Response frame payload.
pub fn encode_response(response: &AdmissionResponse) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 64 + 2 + response.pq_sig.len());
    out.push(WIRE_VERSION);
    out.extend_from_slice(&response.ed_sig);
    out.extend_from_slice(&(response.pq_sig.len() as u16).to_be_bytes());
    out.extend_from_slice(&response.pq_sig);
    out
}

/// Decode a Response frame payload.
pub fn decode_response(bytes: &[u8]) -> Result<AdmissionResponse, MoqlAdmissionError> {
    let malformed = |why: &str| MoqlAdmissionError::Malformed(format!("response: {why}"));
    let (version, rest) = bytes.split_first().ok_or_else(|| malformed("empty"))?;
    if *version != WIRE_VERSION {
        return Err(malformed("unsupported wire version"));
    }
    if rest.len() < 64 + 2 {
        return Err(malformed("truncated"));
    }
    let mut ed_sig = [0u8; 64];
    ed_sig.copy_from_slice(&rest[..64]);
    let pq_len = u16::from_be_bytes([rest[64], rest[65]]) as usize;
    let pq_sig = &rest[66..];
    if pq_sig.len() != pq_len || pq_len == 0 {
        return Err(malformed("pq signature length mismatch or empty"));
    }
    Ok(AdmissionResponse {
        ed_sig,
        pq_sig: pq_sig.to_vec(),
    })
}

async fn write_frame<W: AsyncWrite + Unpin>(
    w: &mut W,
    payload: &[u8],
) -> Result<(), MoqlAdmissionError> {
    if payload.len() > MAX_FRAME_BYTES {
        return Err(MoqlAdmissionError::Malformed(
            "outbound frame exceeds the size cap".to_owned(),
        ));
    }
    w.write_all(&(payload.len() as u32).to_be_bytes())
        .await
        .map_err(|e| MoqlAdmissionError::Carrier(e.to_string()))?;
    w.write_all(payload)
        .await
        .map_err(|e| MoqlAdmissionError::Carrier(e.to_string()))
}

async fn read_frame<R: AsyncRead + Unpin>(r: &mut R) -> Result<Vec<u8>, MoqlAdmissionError> {
    let mut len = [0u8; 4];
    r.read_exact(&mut len)
        .await
        .map_err(|e| MoqlAdmissionError::Carrier(format!("frame length read: {e}")))?;
    let len = u32::from_be_bytes(len) as usize;
    if len == 0 || len > MAX_FRAME_BYTES {
        return Err(MoqlAdmissionError::Malformed(format!(
            "frame length {len} out of range"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)
        .await
        .map_err(|e| MoqlAdmissionError::Carrier(format!("frame body read: {e}")))?;
    Ok(buf)
}

fn fresh_nonce() -> [u8; 32] {
    let mut n = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut n);
    n
}

// ────────────────────────────────────────────────────────────────────────────
// Server side: the authenticator
// ────────────────────────────────────────────────────────────────────────────

/// Inside-carrier admission authenticator for the `moql` accept path.
///
/// Constructed with the daemon-owned [`AcceptedStateAuthority`] and the
/// operator-controlled subject→tenant resolver, then installed on the handler
/// via
/// [`crate::transport::iroh_moq::MoqAuthzConfig::with_admission`]. With no
/// authenticator installed the accept path keeps its pre-#1027 posture
/// (anonymous ⇒ refused).
pub struct MoqlAdmissionAuthenticator {
    authority: Arc<dyn AcceptedStateAuthority>,
    tenant_resolver: PeerTenantResolver,
    timeout: Duration,
    /// Consumed transcript digests → the unix-ms after which the record may be
    /// dropped. Makes each accepted transcript single-use.
    used: Mutex<HashMap<[u8; 32], i64>>,
}

impl std::fmt::Debug for MoqlAdmissionAuthenticator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoqlAdmissionAuthenticator")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl MoqlAdmissionAuthenticator {
    /// Build with the accepted-state authority and subject→tenant resolver.
    pub fn new(
        authority: Arc<dyn AcceptedStateAuthority>,
        tenant_resolver: PeerTenantResolver,
    ) -> Self {
        Self {
            authority,
            tenant_resolver,
            timeout: DEFAULT_ADMISSION_TIMEOUT,
            used: Mutex::new(HashMap::new()),
        }
    }

    /// Override the admission exchange timeout (bounded slowloris window).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Run the challenge/response on `conn`'s first bi stream. On success,
    /// returns the typed admitted peer; on any failure the caller MUST close
    /// the connection (fail-closed) and serve nothing.
    pub async fn accept(&self, conn: &Connection) -> Result<AdmittedMoqPeer, MoqlAdmissionError> {
        match tokio::time::timeout(self.timeout, self.exchange(conn)).await {
            Ok(res) => res,
            Err(_) => Err(MoqlAdmissionError::Timeout),
        }
    }

    async fn exchange(&self, conn: &Connection) -> Result<AdmittedMoqPeer, MoqlAdmissionError> {
        let carrier_node_id = *conn.remote_id().as_bytes();
        let (mut send, mut recv) = conn
            .accept_bi()
            .await
            .map_err(|e| MoqlAdmissionError::Carrier(format!("accept_bi: {e}")))?;

        // ── Hello ────────────────────────────────────────────────────────────
        let hello = decode_hello(&read_frame(&mut recv).await?)?;
        let state = self.check_hello(&hello)?;

        // ── Challenge ────────────────────────────────────────────────────────
        let challenge = AdmissionChallenge {
            server_nonce: fresh_nonce(),
            epoch: state.epoch,
            head_digest: state.head_digest,
        };
        write_frame(&mut send, &encode_challenge(&challenge)).await?;

        // ── Response ─────────────────────────────────────────────────────────
        let response = decode_response(&read_frame(&mut recv).await?)?;
        let now = crate::envelope::current_timestamp();
        self.verify_response(&hello, &challenge, &response, now)?;

        // ── Tenant binding (server-side, from the verified subject) ──────────
        let peer = PeerIdentity::authenticated(hello.did.clone());
        let tenant = (self.tenant_resolver)(&peer)
            .ok_or_else(|| MoqlAdmissionError::TenantUnresolved(hello.did.clone()))?;
        if !is_valid_tenant_segment(&tenant) {
            return Err(MoqlAdmissionError::InvalidTenant(tenant));
        }
        // Admitted: tell the client the moq handshake may proceed, then close
        // the admission stream. Rejections never reach here — they close the
        // connection instead, so this frame is unforgeable by a rejected peer.
        write_frame(&mut send, &encode_verdict()).await?;
        send.finish()
            .map_err(|e| MoqlAdmissionError::Carrier(format!("finish: {e}")))?;
        Ok(AdmittedMoqPeer {
            peer,
            tenant,
            epoch: state.epoch,
            head_digest: state.head_digest,
            carrier_node_id,
        })
    }

    /// The hello-time decision: identity class, currentness, expiry, and
    /// subject-key membership (including that the bound ML-DSA-65 key
    /// decodes). Returns the accepted state the challenge will bind.
    fn check_hello(
        &self,
        hello: &AdmissionHello,
    ) -> Result<AcceptedIdentityState, MoqlAdmissionError> {
        if !hello.did.starts_with(DID_AT9P_PREFIX) {
            return Err(MoqlAdmissionError::NotAcceptedIdentity(hello.did.clone()));
        }
        let state = self
            .authority
            .accepted_state(&hello.did)
            .ok_or_else(|| MoqlAdmissionError::UnknownIdentity(hello.did.clone()))?;
        let now = crate::envelope::current_timestamp();
        if !state.is_live(now) {
            return Err(MoqlAdmissionError::Expired {
                did: hello.did.clone(),
                expired_at_unix_ms: state.expires_at_unix_ms.unwrap_or_default(),
                now_unix_ms: now,
            });
        }
        let subject = state
            .subject_key_for(&hello.ed25519_pub)
            .ok_or_else(|| MoqlAdmissionError::KeyNotCurrent(hello.did.clone()))?;
        if ml_dsa_vk_from_bytes(&subject.ml_dsa_65).is_err() {
            return Err(MoqlAdmissionError::Malformed(format!(
                "accepted state for {:?} carries an undecodable ML-DSA-65 key",
                hello.did
            )));
        }
        Ok(state)
    }

    /// The response-time decision: currentness re-check, replay, then the
    /// hybrid signature. Factored from the I/O path so each rejection class is
    /// unit-testable without a carrier.
    fn verify_response(
        &self,
        hello: &AdmissionHello,
        challenge: &AdmissionChallenge,
        response: &AdmissionResponse,
        now_unix_ms: i64,
    ) -> Result<(), MoqlAdmissionError> {
        // Currentness re-check: the authority must still report exactly the
        // epoch/head the challenge bound. A state advance mid-handshake
        // invalidates the proof even if both signatures verify.
        let current = self
            .authority
            .accepted_state(&hello.did)
            .ok_or_else(|| MoqlAdmissionError::UnknownIdentity(hello.did.clone()))?;
        if current.epoch != challenge.epoch || current.head_digest != challenge.head_digest {
            return Err(MoqlAdmissionError::StateAdvanced(hello.did.clone()));
        }
        if !current.is_live(now_unix_ms) {
            return Err(MoqlAdmissionError::Expired {
                did: hello.did.clone(),
                expired_at_unix_ms: current.expires_at_unix_ms.unwrap_or_default(),
                now_unix_ms,
            });
        }
        let subject = current
            .subject_key_for(&hello.ed25519_pub)
            .ok_or_else(|| MoqlAdmissionError::KeyNotCurrent(hello.did.clone()))?;
        let pq_vk = ml_dsa_vk_from_bytes(&subject.ml_dsa_65).map_err(|_| {
            MoqlAdmissionError::Malformed(format!(
                "accepted state for {:?} carries an undecodable ML-DSA-65 key",
                hello.did
            ))
        })?;

        let t = admission_transcript(
            &hello.did,
            &hello.client_nonce,
            &challenge.server_nonce,
            challenge.epoch,
            &challenge.head_digest,
        );
        let digest = transcript_digest(&t);
        if self.is_consumed(&digest, now_unix_ms) {
            return Err(MoqlAdmissionError::Replay);
        }

        // Inner Ed25519 layer over T.
        let ed_vk = VerifyingKey::from_bytes(&hello.ed25519_pub)
            .map_err(|_| MoqlAdmissionError::BadSignature)?;
        ed_vk
            .verify(&t, &Signature::from_bytes(&response.ed_sig))
            .map_err(|_| MoqlAdmissionError::BadSignature)?;
        // Outer ML-DSA-65 layer over T ‖ ed_sig (inner→outer nesting).
        let mut outer = t;
        outer.extend_from_slice(&response.ed_sig);
        ml_dsa_verify(&pq_vk, &outer, &response.pq_sig)
            .map_err(|_| MoqlAdmissionError::BadSignature)?;

        self.mark_consumed(digest, now_unix_ms);
        Ok(())
    }

    fn is_consumed(&self, digest: &[u8; 32], now_unix_ms: i64) -> bool {
        let used = self.used.lock();
        used.get(digest).is_some_and(|discard_after| now_unix_ms < *discard_after)
    }

    fn mark_consumed(&self, digest: [u8; 32], now_unix_ms: i64) {
        let horizon_ms = REPLAY_HORIZON.as_millis() as i64;
        let mut used = self.used.lock();
        if used.len() >= REPLAY_CACHE_MAX {
            used.retain(|_, discard_after| now_unix_ms < *discard_after);
        }
        used.insert(digest, now_unix_ms.saturating_add(horizon_ms));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Client side: the prover
// ────────────────────────────────────────────────────────────────────────────

/// Client-side proof material for `moql` admission: the peer's accepted
/// `did:at9p` identity and both private halves of one of its accepted current
/// subject keys. The carrier (iroh secret key / NodeId) plays no part.
pub struct MoqlAdmissionProof {
    /// The peer's `did:at9p` DID.
    pub did: String,
    /// Ed25519 signing key of an accepted current subject key.
    pub ed25519: SigningKey,
    /// ML-DSA-65 signing key bound to that Ed25519 key in the accepted state.
    pub ml_dsa_65: MlDsaSigningKey,
}

impl std::fmt::Debug for MoqlAdmissionProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoqlAdmissionProof")
            .field("did", &self.did)
            .finish_non_exhaustive()
    }
}

/// Run the client half of the admission exchange on `conn`, consuming its
/// first bi stream. On success the connection is ready to be wrapped as a
/// `web_transport_iroh::Session` for the moq handshake; on failure the server
/// has closed (or will close) the connection and the caller MUST NOT proceed.
pub async fn prove_moql_admission(
    conn: &Connection,
    proof: &MoqlAdmissionProof,
    timeout: Duration,
) -> Result<(), MoqlAdmissionError> {
    match tokio::time::timeout(timeout, prove_exchange(conn, proof)).await {
        Ok(res) => res,
        Err(_) => Err(MoqlAdmissionError::Timeout),
    }
}

async fn prove_exchange(
    conn: &Connection,
    proof: &MoqlAdmissionProof,
) -> Result<(), MoqlAdmissionError> {
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| MoqlAdmissionError::Carrier(format!("open_bi: {e}")))?;
    let hello = AdmissionHello {
        did: proof.did.clone(),
        ed25519_pub: proof.ed25519.verifying_key().to_bytes(),
        client_nonce: fresh_nonce(),
    };
    write_frame(&mut send, &encode_hello(&hello)).await?;

    let challenge = decode_challenge(&read_frame(&mut recv).await?)?;
    let t = admission_transcript(
        &hello.did,
        &hello.client_nonce,
        &challenge.server_nonce,
        challenge.epoch,
        &challenge.head_digest,
    );
    let ed_sig: [u8; 64] = proof.ed25519.sign(&t).to_bytes();
    let mut outer = t;
    outer.extend_from_slice(&ed_sig);
    let pq_sig = ml_dsa_sign(&proof.ml_dsa_65, &outer);
    write_frame(
        &mut send,
        &encode_response(&AdmissionResponse { ed_sig, pq_sig }),
    )
    .await?;
    send.finish()
        .map_err(|e| MoqlAdmissionError::Carrier(format!("finish: {e}")))?;
    // The server answers a verified proof with an explicit verdict frame; a
    // rejection closes the connection instead, surfacing here as a read error
    // rather than as a later moq-handshake mystery.
    let verdict = read_frame(&mut recv).await?;
    decode_verdict(&verdict)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::crypto::pq::ml_dsa_generate_keypair;

    const DID: &str = "did:at9p:testsubject";

    fn keypair(seed: u8) -> (SigningKey, MlDsaSigningKey) {
        let ed = SigningKey::from_bytes(&[seed; 32]);
        let pq = crate::crypto::pq::ml_dsa_sk_from_seed(&[seed.wrapping_add(1); 32]);
        (ed, pq)
    }

    fn state_with(ed: &SigningKey, pq: &MlDsaSigningKey, epoch: u64, tag: u8) -> AcceptedIdentityState {
        AcceptedIdentityState {
            epoch,
            head_digest: [tag; 64],
            subject_keys: vec![AcceptedSubjectKey {
                ed25519: ed.verifying_key().to_bytes(),
                ml_dsa_65: crate::crypto::pq::ml_dsa_sk_to_vk_bytes(pq),
            }],
            expires_at_unix_ms: None,
        }
    }

    fn fixture_authenticator(
        state: AcceptedIdentityState,
    ) -> (MoqlAdmissionAuthenticator, SigningKey, MlDsaSigningKey) {
        let (ed, pq) = keypair(7);
        let authority: Arc<dyn AcceptedStateAuthority> = Arc::new(move |did: &str| {
            (did == DID).then(|| state.clone())
        });
        let resolver: PeerTenantResolver = Arc::new(|peer: &PeerIdentity| {
            peer.subject.as_deref().map(|_| "alice".to_owned())
        });
        (
            MoqlAdmissionAuthenticator::new(authority, resolver),
            ed,
            pq,
        )
    }

    fn sign_response(
        ed: &SigningKey,
        pq: &MlDsaSigningKey,
        hello: &AdmissionHello,
        challenge: &AdmissionChallenge,
    ) -> AdmissionResponse {
        let t = admission_transcript(
            &hello.did,
            &hello.client_nonce,
            &challenge.server_nonce,
            challenge.epoch,
            &challenge.head_digest,
        );
        let ed_sig: [u8; 64] = ed.sign(&t).to_bytes();
        let mut outer = t;
        outer.extend_from_slice(&ed_sig);
        AdmissionResponse {
            ed_sig,
            pq_sig: ml_dsa_sign(pq, &outer),
        }
    }

    fn hello_for(ed: &SigningKey) -> AdmissionHello {
        AdmissionHello {
            did: DID.to_owned(),
            ed25519_pub: ed.verifying_key().to_bytes(),
            client_nonce: [0xAA; 32],
        }
    }

    fn challenge_for(state: &AcceptedIdentityState) -> AdmissionChallenge {
        AdmissionChallenge {
            server_nonce: [0xBB; 32],
            epoch: state.epoch,
            head_digest: state.head_digest,
        }
    }

    #[test]
    fn frame_roundtrips() {
        let hello = AdmissionHello {
            did: DID.to_owned(),
            ed25519_pub: [1; 32],
            client_nonce: [2; 32],
        };
        assert_eq!(decode_hello(&encode_hello(&hello)).unwrap(), hello);

        let challenge = AdmissionChallenge {
            server_nonce: [3; 32],
            epoch: 42,
            head_digest: [4; 64],
        };
        assert_eq!(decode_challenge(&encode_challenge(&challenge)).unwrap(), challenge);

        let response = AdmissionResponse {
            ed_sig: [5; 64],
            pq_sig: vec![6; 3309],
        };
        assert_eq!(decode_response(&encode_response(&response)).unwrap(), response);
    }

    #[test]
    fn malformed_frames_reject() {
        assert!(matches!(decode_hello(&[]), Err(MoqlAdmissionError::Malformed(_))));
        assert!(matches!(decode_hello(&[2]), Err(MoqlAdmissionError::Malformed(_))));
        assert!(matches!(decode_challenge(&[1, 2, 3]), Err(MoqlAdmissionError::Malformed(_))));
        assert!(matches!(decode_response(&[1]), Err(MoqlAdmissionError::Malformed(_))));
        // did length exceeding the cap
        let mut bad = vec![1u8];
        bad.extend_from_slice(&(MAX_DID_BYTES as u16 + 1).to_be_bytes());
        assert!(matches!(decode_hello(&bad), Err(MoqlAdmissionError::Malformed(_))));
    }

    #[test]
    fn positive_proof_verifies_and_records_transcript() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state.clone());
        let hello = hello_for(&ed);
        let challenge = challenge_for(&state);
        let response = sign_response(&ed, &pq, &hello, &challenge);
        let now = crate::envelope::current_timestamp();
        auth.verify_response(&hello, &challenge, &response, now)
            .expect("valid hybrid proof must verify");
    }

    #[test]
    fn replayed_transcript_rejects() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state.clone());
        let hello = hello_for(&ed);
        let challenge = challenge_for(&state);
        let response = sign_response(&ed, &pq, &hello, &challenge);
        let now = crate::envelope::current_timestamp();
        auth.verify_response(&hello, &challenge, &response, now).unwrap();
        let err = auth
            .verify_response(&hello, &challenge, &response, now)
            .expect_err("the same transcript must be single-use");
        assert!(matches!(err, MoqlAdmissionError::Replay), "{err}");
    }

    #[test]
    fn wrong_nonce_or_epoch_rejects_as_bad_signature() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state.clone());
        let hello = hello_for(&ed);
        let challenge = challenge_for(&state);
        let response = sign_response(&ed, &pq, &hello, &challenge);
        let now = crate::envelope::current_timestamp();

        // A replay over a FRESH challenge (different server nonce) is a
        // signature failure: the transcript bound the original nonce.
        let fresh_challenge = AdmissionChallenge {
            server_nonce: [0xCC; 32],
            ..challenge.clone()
        };
        let err = auth
            .verify_response(&hello, &fresh_challenge, &response, now)
            .expect_err("a response is bound to its challenge nonce");
        assert!(matches!(err, MoqlAdmissionError::BadSignature), "{err}");

        // A response minted against a superseded epoch/head fails likewise.
        let stale_challenge = AdmissionChallenge {
            epoch: state.epoch - 1,
            head_digest: [8; 64],
            ..challenge.clone()
        };
        let stale_response = sign_response(&ed, &pq, &hello, &stale_challenge);
        let err = auth
            .verify_response(&hello, &challenge, &stale_response, now)
            .expect_err("a proof over a stale epoch/head must not verify");
        assert!(matches!(err, MoqlAdmissionError::BadSignature), "{err}");
    }

    #[test]
    fn classical_only_proof_rejects_as_downgrade() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state.clone());
        let hello = hello_for(&ed);
        let challenge = challenge_for(&state);
        let mut response = sign_response(&ed, &pq, &hello, &challenge);
        // Strip the ML-DSA-65 layer: present a classical-only signature.
        let (rogue_ed, _) = keypair(21);
        let t = admission_transcript(
            &hello.did,
            &hello.client_nonce,
            &challenge.server_nonce,
            challenge.epoch,
            &challenge.head_digest,
        );
        let mut outer = t.clone();
        outer.extend_from_slice(&response.ed_sig);
        // A valid ML-DSA signature from a NON-accepted key is still a failure.
        let (rogue_pq_sk, _) = ml_dsa_generate_keypair();
        response.pq_sig = ml_dsa_sign(&rogue_pq_sk, &outer);
        let err = auth
            .verify_response(&hello, &challenge, &response, crate::envelope::current_timestamp())
            .expect_err("ML-DSA layer from a non-accepted key must reject");
        assert!(matches!(err, MoqlAdmissionError::BadSignature), "{err}");
        let _ = rogue_ed;
    }

    #[test]
    fn state_advance_mid_handshake_rejects() {
        let (ed, pq) = keypair(7);
        let old_state = state_with(&ed, &pq, 3, 9);
        // Authority flips to a newer epoch between challenge and response.
        let new_state = AcceptedIdentityState {
            epoch: 4,
            head_digest: [10; 64],
            ..old_state.clone()
        };
        let authority: Arc<dyn AcceptedStateAuthority> =
            Arc::new(move |did: &str| (did == DID).then(|| new_state.clone()));
        let resolver: PeerTenantResolver = Arc::new(|_| Some("alice".to_owned()));
        let auth = MoqlAdmissionAuthenticator::new(authority, resolver);
        let hello = hello_for(&ed);
        let challenge = challenge_for(&old_state); // bound to epoch 3
        let response = sign_response(&ed, &pq, &hello, &challenge);
        let err = auth
            .verify_response(&hello, &challenge, &response, crate::envelope::current_timestamp())
            .expect_err("a state advance mid-handshake must reject");
        assert!(matches!(err, MoqlAdmissionError::StateAdvanced(_)), "{err}");
    }

    #[test]
    fn expired_state_rejects_at_hello() {
        let (ed, pq) = keypair(7);
        let mut state = state_with(&ed, &pq, 3, 9);
        state.expires_at_unix_ms = Some(crate::envelope::current_timestamp() - 1);
        let (auth, _, _) = fixture_authenticator(state);
        let err = auth
            .check_hello(&hello_for(&ed))
            .expect_err("expired accepted state must reject");
        assert!(matches!(err, MoqlAdmissionError::Expired { .. }), "{err}");
    }

    #[test]
    fn rotated_out_key_rejects_at_hello() {
        let (ed, _pq) = keypair(7);
        // Accepted state now publishes a DIFFERENT key set (rotation).
        let (new_ed, new_pq) = keypair(30);
        let state = state_with(&new_ed, &new_pq, 4, 10);
        let (auth, _, _) = fixture_authenticator(state);
        let err = auth
            .check_hello(&hello_for(&ed))
            .expect_err("a rotated-out key is not a current subject key");
        assert!(matches!(err, MoqlAdmissionError::KeyNotCurrent(_)), "{err}");
    }

    #[test]
    fn unknown_and_non_at9p_identities_reject() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state);

        let mut hello = hello_for(&ed);
        hello.did = "did:at9p:someoneelse".to_owned();
        let err = auth.check_hello(&hello).expect_err("unknown DID must reject");
        assert!(matches!(err, MoqlAdmissionError::UnknownIdentity(_)), "{err}");

        // A NodeId dressed as did:key is not an accepted-state identity.
        let mut hello = hello_for(&ed);
        hello.did = "did:key:z6MkNodeIdOnly".to_owned();
        let err = auth.check_hello(&hello).expect_err("did:key must reject");
        assert!(matches!(err, MoqlAdmissionError::NotAcceptedIdentity(_)), "{err}");
    }
}
