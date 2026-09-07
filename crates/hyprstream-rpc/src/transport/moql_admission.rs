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
//!    client's current accepted-state `epoch` and `head_digest`, and the
//!    server identity pinned by the signed `StreamInfo` response.
//! 3. **Response** (client → server): a nested composite signature over the
//!    transcript — the inner Ed25519 layer signs the transcript `T`, the outer
//!    ML-DSA-65 layer signs `T ‖ ed_sig` (the same inner→outer nesting the at9p
//!    record composite uses, `hyprstream-pds::at9p_sign`).
//! 4. **Confirmation** (server → client, success only): a nested server
//!    Ed25519 + ML-DSA-65 signature covers every previous frame, including the
//!    server accepted-state witness and the client's complete hybrid response.
//!    Only after verifying it may the client start the MoQ handshake. Any
//!    rejection instead closes the connection.
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

use crate::crypto::pq::{
    ml_dsa_sign, ml_dsa_sk_to_vk_bytes, ml_dsa_verify, ml_dsa_vk_from_bytes, MlDsaSigningKey,
};
use crate::identity::DID_AT9P_PREFIX;
use crate::moq_authz::{is_valid_tenant_segment, PeerIdentity};
use crate::transport::iroh_moq::PeerTenantResolver;

/// Domain separation tag at the head of every admission transcript.
pub const MOQL_ADMISSION_DOMAIN: &[u8] = b"hyprstream/moql-admission/v1";

/// Wire format version byte. Version 3 binds both locally-observed Iroh
/// endpoint IDs into the hybrid proof and mutual server confirmation.
const WIRE_VERSION: u8 = 3;

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
    /// An admission-enabled server did not have the local private identity
    /// required to prove possession to the client.
    #[error("moql server has no accepted-state signing identity")]
    ServerIdentityUnavailable,
    /// The server's configured private keys do not match its public,
    /// resolver-verified accepted-state witness.
    #[error("moql server signing identity does not match its accepted-state witness")]
    ServerIdentityMismatch,
    /// The local server witness expired, advanced, or lost its published key
    /// pair after startup. It cannot be used to confirm a new or live tunnel.
    #[error("moql server accepted-state identity is no longer current")]
    ServerIdentityNotCurrent,
    /// The remote server's challenge did not carry precisely the witness that
    /// the signed StreamInfo authenticated for this dial.
    #[error("moql challenge server identity does not match resolver witness")]
    WrongServer,
    /// The server did not provide a valid hybrid signature over the completed
    /// admission exchange. The client must never proceed to MoQ in this case.
    #[error("moql server did not mutually confirm the full admission transcript")]
    ServerUnconfirmed,
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
    /// The accepted current Ed25519 key that proved possession. Retained with
    /// the tunnel witness so an active session can be invalidated if that key
    /// is rotated out without changing the tenant mapping.
    pub subject_ed25519: [u8; 32],
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
#[derive(Clone, Debug, PartialEq)]
pub struct AdmissionChallenge {
    /// Fresh random server nonce, single-use per connection.
    pub server_nonce: [u8; 32],
    /// The server's current accepted-state epoch for the DID.
    pub epoch: u64,
    /// The server's current accepted-state head digest for the DID.
    pub head_digest: [u8; 64],
    /// The server accepted-state identity expected by this client from the
    /// signed StreamInfo response. It is signed in the final confirmation.
    pub server_identity: crate::stream_info::MoqlServerIdentity,
}

/// Response (client → server): the nested composite proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionResponse {
    /// Ed25519 signature over the transcript `T`.
    pub ed_sig: [u8; 64],
    /// ML-DSA-65 signature over `T ‖ ed_sig` (inner→outer nesting).
    pub pq_sig: Vec<u8>,
}

/// Server → client mutual confirmation. Both signature halves cover every
/// admission frame: hello, challenge (including the server accepted-state
/// witness), and the client's nested hybrid response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionConfirmation {
    /// Ed25519 signature over [`mutual_confirmation_transcript`].
    pub ed_sig: [u8; 64],
    /// ML-DSA-65 signature over the transcript followed by `ed_sig`.
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

/// Canonical public encoding of a resolver-verified server identity. It is
/// deliberately included in both the challenge and the signed confirmation so
/// a carrier peer cannot substitute a server between RPC resolution and MoQ.
fn encode_server_identity(identity: &crate::stream_info::MoqlServerIdentity) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(2 + identity.did.len() + 8 + 64 + 8 + 32 + 2 + identity.ml_dsa65.len());
    out.extend_from_slice(&(identity.did.len() as u16).to_be_bytes());
    out.extend_from_slice(identity.did.as_bytes());
    out.extend_from_slice(&identity.epoch.to_be_bytes());
    out.extend_from_slice(&identity.head_digest);
    out.extend_from_slice(&identity.expires_at_unix_ms.to_be_bytes());
    out.extend_from_slice(&identity.ed25519);
    out.extend_from_slice(&(identity.ml_dsa65.len() as u16).to_be_bytes());
    out.extend_from_slice(&identity.ml_dsa65);
    out
}

fn decode_server_identity(
    bytes: &[u8],
) -> Result<crate::stream_info::MoqlServerIdentity, MoqlAdmissionError> {
    let malformed = |why: &str| MoqlAdmissionError::Malformed(format!("server identity: {why}"));
    if bytes.len() < 2 {
        return Err(malformed("truncated DID length"));
    }
    let did_len = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
    if did_len == 0 || did_len > MAX_DID_BYTES || bytes.len() < 2 + did_len + 8 + 64 + 8 + 32 + 2 {
        return Err(malformed("invalid DID length or truncated fields"));
    }
    let did = std::str::from_utf8(&bytes[2..2 + did_len])
        .map_err(|_| malformed("DID is not UTF-8"))?
        .to_owned();
    let offset = 2 + did_len;
    let mut epoch = [0u8; 8];
    epoch.copy_from_slice(&bytes[offset..offset + 8]);
    let mut head_digest = vec![0u8; 64];
    head_digest.copy_from_slice(&bytes[offset + 8..offset + 72]);
    let mut expiry = [0u8; 8];
    expiry.copy_from_slice(&bytes[offset + 72..offset + 80]);
    let mut ed25519 = [0u8; 32];
    ed25519.copy_from_slice(&bytes[offset + 80..offset + 112]);
    let pq_len = u16::from_be_bytes([bytes[offset + 112], bytes[offset + 113]]) as usize;
    let pq = &bytes[offset + 114..];
    if pq_len == 0 || pq_len != pq.len() {
        return Err(malformed("ML-DSA-65 length mismatch or empty"));
    }
    Ok(crate::stream_info::MoqlServerIdentity {
        did,
        epoch: u64::from_be_bytes(epoch),
        head_digest,
        expires_at_unix_ms: i64::from_be_bytes(expiry),
        ed25519,
        ml_dsa65: pq.to_vec(),
    })
}

/// The signed transcript both sides reconstruct byte-identically.
pub fn admission_transcript(
    did: &str,
    client_nonce: &[u8; 32],
    server_nonce: &[u8; 32],
    epoch: u64,
    head_digest: &[u8; 64],
    client_carrier_node_id: &[u8; 32],
    server_carrier_node_id: &[u8; 32],
) -> Vec<u8> {
    let mut t = Vec::with_capacity(
        MOQL_ADMISSION_DOMAIN.len() + 1 + 4 + 1 + 2 + did.len() + 32 + 32 + 8 + 64 + 32 + 32,
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
    t.extend_from_slice(client_carrier_node_id);
    t.extend_from_slice(server_carrier_node_id);
    t
}

/// The complete mutually-confirmed admission transcript. Unlike the client
/// proof's preface, this includes the advertised server accepted-state witness
/// and the client's two signature layers, so the server cannot be substituted
/// after the client proves possession.
pub fn mutual_confirmation_transcript(
    hello: &AdmissionHello,
    challenge: &AdmissionChallenge,
    response: &AdmissionResponse,
    client_carrier_node_id: &[u8; 32],
    server_carrier_node_id: &[u8; 32],
) -> Vec<u8> {
    let client = admission_transcript(
        &hello.did,
        &hello.client_nonce,
        &challenge.server_nonce,
        challenge.epoch,
        &challenge.head_digest,
        client_carrier_node_id,
        server_carrier_node_id,
    );
    let server = encode_server_identity(&challenge.server_identity);
    let response = encode_response(response);
    let mut out = Vec::with_capacity(
        MOQL_ADMISSION_DOMAIN.len() + 1 + client.len() + 4 + server.len() + 4 + response.len(),
    );
    out.extend_from_slice(MOQL_ADMISSION_DOMAIN);
    out.extend_from_slice(b"/mutual-confirm/v1\0");
    out.extend_from_slice(&(client.len() as u32).to_be_bytes());
    out.extend_from_slice(&client);
    out.extend_from_slice(&(server.len() as u32).to_be_bytes());
    out.extend_from_slice(&server);
    out.extend_from_slice(&(response.len() as u32).to_be_bytes());
    out.extend_from_slice(&response);
    out
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
    let identity = encode_server_identity(&challenge.server_identity);
    let mut out = Vec::with_capacity(1 + 32 + 8 + 64 + 2 + identity.len());
    out.push(WIRE_VERSION);
    out.extend_from_slice(&challenge.server_nonce);
    out.extend_from_slice(&challenge.epoch.to_be_bytes());
    out.extend_from_slice(&challenge.head_digest);
    out.extend_from_slice(&(identity.len() as u16).to_be_bytes());
    out.extend_from_slice(&identity);
    out
}

/// Decode a Challenge frame payload.
pub fn decode_challenge(bytes: &[u8]) -> Result<AdmissionChallenge, MoqlAdmissionError> {
    let malformed = |why: &str| MoqlAdmissionError::Malformed(format!("challenge: {why}"));
    let (version, rest) = bytes.split_first().ok_or_else(|| malformed("empty"))?;
    if *version != WIRE_VERSION {
        return Err(malformed("unsupported wire version"));
    }
    if rest.len() < 32 + 8 + 64 + 2 {
        return Err(malformed("bad length"));
    }
    let mut server_nonce = [0u8; 32];
    server_nonce.copy_from_slice(&rest[..32]);
    let mut epoch = [0u8; 8];
    epoch.copy_from_slice(&rest[32..40]);
    let mut head_digest = [0u8; 64];
    head_digest.copy_from_slice(&rest[40..104]);
    let identity_len = u16::from_be_bytes([rest[104], rest[105]]) as usize;
    let identity = &rest[106..];
    if identity_len != identity.len() {
        return Err(malformed("server identity length mismatch"));
    }
    Ok(AdmissionChallenge {
        server_nonce,
        epoch: u64::from_be_bytes(epoch),
        head_digest,
        server_identity: decode_server_identity(identity)?,
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

/// Encode a hybrid server confirmation frame.
pub fn encode_confirmation(confirmation: &AdmissionConfirmation) -> Vec<u8> {
    encode_response(&AdmissionResponse {
        ed_sig: confirmation.ed_sig,
        pq_sig: confirmation.pq_sig.clone(),
    })
}

/// Decode a hybrid server confirmation frame.
pub fn decode_confirmation(bytes: &[u8]) -> Result<AdmissionConfirmation, MoqlAdmissionError> {
    let response = decode_response(bytes)?;
    Ok(AdmissionConfirmation {
        ed_sig: response.ed_sig,
        pq_sig: response.pq_sig,
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
#[derive(Clone)]
pub struct MoqlServerIdentityProof {
    /// The public resolver-verified state this server must prove possession of.
    pub identity: crate::stream_info::MoqlServerIdentity,
    /// Private Ed25519 half corresponding to `identity.ed25519`.
    pub ed25519: SigningKey,
    /// Private ML-DSA-65 half corresponding to `identity.ml_dsa65`.
    pub ml_dsa_65: MlDsaSigningKey,
}

impl MoqlServerIdentityProof {
    /// Construct the server confirmation material from the same checkpointed
    /// local identity already used for native client admission. This does not
    /// create a credential: both public keys must exactly match the accepted
    /// witness captured by the native announcement.
    pub fn from_local_admission_proof(
        proof: &MoqlAdmissionProof,
    ) -> Result<Self, MoqlAdmissionError> {
        let identity = proof.expected_server.clone();
        if !identity.did.starts_with(DID_AT9P_PREFIX)
            || identity.did.len() > MAX_DID_BYTES
            || identity.epoch == 0
            || identity.head_digest.len() != 64
            || identity.ml_dsa65.is_empty()
            || identity.ml_dsa65.len() > u16::MAX as usize
            || identity.expires_at_unix_ms <= crate::envelope::current_timestamp()
            || identity.ed25519 != proof.ed25519.verifying_key().to_bytes()
            || identity.ml_dsa65 != ml_dsa_sk_to_vk_bytes(&proof.ml_dsa_65)
        {
            return Err(MoqlAdmissionError::ServerIdentityMismatch);
        }
        Ok(Self {
            identity,
            ed25519: proof.ed25519.clone(),
            ml_dsa_65: proof.ml_dsa_65.clone(),
        })
    }

    fn sign(
        &self,
        hello: &AdmissionHello,
        challenge: &AdmissionChallenge,
        response: &AdmissionResponse,
        client_carrier_node_id: &[u8; 32],
        server_carrier_node_id: &[u8; 32],
    ) -> AdmissionConfirmation {
        let transcript = mutual_confirmation_transcript(
            hello,
            challenge,
            response,
            client_carrier_node_id,
            server_carrier_node_id,
        );
        let ed_sig: [u8; 64] = self.ed25519.sign(&transcript).to_bytes();
        let mut outer = transcript;
        outer.extend_from_slice(&ed_sig);
        AdmissionConfirmation {
            ed_sig,
            pq_sig: ml_dsa_sign(&self.ml_dsa_65, &outer),
        }
    }
}

pub struct MoqlAdmissionAuthenticator {
    authority: Arc<dyn AcceptedStateAuthority>,
    tenant_resolver: PeerTenantResolver,
    timeout: Duration,
    /// Consumed transcript digests → the unix-ms after which the record may be
    /// dropped. Makes each accepted transcript single-use.
    used: Mutex<HashMap<[u8; 32], i64>>,
    /// The server's private accepted identity for the mutual confirmation.
    /// Absence is fail-closed on every admission exchange.
    server_identity: Mutex<Option<MoqlServerIdentityProof>>,
    /// Locally observed Iroh endpoint ID for this server. This is carrier
    /// binding only, never application identity or policy authority.
    server_carrier_node_id: Mutex<Option<[u8; 32]>>,
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
            server_identity: Mutex::new(None),
            server_carrier_node_id: Mutex::new(None),
        }
    }

    /// Install the local accepted-state identity that signs server confirmation
    /// frames. A bare authenticator remains useful for unit decision tests but
    /// cannot admit a network tunnel.
    #[must_use]
    pub fn with_server_identity_and_carrier(
        mut self,
        server_identity: MoqlServerIdentityProof,
        server_carrier_node_id: [u8; 32],
    ) -> Self {
        self.server_identity = Mutex::new(Some(server_identity));
        self.server_carrier_node_id = Mutex::new(Some(server_carrier_node_id));
        self
    }

    /// Attach the local confirmation proof at the service-spawner boundary.
    /// A caller that already supplied one must agree on the same public
    /// accepted-state identity; silently replacing it would make configuration
    /// order an authentication decision.
    pub fn install_server_identity(
        &self,
        server_identity: MoqlServerIdentityProof,
        server_carrier_node_id: [u8; 32],
    ) -> Result<(), MoqlAdmissionError> {
        let mut installed = self.server_identity.lock();
        match installed.as_ref() {
            Some(existing) if existing.identity != server_identity.identity => {
                Err(MoqlAdmissionError::ServerIdentityMismatch)
            }
            Some(_) => Ok(()),
            None => {
                *installed = Some(server_identity);
                *self.server_carrier_node_id.lock() = Some(server_carrier_node_id);
                Ok(())
            }
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
        let server_carrier_node_id = self
            .server_carrier_node_id
            .lock()
            .as_ref()
            .copied()
            .ok_or(MoqlAdmissionError::ServerIdentityUnavailable)?;
        let server_identity = self
            .server_identity
            .lock()
            .clone()
            .ok_or(MoqlAdmissionError::ServerIdentityUnavailable)?;
        if !self.is_server_identity_current(&server_identity) {
            return Err(MoqlAdmissionError::ServerIdentityNotCurrent);
        }
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
            server_identity: server_identity.identity.clone(),
        };
        write_frame(&mut send, &encode_challenge(&challenge)).await?;

        // ── Response ─────────────────────────────────────────────────────────
        let response = decode_response(&read_frame(&mut recv).await?)?;
        let now = crate::envelope::current_timestamp();
        self.verify_response_bound(
            &hello,
            &challenge,
            &response,
            now,
            &carrier_node_id,
            &server_carrier_node_id,
        )?;
        // The server witness is a live authorization fact too. Re-read it
        // after the client response so an expiry or state advance while this
        // exchange was in flight cannot receive a valid confirmation.
        if !self.is_server_identity_current(&server_identity) {
            return Err(MoqlAdmissionError::ServerIdentityNotCurrent);
        }

        // ── Tenant binding (server-side, from the verified subject) ──────────
        let peer = PeerIdentity::authenticated(hello.did.clone());
        let tenant = (self.tenant_resolver)(&peer)
            .ok_or_else(|| MoqlAdmissionError::TenantUnresolved(hello.did.clone()))?;
        if !is_valid_tenant_segment(&tenant) {
            return Err(MoqlAdmissionError::InvalidTenant(tenant));
        }
        // Only after client proof verification does the server prove possession
        // of the resolver-pinned server identity over the *entire* exchange.
        // A bare verdict would be forgeable by an on-path wrong server.
        let confirmation = server_identity.sign(
            &hello,
            &challenge,
            &response,
            &carrier_node_id,
            &server_carrier_node_id,
        );
        write_frame(&mut send, &encode_confirmation(&confirmation)).await?;
        send.finish()
            .map_err(|e| MoqlAdmissionError::Carrier(format!("finish: {e}")))?;
        Ok(AdmittedMoqPeer {
            peer,
            tenant,
            epoch: state.epoch,
            head_digest: state.head_digest,
            subject_ed25519: hello.ed25519_pub,
            carrier_node_id,
        })
    }

    /// Whether a previously admitted tunnel remains bound to the same live
    /// accepted state. This deliberately re-reads the daemon-owned authority:
    /// session admission is not a permanent capability after a state advance,
    /// key rotation, expiry, or withdrawal.
    pub fn is_still_current(&self, admitted: &AdmittedMoqPeer) -> bool {
        let server_identity = self.server_identity.lock().clone();
        let Some(current) = admitted
            .peer
            .subject
            .as_deref()
            .and_then(|did| self.authority.accepted_state(did))
        else {
            return false;
        };
        current.is_live(crate::envelope::current_timestamp())
            && current.epoch == admitted.epoch
            && current.head_digest == admitted.head_digest
            && current.subject_key_for(&admitted.subject_ed25519).is_some()
            // Tenant assignment is live deployment authority: removal or
            // reassignment closes a previously admitted scoped session.
            && (self.tenant_resolver)(&admitted.peer).as_deref() == Some(admitted.tenant.as_str())
            && server_identity
                .as_ref()
                .is_some_and(|server| self.is_server_identity_current(server))
    }

    fn is_server_identity_current(&self, server: &MoqlServerIdentityProof) -> bool {
        let Some(current) = self.authority.accepted_state(&server.identity.did) else {
            return false;
        };
        if server.identity.head_digest.len() != 64 {
            return false;
        }
        let mut head_digest = [0u8; 64];
        head_digest.copy_from_slice(&server.identity.head_digest);
        current.is_live(crate::envelope::current_timestamp())
            && current.epoch == server.identity.epoch
            && current.head_digest == head_digest
            && current
                .subject_key_for(&server.identity.ed25519)
                .is_some_and(|key| key.ml_dsa_65 == server.identity.ml_dsa65)
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
    /// unit-testable without a carrier. The unbound shim is only for those
    /// unit decision tests; the I/O path always uses `verify_response_bound`.
    #[cfg(test)]
    fn verify_response(
        &self,
        hello: &AdmissionHello,
        challenge: &AdmissionChallenge,
        response: &AdmissionResponse,
        now_unix_ms: i64,
    ) -> Result<(), MoqlAdmissionError> {
        self.verify_response_bound(hello, challenge, response, now_unix_ms, &[0; 32], &[1; 32])
    }

    fn verify_response_bound(
        &self,
        hello: &AdmissionHello,
        challenge: &AdmissionChallenge,
        response: &AdmissionResponse,
        now_unix_ms: i64,
        client_carrier_node_id: &[u8; 32],
        server_carrier_node_id: &[u8; 32],
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
            client_carrier_node_id,
            server_carrier_node_id,
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
        used.get(digest)
            .is_some_and(|discard_after| now_unix_ms < *discard_after)
    }

    fn mark_consumed(&self, digest: [u8; 32], now_unix_ms: i64) {
        let horizon_ms = REPLAY_HORIZON.as_millis() as i64;
        let mut used = self.used.lock();
        if used.len() >= REPLAY_CACHE_MAX {
            used.retain(|_, discard_after| now_unix_ms < *discard_after);
        }
        // A sustained stream of valid, distinct admissions can fill the whole
        // replay horizon without making any entry expired. Bound resident state
        // in that case too: retain the newest horizon of transcripts, evicting
        // the entry that will expire first before admitting the new transcript.
        while used.len() >= REPLAY_CACHE_MAX {
            let Some(oldest) = used
                .iter()
                .min_by_key(|(_, discard_after)| *discard_after)
                .map(|(digest, _)| *digest)
            else {
                break;
            };
            used.remove(&oldest);
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
#[derive(Clone)]
pub struct MoqlAdmissionProof {
    /// The peer's `did:at9p` DID.
    pub did: String,
    /// Ed25519 signing key of an accepted current subject key.
    pub ed25519: SigningKey,
    /// ML-DSA-65 signing key bound to that Ed25519 key in the accepted state.
    pub ml_dsa_65: MlDsaSigningKey,
    /// Public accepted-state witness expected from the Iroh server during
    /// mutual admission. The service bootstrap holds its own checkpointed
    /// witness here for server-confirmation construction; the shared reach
    /// resolver replaces it with the remote RPC-authenticated `StreamInfo`
    /// witness for every client dial. Empty/default is never accepted.
    pub expected_server: crate::stream_info::MoqlServerIdentity,
}

impl std::fmt::Debug for MoqlAdmissionProof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MoqlAdmissionProof")
            .field("did", &self.did)
            .finish_non_exhaustive()
    }
}

fn verify_server_challenge(
    expected: &crate::stream_info::MoqlServerIdentity,
    challenge: &AdmissionChallenge,
) -> Result<(), MoqlAdmissionError> {
    if !expected.did.starts_with(DID_AT9P_PREFIX)
        || expected.did.len() > MAX_DID_BYTES
        || expected.epoch == 0
        || expected.head_digest.len() != 64
        || expected.ml_dsa65.is_empty()
        || expected.ml_dsa65.len() > u16::MAX as usize
        || expected.expires_at_unix_ms <= crate::envelope::current_timestamp()
        || &challenge.server_identity != expected
    {
        return Err(MoqlAdmissionError::WrongServer);
    }
    Ok(())
}

fn verify_server_confirmation(
    expected: &crate::stream_info::MoqlServerIdentity,
    hello: &AdmissionHello,
    challenge: &AdmissionChallenge,
    response: &AdmissionResponse,
    confirmation: &AdmissionConfirmation,
    client_carrier_node_id: &[u8; 32],
    server_carrier_node_id: &[u8; 32],
) -> Result<(), MoqlAdmissionError> {
    let ed_vk = VerifyingKey::from_bytes(&expected.ed25519)
        .map_err(|_| MoqlAdmissionError::ServerUnconfirmed)?;
    let pq_vk = ml_dsa_vk_from_bytes(&expected.ml_dsa65)
        .map_err(|_| MoqlAdmissionError::ServerUnconfirmed)?;
    let transcript = mutual_confirmation_transcript(
        hello,
        challenge,
        response,
        client_carrier_node_id,
        server_carrier_node_id,
    );
    ed_vk
        .verify(&transcript, &Signature::from_bytes(&confirmation.ed_sig))
        .map_err(|_| MoqlAdmissionError::ServerUnconfirmed)?;
    let mut outer = transcript;
    outer.extend_from_slice(&confirmation.ed_sig);
    ml_dsa_verify(&pq_vk, &outer, &confirmation.pq_sig)
        .map_err(|_| MoqlAdmissionError::ServerUnconfirmed)
}

/// Run the client half of the admission exchange on `conn`, consuming its
/// first bi stream. On success the connection is ready to be wrapped as a
/// `web_transport_iroh::Session` for the moq handshake; on failure the server
/// has closed (or will close) the connection and the caller MUST NOT proceed.
pub async fn prove_moql_admission(
    conn: &Connection,
    proof: &MoqlAdmissionProof,
    client_carrier_node_id: [u8; 32],
    timeout: Duration,
) -> Result<(), MoqlAdmissionError> {
    match tokio::time::timeout(timeout, prove_exchange(conn, proof, client_carrier_node_id)).await {
        Ok(res) => res,
        Err(_) => Err(MoqlAdmissionError::Timeout),
    }
}

async fn prove_exchange(
    conn: &Connection,
    proof: &MoqlAdmissionProof,
    client_carrier_node_id: [u8; 32],
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
    verify_server_challenge(&proof.expected_server, &challenge)?;
    let t = admission_transcript(
        &hello.did,
        &hello.client_nonce,
        &challenge.server_nonce,
        challenge.epoch,
        &challenge.head_digest,
        &client_carrier_node_id,
        conn.remote_id().as_bytes(),
    );
    let ed_sig: [u8; 64] = proof.ed25519.sign(&t).to_bytes();
    let mut outer = t;
    outer.extend_from_slice(&ed_sig);
    let pq_sig = ml_dsa_sign(&proof.ml_dsa_65, &outer);
    write_frame(
        &mut send,
        &encode_response(&AdmissionResponse {
            ed_sig,
            pq_sig: pq_sig.clone(),
        }),
    )
    .await?;
    send.finish()
        .map_err(|e| MoqlAdmissionError::Carrier(format!("finish: {e}")))?;
    // Do not hand an unconfirmed carrier to the MoQ handshake. The server's
    // hybrid confirmation binds its resolver-pinned identity, the challenge,
    // and this client's full hybrid response into one transcript.
    let confirmation = decode_confirmation(&read_frame(&mut recv).await?)?;
    verify_server_confirmation(
        &proof.expected_server,
        &hello,
        &challenge,
        &AdmissionResponse { ed_sig, pq_sig },
        &confirmation,
        &client_carrier_node_id,
        conn.remote_id().as_bytes(),
    )
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

    fn state_with(
        ed: &SigningKey,
        pq: &MlDsaSigningKey,
        epoch: u64,
        tag: u8,
    ) -> AcceptedIdentityState {
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
        let authority: Arc<dyn AcceptedStateAuthority> =
            Arc::new(move |did: &str| (did == DID).then(|| state.clone()));
        let resolver: PeerTenantResolver =
            Arc::new(|peer: &PeerIdentity| peer.subject.as_deref().map(|_| "alice".to_owned()));
        (MoqlAdmissionAuthenticator::new(authority, resolver), ed, pq)
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
            &[0; 32],
            &[1; 32],
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
            server_identity: Default::default(),
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
            server_identity: crate::stream_info::MoqlServerIdentity {
                did: "did:at9p:test-server".to_owned(),
                epoch: 7,
                head_digest: vec![5; 64],
                expires_at_unix_ms: crate::envelope::current_timestamp() + 60_000,
                ed25519: [6; 32],
                ml_dsa65: vec![7; 1952],
            },
        };
        assert_eq!(
            decode_challenge(&encode_challenge(&challenge)).unwrap(),
            challenge
        );

        let response = AdmissionResponse {
            ed_sig: [5; 64],
            pq_sig: vec![6; 3309],
        };
        assert_eq!(
            decode_response(&encode_response(&response)).unwrap(),
            response
        );

        let confirmation = AdmissionConfirmation {
            ed_sig: [8; 64],
            pq_sig: vec![9; 3309],
        };
        assert_eq!(
            decode_confirmation(&encode_confirmation(&confirmation)).unwrap(),
            confirmation
        );
    }

    #[test]
    fn malformed_frames_reject() {
        assert!(matches!(
            decode_hello(&[]),
            Err(MoqlAdmissionError::Malformed(_))
        ));
        assert!(matches!(
            decode_hello(&[2]),
            Err(MoqlAdmissionError::Malformed(_))
        ));
        assert!(matches!(
            decode_challenge(&[1, 2, 3]),
            Err(MoqlAdmissionError::Malformed(_))
        ));
        assert!(matches!(
            decode_response(&[1]),
            Err(MoqlAdmissionError::Malformed(_))
        ));
        // did length exceeding the cap
        let mut bad = vec![1u8];
        bad.extend_from_slice(&(MAX_DID_BYTES as u16 + 1).to_be_bytes());
        assert!(matches!(
            decode_hello(&bad),
            Err(MoqlAdmissionError::Malformed(_))
        ));
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
        auth.verify_response(&hello, &challenge, &response, now)
            .unwrap();
        let err = auth
            .verify_response(&hello, &challenge, &response, now)
            .expect_err("the same transcript must be single-use");
        assert!(matches!(err, MoqlAdmissionError::Replay), "{err}");
    }

    #[test]
    fn replay_cache_never_exceeds_capacity_when_horizon_is_full() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state);
        let now = crate::envelope::current_timestamp();
        for index in 0..=REPLAY_CACHE_MAX {
            let mut digest = [0u8; 32];
            digest[..8].copy_from_slice(&(index as u64).to_be_bytes());
            auth.mark_consumed(digest, now.saturating_add(index as i64));
        }
        assert_eq!(auth.used.lock().len(), REPLAY_CACHE_MAX);
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
            &[0; 32],
            &[1; 32],
        );
        let mut outer = t.clone();
        outer.extend_from_slice(&response.ed_sig);
        // A valid ML-DSA signature from a NON-accepted key is still a failure.
        let (rogue_pq_sk, _) = ml_dsa_generate_keypair();
        response.pq_sig = ml_dsa_sign(&rogue_pq_sk, &outer);
        let err = auth
            .verify_response(
                &hello,
                &challenge,
                &response,
                crate::envelope::current_timestamp(),
            )
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
            .verify_response(
                &hello,
                &challenge,
                &response,
                crate::envelope::current_timestamp(),
            )
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
    fn current_session_rejects_removed_or_reassigned_tenant() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let authority: Arc<dyn AcceptedStateAuthority> = Arc::new({
            let state = state.clone();
            move |did: &str| (did == DID).then(|| state.clone())
        });
        let tenant = Arc::new(Mutex::new(Some("alice".to_owned())));
        let resolver: PeerTenantResolver = Arc::new({
            let tenant = Arc::clone(&tenant);
            move |_peer| tenant.lock().clone()
        });
        let identity = crate::stream_info::MoqlServerIdentity {
            did: DID.to_owned(), epoch: state.epoch, head_digest: state.head_digest.to_vec(),
            expires_at_unix_ms: crate::envelope::current_timestamp() + 60_000,
            ed25519: ed.verifying_key().to_bytes(), ml_dsa65: crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
        };
        let auth = MoqlAdmissionAuthenticator::new(authority, resolver).with_server_identity_and_carrier(
            MoqlServerIdentityProof { identity, ed25519: ed.clone(), ml_dsa_65: pq },
            [1; 32],
        );
        let admitted = AdmittedMoqPeer {
            peer: PeerIdentity::authenticated(DID.to_owned()), tenant: "alice".to_owned(),
            epoch: state.epoch, head_digest: state.head_digest,
            subject_ed25519: ed.verifying_key().to_bytes(), carrier_node_id: [0; 32],
        };
        assert!(auth.is_still_current(&admitted));
        *tenant.lock() = None;
        assert!(!auth.is_still_current(&admitted));
        *tenant.lock() = Some("bob".to_owned());
        assert!(!auth.is_still_current(&admitted));
    }

    #[test]
    fn unknown_and_non_at9p_identities_reject() {
        let (ed, pq) = keypair(7);
        let state = state_with(&ed, &pq, 3, 9);
        let (auth, _, _) = fixture_authenticator(state);

        let mut hello = hello_for(&ed);
        hello.did = "did:at9p:someoneelse".to_owned();
        let err = auth
            .check_hello(&hello)
            .expect_err("unknown DID must reject");
        assert!(
            matches!(err, MoqlAdmissionError::UnknownIdentity(_)),
            "{err}"
        );

        // A NodeId dressed as did:key is not an accepted-state identity.
        let mut hello = hello_for(&ed);
        hello.did = "did:key:z6MkNodeIdOnly".to_owned();
        let err = auth.check_hello(&hello).expect_err("did:key must reject");
        assert!(
            matches!(err, MoqlAdmissionError::NotAcceptedIdentity(_)),
            "{err}"
        );
    }
}
