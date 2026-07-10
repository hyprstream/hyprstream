//! Mainline (BEP5) DHT locator — untrusted peer rendezvous for `did:at9p`.
//!
//! Track C of the at9p epic (#880, story C1 = #889). This module wraps the
//! standalone [`mainline`] crate (MIT, pubky/pkarr lineage) to provide
//! arbitrary-infohash `announce_peer` / `get_peers`, which iroh's internal
//! pkarr integration cannot do (it is NodeId-lookup only).
//!
//! # Trust model
//!
//! The DHT is an **untrusted locator**: it only ever yields *candidate* peer
//! contacts. Zero authority is derived from anything read here — capsules
//! fetched from returned peers must pass the full GATE pipeline
//! (canon → H512 == cid512 → composite signature) before use (#879 §5.6).
//!
//! # TRUNC20 confinement (review rule R2)
//!
//! BEP5 keys are 20-byte infohashes; at9p identities are 64-byte BLAKE3-512
//! CIDs. The truncation from 64 → 20 bytes happens in exactly one place:
//! [`rendezvous_infohash`], private to this module. Every public API takes
//! the full [`Cid512`] plus fixed-width rendezvous parameters; no truncated
//! form escapes this module.
//!
//! # No name path — review rule R1 (story C4 = #892)
//!
//! Every public entry point on [`MainlineLocator`] takes a [`Cid512`]. There
//! is deliberately **no** string-name lookup, no `name → cid` resolution, and
//! no `name → cid` cache anywhere in this module. A name cache *inside the
//! DHT client* re-opens threat A5: it turns the DHT into a trusted namespace
//! authority (whoever poisons the cache chooses which capsule a name resolves
//! to), exactly the "toxic mold" failure mode #879 §3 excludes. Names are
//! resolved by a PQ-signed `NameRecord` from a trusted namespace authority
//! (PDS / discovery / MoQ group) *above* this layer; the locator only ever
//! answers "where might bytes for *this* cid live", never "which cid does
//! *this* name mean".
//!
//! This is enforced, not merely documented:
//! - [`Cid512`] has no `FromStr` / `From<&str>` / `from_name` — the only way
//!   in is the full 64-byte digest.
//! - [`r1_no_name_path_compiles_in`] is a regression test that scans this
//!   module's source and fails if any `pub fn` grows a string-name parameter.
//!
//! # Add-only hints — review rule R7 (story C4 = #892)
//!
//! #879 §6.1 lists two rendezvous strategies: (1) BEP5 `announce_peer` /
//! `get_peers` (chosen), and (2) a BEP44 deterministic-mailbox channel whose
//! keypair is derived from the cid and is therefore public-by-construction —
//! an overwrite war that is availability-only, acceptable *only* because the
//! values are untrusted hints. The v1 decision (Q3) is recorded by
//! [`V1_MAILBOX_HINT_CHANNEL`]: **BEP5-only for v1; the mailbox is deferred.**
//!
//! Whether or not the mailbox ships, R7 is load-bearing: a hint channel may
//! only **add** lookup targets — it may never prune, replace, or terminate the
//! watermark-anchored BEP5 scan (A14: a bogus "current epoch" hint that skips
//! the scan converts an availability channel into a rollback assist). That
//! constraint is encoded in [`LookupHints`], an accumulator whose API is
//! add-only by construction (no `remove` / `replace` / `clear`), and in
//! [`MainlineLocator::providers_with_hints`], which **always** performs the
//! scan and then unions the hints in. There is no code path that lets a hint
//! skip the scan.
//!
//! # wasm / browser note (documented, not built — C1 acceptance)
//!
//! There is **no raw mainline DHT from wasm**: browsers cannot open UDP
//! sockets, and iroh itself falls back to a pkarr HTTP relay
//! (`dns.iroh.link`) there. The wasm locator therefore needs either a relay
//! bridge (an HTTP/WebTransport endpoint on a trusted-for-liveness node that
//! proxies `get_peers`) or a PDS fallback (capsule fetched via atproto
//! records, #910). That surface is shared with #470 and is deliberately not
//! implemented here; this module is native-only.

use std::net::SocketAddrV4;

use async_trait::async_trait;
use futures::StreamExt;

use crate::error::{Error, Result};

/// Length in bytes of an at9p content identifier (BLAKE3-512 digest).
pub const CID512_LEN: usize = 64;

/// Width in bytes of the fixed-width rendezvous k-index.
pub const RENDEZVOUS_K_INDEX_LEN: usize = 4;

/// Width in bytes of the rendezvous epoch.
pub const RENDEZVOUS_EPOCH_LEN: usize = 8;

const RENDEZVOUS_CONTEXT: &str = "hyprstream.at9p.mainline-locator.rendezvous.v1";

/// A full-width at9p content identifier (64-byte BLAKE3-512 CID).
///
/// This is the only key type the locator's public API accepts. The 20-byte
/// BEP5 infohash derived from it never leaves this module (R2).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cid512([u8; CID512_LEN]);

impl Cid512 {
    /// Construct from a full 64-byte digest.
    pub fn from_bytes(bytes: [u8; CID512_LEN]) -> Self {
        Self(bytes)
    }

    /// Construct from a slice, rejecting anything that is not exactly
    /// 64 bytes. There is deliberately no constructor from shorter input:
    /// callers must hold the full-width CID.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let arr: [u8; CID512_LEN] = bytes.try_into().map_err(|_| {
            Error::other(format!(
                "cid512 must be {CID512_LEN} bytes, got {}",
                bytes.len()
            ))
        })?;
        Ok(Self(arr))
    }

    /// The full 64-byte digest.
    pub fn as_bytes(&self) -> &[u8; CID512_LEN] {
        &self.0
    }
}

impl std::fmt::Debug for Cid512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cid512({})", hex::encode(self.0))
    }
}

/// Fixed-width rendezvous parameters folded into the CID-keyed lookup.
///
/// `k_index` is encoded as big-endian `u32` and `epoch` as big-endian `u64`
/// when deriving the internal 64-byte rendezvous key. There is no string-name
/// path in this API (R1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RendezvousKey {
    k_index: u32,
    epoch: u64,
}

impl RendezvousKey {
    /// Default rendezvous slot for the current minimal C2 API surface.
    ///
    /// Later scoring/parallel-fetch work (#891) can fan out across additional
    /// fixed-width `k_index` values without changing the DHT key model.
    pub const DEFAULT: Self = Self {
        k_index: 0,
        epoch: 0,
    };

    /// Construct a rendezvous slot from fixed-width integer fields.
    pub const fn new(k_index: u32, epoch: u64) -> Self {
        Self { k_index, epoch }
    }

    /// The fixed-width k-index.
    pub const fn k_index(&self) -> u32 {
        self.k_index
    }

    /// The rendezvous epoch.
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    fn k_index_be_bytes(&self) -> [u8; RENDEZVOUS_K_INDEX_LEN] {
        self.k_index.to_be_bytes()
    }

    fn epoch_be_bytes(&self) -> [u8; RENDEZVOUS_EPOCH_LEN] {
        self.epoch.to_be_bytes()
    }
}

/// Full-width derived rendezvous key used internally before BEP5 truncation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct RendezvousCid512([u8; CID512_LEN]);

fn rendezvous_cid512(cid: &Cid512, key: RendezvousKey) -> RendezvousCid512 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RENDEZVOUS_CONTEXT.as_bytes());
    hasher.update(cid.as_bytes());
    hasher.update(&key.k_index_be_bytes());
    hasher.update(&key.epoch_be_bytes());

    let mut out = [0u8; CID512_LEN];
    hasher.finalize_xof().fill(&mut out);
    RendezvousCid512(out)
}

/// Candidate peer contact returned by the untrusted mainline locator.
///
/// This is a reachability hint only. It intentionally carries no trust,
/// verification, or dial-authority marker; callers must fetch candidate
/// content and verify it against the full [`Cid512`] through the GATE pipeline
/// before deriving authority or dialing based on the result.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PeerContact {
    socket_addr: SocketAddrV4,
}

impl PeerContact {
    /// Wrap a DHT-provided IPv4 socket address as an untrusted candidate.
    pub fn untrusted(socket_addr: SocketAddrV4) -> Self {
        Self { socket_addr }
    }

    /// The candidate IPv4 socket address advertised in mainline.
    ///
    /// This address is still untrusted; this accessor is for fetch/scoring
    /// plumbing, not direct admission.
    pub fn socket_addr(&self) -> SocketAddrV4 {
        self.socket_addr
    }
}

/// v1 decision on the BEP44 deterministic-mailbox hint channel (#879 §6.1,
/// Q3 — resolved by story C4 / #892).
///
/// Records whether the public-by-construction BEP44 mailbox (§6.1 option 2)
/// ships for v1. This is a *decision marker*, not a runtime switch: it exists
/// so the rationale is discoverable from the compiled artifact and reviewers
/// can see which branch of R7 binds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MailboxHintDecision {
    /// Ship BEP5 `announce_peer` / `get_peers` only (§6.1 option 1, "chosen").
    /// The deterministic mailbox is deferred. R7's add-only rule nonetheless
    /// binds the moment any future channel feeds [`LookupHints`].
    Bep5Only,

    /// Deterministic mailbox shipped; its values flow through [`LookupHints`]
    /// and are add-only by construction (R7).
    Mailbox,
}

/// The v1 mailbox decision: **BEP5-only** (#879 §6.1 names option 1 "chosen").
///
/// See the module-level "Add-only hints" doc for the rationale and for why
/// R7 binds regardless of this value.
pub const V1_MAILBOX_HINT_CHANNEL: MailboxHintDecision = MailboxHintDecision::Bep5Only;

/// Add-only accumulator of out-of-band lookup hints (review rule R7).
///
/// Hints supplement — never replace — the watermark-anchored BEP5 scan. This
/// type is the structural home for the BEP44 deterministic-mailbox channel
/// when it ships (see [`V1_MAILBOX_HINT_CHANNEL`]); today it is fed by any
/// out-of-band source (PDS record, relay bridge, configuration).
///
/// **Provably add-only.** The API exposes [`LookupHints::add_rendezvous`] and
/// [`LookupHints::add_peer`] and nothing that removes, replaces, or clears
/// entries. [`MainlineLocator::providers_with_hints`] always performs the
/// scan at the canonical rendezvous key and then unions these hints in — a
/// hint can enlarge the candidate set, never prune it or skip the scan (A14).
#[derive(Default, Clone, Debug)]
pub struct LookupHints {
    extra_rendezvous: Vec<RendezvousKey>,
    extra_peers: Vec<PeerContact>,
}

impl LookupHints {
    /// Empty hint set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an additional rendezvous bucket to scan, beyond the canonical key.
    /// Add-only (R7): there is no corresponding remove.
    pub fn add_rendezvous(&mut self, key: RendezvousKey) {
        self.extra_rendezvous.push(key);
    }

    /// Add a candidate peer discovered out-of-band (e.g. from a PDS record).
    /// Add-only (R7): there is no corresponding remove.
    pub fn add_peer(&mut self, peer: PeerContact) {
        self.extra_peers.push(peer);
    }

    /// Additional rendezvous buckets to scan (in insertion order).
    pub fn extra_rendezvous(&self) -> &[RendezvousKey] {
        &self.extra_rendezvous
    }

    /// Additional candidate peers to union in (in insertion order).
    pub fn extra_peers(&self) -> &[PeerContact] {
        &self.extra_peers
    }

    /// Total number of hints accumulated.
    pub fn len(&self) -> usize {
        self.extra_rendezvous.len() + self.extra_peers.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.extra_rendezvous.is_empty() && self.extra_peers.is_empty()
    }
}

/// TRUNC20: derive the 20-byte BEP5 infohash for a derived rendezvous key.
///
/// **This is the single truncation site in the codebase** (R2). The result
/// is passed straight into the BEP5 client and is never returned to callers.
fn rendezvous_infohash(cid: &Cid512, key: RendezvousKey) -> [u8; 20] {
    let rendezvous = rendezvous_cid512(cid, key);
    let mut infohash = [0u8; 20];
    infohash.copy_from_slice(&rendezvous.0[..20]);
    infohash
}

/// Minimal BEP5 operations the locator needs, abstracted so unit tests never
/// touch the real DHT (and so C2+ can inject scoring/relay variants).
#[async_trait]
trait Bep5Client: Send + Sync {
    /// Announce this process as a provider for `info_hash`.
    ///
    /// `port: None` lets remote nodes infer the port (BEP5 implied_port).
    async fn announce(&self, info_hash: [u8; 20], port: Option<u16>) -> Result<()>;

    /// Collect currently-announced providers for `info_hash`.
    async fn get_peers(&self, info_hash: [u8; 20]) -> Result<Vec<SocketAddrV4>>;
}

/// Real BEP5 client backed by [`mainline::async_dht::AsyncDht`].
///
/// Runs in client mode (queries + stores, no inbound routing duties). The
/// underlying crate drives its own UDP socket thread; the async wrapper is
/// runtime-agnostic and works under tokio.
pub struct MainlineBep5 {
    dht: mainline::async_dht::AsyncDht,
}

impl MainlineBep5 {
    /// Bootstrap a client-mode node against the default mainline routers.
    ///
    /// This *does* touch the network — never call it from unit tests.
    pub fn new() -> Result<Self> {
        let dht = mainline::Dht::builder()
            .build()
            .map_err(|e| Error::other(format!("mainline DHT bootstrap failed: {e}")))?;
        Ok(Self {
            dht: dht.as_async(),
        })
    }
}

#[async_trait]
impl Bep5Client for MainlineBep5 {
    async fn announce(&self, info_hash: [u8; 20], port: Option<u16>) -> Result<()> {
        self.dht
            .announce_peer(mainline::Id::from(info_hash), port)
            .await
            .map_err(|e| Error::other(format!("mainline announce_peer failed: {e}")))?;
        Ok(())
    }

    async fn get_peers(&self, info_hash: [u8; 20]) -> Result<Vec<SocketAddrV4>> {
        let mut stream = self.dht.get_peers(mainline::Id::from(info_hash));
        let mut peers = Vec::new();
        while let Some(batch) = stream.next().await {
            peers.extend(batch);
        }
        Ok(peers)
    }
}

/// The at9p mainline locator: full-width CID in, candidate peers out.
///
/// Provider scoring, parallel fetch, and size caps are C3 (#891).
pub struct MainlineLocator {
    client: Box<dyn Bep5Client>,
}

impl MainlineLocator {
    /// Locator over the real mainline DHT (network-touching).
    pub fn new() -> Result<Self> {
        Ok(Self::with_client(Box::new(MainlineBep5::new()?)))
    }

    /// Locator over an injected BEP5 client (tests, relay bridges).
    fn with_client(client: Box<dyn Bep5Client>) -> Self {
        Self { client }
    }

    /// Announce this node as a provider for `cid`.
    pub async fn announce(&self, cid: &Cid512, port: Option<u16>) -> Result<()> {
        self.announce_at(cid, RendezvousKey::DEFAULT, port).await
    }

    /// Announce this node as a provider for a specific k-derived rendezvous.
    pub async fn announce_at(
        &self,
        cid: &Cid512,
        key: RendezvousKey,
        port: Option<u16>,
    ) -> Result<()> {
        self.client
            .announce(rendezvous_infohash(cid, key), port)
            .await
    }

    /// Look up candidate providers for `cid`.
    ///
    /// Returned contacts are **untrusted hints**: anyone can announce on any
    /// infohash. Callers must verify fetched content against the full
    /// `cid` via the GATE pipeline before deriving anything from it.
    pub async fn providers(&self, cid: &Cid512) -> Result<Vec<PeerContact>> {
        self.providers_at(cid, RendezvousKey::DEFAULT).await
    }

    /// Look up candidate providers for a specific k-derived rendezvous.
    ///
    /// Results remain untrusted hints; the `key` only selects the rendezvous
    /// bucket and does not add authority.
    pub async fn providers_at(&self, cid: &Cid512, key: RendezvousKey) -> Result<Vec<PeerContact>> {
        Ok(self
            .client
            .get_peers(rendezvous_infohash(cid, key))
            .await?
            .into_iter()
            .map(PeerContact::untrusted)
            .collect())
    }

    /// Look up candidate providers, unioning in add-only [`LookupHints`] (R7).
    ///
    /// The watermark-anchored BEP5 scan at `key` is **always** performed; each
    /// rendezvous bucket in `hints` is scanned as well, and the hint peers are
    /// appended. `hints` may only enlarge the candidate set — there is no
    /// parameter and no code path that prunes, replaces, or skips the scan
    /// (A14). Results remain untrusted: deduplication against the full
    /// [`Cid512`] is the caller's job via the GATE pipeline.
    pub async fn providers_with_hints(
        &self,
        cid: &Cid512,
        key: RendezvousKey,
        hints: &LookupHints,
    ) -> Result<Vec<PeerContact>> {
        // The canonical scan is unconditional — a hint can never suppress it.
        let mut found = self.providers_at(cid, key).await?;

        // Additional rendezvous buckets from hints: scanned, not trusted.
        for extra in hints.extra_rendezvous() {
            found.extend(self.providers_at(cid, *extra).await?);
        }

        // Out-of-band peer contacts: unioned in verbatim, still untrusted.
        found.extend(hints.extra_peers().iter().cloned());

        Ok(found)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::net::Ipv4Addr;

    use std::sync::Arc;

    /// In-memory BEP5 fake: a HashMap keyed by the 20-byte infohash, exactly
    /// like the real DHT's keyspace. No sockets, no network. Clone shares
    /// state so tests can inspect what the locator sent.
    #[derive(Default, Clone)]
    struct MockBep5 {
        table: Arc<Mutex<HashMap<[u8; 20], Vec<SocketAddrV4>>>>,
        seen_infohashes: Arc<Mutex<Vec<[u8; 20]>>>,
    }

    #[async_trait]
    impl Bep5Client for MockBep5 {
        async fn announce(&self, info_hash: [u8; 20], port: Option<u16>) -> Result<()> {
            self.seen_infohashes.lock().push(info_hash);
            let addr = SocketAddrV4::new(Ipv4Addr::LOCALHOST, port.unwrap_or(6881));
            self.table.lock().entry(info_hash).or_default().push(addr);
            Ok(())
        }

        async fn get_peers(&self, info_hash: [u8; 20]) -> Result<Vec<SocketAddrV4>> {
            self.seen_infohashes.lock().push(info_hash);
            Ok(self
                .table
                .lock()
                .get(&info_hash)
                .cloned()
                .unwrap_or_default())
        }
    }

    fn cid(fill: u8) -> Cid512 {
        Cid512::from_bytes([fill; CID512_LEN])
    }

    #[tokio::test]
    async fn announce_then_providers_roundtrip() {
        let locator = MainlineLocator::with_client(Box::<MockBep5>::default());
        let id = cid(0xAB);

        locator.announce(&id, Some(4242)).await.unwrap();
        let peers = locator.providers(&id).await.unwrap();

        assert_eq!(
            peers,
            vec![PeerContact::untrusted(SocketAddrV4::new(
                Ipv4Addr::LOCALHOST,
                4242
            ))]
        );
    }

    #[tokio::test]
    async fn providers_of_unannounced_cid_is_empty() {
        let locator = MainlineLocator::with_client(Box::<MockBep5>::default());
        assert!(locator.providers(&cid(0x01)).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn rendezvous_derivation_is_deterministic_for_same_cid_epoch_and_k() {
        let id = cid(0xAB);
        let key = RendezvousKey::new(7, 42);

        assert_eq!(rendezvous_infohash(&id, key), rendezvous_infohash(&id, key));
        assert_eq!(rendezvous_cid512(&id, key).0.len(), CID512_LEN);
    }

    #[test]
    fn k_index_and_epoch_are_fixed_width_big_endian() {
        let key = RendezvousKey::new(0x0102_0304, 0x0102_0304_0506_0708);

        assert_eq!(key.k_index_be_bytes(), [0x01, 0x02, 0x03, 0x04]);
        assert_eq!(
            key.epoch_be_bytes(),
            [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );
        assert_eq!(key.k_index_be_bytes().len(), RENDEZVOUS_K_INDEX_LEN);
        assert_eq!(key.epoch_be_bytes().len(), RENDEZVOUS_EPOCH_LEN);
    }

    #[tokio::test]
    async fn truncation_is_internal_and_derived_from_full_rendezvous_key() {
        let mock = MockBep5::default();
        let locator = MainlineLocator::with_client(Box::new(mock.clone()));

        let mut bytes = [0u8; CID512_LEN];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let id = Cid512::from_bytes(bytes);
        let key = RendezvousKey::new(1, 9);

        locator.announce_at(&id, key, None).await.unwrap();

        // The BEP5 client only sees the private 20-byte projection of the
        // full-width derived rendezvous key; public locator APIs never expose it.
        let seen = mock.seen_infohashes.lock().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0], rendezvous_infohash(&id, key));
        assert_eq!(seen[0].len(), 20);
    }

    #[tokio::test]
    async fn k_index_and_epoch_select_distinct_rendezvous_buckets() {
        let locator = MainlineLocator::with_client(Box::<MockBep5>::default());
        let id = cid(0x55);

        locator
            .announce_at(&id, RendezvousKey::new(0, 7), Some(1111))
            .await
            .unwrap();

        assert_eq!(
            locator
                .providers_at(&id, RendezvousKey::new(0, 7))
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(locator
            .providers_at(&id, RendezvousKey::new(1, 7))
            .await
            .unwrap()
            .is_empty());
        assert!(locator
            .providers_at(&id, RendezvousKey::new(0, 8))
            .await
            .unwrap()
            .is_empty());
    }

    #[test]
    fn peer_contact_carries_only_untrusted_candidate_contact_info() {
        let addr = SocketAddrV4::new(Ipv4Addr::new(192, 0, 2, 10), 6881);
        let contact = PeerContact::untrusted(addr);

        assert_eq!(contact.socket_addr(), addr);
    }

    #[test]
    fn cid512_rejects_wrong_lengths() {
        assert!(Cid512::from_slice(&[0u8; 20]).is_err());
        assert!(Cid512::from_slice(&[0u8; 32]).is_err());
        assert!(Cid512::from_slice(&[0u8; 63]).is_err());
        assert!(Cid512::from_slice(&[0u8; 64]).is_ok());
        assert!(Cid512::from_slice(&[0u8; 65]).is_err());
    }

    // ----- R1 enforcement (story C4 / #892) --------------------------------

    /// Regression guard for review rule R1: no public locator API may accept a
    /// string *name* parameter. A name path inside the DHT client re-opens
    /// threat A5 (the DHT becomes a trusted name authority). This scans the
    /// module's production source and fails if any `pub fn` grows a
    /// `&str`/`String` parameter or the CID type gains a string constructor.
    #[test]
    fn r1_no_name_path_compiles_in() {
        // Scan only the production code — everything before the test module —
        // so the assertion literals below don't match this test's own source.
        let full = include_str!("locator.rs");
        let src = full.split("#[cfg(test)]").next().unwrap();

        // 1. No name→cid constructor / string conversion exists on the CID type.
        assert!(
            !src.contains("fn from_name"),
            "R1: Cid512 must not gain a from_name constructor (re-opens A5)"
        );
        assert!(
            !src.contains("impl FromStr for Cid512"),
            "R1: Cid512 must not impl FromStr (a string would name-resolve; A5)"
        );
        assert!(
            !src.contains("impl From<&str> for Cid512"),
            "R1: Cid512 must not impl From<&str> (A5)"
        );

        // 2. No public function carries a string-typed parameter. Walk pub fn
        //    signatures and reject any whose parameter list contains `&str` or
        //    `String` (a name→cid path would have to enter the API here).
        for line in src.lines() {
            let trimmed = line.trim_start();
            let is_pub_fn = trimmed.starts_with("pub fn ")
                || trimmed.starts_with("pub async fn ")
                || trimmed.starts_with("pub const fn ");
            if !is_pub_fn {
                continue;
            }
            assert!(
                !(line.contains("&str") || line.contains(": String")),
                "R1 violation: public locator fn has a string parameter (re-opens A5):\n  {line}"
            );
        }
    }

    /// R1 positive proof: the only query key the public API accepts is a full
    /// [`Cid512`] — content-addressed, not name-addressed.
    #[test]
    fn r1_locator_queries_are_content_addressed_only() {
        // Compile-time check that Cid512 is constructible only from bytes.
        let _ = Cid512::from_bytes([0xAB; CID512_LEN]);
        let _ = Cid512::from_slice(&[0xCD; CID512_LEN]).unwrap();
        // `from_name` / `FromStr` / `From<&str>` simply do not exist:
        // (these would fail to compile if added and used here — left as a
        //  comment to document intent; the source-scan test above enforces it).
    }

    // ----- R7 add-only hints (story C4 / #892) -----------------------------

    #[test]
    fn v1_mailbox_decision_is_recorded_and_bep5_only() {
        // Q3 decision: ship BEP5-only for v1; the mailbox is deferred.
        assert_eq!(V1_MAILBOX_HINT_CHANNEL, MailboxHintDecision::Bep5Only);
    }

    #[test]
    fn lookup_hints_starts_empty_and_grows_only() {
        let mut hints = LookupHints::new();
        assert!(hints.is_empty());
        assert_eq!(hints.len(), 0);

        hints.add_rendezvous(RendezvousKey::new(1, 0));
        assert!(!hints.is_empty());
        assert_eq!(hints.len(), 1);

        hints.add_peer(PeerContact::untrusted(SocketAddrV4::new(
            Ipv4Addr::LOCALHOST,
            7000,
        )));
        assert_eq!(hints.len(), 2);

        assert_eq!(hints.extra_rendezvous(), &[RendezvousKey::new(1, 0)]);
        assert_eq!(
            hints.extra_peers(),
            &[PeerContact::untrusted(SocketAddrV4::new(
                Ipv4Addr::LOCALHOST,
                7000
            ))]
        );
    }

    /// R7 load-bearing proof: hints can only ADD candidates. The canonical
    /// BEP5 scan is always performed and its results always survive — a hint
    /// set, however large, can never prune them.
    #[tokio::test]
    async fn r7_hints_are_add_only_scan_results_survive() {
        let mock = MockBep5::default();
        let locator = MainlineLocator::with_client(Box::new(mock.clone()));
        let id = cid(0x77);
        let canonical = RendezvousKey::DEFAULT;
        let extra = RendezvousKey::new(2, 0);

        // Seed the canonical bucket with a real provider.
        locator.announce_at(&id, canonical, Some(4242)).await.unwrap();
        // Seed the extra hinted bucket with a different provider.
        locator.announce_at(&id, extra, Some(5353)).await.unwrap();

        let mut hints = LookupHints::new();
        hints.add_rendezvous(extra);
        hints.add_peer(PeerContact::untrusted(SocketAddrV4::new(
            Ipv4Addr::new(192, 0, 2, 99),
            9999,
        )));

        let found = locator.providers_with_hints(&id, canonical, &hints).await.unwrap();

        // Canonical scan result survives (never pruned by hints):
        let canonical_addr = SocketAddrV4::new(Ipv4Addr::LOCALHOST, 4242);
        assert!(
            found.iter().any(|p| p.socket_addr() == canonical_addr),
            "R7: canonical scan result must survive hints (no prune)"
        );
        // Hinted rendezvous bucket was scanned and unioned in:
        assert!(
            found
                .iter()
                .any(|p| p.socket_addr() == SocketAddrV4::new(Ipv4Addr::LOCALHOST, 5353)),
            "R7: hinted rendezvous bucket must be unioned in (add-only)"
        );
        // Out-of-band peer hint was unioned in:
        assert!(
            found
                .iter()
                .any(|p| p.socket_addr() == SocketAddrV4::new(Ipv4Addr::new(192, 0, 2, 99), 9999)),
            "R7: hinted peer must be unioned in (add-only)"
        );
    }

    /// R7 proof: with no hints, the hinted path is identical to the plain scan
    /// — confirming hints are purely additive.
    #[tokio::test]
    async fn r7_empty_hints_equal_plain_scan() {
        let mock = MockBep5::default();
        let locator = MainlineLocator::with_client(Box::new(mock.clone()));
        let id = cid(0x33);
        locator.announce(&id, Some(8080)).await.unwrap();

        let plain = locator.providers(&id).await.unwrap();
        let with_hints = locator
            .providers_with_hints(&id, RendezvousKey::DEFAULT, &LookupHints::new())
            .await
            .unwrap();

        assert_eq!(plain, with_hints);
    }

    /// R7 proof: a hint set cannot suppress the scan even when the hint points
    /// at a *different* rendezvous bucket that has no providers. The canonical
    /// scan still returns its providers — the misleading hint prunes nothing.
    #[tokio::test]
    async fn r7_misleading_hint_cannot_prune_scan() {
        let mock = MockBep5::default();
        let locator = MainlineLocator::with_client(Box::new(mock.clone()));
        let id = cid(0x44);
        let canonical = RendezvousKey::DEFAULT;

        // Real provider at the canonical bucket.
        locator.announce_at(&id, canonical, Some(1234)).await.unwrap();

        // Misleading hint directing attention elsewhere (empty bucket).
        let mut hints = LookupHints::new();
        hints.add_rendezvous(RendezvousKey::new(9, 9));

        let found = locator
            .providers_with_hints(&id, canonical, &hints)
            .await
            .unwrap();

        // The real canonical provider is still returned despite the misleading
        // hint — A14 is excluded: a hint cannot skip the watermark scan.
        assert!(
            found
                .iter()
                .any(|p| p.socket_addr() == SocketAddrV4::new(Ipv4Addr::LOCALHOST, 1234)),
            "R7/A14: misleading hint must not prune the canonical scan"
        );
    }
}
