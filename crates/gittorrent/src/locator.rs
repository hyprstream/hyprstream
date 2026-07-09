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
}
