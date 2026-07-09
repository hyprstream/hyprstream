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
//! the full [`Cid512`]; no truncated form escapes this module. Story C2
//! (#890) will replace the raw prefix with the k-derived rendezvous
//! (`k-index` + `epoch-be64`, #879 §5.2) — the confinement point stays here.
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

/// TRUNC20: derive the 20-byte BEP5 infohash for a full-width CID.
///
/// **This is the single truncation site in the codebase** (R2). The result
/// is passed straight into the BEP5 client and is never returned to callers.
///
/// C1 uses the raw 20-byte prefix; C2 (#890) swaps this for the k-derived
/// rendezvous hash (`k-index` + `epoch-be64`) without widening its scope.
fn rendezvous_infohash(cid: &Cid512) -> [u8; 20] {
    let mut infohash = [0u8; 20];
    infohash.copy_from_slice(&cid.0[..20]);
    infohash
}

/// Minimal BEP5 operations the locator needs, abstracted so unit tests never
/// touch the real DHT (and so C2+ can inject scoring/relay variants).
#[async_trait]
pub trait Bep5Client: Send + Sync {
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
/// Thin C1 wrapper — provider scoring, parallel fetch, and size caps are C3
/// (#891); the k-derived rendezvous is C2 (#890).
pub struct MainlineLocator {
    client: Box<dyn Bep5Client>,
}

impl MainlineLocator {
    /// Locator over the real mainline DHT (network-touching).
    pub fn new() -> Result<Self> {
        Ok(Self::with_client(Box::new(MainlineBep5::new()?)))
    }

    /// Locator over an injected BEP5 client (tests, relay bridges).
    pub fn with_client(client: Box<dyn Bep5Client>) -> Self {
        Self { client }
    }

    /// Announce this node as a provider for `cid`.
    pub async fn announce(&self, cid: &Cid512, port: Option<u16>) -> Result<()> {
        self.client.announce(rendezvous_infohash(cid), port).await
    }

    /// Look up candidate providers for `cid`.
    ///
    /// Returned contacts are **untrusted hints**: anyone can announce on any
    /// infohash. Callers must verify fetched content against the full
    /// `cid` via the GATE pipeline before deriving anything from it.
    pub async fn providers(&self, cid: &Cid512) -> Result<Vec<SocketAddrV4>> {
        self.client.get_peers(rendezvous_infohash(cid)).await
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

        assert_eq!(peers, vec![SocketAddrV4::new(Ipv4Addr::LOCALHOST, 4242)]);
    }

    #[tokio::test]
    async fn providers_of_unannounced_cid_is_empty() {
        let locator = MainlineLocator::with_client(Box::<MockBep5>::default());
        assert!(locator.providers(&cid(0x01)).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn truncation_is_internal_and_prefix_derived() {
        let mock = MockBep5::default();
        let locator = MainlineLocator::with_client(Box::new(mock.clone()));

        let mut bytes = [0u8; CID512_LEN];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = i as u8;
        }
        let id = Cid512::from_bytes(bytes);

        locator.announce(&id, None).await.unwrap();

        // The infohash the client saw must be exactly the first 20 bytes of
        // the full-width CID — and nothing wider ever reaches the client.
        let seen = mock.seen_infohashes.lock().clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0][..], bytes[..20]);
    }

    #[tokio::test]
    async fn distinct_cids_sharing_a_20_byte_prefix_collide_on_the_dht() {
        // Documents the TRUNC20 residual: the DHT rendezvous cannot separate
        // CIDs that agree on the derived infohash — the full-width GATE
        // verification after fetch is what restores 512-bit security (#879 §5.2).
        let locator = MainlineLocator::with_client(Box::<MockBep5>::default());

        let mut a = [0x55u8; CID512_LEN];
        let mut b = [0x55u8; CID512_LEN];
        a[63] = 0x00;
        b[63] = 0xFF;
        let (a, b) = (Cid512::from_bytes(a), Cid512::from_bytes(b));
        assert_ne!(a, b);

        locator.announce(&a, Some(1111)).await.unwrap();
        let peers = locator.providers(&b).await.unwrap();
        assert_eq!(peers.len(), 1, "same 20-byte prefix rendezvouses together");
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
