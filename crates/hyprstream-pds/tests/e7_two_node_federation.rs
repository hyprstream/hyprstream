//! E7 (#421) two-node federation harness — publish → resolve → verify → fetch,
//! over a REAL Iroh loopback transport, CPU-only, no second machine.
//!
//! Where E1 (`e2e_content_addressing.rs`) validated the federation spine
//! *in-process* on a single node, E7 stands up **two independent hyprstream
//! nodes on one host** and drives the full lifecycle across the federation
//! transport boundary:
//!
//! ```text
//!   Node A (publisher)                         Node B (resolver)
//!   ------------------                         -----------------
//!   1. register model repo (git2db) -> OID
//!   2. encode OID as git-raw CIDv1 string
//!   3. build ai.hyprstream.model record
//!   4. MST + signed commit (node-A key)
//!   5. emit CAR proof + raw content blob
//!         |  (serve over Iroh, ALPN hyprstream-rpc/1)
//!         +---------------- Iroh loopback --------------->
//!                                                6. dial Node A by EndpointAddr
//!                                                7. fetch CAR proof bytes
//!                                                8. verify_record_proof  (D5)
//!                                                9. extract currentOid, fetch
//!                                                   content blob over Iroh
//!                                               10. content hash == OID  (D5)
//! ```
//!
//! Each node has a DISTINCT identity: its own random Ed25519 node key (→ its own
//! `did:key`, via `hyprstream_rpc::did_web::ed25519_to_did_key`), its own
//! `IrohSubstrate` endpoint, its own TempDir-backed git2db registry, and its own
//! P-256 `#atproto` signing key. Node B trusts NOTHING it didn't verify: it
//! re-derives the record CID, re-walks the MST proof, checks the ES256 commit
//! signature against Node A's published verifying key, and re-hashes the fetched
//! content against the OID carried in the record. That is the D5 untrusted-host
//! posture exercised over a genuine network seam.
//!
//! ## What is REAL here vs. what is the transport stand-in
//!
//! REAL (production code paths, unchanged):
//!   - git2db register + commit → OID                        (step 1)
//!   - hyprstream_rpc::cid::encode_git_oid                    (step 2)
//!   - hyprstream_pds record/MST/commit/CAR + verify_record_proof (steps 3–8,10)
//!   - hyprstream_rpc::did_web::ed25519_to_did_key            (node identity)
//!   - hyprstream_rpc IrohSubstrate two-plane endpoint + dial (the transport)
//!   - the CAR proof + content blob travel over a real Iroh QUIC bidi stream
//!
//! STAND-IN (deliberately minimal, because the typed wire is not built yet):
//!   - the request/response framing on the `hyprstream-rpc/1` ALPN is a tiny
//!     hand-rolled `GET <kind> <key>` line + length-prefixed reply, NOT the
//!     generated Cap'n Proto `getRecord` / `getBlob` RPC. See the MISSING-WIRE
//!     note at the bottom of this file: the PDS record store (#392) is not yet
//!     wired into an RPC service, and there is no CAS fetch-by-OID RPC. Those two
//!     typed methods are the next ticket; this harness proves everything *around*
//!     them is correct so that wiring them is mechanical.
//!
//! CPU-only: git is `file://`, all crypto is in-process, Iroh dials direct by
//! `EndpointAddr` (no DNS / pkarr / relay egress).

#![allow(
    // Integration tests favor unwrap/expect/indexing for readability; the
    // library proper still denies these under `[lints] workspace = true`.
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::collections::BTreeMap;
use std::fs;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use ed25519_dalek::SigningKey as NodeSigningKey;
use git2db::config::RepositoryConfig;
use git2db::{Git2DB, Git2DBConfig, GitManager};
use p256::ecdsa::{SigningKey, VerifyingKey};
use tempfile::TempDir;

use hyprstream_pds::car::{build_record_proof_car, parse_car_v1, verify_record_proof};
use hyprstream_pds::cid::Cid as PdsCid;
use hyprstream_pds::commit::{Commit, UnsignedCommit};
use hyprstream_pds::mst::{Node, Proof};
use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
use hyprstream_pds::tid::Tid;

use hyprstream_rpc::cid::{decode_cid, Codec, HashAlgo};
use hyprstream_rpc::did_web::ed25519_to_did_key;
use hyprstream_rpc::transport::iroh_substrate::{
    IrohSubstrate, NoopHandler, ALPN_HYPRSTREAM_RPC,
};

use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use iroh::{EndpointAddr, TransportAddr};

// ─────────────────────────────────────────────────────────────────────────────
// git2db harness (mirrors E1's create/register helpers)
// ─────────────────────────────────────────────────────────────────────────────

/// `file://` URLs don't support shallow clone, so disable it on the global
/// `GitManager` that `Git2DB::open` uses. Idempotent across the process.
fn init_git_manager_no_shallow() {
    let mut config = Git2DBConfig::default();
    config.repository = RepositoryConfig {
        prefer_shallow: false,
        shallow_depth: None,
        auto_init: true,
        auto_init_submodules: false,
    };
    let _ = GitManager::init_with_config(config);
}

/// Seed a real git repo with one commit, returning its head `Oid`. This is the
/// "model repo" stand-in — in production it would carry weights under git-xet.
async fn create_model_repo(path: &std::path::Path) -> Result<git2::Oid> {
    use git2::{Repository, Signature};
    let repo = Repository::init(path)?;
    let sig = Signature::now("E7 Federation", "e7@hyprstream.ai")?;
    fs::write(path.join("README.md"), "# qwen3 model repo (node A)\n")?;
    fs::write(path.join("config.json"), r#"{"arch":"qwen3","dim":4096}"#)?;
    let tree_id = {
        let mut index = repo.index()?;
        index.add_path(std::path::Path::new("README.md"))?;
        index.add_path(std::path::Path::new("config.json"))?;
        index.write()?;
        index.write_tree()?
    };
    let tree = repo.find_tree(tree_id)?;
    let oid = repo.commit(Some("HEAD"), &sig, &sig, "Initial model import", &tree, &[])?;
    Ok(oid)
}

/// Register `source_path` into `registry` and make a tracked commit through the
/// git2db handle, so the OID we content-address is produced by git2db's own
/// commit path. Returns `(head_oid, raw_commit_object_bytes)`.
///
/// The raw commit object bytes are what a peer would fetch to verify the OID by
/// content hash: `sha1(git_object_bytes) == oid`. We read them out of the repo's
/// object database via libgit2 (`odb().read(oid)`), the same bytes any git
/// transport (gittorrent / cas-serve) would ship.
async fn register_commit_and_read_object(
    registry: &mut Git2DB,
    registry_root: &std::path::Path,
    repo_name: &str,
    source_path: &std::path::Path,
) -> Result<(git2::Oid, Vec<u8>)> {
    let url = format!("file://{}", source_path.display());
    let repo_id = registry.add_repository(repo_name, &url).await?;
    let handle = registry.repo(&repo_id)?;

    let worktree = handle.worktree()?;
    fs::write(
        worktree.join("WEIGHTS.bin"),
        b"\x00\x01\x02qwen3-weights-stub-node-A\x03\x04",
    )?;
    handle.staging().add("WEIGHTS.bin").await?;
    let oid = handle.commit("Add model weights (E7 fixture)").await?;

    // Live HEAD must match (git2db really advanced the ref, not just returned a value).
    let registry_root_abs = registry_root.canonicalize()?;
    assert!(
        handle.worktree()?.starts_with(&registry_root_abs),
        "git2db repo worktree should live under the registry root"
    );
    let repo = handle.open_repo()?;
    let head_oid = repo
        .head()?
        .target()
        .ok_or_else(|| anyhow!("git2db commit left HEAD unborn"))?;
    assert_eq!(head_oid, oid, "git2db commit OID must equal live HEAD");

    // Read the raw commit object bytes from the object database — the content a
    // peer hashes to confirm the OID. (libgit2 `odb.read` returns the inflated
    // object payload; the git OID is sha1("commit <len>\0" + payload). We ship
    // the payload and let the verifier reconstruct the framed hash.)
    let odb = repo.odb()?;
    let obj = odb.read(oid)?;
    let raw = obj.data().to_vec();
    Ok((oid, raw))
}

/// Recompute a git object's OID from its inflated payload + type, the way git
/// itself addresses objects: `sha1("<type> <len>\0" <payload>)`. `git2`'s
/// `Oid::hash_object` applies exactly that framing + sha1, so this is the same
/// content-addressing function git uses. This is the integrity check Node B runs
/// against the OID it pulled from the (verified) PDS record.
fn git_oid_of_object(obj_type: git2::ObjectType, payload: &[u8]) -> Result<git2::Oid> {
    git2::Oid::hash_object(obj_type, payload).context("git2 hash_object")
}

// ─────────────────────────────────────────────────────────────────────────────
// A node's PUBLISHED federation artifacts (what crosses the wire)
// ─────────────────────────────────────────────────────────────────────────────

/// Everything Node A makes available to peers for one model record. In a wired
/// system the CAR comes from a `getRecord` RPC and the content from a `getBlob` /
/// CAS-by-OID RPC; here they are bytes served over a raw Iroh bidi stream.
#[derive(Clone, Debug)]
struct PublishedModel {
    /// The federated address peers resolve: at://<did>/ai.hyprstream.model/<rkey>.
    at_uri: String,
    /// CAR proof bytes: commit + MST path + record block (what `getRecord` returns).
    /// These are the bytes that travel over Iroh; Node B parses them to confirm
    /// the transport delivered the commit + record blocks intact.
    car: Vec<u8>,
    /// The record's own CID (so the resolver can pick it out of the CAR blocks).
    record_cid: PdsCid,
    /// The MST inclusion proof the host claims for this record. Carried as part
    /// of the host's claim; `verify_record_proof` cryptographically validates it
    /// (each step's node CID must chain from the signed `commit.data` root), so a
    /// lying host cannot forge a proof that verifies against an unsigned record.
    /// (A self-contained CAR→proof rebuild helper does not exist in the public
    /// MST API yet — see the MISSING-WIRE note; the proof is the host's claim and
    /// the verifier re-checks it end to end.)
    proof: Proof,
    /// Node A's published `#atproto` P-256 verifying key (peers learn this from
    /// the DID document; here we hand it over out-of-band, as the DID-doc fetch
    /// is orthogonal to the federation lifecycle under test).
    atproto_vk: VerifyingKey,
    /// The raw git commit-object payload addressed by the record's currentOid
    /// (what a `getBlob`/CAS-by-OID RPC would return). Keyed by OID hex.
    content_by_oid: BTreeMap<String, (git2::ObjectType, Vec<u8>)>,
}

/// One independent in-process hyprstream node: distinct node key → DID, distinct
/// git2db registry + PDS signing key, and (for the publisher) its IrohSubstrate.
struct FederationNode {
    /// Ed25519 node key → did:key identity + Iroh endpoint key.
    node_key: [u8; 32],
    /// did:key derived from the node key (the node's federation identity).
    did: String,
    /// Per-node temp root holding the git2db registry (kept alive for the test).
    _temp: TempDir,
}

impl FederationNode {
    /// Build a node with a fresh random identity and an empty git2db registry.
    async fn new(label: &str) -> Result<(Self, Git2DB, std::path::PathBuf)> {
        let mut node_key = [0u8; 32];
        // ed25519-dalek's SigningKey::generate gives uniformly-random key bytes.
        let sk = NodeSigningKey::generate(&mut rand::rngs::OsRng);
        node_key.copy_from_slice(sk.as_bytes());
        let did = ed25519_to_did_key(&node_key);

        let temp = TempDir::new()?;
        let registry_root = temp.path().join(format!("{label}-registry"));
        fs::create_dir_all(&registry_root)?;
        let registry = Git2DB::open(&registry_root).await?;
        Ok((
            FederationNode {
                node_key,
                did,
                _temp: temp,
            },
            registry,
            registry_root,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Iroh transport: a minimal "PDS proof" protocol handler on ALPN hyprstream-rpc/1
// ─────────────────────────────────────────────────────────────────────────────
//
// Wire framing (intentionally tiny — see MISSING-WIRE note):
//   request:  one bidi stream carrying a UTF-8 line "GET CAR <at_uri>" or
//             "GET BLOB <oid_hex>", terminated by FIN.
//   response: the requested bytes, terminated by FIN. An unknown key replies
//             with the 5-byte sentinel b"ERR\0\0".
//
// This is the smallest thing that proves the bytes cross a real Iroh QUIC bidi
// stream between two endpoints; the production handler would terminate the
// generated Cap'n Proto service instead.

const ERR_SENTINEL: &[u8] = b"ERR\0\0";
const MAX_REPLY: usize = 16 * 1024 * 1024; // 16 MiB ceiling for the test reader.

#[derive(Clone, Debug)]
struct PdsProofHandler {
    published: Arc<PublishedModel>,
}

impl ProtocolHandler for PdsProofHandler {
    async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
        // Serve every bidi stream the peer opens until it closes the connection.
        loop {
            let (mut send, mut recv) = match conn.accept_bi().await {
                Ok(s) => s,
                // Peer closed the connection — clean end of service.
                Err(_) => return Ok(()),
            };
            let req = recv
                .read_to_end(4096)
                .await
                .map_err(AcceptError::from_err)?;
            let reply = self.respond(&req);
            send.write_all(&reply).await.map_err(AcceptError::from_err)?;
            send.finish().map_err(AcceptError::from_err)?;
            let _ = send.stopped().await;
        }
    }
}

impl PdsProofHandler {
    fn respond(&self, req: &[u8]) -> Vec<u8> {
        let line = match std::str::from_utf8(req) {
            Ok(s) => s.trim(),
            Err(_) => return ERR_SENTINEL.to_vec(),
        };
        let mut parts = line.splitn(3, ' ');
        match (parts.next(), parts.next(), parts.next()) {
            (Some("GET"), Some("CAR"), Some(at_uri)) if at_uri == self.published.at_uri => {
                self.published.car.clone()
            }
            (Some("GET"), Some("BLOB"), Some(oid_hex)) => self
                .published
                .content_by_oid
                .get(oid_hex)
                .map(|(_ty, bytes)| bytes.clone())
                .unwrap_or_else(|| ERR_SENTINEL.to_vec()),
            _ => ERR_SENTINEL.to_vec(),
        }
    }
}

/// `EndpointAddr` for a substrate from its bound sockets (direct dial — no DNS).
fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
    EndpointAddr::from_parts(
        substrate.endpoint_id(),
        substrate
            .endpoint()
            .bound_sockets()
            .into_iter()
            .map(TransportAddr::Ip),
    )
}

/// Client-side fetch: open one bidi stream, send `line`, read the reply to FIN.
async fn fetch_over_iroh(client: &IrohSubstrate, addr: &EndpointAddr, line: &str) -> Result<Vec<u8>> {
    let conn = client
        .connect(addr.clone(), ALPN_HYPRSTREAM_RPC)
        .await
        .context("dial Node A over Iroh")?;
    let (mut send, mut recv) = conn.open_bi().await?;
    send.write_all(line.as_bytes()).await?;
    send.finish()?;
    let reply = recv.read_to_end(MAX_REPLY).await?;
    if reply == ERR_SENTINEL {
        bail!("Node A returned ERR for request {line:?}");
    }
    Ok(reply)
}

// ─────────────────────────────────────────────────────────────────────────────
// Node A: build the published federation artifacts (steps 1–5)
// ─────────────────────────────────────────────────────────────────────────────

/// Run Node A's publish lifecycle and return the artifacts it would serve, plus
/// the random P-256 signing key (kept by the caller only to prove negative cases
/// can't forge it; the public verifying key travels inside `PublishedModel`).
async fn node_a_publish(
    node: &FederationNode,
    registry: &mut Git2DB,
    registry_root: &std::path::Path,
) -> Result<PublishedModel> {
    // ── Step 1: register a model repo → real git commit OID ──────────────────
    let source_repo = registry_root.parent().unwrap().join("qwen3-source");
    fs::create_dir_all(&source_repo)?;
    let _seed = create_model_repo(&source_repo).await?;
    let (head_oid, commit_obj_payload) =
        register_commit_and_read_object(registry, registry_root, "qwen3", &source_repo)
            .await
            .context("node A step 1: git2db register + commit")?;
    let oid_hex = head_oid.to_string();
    assert_eq!(oid_hex.len(), 40, "sha1 git OID is 40 hex chars");

    // ── Step 2: encode the OID as a git-raw CIDv1 string ─────────────────────
    let current_oid_cid =
        hyprstream_rpc::cid::encode_git_oid(&oid_hex).context("node A step 2: encode_git_oid")?;

    // ── Step 3: build the ai.hyprstream.model record (3 fields) ──────────────
    // The record's `repo` at-uri is the node's OWN did:key identity — this is
    // what makes the two nodes' records independently addressable.
    let repo_at_uri = format!("at://{}", node.did);
    let created_at = "2026-06-24T00:00:00.000Z";
    let record = ModelRecord::new(&repo_at_uri, &current_oid_cid, created_at)
        .context("node A step 3: ModelRecord::new")?;

    // ── Step 4: MST (with siblings) → signed commit over the root ────────────
    let mut records: BTreeMap<Tid, ModelRecord> = BTreeMap::new();
    let mut record_cids: BTreeMap<Tid, PdsCid> = BTreeMap::new();
    let base = 1_700_000_000_000_000_u64;
    let siblings = [
        ("bafyreiexampleoidSIBLING00000000a", 20u16),
        (current_oid_cid.as_str(), 21),
        ("bafyreiexampleoidSIBLING00000000c", 22),
    ];
    for (i, (sib, clk)) in siblings.iter().enumerate() {
        let rec = if i == 1 {
            record.clone()
        } else {
            ModelRecord::new(&repo_at_uri, *sib, created_at)?
        };
        let tid = Tid::from_micros(base + i as u64 * 1000, *clk);
        record_cids.insert(tid, rec.cid());
        records.insert(tid, rec);
    }
    let tree = Node::from_records(COLLECTION_NSID, &record_cids);
    let root_cid = tree.root_cid();

    let target_tid = *records
        .iter()
        .find(|(_, r)| r.current_oid == current_oid_cid)
        .map(|(t, _)| t)
        .ok_or_else(|| anyhow!("target record not in tree"))?;
    let target_record = records.get(&target_tid).cloned().unwrap();
    let target_cid = record_cids[&target_tid];

    // Sign the commit with the node's P-256 #atproto key. `commit.did` is bound
    // to the node's did:key, tying the signed pointer to the node identity.
    let signing_key = SigningKey::random(&mut rand::rngs::OsRng);
    let atproto_vk = VerifyingKey::from(&signing_key);
    let unsigned = UnsignedCommit::new(node.did.clone(), root_cid, Tid::now(), None);
    let commit = Commit::sign(&unsigned, &signing_key);
    commit.verify(&atproto_vk).context("node A: own commit must verify")?;

    // ── Step 5: emit the CAR proof for the target record ─────────────────────
    let proof = tree
        .proof(COLLECTION_NSID, &target_tid)
        .ok_or_else(|| anyhow!("no MST proof for target record"))?;
    let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
    let car = build_record_proof_car(&commit, &proof, &node_blocks, &target_record);

    // Index the raw content by OID hex for the BLOB fetch path.
    let mut content_by_oid = BTreeMap::new();
    content_by_oid.insert(
        oid_hex.clone(),
        (git2::ObjectType::Commit, commit_obj_payload),
    );

    // The federated at-uri peers resolve.
    let rkey = target_tid.encode();
    let at_uri = format!("at://{}/{}/{}", node.did, COLLECTION_NSID, rkey);

    Ok(PublishedModel {
        at_uri,
        car,
        record_cid: target_cid,
        proof,
        atproto_vk,
        content_by_oid,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Node B: resolve → verify → fetch → content-hash (steps 6–10)
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve + verify a fetched CAR against the host's claimed inclusion `proof`.
/// This is the D5 untrusted-host check: nothing is trusted until the commit
/// signature + MST path + record CID all check out.
///
/// The fetched CAR (which crossed the Iroh wire) is parsed to confirm the
/// transport delivered the commit + record blocks, and the record is decoded
/// FROM the CAR (not from any out-of-band copy) so a tampered CAR would surface
/// here. The `proof` is the host's claim; `verify_record_proof` re-walks it and
/// re-checks the ES256 commit signature, so the host cannot lie its way past it.
fn resolve_and_verify(
    car: &[u8],
    record_cid: &PdsCid,
    proof: &Proof,
    atproto_vk: &VerifyingKey,
) -> Result<ModelRecord> {
    let (roots, blocks) = parse_car_v1(car).context("Node B: parse CAR")?;
    let commit_cid = *roots.first().ok_or_else(|| anyhow!("CAR has no root"))?;
    let mut commit: Option<Commit> = None;
    let mut record: Option<ModelRecord> = None;
    for (cid, bytes) in &blocks {
        if *cid == commit_cid {
            commit = Some(Commit::from_dag_cbor(bytes).context("decode commit block")?);
        } else if *cid == *record_cid {
            record = Some(ModelRecord::from_dag_cbor(bytes).context("decode record block")?);
        }
    }
    let commit = commit.ok_or_else(|| anyhow!("CAR missing commit block"))?;
    let record = record.ok_or_else(|| anyhow!("CAR missing record block"))?;

    // D5: verify commit signature + MST path + record CID, all at once. The
    // record passed in came out of the fetched CAR, so this also catches a CAR
    // whose record block was tampered (its CID would no longer match the path).
    verify_record_proof(&commit, atproto_vk, proof, &record)
        .context("Node B: verify_record_proof (D5)")?;
    Ok(record)
}

// ─────────────────────────────────────────────────────────────────────────────
// The end-to-end two-node test (happy path, steps 1–10 over Iroh loopback)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e7_two_node_federation_publish_resolve_verify_fetch() -> Result<()> {
    init_git_manager_no_shallow();

    // ── Two independent nodes, distinct identities ───────────────────────────
    let (node_a, mut reg_a, reg_a_root) = FederationNode::new("nodeA").await?;
    let (node_b, _reg_b, _reg_b_root) = FederationNode::new("nodeB").await?;
    assert_ne!(node_a.node_key, node_b.node_key, "nodes must have distinct keys");
    assert_ne!(node_a.did, node_b.did, "nodes must have distinct DIDs");
    assert!(node_a.did.starts_with("did:key:z"), "did:key form: {}", node_a.did);

    // ── Node A publishes (steps 1–5) ─────────────────────────────────────────
    let published = Arc::new(node_a_publish(&node_a, &mut reg_a, &reg_a_root).await?);
    assert!(
        published.at_uri.contains(&node_a.did),
        "published at-uri must carry Node A's DID"
    );

    // ── Node A serves its artifacts over Iroh (ALPN hyprstream-rpc/1) ─────────
    let server = IrohSubstrate::new(
        node_a.node_key,
        NoopHandler::new("moq-not-wired"),
        PdsProofHandler {
            published: Arc::clone(&published),
        },
    )
    .await?;
    let server_addr = direct_addr(&server);

    // ── Node B is a client-only substrate ────────────────────────────────────
    let client = IrohSubstrate::new(
        node_b.node_key,
        NoopHandler::new("b-moq"),
        NoopHandler::new("b-rpc"),
    )
    .await?;

    // ── Step 6+7: Node B dials Node A and fetches the CAR proof over Iroh ─────
    let car_req = format!("GET CAR {}", published.at_uri);
    let fetched_car = fetch_over_iroh(&client, &server_addr, &car_req).await?;
    assert_eq!(
        fetched_car, published.car,
        "CAR fetched over Iroh must be byte-identical to what Node A served"
    );

    // ── Step 8: Node B verifies the CAR proof (D5) and gets the record ───────
    let record = resolve_and_verify(
        &fetched_car,
        &published.record_cid,
        &published.proof,
        &published.atproto_vk,
    )
    .context("Node B: resolve + verify over the federation boundary")?;
    assert!(
        record.repo.contains(&node_a.did),
        "verified record must be owned by Node A's DID"
    );

    // ── Step 9: extract currentOid, fetch the content blob over Iroh ─────────
    let cid = decode_cid(&record.current_oid).context("decode currentOid CID")?;
    assert_eq!(cid.codec, Codec::GitRaw, "currentOid must be a git-raw CID");
    assert_eq!(cid.multihash.algo, HashAlgo::Sha1, "sha1 git OID");
    // The OID hex is the multihash digest, lowercase hex.
    let oid_hex = hex_encode(&cid.multihash.digest);
    let blob_req = format!("GET BLOB {oid_hex}");
    let content = fetch_over_iroh(&client, &server_addr, &blob_req).await?;
    assert!(!content.is_empty(), "fetched content must be non-empty");

    // ── Step 10: verify the content matches the OID by hash (D5) ─────────────
    let recomputed = git_oid_of_object(git2::ObjectType::Commit, &content)?;
    assert_eq!(
        recomputed.to_string(),
        oid_hex,
        "fetched content's git OID must equal the record's currentOid"
    );

    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// D5 negative tests: a forged/tampered artifact must be REJECTED by Node B
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e7_d5_tampered_record_rejected() -> Result<()> {
    init_git_manager_no_shallow();

    let (node_a, mut reg_a, reg_a_root) = FederationNode::new("nodeA-neg").await?;
    let published = node_a_publish(&node_a, &mut reg_a, &reg_a_root).await?;

    // Decode the honest commit out of the served CAR (the same bytes a peer
    // fetches over Iroh).
    let (roots, blocks) = parse_car_v1(&published.car)?;
    let commit_cid = roots[0];
    let commit = blocks
        .iter()
        .find(|(c, _)| *c == commit_cid)
        .map(|(_, b)| Commit::from_dag_cbor(b))
        .ok_or_else(|| anyhow!("no commit in CAR"))??;

    // (a) Forged record: a malicious host serves a record whose CID is NOT the
    //     value the (honest, signed) MST path addresses. verify_record_proof must
    //     reject it — the host can lie about the record bytes but the proof's
    //     terminal entry value won't match the forged record's CID.
    let forged = ModelRecord::new(
        format!("at://{}", node_a.did),
        "bafyreiexampleoidFORGEDNOTONPATH0",
        "2026-06-24T00:00:00.000Z",
    )?;
    assert!(
        verify_record_proof(&commit, &published.atproto_vk, &published.proof, &forged).is_err(),
        "D5: a forged record (wrong CID) must be rejected by verify_record_proof"
    );

    // (b) Wrong signer: the commit verified against an attacker's key must fail
    //     the ES256 signature check, even with the honest record + proof.
    let honest_record = resolve_and_verify(
        &published.car,
        &published.record_cid,
        &published.proof,
        &published.atproto_vk,
    )?;
    let attacker_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
    assert!(
        verify_record_proof(&commit, &attacker_vk, &published.proof, &honest_record).is_err(),
        "D5: a commit verified against the wrong key must be rejected"
    );

    // (c) Tampered MST root: point the commit at a different `data` root. The
    //     signature no longer covers it, so verification fails.
    let mut tampered = commit.clone();
    tampered.data = PdsCid::from_dag_cbor(b"not the real MST root");
    assert!(
        verify_record_proof(&tampered, &published.atproto_vk, &published.proof, &honest_record)
            .is_err(),
        "D5: a commit whose data (root) was tampered must be rejected"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e7_d5_tampered_content_blob_rejected() -> Result<()> {
    init_git_manager_no_shallow();

    let (node_a, mut reg_a, reg_a_root) = FederationNode::new("nodeA-blob").await?;
    let published = node_a_publish(&node_a, &mut reg_a, &reg_a_root).await?;

    // Node B verifies the record honestly, then a malicious host serves a
    // CORRUPTED content blob for the (correct) OID. The content-hash check must
    // catch it: sha1(framed corrupted bytes) != OID.
    let record = resolve_and_verify(
        &published.car,
        &published.record_cid,
        &published.proof,
        &published.atproto_vk,
    )?;
    let cid = decode_cid(&record.current_oid)?;
    let oid_hex = hex_encode(&cid.multihash.digest);

    let (_ty, honest) = published
        .content_by_oid
        .get(&oid_hex)
        .cloned()
        .ok_or_else(|| anyhow!("no content for OID"))?;
    // Corrupt one byte.
    let mut corrupted = honest.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 0xff;

    let recomputed = git_oid_of_object(git2::ObjectType::Commit, &corrupted)?;
    assert_ne!(
        recomputed.to_string(),
        oid_hex,
        "D5: corrupted content must NOT hash to the record's currentOid"
    );
    // And the honest bytes still match (sanity that the check is real, not always-false).
    let honest_oid = git_oid_of_object(git2::ObjectType::Commit, &honest)?;
    assert_eq!(honest_oid.to_string(), oid_hex, "honest content must match OID");
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// helpers
// ─────────────────────────────────────────────────────────────────────────────

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// MISSING-WIRE note — what the NEXT ticket must build for true federated fetch
// ─────────────────────────────────────────────────────────────────────────────
//
// This harness validates federation CORRECTNESS end to end over a real Iroh
// loopback: distinct node identities (did:key), publish → resolve → D5-verify →
// content-hash, plus forged/tampered rejection. Two typed RPC wires are NOT yet
// built, so this test ships the federation bytes over a hand-rolled bidi framing
// instead of the generated Cap'n Proto service:
//
//   1. PDS record fetch over RPC ("getRecord"). The atproto PDS record store
//      (`hyprstream-pds`, #392) is a pure in-process crypto/metadata layer — by
//      design it has "no networking" (see lib.rs). No crate depends on it, and
//      there is no `com.atproto.repo.getRecord` / `getRepo` RPC method anywhere
//      in `hyprstream-rpc` or the service layer. `services/model.rs` (#392)
//      explicitly logs the federated `at://` attempt and falls through to local
//      resolution because "the record store is not yet wired up".
//        NEEDED: a capnp RPC method on the model/registry service that returns a
//        CAR proof for `at://<did>/ai.hyprstream.model/<rkey>`, terminated by an
//        `IrohRpcProtocolHandler` on ALPN `hyprstream-rpc/1` (the same handler
//        `policy_over_iroh.rs` uses), so a peer calls a typed `getRecord` rather
//        than this test's `GET CAR <at_uri>` line.
//
//   2. Content/weights fetch by OID over RPC ("getBlob" / CAS-by-OID). There is
//      no RPC to fetch a git object or XET shard by OID/CID from a remote node.
//      `cas-serve` exists but is not exposed as a fetch-by-OID RPC, and
//      `services/model.rs` loads weights from LOCAL worktree paths only.
//        NEEDED: a CAS/registry RPC `getBlob(cid) -> bytes` (git-raw OID and
//        xet-xorb/xet-shard CIDs) so a peer streams the content addressed by the
//        verified record's `currentOid`, replacing this test's `GET BLOB <oid>`.
//
// Also absent (smaller): a public MST helper to rebuild a `Proof` from CAR
// blocks alone (`Node::from_blocks` / proof-from-blocks). Today `verify` consumes
// a `Proof` whose `NodeData` blocks are inlined, and `Node::proof` only builds
// from a full `from_records` tree. The CAR already carries those node blocks, so
// a `Proof::from_car(car, record_cid)` constructor would let a thin verifier work
// from the CAR alone; this harness carries the host-supplied `Proof` (which
// `verify_record_proof` still validates cryptographically) as the interim.
//
// DEFERRED-TO-REAL-NETWORK (the 2×MI210 box, not a correctness gap): NAT
// traversal, pkarr/DNS discovery, relay fallback, and real inter-host latency.
// Those are exercised by the same `IrohSubstrate` path; here we dial direct by
// `EndpointAddr` (loopback) to keep the test hermetic.
