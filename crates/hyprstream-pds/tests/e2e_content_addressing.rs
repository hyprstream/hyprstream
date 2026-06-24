//! E1 (#421) end-to-end: content addressing + identity (CPU, no network).
//!
//! Validates the full federated "signed mutable pointer → immutable OID" spine,
//! composing four crates that have unit tests but no cross-crate integration
//! coverage until this test:
//!
//! 1. **git2db** — register a model repo, make a commit, obtain a real git
//!    commit `Oid`. (This is the "register a model repo → git commit OID" step.)
//! 2. **hyprstream-rpc::cid** — encode that hex OID as a canonical CIDv1
//!    `git-raw` base32 string (`encode_git_oid`). This is the form carried by
//!    `ai.hyprstream.model.currentOid` and resolved by federated peers.
//! 3. **hyprstream-pds** — build an `ai.hyprstream.model` record (3 fields:
//!    `repo`, `currentOid`, `createdAt`), DAG-CBOR-encode it deterministically,
//!    insert its CID into an MST, sign a commit over the MST root, and emit a
//!    CAR proof. Then `verify_record_proof` checks the commit signature, the MST
//!    inclusion path, and the record CID — the D5 untrusted-host posture.
//! 4. **ModelRef prefix-dispatch** — `at://did:web:…/ai.hyprstream.model/<rkey>`
//!    routes to the federated branch (resolve via PDS record); `qwen3:main`
//!    routes to the local branch (resolve via local registry). This is the
//!    routing grammar that lets a single `modelRef :Text` field serve both
//!    federated and local model lookups (#395).
//!
//! The test is CPU-only (no libtorch, no network): git operations are against a
//! `file://` tempdir, PDS operations are in-process, and the P-256 signing key
//! is random per run.

#![allow(
    // Integration tests are allowed to use unwrap/expect/indexing for readability;
    // the library proper still denies these under `[lints] workspace = true`.
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing
)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use anyhow::{anyhow, bail, Context, Result};
use git2db::config::RepositoryConfig;
use git2db::{Git2DB, Git2DBConfig, GitManager};
use hyprstream_pds::car::{build_record_proof_car, parse_car_v1, verify_record_proof};
use hyprstream_pds::cid::Cid as PdsCid;
use hyprstream_pds::commit::{Commit, UnsignedCommit};
use hyprstream_pds::mst::Node;
use hyprstream_pds::record::{ModelRecord, COLLECTION_NSID};
use hyprstream_pds::tid::Tid;
use p256::ecdsa::{SigningKey, VerifyingKey};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────────────────
// ModelRef prefix-dispatch (step 6)
// ─────────────────────────────────────────────────────────────────────────────
//
// The `modelRef` field on the wire is a single string (`modelRef :Text` in the
// capnp schema, #395). Its grammar routes resolution to one of two branches:
//
//   federated := "at://" <did> "/ai.hyprstream.model/" <rkey>
//   local     := <name> [ ":" <tag> ]            (e.g. "qwen3:main", "qwen3")
//
// `at://` is the federated prefix: the record lives in a PDS, addressed by DID
// + collection + rkey, and the host must verify a CAR proof before trusting it
// (D5). Anything else is a local ref: resolved against the node's own registry
// (git2db), no federation/proof step. This mirrors how `at://` URIs are
// distinguished from bare handles in atproto/ATProtocol generally.

/// A parsed `modelRef` — the routing decision a resolver makes.
#[derive(Clone, Debug, Eq, PartialEq)]
enum ModelRef {
    /// `at://<did>/ai.hyprstream.model/<rkey>` — resolve via the federated PDS
    /// layer (look up the record, verify its CAR proof, follow `currentOid`).
    Federated { did: String, rkey: String },
    /// `<name>[:<tag>]` — resolve via the local model registry (git2db). `tag`
    /// defaults to `"main"` when absent, matching the OCI/docker convention the
    /// local registry already uses.
    Local { name: String, tag: String },
}

impl ModelRef {
    /// Parse a `modelRef` string into its dispatch branch.
    ///
    /// - `at://…/ai.hyprstream.model/<rkey>` → [`ModelRef::Federated`]
    /// - `<name>` or `<name>:<tag>` → [`ModelRef::Local`] (tag defaults to `main`)
    ///
    /// The federated form requires the `ai.hyprstream.model` collection and a
    /// non-empty DID + rkey; anything malformed falls through to `Local` only if
    /// it lacks the `at://` prefix (a malformed `at://` URI is an error, since
    /// silently treating a truncated federated ref as local would hide a bug).
    fn parse(s: &str) -> Result<Self> {
        if let Some(rest) = s.strip_prefix("at://") {
            // Federated form. Split off the DID (first path segment), then the
            // collection, then the rkey.
            let did = rest
                .split('/')
                .next()
                .filter(|d| !d.is_empty())
                .ok_or_else(|| anyhow!("federated modelRef missing DID: {s:?}"))?;
            let after_did = rest
                .strip_prefix(did)
                .and_then(|r| r.strip_prefix('/'))
                .ok_or_else(|| anyhow!("federated modelRef malformed after DID: {s:?}"))?;
            // Expect "<collection>/<rkey>"; collection must be our NSID.
            let (collection, rkey) = after_did
                .split_once('/')
                .ok_or_else(|| anyhow!("federated modelRef missing rkey: {s:?}"))?;
            if collection != COLLECTION_NSID {
                bail!(
                    "federated modelRef collection {collection:?} is not {COLLECTION_NSID:?}"
                );
            }
            if rkey.is_empty() {
                bail!("federated modelRef rkey is empty: {s:?}");
            }
            Ok(ModelRef::Federated {
                did: did.to_owned(),
                rkey: rkey.to_owned(),
            })
        } else {
            // Local form: "<name>" or "<name>:<tag>". Name must be non-empty and
            // tag-less names get the default `main` tag.
            let (name, tag) = match s.split_once(':') {
                Some((n, t)) if !n.is_empty() && !t.is_empty() => (n.to_owned(), t.to_owned()),
                None if !s.is_empty() => (s.to_owned(), "main".to_owned()),
                _ => bail!("local modelRef must be '<name>' or '<name>:<tag>': {s:?}"),
            };
            Ok(ModelRef::Local { name, tag })
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// git2db harness — mirrors crates/git2db/tests/v2_core_operations.rs
// ─────────────────────────────────────────────────────────────────────────────

/// Disable shallow cloning for the global `GitManager`, which `Git2DB::open`
/// uses. `file://` URLs don't support shallow clones, so the default config
/// (prefer_shallow = true) would fail the clone in `add_repository`.
fn init_git_manager_no_shallow() {
    let mut config = Git2DBConfig::default();
    config.repository = RepositoryConfig {
        prefer_shallow: false,
        shallow_depth: None,
        auto_init: true,
        auto_init_submodules: false,
    };
    // Ignore the "already initialized" error — the global is process-wide.
    let _ = GitManager::init_with_config(config);
}

/// Create a real git repo at `path` with one initial commit (`README.md`),
/// using git2db's re-exported libgit2 binding. Returns the head commit `Oid`.
///
/// This is the "model repo" stand-in: in production it holds model weights under
/// git-xet; here it just needs a real commit OID for the content-addressing
/// pipeline to encode. The OID is produced by libgit2 (the same library git2db
/// uses for every repo it registers), so it is byte-identical to what a
/// registered model repo's `HEAD` would report.
async fn create_model_repo(path: &std::path::Path) -> Result<git2::Oid> {
    use git2::{Repository, Signature};
    let repo = Repository::init(path)?;
    // CI / Nix sandboxes often lack ~/.gitconfig; use an explicit signature.
    let sig = Signature::now("E2E Content Addressing", "e2e@hyprstream.ai")?;
    // Stage a README so the tree is non-empty (a real model repo would have
    // weights + a config blob).
    fs::write(path.join("README.md"), "# qwen3 model repo\n")?;
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

/// Register `source_path` as a git2db repo, then make a second commit through
/// the `RepositoryHandle` so the OID we feed into the PDS record comes from
/// git2db's own commit path (not just raw libgit2). Returns `(repo_name, oid)`.
///
/// This validates step 1 ("register a model repo (git2db) → git commit OID")
/// through the actual git2db registry + handle API.
async fn register_and_commit_via_git2db(
    registry: &mut Git2DB,
    registry_root: &std::path::Path,
    repo_name: &str,
    source_path: &std::path::Path,
) -> Result<git2::Oid> {
    let url = format!("file://{}", source_path.display());
    let repo_id = registry.add_repository(repo_name, &url).await?;
    let handle = registry.repo(&repo_id)?;

    // Make a tracked change through git2db's staging → commit API so the OID is
    // produced by git2db (exercising its commit path, not just libgit2's).
    let worktree = handle.worktree()?;
    fs::write(worktree.join("WEIGHTS.bin"), b"\x00\x01\x02qwen3-weights-stub\x03\x04")?;
    handle.staging().add("WEIGHTS.bin").await?;
    let oid = handle.commit("Add model weights (E2E fixture)").await?;

    // Sanity: the registry knows about this repo by name, the worktree lives
    // under the registry root, and the commit OID matches the repo's live HEAD
    // (i.e. git2db really did advance HEAD, not just return a value). We read
    // HEAD via open_repo() rather than `current_oid()` because the latter reads
    // registry *metadata* (set at registration time), which is not refreshed by
    // `commit()`; the live HEAD is the authoritative check.
    assert_eq!(handle.name()?, Some(repo_name));
    let registry_root_abs = registry_root.canonicalize()?;
    assert!(
        handle.worktree()?.starts_with(&registry_root_abs),
        "git2db repo worktree should live under the registry root"
    );
    let repo = handle.open_repo()?;
    let head = repo.head()?;
    let head_oid = head
        .target()
        .ok_or_else(|| anyhow!("git2db commit left HEAD unborn"))?;
    assert_eq!(
        head_oid, oid,
        "git2db commit OID must equal the repo's live HEAD"
    );
    Ok(oid)
}

// ─────────────────────────────────────────────────────────────────────────────
// The end-to-end test (steps 1–5)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn e2e_content_addressing_record_mst_proof() -> Result<()> {
    init_git_manager_no_shallow();

    let temp = TempDir::new()?;
    let registry_root = temp.path().join("registry");
    fs::create_dir_all(&registry_root)?;
    let mut registry = Git2DB::open(&registry_root).await?;

    // ── Step 1: register a model repo (git2db) → real git commit OID ─────────
    let source_repo = temp.path().join("qwen3-source");
    fs::create_dir_all(&source_repo)?;
    // Seed the source repo (raw libgit2 init commit — establishes HEAD).
    let _seed_oid = create_model_repo(&source_repo).await?;
    // Register via git2db and add a second commit through its handle API. The
    // OID we content-address is the one git2db produced.
    let head_oid = register_and_commit_via_git2db(
        &mut registry,
        &registry_root,
        "qwen3",
        &source_repo,
    )
    .await
    .context("step 1: git2db register + commit")?;
    let oid_hex = head_oid.to_string();
    // A real git OID is 40 hex chars (sha1) or 64 (sha256); libgit2 default is sha1.
    assert_eq!(oid_hex.len(), 40, "git2 commit OID should be 40-char sha1 hex");

    // ── Step 2: encode the OID as a CID (hyprstream-rpc/src/cid.rs) ──────────
    let current_oid_cid = hyprstream_rpc::cid::encode_git_oid(&oid_hex)
        .context("step 2: encode_git_oid")?;
    // Canonical CIDv1 base32 form: leading 'b' multibase prefix.
    assert!(
        current_oid_cid.starts_with('b'),
        "CIDv1 canonical base32 must start with 'b': {current_oid_cid}"
    );
    // Round-trips through decode_cid: the codec is git-raw and the multihash
    // algorithm matches the OID length (sha1 for 40-hex / 20-byte digests).
    let decoded = hyprstream_rpc::cid::decode_cid(&current_oid_cid)?;
    assert_eq!(decoded.codec, hyprstream_rpc::cid::Codec::GitRaw);
    assert_eq!(decoded.multihash.algo, hyprstream_rpc::cid::HashAlgo::Sha1);
    // The multihash digest is the raw OID bytes (big-endian sha1).
    let oid_bytes = head_oid.as_bytes();
    assert_eq!(decoded.multihash.digest, oid_bytes);
    // Determinism: encoding the same OID twice yields the same string.
    let again = hyprstream_rpc::cid::encode_git_oid(&oid_hex)?;
    assert_eq!(current_oid_cid, again, "encode_git_oid must be deterministic");

    // ── Step 3: build an `ai.hyprstream.model` record (3 fields) ─────────────
    let did = "did:web:alice.hyprstream.ai";
    let repo_at_uri = format!("at://{did}");
    let created_at = "2026-06-23T12:34:56.789Z";
    let record = ModelRecord::new(&repo_at_uri, &current_oid_cid, created_at)
        .context("step 3: ModelRecord::new")?;
    // The 3 lexicon fields, in this order:
    assert_eq!(record.repo, repo_at_uri);
    assert_eq!(record.current_oid, current_oid_cid);
    assert_eq!(record.created_at, created_at);

    // ── Step 4: DAG-CBOR encode → CID → MST → signed commit → CAR proof ─────
    // Deterministic DAG-CBOR: same record → same bytes → same CID.
    let record_cid = record.cid();
    let record_bytes = record.to_dag_cbor();
    let record_again = ModelRecord::from_dag_cbor(&record_bytes)?;
    assert_eq!(record, record_again);
    assert_eq!(record_again.cid(), record_cid, "re-decoded record CID must match");

    // Insert the record's CID into an MST keyed by its TID rkey. We add a couple
    // of sibling records so the tree has >1 node (exercises the proof descent +
    // shared-prefix compression rather than a degenerate single-leaf tree).
    let mut records: BTreeMap<Tid, ModelRecord> = BTreeMap::new();
    let mut record_cids: BTreeMap<Tid, PdsCid> = BTreeMap::new();
    let base_micros = 1_700_000_000_000_000_u64;
    let siblings = [
        ("bafyreiexampleoidSIBLING00000000a", 10u16),
        (current_oid_cid.as_str(), 11),
        ("bafyreiexampleoidSIBLING00000000c", 12),
    ];
    for (i, (sib_oid, clk)) in siblings.iter().enumerate() {
        // The middle sibling is the real record (its currentOid is the git OID CID).
        let rec = if i == 1 {
            record.clone()
        } else {
            ModelRecord::new(&repo_at_uri, *sib_oid, created_at)?
        };
        let tid = Tid::from_micros(base_micros + i as u64 * 1000, *clk);
        record_cids.insert(tid, rec.cid());
        records.insert(tid, rec);
    }
    let tree = Node::from_records(COLLECTION_NSID, &record_cids);
    let root_cid = tree.root_cid();
    // The tree must contain our target record; find its TID (the middle one).
    let target_tid = records
        .iter()
        .find(|(_, r)| r.current_oid == current_oid_cid)
        .map(|(t, _)| *t)
        .ok_or_else(|| anyhow!("target record not in tree"))?;
    let target_record = records.get(&target_tid).cloned().unwrap();
    let target_cid = record_cids.get(&target_tid).copied().unwrap();
    assert_eq!(target_record.cid(), target_cid);

    // Sign a commit over the MST root with a P-256 (#atproto) key.
    let signing_key = SigningKey::random(&mut rand::rngs::OsRng);
    let verifying_key = VerifyingKey::from(&signing_key);
    let unsigned = UnsignedCommit::new(did, root_cid, Tid::now(), None);
    let commit = Commit::sign(&unsigned, &signing_key);
    // The signed commit verifies under the right key and fails under another.
    commit.verify(&verifying_key).context("commit must verify")?;
    let other_vk = VerifyingKey::from(&SigningKey::random(&mut rand::rngs::OsRng));
    assert!(commit.verify(&other_vk).is_err(), "commit must reject a wrong key");

    // MST inclusion proof: from the commit's `data` root down to the record.
    let proof = tree
        .proof(COLLECTION_NSID, &target_tid)
        .ok_or_else(|| anyhow!("no MST proof for target record"))?;
    proof.verify(&root_cid, &target_cid).context("MST proof must verify")?;

    // Build a CAR proof (commit + path nodes + record) and parse it back.
    let (_root_data, node_blocks) = tree.to_node_data_with_blocks();
    let car = build_record_proof_car(&commit, &proof, &node_blocks, &target_record);
    let (car_roots, car_blocks) = parse_car_v1(&car)?;
    assert_eq!(car_roots, vec![commit.cid()], "CAR root must be the commit CID");
    let car_block_cids: BTreeSet<PdsCid> = car_blocks.iter().map(|(c, _)| *c).collect();
    assert!(car_block_cids.contains(&commit.cid()), "CAR must contain commit block");
    assert!(car_block_cids.contains(&target_cid), "CAR must contain record block");

    // ── Step 5: verifyRecordProof + tamper detection ─────────────────────────
    // Happy path: commit sig + MST path + record CID all check out.
    verify_record_proof(&commit, &verifying_key, &proof, &target_record)
        .context("step 5: verify_record_proof happy path")?;

    // Tamper (a): wrong record — a different record's CID isn't addressed by the
    // path's terminal entry, so the proof must reject it.
    let wrong_record = ModelRecord::new(
        &repo_at_uri,
        "bafyreiexampleoidNOTADDRESSED0000",
        created_at,
    )?;
    assert!(
        verify_record_proof(&commit, &verifying_key, &proof, &wrong_record).is_err(),
        "verify_record_proof must reject a wrong record (CID not on the path)"
    );

    // Tamper (b): tampered commit root — point the commit at a different MST root.
    // The signature no longer covers the tampered `data`, so commit.verify fails
    // inside verify_record_proof.
    let mut tampered_commit = commit.clone();
    tampered_commit.data = PdsCid::from_dag_cbor(b"not the real MST root");
    assert!(
        verify_record_proof(&tampered_commit, &verifying_key, &proof, &target_record).is_err(),
        "verify_record_proof must reject a commit whose data (root) was tampered"
    );

    // Tamper (c): wrong verifying key — the commit was signed by a different key.
    assert!(
        verify_record_proof(&commit, &other_vk, &proof, &target_record).is_err(),
        "verify_record_proof must reject a commit signed by a different key"
    );

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelRef prefix-dispatch (step 6)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn model_ref_prefix_dispatch() {
    // Federated form: at://<did>/ai.hyprstream.model/<rkey> → Federated.
    let fed = ModelRef::parse("at://did:web:alice.hyprstream.ai/ai.hyprstream.model/qwen3")
        .expect("federated ref parses");
    assert_eq!(
        fed,
        ModelRef::Federated {
            did: "did:web:alice.hyprstream.ai".to_owned(),
            rkey: "qwen3".to_owned(),
        }
    );

    // A TID-shaped rkey (13 chars, base32) — the realistic federated case.
    let fed_tid = ModelRef::parse("at://did:web:bob.hyprstream.ai/ai.hyprstream.model/3zztslq4be52u")
        .expect("federated TID ref parses");
    assert_eq!(
        fed_tid,
        ModelRef::Federated {
            did: "did:web:bob.hyprstream.ai".to_owned(),
            rkey: "3zztslq4be52u".to_owned(),
        }
    );

    // Local form: name:tag → Local (tag preserved).
    let local_tagged = ModelRef::parse("qwen3:main").expect("local tagged ref parses");
    assert_eq!(
        local_tagged,
        ModelRef::Local {
            name: "qwen3".to_owned(),
            tag: "main".to_owned(),
        }
    );

    // Local form: bare name → Local with default tag "main".
    let local_bare = ModelRef::parse("qwen3").expect("local bare ref parses");
    assert_eq!(
        local_bare,
        ModelRef::Local {
            name: "qwen3".to_owned(),
            tag: "main".to_owned(),
        }
    );

    // Local form: non-default tag.
    let local_instruct = ModelRef::parse("qwen3:instruct").expect("local instruct ref parses");
    assert_eq!(
        local_instruct,
        ModelRef::Local {
            name: "qwen3".to_owned(),
            tag: "instruct".to_owned(),
        }
    );

    // ── Negative cases: malformed refs are rejected, not silently misrouted ─
    // Federated ref with the wrong collection — reject (don't fall back to local).
    assert!(
        ModelRef::parse("at://did:web:x/ai.hyprstream.somethingelse/qwen3").is_err(),
        "federated ref with wrong collection must be rejected, not treated as local"
    );
    // Federated ref missing the rkey — reject.
    assert!(
        ModelRef::parse("at://did:web:x/ai.hyprstream.model/").is_err(),
        "federated ref with empty rkey must be rejected"
    );
    // Federated ref missing the DID — reject.
    assert!(
        ModelRef::parse("at:///ai.hyprstream.model/qwen3").is_err(),
        "federated ref with empty DID must be rejected"
    );
    // Local ref with empty name — reject.
    assert!(ModelRef::parse("").is_err(), "empty modelRef must be rejected");
    assert!(
        ModelRef::parse(":main").is_err(),
        "local ref with empty name must be rejected"
    );
    // Local ref with empty tag — reject (the `:` promises a tag).
    assert!(
        ModelRef::parse("qwen3:").is_err(),
        "local ref with empty tag must be rejected"
    );
}

/// A test-only modelRef dispatcher: confirms that the parsed branch maps to the
/// right resolver backend (PDS proof verification vs. local registry lookup).
/// This is the routing decision a real `resolveModelRef` would make; here we
/// just assert the branch, since the actual resolvers are exercised above.
#[test]
fn model_ref_routes_to_correct_branch() {
    use ModelRef::{Federated, Local};

    // Federated refs would hit the PDS path (verifyRecordProof).
    for s in [
        "at://did:web:alice.hyprstream.ai/ai.hyprstream.model/qwen3",
        "at://did:web:bob.hyprstream.ai/ai.hyprstream.model/3zztslq4be52u",
    ] {
        match ModelRef::parse(s).expect("parses") {
            Federated { .. } => {} // PDS branch
            Local { .. } => panic!("`{s}` must route to the federated (PDS) branch, not local"),
        }
    }

    // Local refs would hit the git2db registry.
    for s in ["qwen3:main", "qwen3", "llama-3.1-8b:instruct"] {
        match ModelRef::parse(s).expect("parses") {
            Local { .. } => {} // registry branch
            Federated { .. } => panic!("`{s}` must route to the local (registry) branch, not PDS"),
        }
    }
}
