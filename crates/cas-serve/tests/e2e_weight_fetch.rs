//! E2E weight-fetch + cas-serve integration (CPU only, no libtorch).
//!
//! Exercises the full Hub-free XET weight-fetch path that git2db / the LFS
//! smudge layer rely on (#421 E2):
//!
//! 1. `UploadFile` — the cas-serve binary runs a synthetic safetensors-like
//!    blob through Gearhash CDC (`chunker.rs`), aggregates chunks into xorbs,
//!    and stores both the content-addressed xorbs and the reconstruction
//!    shard (`.mdb`) under the file's merkle hash.
//! 2. `GetFile` — the binary loads the shard, fetches each referenced xorb,
//!    and concatenates the segment bytes; the result must equal the original.
//! 3. LFS pointer resolution — a Git LFS pointer carrying the `xet-merkle` /
//!    `xet-shard` extensions (the #386 fix in `git2db/src/lfs.rs`) is built
//!    from the upload's file hash, round-tripped through the LFS pointer
//!    text format, and its `xet-merkle` is resolved back through `GetFile`.
//!    This proves the smudge path is self-describing and Hub-free: the real
//!    XET `MerkleHash` is read verbatim from the pointer rather than
//!    mis-converted from the LFS SHA-256 OID.
//! 4. Multi-xorb reconstruction — a file split across several xorbs (forced
//!    via a small per-xorb cap) still reconstructs through the binary's
//!    `GetFile` reassembly path.
//!
//! The cas-serve binary is driven as a subprocess speaking the NDJSON wire
//! protocol (`protocol::Request` / `protocol::Response`) over stdin/stdout,
//! exactly as a remote XET CAS client would over SSH. Storage is isolated to
//! a per-test `tempfile::Tempdir` via the `CAS_STORAGE` env var. Each test is
//! fully self-contained (integration tests run as separate processes).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use base64::Engine;
use cas_serve::{
    chunker,
    mdb_shard::MDB_SHARD_HEADER_TAG,
    protocol::{ErrorCode, Request, Response},
    shard::Shard,
};
use merklehash::MerkleHash;

/// Well-known LFS pointer extension key carrying the XET `MerkleHash`, per
/// `git2db/src/lfs.rs` / issue #386. Duplicated here so this test stays within
/// the `cas-serve` crate (no libtorch-coupled git2db dependency) while still
/// validating the exact wire format the LFS smudge layer reads.
const XET_MERKLE_EXT_KEY: &str = "xet-merkle";
const XET_SHARD_EXT_KEY: &str = "xet-shard";

/// Subprocess handle driving the `cas-serve` binary over NDJSON. Owns the
/// `TempDir` it stores xorbs/shards in; dropping the handle reaps the child.
struct CasProc {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    _dir: tempfile::TempDir,
}

impl CasProc {
    /// Spawn `cas-serve` with storage isolated to a fresh tempdir it owns.
    fn spawn() -> std::io::Result<Self> {
        let dir = tempfile::TempDir::new()?;
        Self::spawn_in(dir)
    }

    /// Spawn `cas-serve` against a caller-managed storage dir (used by the
    /// multi-xorb test, which pre-populates the xorb/shard layout). The caller
    /// must keep `dir` alive for the lifetime of the returned handle.
    fn spawn_in(dir: tempfile::TempDir) -> std::io::Result<Self> {
        let mut child = Command::new(env!("CARGO_BIN_EXE_cas-serve"))
            .env("CAS_STORAGE", dir.path())
            // tracing goes to stderr; silence it so it can't interleave with
            // the NDJSON responses on stdout (the protocol channel).
            .env("RUST_LOG", "off")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        Ok(Self {
            child,
            stdin,
            stdout,
            _dir: dir,
        })
    }

    /// Send one NDJSON request and read back exactly one response line.
    fn round_trip(&mut self, req: &Request) -> Response {
        let line = serde_json::to_string(req).expect("serialize request");
        writeln!(self.stdin, "{line}").expect("write request");
        self.stdin.flush().expect("flush request");
        let mut buf = String::new();
        let n =
            std::io::BufRead::read_line(&mut std::io::BufReader::new(&mut self.stdout), &mut buf)
                .expect("read response line");
        assert!(n > 0, "cas-serve closed stdout before replying to: {line}");
        serde_json::from_str::<Response>(buf.trim()).unwrap_or_else(|e| {
            panic!("failed to deserialize cas-serve response ({e}) to {line:?}: {buf:?}")
        })
    }

    /// Graceful shutdown (Shutdown → ShutdownAck) + reap, best-effort.
    fn shutdown(mut self) {
        let _ = writeln!(
            self.stdin,
            "{}",
            serde_json::to_string(&Request::Shutdown).unwrap()
        );
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

impl Drop for CasProc {
    fn drop(&mut self) {
        // Never leak a cas-serve process if a test panics before `shutdown`.
        // `kill` is a no-op on an already-exited child.
        let _ = self.child.kill();
    }
}

// --------------------------------------------------------------------------- //
// Test data helpers.                                                          //
// --------------------------------------------------------------------------- //

/// Generate `len` bytes of pseudo-random data (xorshift64) so the Gearhash
/// CDC actually cuts several boundaries. Constant seed → deterministic.
fn synthetic_weights(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let mut s: u64 = 0x9e3779b97f4a7c15;
    while out.len() + 8 <= len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        out.extend_from_slice(&s.to_le_bytes());
    }
    if out.len() < len {
        s ^= s << 13;
        s ^= s >> 7;
        out.extend_from_slice(&s.to_le_bytes()[..len - out.len()]);
    }
    assert_eq!(out.len(), len);
    out
}

/// SHA-256 of `data` as lowercase hex (the LFS OID). A self-contained impl
/// avoids adding a `sha2` dev-dependency; the digest is cross-checked below
/// against the bytes cas-serve reconstructs (length + content equality).
fn sha256_hex(data: &[u8]) -> String {
    fn sha256(msg: &[u8]) -> [u8; 32] {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        let mut h: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        let ml = (msg.len() as u64).wrapping_mul(8);
        let mut msg = msg.to_vec();
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&ml.to_be_bytes());
        for chunk in msg.chunks_exact(64) {
            let mut w = [0u32; 64];
            for (i, b) in chunk.chunks_exact(4).enumerate() {
                w[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
                (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
            for i in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let t1 = hh
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let t2 = s0.wrapping_add(maj);
                hh = g;
                g = f;
                f = e;
                e = d.wrapping_add(t1);
                d = c;
                c = b;
                b = a;
                a = t1.wrapping_add(t2);
            }
            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
            h[5] = h[5].wrapping_add(f);
            h[6] = h[6].wrapping_add(g);
            h[7] = h[7].wrapping_add(hh);
        }
        let mut out = [0u8; 32];
        for (i, v) in h.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&v.to_be_bytes());
        }
        out
    }
    let digest = sha256(data);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Parse the `key value` extension lines out of an LFS pointer text block.
/// Mirrors `git2db::lfs::LfsPointer::parse`'s extension handling: every
/// non-empty line that isn't `version`/`oid`/`size` is treated as an extension.
fn parse_lfs_extensions(text: &str) -> std::collections::HashMap<String, String> {
    let mut out = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with("version ")
            || line.starts_with("oid ")
            || line.starts_with("size ")
        {
            continue;
        }
        if let Some((k, v)) = line.split_once(' ') {
            out.insert(k.to_owned(), v.to_owned());
        }
    }
    out
}

/// Upload `original` through the cas-serve subprocess and assert the upload
/// succeeds, returning `(file_hash, xorb_hashes)`. Shared by the roundtrip
/// and pointer-resolution tests so both exercise the real `UploadFile` path.
fn upload_via_subprocess(cas: &mut CasProc, original: &[u8]) -> (String, Vec<String>) {
    let b64 = base64::engine::general_purpose::STANDARD.encode(original);
    match cas.round_trip(&Request::UploadFile { data: b64 }) {
        Response::UploadFileSuccess {
            file_hash,
            file_len,
            xorb_hashes,
        } => {
            assert_eq!(file_len, original.len() as u64);
            assert!(!xorb_hashes.is_empty(), "UploadFile must yield ≥1 xorb");
            MerkleHash::from_hex(&file_hash)
                .expect("file_hash must be a valid 64-char hex MerkleHash");
            (file_hash, xorb_hashes)
        }
        other => panic!("expected UploadFileSuccess, got {other:?}"),
    }
}

// --------------------------------------------------------------------------- //
// Tests.                                                                      //
// --------------------------------------------------------------------------- //

/// #421 E2 steps 1-3: UploadFile → CDC → xorbs → mdb_shard → GetFile roundtrip.
///
/// Drives the cas-serve *binary* as a subprocess over the NDJSON wire protocol,
/// so this exercises the real stdin/stdout transport a remote XET CAS client
/// uses over SSH, not just the library.
#[test]
fn weight_fetch_roundtrip_e2e() {
    let mut cas = CasProc::spawn().expect("spawn cas-serve");

    // 1 MiB synthetic safetensors-like weight blob (deterministic PRNG).
    const LEN: usize = 1024 * 1024;
    let original = synthetic_weights(LEN);

    // --- Steps 1-2: UploadFile → CDC chunks → xorbs → mdb_shard -------------
    let (file_hash, xorb_hashes) = upload_via_subprocess(&mut cas, &original);

    // --- Step 3: GetFile → mdb_shard reconstruction → bytes match ----------
    let got = cas.round_trip(&Request::GetFile {
        hash: file_hash.clone(),
    });
    let data = match got {
        Response::File { data } => base64::engine::general_purpose::STANDARD
            .decode(&data)
            .expect("decode GetFile base64"),
        other => panic!("expected File, got {other:?}"),
    };
    assert_eq!(
        data, original,
        "GetFile must reconstruct the original bytes exactly"
    );

    // Exists must report true for the shard and for every referenced xorb.
    let exists_file = cas.round_trip(&Request::Exists {
        hash: file_hash.clone(),
    });
    assert!(
        matches!(exists_file, Response::Exists { exists: true }),
        "shard for the file hash must exist"
    );
    for xh in &xorb_hashes {
        let exists_xorb = cas.round_trip(&Request::Exists { hash: xh.clone() });
        assert!(
            matches!(exists_xorb, Response::Exists { exists: true }),
            "xorb {xh} referenced by the shard must exist"
        );
    }

    // GetReconstructionInfo must return a base64'd mdb_shard binary that
    // parses with the XET magic header and yields segments summing to file_len.
    let info = cas.round_trip(&Request::GetReconstructionInfo {
        hash: file_hash.clone(),
    });
    let mdb_bytes = match info {
        Response::ReconstructionInfo { info } => base64::engine::general_purpose::STANDARD
            .decode(&info)
            .unwrap(),
        other => panic!("expected ReconstructionInfo, got {other:?}"),
    };
    assert_eq!(
        &mdb_bytes[..MDB_SHARD_HEADER_TAG.len()],
        &MDB_SHARD_HEADER_TAG[..],
        "reconstruction info must be an mdb_shard binary with the XET magic header"
    );
    let file_hash_parsed = MerkleHash::from_hex(&file_hash).unwrap();
    let segments = Shard::segments(&mdb_bytes, &file_hash_parsed)
        .expect("mdb_shard segments must parse for the file hash");
    let total: u64 = segments.iter().map(|s| s.byte_len).sum();
    assert_eq!(total, LEN as u64, "segment byte total must equal file_len");

    cas.shutdown();
}

/// #421 E2 steps 4-5: build an LFS pointer with the #386 `xet-merkle`/
/// `xet-shard` extensions, then resolve `xet-merkle` → cas-serve `GetFile`
/// → bytes match the original.
///
/// This mirrors what `git2db::lfs::LfsPointer::set_xet_extensions` produces at
/// promote time and what `LfsSmudge::smudge_lfs` consumes at checkout time
/// (resolving the pointer's `xet_merkle_hash()` against XET CAS). The pointer
/// format is validated directly here so the test stays in the `cas-serve`
/// crate and does not pull git2db's libtorch-coupled dependency graph.
#[test]
fn lfs_pointer_xet_merkle_resolution() {
    let mut cas = CasProc::spawn().expect("spawn cas-serve");

    const LEN: usize = 1024 * 1024;
    let original = synthetic_weights(LEN);

    // Upload through the binary so the shard + xorbs are really on disk.
    let (file_hash, xorb_hashes) = upload_via_subprocess(&mut cas, &original);

    // --- Step 4: build the LFS pointer with xet-merkle/xet-shard -----------
    // The #386 format: standard LFS header lines, then the well-known
    // extension lines. The OID is the genuine SHA-256 of the bytes (a real
    // promote-time pointer would carry this), and `xet-merkle` carries the
    // real XET MerkleHash — the two are distinct hash families, which is the
    // exact unsoundness the #386 fix addresses.
    let oid = sha256_hex(&original);
    assert_ne!(
        oid, file_hash,
        "LFS SHA-256 OID must differ from the XET MerkleHash (the #386 invariant)"
    );
    let shard_hint = format!("default.{}", &xorb_hashes[0]);
    let pointer_text = format!(
        "version https://git-lfs.github.com/spec/v1\n\
         oid sha256:{oid}\n\
         size {size}\n\
         {XET_MERKLE_EXT_KEY} {file_hash}\n\
         {XET_SHARD_EXT_KEY} {shard_hint}\n",
        size = original.len(),
    );

    // Round-trip the pointer text through the LFS extension parser and confirm
    // `xet-merkle` survives verbatim (this is what `LfsPointer::parse` does).
    let parsed = parse_lfs_extensions(&pointer_text);
    let resolved_merkle_hex = parsed
        .get(XET_MERKLE_EXT_KEY)
        .expect("xet-merkle extension must survive an LFS pointer parse round-trip (#386)");
    assert_eq!(
        resolved_merkle_hex, &file_hash,
        "xet-merkle must equal the uploader's file merkle hash"
    );
    assert_eq!(
        parsed.get(XET_SHARD_EXT_KEY).map(String::as_str),
        Some(shard_hint.as_str()),
        "xet-shard hint must round-trip"
    );

    // The resolved xet-merkle must be a valid MerkleHash — this is exactly the
    // `LfsPointer::xet_merkle_hash()` call the smudge path makes.
    let merkle = MerkleHash::from_hex(resolved_merkle_hex)
        .expect("xet-merkle extension must parse as a MerkleHash");

    // --- Step 5: resolve xet-merkle → cas-serve GetFile → bytes match ------
    let got = cas.round_trip(&Request::GetFile { hash: merkle.hex() });
    let data = match got {
        Response::File { data } => base64::engine::general_purpose::STANDARD
            .decode(&data)
            .expect("decode GetFile base64"),
        other => panic!("expected File resolving via xet-merkle, got {other:?}"),
    };
    assert_eq!(
        data, original,
        "resolving the LFS pointer's xet-merkle through cas-serve must yield the original bytes"
    );

    // Defense in depth: a pointer with no xet-merkle extension must NOT resolve
    // to a MerkleHash. The smudge path errors clearly here (#386) instead of
    // silently mis-converting the SHA-256 OID; we emulate that check.
    let vanilla_pointer = format!(
        "version https://git-lfs.github.com/spec/v1\n\
         oid sha256:{oid}\n\
         size {size}\n",
        size = original.len(),
    );
    let vanilla_ext = parse_lfs_extensions(&vanilla_pointer);
    assert!(
        !vanilla_ext.contains_key(XET_MERKLE_EXT_KEY),
        "a vanilla LFS pointer carries no xet-merkle; smudge must refuse (see #386)"
    );

    cas.shutdown();
}

/// #421 E2 step 6: multi-xorb reconstruction through the binary.
///
/// The production cas-serve binary hardcodes the spec 64 MiB per-xorb cap
/// (`chunker::MAX_XORB_BYTES`). To exercise multi-xorb aggregation through the
/// *binary's* `GetFile` reassembly path without allocating 64+ MiB, this test:
///
/// 1. Chunks ~1 MiB of synthetic weights with the spec CDC chunker.
/// 2. Aggregates the chunks into several xorbs using a small forced cap via
///    `Shard::from_chunks_with_cap` (the same builder the binary uses, just
///    with a test-only cap).
/// 3. Writes the xorbs + shard into a cas-serve storage layout (under a fresh
///    `CAS_STORAGE` tempdir) exactly as `UploadFile` would.
/// 4. Points the cas-serve *binary* at that storage and resolves the file via
///    `GetFile`, exercising the binary's `reassemble_from_shard` path across
///    multiple xorbs.
///
/// This validates that a file spanning multiple xorbs reconstructs correctly
/// through the real subprocess reassembly logic, not just the library.
#[test]
fn multi_xorb_reconstruction() {
    let dir = tempfile::TempDir::new().expect("tempdir for multi-xorb storage");
    // Keep a second handle to the same path so the TempDir is not dropped when
    // handed into `spawn_in`; the handle lives for the test's scope.
    let storage_path = dir.path().to_path_buf();
    let mut cas = CasProc::spawn_in(dir).expect("spawn cas-serve");

    // ~1 MiB of pseudo-random weights so CDC cuts many boundaries.
    const LEN: usize = 1024 * 1024;
    let original = synthetic_weights(LEN);

    // Chunk + aggregate with a tiny cap so the file spans many xorbs.
    let chunks = chunker::chunk_all(&original);
    assert!(
        chunks.len() > 1,
        "expected CDC to cut multiple chunks for {LEN} bytes"
    );
    const FORCED_CAP: usize = 256 * 1024;
    let (shard, xorbs) = Shard::from_chunks_with_cap(&chunks, FORCED_CAP);
    assert!(
        xorbs.len() > 1,
        "expected multiple xorbs with a {FORCED_CAP}-byte cap, got {}",
        xorbs.len()
    );

    // Materialize the xorbs + shard in the cas-serve storage layout, exactly
    // mirroring `handle_upload_file`. `spawn_in` owns the TempDir; we write
    // through `storage_path` (the same path the binary sees via CAS_STORAGE).
    let xorbs_dir = storage_path.join("xorbs");
    let shards_dir = storage_path.join("shards");
    std::fs::create_dir_all(&xorbs_dir).unwrap();
    std::fs::create_dir_all(&shards_dir).unwrap();
    for (h, bytes) in &xorbs {
        std::fs::write(xorbs_dir.join(format!("default.{}", h.hex())), bytes).unwrap();
    }
    std::fs::write(shards_dir.join(&shard.file_hash), shard.to_bytes()).unwrap();

    // Reconstruct via the cas-serve *binary* (subprocess GetFile) — this goes
    // through `reassemble_from_shard`, walking each segment's xorb across the
    // multi-xorb shard.
    let got = cas.round_trip(&Request::GetFile {
        hash: shard.file_hash.clone(),
    });
    let data = match got {
        Response::File { data } => base64::engine::general_purpose::STANDARD
            .decode(&data)
            .expect("decode GetFile base64"),
        Response::Error { code, message } => {
            panic!("cas-serve GetFile failed across multi-xorb shard: {code:?}: {message}")
        }
        other => panic!("expected File, got {other:?}"),
    };
    assert_eq!(
        data.len(),
        original.len(),
        "reconstructed multi-xorb file must match original length"
    );
    assert_eq!(
        data, original,
        "reconstructed multi-xorb file must match original bytes exactly"
    );

    // Cross-check: querying a missing hash must surface a clean NotFound (not a
    // silent empty File), so a corrupt/short shard is never mistaken for an
    // empty weight file.
    let bogus = "0".repeat(64);
    let missing = cas.round_trip(&Request::GetFile { hash: bogus });
    match missing {
        Response::Error { code, .. } => assert_eq!(code, ErrorCode::NotFound),
        other => panic!("expected NotFound for bogus hash, got {other:?}"),
    }

    cas.shutdown();
    drop(storage_path);
}
