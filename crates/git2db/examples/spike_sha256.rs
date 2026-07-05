//! SPIKE (#810 / epic #809, W1): can our pinned git2-rs / vendored libgit2 open a
//! SHA-256 object-format repository at all?
//!
//! Reproduction harness. Usage:
//!
//! ```sh
//! git init --object-format=sha256 /tmp/sha256repo
//! git -C /tmp/sha256repo -c user.email=a@b.c -c user.name=a commit \
//!     --allow-empty -m init
//! cargo run -p git2db --example spike_sha256 -- /tmp/sha256repo
//! ```
//!
//! Result on git2 0.20.3 + vendored libgit2-sys 0.18.3 (libgit2 1.9.2):
//!
//! ```text
//! OPEN ERR: class=Repository code=Invalid msg=unknown object format 'sha256'
//! ```
//!
//! i.e. NOT CAPABLE — libgit2 rejects the repo at `Repository::open` because it was
//! not compiled with `GIT_EXPERIMENTAL_SHA256`. A SHA-1 repo opens fine. See the PR
//! body for the full writeup and the upgrade path that would unblock W1.
//!
//! ## Phase 2 (compile-time enable) — result: ABI-UNSAFE, still not viable
//!
//! Patching the vendored `libgit2-sys` build to define `GIT_EXPERIMENTAL_SHA256`
//! (a `cc::Build::define(...)` on the C compile; the crate uses `cc`, not cmake — there
//! is no `EXPERIMENTAL_SHA256` cmake knob and no feature/env knob exposed) makes the C
//! library accept sha256 repos, and `Repository::open` then succeeds. BUT the flag is
//! ABI-breaking: with it, C `struct git_oid` becomes `{ unsigned char type; unsigned char
//! id[32]; }` (33 bytes), while libgit2-sys's Rust binding still declares `{ id: [u8; 20] }`
//! and gates nothing on the flag. Result — every OID read is corrupted, for BOTH sha256
//! AND previously-working sha1 repos:
//!
//! ```text
//! sha256 real HEAD:  eea50b3...0dca157b (64 hex)  → git2-rs read: eea50b34...337692ff  → commit NOT FOUND
//! sha1   real HEAD:  c162c8f...335c6112d (40 hex)  → git2-rs read: c162c8f...335c611fc  → commit NOT FOUND (REGRESSED)
//! ```
//!
//! So a libgit2-sys-only C define is unsafe and regresses SHA-1. A real enable needs a
//! coordinated fork of BOTH `libgit2-sys` (define + rewrite `git_oid` struct + all
//! sha256-gated extern signatures) AND `git2-rs` (its safe `Oid` hardcodes 20-byte/40-hex
//! SHA-1), tracking an explicitly-unstable upstream ABI. See PR body for cost/recommendation.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout)]

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: spike_sha256 <repo-path>");
    match git2::Repository::open(&path) {
        Ok(repo) => {
            println!("OPEN OK");
            if let Ok(head) = repo.head() {
                if let Some(oid) = head.target() {
                    println!("HEAD OID: {} (len={})", oid, oid.to_string().len());
                    match repo.find_commit(oid) {
                        Ok(c) => println!("READ COMMIT OK: {:?}", c.summary()),
                        Err(e) => println!("READ COMMIT ERR: {e}"),
                    }
                }
            }
            if let Ok(cfg) = repo.config() {
                match cfg.get_string("extensions.objectformat") {
                    Ok(v) => println!("extensions.objectFormat = {v}"),
                    Err(e) => println!("objectFormat read ERR: {e}"),
                }
            }
        }
        Err(e) => println!(
            "OPEN ERR: class={:?} code={:?} msg={}",
            e.class(),
            e.code(),
            e.message()
        ),
    }
}
