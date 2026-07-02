//! Trivial `wasm32-wasip1` guest for the #506 Profile-B validation.
//!
//! It does ONLY filesystem ops via `std::fs` — which on `wasm32-wasip1` lower to
//! WASI preview1 `path_open`/`fd_read`/`fd_write`/`fd_readdir` against the single
//! preopen the host grants. In Profile B that preopen is a Subject-scoped
//! `hyprstream_vfs::Mount`, so every op here MUST land in the VFS, never the host.
//!
//! Sequence (all under the preopen root, mounted at guest path `.` / `/`):
//!   1. read  `seed.txt`   (host pre-populates it in the Mount) -> expect "hello-vfs"
//!   2. write `out.txt`    with the seed reversed
//!   3. read  `out.txt`    back and verify the round-trip
//!   4. readdir `.`        and verify both names are present
//!   5. write `done.txt`   = "ok"  (the host asserts this appears in the Mount)
//!
//! Exit code 0 == every step passed. Any mismatch -> nonzero exit (the host turns a
//! nonzero `proc_exit` into a test failure). We deliberately do NOT touch any
//! absolute host path, the clock, randomness, or sockets — there is no capability
//! for them in Profile B.

use std::fs;
use std::process::exit;

fn run() -> Result<(), String> {
    // The preopen is the Mount root. wasip1 resolves relative paths against the
    // preopen, so a bare name addresses the VFS.
    let seed = fs::read_to_string("seed.txt").map_err(|e| format!("read seed.txt: {e}"))?;
    let seed = seed.trim_end_matches('\n').to_string();
    if seed != "hello-vfs" {
        return Err(format!("seed.txt mismatch: got {seed:?}"));
    }

    let reversed: String = seed.chars().rev().collect();
    fs::write("out.txt", reversed.as_bytes()).map_err(|e| format!("write out.txt: {e}"))?;

    let back = fs::read_to_string("out.txt").map_err(|e| format!("read out.txt: {e}"))?;
    if back != reversed {
        return Err(format!("out.txt round-trip mismatch: {back:?} != {reversed:?}"));
    }

    // readdir the preopen root; expect at least seed.txt + out.txt.
    let mut names: Vec<String> = fs::read_dir(".")
        .map_err(|e| format!("readdir .: {e}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    if !names.iter().any(|n| n == "seed.txt") {
        return Err(format!("readdir missing seed.txt: {names:?}"));
    }
    if !names.iter().any(|n| n == "out.txt") {
        return Err(format!("readdir missing out.txt: {names:?}"));
    }

    // Final success marker — the host asserts this lands in the Mount.
    fs::write("done.txt", b"ok").map_err(|e| format!("write done.txt: {e}"))?;
    Ok(())
}

fn main() {
    match run() {
        Ok(()) => exit(0),
        Err(e) => {
            // stderr is a sink in Profile B, but emit anyway for native debugging.
            eprintln!("fsguest: {e}");
            exit(1);
        }
    }
}
