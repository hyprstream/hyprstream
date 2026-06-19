//! In-memory `ZeroCopyReader` / `ZeroCopyWriter` adapters.
//!
//! `fuse_backend_rs`'s `read`/`write` ops do not hand back a buffer; they copy
//! through the `ZeroCopyReader` / `ZeroCopyWriter` traits, which in turn move
//! bytes to/from the backend's real file via [`FileReadWriteVolatile`]. The VFS
//! `Mount::read`/`write` API, however, is buffer-based (`Vec<u8>` / `&[u8]`).
//! These two tiny adapters bridge the gap with a heap buffer — no host fd, no
//! mmap, so they work uniformly over passthrough, overlay and RAFS backends.

use std::cmp::min;
use std::io::{self, Read, Write};

use fuse_backend_rs::api::filesystem::{ZeroCopyReader, ZeroCopyWriter};
use fuse_backend_rs::file_buf::FileVolatileSlice;
use fuse_backend_rs::file_traits::FileReadWriteVolatile;

/// Chunk size for staging copies through a heap buffer.
const STAGE_CHUNK: usize = 128 * 1024;

// ─────────────────────────────────────────────────────────────────────────────
// MemWriter — sink for FileSystem::read
// ─────────────────────────────────────────────────────────────────────────────

/// A [`ZeroCopyWriter`] that accumulates everything written into an owned
/// `Vec<u8>`. Used as the `w` argument of `FileSystem::read`.
pub struct MemWriter {
    buf: Vec<u8>,
}

impl MemWriter {
    pub fn with_capacity(cap: usize) -> Self {
        Self { buf: Vec::with_capacity(cap) }
    }

    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }
}

impl Write for MemWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl ZeroCopyWriter for MemWriter {
    /// Copy up to `count` bytes out of the backend file `f` (starting at `off`)
    /// into our buffer, staging through a heap chunk.
    fn write_from(&mut self, f: &mut dyn FileReadWriteVolatile, count: usize, off: u64) -> io::Result<usize> {
        let mut stage = vec![0u8; min(count, STAGE_CHUNK)];
        // SAFETY: `stage` is a live, exclusively-borrowed heap buffer for the
        // duration of the volatile read; no other accessor exists.
        let slice = unsafe { FileVolatileSlice::from_mut_slice(&mut stage) };
        let n = f.read_at_volatile(slice, off)?;
        self.buf.extend_from_slice(&stage[..n]);
        Ok(n)
    }

    fn available_bytes(&self) -> usize {
        // The destination is an unbounded heap Vec.
        usize::MAX
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemReader — source for FileSystem::write
// ─────────────────────────────────────────────────────────────────────────────

/// A [`ZeroCopyReader`] backed by a caller-owned byte slice. Used as the `r`
/// argument of `FileSystem::write`.
pub struct MemReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MemReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl Read for MemReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = min(out.len(), self.data.len() - self.pos);
        out[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

impl ZeroCopyReader for MemReader<'_> {
    /// Copy up to `count` bytes from our slice into the backend file `f` at
    /// `off`, staging through a heap chunk.
    fn read_to(&mut self, f: &mut dyn FileReadWriteVolatile, count: usize, off: u64) -> io::Result<usize> {
        let remaining = self.data.len() - self.pos;
        let take = min(min(count, remaining), STAGE_CHUNK);
        if take == 0 {
            return Ok(0);
        }
        let mut stage = self.data[self.pos..self.pos + take].to_vec();
        // SAFETY: `stage` is a live, exclusively-borrowed heap buffer for the
        // duration of the volatile write; no other accessor exists.
        let slice = unsafe { FileVolatileSlice::from_mut_slice(&mut stage) };
        let n = f.write_at_volatile(slice, off)?;
        self.pos += n;
        Ok(n)
    }
}
