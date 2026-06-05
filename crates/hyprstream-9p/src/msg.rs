//! 9P2000.L message types and wire format serialization.
//!
//! Minimal implementation — only the message types needed for a 9P client
//! to walk, open, read, write, readdir, stat, and clunk.
//!
//! Wire format: all little-endian, length-prefixed.
//! ```text
//! [4 bytes: total size including this field]
//! [1 byte: message type]
//! [2 bytes: tag]
//! [... type-specific fields ...]
//! ```

use std::io::{Cursor, Read};

// ─────────────────────────────────────────────────────────────────────────────
// Message type constants (9P2000.L)
// ─────────────────────────────────────────────────────────────────────────────

pub const TVERSION: u8 = 100;
pub const RVERSION: u8 = 101;
pub const TATTACH: u8 = 104;
pub const RATTACH: u8 = 105;
pub const RLERROR: u8 = 7;
pub const TWALK: u8 = 110;
pub const RWALK: u8 = 111;
pub const TLOPEN: u8 = 12;
pub const RLOPEN: u8 = 13;
pub const TREAD: u8 = 116;
pub const RREAD: u8 = 117;
pub const TWRITE: u8 = 118;
pub const RWRITE: u8 = 119;
pub const TCLUNK: u8 = 120;
pub const RCLUNK: u8 = 121;
pub const TGETATTR: u8 = 24;
pub const RGETATTR: u8 = 25;
pub const TREADDIR: u8 = 40;
pub const RREADDIR: u8 = 41;

// ─────────────────────────────────────────────────────────────────────────────
// QID — unique file identifier
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct Qid {
    pub qtype: u8,
    pub version: u32,
    pub path: u64,
}

impl Qid {
    pub fn read_from(r: &mut Cursor<&[u8]>) -> anyhow::Result<Self> {
        Ok(Qid {
            qtype: read_u8(r)?,
            version: read_u32(r)?,
            path: read_u64(r)?,
        })
    }

    pub fn write_to(&self, w: &mut Vec<u8>) {
        w.push(self.qtype);
        w.extend_from_slice(&self.version.to_le_bytes());
        w.extend_from_slice(&self.path.to_le_bytes());
    }

    pub fn is_dir(&self) -> bool {
        self.qtype & 0x80 != 0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Wire format helpers
// ─────────────────────────────────────────────────────────────────────────────

fn read_u8(r: &mut Cursor<&[u8]>) -> anyhow::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(r: &mut Cursor<&[u8]>) -> anyhow::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut Cursor<&[u8]>) -> anyhow::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut Cursor<&[u8]>) -> anyhow::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string(r: &mut Cursor<&[u8]>) -> anyhow::Result<String> {
    let len = read_u16(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

fn read_data(r: &mut Cursor<&[u8]>) -> anyhow::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_string(w: &mut Vec<u8>, s: &str) {
    w.extend_from_slice(&(s.len() as u16).to_le_bytes());
    w.extend_from_slice(s.as_bytes());
}

#[allow(dead_code)]
fn write_data(w: &mut Vec<u8>, data: &[u8]) {
    w.extend_from_slice(&(data.len() as u32).to_le_bytes());
    w.extend_from_slice(data);
}

// ─────────────────────────────────────────────────────────────────────────────
// T-messages (client → server)
// ─────────────────────────────────────────────────────────────────────────────

/// Serialize a T-message to wire format (including length prefix).
pub fn encode_tmessage(tag: u16, msg_type: u8, body: &[u8]) -> Vec<u8> {
    let size = 4 + 1 + 2 + body.len();
    let mut buf = Vec::with_capacity(size);
    buf.extend_from_slice(&(size as u32).to_le_bytes());
    buf.push(msg_type);
    buf.extend_from_slice(&tag.to_le_bytes());
    buf.extend_from_slice(body);
    buf
}

pub fn tversion(tag: u16, msize: u32, version: &str) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&msize.to_le_bytes());
    write_string(&mut body, version);
    encode_tmessage(tag, TVERSION, &body)
}

pub fn tattach(tag: u16, fid: u32, afid: u32, uname: &str, aname: &str) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&afid.to_le_bytes());
    write_string(&mut body, uname);
    write_string(&mut body, aname);
    // 9P2000.L: n_uname
    body.extend_from_slice(&0u32.to_le_bytes());
    encode_tmessage(tag, TATTACH, &body)
}

pub fn twalk(tag: u16, fid: u32, newfid: u32, wnames: &[&str]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&newfid.to_le_bytes());
    body.extend_from_slice(&(wnames.len() as u16).to_le_bytes());
    for name in wnames {
        write_string(&mut body, name);
    }
    encode_tmessage(tag, TWALK, &body)
}

pub fn tlopen(tag: u16, fid: u32, flags: u32) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&flags.to_le_bytes());
    encode_tmessage(tag, TLOPEN, &body)
}

pub fn tread(tag: u16, fid: u32, offset: u64, count: u32) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&offset.to_le_bytes());
    body.extend_from_slice(&count.to_le_bytes());
    encode_tmessage(tag, TREAD, &body)
}

pub fn twrite(tag: u16, fid: u32, offset: u64, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&offset.to_le_bytes());
    body.extend_from_slice(&(data.len() as u32).to_le_bytes());
    body.extend_from_slice(data);
    encode_tmessage(tag, TWRITE, &body)
}

pub fn tclunk(tag: u16, fid: u32) -> Vec<u8> {
    encode_tmessage(tag, TCLUNK, &fid.to_le_bytes())
}

pub fn tgetattr(tag: u16, fid: u32, request_mask: u64) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&request_mask.to_le_bytes());
    encode_tmessage(tag, TGETATTR, &body)
}

pub fn treaddir(tag: u16, fid: u32, offset: u64, count: u32) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&fid.to_le_bytes());
    body.extend_from_slice(&offset.to_le_bytes());
    body.extend_from_slice(&count.to_le_bytes());
    encode_tmessage(tag, TREADDIR, &body)
}

// ─────────────────────────────────────────────────────────────────────────────
// R-messages (server → client) — parsed responses
// ─────────────────────────────────────────────────────────────────────────────

/// Parsed R-message from server.
#[derive(Debug)]
pub enum Response {
    Version { msize: u32, version: String },
    Attach { qid: Qid },
    Walk { qids: Vec<Qid> },
    Lopen { qid: Qid, iounit: u32 },
    Read { data: Vec<u8> },
    Write { count: u32 },
    Clunk,
    Getattr { qid: Qid, mode: u32, size: u64, mtime_sec: u64 },
    Readdir { data: Vec<u8> },
    Error { ecode: u32 },
}

/// Parse an R-message from wire bytes (including length prefix).
pub fn parse_response(buf: &[u8]) -> anyhow::Result<(u16, Response)> {
    if buf.len() < 7 {
        anyhow::bail!("9P response too short: {} bytes", buf.len());
    }

    let mut r = Cursor::new(buf);
    let _size = read_u32(&mut r)?;
    let msg_type = read_u8(&mut r)?;
    let tag = read_u16(&mut r)?;

    let response = match msg_type {
        RLERROR => {
            let ecode = read_u32(&mut r)?;
            Response::Error { ecode }
        }
        RVERSION => {
            let msize = read_u32(&mut r)?;
            let version = read_string(&mut r)?;
            Response::Version { msize, version }
        }
        RATTACH => {
            let qid = Qid::read_from(&mut r)?;
            Response::Attach { qid }
        }
        RWALK => {
            let nwqid = read_u16(&mut r)?;
            let mut qids = Vec::with_capacity(nwqid as usize);
            for _ in 0..nwqid {
                qids.push(Qid::read_from(&mut r)?);
            }
            Response::Walk { qids }
        }
        RLOPEN => {
            let qid = Qid::read_from(&mut r)?;
            let iounit = read_u32(&mut r)?;
            Response::Lopen { qid, iounit }
        }
        RREAD => {
            let data = read_data(&mut r)?;
            Response::Read { data }
        }
        RWRITE => {
            let count = read_u32(&mut r)?;
            Response::Write { count }
        }
        RCLUNK => Response::Clunk,
        RGETATTR => {
            let _valid = read_u64(&mut r)?;
            let qid = Qid::read_from(&mut r)?;
            let mode = read_u32(&mut r)?;
            let _uid = read_u32(&mut r)?;
            let _gid = read_u32(&mut r)?;
            let _nlink = read_u64(&mut r)?;
            let _rdev = read_u64(&mut r)?;
            let size = read_u64(&mut r)?;
            let _blksize = read_u64(&mut r)?;
            let _blocks = read_u64(&mut r)?;
            let _atime_sec = read_u64(&mut r)?;
            let _atime_nsec = read_u64(&mut r)?;
            let mtime_sec = read_u64(&mut r)?;
            let _mtime_nsec = read_u64(&mut r)?;
            let _ctime_sec = read_u64(&mut r)?;
            let _ctime_nsec = read_u64(&mut r)?;
            let _btime_sec = read_u64(&mut r)?;
            let _btime_nsec = read_u64(&mut r)?;
            let _gen = read_u64(&mut r)?;
            let _data_version = read_u64(&mut r)?;
            Response::Getattr { qid, mode, size, mtime_sec }
        }
        RREADDIR => {
            let data = read_data(&mut r)?;
            Response::Readdir { data }
        }
        _ => anyhow::bail!("unknown 9P response type: {msg_type}"),
    };

    Ok((tag, response))
}

// ─────────────────────────────────────────────────────────────────────────────
// Readdir entry parsing
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed directory entry from Rreaddir data.
#[derive(Debug)]
pub struct DirEntryP9 {
    pub qid: Qid,
    pub offset: u64,
    pub dtype: u8,
    pub name: String,
}

/// Parse Rreaddir data into directory entries.
pub fn parse_readdir_entries(data: &[u8]) -> anyhow::Result<Vec<DirEntryP9>> {
    let mut entries = Vec::new();
    let mut r = Cursor::new(data);

    while (r.position() as usize) < data.len() {
        let qid = Qid::read_from(&mut r)?;
        let offset = read_u64(&mut r)?;
        let dtype = read_u8(&mut r)?;
        let name = read_string(&mut r)?;
        entries.push(DirEntryP9 { qid, offset, dtype, name });
    }

    Ok(entries)
}
