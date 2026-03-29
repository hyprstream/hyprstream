//! 9P2000.L wire protocol types and codec.
//!
//! Implements the subset of 9P2000.L needed for VFS federation:
//! version, attach, walk, lopen, read, write, clunk, readdir, getattr, statfs,
//! flush, and lerror.
//!
//! Wire format: all integers are little-endian. Strings are u16-length-prefixed
//! UTF-8. Every frame is: `u32 size` (inclusive) + `u8 type` + `u16 tag` + payload.
//!
//! Reference: <https://github.com/chaos/diod/blob/master/protocol.md>

use std::io::{self, Cursor, Read, Write};

// ─────────────────────────────────────────────────────────────────────────────
// Message type constants (from linux include/net/9p/9p.h)
// ─────────────────────────────────────────────────────────────────────────────

const TLERROR: u8 = 6;
const RLERROR: u8 = TLERROR + 1;
const TSTATFS: u8 = 8;
const RSTATFS: u8 = TSTATFS + 1;
const TLOPEN: u8 = 12;
const RLOPEN: u8 = TLOPEN + 1;
const TGETATTR: u8 = 24;
const RGETATTR: u8 = TGETATTR + 1;
const TREADDIR: u8 = 40;
const RREADDIR: u8 = TREADDIR + 1;
const TVERSION: u8 = 100;
const RVERSION: u8 = TVERSION + 1;
const TAUTH: u8 = 102;
#[allow(dead_code)]
const RAUTH: u8 = TAUTH + 1;
const TATTACH: u8 = 104;
const RATTACH: u8 = TATTACH + 1;
const TFLUSH: u8 = 108;
const RFLUSH: u8 = TFLUSH + 1;
const TWALK: u8 = 110;
const RWALK: u8 = TWALK + 1;
const TREAD: u8 = 116;
const RREAD: u8 = TREAD + 1;
const TWRITE: u8 = 118;
const RWRITE: u8 = TWRITE + 1;
const TCLUNK: u8 = 120;
const RCLUNK: u8 = TCLUNK + 1;

// ─────────────────────────────────────────────────────────────────────────────
// QID
// ─────────────────────────────────────────────────────────────────────────────

/// QID type: directory.
pub const QTDIR: u8 = 0x80;
/// QID type: regular file.
pub const QTFILE: u8 = 0x00;

/// 9P2000.L unique file identifier (13 bytes on wire).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qid {
    pub ty: u8,
    pub version: u32,
    pub path: u64,
}

impl Qid {
    fn encode<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&[self.ty])?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.path.to_le_bytes())
    }

    fn decode<R: Read>(r: &mut R) -> io::Result<Self> {
        let ty = read_u8(r)?;
        let version = read_u32(r)?;
        let path = read_u64(r)?;
        Ok(Self { ty, version, path })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dirent (for readdir data encoding)
// ─────────────────────────────────────────────────────────────────────────────

/// 9P2000.L directory entry (within readdir data buffer).
#[derive(Clone, Debug)]
pub struct Dirent {
    pub qid: Qid,
    pub offset: u64,
    pub ty: u8,
    pub name: String,
}

impl Dirent {
    /// Wire size of this dirent.
    pub fn byte_size(&self) -> usize {
        13 + 8 + 1 + 2 + self.name.len() // qid + offset + ty + name_len + name
    }

    /// Encode into writer.
    pub fn encode<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.qid.encode(w)?;
        w.write_all(&self.offset.to_le_bytes())?;
        w.write_all(&[self.ty])?;
        write_string(w, &self.name)
    }

    /// Decode from reader.
    pub fn decode<R: Read>(r: &mut R) -> io::Result<Self> {
        let qid = Qid::decode(r)?;
        let offset = read_u64(r)?;
        let ty = read_u8(r)?;
        let name = read_string(r)?;
        Ok(Self { qid, offset, ty, name })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// T-messages (client → server)
// ─────────────────────────────────────────────────────────────────────────────

/// Client-to-server message.
#[derive(Debug)]
pub enum Tmessage {
    Version(Tversion),
    Auth(Tauth),
    Attach(Tattach),
    Flush(Tflush),
    Walk(Twalk),
    Lopen(Tlopen),
    Read(Tread),
    Write(Twrite),
    Clunk(Tclunk),
    Readdir(Treaddir),
    GetAttr(Tgetattr),
    Statfs(Tstatfs),
}

#[derive(Debug)]
pub struct Tversion {
    pub msize: u32,
    pub version: String,
}

#[derive(Debug)]
pub struct Tauth {
    pub afid: u32,
    pub uname: String,
    pub aname: String,
    pub n_uname: u32,
}

#[derive(Debug)]
pub struct Tattach {
    pub fid: u32,
    pub afid: u32,
    pub uname: String,
    pub aname: String,
    pub n_uname: u32,
}

#[derive(Debug)]
pub struct Tflush {
    pub oldtag: u16,
}

#[derive(Debug)]
pub struct Twalk {
    pub fid: u32,
    pub newfid: u32,
    pub wnames: Vec<String>,
}

#[derive(Debug)]
pub struct Tlopen {
    pub fid: u32,
    pub flags: u32,
}

#[derive(Debug)]
pub struct Tread {
    pub fid: u32,
    pub offset: u64,
    pub count: u32,
}

#[derive(Debug)]
pub struct Twrite {
    pub fid: u32,
    pub offset: u64,
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Tclunk {
    pub fid: u32,
}

#[derive(Debug)]
pub struct Treaddir {
    pub fid: u32,
    pub offset: u64,
    pub count: u32,
}

#[derive(Debug)]
pub struct Tgetattr {
    pub fid: u32,
    pub request_mask: u64,
}

#[derive(Debug)]
pub struct Tstatfs {
    pub fid: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// R-messages (server → client)
// ─────────────────────────────────────────────────────────────────────────────

/// Server-to-client message.
#[derive(Debug)]
pub enum Rmessage {
    Version(Rversion),
    Attach(Rattach),
    Flush,
    Walk(Rwalk),
    Lopen(Rlopen),
    Read(Rread),
    Write(Rwrite),
    Clunk,
    Readdir(Rreaddir),
    GetAttr(Rgetattr),
    Statfs(Rstatfs),
    Lerror(Rlerror),
}

#[derive(Debug)]
pub struct Rversion {
    pub msize: u32,
    pub version: String,
}

#[derive(Debug)]
pub struct Rlerror {
    pub ecode: u32,
}

#[derive(Debug)]
pub struct Rattach {
    pub qid: Qid,
}

#[derive(Debug)]
pub struct Rwalk {
    pub wqids: Vec<Qid>,
}

#[derive(Debug)]
pub struct Rlopen {
    pub qid: Qid,
    pub iounit: u32,
}

#[derive(Debug)]
pub struct Rread {
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Rwrite {
    pub count: u32,
}

#[derive(Debug)]
pub struct Rreaddir {
    pub data: Vec<u8>,
}

#[derive(Debug)]
pub struct Rgetattr {
    pub valid: u64,
    pub qid: Qid,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub nlink: u64,
    pub rdev: u64,
    pub size: u64,
    pub blksize: u64,
    pub blocks: u64,
    pub atime_sec: u64,
    pub atime_nsec: u64,
    pub mtime_sec: u64,
    pub mtime_nsec: u64,
    pub ctime_sec: u64,
    pub ctime_nsec: u64,
    pub btime_sec: u64,
    pub btime_nsec: u64,
    pub gen: u64,
    pub data_version: u64,
}

#[derive(Debug)]
pub struct Rstatfs {
    pub ty: u32,
    pub bsize: u32,
    pub blocks: u64,
    pub bfree: u64,
    pub bavail: u64,
    pub files: u64,
    pub ffree: u64,
    pub fsid: u64,
    pub namelen: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// Frames (size + type + tag + message)
// ─────────────────────────────────────────────────────────────────────────────

/// Complete T-frame (client request).
#[derive(Debug)]
pub struct Tframe {
    pub tag: u16,
    pub msg: Tmessage,
}

/// Complete R-frame (server response).
#[derive(Debug)]
pub struct Rframe {
    pub tag: u16,
    pub msg: Rmessage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Wire encoding / decoding
// ─────────────────────────────────────────────────────────────────────────────

// Primitive readers/writers.

fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> io::Result<String> {
    let len = read_u16(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn read_data<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let len = read_u32(r)? as usize;
    if len > 32 * 1024 * 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "data too large"));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_string<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    if bytes.len() > u16::MAX as usize {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "string too long"));
    }
    w.write_all(&(bytes.len() as u16).to_le_bytes())?;
    w.write_all(bytes)
}

fn write_data<W: Write>(w: &mut W, data: &[u8]) -> io::Result<()> {
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    w.write_all(data)
}

// ── Tmessage encoding ─────────────────────────────────────────────────────

impl Tmessage {
    fn type_byte(&self) -> u8 {
        match self {
            Self::Version(_) => TVERSION,
            Self::Auth(_) => TAUTH,
            Self::Attach(_) => TATTACH,
            Self::Flush(_) => TFLUSH,
            Self::Walk(_) => TWALK,
            Self::Lopen(_) => TLOPEN,
            Self::Read(_) => TREAD,
            Self::Write(_) => TWRITE,
            Self::Clunk(_) => TCLUNK,
            Self::Readdir(_) => TREADDIR,
            Self::GetAttr(_) => TGETATTR,
            Self::Statfs(_) => TSTATFS,
        }
    }

    fn encode_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Self::Version(v) => {
                w.write_all(&v.msize.to_le_bytes())?;
                write_string(w, &v.version)
            }
            Self::Auth(a) => {
                w.write_all(&a.afid.to_le_bytes())?;
                write_string(w, &a.uname)?;
                write_string(w, &a.aname)?;
                w.write_all(&a.n_uname.to_le_bytes())
            }
            Self::Attach(a) => {
                w.write_all(&a.fid.to_le_bytes())?;
                w.write_all(&a.afid.to_le_bytes())?;
                write_string(w, &a.uname)?;
                write_string(w, &a.aname)?;
                w.write_all(&a.n_uname.to_le_bytes())
            }
            Self::Flush(f) => w.write_all(&f.oldtag.to_le_bytes()),
            Self::Walk(tw) => {
                w.write_all(&tw.fid.to_le_bytes())?;
                w.write_all(&tw.newfid.to_le_bytes())?;
                w.write_all(&(tw.wnames.len() as u16).to_le_bytes())?;
                for name in &tw.wnames {
                    write_string(w, name)?;
                }
                Ok(())
            }
            Self::Lopen(o) => {
                w.write_all(&o.fid.to_le_bytes())?;
                w.write_all(&o.flags.to_le_bytes())
            }
            Self::Read(r) => {
                w.write_all(&r.fid.to_le_bytes())?;
                w.write_all(&r.offset.to_le_bytes())?;
                w.write_all(&r.count.to_le_bytes())
            }
            Self::Write(tw) => {
                w.write_all(&tw.fid.to_le_bytes())?;
                w.write_all(&tw.offset.to_le_bytes())?;
                write_data(w, &tw.data)
            }
            Self::Clunk(c) => w.write_all(&c.fid.to_le_bytes()),
            Self::Readdir(r) => {
                w.write_all(&r.fid.to_le_bytes())?;
                w.write_all(&r.offset.to_le_bytes())?;
                w.write_all(&r.count.to_le_bytes())
            }
            Self::GetAttr(g) => {
                w.write_all(&g.fid.to_le_bytes())?;
                w.write_all(&g.request_mask.to_le_bytes())
            }
            Self::Statfs(s) => w.write_all(&s.fid.to_le_bytes()),
        }
    }

    fn decode_payload<R: Read>(ty: u8, r: &mut R) -> io::Result<Self> {
        match ty {
            TVERSION => Ok(Self::Version(Tversion {
                msize: read_u32(r)?,
                version: read_string(r)?,
            })),
            TAUTH => Ok(Self::Auth(Tauth {
                afid: read_u32(r)?,
                uname: read_string(r)?,
                aname: read_string(r)?,
                n_uname: read_u32(r)?,
            })),
            TATTACH => Ok(Self::Attach(Tattach {
                fid: read_u32(r)?,
                afid: read_u32(r)?,
                uname: read_string(r)?,
                aname: read_string(r)?,
                n_uname: read_u32(r)?,
            })),
            TFLUSH => Ok(Self::Flush(Tflush { oldtag: read_u16(r)? })),
            TWALK => {
                let fid = read_u32(r)?;
                let newfid = read_u32(r)?;
                let nwnames = read_u16(r)?;
                let mut wnames = Vec::with_capacity(nwnames as usize);
                for _ in 0..nwnames {
                    wnames.push(read_string(r)?);
                }
                Ok(Self::Walk(Twalk { fid, newfid, wnames }))
            }
            TLOPEN => Ok(Self::Lopen(Tlopen {
                fid: read_u32(r)?,
                flags: read_u32(r)?,
            })),
            TREAD => Ok(Self::Read(Tread {
                fid: read_u32(r)?,
                offset: read_u64(r)?,
                count: read_u32(r)?,
            })),
            TWRITE => Ok(Self::Write(Twrite {
                fid: read_u32(r)?,
                offset: read_u64(r)?,
                data: read_data(r)?,
            })),
            TCLUNK => Ok(Self::Clunk(Tclunk { fid: read_u32(r)? })),
            TREADDIR => Ok(Self::Readdir(Treaddir {
                fid: read_u32(r)?,
                offset: read_u64(r)?,
                count: read_u32(r)?,
            })),
            TGETATTR => Ok(Self::GetAttr(Tgetattr {
                fid: read_u32(r)?,
                request_mask: read_u64(r)?,
            })),
            TSTATFS => Ok(Self::Statfs(Tstatfs { fid: read_u32(r)? })),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown T-message type {ty}"),
            )),
        }
    }
}

// ── Rmessage encoding ─────────────────────────────────────────────────────

impl Rmessage {
    fn type_byte(&self) -> u8 {
        match self {
            Self::Version(_) => RVERSION,
            Self::Attach(_) => RATTACH,
            Self::Flush => RFLUSH,
            Self::Walk(_) => RWALK,
            Self::Lopen(_) => RLOPEN,
            Self::Read(_) => RREAD,
            Self::Write(_) => RWRITE,
            Self::Clunk => RCLUNK,
            Self::Readdir(_) => RREADDIR,
            Self::GetAttr(_) => RGETATTR,
            Self::Statfs(_) => RSTATFS,
            Self::Lerror(_) => RLERROR,
        }
    }

    fn encode_payload<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Self::Version(v) => {
                w.write_all(&v.msize.to_le_bytes())?;
                write_string(w, &v.version)
            }
            Self::Attach(a) => a.qid.encode(w),
            Self::Flush => Ok(()),
            Self::Walk(rw) => {
                w.write_all(&(rw.wqids.len() as u16).to_le_bytes())?;
                for qid in &rw.wqids {
                    qid.encode(w)?;
                }
                Ok(())
            }
            Self::Lopen(o) => {
                o.qid.encode(w)?;
                w.write_all(&o.iounit.to_le_bytes())
            }
            Self::Read(r) => write_data(w, &r.data),
            Self::Write(rw) => w.write_all(&rw.count.to_le_bytes()),
            Self::Clunk => Ok(()),
            Self::Readdir(r) => write_data(w, &r.data),
            Self::GetAttr(g) => {
                w.write_all(&g.valid.to_le_bytes())?;
                g.qid.encode(w)?;
                w.write_all(&g.mode.to_le_bytes())?;
                w.write_all(&g.uid.to_le_bytes())?;
                w.write_all(&g.gid.to_le_bytes())?;
                w.write_all(&g.nlink.to_le_bytes())?;
                w.write_all(&g.rdev.to_le_bytes())?;
                w.write_all(&g.size.to_le_bytes())?;
                w.write_all(&g.blksize.to_le_bytes())?;
                w.write_all(&g.blocks.to_le_bytes())?;
                w.write_all(&g.atime_sec.to_le_bytes())?;
                w.write_all(&g.atime_nsec.to_le_bytes())?;
                w.write_all(&g.mtime_sec.to_le_bytes())?;
                w.write_all(&g.mtime_nsec.to_le_bytes())?;
                w.write_all(&g.ctime_sec.to_le_bytes())?;
                w.write_all(&g.ctime_nsec.to_le_bytes())?;
                w.write_all(&g.btime_sec.to_le_bytes())?;
                w.write_all(&g.btime_nsec.to_le_bytes())?;
                w.write_all(&g.gen.to_le_bytes())?;
                w.write_all(&g.data_version.to_le_bytes())
            }
            Self::Statfs(s) => {
                w.write_all(&s.ty.to_le_bytes())?;
                w.write_all(&s.bsize.to_le_bytes())?;
                w.write_all(&s.blocks.to_le_bytes())?;
                w.write_all(&s.bfree.to_le_bytes())?;
                w.write_all(&s.bavail.to_le_bytes())?;
                w.write_all(&s.files.to_le_bytes())?;
                w.write_all(&s.ffree.to_le_bytes())?;
                w.write_all(&s.fsid.to_le_bytes())?;
                w.write_all(&s.namelen.to_le_bytes())
            }
            Self::Lerror(e) => w.write_all(&e.ecode.to_le_bytes()),
        }
    }

    fn decode_payload<R: Read>(ty: u8, r: &mut R) -> io::Result<Self> {
        match ty {
            RVERSION => Ok(Self::Version(Rversion {
                msize: read_u32(r)?,
                version: read_string(r)?,
            })),
            RATTACH => Ok(Self::Attach(Rattach { qid: Qid::decode(r)? })),
            RFLUSH => Ok(Self::Flush),
            RWALK => {
                let nwqids = read_u16(r)?;
                let mut wqids = Vec::with_capacity(nwqids as usize);
                for _ in 0..nwqids {
                    wqids.push(Qid::decode(r)?);
                }
                Ok(Self::Walk(Rwalk { wqids }))
            }
            RLOPEN => Ok(Self::Lopen(Rlopen {
                qid: Qid::decode(r)?,
                iounit: read_u32(r)?,
            })),
            RREAD => Ok(Self::Read(Rread { data: read_data(r)? })),
            RWRITE => Ok(Self::Write(Rwrite { count: read_u32(r)? })),
            RCLUNK => Ok(Self::Clunk),
            RREADDIR => Ok(Self::Readdir(Rreaddir { data: read_data(r)? })),
            RGETATTR => Ok(Self::GetAttr(Rgetattr {
                valid: read_u64(r)?,
                qid: Qid::decode(r)?,
                mode: read_u32(r)?,
                uid: read_u32(r)?,
                gid: read_u32(r)?,
                nlink: read_u64(r)?,
                rdev: read_u64(r)?,
                size: read_u64(r)?,
                blksize: read_u64(r)?,
                blocks: read_u64(r)?,
                atime_sec: read_u64(r)?,
                atime_nsec: read_u64(r)?,
                mtime_sec: read_u64(r)?,
                mtime_nsec: read_u64(r)?,
                ctime_sec: read_u64(r)?,
                ctime_nsec: read_u64(r)?,
                btime_sec: read_u64(r)?,
                btime_nsec: read_u64(r)?,
                gen: read_u64(r)?,
                data_version: read_u64(r)?,
            })),
            RSTATFS => Ok(Self::Statfs(Rstatfs {
                ty: read_u32(r)?,
                bsize: read_u32(r)?,
                blocks: read_u64(r)?,
                bfree: read_u64(r)?,
                bavail: read_u64(r)?,
                files: read_u64(r)?,
                ffree: read_u64(r)?,
                fsid: read_u64(r)?,
                namelen: read_u32(r)?,
            })),
            RLERROR => Ok(Self::Lerror(Rlerror { ecode: read_u32(r)? })),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown R-message type {ty}"),
            )),
        }
    }
}

// ── Frame encode/decode ───────────────────────────────────────────────────

impl Tframe {
    /// Encode this frame to a byte buffer.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        // Encode payload first to know its size.
        let mut payload = Vec::new();
        self.msg.encode_payload(&mut payload)?;
        // Frame: size(4) + type(1) + tag(2) + payload
        let size = (4 + 1 + 2 + payload.len()) as u32;
        let mut buf = Vec::with_capacity(size as usize);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.push(self.msg.type_byte());
        buf.extend_from_slice(&self.tag.to_le_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Decode a Tframe from a byte slice that contains a complete frame.
    ///
    /// Returns the frame and number of bytes consumed, or `None` if the
    /// buffer does not yet contain a complete frame.
    pub fn decode(buf: &[u8]) -> io::Result<Option<(Self, usize)>> {
        if buf.len() < 4 {
            return Ok(None);
        }
        let size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if size < 7 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "9P frame too small",
            ));
        }
        if buf.len() < size {
            return Ok(None);
        }
        let mut cursor = Cursor::new(&buf[4..size]);
        let ty = read_u8(&mut cursor)?;
        let tag = read_u16(&mut cursor)?;
        let msg = Tmessage::decode_payload(ty, &mut cursor)?;
        Ok(Some((Self { tag, msg }, size)))
    }
}

impl Rframe {
    /// Encode this frame to a byte buffer.
    pub fn encode(&self) -> io::Result<Vec<u8>> {
        let mut payload = Vec::new();
        self.msg.encode_payload(&mut payload)?;
        let size = (4 + 1 + 2 + payload.len()) as u32;
        let mut buf = Vec::with_capacity(size as usize);
        buf.extend_from_slice(&size.to_le_bytes());
        buf.push(self.msg.type_byte());
        buf.extend_from_slice(&self.tag.to_le_bytes());
        buf.extend_from_slice(&payload);
        Ok(buf)
    }

    /// Decode an Rframe from a byte slice that contains a complete frame.
    ///
    /// Returns the frame and number of bytes consumed, or `None` if the
    /// buffer does not yet contain a complete frame.
    pub fn decode(buf: &[u8]) -> io::Result<Option<(Self, usize)>> {
        if buf.len() < 4 {
            return Ok(None);
        }
        let size = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        if size < 7 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "9P frame too small",
            ));
        }
        if buf.len() < size {
            return Ok(None);
        }
        let mut cursor = Cursor::new(&buf[4..size]);
        let ty = read_u8(&mut cursor)?;
        let tag = read_u16(&mut cursor)?;
        let msg = Rmessage::decode_payload(ty, &mut cursor)?;
        Ok(Some((Self { tag, msg }, size)))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_tversion() {
        let frame = Tframe {
            tag: 0xFFFF,
            msg: Tmessage::Version(Tversion {
                msize: 8192,
                version: "9P2000.L".into(),
            }),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, consumed) = Tframe::decode(&encoded).unwrap().unwrap();
        assert_eq!(consumed, encoded.len());
        assert_eq!(decoded.tag, 0xFFFF);
        match decoded.msg {
            Tmessage::Version(v) => {
                assert_eq!(v.msize, 8192);
                assert_eq!(v.version, "9P2000.L");
            }
            other => panic!("expected Version, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_rversion() {
        let frame = Rframe {
            tag: 0xFFFF,
            msg: Rmessage::Version(Rversion {
                msize: 8192,
                version: "9P2000.L".into(),
            }),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, consumed) = Rframe::decode(&encoded).unwrap().unwrap();
        assert_eq!(consumed, encoded.len());
        match decoded.msg {
            Rmessage::Version(v) => {
                assert_eq!(v.msize, 8192);
                assert_eq!(v.version, "9P2000.L");
            }
            other => panic!("expected Version, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_twalk() {
        let frame = Tframe {
            tag: 1,
            msg: Tmessage::Walk(Twalk {
                fid: 0,
                newfid: 1,
                wnames: vec!["srv".into(), "model".into(), "status".into()],
            }),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Tframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Tmessage::Walk(w) => {
                assert_eq!(w.fid, 0);
                assert_eq!(w.newfid, 1);
                assert_eq!(w.wnames, vec!["srv", "model", "status"]);
            }
            other => panic!("expected Walk, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_rwalk() {
        let frame = Rframe {
            tag: 1,
            msg: Rmessage::Walk(Rwalk {
                wqids: vec![
                    Qid { ty: QTDIR, version: 0, path: 1 },
                    Qid { ty: QTDIR, version: 0, path: 2 },
                    Qid { ty: QTFILE, version: 0, path: 3 },
                ],
            }),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Rframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Rmessage::Walk(w) => {
                assert_eq!(w.wqids.len(), 3);
                assert_eq!(w.wqids[2].ty, QTFILE);
                assert_eq!(w.wqids[2].path, 3);
            }
            other => panic!("expected Walk, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_tread_rread() {
        let tframe = Tframe {
            tag: 5,
            msg: Tmessage::Read(Tread { fid: 3, offset: 0, count: 4096 }),
        };
        let encoded = tframe.encode().unwrap();
        let (decoded, _) = Tframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Tmessage::Read(r) => {
                assert_eq!(r.fid, 3);
                assert_eq!(r.offset, 0);
                assert_eq!(r.count, 4096);
            }
            other => panic!("expected Read, got {:?}", other),
        }

        let rframe = Rframe {
            tag: 5,
            msg: Rmessage::Read(Rread { data: b"hello world".to_vec() }),
        };
        let encoded = rframe.encode().unwrap();
        let (decoded, _) = Rframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Rmessage::Read(r) => assert_eq!(r.data, b"hello world"),
            other => panic!("expected Read, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_twrite_rwrite() {
        let tframe = Tframe {
            tag: 6,
            msg: Tmessage::Write(Twrite {
                fid: 2,
                offset: 0,
                data: b"set temperature 0.7".to_vec(),
            }),
        };
        let encoded = tframe.encode().unwrap();
        let (decoded, _) = Tframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Tmessage::Write(w) => {
                assert_eq!(w.fid, 2);
                assert_eq!(w.data, b"set temperature 0.7");
            }
            other => panic!("expected Write, got {:?}", other),
        }

        let rframe = Rframe {
            tag: 6,
            msg: Rmessage::Write(Rwrite { count: 19 }),
        };
        let encoded = rframe.encode().unwrap();
        let (decoded, _) = Rframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Rmessage::Write(w) => assert_eq!(w.count, 19),
            other => panic!("expected Write, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_lerror() {
        let frame = Rframe {
            tag: 3,
            msg: Rmessage::Lerror(Rlerror { ecode: 2 }), // ENOENT
        };
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Rframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Rmessage::Lerror(e) => assert_eq!(e.ecode, 2),
            other => panic!("expected Lerror, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_tattach() {
        let frame = Tframe {
            tag: 0,
            msg: Tmessage::Attach(Tattach {
                fid: 0,
                afid: u32::MAX,
                uname: "nobody".into(),
                aname: "".into(),
                n_uname: u32::MAX,
            }),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Tframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Tmessage::Attach(a) => {
                assert_eq!(a.fid, 0);
                assert_eq!(a.uname, "nobody");
            }
            other => panic!("expected Attach, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_rgetattr() {
        let frame = Rframe {
            tag: 10,
            msg: Rmessage::GetAttr(Rgetattr {
                valid: 0x7FF,
                qid: Qid { ty: QTFILE, version: 1, path: 42 },
                mode: 0o100444,
                uid: 1000,
                gid: 1000,
                nlink: 1,
                rdev: 0,
                size: 256,
                blksize: 4096,
                blocks: 1,
                atime_sec: 1711700000,
                atime_nsec: 0,
                mtime_sec: 1711700000,
                mtime_nsec: 0,
                ctime_sec: 1711700000,
                ctime_nsec: 0,
                btime_sec: 0,
                btime_nsec: 0,
                gen: 0,
                data_version: 0,
            }),
        };
        let encoded = frame.encode().unwrap();
        let (decoded, _) = Rframe::decode(&encoded).unwrap().unwrap();
        match decoded.msg {
            Rmessage::GetAttr(g) => {
                assert_eq!(g.size, 256);
                assert_eq!(g.qid.path, 42);
                assert_eq!(g.mode, 0o100444);
            }
            other => panic!("expected GetAttr, got {:?}", other),
        }
    }

    #[test]
    fn roundtrip_dirent() {
        let d = Dirent {
            qid: Qid { ty: QTFILE, version: 0, path: 7 },
            offset: 1,
            ty: QTFILE,
            name: "status".into(),
        };
        let mut buf = Vec::new();
        d.encode(&mut buf).unwrap();
        let decoded = Dirent::decode(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.name, "status");
        assert_eq!(decoded.qid.path, 7);
    }

    #[test]
    fn incomplete_buffer_returns_none() {
        assert!(Tframe::decode(&[0, 0, 0]).unwrap().is_none());
        assert!(Rframe::decode(&[0, 0, 0]).unwrap().is_none());

        // Size says 100 but buffer has only 10.
        let buf = [100u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert!(Tframe::decode(&buf).unwrap().is_none());
    }

    #[test]
    fn too_small_frame_errors() {
        // Size = 3, which is less than minimum 7.
        let buf = [3u8, 0, 0, 0, 0, 0, 0];
        assert!(Tframe::decode(&buf).is_err());
    }
}
