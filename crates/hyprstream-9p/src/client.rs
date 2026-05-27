//! 9P2000.L client — sends T-messages, receives R-messages.
//!
//! Transport-agnostic: the caller provides `send` and `recv` functions.
//! On wasm32, these would be backed by DMA ring buffers (SharedArrayBuffer).
//! On native, these could be backed by pipes, TCP, or QUIC streams.

use std::cell::Cell;

use crate::msg::{self, Qid, Response};

/// Transport trait for sending/receiving 9P messages.
///
/// Implementations provide the actual byte transport (DMA ring, pipe, etc.).
/// Messages are already length-prefixed by the 9P serializer.
#[async_trait::async_trait(?Send)]
pub trait P9Transport {
    /// Send a complete 9P message (including length prefix).
    async fn send(&self, data: &[u8]) -> anyhow::Result<()>;

    /// Receive a complete 9P message (including length prefix).
    async fn recv(&self) -> anyhow::Result<Vec<u8>>;
}

/// 9P2000.L client.
///
/// Manages fid allocation, tag sequencing, and protocol handshake.
/// All operations are async — the transport determines the actual I/O.
pub struct P9Client<T: P9Transport> {
    transport: T,
    next_fid: Cell<u32>,
    next_tag: Cell<u16>,
    root_fid: u32,
    #[allow(dead_code)]
    msize: u32,
}

impl<T: P9Transport> P9Client<T> {
    /// Create a new client and perform version + attach handshake.
    pub async fn connect(transport: T, uname: &str, aname: &str) -> anyhow::Result<Self> {
        let client = Self {
            transport,
            next_fid: Cell::new(1), // 0 reserved for root
            next_tag: Cell::new(1), // 0 is NOTAG
            root_fid: 0,
            msize: 8192,
        };

        // Version negotiation
        let tag = client.alloc_tag();
        let msg = msg::tversion(tag, 65536, "9P2000.L");
        client.transport.send(&msg).await?;
        let resp = client.transport.recv().await?;
        let (_, response) = msg::parse_response(&resp)?;
        match response {
            Response::Version { msize, version } => {
                if version != "9P2000.L" {
                    anyhow::bail!("server does not support 9P2000.L: {version}");
                }
                // Update msize to what server accepts
                let _ = msize; // TODO: use negotiated msize
            }
            Response::Error { ecode } => anyhow::bail!("version failed: errno {ecode}"),
            _ => anyhow::bail!("unexpected response to Tversion"),
        }

        // Attach to root
        let tag = client.alloc_tag();
        let msg = msg::tattach(tag, 0, u32::MAX, uname, aname);
        client.transport.send(&msg).await?;
        let resp = client.transport.recv().await?;
        let (_, response) = msg::parse_response(&resp)?;
        match response {
            Response::Attach { qid: _ } => {}
            Response::Error { ecode } => anyhow::bail!("attach failed: errno {ecode}"),
            _ => anyhow::bail!("unexpected response to Tattach"),
        }

        Ok(client)
    }

    fn alloc_fid(&self) -> u32 {
        let fid = self.next_fid.get();
        self.next_fid.set(fid.wrapping_add(1));
        fid
    }

    fn alloc_tag(&self) -> u16 {
        let tag = self.next_tag.get();
        self.next_tag.set(tag.wrapping_add(1));
        if self.next_tag.get() == 0 { self.next_tag.set(1); } // skip NOTAG
        tag
    }

    /// Send a T-message and receive the R-message, checking for errors.
    async fn rpc(&self, msg: &[u8]) -> anyhow::Result<Response> {
        self.transport.send(msg).await?;
        let resp = self.transport.recv().await?;
        let (_, response) = msg::parse_response(&resp)?;
        if let Response::Error { ecode } = &response {
            anyhow::bail!("9P error: errno {ecode}");
        }
        Ok(response)
    }

    // ────────────────────────────────────────────────────────────────────────
    // Public operations
    // ────────────────────────────────────────────────────────────────────────

    /// Walk a path from root, returning a new fid.
    pub async fn walk(&self, components: &[&str]) -> anyhow::Result<(u32, Vec<Qid>)> {
        let newfid = self.alloc_fid();
        let tag = self.alloc_tag();
        let msg = msg::twalk(tag, self.root_fid, newfid, components);
        let resp = self.rpc(&msg).await?;
        match resp {
            Response::Walk { qids } => Ok((newfid, qids)),
            _ => anyhow::bail!("unexpected response to Twalk"),
        }
    }

    /// Open a fid for I/O.
    pub async fn open(&self, fid: u32, flags: u32) -> anyhow::Result<(Qid, u32)> {
        let tag = self.alloc_tag();
        let msg = msg::tlopen(tag, fid, flags);
        let resp = self.rpc(&msg).await?;
        match resp {
            Response::Lopen { qid, iounit } => Ok((qid, iounit)),
            _ => anyhow::bail!("unexpected response to Tlopen"),
        }
    }

    /// Read data from an open fid.
    pub async fn read(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<u8>> {
        let tag = self.alloc_tag();
        let msg = msg::tread(tag, fid, offset, count);
        let resp = self.rpc(&msg).await?;
        match resp {
            Response::Read { data } => Ok(data),
            _ => anyhow::bail!("unexpected response to Tread"),
        }
    }

    /// Write data to an open fid.
    pub async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> anyhow::Result<u32> {
        let tag = self.alloc_tag();
        let msg = msg::twrite(tag, fid, offset, data);
        let resp = self.rpc(&msg).await?;
        match resp {
            Response::Write { count } => Ok(count),
            _ => anyhow::bail!("unexpected response to Twrite"),
        }
    }

    /// Close a fid.
    pub async fn clunk(&self, fid: u32) -> anyhow::Result<()> {
        let tag = self.alloc_tag();
        let msg = msg::tclunk(tag, fid);
        self.rpc(&msg).await?;
        Ok(())
    }

    /// Get file attributes.
    pub async fn getattr(&self, fid: u32) -> anyhow::Result<(Qid, u32, u64, u64)> {
        let tag = self.alloc_tag();
        // P9_GETATTR_BASIC = 0x7ff
        let msg = msg::tgetattr(tag, fid, 0x7ff);
        let resp = self.rpc(&msg).await?;
        match resp {
            Response::Getattr { qid, mode, size, mtime_sec } => Ok((qid, mode, size, mtime_sec)),
            _ => anyhow::bail!("unexpected response to Tgetattr"),
        }
    }

    /// Read directory entries.
    pub async fn readdir(&self, fid: u32, offset: u64, count: u32) -> anyhow::Result<Vec<msg::DirEntryP9>> {
        let tag = self.alloc_tag();
        let msg = msg::treaddir(tag, fid, offset, count);
        let resp = self.rpc(&msg).await?;
        match resp {
            Response::Readdir { data } => msg::parse_readdir_entries(&data),
            _ => anyhow::bail!("unexpected response to Treaddir"),
        }
    }

    /// Get the root fid.
    pub fn root_fid(&self) -> u32 {
        self.root_fid
    }
}
