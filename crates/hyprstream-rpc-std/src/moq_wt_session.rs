//! `web_transport_trait::Session` over a browser `web_sys::WebTransport`.
//!
//! moq-net is generic over [`web_transport_trait::Session`] (the same trait
//! `web-transport-quinn` implements on native). No published `web-transport-wasm`
//! version implements that trait, so this module adapts `web-transport-wasm`'s
//! concrete `Session`/`SendStream`/`RecvStream` (which do the messy WHATWG-stream
//! bridging over `web_sys::WebTransport`) to the trait moq-net consumes.
//!
//! wasm-only: `web_sys::WebTransport` is `!Sync`, which is why moq-net's
//! `SessionInner` bound is relaxed on wasm (see the `third_party/moq-net` fork).
#![cfg(target_arch = "wasm32")]

use bytes::Bytes;
use web_transport_wasm as wtw;

/// Error wrapper implementing [`web_transport_trait::Error`] over `wtw::Error`.
#[derive(Debug)]
pub struct WtError(pub wtw::Error);

impl std::fmt::Display for WtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for WtError {}

impl web_transport_trait::Error for WtError {
    fn session_error(&self) -> Option<(u32, String)> {
        // web-transport-wasm surfaces an application close code (u8) if any.
        self.0.code().map(|c| (c as u32, self.0.to_string()))
    }
}

impl From<wtw::Error> for WtError {
    fn from(e: wtw::Error) -> Self {
        WtError(e)
    }
}

/// Adapter implementing [`web_transport_trait::Session`] over `wtw::Session`.
#[derive(Clone)]
pub struct WtSession(pub wtw::Session);

impl web_transport_trait::Session for WtSession {
    type SendStream = WtSendStream;
    type RecvStream = WtRecvStream;
    type Error = WtError;

    async fn accept_uni(&self) -> Result<Self::RecvStream, Self::Error> {
        Ok(WtRecvStream(self.0.accept_uni().await?))
    }

    async fn accept_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), Self::Error> {
        let (s, r) = self.0.accept_bi().await?;
        Ok((WtSendStream(s), WtRecvStream(r)))
    }

    async fn open_bi(&self) -> Result<(Self::SendStream, Self::RecvStream), Self::Error> {
        let (s, r) = self.0.open_bi().await?;
        Ok((WtSendStream(s), WtRecvStream(r)))
    }

    async fn open_uni(&self) -> Result<Self::SendStream, Self::Error> {
        Ok(WtSendStream(self.0.open_uni().await?))
    }

    fn send_datagram(&self, payload: Bytes) -> Result<(), Self::Error> {
        // web-transport-wasm's send_datagram is async, but the trait requires a
        // sync return. Datagrams are best-effort/lossy by definition, and the moq
        // stream path does not depend on them, so fire-and-forget on the worker's
        // single JS thread rather than block.
        let session = self.0.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let _ = session.send_datagram(payload).await;
        });
        Ok(())
    }

    async fn recv_datagram(&self) -> Result<Bytes, Self::Error> {
        Ok(self.0.recv_datagram().await?)
    }

    fn max_datagram_size(&self) -> usize {
        // WebTransport does not expose a max datagram size; report a conservative
        // floor. moq streams don't use datagrams, so this is informational only.
        1200
    }

    fn protocol(&self) -> Option<&str> {
        self.0.protocol()
    }

    fn close(&self, code: u32, reason: &str) {
        self.0.close(code, reason);
    }

    async fn closed(&self) -> Self::Error {
        WtError(self.0.closed().await)
    }
}

/// Adapter over `wtw::SendStream`.
pub struct WtSendStream(pub wtw::SendStream);

impl web_transport_trait::SendStream for WtSendStream {
    type Error = WtError;

    async fn write(&mut self, buf: &[u8]) -> Result<usize, Self::Error> {
        // web-transport-wasm's write() writes the whole buffer (backpressured);
        // the trait's contract returns the number of bytes accepted.
        self.0.write(buf).await?;
        Ok(buf.len())
    }

    fn set_priority(&mut self, order: u8) {
        self.0.set_priority(order as i32);
    }

    fn finish(&mut self) -> Result<(), Self::Error> {
        self.0.finish()?;
        Ok(())
    }

    fn reset(&mut self, code: u32) {
        self.0.reset(&code.to_string());
    }

    async fn closed(&mut self) -> Result<(), Self::Error> {
        self.0.closed().await?;
        Ok(())
    }
}

/// Adapter over `wtw::RecvStream`.
pub struct WtRecvStream(pub wtw::RecvStream);

impl web_transport_trait::RecvStream for WtRecvStream {
    type Error = WtError;

    async fn read(&mut self, dst: &mut [u8]) -> Result<Option<usize>, Self::Error> {
        // web-transport-wasm read(max) returns at most `max` bytes as a fresh
        // chunk; copy into the caller's buffer to match the trait's read-into API.
        match self.0.read(dst.len()).await? {
            Some(chunk) => {
                let n = chunk.len().min(dst.len());
                dst[..n].copy_from_slice(&chunk[..n]);
                Ok(Some(n))
            }
            None => Ok(None),
        }
    }

    fn stop(&mut self, code: u32) {
        self.0.stop(&code.to_string());
    }

    async fn closed(&mut self) -> Result<(), Self::Error> {
        self.0.closed().await?;
        Ok(())
    }
}
