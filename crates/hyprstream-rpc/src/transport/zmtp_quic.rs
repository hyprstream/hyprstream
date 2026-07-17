//! ZMTP 3.1 over QUIC transport implementation.
//!
//! This module implements ZMTP (ZeroMQ Message Transport Protocol) version 3.1
//! running over QUIC streams. QUIC provides TLS 1.3 encryption at the transport
//! layer, so the ZMTP handshake uses the NULL mechanism (CurveZMQ not needed).
//!
//! # Protocol Mapping
//!
//! ```text
//! TCP connection     → QUIC connection (+ TLS 1.3)
//! TCP byte stream    → Single QUIC bidirectional stream (ordered, reliable)
//! ZMTP greeting      → Sent as first 64 bytes on the QUIC stream
//! ZMTP handshake     → NULL mechanism (TLS handles encryption)
//! ZMTP message frames → Identical framing on the QUIC stream
//! ```
//!
//! # ZMTP Frame Format
//!
//! ```text
//! Flags (1 byte):
//!   Bit 0 (LSB): MORE — 1 if more frames follow, 0 if last frame
//!   Bit 1:       LONG — 1 if size is 8-byte uint64, 0 if size is 1-byte uint8
//!   Bit 2:       COMMAND — 1 = command frame, 0 = message frame
//!   Bits 3-7:    Reserved (must be 0)
//!
//! Size:
//!   LONG=0: 1 byte  (uint8, frames 0–255 bytes)
//!   LONG=1: 8 bytes (uint64 big-endian, frames up to 2^64 bytes)
//!
//! Body: exactly `size` bytes
//! ```

use anyhow::{anyhow, bail, ensure, Result};
use bytes::Bytes;
use std::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::sync::Notify;
use tracing::{debug, error, trace};

// Re-export pure framing types from zmtp_framing (single source of truth).
pub use crate::zmtp_framing::{ZmqSocketType, GREETING_SIZE, ZMTP_VERSION_MAJOR, ZMTP_VERSION_MINOR, MECHANISM_NULL};
use crate::zmtp_framing;

/// ALPN protocol identifier for ZMTP over QUIC.
pub const ALPN_ZMTP3: &[u8] = b"zmtp3";

/// A parsed ZMTP frame (with `Bytes` data for async I/O efficiency).
#[derive(Debug, Clone)]
pub struct ZmtpFrame {
    pub more: bool,
    pub command: bool,
    pub data: Bytes,
}

#[cfg(test)]
impl ZmtpFrame {
    /// Convert from the owned framing type.
    fn from_owned(f: zmtp_framing::ZmtpFrame) -> Self {
        Self {
            more: f.more,
            command: f.command,
            data: Bytes::from(f.data),
        }
    }
}

/// A parsed ZMTP command (with `Bytes` body for async I/O efficiency).
#[derive(Debug, Clone)]
pub struct ZmtpCommand {
    pub name: String,
    pub body: Bytes,
}

impl ZmtpCommand {
    /// Convert from the owned framing type.
    fn from_owned(c: zmtp_framing::ZmtpCommand) -> Self {
        Self {
            name: c.name,
            body: Bytes::from(c.body),
        }
    }
}

/// ZMTP 3.1 framing over a byte stream.
///
/// Transport-only — no socket type semantics. Socket semantics live in
/// QuicReq/QuicRep/etc.
pub struct ZmtpStream<S> {
    stream: S,
}

impl<S: AsyncRead + AsyncWrite + Unpin> ZmtpStream<S> {
    /// Create a new ZMTP stream wrapper.
    pub fn new(stream: S) -> Self {
        Self { stream }
    }

    /// Perform ZMTP greeting + NULL mechanism handshake.
    ///
    /// # Arguments
    ///
    /// * `socket_type` - The ZMQ socket type (sent in READY metadata)
    /// * `is_server` - Whether this side is the server (affects READY order)
    ///
    /// # NULL Handshake Sequence (RFC 37)
    ///
    /// 1. Both sides send 64-byte greeting (can overlap)
    /// 2. Both sides read and validate peer greeting
    /// 3. CLIENT sends READY command first
    /// 4. Server reads client READY, validates, then sends READY
    pub async fn handshake(&mut self, socket_type: ZmqSocketType, is_server: bool) -> Result<()> {
        // 1. Build and send greeting (delegates to zmtp_framing)
        let greeting = zmtp_framing::build_greeting();
        self.stream.write_all(&greeting).await?;

        // 2. Read peer greeting
        let mut peer_greeting = [0u8; GREETING_SIZE];
        self.stream.read_exact(&mut peer_greeting).await?;
        zmtp_framing::validate_greeting(&peer_greeting)?;

        // 3. Asymmetric READY exchange (RFC 37: client sends first)
        let ready_metadata = zmtp_framing::build_ready_metadata(socket_type);

        if !is_server {
            self.send_command("READY", &ready_metadata).await?;
            let cmd = self.recv_command().await?;
            zmtp_framing::validate_ready_command(&zmtp_framing::ZmtpCommand {
                name: cmd.name.clone(),
                body: cmd.body.to_vec(),
            })?;
        } else {
            let cmd = self.recv_command().await?;
            zmtp_framing::validate_ready_command(&zmtp_framing::ZmtpCommand {
                name: cmd.name.clone(),
                body: cmd.body.to_vec(),
            })?;
            self.send_command("READY", &ready_metadata).await?;
        }

        debug!(socket_type = ?socket_type, is_server, "ZMTP handshake complete");
        Ok(())
    }

    /// Receive one ZMTP frame (message or command).
    pub async fn recv_frame(&mut self) -> Result<ZmtpFrame> {
        // Read flags byte
        let mut flags_buf = [0u8; 1];
        self.stream.read_exact(&mut flags_buf).await?;
        let flags = flags_buf[0];

        let more = (flags & 0x01) != 0;
        let long = (flags & 0x02) != 0;
        let command = (flags & 0x04) != 0;

        // Validate reserved bits
        ensure!(
            flags & 0xF8 == 0,
            "ZMTP frame has reserved bits set: 0x{:02X}",
            flags
        );

        // Read size
        let size: usize = if long {
            let mut buf = [0u8; 8];
            self.stream.read_exact(&mut buf).await?;
            let raw_size = u64::from_be_bytes(buf);
            ensure!(
                raw_size <= crate::zmtp_framing::MAX_FRAME_SIZE as u64,
                "frame size {} exceeds maximum {}",
                raw_size,
                crate::zmtp_framing::MAX_FRAME_SIZE
            );
            raw_size as usize
        } else {
            let mut buf = [0u8; 1];
            self.stream.read_exact(&mut buf).await?;
            buf[0] as usize
        };

        // Read body
        let mut data = vec![0u8; size];
        self.stream.read_exact(&mut data).await?;

        trace!(more, command, size, "Received ZMTP frame");

        Ok(ZmtpFrame {
            more,
            command,
            data: Bytes::from(data),
        })
    }

    /// Receive a command frame specifically (validates COMMAND bit set, MORE=0).
    pub async fn recv_command(&mut self) -> Result<ZmtpCommand> {
        let frame = self.recv_frame().await?;

        ensure!(frame.command, "Expected command frame, got message frame");
        ensure!(!frame.more, "Command frames must have MORE=0");

        ZmtpCommand::parse(&frame.data)
    }

    /// Send a ZMTP command frame (COMMAND bit set, MORE=0).
    ///
    /// Delegates to `zmtp_framing::encode_command` for single source of truth.
    pub async fn send_command(&mut self, name: &str, body: &[u8]) -> Result<()> {
        let encoded = crate::zmtp_framing::encode_command(name, body);
        self.stream.write_all(&encoded).await?;
        trace!(name, body_len = body.len(), "Sent ZMTP command");
        Ok(())
    }

    /// Send one ZMTP message frame (COMMAND=0).
    ///
    /// Delegates to `zmtp_framing::encode_frame` for single source of truth.
    pub async fn send_frame(&mut self, more: bool, data: &[u8]) -> Result<()> {
        let encoded = crate::zmtp_framing::encode_frame(more, false, data);
        self.stream.write_all(&encoded).await?;
        trace!(more, size = data.len(), "Sent ZMTP frame");
        Ok(())
    }

    /// Send one ZMQ multipart message using ZMTP flags.
    pub async fn send_multipart(&mut self, parts: &[Bytes]) -> Result<()> {
        for (i, part) in parts.iter().enumerate() {
            let more = i < parts.len() - 1;
            self.send_frame(more, part).await?;
        }
        trace!(num_parts = parts.len(), "Sent multipart message");
        Ok(())
    }

    /// Receive one ZMQ multipart message, assembling frames via MORE flag.
    pub async fn recv_multipart(&mut self) -> Result<Multipart> {
        let mut parts = Vec::new();

        loop {
            let frame = self.recv_frame().await?;

            // Message frames shouldn't have COMMAND bit set
            if frame.command {
                bail!("Received unexpected command frame in message stream");
            }

            let more = frame.more;
            parts.push(frame.data);

            if !more {
                break;
            }
        }

        trace!(num_parts = parts.len(), "Received multipart message");
        Ok(Multipart { parts })
    }
}

// Greeting, metadata, and command functions are now in crate::zmtp_framing.
// ZmtpStream delegates to those pure functions for encoding/decoding.

impl ZmtpCommand {
    /// Parse a command from frame body data.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let owned = zmtp_framing::ZmtpCommand::parse(data)?;
        Ok(Self::from_owned(owned))
    }
}

/// ZMQ multipart message (owned).
#[derive(Debug, Clone, Default)]
pub struct Multipart {
    /// Message parts.
    pub parts: Vec<Bytes>,
}

impl Multipart {
    /// Create an empty multipart message.
    pub fn new() -> Self {
        Self { parts: Vec::new() }
    }

    /// Create a multipart message from parts.
    pub fn from_parts(parts: Vec<Bytes>) -> Self {
        Self { parts }
    }

    /// Add a part to the message.
    pub fn push(&mut self, part: Bytes) {
        self.parts.push(part);
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Get number of parts.
    pub fn len(&self) -> usize {
        self.parts.len()
    }
}

// ============================================================================
// QUIC Socket Types (server-side)
// ============================================================================

fn owned_quic_server_config(tls: rustls::ServerConfig) -> Result<quinn::ServerConfig> {
    crate::transport::pq_provider::validate_internal_mesh_crypto_provider(
        tls.crypto_provider(),
    )?;
    Ok(quinn::ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(tls)?,
    )))
}

fn owned_quic_client_config(tls: rustls::ClientConfig) -> Result<quinn::ClientConfig> {
    crate::transport::pq_provider::validate_internal_mesh_crypto_provider(
        tls.crypto_provider(),
    )?;
    Ok(quinn::ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(tls)?,
    )))
}

/// Server-side REP socket over QUIC.
///
/// Accepts QUIC connections; each connection carries one ZmtpStream for ZMTP exchange.
/// QuicRep enforces the REP state machine (recv then send alternation).
pub struct QuicRep {
    endpoint: quinn::Endpoint,
}

impl QuicRep {
    /// Bind a QUIC REP socket to the given address.
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address to bind to
    /// * `tls` - TLS server configuration
    pub fn bind(addr: SocketAddr, tls: rustls::ServerConfig) -> Result<Self> {
        let endpoint = quinn::Endpoint::server(owned_quic_server_config(tls)?, addr)?;

        Ok(Self { endpoint })
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint
            .local_addr()
            .map_err(|e| anyhow!("Failed to get local address: {}", e))
    }

    /// Accept loop: accepts incoming connections and handles them.
    ///
    /// Each incoming QUIC connection can have multiple bidi streams, each
    /// representing one REQ/REP exchange.
    ///
    /// # Arguments
    ///
    /// * `handler` - Async function to handle each request
    /// * `shutdown` - Shutdown signal
    pub async fn accept_loop<H, Fut>(
        &self,
        handler: H,
        shutdown: Arc<Notify>,
    ) -> Result<()>
    where
        H: Fn(Multipart) -> Fut + Clone + Send + 'static,
        Fut: std::future::Future<Output = Result<Multipart>> + Send,
    {
        loop {
            tokio::select! {
                Some(incoming) = self.endpoint.accept() => {
                    let handler = handler.clone();
                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(conn) => {
                                if let Err(e) = Self::handle_connection(conn, handler).await {
                                    error!("Connection error: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("Incoming connection failed: {}", e);
                            }
                        }
                    });
                }
                _ = shutdown.notified() => {
                    debug!("QUIC accept loop shutting down");
                    break;
                }
            }
        }

        // Graceful shutdown: wait for connections to drain
        self.endpoint.wait_idle().await;
        Ok(())
    }

    /// Accept loop for RequestService handlers.
    ///
    /// This variant handles the full envelope processing pipeline:
    /// 1. ZMTP handshake
    /// 2. SignedEnvelope verification
    /// 3. Service handle_request dispatch
    /// 4. Signed response serialization
    ///
    /// Uses `spawn_local` because `RequestService` is `?Send`.
    ///
    /// # Arguments
    ///
    /// * `service` - RequestService implementation (wrapped in Rc)
    /// * `server_pubkey` - Server's Ed25519 verifying key
    /// * `signing_key` - Server's Ed25519 signing key
    /// * `nonce_cache` - Nonce cache for replay protection (wrapped in Arc)
    /// * `shutdown` - Shutdown signal
    pub async fn accept_loop_service<S>(
        &self,
        service: Rc<S>,
        server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
        shutdown: Arc<Notify>,
    ) -> Result<()>
    where
        S: crate::service::RequestService + 'static,
    {
        loop {
            tokio::select! {
                Some(incoming) = self.endpoint.accept() => {
                    let service = Rc::clone(&service);
                    let nonce_cache = Arc::clone(&nonce_cache);
                    let signing_key = signing_key.clone();  // Clone before spawn

                    // Use spawn_local because RequestService is ?Send
                    tokio::task::spawn_local(async move {
                        match incoming.await {
                            Ok(conn) => {
                                if let Err(e) = Self::handle_connection_service(
                                    conn,
                                    service,
                                    server_pubkey,
                                    signing_key,
                                    nonce_cache,
                                ).await {
                                    error!("Connection error: {}", e);
                                }
                            }
                            Err(e) => {
                                error!("Incoming connection failed: {}", e);
                            }
                        }
                    });
                }
                _ = shutdown.notified() => {
                    debug!("QUIC accept loop shutting down");
                    break;
                }
            }
        }

        // Graceful shutdown: wait for connections to drain
        self.endpoint.wait_idle().await;
        Ok(())
    }

    async fn handle_connection_service<S>(
        conn: quinn::Connection,
        service: Rc<S>,
        server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
    ) -> Result<()>
    where
        S: crate::service::RequestService + 'static,
    {
        let remote = conn.remote_address();
        debug!(?remote, "QUIC connection established");

        loop {
            // Accept bidi streams from this connection (one per REQ/REP exchange)
            match conn.accept_bi().await {
                Ok((send, recv)) => {
                    let service = Rc::clone(&service);
                    let nonce_cache = Arc::clone(&nonce_cache);
                    let signing_key = signing_key.clone();  // Clone before spawn

                    tokio::task::spawn_local(async move {
                        if let Err(e) = Self::handle_stream_service(
                            send,
                            recv,
                            service,
                            server_pubkey,
                            signing_key,
                            nonce_cache,
                        ).await {
                            debug!("Stream error: {}", e);
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                    debug!(?remote, "Connection closed by peer");
                    break;
                }
                Err(e) => {
                    debug!(?remote, "Connection error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_stream_service<S>(
        send: quinn::SendStream,
        recv: quinn::RecvStream,
        service: Rc<S>,
        _server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
    ) -> Result<()>
    where
        S: crate::service::RequestService + 'static,
    {
        let mut stream = ZmtpStream::new(QuicStream { send, recv });

        // ZMTP handshake (server side)
        stream.handshake(ZmqSocketType::Rep, true).await?;

        // Receive request (single-frame: SignedEnvelope bytes)
        let request = stream.recv_multipart().await?;
        if request.parts.is_empty() {
            return Err(anyhow!("empty request"));
        }

        let raw_bytes = &request.parts[0];

        // Process through envelope pipeline.
        // AnySigner: per-service keys mean each client signs with its own key,
        // not a shared root key. The envelope signature is still verified — we
        // just don't pin it to the server's own key. JWT claims + Casbin handle
        // authorization.
        // subsecond::call wraps dispatch for hot-patching during dev
        // INV-2 (#1042): this legacy ZMTP entry point terminates a QUIC
        // stream — an untrusted carrier even on a loopback address.
        let response_bytes = subsecond::call(|| crate::service::dispatch::process_request(
            raw_bytes,
            &*service,
            EnvelopeVerification::AnySigner,
            &signing_key,
            &nonce_cache,
            crate::transport::carrier::CarrierContext::quic(),
        )).await?;

        // Send response
        let response = Multipart {
            parts: vec![Bytes::from(response_bytes)],
        };
        stream.send_multipart(&response.parts).await?;

        // Signal end of response. Any streaming pump was already spawned inside
        // process_request (#186).
        stream.stream.send.finish()?;

        Ok(())
    }

    async fn handle_connection<H, Fut>(conn: quinn::Connection, handler: H) -> Result<()>
    where
        H: Fn(Multipart) -> Fut + Send + Clone + 'static,
        Fut: std::future::Future<Output = Result<Multipart>> + Send,
    {
        let remote = conn.remote_address();
        debug!(?remote, "QUIC connection established");

        loop {
            // Accept bidi streams from this connection (one per REQ/REP exchange)
            match conn.accept_bi().await {
                Ok((send, recv)) => {
                    let handler = handler.clone();
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_stream(send, recv, handler).await {
                            debug!("Stream error: {}", e);
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                    debug!(?remote, "Connection closed by peer");
                    break;
                }
                Err(e) => {
                    debug!(?remote, "Connection error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_stream<H, Fut>(
        send: quinn::SendStream,
        recv: quinn::RecvStream,
        handler: H,
    ) -> Result<()>
    where
        H: Fn(Multipart) -> Fut + Send,
        Fut: std::future::Future<Output = Result<Multipart>> + Send,
    {
        let mut stream = ZmtpStream::new(QuicStream { send, recv });

        // Perform ZMTP handshake as server
        stream.handshake(ZmqSocketType::Rep, true).await?;

        // REP: receive request, send reply (each request gets its own stream)
        let request = match stream.recv_multipart().await {
            Ok(msg) => msg,
            Err(e) => {
                // Stream closed is normal at end of request
                if e.downcast_ref::<std::io::Error>()
                    .is_some_and(|io_err| io_err.kind() == std::io::ErrorKind::UnexpectedEof)
                {
                    return Ok(());
                }
                return Err(e);
            }
        };

        // Process request
        let response = handler(request).await?;

        // Send reply
        stream.send_multipart(&response.parts).await?;

        Ok(())
    }
}

/// Client-side REQ socket over QUIC.
///
/// Opens one connection; each request opens a new bidi stream (ZMTP semantics).
#[allow(dead_code)] // endpoint held to keep connection alive
pub struct QuicReq {
    conn: quinn::Connection,
    endpoint: quinn::Endpoint,
}

impl QuicReq {
    /// Connect to a QUIC REP server.
    ///
    /// # Arguments
    ///
    /// * `addr` - Server address
    /// * `server_name` - Server hostname for TLS validation
    /// * `tls` - TLS client configuration
    pub async fn connect(
        addr: SocketAddr,
        server_name: &str,
        tls: rustls::ClientConfig,
    ) -> Result<Self> {
        let mut endpoint = quinn::Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(owned_quic_client_config(tls)?);

        let conn = endpoint
            .connect(addr, server_name)?
            .await
            .map_err(|e| anyhow!("QUIC connection failed: {}", e))?;

        Ok(Self { conn, endpoint })
    }

    /// Send request, receive reply. Each call opens a new bidi stream.
    ///
    /// QuicReq enforces REQ semantics: send then recv, one at a time.
    pub async fn request(&self, msg: &Multipart) -> Result<Multipart> {
        let (send, recv) = self.conn.open_bi().await?;

        let mut stream = ZmtpStream::new(QuicStream { send, recv });
        stream.handshake(ZmqSocketType::Req, false).await?;

        // Send request
        stream.send_multipart(&msg.parts).await?;

        // Signal end of request (important for REP to know we're done)
        stream.stream.send.finish()?;

        // Receive reply
        let response = stream.recv_multipart().await?;

        Ok(response)
    }

    /// Close the connection.
    pub fn close(&self) {
        self.conn.close(0u32.into(), b"client closing");
    }
}

impl Drop for QuicReq {
    fn drop(&mut self) {
        self.close();
        // Endpoint will be cleaned up when dropped
    }
}

// ============================================================================
// PUB/SUB Socket Types
// ============================================================================

/// Server-side XPUB socket over QUIC.
///
/// Accepts subscriber connections; routes publications by topic prefix.
/// Uses tokio::broadcast per topic for fan-out.
pub struct QuicXPub {
    endpoint: quinn::Endpoint,
    /// Topic prefix → broadcast sender; subscribers hold receivers
    topics: Arc<dashmap::DashMap<Vec<u8>, tokio::sync::broadcast::Sender<Bytes>>>,
}

impl QuicXPub {
    /// Bind a QUIC XPUB socket to the given address.
    pub fn bind(addr: SocketAddr, tls: rustls::ServerConfig) -> Result<Self> {
        let endpoint = quinn::Endpoint::server(owned_quic_server_config(tls)?, addr)?;

        Ok(Self {
            endpoint,
            topics: Arc::new(dashmap::DashMap::new()),
        })
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint
            .local_addr()
            .map_err(|e| anyhow!("Failed to get local address: {}", e))
    }

    /// Publish a message to all subscribers matching the topic prefix.
    ///
    /// If no subscribers exist for this topic, the message is dropped.
    pub fn publish(&self, topic: &[u8], data: Bytes) {
        // Find all matching topic prefixes and send to their broadcast channels
        for entry in self.topics.iter() {
            let (prefix, sender) = entry.pair();
            if topic.starts_with(prefix) {
                // Ignore send errors (no active receivers)
                let _ = sender.send(data.clone());
            }
        }
    }

    /// Accept loop: accepts incoming subscriber connections.
    pub async fn accept_loop(&self, shutdown: Arc<Notify>) -> Result<()> {
        loop {
            tokio::select! {
                Some(incoming) = self.endpoint.accept() => {
                    let topics = Arc::clone(&self.topics);
                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(conn) => {
                                if let Err(e) = Self::handle_subscriber(conn, topics).await {
                                    debug!("Subscriber connection error: {}", e);
                                }
                            }
                            Err(e) => {
                                debug!("Incoming subscriber connection failed: {}", e);
                            }
                        }
                    });
                }
                _ = shutdown.notified() => {
                    debug!("XPUB accept loop shutting down");
                    break;
                }
            }
        }

        self.endpoint.wait_idle().await;
        Ok(())
    }

    async fn handle_subscriber(
        conn: quinn::Connection,
        topics: Arc<dashmap::DashMap<Vec<u8>, tokio::sync::broadcast::Sender<Bytes>>>,
    ) -> Result<()> {
        let remote = conn.remote_address();
        debug!(?remote, "XPUB subscriber connection established");

        // Accept bidi streams (one per subscription)
        loop {
            match conn.accept_bi().await {
                Ok((send, recv)) => {
                    let topics = Arc::clone(&topics);
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_subscription(send, recv, topics).await {
                            debug!("Subscription error: {}", e);
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                    debug!(?remote, "Subscriber disconnected");
                    break;
                }
                Err(e) => {
                    debug!(?remote, "Subscriber connection error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_subscription(
        send: quinn::SendStream,
        recv: quinn::RecvStream,
        topics: Arc<dashmap::DashMap<Vec<u8>, tokio::sync::broadcast::Sender<Bytes>>>,
    ) -> Result<()> {
        let mut stream = ZmtpStream::new(QuicStream { send, recv });

        // Perform ZMTP handshake as server (XPUB)
        stream.handshake(ZmqSocketType::XPub, true).await?;

        // For simplicity: read one SUBSCRIBE command, then forward messages
        // Real implementation would track multiple subscriptions per connection
        let cmd = stream.recv_command().await?;

        if cmd.name != "SUBSCRIBE" {
            bail!("Expected SUBSCRIBE command, got {}", cmd.name);
        }

        let topic_prefix = cmd.body.to_vec();
        debug!(topic = ?std::str::from_utf8(&topic_prefix), "Subscriber subscribed");

        // Get or create broadcast channel for this topic
        let sender = topics
            .entry(topic_prefix.clone())
            .or_insert_with(|| {
                let (tx, _rx) = tokio::sync::broadcast::channel(256);
                tx
            })
            .clone();

        // Subscribe and forward messages to this stream
        let mut rx = sender.subscribe();
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Ok(data) => {
                            // Send as multipart: [topic, data]
                            let parts = vec![Bytes::copy_from_slice(&topic_prefix), data];
                            if stream.send_multipart(&parts).await.is_err() {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                // Also check for incoming commands (e.g., CANCEL)
                cmd = stream.recv_command() => {
                    match cmd {
                        Ok(c) if c.name == "CANCEL" => {
                            debug!(topic = ?std::str::from_utf8(&c.body), "Subscriber unsubscribed");
                            break;
                        }
                        Ok(c) => {
                            debug!("Unexpected command: {}", c.name);
                        }
                        Err(_) => break,
                    }
                }
            }
        }

        // Clean up topic entry if no receivers remain
        if sender.receiver_count() == 0 {
            topics.remove(&topic_prefix);
        }

        Ok(())
    }
}

/// Client-side SUB socket over QUIC.
///
/// Connects to QuicXPub; sends ZMTP subscription commands.
#[allow(dead_code)] // endpoint held to keep connection alive
pub struct QuicSub {
    conn: quinn::Connection,
    endpoint: quinn::Endpoint,
}

impl QuicSub {
    /// Connect to a QUIC XPUB server.
    pub async fn connect(
        addr: SocketAddr,
        server_name: &str,
        tls: rustls::ClientConfig,
    ) -> Result<Self> {
        let mut endpoint = quinn::Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(owned_quic_client_config(tls)?);

        let conn = endpoint
            .connect(addr, server_name)?
            .await
            .map_err(|e| anyhow!("QUIC connection failed: {}", e))?;

        Ok(Self { conn, endpoint })
    }

    /// Subscribe to a topic prefix.
    ///
    /// Returns a receiver for messages matching this prefix.
    pub async fn subscribe(&self, prefix: &[u8]) -> Result<tokio::sync::broadcast::Receiver<Bytes>> {
        let (send, recv_stream) = self.conn.open_bi().await?;
        let mut stream = ZmtpStream::new(QuicStream { send, recv: recv_stream });

        // Perform ZMTP handshake as client (SUB)
        stream.handshake(ZmqSocketType::Sub, false).await?;

        // Send SUBSCRIBE command
        stream.send_command("SUBSCRIBE", prefix).await?;

        // Create a broadcast channel to forward received messages
        let (tx, rx) = tokio::sync::broadcast::channel(256);

        // Spawn a task to receive messages and forward them
        tokio::spawn(async move {
            while let Ok(msg) = stream.recv_multipart().await {
                // parts[0] = topic prefix, parts[1] = data payload
                let data = msg.parts.get(1).or_else(|| msg.parts.first());
                if let Some(data) = data {
                    let _ = tx.send(data.clone());
                }
            }
        });

        Ok(rx)
    }

    /// Unsubscribe from a topic prefix.
    pub async fn unsubscribe(&self, prefix: &[u8]) -> Result<()> {
        let (send, recv) = self.conn.open_bi().await?;
        let mut stream = ZmtpStream::new(QuicStream { send, recv });

        // Perform ZMTP handshake as client (SUB)
        stream.handshake(ZmqSocketType::Sub, false).await?;

        // Send CANCEL command
        stream.send_command("CANCEL", prefix).await?;

        Ok(())
    }

    /// Close the connection.
    pub fn close(&self) {
        self.conn.close(0u32.into(), b"subscriber closing");
    }
}

impl Drop for QuicSub {
    fn drop(&mut self) {
        self.close();
        // Endpoint will be cleaned up when dropped
    }
}

// ============================================================================
// PUSH/PULL Socket Types
// ============================================================================

/// PUSH socket: opens connection, sends frames on a persistent bidi stream.
#[allow(dead_code)] // endpoint held to keep connection alive
pub struct QuicPush {
    conn: quinn::Connection,
    endpoint: quinn::Endpoint,
}

impl QuicPush {
    /// Connect to a QUIC PULL server.
    pub async fn connect(
        addr: SocketAddr,
        server_name: &str,
        tls: rustls::ClientConfig,
    ) -> Result<Self> {
        let mut endpoint = quinn::Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(owned_quic_client_config(tls)?);

        let conn = endpoint
            .connect(addr, server_name)?
            .await
            .map_err(|e| anyhow!("QUIC connection failed: {}", e))?;

        Ok(Self { conn, endpoint })
    }

    /// Send a message to the PULL socket.
    pub async fn send(&self, msg: &Multipart) -> Result<()> {
        let (send, recv) = self.conn.open_bi().await?;

        let mut stream = ZmtpStream::new(QuicStream { send, recv });
        stream.handshake(ZmqSocketType::Push, false).await?;
        stream.send_multipart(&msg.parts).await?;
        stream.stream.send.finish()?;

        Ok(())
    }

    /// Close the connection.
    pub fn close(&self) {
        self.conn.close(0u32.into(), b"push closing");
    }
}

impl Drop for QuicPush {
    fn drop(&mut self) {
        self.close();
        // Endpoint will be cleaned up when dropped
    }
}

/// PULL socket: accepts connections from multiple PUSHers, fans in to single receiver.
pub struct QuicPull {
    endpoint: quinn::Endpoint,
    /// Incoming frames from all PUSH connections forwarded here
    tx: tokio::sync::mpsc::UnboundedSender<Multipart>,
    /// Receiver for messages (exposed to user)
    rx: Option<tokio::sync::mpsc::UnboundedReceiver<Multipart>>,
}

impl QuicPull {
    /// Bind a QUIC PULL socket to the given address.
    pub fn bind(addr: SocketAddr, tls: rustls::ServerConfig) -> Result<Self> {
        let endpoint = quinn::Endpoint::server(owned_quic_server_config(tls)?, addr)?;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        Ok(Self {
            endpoint,
            tx,
            rx: Some(rx),
        })
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint
            .local_addr()
            .map_err(|e| anyhow!("Failed to get local address: {}", e))
    }

    /// Receive the next message.
    pub async fn recv(&mut self) -> Option<Multipart> {
        self.rx.as_mut()?.recv().await
    }

    /// Accept loop: accepts incoming push connections.
    pub async fn accept_loop(&self, shutdown: Arc<Notify>) -> Result<()> {
        loop {
            tokio::select! {
                Some(incoming) = self.endpoint.accept() => {
                    let tx = self.tx.clone();
                    tokio::spawn(async move {
                        match incoming.await {
                            Ok(conn) => {
                                if let Err(e) = Self::handle_pusher(conn, tx).await {
                                    debug!("Pusher connection error: {}", e);
                                }
                            }
                            Err(e) => {
                                debug!("Incoming pusher connection failed: {}", e);
                            }
                        }
                    });
                }
                _ = shutdown.notified() => {
                    debug!("PULL accept loop shutting down");
                    break;
                }
            }
        }

        self.endpoint.wait_idle().await;
        Ok(())
    }

    async fn handle_pusher(
        conn: quinn::Connection,
        tx: tokio::sync::mpsc::UnboundedSender<Multipart>,
    ) -> Result<()> {
        let remote = conn.remote_address();
        debug!(?remote, "Pusher connection established");

        // Accept bidi streams (one per message)
        loop {
            match conn.accept_bi().await {
                Ok((send, recv)) => {
                    let tx = tx.clone();
                    tokio::spawn(async move {
                        let mut stream = ZmtpStream::new(QuicStream { send, recv });
                        if stream.handshake(ZmqSocketType::Pull, true).await.is_err() {
                            return;
                        }
                        if let Ok(msg) = stream.recv_multipart().await {
                            let _ = tx.send(msg);
                        }
                    });
                }
                Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                    debug!(?remote, "Pusher disconnected");
                    break;
                }
                Err(e) => {
                    debug!(?remote, "Pusher connection error: {}", e);
                    break;
                }
            }
        }

        Ok(())
    }
}

/// Wrapper for Quinn bidi stream to implement AsyncRead/AsyncWrite.
struct QuicStream {
    send: quinn::SendStream,
    recv: quinn::RecvStream,
}

impl AsyncRead for QuicStream {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.recv).poll_read(cx, buf)
    }
}

impl AsyncWrite for QuicStream {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        std::pin::Pin::new(&mut self.send)
            .poll_write(cx, buf)
            .map_err(std::io::Error::other)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.send)
            .poll_flush(cx)
            .map_err(std::io::Error::other)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        std::pin::Pin::new(&mut self.send)
            .poll_shutdown(cx)
            .map_err(std::io::Error::other)
    }
}

// ============================================================================
// QuicServiceLoop - QUIC Transport Service Loop
// ============================================================================

/// Service loop for hosting RequestService over QUIC transport.
///
/// This struct wraps a `QuicRep` socket and a `RequestService` implementation,
/// running the QUIC accept loop in a dedicated thread with its own tokio runtime.
///
/// # Threading Model
///
/// `RequestService` is `#[async_trait(?Send)]` - handlers are not `Send`. This requires:
/// - Single-threaded runtime (`new_current_thread`)
/// - `LocalSet` for non-Send futures
/// - `spawn_local` instead of `tokio::spawn`
/// - Service wrapped in `Rc<S>` (not `Arc<S>`)
///
/// # Usage
///
/// ```ignore
/// use hyprstream_rpc::transport::zmtp_quic::{QuicRep, QuicServiceLoop};
/// use hyprstream_rpc::service::spawner::{Spawnable, DualSpawnable};
///
/// let (tls, cert_der) = server_tls_self_signed("hyprstream.local")?;
/// let quic_rep = QuicRep::bind(addr, tls)?;
/// let service = MyService::new();
///
/// let quic_loop = QuicServiceLoop::new(quic_rep, service, cert_der);
///
/// // Use with DualSpawnable to run alongside ZMQ:
/// let dual = DualSpawnable::new(zmq_loop, quic_loop);
/// Ok(Box::new(dual))
/// ```
pub struct QuicServiceLoop<S>
where
    S: crate::service::RequestService + Send + Sync + 'static,
{
    rep: QuicRep,
    service: S,
    /// Server leaf certificate DER bytes — used to pin the registered QUIC
    /// endpoint (self-signed, so clients pin the SHA-256 rather than CA-validate).
    cert_der: Vec<u8>,
    /// Service name
    name: String,
    /// QUIC endpoint address
    addr: SocketAddr,
}

impl<S> QuicServiceLoop<S>
where
    S: crate::service::RequestService + Send + Sync + 'static,
{
    /// Create a new QUIC service loop.
    pub fn new(rep: QuicRep, service: S, cert_der: Vec<u8>) -> Result<Self> {
        let addr = rep.local_addr()?;
        let name = service.name().to_owned();
        Ok(Self {
            rep,
            service,
            cert_der,
            name,
            addr,
        })
    }

    /// Get the QUIC endpoint address.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }
}

impl<S> crate::service::Spawnable for QuicServiceLoop<S>
where
    S: crate::service::RequestService + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn registrations(&self) -> Vec<(crate::registry::SocketKind, crate::transport::TransportConfig)> {
        // Register the QUIC endpoint *pinned* by the self-signed leaf cert's
        // SHA-256 — a plain (WebPki) registration would make clients CA-validate
        // a self-signed cert and fail.
        let pin = crate::transport::quinn_transport::cert_sha256(&self.cert_der);
        vec![(
            crate::registry::SocketKind::Rep,
            crate::transport::TransportConfig::quic_pinned(self.addr, "hyprstream.local", pin),
        )]
    }

    fn run(
        self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> crate::error::Result<()> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| crate::error::RpcError::Other(format!("runtime build: {}", e)))?;

        let local = tokio::task::LocalSet::new();

        local.block_on(&rt, async move {
            let service = Rc::new(self.service);
            let server_pubkey = service.signing_key().verifying_key();
            let signing_key = service.signing_key();
            let nonce_cache = Arc::new(crate::envelope::InMemoryNonceCache::new());

            // Signal ready
            if let Some(tx) = on_ready {
                let _ = tx.send(());
            }

            // Run accept loop
            if let Err(e) = self.rep.accept_loop_service(service, server_pubkey, signing_key, nonce_cache, shutdown).await {
                tracing::error!("QUIC accept loop error: {}", e);
            }
        });

        Ok(())
    }
}


/// Compute SHA-256 hash of certificate for browser `serverCertificateHashes` (base64).
pub fn cert_hash(cert_der: &[u8]) -> String {
    use base64::Engine;
    use sha2::{Sha256, Digest};
    let hash = Sha256::digest(cert_der);
    base64::engine::general_purpose::STANDARD.encode(hash)
}

// ============================================================================
// TLS Helpers
// ============================================================================

/// Generate a self-signed TLS certificate for development/testing.
///
/// Returns (ServerConfig, cert_der_bytes).
pub fn server_tls_self_signed(name: &str) -> Result<(rustls::ServerConfig, Vec<u8>)> {
    let cert_key = rcgen::generate_simple_self_signed(vec![name.to_owned()])?;

    let cert_der = cert_key.cert.der().to_vec();
    let key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(cert_key.key_pair.serialize_der());

    let mut cfg = rustls::ServerConfig::builder_with_provider(
        crate::transport::pq_provider::internal_mesh_crypto_provider(),
    )
        .with_safe_default_protocol_versions()?
        .with_no_client_auth()
        .with_single_cert(vec![cert_der.clone().into()], key_der.into())?;

    cfg.alpn_protocols = vec![ALPN_ZMTP3.to_vec()];

    Ok((cfg, cert_der))
}

/// Create a TLS client config that pins a specific server certificate.
///
/// Use this for self-signed certificates where you don't want to trust
/// the system CA store.
pub fn client_tls_pinned(cert_der: &[u8]) -> Result<rustls::ClientConfig> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.add(rustls::pki_types::CertificateDer::from_slice(cert_der))?;

    let cfg = rustls::ClientConfig::builder_with_provider(
        crate::transport::pq_provider::internal_mesh_crypto_provider(),
    )
        .with_safe_default_protocol_versions()?
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let mut cfg = cfg;
    cfg.alpn_protocols = vec![ALPN_ZMTP3.to_vec()];

    Ok(cfg)
}

/// Create a TLS client config that trusts system root certificates.
pub fn client_tls_system_roots() -> Result<rustls::ClientConfig> {
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());

    let cfg = rustls::ClientConfig::builder_with_provider(
        crate::transport::pq_provider::internal_mesh_crypto_provider(),
    )
        .with_safe_default_protocol_versions()?
        .with_root_certificates(root_store)
        .with_no_client_auth();

    let mut cfg = cfg;
    cfg.alpn_protocols = vec![ALPN_ZMTP3.to_vec()];

    Ok(cfg)
}

// ============================================================================
// Request Processing Helper
// ============================================================================

/// Envelope verification mode for `process_request`.
///
/// Two modes exist for different transport security models:
///
/// - **FixedSigner**: ZMQ (internal service-to-service) — envelope signer must match
///   known `server_pubkey` for mutual authentication. Peers pre-share keys.
/// - **AnySigner**: WebTransport (external browser clients) — any valid Ed25519 signer
///   accepted. TLS 1.3 provides transport-layer authentication.
///
/// Both modes share: timestamp window (5 min), nonce replay protection, JWT claims
/// verification. The only difference is step 1 of 4 in the verification pipeline.
pub use crate::envelope::EnvelopeVerification;

// process_request moved to crate::service::dispatch (#148) — transport-neutral
// dispatch core, shared by RequestLoop / WebTransport / LocalServiceBridge.

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::zmtp_framing::{build_greeting, validate_greeting, build_ready_metadata};
    use quinn::crypto::rustls::HandshakeData;

    #[test]
    fn test_greeting_format() {
        let greeting = build_greeting();

        // Check size
        assert_eq!(greeting.len(), GREETING_SIZE);

        // Check signature
        assert_eq!(greeting[0], 0xFF);
        assert!(greeting[1..9].iter().all(|&b| b == 0));
        assert_eq!(greeting[9], 0x7F);

        // Check version
        assert_eq!(greeting[10], ZMTP_VERSION_MAJOR);
        assert_eq!(greeting[11], ZMTP_VERSION_MINOR);

        // Check mechanism
        assert_eq!(&greeting[12..16], b"NULL");

        // Check as-server (must be 0x00 for NULL mechanism)
        assert_eq!(greeting[32], 0x00);
    }

    #[test]
    fn test_greeting_roundtrip() {
        let greeting = build_greeting();
        validate_greeting(&greeting).expect("greeting should validate");
    }

    #[test]
    fn test_greeting_rejects_wrong_version() {
        let mut greeting = build_greeting();
        greeting[10] = 99; // Wrong major version

        let result = validate_greeting(&greeting);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("version mismatch"));
    }

    #[test]
    fn test_greeting_rejects_wrong_mechanism() {
        let mut greeting = build_greeting();
        greeting[12..16].copy_from_slice(b"CURV");

        let result = validate_greeting(&greeting);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mechanism mismatch"));
    }

    #[test]
    fn test_ready_metadata_encoding() {
        let metadata = build_ready_metadata(ZmqSocketType::Rep);

        // Parse it back
        assert!(!metadata.is_empty());

        // First byte is name length
        let name_len = metadata[0] as usize;
        assert_eq!(name_len, 11); // "Socket-Type"

        // Name follows
        let name = std::str::from_utf8(&metadata[1..1 + name_len]).unwrap();
        assert_eq!(name, "Socket-Type");

        // Then 4-byte value length
        let value_len_start = 1 + name_len;
        let value_len = u32::from_be_bytes([
            metadata[value_len_start],
            metadata[value_len_start + 1],
            metadata[value_len_start + 2],
            metadata[value_len_start + 3],
        ]) as usize;
        assert_eq!(value_len, 3); // "REP"

        // Value follows
        let value = std::str::from_utf8(&metadata[value_len_start + 4..value_len_start + 4 + value_len]).unwrap();
        assert_eq!(value, "REP");
    }

    #[test]
    fn test_tls_self_signed_generates_valid_cert() {
        let (config, cert_der) = server_tls_self_signed("test.local").unwrap();

        // Check ALPN is set
        assert!(config.alpn_protocols.iter().any(|p| p == ALPN_ZMTP3));

        // Check cert is non-empty
        assert!(!cert_der.is_empty());
    }

    #[test]
    fn test_client_tls_pinned_accepts_matching_cert() {
        let (_server_config, cert_der) = server_tls_self_signed("test.local").unwrap();

        // Should succeed
        let client_config = client_tls_pinned(&cert_der);
        assert!(client_config.is_ok());

        // Check ALPN is set
        let config = client_config.unwrap();
        assert!(config.alpn_protocols.iter().any(|p| p == ALPN_ZMTP3));
    }

    #[tokio::test]
    async fn zmtp_live_handshake_negotiates_x25519mlkem768() {
        let (server_tls, cert_der) = server_tls_self_signed("localhost").unwrap();
        let server = QuicRep::bind("127.0.0.1:0".parse().unwrap(), server_tls).unwrap();
        let addr = server.local_addr().unwrap();
        let endpoint = server.endpoint.clone();

        let accepted = tokio::spawn(async move {
            let connection = endpoint
                .accept()
                .await
                .expect("incoming zmtp connection")
                .await
                .expect("complete zmtp handshake");
            connection
                .handshake_data()
                .expect("completed quinn connection has handshake data")
                .downcast::<HandshakeData>()
                .expect("quinn rustls handshake data")
                .negotiated_key_exchange_group
        });

        let client = QuicReq::connect(addr, "localhost", client_tls_pinned(&cert_der).unwrap())
            .await
            .unwrap();
        let client_group = client
            .conn
            .handshake_data()
            .expect("completed quinn connection has handshake data")
            .downcast::<HandshakeData>()
            .expect("quinn rustls handshake data")
            .negotiated_key_exchange_group;

        assert_eq!(client_group, rustls::NamedGroup::X25519MLKEM768);
        assert_eq!(
            accepted.await.unwrap(),
            rustls::NamedGroup::X25519MLKEM768
        );
    }

    #[tokio::test]
    async fn zmtp_internal_mesh_rejects_classical_only_peer() {
        let err = QuicReq::connect(
            "127.0.0.1:9".parse().unwrap(),
            "localhost",
            classical_client_config(),
        )
        .await
        .err()
        .expect("public REQ seam must reject classical config before dialing");
        assert!(err.to_string().contains("owned-mesh crypto policy mismatch"));
    }

    fn classical_client_config() -> rustls::ClientConfig {
        let mut config = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(rustls::RootCertStore::empty())
        .with_no_client_auth();
        config.alpn_protocols = vec![ALPN_ZMTP3.to_vec()];
        config
    }

    fn classical_server_config() -> rustls::ServerConfig {
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(
            cert.key_pair.serialize_der(),
        );
        let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(vec![cert.cert.der().clone()], key.into())
        .unwrap();
        config.alpn_protocols = vec![ALPN_ZMTP3.to_vec()];
        config
    }

    fn assert_owned_policy_error<T>(result: Result<T>, seam: &str) {
        let err = result.err().unwrap_or_else(|| panic!("{seam} accepted classical config"));
        assert!(
            err.to_string().contains("owned-mesh crypto policy mismatch"),
            "{seam} returned the wrong error: {err}"
        );
    }

    #[test]
    fn zmtp_rep_rejects_classical_only_config_before_bind() {
        let err = QuicRep::bind("127.0.0.1:0".parse().unwrap(), classical_server_config())
            .err()
            .expect("public REP seam must reject classical config before bind");
        assert!(err.to_string().contains("owned-mesh crypto policy mismatch"));
    }

    #[tokio::test]
    async fn all_owned_zmtp_socket_families_reject_classical_configs() {
        let bind_addr = "127.0.0.1:0".parse().unwrap();
        assert_owned_policy_error(
            QuicXPub::bind(bind_addr, classical_server_config()),
            "XPUB bind",
        );
        assert_owned_policy_error(
            QuicPull::bind(bind_addr, classical_server_config()),
            "PULL bind",
        );

        let dial_addr = "127.0.0.1:9".parse().unwrap();
        assert_owned_policy_error(
            QuicSub::connect(dial_addr, "localhost", classical_client_config()).await,
            "SUB connect",
        );
        assert_owned_policy_error(
            QuicPush::connect(dial_addr, "localhost", classical_client_config()).await,
            "PUSH connect",
        );
    }

    #[test]
    fn test_command_parse() {
        let data = b"\x05READY\x00\x00\x00\x03foo";
        let cmd = ZmtpCommand::parse(data).unwrap();

        assert_eq!(cmd.name, "READY");
        assert_eq!(cmd.body.as_ref(), b"\x00\x00\x00\x03foo");
    }

    #[test]
    fn test_command_parse_empty_name() {
        let data = b"";
        let result = ZmtpCommand::parse(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_multipart() {
        let mut msg = Multipart::new();
        assert!(msg.is_empty());
        assert_eq!(msg.len(), 0);

        msg.push(Bytes::from("part1"));
        msg.push(Bytes::from("part2"));

        assert!(!msg.is_empty());
        assert_eq!(msg.len(), 2);
    }

    // ========================================================================
    // Frame encoding/decoding tests
    // ========================================================================

    /// Encode a ZMTP frame to bytes (delegates to zmtp_framing).
    fn encode_frame_for_test(more: bool, command: bool, data: &[u8]) -> Vec<u8> {
        zmtp_framing::encode_frame(more, command, data)
    }

    /// Decode a ZMTP frame from bytes (delegates to zmtp_framing).
    fn decode_frame_for_test(buf: &[u8]) -> Result<(ZmtpFrame, usize)> {
        let (owned, consumed) = zmtp_framing::decode_frame(buf)?;
        Ok((ZmtpFrame::from_owned(owned), consumed))
    }

    #[test]
    fn test_frame_short_encode_decode() {
        // Frame <=255 bytes: short format
        let data = vec![0x42u8; 100];
        let encoded = encode_frame_for_test(false, false, &data);
        let (decoded, consumed) = decode_frame_for_test(&encoded).unwrap();

        assert_eq!(consumed, encoded.len());
        assert!(!decoded.more);
        assert!(!decoded.command);
        assert_eq!(decoded.data.as_ref(), data.as_slice());
    }

    #[test]
    fn test_frame_long_encode_decode() {
        // Frame >255 bytes: long format
        let data = vec![0xABu8; 500];
        let encoded = encode_frame_for_test(false, false, &data);
        let (decoded, consumed) = decode_frame_for_test(&encoded).unwrap();

        assert_eq!(consumed, encoded.len());
        assert!(!decoded.more);
        assert!(!decoded.command);
        assert_eq!(decoded.data.as_ref(), data.as_slice());

        // Long frame should have 8-byte size
        assert_eq!(encoded[0] & 0x02, 0x02); // LONG bit set
    }

    #[test]
    fn test_frame_more_flag() {
        let data = b"test";
        let encoded_more = encode_frame_for_test(true, false, data);
        let encoded_last = encode_frame_for_test(false, false, data);

        let (decoded_more, _) = decode_frame_for_test(&encoded_more).unwrap();
        let (decoded_last, _) = decode_frame_for_test(&encoded_last).unwrap();

        assert!(decoded_more.more);
        assert!(!decoded_last.more);
    }

    #[test]
    fn test_command_frame_flags() {
        let data = b"READY";
        let encoded = encode_frame_for_test(false, true, data);
        let (decoded, _) = decode_frame_for_test(&encoded).unwrap();

        assert!(decoded.command);
        assert!(!decoded.more); // Commands are always single-frame
    }

    #[test]
    fn test_command_rejects_more_flag() {
        // Per RFC 37, command frames must have MORE=0
        let data = b"READY";
        let encoded = encode_frame_for_test(true, true, data); // Invalid: MORE=1 for command
        let (decoded, _) = decode_frame_for_test(&encoded).unwrap();

        // The decode should succeed, but this would be rejected by higher-level validation
        assert!(decoded.command);
        assert!(decoded.more); // Invalid state
    }

    #[test]
    fn test_ready_metadata_roundtrip() {
        let metadata = zmtp_framing::build_ready_metadata(ZmqSocketType::Rep);

        // Parse the metadata
        let name_len = metadata[0] as usize;
        let name = std::str::from_utf8(&metadata[1..1 + name_len]).unwrap();

        let value_len_start = 1 + name_len;
        let value_len = u32::from_be_bytes([
            metadata[value_len_start],
            metadata[value_len_start + 1],
            metadata[value_len_start + 2],
            metadata[value_len_start + 3],
        ]) as usize;
        let value = std::str::from_utf8(&metadata[value_len_start + 4..value_len_start + 4 + value_len]).unwrap();

        // Build again for REQ
        let metadata_req = zmtp_framing::build_ready_metadata(ZmqSocketType::Req);
        let name_len_req = metadata_req[0] as usize;
        let value_len_start_req = 1 + name_len_req;
        let value_len_req = u32::from_be_bytes([
            metadata_req[value_len_start_req],
            metadata_req[value_len_start_req + 1],
            metadata_req[value_len_start_req + 2],
            metadata_req[value_len_start_req + 3],
        ]) as usize;
        let value_req = std::str::from_utf8(&metadata_req[value_len_start_req + 4..value_len_start_req + 4 + value_len_req]).unwrap();

        assert_eq!(name, "Socket-Type");
        assert_eq!(value, "REP");
        assert_eq!(value_req, "REQ");
    }

    #[test]
    fn test_subscribe_command() {
        // SUBSCRIBE command: body is topic prefix
        let topic = b"my.topic";
        let cmd = ZmtpCommand {
            name: "SUBSCRIBE".to_owned(),
            body: Bytes::copy_from_slice(topic),
        };

        assert_eq!(cmd.name, "SUBSCRIBE");
        assert_eq!(cmd.body.as_ref(), topic);
    }

    #[test]
    fn test_cancel_command() {
        // CANCEL command: body is topic prefix
        let topic = b"my.topic";
        let cmd = ZmtpCommand {
            name: "CANCEL".to_owned(),
            body: Bytes::copy_from_slice(topic),
        };

        assert_eq!(cmd.name, "CANCEL");
        assert_eq!(cmd.body.as_ref(), topic);
    }

    #[test]
    fn test_multipart_roundtrip() {
        // Test multipart message encoding
        let parts: Vec<Bytes> = vec![
            Bytes::from("identity"),
            Bytes::from("delimiter"),
            Bytes::from("payload"),
        ];

        // Encode all parts
        let mut encoded = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            let more = i < parts.len() - 1;
            encoded.extend(encode_frame_for_test(more, false, part));
        }

        // Decode and verify
        let mut offset = 0;
        let mut decoded_parts = Vec::new();
        for _expected in &parts {
            let (frame, consumed) = decode_frame_for_test(&encoded[offset..]).unwrap();
            decoded_parts.push(frame.data);
            offset += consumed;
        }

        assert_eq!(decoded_parts, parts);
    }

    // ========================================================================
    // ZmtpStream tests (using in-memory duplex streams)
    // ========================================================================

    #[test]
    fn test_zmtpstream_from_split() {
        // Create in-memory duplex streams for testing
        let (client_stream, server_stream) = tokio::io::duplex(8192);

        // This test verifies the constructor exists and works
        // Full handshake tests would require async runtime
        let _client_stream = client_stream;
        let _server_stream = server_stream;
    }
}
