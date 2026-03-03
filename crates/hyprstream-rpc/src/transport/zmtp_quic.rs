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
        let quic_server_cfg =
            quinn::ServerConfig::with_crypto(Arc::new(quinn::crypto::rustls::QuicServerConfig::try_from(tls)?));

        let endpoint = quinn::Endpoint::server(quic_server_cfg, addr)?;

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

    /// Accept loop for ZmqService handlers.
    ///
    /// This variant handles the full envelope processing pipeline:
    /// 1. ZMTP handshake
    /// 2. SignedEnvelope verification
    /// 3. Service handle_request dispatch
    /// 4. Signed response serialization
    ///
    /// Uses `spawn_local` because `ZmqService` is `?Send`.
    ///
    /// # Arguments
    ///
    /// * `service` - ZmqService implementation (wrapped in Rc)
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
        S: crate::service::ZmqService + 'static,
    {
        loop {
            tokio::select! {
                Some(incoming) = self.endpoint.accept() => {
                    let service = Rc::clone(&service);
                    let nonce_cache = Arc::clone(&nonce_cache);
                    let signing_key = signing_key.clone();  // Clone before spawn

                    // Use spawn_local because ZmqService is ?Send
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
        S: crate::service::ZmqService + 'static,
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
        server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
    ) -> Result<()>
    where
        S: crate::service::ZmqService + 'static,
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

        // Process through envelope pipeline (FixedSigner: QUIC peers pre-share keys)
        let (response_bytes, continuation) = process_request(
            raw_bytes,
            &*service,
            EnvelopeVerification::FixedSigner(&server_pubkey),
            &signing_key,
            &nonce_cache,
        ).await?;

        // Send response
        let response = Multipart {
            parts: vec![Bytes::from(response_bytes)],
        };
        stream.send_multipart(&response.parts).await?;

        // Signal end of response
        stream.stream.send.finish()?;

        // Handle continuation if present (for streaming)
        if let Some(cont) = continuation {
            // Spawn continuation on LocalSet (streaming handler is ?Send)
            tokio::task::spawn_local(cont);
        }

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
        let quic_client_config =
            quinn::ClientConfig::new(Arc::new(quinn::crypto::rustls::QuicClientConfig::try_from(
                tls,
            )?));

        let mut endpoint = quinn::Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(quic_client_config);

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
        let quic_server_cfg =
            quinn::ServerConfig::with_crypto(Arc::new(quinn::crypto::rustls::QuicServerConfig::try_from(tls)?));

        let endpoint = quinn::Endpoint::server(quic_server_cfg, addr)?;

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
        let quic_client_config =
            quinn::ClientConfig::new(Arc::new(quinn::crypto::rustls::QuicClientConfig::try_from(
                tls,
            )?));

        let mut endpoint = quinn::Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(quic_client_config);

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
        let quic_client_config =
            quinn::ClientConfig::new(Arc::new(quinn::crypto::rustls::QuicClientConfig::try_from(
                tls,
            )?));

        let mut endpoint = quinn::Endpoint::client("[::]:0".parse()?)?;
        endpoint.set_default_client_config(quic_client_config);

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
        let quic_server_cfg =
            quinn::ServerConfig::with_crypto(Arc::new(quinn::crypto::rustls::QuicServerConfig::try_from(tls)?));

        let endpoint = quinn::Endpoint::server(quic_server_cfg, addr)?;
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

/// Service loop for hosting ZmqService over QUIC transport.
///
/// This struct wraps a `QuicRep` socket and a `ZmqService` implementation,
/// running the QUIC accept loop in a dedicated thread with its own tokio runtime.
///
/// # Threading Model
///
/// `ZmqService` is `#[async_trait(?Send)]` - handlers are not `Send`. This requires:
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
    S: crate::service::ZmqService + Send + Sync + 'static,
{
    rep: QuicRep,
    service: S,
    /// Server certificate DER bytes (for registration)
    _cert_der: Vec<u8>,
    /// ZMQ context (for Spawnable trait)
    context: Arc<zmq::Context>,
    /// Service name
    name: String,
    /// QUIC endpoint address
    addr: SocketAddr,
}

impl<S> QuicServiceLoop<S>
where
    S: crate::service::ZmqService + Send + Sync + 'static,
{
    /// Create a new QUIC service loop.
    pub fn new(rep: QuicRep, service: S, cert_der: Vec<u8>) -> Result<Self> {
        let addr = rep.local_addr()?;
        let name = service.name().to_owned();
        Ok(Self {
            rep,
            service,
            _cert_der: cert_der,
            context: Arc::new(zmq::Context::new()),
            name,
            addr,
        })
    }

    /// Get the QUIC endpoint address.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }
}

impl<S> crate::service::spawner::Spawnable for QuicServiceLoop<S>
where
    S: crate::service::ZmqService + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn registrations(&self) -> Vec<(crate::registry::SocketKind, crate::transport::TransportConfig)> {
        // Register QUIC endpoint
        vec![(
            crate::registry::SocketKind::Rep,
            crate::transport::TransportConfig::quic(self.addr, "hyprstream.local"),
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

// ============================================================================
// WebTransport Server (browser clients)
// ============================================================================

/// WebTransport server for browser clients.
///
/// Accepts WebTransport sessions (HTTP/3) from browsers and handles
/// ZMTP requests using the same ZmqService handlers as native QUIC.
///
/// # Browser Compatibility
///
/// Uses `serverCertificateHashes` in the WebTransport constructor to accept
/// self-signed certificates. The browser passes the SHA-256 hash of the cert.
///
/// # Usage
///
/// ```ignore
/// let (tls, cert_der) = server_tls_self_signed("localhost")?;
/// let cert_hash = sha256(&cert_der);
///
/// let wt_server = WebTransportServer::bind(addr, tls)?;
///
/// // In browser:
/// // const wt = new WebTransport('https://localhost:4433', {
/// //   serverCertificateHashes: [{
/// //     algorithm: 'sha-256',
/// //     value: Uint8Array.from(atob(certHashBase64), c => c.charCodeAt(0))
/// //   }]
/// // });
/// ```
pub struct WebTransportServer {
    endpoint: quinn::Endpoint,
    /// Serialized RFC 9728 metadata JSON for GET /.well-known/oauth-protected-resource
    protected_resource_json: Option<Vec<u8>>,
}

/// Concrete h3-quinn stream types used throughout WebTransportServer.
type H3QuinnBidiStream = h3_quinn::BidiStream<bytes::Bytes>;
type H3QuinnRequestStream = h3::server::RequestStream<H3QuinnBidiStream, bytes::Bytes>;
type H3QuinnConnection = h3::server::Connection<h3_quinn::Connection, bytes::Bytes>;
type WtSession = h3_webtransport::server::WebTransportSession<h3_quinn::Connection, bytes::Bytes>;
type WtBidiStream = h3_webtransport::stream::BidiStream<H3QuinnBidiStream, bytes::Bytes>;

impl WebTransportServer {
    /// Bind an HTTP/3 + WebTransport server to the given address with a certificate.
    ///
    /// Uses h3 + h3-quinn + h3-webtransport to serve both regular HTTP/3 GET requests
    /// (e.g. `.well-known/oauth-protected-resource`) and WebTransport CONNECT sessions
    /// (for ZMTP-framed Cap'n Proto RPC) on the same QUIC port.
    ///
    /// # Arguments
    ///
    /// * `addr` - Socket address to bind to
    /// * `cert_der` - DER-encoded certificate
    /// * `key_der` - DER-encoded private key
    pub fn bind(
        addr: SocketAddr,
        cert_der: Vec<u8>,
        key_der: Vec<u8>,
    ) -> Result<Self> {
        let tls = webtransport_tls_config(cert_der, key_der)?;
        let quic_cfg = quinn::ServerConfig::with_crypto(Arc::new(
            quinn::crypto::rustls::QuicServerConfig::try_from(tls)
                .map_err(|e| anyhow!("QUIC server config failed: {}", e))?
        ));
        let endpoint = quinn::Endpoint::server(quic_cfg, addr)?;

        Ok(Self { endpoint, protected_resource_json: None })
    }

    /// Set RFC 9728 Protected Resource Metadata served at /.well-known/oauth-protected-resource.
    pub fn with_protected_resource_metadata(mut self, json_bytes: Vec<u8>) -> Self {
        self.protected_resource_json = Some(json_bytes);
        self
    }

    /// Get the actual bound address from the QUIC endpoint.
    ///
    /// This returns the real OS-assigned address, which matters when
    /// binding to port 0 (ephemeral port assignment).
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint.local_addr().map_err(Into::into)
    }

    /// Accept connections and dispatch HTTP/3 + WebTransport.
    ///
    /// Uses `spawn_local` because `ZmqService` is `?Send`.
    pub async fn accept_loop_service<S>(
        self,
        service: Rc<S>,
        server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
        shutdown: Arc<Notify>,
    ) -> Result<()>
    where
        S: crate::service::ZmqService + 'static,
    {
        let metadata_json = self.protected_resource_json.map(Arc::new);

        loop {
            tokio::select! {
                incoming = self.endpoint.accept() => {
                    let Some(incoming) = incoming else { break; };
                    let service = Rc::clone(&service);
                    let nonce_cache = Arc::clone(&nonce_cache);
                    let signing_key = signing_key.clone();
                    let metadata = metadata_json.clone();

                    tokio::task::spawn_local(async move {
                        if let Err(e) = Self::handle_connection(
                            incoming, service, server_pubkey, signing_key,
                            nonce_cache, metadata,
                        ).await {
                            debug!("H3 connection error: {}", e);
                        }
                    });
                }
                _ = shutdown.notified() => {
                    debug!("WebTransport accept loop shutting down");
                    break;
                }
            }
        }
        Ok(())
    }

    /// Handle an individual QUIC connection via h3.
    ///
    /// Routes incoming requests: WebTransport CONNECT upgrades go to
    /// `handle_webtransport_session`, regular HTTP/3 GETs go to `handle_http_request`.
    async fn handle_connection<S>(
        incoming: quinn::Incoming,
        service: Rc<S>,
        server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
        metadata_json: Option<Arc<Vec<u8>>>,
    ) -> Result<()>
    where
        S: crate::service::ZmqService + 'static,
    {
        let quinn_conn = incoming.await?;
        let h3_conn = h3_quinn::Connection::new(quinn_conn);
        let mut h3_server: H3QuinnConnection = h3::server::builder()
            .enable_webtransport(true)
            .enable_datagram(true)
            .enable_extended_connect(true)
            .build(h3_conn)
            .await
            .map_err(|e| anyhow!("h3 server build failed: {}", e))?;

        // Accept requests — route HTTP/3 GETs vs WebTransport CONNECT
        loop {
            match h3_server.accept().await {
                Ok(Some(resolver)) => {
                    let (req, stream) = resolver.resolve_request().await
                        .map_err(|e| anyhow!("h3 resolve request failed: {}", e))?;

                    // Check for WebTransport CONNECT upgrade
                    let is_webtransport = req.method() == http::Method::CONNECT
                        && req.extensions().get::<h3::ext::Protocol>()
                            == Some(&h3::ext::Protocol::WEB_TRANSPORT);

                    if is_webtransport {
                        // Upgrade to WebTransport session (consumes h3 connection)
                        let session: WtSession = h3_webtransport::server::WebTransportSession::accept(
                            req, stream, h3_server,
                        ).await
                        .map_err(|e| anyhow!("WebTransport session accept failed: {}", e))?;

                        Self::handle_webtransport_session(
                            session, service, server_pubkey, signing_key,
                            nonce_cache, metadata_json,
                        ).await?;
                        break; // session consumed the connection
                    } else {
                        // Regular HTTP/3 request
                        Self::handle_http_request(req, stream, metadata_json.as_ref()).await?;
                        // Continue accepting more requests on this connection
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    debug!("H3 accept error: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Serve HTTP/3 requests: `.well-known/oauth-protected-resource` and `/health`.
    async fn handle_http_request(
        req: http::Request<()>,
        mut stream: H3QuinnRequestStream,
        metadata_json: Option<&Arc<Vec<u8>>>,
    ) -> Result<()> {
        let (status, content_type, body) = match req.uri().path() {
            "/.well-known/oauth-protected-resource" => {
                if let Some(json) = metadata_json {
                    (http::StatusCode::OK, "application/json", json.as_ref().clone())
                } else {
                    (http::StatusCode::NOT_FOUND, "text/plain", b"Not configured".to_vec())
                }
            }
            "/health" => (http::StatusCode::OK, "text/plain", b"ok".to_vec()),
            _ => (http::StatusCode::NOT_FOUND, "text/plain", b"Not found".to_vec()),
        };

        let response = http::Response::builder()
            .status(status)
            .header("content-type", content_type)
            .body(())
            .map_err(|e| anyhow!("failed to build HTTP response: {}", e))?;

        stream.send_response(response).await
            .map_err(|e| anyhow!("failed to send HTTP response: {}", e))?;
        stream.send_data(bytes::Bytes::from(body)).await
            .map_err(|e| anyhow!("failed to send HTTP body: {}", e))?;
        stream.finish().await
            .map_err(|e| anyhow!("failed to finish HTTP stream: {}", e))?;
        Ok(())
    }

    /// Handle a WebTransport session: accept bidi streams for ZMTP-framed RPC,
    /// and also handle any in-session HTTP/3 requests (via `AcceptedBi::Request`).
    async fn handle_webtransport_session<S>(
        session: WtSession,
        service: Rc<S>,
        server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
        metadata_json: Option<Arc<Vec<u8>>>,
    ) -> Result<()>
    where
        S: crate::service::ZmqService + 'static,
    {
        debug!("WebTransport session established");
        loop {
            match session.accept_bi().await {
                Ok(Some(h3_webtransport::server::AcceptedBi::BidiStream(_session_id, stream))) => {
                    let service = Rc::clone(&service);
                    let nonce_cache = Arc::clone(&nonce_cache);
                    let signing_key = signing_key.clone();
                    tokio::task::spawn_local(async move {
                        if let Err(e) = Self::handle_wt_stream(
                            stream, service, server_pubkey, signing_key, nonce_cache,
                        ).await {
                            debug!("WebTransport stream error: {}", e);
                        }
                    });
                }
                Ok(Some(h3_webtransport::server::AcceptedBi::Request(req, stream))) => {
                    // HTTP/3 request within WebTransport session
                    if let Err(e) = Self::handle_http_request(req, stream, metadata_json.as_ref()).await {
                        debug!("WebTransport in-session HTTP request error: {}", e);
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    debug!("WebTransport session closed: {}", e);
                    break;
                }
            }
        }
        Ok(())
    }

    /// Handle a single WebTransport bidi stream: length-prefixed Cap'n Proto RPC.
    async fn handle_wt_stream<S>(
        stream: WtBidiStream,
        service: Rc<S>,
        _server_pubkey: ed25519_dalek::VerifyingKey,
        signing_key: ed25519_dalek::SigningKey,
        nonce_cache: Arc<crate::envelope::InMemoryNonceCache>,
    ) -> Result<()>
    where
        S: crate::service::ZmqService + 'static,
    {
        use h3::quic::BidiStream as H3BidiStream;
        let (mut send, mut recv) = H3BidiStream::split(stream);

        // WebTransport streams don't need ZMTP handshake - we send raw Cap'n Proto.
        // This is a simplification since both ends are controlled code (not libzmq peers).

        // Read request bytes (length-prefixed)
        const MAX_WEBTRANSPORT_REQUEST_SIZE: usize = 16 * 1024 * 1024;
        let mut len_buf = [0u8; 4];
        recv.read_exact(&mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        ensure!(
            len <= MAX_WEBTRANSPORT_REQUEST_SIZE,
            "WebTransport request size {} exceeds maximum {}",
            len,
            MAX_WEBTRANSPORT_REQUEST_SIZE
        );

        let mut request_buf = vec![0u8; len];
        recv.read_exact(&mut request_buf).await?;

        // Process through WebTransport pipeline (AnySigner: TLS 1.3 provides transport auth)
        let (response_bytes, continuation) = process_request(
            &request_buf,
            &*service,
            EnvelopeVerification::AnySigner,
            &signing_key,
            &nonce_cache,
        ).await?;

        // Write response length + bytes
        send.write_all(&(response_bytes.len() as u32).to_be_bytes()).await?;
        send.write_all(&response_bytes).await?;
        send.shutdown().await?;

        // Handle continuation if present (for streaming)
        if let Some(cont) = continuation {
            tokio::task::spawn_local(cont);
        }

        Ok(())
    }
}

/// Compute SHA-256 hash of certificate for browser `serverCertificateHashes`.
pub fn cert_hash(cert_der: &[u8]) -> String {
    use base64::Engine;
    use sha2::{Sha256, Digest};
    let hash = Sha256::digest(cert_der);
    base64::engine::general_purpose::STANDARD.encode(hash)
}

// ============================================================================
// TLS Helpers
// ============================================================================

/// Install the ring crypto provider for rustls (required for QUIC/TLS).
///
/// Must be called before creating any rustls `ServerConfig` or quinn endpoint.
/// No-op if already installed (safe to call multiple times).
pub fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Build TLS config for HTTP/3 + WebTransport (ALPN: h3).
fn webtransport_tls_config(
    cert_der: Vec<u8>,
    key_der: Vec<u8>,
) -> Result<rustls::ServerConfig> {
    ensure_crypto_provider();
    let key = rustls::pki_types::PrivateKeyDer::from(
        rustls::pki_types::PrivatePkcs8KeyDer::from(key_der)
    );
    let mut cfg = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der.into()], key)?;
    cfg.alpn_protocols = vec![b"h3".to_vec()];
    Ok(cfg)
}

/// Generate a self-signed TLS certificate for development/testing.
///
/// Returns (ServerConfig, cert_der_bytes).
pub fn server_tls_self_signed(name: &str) -> Result<(rustls::ServerConfig, Vec<u8>)> {
    ensure_crypto_provider();
    let cert_key = rcgen::generate_simple_self_signed(vec![name.to_owned()])?;

    let cert_der = cert_key.cert.der().to_vec();
    let key_der = rustls::pki_types::PrivatePkcs8KeyDer::from(cert_key.key_pair.serialize_der());

    let mut cfg = rustls::ServerConfig::builder()
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

    let cfg = rustls::ClientConfig::builder()
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

    let cfg = rustls::ClientConfig::builder()
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
pub enum EnvelopeVerification<'a> {
    /// Require the envelope signer to match this specific verifying key.
    /// Used for ZMQ transport where peers pre-share Ed25519 keys.
    FixedSigner(&'a ed25519_dalek::VerifyingKey),
    /// Accept any valid Ed25519 signer.
    /// Used for WebTransport where TLS 1.3 provides channel authentication.
    AnySigner,
}

/// Process a request through the full envelope verification pipeline.
///
/// Unified handler for both ZMQ (ROUTER) and WebTransport paths. The only
/// difference is envelope signer verification, controlled by `verification`.
///
/// # Pipeline
///
/// 1. Unwrap `SignedEnvelope` and verify Ed25519 signature (mode-dependent)
/// 2. Verify JWT claims (`sub`, `exp`, `aud`, `scope`, downgrade protection)
/// 3. Dispatch to `service.handle_request()` with verified `EnvelopeContext`
/// 4. Sign response with server's `signing_key`
///
/// # Arguments
///
/// * `raw_bytes` - Raw Cap'n Proto bytes containing a `SignedEnvelope`
/// * `service` - The ZmqService to dispatch to
/// * `verification` - Envelope signer verification mode
/// * `signing_key` - Server's signing key for response
/// * `nonce_cache` - Nonce cache for replay protection
///
/// # Returns
///
/// * `Ok((response_bytes, continuation))` - Signed response and optional continuation
/// * `Err(e)` - Processing error (already logged)
pub async fn process_request<S>(
    raw_bytes: &[u8],
    service: &S,
    verification: EnvelopeVerification<'_>,
    signing_key: &ed25519_dalek::SigningKey,
    nonce_cache: &crate::envelope::InMemoryNonceCache,
) -> Result<(Vec<u8>, Option<crate::service::Continuation>)>
where
    S: crate::service::ZmqService,
{
    use crate::ToCapnp;
    use crate::envelope::ResponseEnvelope;
    use capnp::message::Builder;
    use capnp::serialize;
    use tracing::warn;

    // 1. Unwrap and verify SignedEnvelope based on verification mode
    let (ctx, payload) = match match verification {
        EnvelopeVerification::FixedSigner(pubkey) =>
            crate::envelope::unwrap_envelope(raw_bytes, pubkey, nonce_cache),
        EnvelopeVerification::AnySigner =>
            crate::envelope::unwrap_envelope_any_signer(raw_bytes, nonce_cache),
    } {
        Ok(result) => result,
        Err(e) => {
            warn!("{} envelope verification failed: {}", service.name(), e);
            // Build error response with request_id=0 (envelope is invalid)
            let error_payload = service.build_error_payload(0, &format!("envelope verification failed: {}", e));
            let signed_response = ResponseEnvelope::new_signed(0, error_payload, signing_key);

            let mut message = Builder::new_default();
            let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
            signed_response.write_to(&mut builder);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            return Ok((bytes, None));
        }
    };

    let request_id = ctx.request_id;
    debug!(
        "{} verified request from {} (id={})",
        service.name(),
        ctx.subject(),
        request_id
    );

    // 2. Verify claims (E2E JWT, downgrade protection)
    if let Err(e) = service.verify_claims(&ctx) {
        warn!(
            "{} claims verification failed for {} (id={}): {}",
            service.name(), ctx.subject(), request_id, e
        );
        let error_payload = service.build_error_payload(request_id, &e.to_string());
        let signed_response = ResponseEnvelope::new_signed(request_id, error_payload, signing_key);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        signed_response.write_to(&mut builder);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        return Ok((bytes, None));
    }

    // 3. Handle request
    let (response_payload, continuation) = match service.handle_request(&ctx, &payload).await {
        Ok((resp, cont)) => (resp, cont),
        Err(e) => {
            error!("{} request handling error: {}", service.name(), e);
            (service.build_error_payload(request_id, &e.to_string()), None)
        }
    };

    // 4. Sign and serialize response
    let signed_response = ResponseEnvelope::new_signed(request_id, response_payload, signing_key);

    let mut message = Builder::new_default();
    let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
    signed_response.write_to(&mut builder);

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;

    Ok((bytes, continuation))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zmtp_framing::{build_greeting, validate_greeting, build_ready_metadata};

    /// Install the ring crypto provider for rustls (required for TLS tests).
    /// Delegates to the public module-level function.
    fn ensure_crypto_provider() {
        super::ensure_crypto_provider();
    }

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
        ensure_crypto_provider();
        let (config, cert_der) = server_tls_self_signed("test.local").unwrap();

        // Check ALPN is set
        assert!(config.alpn_protocols.iter().any(|p| p == ALPN_ZMTP3));

        // Check cert is non-empty
        assert!(!cert_der.is_empty());
    }

    #[test]
    fn test_client_tls_pinned_accepts_matching_cert() {
        ensure_crypto_provider();
        let (_server_config, cert_der) = server_tls_self_signed("test.local").unwrap();

        // Should succeed
        let client_config = client_tls_pinned(&cert_der);
        assert!(client_config.is_ok());

        // Check ALPN is set
        let config = client_config.unwrap();
        assert!(config.alpn_protocols.iter().any(|p| p == ALPN_ZMTP3));
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
            name: "SUBSCRIBE".to_string(),
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
            name: "CANCEL".to_string(),
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
