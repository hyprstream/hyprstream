//! QuicStreamBridge — Bridges ZMQ SUB → WebTransport for streaming data.
//!
//! When a client requests a streaming operation (e.g., model inference) via
//! `RequestLoop`, the service returns a `StreamInfo` response containing
//! the topic and server's DH public key. The client then opens a second
//! WebTransport stream for subscription.
//!
//! This bridge:
//! 1. Subscribes to topics on the ZMQ XPUB (StreamService output)
//! 2. Forwards `StreamBlock`s to matching WebTransport sessions
//! 3. StreamBlocks carry chained HMAC-SHA256 for E2E verification
//!
//! # Architecture
//!
//! ```text
//! StreamService (XPUB) ───SUB───► QuicStreamBridge ───WebTransport───► Browser
//!                                   │
//!                                   └─ manages per-topic subscriptions
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use tokio::sync::{mpsc, Notify};
use tracing::{debug, error, warn};

use crate::transport::TransportConfig;

/// Manages streaming subscriptions from ZMQ SUB to WebTransport sessions.
///
/// Each active subscription maps a topic to a channel that feeds
/// WebTransport stream writers. When a client subscribes to a topic,
/// the bridge creates a ZMQ SUB socket subscribed to that topic and
/// forwards all received StreamBlocks to the WebTransport stream.
pub struct QuicStreamBridge {
    /// ZMQ context for creating SUB sockets
    zmq_context: Arc<zmq::Context>,
    /// Transport config for connecting to StreamService XPUB
    sub_transport: TransportConfig,
    /// Active topic subscriptions: topic -> sender for WebTransport forwarding
    subscriptions: HashMap<String, Vec<mpsc::UnboundedSender<Vec<u8>>>>,
    /// Shutdown signal
    shutdown: Arc<Notify>,
}

impl QuicStreamBridge {
    /// Create a new stream bridge.
    ///
    /// # Arguments
    ///
    /// * `zmq_context` - ZMQ context for socket creation
    /// * `sub_transport` - Transport config for connecting to StreamService's XPUB endpoint
    pub fn new(
        zmq_context: Arc<zmq::Context>,
        sub_transport: TransportConfig,
    ) -> Self {
        Self {
            zmq_context,
            sub_transport,
            subscriptions: HashMap::new(),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Subscribe to a topic, returning a receiver for StreamBlock bytes.
    ///
    /// The receiver yields raw Cap'n Proto serialized StreamBlock messages.
    /// The caller (WebTransport handler) forwards these to the browser client.
    ///
    /// Multiple subscribers to the same topic share the underlying ZMQ SUB socket.
    pub fn subscribe(&mut self, topic: &str) -> mpsc::UnboundedReceiver<Vec<u8>> {
        let (tx, rx) = mpsc::unbounded_channel();

        let senders = self.subscriptions.entry(topic.to_string()).or_default();

        // If this is the first subscriber for this topic, start the ZMQ SUB task
        if senders.is_empty() {
            let zmq_context = Arc::clone(&self.zmq_context);
            let sub_transport = self.sub_transport.clone();
            let topic_owned = topic.to_string();
            let shutdown = Arc::clone(&self.shutdown);
            let tx_clone = tx.clone();

            tokio::spawn(async move {
                if let Err(e) = Self::zmq_sub_loop(
                    zmq_context,
                    sub_transport,
                    &topic_owned,
                    tx_clone,
                    shutdown,
                ).await {
                    warn!("ZMQ SUB loop for topic '{}' ended: {}", topic_owned, e);
                }
            });
        }

        senders.push(tx);
        rx
    }

    /// Remove a subscription sender (cleanup on WebTransport disconnect).
    pub fn unsubscribe(&mut self, topic: &str) {
        if let Some(senders) = self.subscriptions.get_mut(topic) {
            // Remove closed senders
            senders.retain(|tx| !tx.is_closed());
            if senders.is_empty() {
                self.subscriptions.remove(topic);
            }
        }
    }

    /// Signal shutdown to all subscription loops.
    pub fn shutdown(&self) {
        self.shutdown.notify_waiters();
    }

    /// Internal: ZMQ SUB loop that receives from XPUB and forwards to channel.
    async fn zmq_sub_loop(
        zmq_context: Arc<zmq::Context>,
        sub_transport: TransportConfig,
        topic: &str,
        tx: mpsc::UnboundedSender<Vec<u8>>,
        shutdown: Arc<Notify>,
    ) -> Result<()> {
        use tmq::FromZmqSocket;
        use futures::StreamExt;

        // Create raw ZMQ SUB socket, connect, then wrap with tmq
        let mut socket = zmq_context.socket(zmq::SUB)
            .map_err(|e| anyhow!("failed to create SUB socket: {}", e))?;
        socket.set_linger(0).ok();

        // Connect to StreamService XPUB
        sub_transport.connect(&mut socket)
            .map_err(|e| anyhow!("failed to connect SUB to {}: {}", sub_transport.zmq_endpoint(), e))?;

        // Wrap raw socket → SubscribeWithoutTopic → Subscribe
        let without_topic = tmq::SubscribeWithoutTopic::from_zmq_socket(socket)
            .map_err(|e| anyhow!("failed to wrap SUB socket: {}", e))?;
        let mut subscriber = without_topic.subscribe(topic.as_bytes())
            .map_err(|e| anyhow!("failed to subscribe to '{}': {}", topic, e))?;

        debug!("QuicStreamBridge: subscribed to topic '{}'", topic);

        loop {
            tokio::select! {
                biased;

                _ = shutdown.notified() => {
                    debug!("QuicStreamBridge: shutdown for topic '{}'", topic);
                    break;
                }

                Some(result) = subscriber.next() => {
                    match result {
                        Ok(msg) => {
                            // ZMQ SUB messages: [topic_frame, ...data_frames]
                            // Skip the topic frame, concatenate the rest
                            let frames: Vec<zmq::Message> = msg.into_iter().collect();
                            if frames.len() < 2 {
                                continue;
                            }
                            let data: Vec<u8> = frames[1..]
                                .iter()
                                .flat_map(|f: &zmq::Message| f.iter().copied())
                                .collect();

                            if tx.send(data).is_err() {
                                // Receiver dropped — client disconnected
                                debug!("QuicStreamBridge: subscriber for '{}' disconnected", topic);
                                break;
                            }
                        }
                        Err(e) => {
                            error!("QuicStreamBridge: SUB recv error for '{}': {}", topic, e);
                            break;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
