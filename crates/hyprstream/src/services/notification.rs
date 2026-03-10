//! Notification service — blind relay with broadcast encryption.
//!
//! Mediates pubkey exchange and routes encrypted capsules between publishers
//! and subscribers. NotificationService never sees plaintext payloads or
//! DH shared secrets.
//!
//! # Flow
//!
//! 1. Subscriber calls `subscribe(scope, ephemeral_pubkey)` → NS registers
//!    topic with StreamService, returns topic + endpoint
//! 2. Publisher calls `publishIntent(scope, publisher_pubkey)` → NS returns
//!    rerandomized (blinded) subscriber pubkeys
//! 3. Publisher encrypts payload, wraps data_key per subscriber, calls `deliver`
//! 4. NS routes encrypted capsules to subscriber topics via StreamPublisher
//! 5. Subscribers decrypt using blinding-aware DH

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use async_trait::async_trait;
use capnp::message::Builder as CapnpBuilder;
use capnp::serialize as capnp_serialize;
use hyprstream_rpc::crypto::notification::pubkey_fingerprint;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::registry::{global as registry, SocketKind};
use hyprstream_rpc::streaming::StreamChannel;
use hyprstream_rpc::transport::TransportConfig;
use parking_lot::RwLock;
use tracing::{debug, error, trace, warn};
use uuid::Uuid;

use crate::services::{Continuation, EnvelopeContext, PolicyClient, ZmqService};
use crate::services::generated::notification_client::{
    NotificationClient, NotificationHandler, NotificationResponseVariant,
    ErrorInfo, PingInfo,
    SubscribeRequest, SubscribeResponse,
    PublishIntentRequest, PublishIntentResponse,
    DeliverRequest, DeliverResponse,
    UnsubscribeRequest,
    SubscriptionInfo, SubscriptionList,
    dispatch_notification, serialize_response,
};

use hyprstream_rpc::crypto::rerandomize_pubkey;

/// Service name for endpoint registry.
const SERVICE_NAME: &str = "notification";

/// Default subscription TTL (10 minutes).
const DEFAULT_TTL_SECS: u32 = 600;

/// Maximum subscription TTL (1 hour).
const MAX_TTL_SECS: u32 = 3600;

/// Intent expiry time (30 seconds).
const INTENT_EXPIRY: Duration = Duration::from_secs(30);

/// Background expiry interval (30 seconds).
const EXPIRY_INTERVAL: Duration = Duration::from_secs(30);

// ============================================================================
// Subscriber Registry
// ============================================================================

/// A registered notification subscriber.
#[allow(dead_code)] // created_at used by diagnostic/debugging; expires_at used by expire()
struct Subscriber {
    id: Uuid,
    /// Verified identity from EnvelopeContext (for unsubscribe auth).
    subject: String,
    /// Parsed claim scope pattern: "serve:model:*".
    scope_pattern: String,
    /// Client Ristretto255 pubkey (32 bytes).
    ephemeral_pubkey: [u8; 32],
    /// Blake3(pubkey)[..16] for capsule routing (128-bit).
    pubkey_fingerprint: [u8; 16],
    /// Pre-registered XPUB topic.
    registered_topic: String,
    created_at: Instant,
    /// Wall-clock creation time for serialization.
    created_at_epoch: i64,
    expires_at: Instant,
    /// Wall-clock expiry time for serialization.
    expires_at_epoch: i64,
}

/// Registry of active subscribers indexed multiple ways.
struct SubscriberRegistry {
    by_id: HashMap<Uuid, Subscriber>,
    by_fingerprint: HashMap<[u8; 16], Uuid>,
    /// Scope prefix → subscriber IDs for fast scope matching.
    by_scope_prefix: HashMap<String, HashSet<Uuid>>,
}

impl SubscriberRegistry {
    fn new() -> Self {
        Self {
            by_id: HashMap::new(),
            by_fingerprint: HashMap::new(),
            by_scope_prefix: HashMap::new(),
        }
    }

    /// Insert a subscriber. Returns error if fingerprint collides.
    fn insert(&mut self, sub: Subscriber) -> Result<(), String> {
        if self.by_fingerprint.contains_key(&sub.pubkey_fingerprint) {
            return Err("pubkey fingerprint collision".to_owned());
        }
        let id = sub.id;
        let fingerprint = sub.pubkey_fingerprint;
        let scope = sub.scope_pattern.clone();

        // Index by scope prefix (first two segments for fast lookup)
        let prefix = scope_prefix(&scope);
        self.by_scope_prefix.entry(prefix).or_default().insert(id);
        self.by_fingerprint.insert(fingerprint, id);
        self.by_id.insert(id, sub);
        Ok(())
    }

    /// Remove a subscriber by ID. Returns the subscriber if found.
    fn remove(&mut self, id: &Uuid) -> Option<Subscriber> {
        if let Some(sub) = self.by_id.remove(id) {
            self.by_fingerprint.remove(&sub.pubkey_fingerprint);
            let prefix = scope_prefix(&sub.scope_pattern);
            if let Some(set) = self.by_scope_prefix.get_mut(&prefix) {
                set.remove(id);
                if set.is_empty() {
                    self.by_scope_prefix.remove(&prefix);
                }
            }
            Some(sub)
        } else {
            None
        }
    }

    /// Find subscribers matching a scope pattern (simple glob: `*` matches any suffix).
    ///
    /// Uses `by_scope_prefix` index for initial candidate filtering, then applies
    /// full glob match. Falls back to full scan for `*` pattern matches.
    fn match_scope(&self, scope: &str) -> Vec<&Subscriber> {
        let prefix = scope_prefix(scope);

        // Collect candidate IDs from prefix index
        let mut candidates: HashSet<Uuid> = HashSet::new();
        for (idx_prefix, ids) in &self.by_scope_prefix {
            // Include entries where the index prefix matches the scope's prefix,
            // or where the index prefix is a wildcard-like pattern
            if idx_prefix == &prefix || idx_prefix == "*" || prefix.starts_with(idx_prefix) {
                candidates.extend(ids);
            }
        }

        // Also include subscribers with wildcard-only patterns (e.g., "*")
        if let Some(ids) = self.by_scope_prefix.get("*") {
            candidates.extend(ids);
        }

        // Apply full glob match on candidates
        candidates.iter()
            .filter_map(|id| self.by_id.get(id))
            .filter(|sub| scope_matches(&sub.scope_pattern, scope))
            .collect()
    }

    /// Remove expired subscribers. Returns count removed.
    fn expire(&mut self, now: Instant) -> usize {
        let expired: Vec<Uuid> = self.by_id
            .values()
            .filter(|sub| now >= sub.expires_at)
            .map(|sub| sub.id)
            .collect();
        let count = expired.len();
        for id in &expired {
            self.remove(id);
        }
        count
    }

    /// List subscriptions for a given subject.
    fn list_for_subject(&self, subject: &str) -> Vec<&Subscriber> {
        self.by_id
            .values()
            .filter(|sub| sub.subject == subject)
            .collect()
    }

    fn count(&self) -> usize {
        self.by_id.len()
    }
}

/// Extract first two colon-separated segments as scope prefix for indexing.
fn scope_prefix(scope: &str) -> String {
    let parts: Vec<&str> = scope.splitn(3, ':').collect();
    match parts.len() {
        0 => String::new(),
        1 => parts[0].to_owned(),
        _ => format!("{}:{}", parts[0], parts[1]),
    }
}

/// Check if a subscriber's scope pattern matches a publisher's scope.
///
/// Simple `keyMatch`-style glob: `serve:model:*` matches `serve:model:qwen3`.
fn scope_matches(pattern: &str, scope: &str) -> bool {
    if pattern == scope || pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix(":*") {
        scope.starts_with(prefix) && scope[prefix.len()..].starts_with(':')
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        scope.starts_with(prefix)
    } else {
        pattern == scope
    }
}

// ============================================================================
// Pending Intent
// ============================================================================

/// A pending publish intent awaiting delivery.
struct PendingIntent {
    scope: String,
    /// Verified caller identity — delivery must come from same subject.
    publisher_subject: String,
    /// Publisher's ephemeral Ristretto pubkey (32 bytes).
    publisher_pubkey: [u8; 32],
    /// Matched subscribers with blinding info.
    matched_subscribers: Vec<MatchedSubscriber>,
    created_at: Instant,
}

/// A subscriber matched to a publish intent.
struct MatchedSubscriber {
    #[allow(dead_code)] // Used for logging/diagnostics
    id: Uuid,
    /// Blinding scalar r_i for Ristretto rerandomization.
    blinding_scalar: [u8; 32],
    /// Blake3(blinded_pubkey)[..16] — used for capsule routing.
    blinded_fingerprint: [u8; 16],
    /// The registered topic for this subscriber.
    topic: String,
}

// ============================================================================
// NotificationBlock serialization
// ============================================================================

/// Serialize a NotificationBlock as Cap'n Proto bytes.
///
/// NS constructs this from the DeliverRequest fields + intent metadata.
/// The subscriber parses it to extract all fields needed for decryption.
fn serialize_notification_block(
    publisher_pubkey: &[u8; 32],
    blinding_scalar: &[u8; 32],
    wrapped_key: &[u8],
    key_nonce: &[u8],
    encrypted_payload: &[u8],
    nonce: &[u8],
    intent_id: &str,
    scope: &str,
    publisher_mac: &[u8],
) -> Vec<u8> {
    let mut builder = CapnpBuilder::new_default();
    {
        let mut block = builder.init_root::<crate::notification_capnp::notification_block::Builder<'_>>();
        block.set_publisher_pubkey(publisher_pubkey);
        block.set_blinding_scalar(blinding_scalar);
        block.set_wrapped_key(wrapped_key);
        block.set_key_nonce(key_nonce);
        block.set_encrypted_payload(encrypted_payload);
        block.set_nonce(nonce);
        block.set_intent_id(intent_id);
        block.set_scope(scope);
        block.set_publisher_mac(publisher_mac);
    }
    let mut output = Vec::new();
    #[allow(clippy::expect_used)] // writing to Vec<u8> is infallible
    capnp_serialize::write_message(&mut output, &builder).expect("capnp serialization to Vec is infallible");
    output
}

// ============================================================================
// NotificationService
// ============================================================================

/// Notification service: blind relay with broadcast encryption.
pub struct NotificationService {
    subscribers: Arc<RwLock<SubscriberRegistry>>,
    pending_intents: Arc<RwLock<HashMap<String, PendingIntent>>>,
    delivery_counter: AtomicU64,
    /// StreamChannel for topic registration and publishing to StreamService.
    stream_channel: StreamChannel,
    // Auth
    signing_key: Arc<SigningKey>,
    policy_client: Option<PolicyClient>,
    expected_audience: Option<String>,
    /// Local OAuth issuer URL for distinguishing local vs. federated JWTs.
    local_issuer_url: Option<String>,
    /// Federation key source for verifying externally-issued JWTs.
    federation_key_source: Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>>,
    // Infrastructure
    context: Arc<zmq::Context>,
    transport: TransportConfig,
}

impl NotificationService {
    /// Create a new notification service.
    pub fn new(
        signing_key: Arc<SigningKey>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
    ) -> Self {
        let stream_channel = StreamChannel::new(
            Arc::clone(&context),
            (*signing_key).clone(),
        );

        let subscribers = Arc::new(RwLock::new(SubscriberRegistry::new()));

        // Spawn background expiry task
        let subs_for_expiry = Arc::clone(&subscribers);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(EXPIRY_INTERVAL).await;
                let expired = subs_for_expiry.write().expire(Instant::now());
                if expired > 0 {
                    debug!("Expired {} notification subscriptions", expired);
                }
            }
        });

        Self {
            subscribers,
            pending_intents: Arc::new(RwLock::new(HashMap::new())),
            delivery_counter: AtomicU64::new(0),
            stream_channel,
            signing_key,
            policy_client: None,
            expected_audience: None,
            local_issuer_url: None,
            federation_key_source: None,
            context,
            transport,
        }
    }

    /// Set the policy client for authorization checks.
    pub fn with_policy_client(mut self, client: PolicyClient) -> Self {
        self.policy_client = Some(client);
        self
    }

    /// Set expected audience for JWT validation.
    pub fn with_expected_audience(mut self, audience: String) -> Self {
        self.expected_audience = Some(audience);
        self
    }

    /// Set the local OAuth issuer URL for distinguishing local vs. federated JWTs.
    pub fn with_local_issuer_url(mut self, url: String) -> Self {
        self.local_issuer_url = Some(url);
        self
    }

    /// Set the federation key source for verifying externally-issued JWTs.
    pub fn with_federation_key_source(
        mut self,
        src: std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>,
    ) -> Self {
        self.federation_key_source = Some(src);
        self
    }

    /// Generate a unique XPUB topic for a subscriber.
    fn generate_topic(&self) -> String {
        format!("notify/{}", Uuid::new_v4())
    }
}

// ============================================================================
// NotificationHandler implementation
// ============================================================================

#[async_trait(?Send)]
impl NotificationHandler for NotificationService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> Result<()> {
        if ctx.identity.is_local() {
            return Ok(());
        }
        if let Some(ref policy_client) = self.policy_client {
            let subject = ctx.subject().to_string();
            let allowed = policy_client
                .check(&subject, "*", resource, operation)
                .await
                .unwrap_or_else(|e| {
                    warn!("Notification policy check failed for {}: {}", subject, e);
                    false
                });
            if allowed {
                Ok(())
            } else {
                anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
            }
        } else {
            Ok(())
        }
    }

    async fn handle_subscribe(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &SubscribeRequest,
    ) -> Result<NotificationResponseVariant> {
        let scope_pattern = data.scope_pattern.clone();
        let ephemeral_pubkey_data = &data.ephemeral_pubkey;
        let ttl_seconds = data.ttl_seconds;

        // Validate pubkey length
        if ephemeral_pubkey_data.len() != 32 {
            return Ok(NotificationResponseVariant::Error(ErrorInfo {
                message: format!("ephemeral pubkey must be 32 bytes, got {}", ephemeral_pubkey_data.len()),
                code: "INVALID_PUBKEY".to_owned(),
                details: String::new(),
            }));
        }

        let mut ephemeral_pubkey = [0u8; 32];
        ephemeral_pubkey.copy_from_slice(ephemeral_pubkey_data);
        let fingerprint = pubkey_fingerprint(&ephemeral_pubkey);

        // TTL with bounds
        let ttl = if ttl_seconds == 0 {
            DEFAULT_TTL_SECS
        } else {
            ttl_seconds.min(MAX_TTL_SECS)
        };

        let now = Instant::now();
        let now_epoch = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64;
        let ttl_secs = u64::from(ttl);
        let sub_id = Uuid::new_v4();
        let topic = self.generate_topic();
        let subject = ctx.subject().to_string();

        // Pre-register topic with StreamService so messages are buffered
        // even before the subscriber connects their SUB socket.
        let expiry = chrono::Utc::now().timestamp() + ttl_secs as i64;
        if let Err(e) = self.stream_channel.register_topic(&topic, expiry, None).await {
            warn!("Failed to register topic {} with StreamService: {}", topic, e);
            return Ok(NotificationResponseVariant::Error(ErrorInfo {
                message: "Failed to register notification topic".to_owned(),
                code: "STREAM_ERROR".to_owned(),
                details: e.to_string(),
            }));
        }

        let subscriber = Subscriber {
            id: sub_id,
            subject,
            scope_pattern,
            ephemeral_pubkey,
            pubkey_fingerprint: fingerprint,
            registered_topic: topic.clone(),
            created_at: now,
            created_at_epoch: now_epoch,
            expires_at: now + Duration::from_secs(ttl_secs),
            expires_at_epoch: now_epoch + ttl_secs as i64,
        };

        let mut reg = self.subscribers.write();
        if let Err(e) = reg.insert(subscriber) {
            return Ok(NotificationResponseVariant::Error(ErrorInfo {
                message: e,
                code: "FINGERPRINT_COLLISION".to_owned(),
                details: String::new(),
            }));
        }
        drop(reg);

        debug!("Subscriber {} registered for topic {}", sub_id, topic);

        // Get StreamService endpoint for client to connect SUB socket
        let stream_endpoint = self.stream_channel.stream_endpoint();

        Ok(NotificationResponseVariant::SubscribeResult(SubscribeResponse {
            subscription_id: sub_id.to_string(),
            assigned_topic: topic,
            stream_endpoint,
        }))
    }

    async fn handle_publish_intent(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &PublishIntentRequest,
    ) -> Result<NotificationResponseVariant> {
        let scope = data.scope.clone();
        let publisher_pubkey_data = &data.publisher_pubkey;

        if publisher_pubkey_data.len() != 32 {
            return Ok(NotificationResponseVariant::Error(ErrorInfo {
                message: format!("publisher pubkey must be 32 bytes, got {}", publisher_pubkey_data.len()),
                code: "INVALID_PUBKEY".to_owned(),
                details: String::new(),
            }));
        }

        let mut publisher_pubkey = [0u8; 32];
        publisher_pubkey.copy_from_slice(publisher_pubkey_data);

        // Match subscribers by scope
        let reg = self.subscribers.read();
        let matched = reg.match_scope(&scope);

        if matched.is_empty() {
            drop(reg);
            return Ok(NotificationResponseVariant::Error(ErrorInfo {
                message: format!("No subscribers for scope '{}'", scope),
                code: "NO_SUBSCRIBERS".to_owned(),
                details: String::new(),
            }));
        }

        // Rerandomize each subscriber's pubkey
        let mut matched_subscribers = Vec::with_capacity(matched.len());
        let mut blinded_pubkeys: Vec<Vec<u8>> = Vec::with_capacity(matched.len());

        for sub in &matched {
            let sub_pubkey = hyprstream_rpc::crypto::RistrettoPublic::from_bytes(&sub.ephemeral_pubkey);
            match sub_pubkey {
                Some(pk) => {
                    let (blinded, r_bytes) = rerandomize_pubkey(&pk);
                    let blinded_bytes = blinded.to_bytes();
                    let blinded_fp = pubkey_fingerprint(&blinded_bytes);

                    blinded_pubkeys.push(blinded_bytes.to_vec());
                    matched_subscribers.push(MatchedSubscriber {
                        id: sub.id,
                        blinding_scalar: r_bytes,
                        blinded_fingerprint: blinded_fp,
                        topic: sub.registered_topic.clone(),
                    });
                }
                None => {
                    warn!("Invalid pubkey for subscriber {}, skipping", sub.id);
                }
            }
        }
        drop(reg);

        let intent_id = Uuid::new_v4().to_string();
        let publisher_subject = ctx.subject().to_string();

        let intent = PendingIntent {
            scope,
            publisher_subject,
            publisher_pubkey,
            matched_subscribers,
            created_at: Instant::now(),
        };

        self.pending_intents.write().insert(intent_id.clone(), intent);

        debug!("PublishIntent {} created with {} recipients", intent_id, blinded_pubkeys.len());

        Ok(NotificationResponseVariant::PublishIntentResult(PublishIntentResponse {
            intent_id,
            recipient_pubkeys: blinded_pubkeys,
        }))
    }

    async fn handle_deliver(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &DeliverRequest,
    ) -> Result<NotificationResponseVariant> {
        let intent_id = data.intent_id.clone();
        let caller = ctx.subject().to_string();

        // Verify caller identity and check expiry BEFORE removing from map.
        // This prevents an attacker from destroying valid intents by sending
        // unauthorized deliver requests.
        {
            let intents = self.pending_intents.read();
            match intents.get(&intent_id) {
                Some(i) => {
                    if caller != i.publisher_subject {
                        return Ok(NotificationResponseVariant::Error(ErrorInfo {
                            message: "Unauthorized: intent belongs to a different subject".to_owned(),
                            code: "UNAUTHORIZED".to_owned(),
                            details: String::new(),
                        }));
                    }
                    if i.created_at.elapsed() > INTENT_EXPIRY {
                        // Drop read lock before acquiring write lock to remove expired intent
                        drop(intents);
                        self.pending_intents.write().remove(&intent_id);
                        return Ok(NotificationResponseVariant::Error(ErrorInfo {
                            message: "Intent expired".to_owned(),
                            code: "EXPIRED".to_owned(),
                            details: String::new(),
                        }));
                    }
                }
                None => {
                    return Ok(NotificationResponseVariant::Error(ErrorInfo {
                        message: format!("Intent '{}' not found", intent_id),
                        code: "NOT_FOUND".to_owned(),
                        details: String::new(),
                    }));
                }
            }
        }

        // Now safe to remove — caller is verified and intent is not expired.
        // The prior lookup confirmed it exists; `remove` returns None only if a
        // concurrent caller already consumed it (which this service's single-task
        // dispatch prevents).
        #[allow(clippy::expect_used)]
        let intent = self.pending_intents.write().remove(&intent_id)
            .expect("intent verified above and only removed here");

        // Build a lookup from blinded_fingerprint → capsule for routing
        let mut capsule_map: HashMap<[u8; 16], &crate::services::generated::notification_client::RecipientCapsule> =
            HashMap::with_capacity(data.capsules.len());
        for capsule in &data.capsules {
            if capsule.pubkey_fingerprint.len() == 16 {
                let mut fp = [0u8; 16];
                fp.copy_from_slice(&capsule.pubkey_fingerprint);
                capsule_map.insert(fp, capsule);
            }
        }

        // Route capsules to subscriber topics via StreamPublisher
        let mut delivered: u32 = 0;
        for matched_sub in &intent.matched_subscribers {
            let capsule = match capsule_map.get(&matched_sub.blinded_fingerprint) {
                Some(c) => c,
                None => {
                    warn!(
                        "No capsule for subscriber fingerprint {:?} in intent {}",
                        &matched_sub.blinded_fingerprint[..4], intent_id
                    );
                    continue;
                }
            };

            // Serialize NotificationBlock with all fields the subscriber needs
            let block_bytes = serialize_notification_block(
                &intent.publisher_pubkey,
                &matched_sub.blinding_scalar,
                &capsule.wrapped_key,
                &capsule.key_nonce,
                &data.encrypted_payload,
                &data.nonce,
                &intent_id,
                &intent.scope,
                &capsule.mac,
            );

            // Publish to the subscriber's pre-registered topic via StreamPublisher
            match self.stream_channel.publisher_for_topic(&matched_sub.topic).await {
                Ok(mut publisher) => {
                    if let Err(e) = publisher.publish_data(&block_bytes).await {
                        error!("Failed to publish to topic {}: {}", matched_sub.topic, e);
                        continue;
                    }
                    if let Err(e) = publisher.complete(b"").await {
                        error!("Failed to complete topic {}: {}", matched_sub.topic, e);
                        continue;
                    }
                    delivered += 1;
                }
                Err(e) => {
                    error!("Failed to create publisher for topic {}: {}", matched_sub.topic, e);
                }
            }
        }

        self.delivery_counter.fetch_add(u64::from(delivered), Ordering::Relaxed);
        debug!("Deliver for intent {} to {}/{} recipients", intent_id, delivered, intent.matched_subscribers.len());

        Ok(NotificationResponseVariant::DeliverResult(DeliverResponse {
            delivered_count: delivered,
        }))
    }

    async fn handle_unsubscribe(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
        data: &UnsubscribeRequest,
    ) -> Result<NotificationResponseVariant> {
        let subscription_id = &data.subscription_id;
        let uuid = match Uuid::parse_str(subscription_id) {
            Ok(u) => u,
            Err(_) => {
                return Ok(NotificationResponseVariant::Error(ErrorInfo {
                    message: "Invalid subscription ID format".to_owned(),
                    code: "INVALID_ID".to_owned(),
                    details: String::new(),
                }));
            }
        };

        let mut reg = self.subscribers.write();
        match reg.by_id.get(&uuid) {
            Some(sub) if sub.subject == ctx.subject().to_string() => {}
            Some(_) => {
                return Ok(NotificationResponseVariant::Error(ErrorInfo {
                    message: "Cannot unsubscribe: subscription belongs to another identity".to_owned(),
                    code: "UNAUTHORIZED".to_owned(),
                    details: String::new(),
                }));
            }
            None => {
                return Ok(NotificationResponseVariant::Error(ErrorInfo {
                    message: "Subscription not found".to_owned(),
                    code: "NOT_FOUND".to_owned(),
                    details: String::new(),
                }));
            }
        }
        reg.remove(&uuid);
        drop(reg);

        debug!("Unsubscribed {}", subscription_id);
        Ok(NotificationResponseVariant::UnsubscribeResult)
    }

    async fn handle_list_subscriptions(
        &self,
        ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<NotificationResponseVariant> {
        let subject = ctx.subject().to_string();
        let reg = self.subscribers.read();
        let subs = reg.list_for_subject(&subject);

        let infos: Vec<SubscriptionInfo> = subs
            .iter()
            .map(|sub| SubscriptionInfo {
                subscription_id: sub.id.to_string(),
                scope_pattern: sub.scope_pattern.clone(),
                created_at: sub.created_at_epoch,
                expires_at: sub.expires_at_epoch,
            })
            .collect();
        drop(reg);

        Ok(NotificationResponseVariant::ListSubscriptionsResult(SubscriptionList {
            subscriptions: infos,
        }))
    }

    async fn handle_ping(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> Result<NotificationResponseVariant> {
        let reg = self.subscribers.read();
        let active = reg.count() as u32;
        drop(reg);

        let total = self.delivery_counter.load(Ordering::Relaxed);

        Ok(NotificationResponseVariant::PingResult(PingInfo {
            status: "ok".to_owned(),
            active_subscriptions: active,
            total_delivered: total,
        }))
    }
}

// ============================================================================
// ZmqService implementation
// ============================================================================

#[async_trait(?Send)]
impl ZmqService for NotificationService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        trace!(
            "Notification request from {} (id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_notification(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        "notification"
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        (*self.signing_key).clone()
    }

    fn expected_audience(&self) -> Option<&str> {
        self.expected_audience.as_deref()
    }

    fn local_issuer_url(&self) -> Option<&str> {
        self.local_issuer_url.as_deref()
    }

    fn federation_key_source(
        &self,
    ) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>> {
        self.federation_key_source.clone()
    }

    fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
        let variant = NotificationResponseVariant::Error(ErrorInfo {
            message: error.to_owned(),
            code: "INTERNAL".to_owned(),
            details: String::new(),
        });
        serialize_response(request_id, &variant).unwrap_or_default()
    }
}

// ============================================================================
// NotificationClient construction
// ============================================================================

impl NotificationClient {
    /// Create a new notification client (endpoint from registry).
    pub fn new(signing_key: SigningKey, identity: RequestIdentity) -> Self {
        let endpoint = registry().endpoint(SERVICE_NAME, SocketKind::Rep).to_zmq_string();
        crate::services::core::create_service_client(&endpoint, signing_key, identity)
    }
}

// ============================================================================
// NotificationPublisher (hand-written crypto wrapper)
// ============================================================================

/// Publisher-side helper for encrypted notification delivery.
///
/// Wraps `NotificationClient` with two-phase DH + AES-GCM broadcast encryption.
/// Used by services (e.g., ModelService) to publish encrypted events.
///
/// # Usage
///
/// ```ignore
/// let publisher = NotificationPublisher::new(notification_client, signing_key);
/// let delivered = publisher.publish(
///     "serve:model:qwen3",
///     &serde_json::to_vec(&event)?,
/// ).await?;
/// ```
pub struct NotificationPublisher {
    client: NotificationClient,
    /// Ed25519 signing key for attestation inside encrypted payloads.
    #[allow(dead_code)] // Used in Phase 10 (Ed25519 attestation inside encrypted payload)
    signing_key: SigningKey,
}

impl NotificationPublisher {
    /// Create a new notification publisher.
    pub fn new(client: NotificationClient, signing_key: SigningKey) -> Self {
        Self { client, signing_key }
    }

    /// Two-phase encrypted publish:
    ///
    /// 1. `publishIntent` → get blinded subscriber pubkeys from NS
    /// 2. Generate data_key, encrypt payload, wrap key per subscriber
    /// 3. `deliver` → NS routes capsules to subscriber topics
    ///
    /// Returns the number of subscribers who received the notification,
    /// or 0 if no subscribers matched the scope.
    pub async fn publish(&self, scope: &str, payload: &[u8]) -> Result<u32> {
        use hyprstream_rpc::crypto::key_exchange::generate_ephemeral_keypair;
        use hyprstream_rpc::crypto::notification::BroadcastEncryptor;

        // Generate fresh ephemeral keypair for this publish
        let (ephemeral_secret, ephemeral_pub) = generate_ephemeral_keypair();
        let pub_bytes = ephemeral_pub.to_bytes();

        // Phase 1: publishIntent → get blinded subscriber pubkeys
        let intent_resp = self.client.publish_intent(scope, &pub_bytes).await?;

        let blinded_pubkeys: Vec<[u8; 32]> = intent_resp.recipient_pubkeys.iter()
            .filter_map(|pk| {
                if pk.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(pk);
                    Some(arr)
                } else {
                    None
                }
            })
            .collect();

        if blinded_pubkeys.is_empty() {
            return Ok(0);
        }

        // Phase 2: Encrypt payload for all subscribers
        let encryptor = BroadcastEncryptor::new(intent_resp.intent_id.clone(), scope.to_owned());
        let encrypted = encryptor.encrypt(
            &ephemeral_secret.scalar().to_bytes(),
            &pub_bytes,
            &blinded_pubkeys,
            payload,
        )?;

        // Phase 3: deliver → NS routes capsules
        let capsules: Vec<_> = encrypted.capsules.into_iter().map(|c| {
            crate::services::generated::notification_client::RecipientCapsule {
                pubkey_fingerprint: c.fingerprint.to_vec(),
                wrapped_key: c.wrapped_key,
                key_nonce: c.key_nonce.to_vec(),
                mac: c.mac.to_vec(),
            }
        }).collect();

        let deliver_resp = self.client.deliver(
            &intent_resp.intent_id,
            &capsules,
            &encrypted.ciphertext,
            &encrypted.nonce,
        ).await?;

        Ok(deliver_resp.delivered_count)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_matches() {
        assert!(scope_matches("serve:model:*", "serve:model:qwen3"));
        assert!(scope_matches("serve:model:*", "serve:model:llama2"));
        assert!(!scope_matches("serve:model:*", "train:model:qwen3"));
        assert!(!scope_matches("serve:model:*", "serve:adapter:foo"));
        assert!(scope_matches("*", "anything:here"));
        assert!(scope_matches("serve:model:qwen3", "serve:model:qwen3"));
        assert!(!scope_matches("serve:model:qwen3", "serve:model:qwen4"));
    }

    #[test]
    fn test_scope_prefix() {
        assert_eq!(scope_prefix("serve:model:qwen3"), "serve:model");
        assert_eq!(scope_prefix("serve:model:*"), "serve:model");
        assert_eq!(scope_prefix("serve"), "serve");
        assert_eq!(scope_prefix(""), "");
    }

    #[test]
    fn test_subscriber_registry_basic() {
        let mut reg = SubscriberRegistry::new();
        let now = Instant::now();

        let sub = Subscriber {
            id: Uuid::new_v4(),
            subject: "local:test".to_owned(),
            scope_pattern: "serve:model:*".to_owned(),
            ephemeral_pubkey: [1u8; 32],
            pubkey_fingerprint: pubkey_fingerprint(&[1u8; 32]),
            registered_topic: "notify/test".to_owned(),
            created_at: now,
            created_at_epoch: 1000,
            expires_at: now + Duration::from_secs(600),
            expires_at_epoch: 1600,
        };
        let id = sub.id;

        assert!(reg.insert(sub).is_ok());
        assert_eq!(reg.count(), 1);

        // Match scope
        let matched = reg.match_scope("serve:model:qwen3");
        assert_eq!(matched.len(), 1);

        // No match
        let matched = reg.match_scope("train:model:qwen3");
        assert!(matched.is_empty());

        // Remove
        assert!(reg.remove(&id).is_some());
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn test_fingerprint_collision_rejected() {
        let mut reg = SubscriberRegistry::new();
        let now = Instant::now();

        let sub1 = Subscriber {
            id: Uuid::new_v4(),
            subject: "local:a".to_owned(),
            scope_pattern: "serve:model:*".to_owned(),
            ephemeral_pubkey: [1u8; 32],
            pubkey_fingerprint: pubkey_fingerprint(&[1u8; 32]),
            registered_topic: "notify/a".to_owned(),
            created_at: now,
            created_at_epoch: 1000,
            expires_at: now + Duration::from_secs(600),
            expires_at_epoch: 1600,
        };
        let sub2 = Subscriber {
            id: Uuid::new_v4(),
            subject: "local:b".to_owned(),
            scope_pattern: "serve:model:*".to_owned(),
            ephemeral_pubkey: [1u8; 32], // Same pubkey → same fingerprint
            pubkey_fingerprint: pubkey_fingerprint(&[1u8; 32]),
            registered_topic: "notify/b".to_owned(),
            created_at: now,
            created_at_epoch: 1000,
            expires_at: now + Duration::from_secs(600),
            expires_at_epoch: 1600,
        };

        assert!(reg.insert(sub1).is_ok());
        assert!(reg.insert(sub2).is_err()); // Collision
    }

    #[test]
    fn test_expiry() {
        let mut reg = SubscriberRegistry::new();
        let now = Instant::now();

        let sub = Subscriber {
            id: Uuid::new_v4(),
            subject: "local:test".to_owned(),
            scope_pattern: "serve:model:*".to_owned(),
            ephemeral_pubkey: [2u8; 32],
            pubkey_fingerprint: pubkey_fingerprint(&[2u8; 32]),
            registered_topic: "notify/expiry-test".to_owned(),
            created_at: now,
            created_at_epoch: 1000,
            // Already expired
            expires_at: now - Duration::from_secs(1),
            expires_at_epoch: 999,
        };

        assert!(reg.insert(sub).is_ok());
        assert_eq!(reg.count(), 1);
        assert_eq!(reg.expire(now), 1);
        assert_eq!(reg.count(), 0);
    }
}
