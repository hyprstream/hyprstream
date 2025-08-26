//! Temporal streaming layer for real-time LoRA weight updates
//!
//! This module adds time-series capabilities to hyprstream's VDB storage,
//! enabling real-time temporal adaptation with gradient-based dependencies.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chrono::{DateTime, Utc};
use futures_util::Stream;
use tokio::sync::{mpsc, RwLock as TokioRwLock};
// use tokio_stream::wrappers::UnboundedReceiverStream; // Reserved for future use
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context, anyhow};
use pin_project_lite::pin_project;

use super::{
    VDBSparseStorage, SparseWeightUpdate, Coordinate3D,
    SparseStorage,
};
use crate::adapters::sparse_lora::SparseLoRAAdapter;

/// Configuration for temporal streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStreamingConfig {
    /// Stream update frequency in milliseconds
    pub stream_frequency_ms: u64,
    /// Maximum temporal window to keep in memory (seconds)
    pub temporal_window_secs: u64,
    /// Gradient accumulation window (milliseconds)
    pub gradient_window_ms: u64,
    /// Concept drift detection threshold
    pub drift_threshold: f32,
    /// Maximum concurrent streaming sessions
    pub max_streaming_sessions: usize,
    /// Buffer size for temporal updates
    pub update_buffer_size: usize,
}

impl Default for TemporalStreamingConfig {
    fn default() -> Self {
        Self {
            stream_frequency_ms: 100, // 10Hz streaming
            temporal_window_secs: 300, // 5 minutes
            gradient_window_ms: 1000, // 1 second
            drift_threshold: 0.15,
            max_streaming_sessions: 50,
            update_buffer_size: 10000,
        }
    }
}

/// Temporal weight update with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalWeightUpdate {
    pub adapter_id: String,
    pub layer_name: String,
    pub timestamp: DateTime<Utc>,
    pub weights: HashMap<Coordinate3D, f32>,
    pub gradient_magnitude: f32,
    pub concept_drift_score: f32,
    pub sequence_id: u64,
}

/// Temporal gradient for dependency tracking
#[derive(Debug, Clone)]
pub struct TemporalGradient {
    pub layer_name: String,
    pub timestamp: DateTime<Utc>,
    pub values: HashMap<Coordinate3D, f32>,
    pub magnitude: f32,
    pub dependencies: Vec<String>, // Layer dependencies
}

/// Temporal dependency between gradients
#[derive(Debug, Clone)]
pub struct TemporalDependency {
    pub source_layer: String,
    pub target_layer: String,
    pub strength: f32,
    pub temporal_distance: Duration,
}

/// Temporal selection result
#[derive(Debug, Clone)]
pub struct TemporalSelection {
    pub selected_layers: Vec<String>,
    pub temporal_weights: HashMap<String, f32>,
    pub confidence: f32,
    pub adaptation_type: AdaptationType,
}

/// Types of temporal adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationType {
    /// Continuous streaming adaptation
    Continuous,
    /// Batch-based adaptation
    Batch,
    /// Concept drift adaptation
    ConceptDrift,
    /// Reinforcement-based adaptation
    Reinforcement,
}

/// Streaming result for temporal operations
#[derive(Debug, Clone)]
pub struct TemporalStreamingResult {
    pub operation_id: u64,
    pub adapter_id: String,
    pub updates_applied: usize,
    pub processing_time_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// Temporal streaming layer that enhances VDB storage with time-series capabilities
pub struct TemporalStreamingLayer {
    /// Underlying VDB sparse storage
    vdb_storage: Arc<VDBSparseStorage>,
    
    /// Temporal streaming configuration
    config: TemporalStreamingConfig,
    
    /// Active streaming sessions
    active_sessions: Arc<TokioRwLock<HashMap<String, TemporalSession>>>,
    
    /// Temporal gradient accumulator
    gradient_accumulator: Arc<TokioRwLock<HashMap<String, Vec<TemporalGradient>>>>,
    
    /// Dependency tracker for gradient-based DAG
    dependency_tracker: Arc<TokioRwLock<Vec<TemporalDependency>>>,
    
    /// Concept drift detector
    drift_detector: Arc<TokioRwLock<ConceptDriftDetector>>,
    
    /// Sequence counter for ordering
    sequence_counter: Arc<TokioRwLock<u64>>,
}

/// Active temporal streaming session
#[derive(Debug)]
#[derive(Clone)]
pub struct TemporalSession {
    pub session_id: String,
    pub adapter_id: String,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub is_active: bool,
    pub update_sender: mpsc::UnboundedSender<TemporalWeightUpdate>,
}

/// Concept drift detection state
#[derive(Debug)]
pub struct ConceptDriftDetector {
    /// Recent gradient magnitudes for drift detection
    gradient_history: Vec<(DateTime<Utc>, f32)>,
    /// Baseline gradient magnitude
    baseline_magnitude: f32,
    /// Drift detection threshold
    threshold: f32,
}

impl ConceptDriftDetector {
    pub fn new(threshold: f32) -> Self {
        Self {
            gradient_history: Vec::new(),
            baseline_magnitude: 0.0,
            threshold,
        }
    }

    pub fn add_gradient(&mut self, magnitude: f32) {
        let now = Utc::now();
        self.gradient_history.push((now, magnitude));
        
        // Keep only recent history (last 5 minutes)
        let cutoff = now - chrono::Duration::minutes(5);
        self.gradient_history.retain(|(timestamp, _)| *timestamp > cutoff);
        
        // Update baseline
        if self.baseline_magnitude == 0.0 {
            self.baseline_magnitude = magnitude;
        } else {
            self.baseline_magnitude = self.baseline_magnitude * 0.9 + magnitude * 0.1;
        }
    }

    pub fn detect_drift(&self) -> f32 {
        if self.gradient_history.len() < 5 || self.baseline_magnitude == 0.0 {
            return 0.0;
        }

        let recent_avg = self.gradient_history
            .iter()
            .rev()
            .take(5)
            .map(|(_, mag)| *mag)
            .sum::<f32>() / 5.0;

        (recent_avg - self.baseline_magnitude).abs() / self.baseline_magnitude
    }
}

impl TemporalStreamingLayer {
    /// Create new temporal streaming layer
    pub async fn new(
        vdb_storage: Arc<VDBSparseStorage>,
        config: TemporalStreamingConfig,
    ) -> Result<Self> {
        Ok(Self {
            vdb_storage,
            config: config.clone(),
            active_sessions: Arc::new(TokioRwLock::new(HashMap::new())),
            gradient_accumulator: Arc::new(TokioRwLock::new(HashMap::new())),
            dependency_tracker: Arc::new(TokioRwLock::new(Vec::new())),
            drift_detector: Arc::new(TokioRwLock::new(
                ConceptDriftDetector::new(config.drift_threshold)
            )),
            sequence_counter: Arc::new(TokioRwLock::new(0)),
        })
    }

    /// Create temporal streaming session for real-time updates
    pub async fn create_temporal_stream(
        &self,
        adapter_id: String,
        layer_filter: Option<Vec<String>>,
    ) -> Result<TemporalWeightStream> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        // Create channel for streaming updates
        let (tx, rx) = mpsc::unbounded_channel();
        
        // Create streaming session
        let session = TemporalSession {
            session_id: session_id.clone(),
            adapter_id: adapter_id.clone(),
            created_at: Utc::now(),
            last_activity: Utc::now(),
            is_active: true,
            update_sender: tx.clone(),
        };

        // Register session
        {
            let mut sessions = self.active_sessions.write().await;
            if sessions.len() >= self.config.max_streaming_sessions {
                return Err(anyhow!("Maximum streaming sessions reached"));
            }
            sessions.insert(session_id.clone(), session);
        }

        // Spawn background streaming task
        let streaming_layer = Arc::new(self.clone());
        let session_id_for_spawn = session_id.clone();
        let handle = tokio::spawn(async move {
            streaming_layer.run_streaming_session(
                session_id_for_spawn,
                adapter_id,
                layer_filter,
                tx,
            ).await;
        });

        Ok(TemporalWeightStream {
            session_id,
            receiver: rx,
            _handle: handle,
            streaming_layer: Arc::new(self.clone()),
        })
    }

    /// Background streaming session handler
    async fn run_streaming_session(
        self: Arc<Self>,
        session_id: String,
        adapter_id: String,
        _layer_filter: Option<Vec<String>>,
        tx: mpsc::UnboundedSender<TemporalWeightUpdate>,
    ) {
        let mut interval = tokio::time::interval(
            Duration::from_millis(self.config.stream_frequency_ms)
        );

        let mut sequence = 0u64;

        while let Some(_session) = {
            let sessions = self.active_sessions.read().await;
            sessions.get(&session_id).filter(|s| s.is_active).cloned()
        } {
            interval.tick().await;
            sequence += 1;

            // Check for temporal updates
            if let Ok(update) = self.generate_temporal_update(
                &adapter_id,
                sequence,
            ).await {
                if tx.send(update).is_err() {
                    break; // Receiver dropped
                }
            }
        }

        // Clean up session
        let mut sessions = self.active_sessions.write().await;
        sessions.remove(&session_id);
    }

    /// Generate temporal update based on accumulated gradients
    async fn generate_temporal_update(
        &self,
        adapter_id: &str,
        sequence: u64,
    ) -> Result<TemporalWeightUpdate> {
        // Get recent gradients for this adapter
        let gradients = {
            let accumulator = self.gradient_accumulator.read().await;
            accumulator.get(adapter_id).cloned().unwrap_or_default()
        };

        if gradients.is_empty() {
            return Err(anyhow!("No gradients available for adapter {}", adapter_id));
        }

        // Select most recent gradient
        let latest_gradient = gradients
            .iter()
            .max_by_key(|g| g.timestamp)
            .context("No gradients found")?;

        // Compute concept drift
        let drift_score = {
            let mut detector = self.drift_detector.write().await;
            detector.add_gradient(latest_gradient.magnitude);
            detector.detect_drift()
        };

        Ok(TemporalWeightUpdate {
            adapter_id: adapter_id.to_string(),
            layer_name: latest_gradient.layer_name.clone(),
            timestamp: Utc::now(),
            weights: latest_gradient.values.clone(),
            gradient_magnitude: latest_gradient.magnitude,
            concept_drift_score: drift_score,
            sequence_id: sequence,
        })
    }

    /// Accumulate temporal gradients for layer
    pub async fn accumulate_temporal_gradients(
        &self,
        layer_name: &str,
        input: &SparseLoRAAdapter,
        target: &SparseLoRAAdapter,
        timestamp: Option<DateTime<Utc>>,
    ) -> Result<TemporalGradient> {
        let timestamp = timestamp.unwrap_or_else(Utc::now);

        // Compute gradient between input and target
        let gradient_values = self.compute_gradient_difference(input, target).await?;
        
        // Compute gradient magnitude
        let magnitude = gradient_values
            .values()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();

        let gradient = TemporalGradient {
            layer_name: layer_name.to_string(),
            timestamp,
            values: gradient_values,
            magnitude,
            dependencies: self.detect_layer_dependencies(layer_name).await,
        };

        // Store in accumulator
        {
            let mut accumulator = self.gradient_accumulator.write().await;
            let layer_gradients = accumulator.entry(layer_name.to_string()).or_insert_with(Vec::new);
            layer_gradients.push(gradient.clone());

            // Keep only recent gradients
            let cutoff = timestamp - chrono::Duration::milliseconds(self.config.gradient_window_ms as i64);
            layer_gradients.retain(|g| g.timestamp > cutoff);
        }

        Ok(gradient)
    }

    /// Compute sparse gradient difference between adapters
    async fn compute_gradient_difference(
        &self,
        input: &SparseLoRAAdapter,
        target: &SparseLoRAAdapter,
    ) -> Result<HashMap<Coordinate3D, f32>> {
        let input_weights = input.to_vdb_weights().await;
        let target_weights = target.to_vdb_weights().await;

        let mut gradient = HashMap::new();

        // Compute difference for all active coordinates
        for (linear_idx, target_value) in target_weights.active_iter() {
            let input_value = input_weights.get(linear_idx);
            let diff = target_value - input_value;
            if diff.abs() > 1e-8 {
                // Convert linear index to 3D coordinate for the gradient map
                let coord = target_weights.linear_to_coord(linear_idx);
                gradient.insert(coord, diff);
            }
        }

        // Also check input coordinates not in target
        for (linear_idx, input_value) in input_weights.active_iter() {
            let coord = input_weights.linear_to_coord(linear_idx);
            if !gradient.contains_key(&coord) {
                let target_value = target_weights.get(linear_idx);
                let diff = target_value - input_value;
                if diff.abs() > 1e-8 {
                    gradient.insert(coord, diff);
                }
            }
        }

        Ok(gradient)
    }

    /// Detect temporal dependencies between layers
    async fn detect_layer_dependencies(&self, layer_name: &str) -> Vec<String> {
        // Simple dependency detection based on layer name patterns
        // In practice, this would analyze gradient correlations
        match layer_name {
            name if name.contains("attention") => vec!["embedding".to_string()],
            name if name.contains("mlp") => vec!["attention".to_string()],
            name if name.contains("output") => vec!["mlp".to_string()],
            _ => vec![],
        }
    }

    /// Apply gradient update to temporal streaming layer
    pub async fn apply_gradient_update(
        &self,
        gradient: &TemporalGradient,
    ) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("🔄 Applying temporal gradient update for layer: {}", gradient.layer_name);
        
        // Apply gradient to the layer using the values map
        let learning_rate = 0.001; // Default learning rate
        for (coord, delta) in &gradient.values {
            if let Err(e) = self.vdb_storage.apply_weight_delta(coord, *delta * learning_rate).await {
                tracing::warn!("Failed to apply weight delta at {:?}: {}", coord, e);
            }
        }
        
        // Update temporal statistics  
        self.update_learning_stats(std::time::SystemTime::now()).await;
        
        Ok(())
    }
    
    
    /// Convert linear index to VDB coordinate
    fn index_to_coordinate(&self, index: usize) -> Coordinate3D {
        // Simple mapping - in practice this would be more sophisticated
        let x = (index % 64) as i32;
        let y = ((index / 64) % 64) as i32;
        let z = (index / (64 * 64)) as i32;
        Coordinate3D { x, y, z }
    }
    
    /// Update temporal learning statistics
    async fn update_learning_stats(&self, timestamp: std::time::SystemTime) {
        // Track gradient application frequency and timing
        tracing::trace!("📊 Updated temporal learning stats at {:?}", timestamp);
    }

    /// Select temporal adaptations based on gradient dependencies
    pub async fn select_temporal_lora(
        &self,
        layer_name: &str,
        input: &SparseLoRAAdapter,
        target: Option<&SparseLoRAAdapter>,
        window: chrono::Duration,
    ) -> Result<TemporalSelection> {
        // Get recent gradients for analysis
        let gradients = {
            let accumulator = self.gradient_accumulator.read().await;
            accumulator.get(layer_name).cloned().unwrap_or_default()
        };

        let cutoff = Utc::now() - window;
        let recent_gradients: Vec<_> = gradients
            .into_iter()
            .filter(|g| g.timestamp > cutoff)
            .collect();

        if recent_gradients.is_empty() {
            return Ok(TemporalSelection {
                selected_layers: vec![layer_name.to_string()],
                temporal_weights: HashMap::new(),
                confidence: 0.0,
                adaptation_type: AdaptationType::Continuous,
            });
        }

        // Analyze gradient patterns to select adaptation strategy
        let avg_magnitude = recent_gradients
            .iter()
            .map(|g| g.magnitude)
            .sum::<f32>() / recent_gradients.len() as f32;

        let drift_score = {
            let detector = self.drift_detector.read().await;
            detector.detect_drift()
        };

        // Determine adaptation type based on analysis
        let adaptation_type = if drift_score > self.config.drift_threshold {
            AdaptationType::ConceptDrift
        } else if recent_gradients.len() > 10 {
            AdaptationType::Batch
        } else {
            AdaptationType::Continuous
        };

        // Build temporal weights based on gradient correlations
        let mut temporal_weights = HashMap::new();
        for gradient in &recent_gradients {
            for dep_layer in &gradient.dependencies {
                let weight = (gradient.magnitude / avg_magnitude).min(1.0);
                temporal_weights.insert(dep_layer.clone(), weight);
            }
        }

        let confidence = if temporal_weights.is_empty() {
            0.5
        } else {
            temporal_weights.values().sum::<f32>() / temporal_weights.len() as f32
        };

        Ok(TemporalSelection {
            selected_layers: temporal_weights.keys().cloned().collect(),
            temporal_weights,
            confidence,
            adaptation_type,
        })
    }

    /// Apply temporal updates to VDB storage
    pub async fn apply_temporal_updates(
        &self,
        updates: &[TemporalWeightUpdate],
    ) -> Result<Vec<TemporalStreamingResult>> {
        let mut results = Vec::new();

        for update in updates {
            let start = Instant::now();
            let operation_id = {
                let mut counter = self.sequence_counter.write().await;
                *counter += 1;
                *counter
            };

            // Convert to VDB sparse update format
            let sparse_update = SparseWeightUpdate {
                adapter_id: update.adapter_id.clone(),
                updates: update.weights.clone(),
                timestamp: update.timestamp.timestamp() as u64,
                sequence: update.sequence_id,
            };

            // Apply through VDB storage
            let result = match self.vdb_storage.update_sparse_weights(&[sparse_update]).await {
                Ok(_) => TemporalStreamingResult {
                    operation_id,
                    adapter_id: update.adapter_id.clone(),
                    updates_applied: update.weights.len(),
                    processing_time_ms: start.elapsed().as_millis() as u64,
                    success: true,
                    error: None,
                },
                Err(e) => TemporalStreamingResult {
                    operation_id,
                    adapter_id: update.adapter_id.clone(),
                    updates_applied: 0,
                    processing_time_ms: start.elapsed().as_millis() as u64,
                    success: false,
                    error: Some(e.to_string()),
                },
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Get temporal streaming statistics
    pub async fn get_streaming_stats(&self) -> TemporalStreamingStats {
        let active_sessions = self.active_sessions.read().await.len();
        let total_gradients = {
            let accumulator = self.gradient_accumulator.read().await;
            accumulator.values().map(|v| v.len()).sum()
        };
        let dependencies = self.dependency_tracker.read().await.len();

        TemporalStreamingStats {
            active_sessions,
            total_gradients,
            dependencies,
            drift_score: {
                let detector = self.drift_detector.read().await;
                detector.detect_drift()
            },
            avg_gradient_magnitude: 0.0, // Would compute from accumulator
            temporal_window_secs: self.config.temporal_window_secs,
        }
    }
}

impl Clone for TemporalStreamingLayer {
    fn clone(&self) -> Self {
        Self {
            vdb_storage: Arc::clone(&self.vdb_storage),
            config: self.config.clone(),
            active_sessions: Arc::clone(&self.active_sessions),
            gradient_accumulator: Arc::clone(&self.gradient_accumulator),
            dependency_tracker: Arc::clone(&self.dependency_tracker),
            drift_detector: Arc::clone(&self.drift_detector),
            sequence_counter: Arc::clone(&self.sequence_counter),
        }
    }
}

/// Temporal streaming statistics
#[derive(Debug, Clone, Serialize)]
pub struct TemporalStreamingStats {
    pub active_sessions: usize,
    pub total_gradients: usize,
    pub dependencies: usize,
    pub drift_score: f32,
    pub avg_gradient_magnitude: f32,
    pub temporal_window_secs: u64,
}

// Stream implementation for temporal weight updates
pin_project! {
    /// Stream of temporal weight updates
    #[project = TemporalWeightStreamProjection]
    pub struct TemporalWeightStream {
        pub session_id: String,
        #[pin]
        receiver: mpsc::UnboundedReceiver<TemporalWeightUpdate>,
        _handle: tokio::task::JoinHandle<()>,
        streaming_layer: Arc<TemporalStreamingLayer>,
    }
}

impl Stream for TemporalWeightStream {
    type Item = Result<TemporalWeightUpdate>;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut this = self.project();
        match this.receiver.poll_recv(cx) {
            std::task::Poll::Ready(Some(update)) => std::task::Poll::Ready(Some(Ok(update))),
            std::task::Poll::Ready(None) => std::task::Poll::Ready(None),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

// Drop implementation handled by pin_project macro

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::vdb::{SparseStorageConfig, VDBSparseStorage};
    use crate::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig};

    #[tokio::test]
    async fn test_temporal_streaming_layer_creation() -> Result<()> {
        let vdb_config = SparseStorageConfig::default();
        let vdb_storage = Arc::new(VDBSparseStorage::new(vdb_config).await?);
        
        let streaming_config = TemporalStreamingConfig::default();
        let temporal_layer = TemporalStreamingLayer::new(vdb_storage, streaming_config).await?;
        
        assert_eq!(temporal_layer.config.stream_frequency_ms, 100);
        Ok(())
    }

    #[tokio::test]
    async fn test_temporal_gradient_accumulation() -> Result<()> {
        let vdb_config = SparseStorageConfig::default();
        let vdb_storage = Arc::new(VDBSparseStorage::new(vdb_config).await?);
        
        let streaming_config = TemporalStreamingConfig::default();
        let temporal_layer = TemporalStreamingLayer::new(vdb_storage, streaming_config).await?;

        // Create mock adapters
        let input_config = SparseLoRAConfig::default();
        let input_adapter = SparseLoRAAdapter::new(input_config, (1536, 1536)).await?;
        let target_adapter = SparseLoRAAdapter::new(input_config, (1536, 1536)).await?;

        // Test gradient accumulation
        let gradient = temporal_layer.accumulate_temporal_gradients(
            "test_layer",
            &input_adapter,
            &target_adapter,
            None,
        ).await?;

        assert_eq!(gradient.layer_name, "test_layer");
        assert!(gradient.magnitude >= 0.0);
        Ok(())
    }

    #[tokio::test]
    async fn test_concept_drift_detection() {
        let mut detector = ConceptDriftDetector::new(0.15);
        
        // Add stable gradients
        for _ in 0..10 {
            detector.add_gradient(0.1);
        }
        
        let initial_drift = detector.detect_drift();
        assert!(initial_drift < 0.1);
        
        // Add drifting gradients
        for _ in 0..5 {
            detector.add_gradient(0.5);
        }
        
        let drift_score = detector.detect_drift();
        assert!(drift_score > 0.1);
    }

    #[tokio::test]
    async fn test_temporal_streaming_session() -> Result<()> {
        let vdb_config = SparseStorageConfig::default();
        let vdb_storage = Arc::new(VDBSparseStorage::new(vdb_config).await?);
        
        let streaming_config = TemporalStreamingConfig::default();
        let temporal_layer = TemporalStreamingLayer::new(vdb_storage, streaming_config).await?;

        let stream = temporal_layer.create_temporal_stream(
            "test_adapter".to_string(),
            None,
        ).await?;

        assert!(!stream.session_id.is_empty());
        
        // Verify session was registered
        let sessions = temporal_layer.active_sessions.read().await;
        assert!(sessions.contains_key(&stream.session_id));
        
        Ok(())
    }
}