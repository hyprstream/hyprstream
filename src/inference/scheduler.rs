//! Continuous batching scheduler with LoRA adapter routing
//! 
//! Manages sequence scheduling, batching, and memory operations

use anyhow::{Result, anyhow};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use chrono::{DateTime, Utc};

use crate::inference::block_engine::{BlockEngine, AllocStatus};
use crate::runtime::conversation_router::ConversationRouter;
use crate::config::GenerationRequest;

/// Sequence ID type
pub type SeqID = usize;

/// Sequence status in the scheduler
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceStatus {
    /// Waiting to be scheduled
    Pending,
    /// Currently running
    Running,
    /// Swapped out to CPU/VDB
    SwappedOut,
    /// Generation completed
    Finished,
    /// Aborted due to error
    Aborted,
}

/// Individual sequence in generation
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence ID
    pub seq_id: SeqID,
    /// Input prompt tokens
    pub prompt_tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,
    /// Current status
    pub status: SequenceStatus,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Associated LoRA adapter ID
    pub lora_adapter_id: Option<String>,
    /// Generation parameters
    pub generation_params: GenerationParams,
}

impl Sequence {
    pub fn new(seq_id: SeqID, prompt_tokens: Vec<u32>, params: GenerationParams) -> Self {
        let now = Utc::now();
        Self {
            seq_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            status: SequenceStatus::Pending,
            created_at: now,
            updated_at: now,
            lora_adapter_id: None,
            generation_params: params,
        }
    }
    
    pub fn get_len(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }
    
    pub fn get_prompt_len(&self) -> usize {
        self.prompt_tokens.len()
    }
    
    pub fn is_finished(&self) -> bool {
        matches!(self.status, SequenceStatus::Finished | SequenceStatus::Aborted)
    }
}

/// Generation parameters for a sequence
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub stop_tokens: Vec<u32>,
    pub stream: bool,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            stop_tokens: vec![2], // EOS token
            stream: false,
        }
    }
}

/// Group of sequences (for beam search or parallel sampling)
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    /// Group ID
    pub group_id: usize,
    /// Sequences in this group
    pub sequences: HashMap<SeqID, Arc<RwLock<Sequence>>>,
    /// Sampling parameters
    pub sampling_params: SamplingParams,
    /// Response channel for streaming
    pub response_tx: Option<mpsc::UnboundedSender<GenerationChunk>>,
    /// Priority for scheduling
    pub priority: f32,
}

impl SequenceGroup {
    pub fn new(
        group_id: usize,
        sequences: Vec<Sequence>,
        sampling_params: SamplingParams,
    ) -> Self {
        let mut seq_map = HashMap::new();
        for seq in sequences {
            seq_map.insert(seq.seq_id, Arc::new(RwLock::new(seq)));
        }
        
        Self {
            group_id,
            sequences: seq_map,
            sampling_params,
            response_tx: None,
            priority: 1.0,
        }
    }
    
    pub async fn get_seqs(&self) -> Vec<(SeqID, Sequence)> {
        let mut seqs = Vec::new();
        for (&id, seq) in &self.sequences {
            let seq_guard = seq.read().await;
            seqs.push((id, seq_guard.clone()));
        }
        seqs
    }
    
    pub async fn get_status(&self) -> SequenceStatus {
        // Return the most restrictive status
        for seq in self.sequences.values() {
            let seq_guard = seq.read().await;
            if seq_guard.status == SequenceStatus::Pending {
                return SequenceStatus::Pending;
            }
        }
        SequenceStatus::Running
    }
    
    pub async fn get_prompt_len(&self) -> usize {
        let mut max_len = 0;
        for seq in self.sequences.values() {
            let seq_guard = seq.read().await;
            max_len = max_len.max(seq_guard.get_prompt_len());
        }
        max_len
    }
}

/// Sampling parameters for generation
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub n: usize,  // Number of sequences to generate
    pub best_of: Option<usize>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub use_beam_search: bool,
    pub early_stopping: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            best_of: None,
            temperature: 0.7,
            top_p: 0.9,
            top_k: Some(40),
            use_beam_search: false,
            early_stopping: true,
        }
    }
}

/// Generation chunk for streaming
#[derive(Debug, Clone)]
pub struct GenerationChunk {
    pub seq_id: SeqID,
    pub token: String,
    pub token_id: u32,
    pub finish_reason: Option<FinishReason>,
}

/// Reason for finishing generation
#[derive(Debug, Clone)]
pub enum FinishReason {
    MaxTokens,
    StopToken,
    EOS,
}

/// Scheduler output for batch processing
pub struct SchedulerOutput {
    /// Sequences scheduled for this iteration
    pub scheduled_groups: Vec<Arc<SequenceGroup>>,
    /// Blocks to swap from CPU to GPU
    pub blocks_to_swap_in: HashMap<usize, usize>,
    /// Blocks to swap from GPU to CPU
    pub blocks_to_swap_out: HashMap<usize, usize>,
    /// Blocks to copy (for copy-on-write)
    pub blocks_to_copy: HashMap<usize, Vec<usize>>,
    /// Sequences that couldn't be scheduled
    pub preempted_groups: Vec<Arc<SequenceGroup>>,
}

/// Configuration for the scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum number of sequences to run concurrently
    pub max_num_seqs: usize,
    /// Maximum number of batched tokens
    pub max_num_batched_tokens: usize,
    /// Enable swapping to CPU
    pub enable_swapping: bool,
    /// Scheduling policy
    pub policy: SchedulingPolicy,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 2048,
            enable_swapping: true,
            policy: SchedulingPolicy::FCFS,
        }
    }
}

/// Scheduling policy
#[derive(Debug, Clone)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    FCFS,
    /// Priority-based
    Priority,
    /// Shortest job first
    SJF,
}

/// Continuous batching scheduler with LoRA routing
pub struct HyprScheduler {
    /// Configuration
    config: SchedulerConfig,
    /// Block engine for memory management
    block_engine: Arc<BlockEngine>,
    /// Conversation router for LoRA selection
    conversation_router: Arc<ConversationRouter>,
    /// Waiting queue
    waiting: Arc<RwLock<VecDeque<Arc<SequenceGroup>>>>,
    /// Running sequences
    running: Arc<RwLock<VecDeque<Arc<SequenceGroup>>>>,
    /// Swapped out sequences
    swapped: Arc<RwLock<VecDeque<Arc<SequenceGroup>>>>,
    /// Next sequence ID
    next_seq_id: Arc<RwLock<SeqID>>,
    /// Next group ID
    next_group_id: Arc<RwLock<usize>>,
}

impl HyprScheduler {
    /// Create new scheduler
    pub async fn new(
        config: SchedulerConfig,
        block_engine: Arc<BlockEngine>,
        conversation_router: Arc<ConversationRouter>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            block_engine,
            conversation_router,
            waiting: Arc::new(RwLock::new(VecDeque::new())),
            running: Arc::new(RwLock::new(VecDeque::new())),
            swapped: Arc::new(RwLock::new(VecDeque::new())),
            next_seq_id: Arc::new(RwLock::new(0)),
            next_group_id: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Add a generation request to the scheduler
    pub async fn add_request(&self, request: GenerationRequest) -> Result<SeqID> {
        // Get next sequence ID
        let seq_id = {
            let mut id = self.next_seq_id.write().await;
            let current = *id;
            *id += 1;
            current
        };
        
        // Get next group ID
        let group_id = {
            let mut id = self.next_group_id.write().await;
            let current = *id;
            *id += 1;
            current
        };
        
        // Tokenize prompt (simplified - should use actual tokenizer)
        let prompt_tokens = self.tokenize(&request.prompt);
        
        // Create sequence
        let params = GenerationParams {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            top_k: request.top_k,
            stop_tokens: vec![2], // EOS
            stream: request.stream,
        };
        
        let mut sequence = Sequence::new(seq_id, prompt_tokens, params);
        
        // Route to LoRA adapter if specified
        if let Some(adapters) = &request.active_adapters {
            if !adapters.is_empty() {
                sequence.lora_adapter_id = Some(adapters[0].clone());
            }
        }
        
        // Create sequence group
        let sampling_params = SamplingParams::default();
        let group = SequenceGroup::new(group_id, vec![sequence], sampling_params);
        
        // Add to waiting queue
        {
            let mut waiting = self.waiting.write().await;
            waiting.push_back(Arc::new(group));
        }
        
        Ok(seq_id)
    }
    
    /// Schedule sequences for next iteration
    pub async fn schedule(&self) -> Result<SchedulerOutput> {
        let mut scheduled_groups = Vec::new();
        let mut blocks_to_swap_in = HashMap::new();
        let mut blocks_to_swap_out = HashMap::new();
        let blocks_to_copy = HashMap::new();
        let mut preempted_groups = Vec::new();
        
        // First, try to schedule running sequences
        let mut num_scheduled_tokens = 0;
        let mut num_scheduled_seqs = 0;
        
        {
            let running = self.running.read().await;
            for group in running.iter() {
                let group_tokens = self.get_group_tokens(group).await;
                
                if num_scheduled_tokens + group_tokens <= self.config.max_num_batched_tokens
                    && num_scheduled_seqs < self.config.max_num_seqs
                {
                    scheduled_groups.push(group.clone());
                    num_scheduled_tokens += group_tokens;
                    num_scheduled_seqs += group.sequences.len();
                }
            }
        }
        
        // Try to add waiting sequences
        {
            let mut waiting = self.waiting.write().await;
            let mut to_schedule = Vec::new();
            
            while let Some(group) = waiting.front() {
                let prompt_len = group.get_prompt_len().await;
                let num_blocks = (prompt_len + 15) / 16; // Assuming block size 16
                
                // Check if we can allocate
                let can_allocate = self.block_engine.can_allocate(group.group_id, num_blocks).await;
                
                match can_allocate {
                    AllocStatus::Ok => {
                        if num_scheduled_tokens + prompt_len <= self.config.max_num_batched_tokens
                            && num_scheduled_seqs < self.config.max_num_seqs
                        {
                            // Allocate blocks
                            self.block_engine.allocate(group.group_id, num_blocks).await?;
                            
                            let group = waiting.pop_front().unwrap();
                            to_schedule.push(group.clone());
                            scheduled_groups.push(group);
                            num_scheduled_tokens += prompt_len;
                            num_scheduled_seqs += 1;
                        } else {
                            break;
                        }
                    }
                    AllocStatus::Later => {
                        // Need to preempt or swap
                        if self.config.enable_swapping {
                            // Try swapping out some sequences
                            let swaps = self.perform_swapping(num_blocks).await?;
                            blocks_to_swap_out.extend(swaps);
                        } else {
                            break;
                        }
                    }
                    AllocStatus::Impossible => {
                        // Skip this sequence
                        preempted_groups.push(waiting.pop_front().unwrap());
                    }
                }
            }
            
            // Move scheduled sequences to running
            let mut running = self.running.write().await;
            for group in to_schedule {
                running.push_back(group);
            }
        }
        
        // Try to swap in previously swapped sequences
        if self.config.enable_swapping && !self.swapped.read().await.is_empty() {
            let swaps = self.try_swap_in().await?;
            blocks_to_swap_in.extend(swaps);
        }
        
        Ok(SchedulerOutput {
            scheduled_groups,
            blocks_to_swap_in,
            blocks_to_swap_out,
            blocks_to_copy,
            preempted_groups,
        })
    }
    
    /// Update sequence with generated tokens
    pub async fn update_sequence(
        &self,
        seq_id: SeqID,
        new_tokens: Vec<u32>,
        finished: bool,
    ) -> Result<()> {
        // Find sequence in running groups
        let mut running = self.running.write().await;
        
        for group in running.iter() {
            if let Some(seq) = group.sequences.get(&seq_id) {
                let mut seq_guard = seq.write().await;
                seq_guard.generated_tokens.extend(new_tokens);
                seq_guard.updated_at = Utc::now();
                
                if finished {
                    seq_guard.status = SequenceStatus::Finished;
                }
                
                break;
            }
        }
        
        // Remove finished groups
        running.retain(|group| {
            let all_finished = group.sequences.values().all(|seq| {
                let seq = seq.try_read();
                seq.map(|s| s.is_finished()).unwrap_or(false)
            });
            !all_finished
        });
        
        Ok(())
    }
    
    /// Get total tokens for a group
    async fn get_group_tokens(&self, group: &Arc<SequenceGroup>) -> usize {
        let mut total = 0;
        for seq in group.sequences.values() {
            let seq_guard = seq.read().await;
            total += seq_guard.get_len();
        }
        total
    }
    
    /// Perform swapping to free memory
    async fn perform_swapping(&self, needed_blocks: usize) -> Result<HashMap<usize, usize>> {
        // Simple strategy: swap out the oldest running sequence
        let mut swaps = HashMap::new();
        
        let mut running = self.running.write().await;
        let mut swapped = self.swapped.write().await;
        
        if let Some(group) = running.pop_front() {
            // Swap out this group's blocks
            let group_swaps = self.block_engine.swap_out(group.group_id, needed_blocks).await?;
            swaps.extend(group_swaps);
            
            // Move to swapped queue
            swapped.push_back(group);
        }
        
        Ok(swaps)
    }
    
    /// Try to swap in previously swapped sequences
    async fn try_swap_in(&self) -> Result<HashMap<usize, usize>> {
        let mut swaps = HashMap::new();
        
        let mut swapped = self.swapped.write().await;
        let mut running = self.running.write().await;
        
        if let Some(group) = swapped.front() {
            let num_blocks = 10; // Estimate
            let can_allocate = self.block_engine.can_allocate(group.group_id, num_blocks).await;
            
            if can_allocate == AllocStatus::Ok {
                let group = swapped.pop_front().unwrap();
                let group_swaps = self.block_engine.swap_in(group.group_id, num_blocks).await?;
                swaps.extend(group_swaps);
                running.push_back(group);
            }
        }
        
        Ok(swaps)
    }
    
    /// Simple tokenization (should use actual tokenizer)
    fn tokenize(&self, text: &str) -> Vec<u32> {
        text.chars().map(|c| c as u32).collect()
    }
    
    /// Get scheduler statistics
    pub async fn get_stats(&self) -> SchedulerStats {
        let waiting_count = self.waiting.read().await.len();
        let running_count = self.running.read().await.len();
        let swapped_count = self.swapped.read().await.len();
        
        SchedulerStats {
            num_waiting: waiting_count,
            num_running: running_count,
            num_swapped: swapped_count,
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub num_waiting: usize,
    pub num_running: usize,
    pub num_swapped: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_scheduler_creation() {
        let block_engine = Arc::new(
            BlockEngine::new(16, 100, 100).await.unwrap()
        );
        
        //
        // let router = Arc::new(ConversationRouter::new(temporal_layer).await);

        // Create a dummy router for testing
        let router = Arc::new(ConversationRouter::default());
        
        let scheduler = HyprScheduler::new(
            SchedulerConfig::default(),
            block_engine,
            router,
        ).await;
        
        assert!(scheduler.is_ok());
    }
}