//! Generation quality metrics for self-supervised training
//!
//! This module provides O(1) online metrics accumulation during text generation,
//! capturing quality signals from logits that can be used for reinforcement learning.
//!
//! Key metrics:
//! - Perplexity: Model confidence (lower = more confident)
//! - Entropy: Decision uncertainty (lower = more certain)
//! - Entropy variance: Consistency of confidence (lower = more consistent)
//! - Repetition ratio: N-gram repetition detection (lower = less repetitive)

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Size of ring buffer for n-gram detection
const NGRAM_BUFFER_SIZE: usize = 32;

/// N-gram size for repetition detection (trigrams)
const NGRAM_SIZE: usize = 3;

/// Number of hash functions for bloom filter (k=3 gives ~2% FP rate at 50% fill)
const BLOOM_K: usize = 3;

/// Bloom filter size in bits (2048 bits = 256 bytes)
const BLOOM_BITS: usize = 2048;

/// O(1) online metrics accumulator using Welford's algorithm
///
/// This accumulator maintains running statistics without storing per-token data,
/// using approximately 360 bytes of memory regardless of sequence length.
///
/// # Example
/// ```ignore
/// let mut acc = GenerationMetricsAccumulator::new();
/// for token in generated_tokens {
///     acc.add_token(log_prob, entropy, token_id);
/// }
/// let metrics = acc.finalize();
/// println!("Perplexity: {}", metrics.perplexity);
/// ```
#[derive(Debug, Clone)]
pub struct GenerationMetricsAccumulator {
    // Welford's algorithm for log_prob statistics
    log_prob_mean: f64,
    log_prob_m2: f64,

    // Welford's algorithm for entropy statistics
    entropy_mean: f64,
    entropy_m2: f64,

    // Token count
    count: u64,

    // Ring buffer for n-gram detection
    recent_tokens: VecDeque<u32>,

    // N-gram repetition counter
    ngram_repeats: u32,

    // Bloom filter for n-gram tracking (256 bytes = 2048 bits)
    // Uses k=3 hash functions for ~2% false positive rate at 50% fill
    bloom_filter: [u64; 32],

    // Count of unique n-grams inserted (approximate, counts insertions not unique)
    ngram_insert_count: u32,
}

impl Default for GenerationMetricsAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl GenerationMetricsAccumulator {
    /// Create a new metrics accumulator
    pub fn new() -> Self {
        Self {
            log_prob_mean: 0.0,
            log_prob_m2: 0.0,
            entropy_mean: 0.0,
            entropy_m2: 0.0,
            count: 0,
            recent_tokens: VecDeque::with_capacity(NGRAM_BUFFER_SIZE),
            ngram_repeats: 0,
            bloom_filter: [0u64; 32],
            ngram_insert_count: 0,
        }
    }

    /// Add a token's metrics to the accumulator
    ///
    /// # Arguments
    /// * `log_prob` - Log probability of the sampled token (negative value)
    /// * `entropy` - Entropy of the probability distribution (in nats)
    /// * `token_id` - The sampled token ID (for n-gram tracking)
    ///
    /// # Complexity
    /// O(1) time and space per token
    pub fn add_token(&mut self, log_prob: f32, entropy: f32, token_id: u32) {
        self.count += 1;
        let n = self.count as f64;

        // Welford's online algorithm for log_prob
        let log_prob_f64 = log_prob as f64;
        let delta_lp = log_prob_f64 - self.log_prob_mean;
        self.log_prob_mean += delta_lp / n;
        let delta2_lp = log_prob_f64 - self.log_prob_mean;
        self.log_prob_m2 += delta_lp * delta2_lp;

        // Welford's online algorithm for entropy
        let entropy_f64 = entropy as f64;
        let delta_e = entropy_f64 - self.entropy_mean;
        self.entropy_mean += delta_e / n;
        let delta2_e = entropy_f64 - self.entropy_mean;
        self.entropy_m2 += delta_e * delta2_e;

        // N-gram tracking
        self.recent_tokens.push_back(token_id);
        if self.recent_tokens.len() > NGRAM_BUFFER_SIZE {
            self.recent_tokens.pop_front();
        }

        // Check for n-gram repetition when we have enough tokens
        if self.recent_tokens.len() >= NGRAM_SIZE {
            let ngram_hash = self.compute_current_ngram_hash();

            // Check if this n-gram might already be in the bloom filter
            if self.bloom_contains(ngram_hash) {
                // Likely repeat (with ~2% false positive rate at 50% fill)
                self.ngram_repeats += 1;
            } else {
                // New n-gram, add to bloom filter
                self.bloom_insert(ngram_hash);
                self.ngram_insert_count += 1;
            }
        }
    }

    /// Insert a hash into the bloom filter
    ///
    /// Uses k=3 hash functions derived from the base hash via mixing.
    fn bloom_insert(&mut self, hash: u64) {
        for i in 0..BLOOM_K {
            let bit_index = self.bloom_hash(hash, i);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            self.bloom_filter[word_index] |= 1 << bit_offset;
        }
    }

    /// Check if a hash might be in the bloom filter
    ///
    /// Returns true if the element might be present (with some false positive rate),
    /// false if the element is definitely not present.
    fn bloom_contains(&self, hash: u64) -> bool {
        for i in 0..BLOOM_K {
            let bit_index = self.bloom_hash(hash, i);
            let word_index = bit_index / 64;
            let bit_offset = bit_index % 64;
            if (self.bloom_filter[word_index] & (1 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute the i-th hash function for the bloom filter
    ///
    /// Uses the double hashing technique: h_i(x) = h1(x) + i * h2(x)
    /// where h1 and h2 are derived from the input hash.
    fn bloom_hash(&self, hash: u64, i: usize) -> usize {
        // Split hash into two 32-bit values for double hashing
        let h1 = (hash & 0xFFFFFFFF) as usize;
        let h2 = ((hash >> 32) | 1) as usize; // Ensure h2 is odd for better distribution

        // Double hashing: h_i(x) = (h1 + i * h2) mod m
        (h1.wrapping_add(i.wrapping_mul(h2))) % BLOOM_BITS
    }

    /// Compute hash of the current n-gram (last NGRAM_SIZE tokens)
    fn compute_current_ngram_hash(&self) -> u64 {
        let len = self.recent_tokens.len();
        if len < NGRAM_SIZE {
            return 0;
        }

        // Simple hash combining last NGRAM_SIZE tokens
        let mut hash: u64 = 0;
        for i in (len - NGRAM_SIZE)..len {
            let token = self.recent_tokens[i] as u64;
            hash = hash.wrapping_mul(31).wrapping_add(token);
        }
        // Ensure non-zero for bloom filter logic
        if hash == 0 {
            hash = 1;
        }
        hash
    }

    /// Finalize and return aggregated quality metrics
    ///
    /// # Returns
    /// `GenerationQualityMetrics` containing perplexity, entropy stats, and repetition ratio
    pub fn finalize(&self) -> GenerationQualityMetrics {
        if self.count == 0 {
            return GenerationQualityMetrics::default();
        }

        // Perplexity = exp(-avg_log_prob)
        let perplexity = (-self.log_prob_mean).exp() as f32;

        // Average entropy
        let avg_entropy = self.entropy_mean as f32;

        // Entropy variance (Welford's finalization)
        let entropy_variance = if self.count > 1 {
            (self.entropy_m2 / (self.count - 1) as f64) as f32
        } else {
            0.0
        };

        // Repetition ratio: ngram_repeats / total_possible_ngrams
        let total_ngrams = if self.count >= NGRAM_SIZE as u64 {
            self.count - NGRAM_SIZE as u64 + 1
        } else {
            0
        };
        let repetition_ratio = if total_ngrams > 0 {
            self.ngram_repeats as f32 / total_ngrams as f32
        } else {
            0.0
        };

        GenerationQualityMetrics {
            perplexity,
            avg_entropy,
            entropy_variance,
            repetition_ratio,
            token_count: self.count as u32,
        }
    }

    /// Get current token count
    pub fn token_count(&self) -> u64 {
        self.count
    }

    /// Check if any tokens have been added
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Aggregated quality metrics from a generation (~24 bytes)
///
/// These metrics can be used for:
/// - Quality scoring in self-supervised RL
/// - Monitoring generation quality
/// - Detecting degraded model performance
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GenerationQualityMetrics {
    /// Perplexity: exp(-avg_log_prob)
    /// Lower = model is more confident in its generations
    /// Typical range: 1.0 (perfect) to 100+ (very uncertain)
    pub perplexity: f32,

    /// Average entropy of probability distributions (in nats)
    /// Lower = more decisive token selection
    /// Typical range: 0.0 (deterministic) to ~10.0 (very uncertain)
    pub avg_entropy: f32,

    /// Variance of entropy across tokens
    /// Lower = more consistent confidence throughout generation
    pub entropy_variance: f32,

    /// Ratio of repeated n-grams to total n-grams
    /// Lower = less repetitive output
    /// Range: 0.0 (no repeats) to 1.0 (all repeats)
    pub repetition_ratio: f32,

    /// Number of tokens generated
    pub token_count: u32,
}

impl GenerationQualityMetrics {
    /// Check if metrics indicate potentially problematic generation
    pub fn is_concerning(&self) -> bool {
        self.perplexity > 50.0 || self.repetition_ratio > 0.3 || self.entropy_variance > 2.0
    }
}

/// Per-session metrics storage for tracking quality across conversation turns
#[derive(Debug, Clone, Default)]
pub struct SessionMetrics {
    /// Quality metrics for each turn in the session
    turn_metrics: Vec<GenerationQualityMetrics>,

    /// Running sum of perplexity for session average
    session_perplexity_sum: f64,

    /// Running sum of tokens for weighted averages
    session_token_sum: u64,
}

impl SessionMetrics {
    /// Create a new session metrics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Add metrics from a completed generation turn
    pub fn add_turn(&mut self, metrics: GenerationQualityMetrics) {
        self.session_perplexity_sum += metrics.perplexity as f64 * metrics.token_count as f64;
        self.session_token_sum += metrics.token_count as u64;
        self.turn_metrics.push(metrics);
    }

    /// Get the number of turns in this session
    pub fn turn_count(&self) -> usize {
        self.turn_metrics.len()
    }

    /// Get metrics for a specific turn
    pub fn get_turn(&self, index: usize) -> Option<&GenerationQualityMetrics> {
        self.turn_metrics.get(index)
    }

    /// Get all turn metrics
    pub fn turns(&self) -> &[GenerationQualityMetrics] {
        &self.turn_metrics
    }

    /// Compute session-level average perplexity (token-weighted)
    pub fn avg_perplexity(&self) -> f32 {
        if self.session_token_sum == 0 {
            return 0.0;
        }
        (self.session_perplexity_sum / self.session_token_sum as f64) as f32
    }

    /// Get total tokens generated in this session
    pub fn total_tokens(&self) -> u64 {
        self.session_token_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_empty() {
        let acc = GenerationMetricsAccumulator::new();
        assert!(acc.is_empty());
        let metrics = acc.finalize();
        assert_eq!(metrics.token_count, 0);
    }

    #[test]
    fn test_accumulator_single_token() {
        let mut acc = GenerationMetricsAccumulator::new();
        acc.add_token(-2.0, 1.5, 100);

        let metrics = acc.finalize();
        assert_eq!(metrics.token_count, 1);
        // perplexity = exp(-(-2.0)) = exp(2.0) ≈ 7.39
        assert!((metrics.perplexity - 7.389).abs() < 0.01);
        assert!((metrics.avg_entropy - 1.5).abs() < 0.01);
        assert_eq!(metrics.entropy_variance, 0.0); // No variance with 1 sample
    }

    #[test]
    fn test_accumulator_multiple_tokens() {
        let mut acc = GenerationMetricsAccumulator::new();

        // Add tokens with consistent log_prob and entropy
        for i in 0..10 {
            acc.add_token(-1.0, 2.0, i);
        }

        let metrics = acc.finalize();
        assert_eq!(metrics.token_count, 10);
        // perplexity = exp(-(-1.0)) = e ≈ 2.718
        assert!((metrics.perplexity - std::f32::consts::E).abs() < 0.01);
        assert!((metrics.avg_entropy - 2.0).abs() < 0.01);
        // Variance should be near 0 with identical values
        assert!(metrics.entropy_variance < 0.01);
    }

    #[test]
    fn test_ngram_repetition() {
        let mut acc = GenerationMetricsAccumulator::new();

        // Generate pattern: 1,2,3,1,2,3,1,2,3 (repeated trigram)
        for _ in 0..3 {
            acc.add_token(-1.0, 1.0, 1);
            acc.add_token(-1.0, 1.0, 2);
            acc.add_token(-1.0, 1.0, 3);
        }

        let metrics = acc.finalize();
        // Should detect repetition
        assert!(metrics.repetition_ratio > 0.0);
    }

    #[test]
    fn test_session_metrics() {
        let mut session = SessionMetrics::new();

        // Add some turns
        session.add_turn(GenerationQualityMetrics {
            perplexity: 5.0,
            avg_entropy: 2.0,
            entropy_variance: 0.5,
            repetition_ratio: 0.1,
            token_count: 50,
        });

        session.add_turn(GenerationQualityMetrics {
            perplexity: 3.0,
            avg_entropy: 1.5,
            entropy_variance: 0.3,
            repetition_ratio: 0.05,
            token_count: 100,
        });

        assert_eq!(session.turn_count(), 2);
        assert_eq!(session.total_tokens(), 150);

        // Weighted average: (5*50 + 3*100) / 150 = 550/150 ≈ 3.67
        assert!((session.avg_perplexity() - 3.67).abs() < 0.1);
    }
}
