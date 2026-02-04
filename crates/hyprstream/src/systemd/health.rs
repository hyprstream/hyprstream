//! Health-based watchdog integration
//!
//! Provides a health checker that only pings the systemd watchdog when
//! the service is healthy. If health checks fail, the watchdog won't be
//! pinged and systemd will eventually restart the service.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │              HealthChecker                  │
//! │  ┌─────────────────────────────────────┐   │
//! │  │  check_zmq_connectivity()           │   │
//! │  │  check_http_endpoint()              │   │
//! │  │  check_memory_pressure()            │   │
//! │  └─────────────────────────────────────┘   │
//! │                    │                        │
//! │                    ▼                        │
//! │  ┌─────────────────────────────────────┐   │
//! │  │  is_healthy() -> bool               │   │
//! │  │  (all checks must pass)             │   │
//! │  └─────────────────────────────────────┘   │
//! └─────────────────────────────────────────────┘
//!                      │
//!                      ▼
//! ┌─────────────────────────────────────────────┐
//! │           run_watchdog()                    │
//! │  if is_healthy() { SystemdNotifier::watchdog() }
//! └─────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_core::systemd::health::{HealthChecker, run_watchdog};
//!
//! let checker = HealthChecker::new();
//!
//! // Spawn watchdog task (runs until shutdown)
//! tokio::spawn(run_watchdog(checker));
//! ```

use super::SystemdNotifier;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// Timeout for individual health checks
    pub check_timeout: Duration,

    /// HTTP endpoint to check (if any)
    pub http_endpoint: Option<String>,

    /// Memory pressure threshold (0.0 - 1.0)
    /// If memory usage exceeds this, service is unhealthy
    pub memory_threshold: f64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_timeout: Duration::from_secs(5),
            http_endpoint: None,
            memory_threshold: 0.95, // 95% memory usage threshold
        }
    }
}

/// Health checker for watchdog integration
///
/// Performs health checks and reports overall service health.
/// Only healthy services should ping the watchdog.
pub struct HealthChecker {
    config: HealthConfig,
}

impl HealthChecker {
    /// Create a new health checker with default configuration
    pub fn new() -> Self {
        Self {
            config: HealthConfig::default(),
        }
    }

    /// Create a new health checker with custom configuration
    pub fn with_config(config: HealthConfig) -> Self {
        Self { config }
    }

    /// Set the HTTP endpoint to check
    pub fn with_http_endpoint(mut self, endpoint: String) -> Self {
        self.config.http_endpoint = Some(endpoint);
        self
    }

    /// Aggregate health check - returns true only if ALL checks pass
    ///
    /// Currently checks:
    /// 1. HTTP endpoint (if configured)
    /// 2. Memory pressure
    ///
    /// # Returns
    ///
    /// `true` if all health checks pass, `false` otherwise.
    pub async fn is_healthy(&self) -> bool {
        // 1. Check HTTP endpoint if configured
        let http_ok = if self.config.http_endpoint.is_some() {
            self.check_http_endpoint().await
        } else {
            true
        };

        // 2. Check memory pressure
        let memory_ok = self.check_memory_pressure();

        let healthy = http_ok && memory_ok;

        if !healthy {
            warn!(
                "Health check failed: http={}, memory={}",
                http_ok, memory_ok
            );
        } else {
            debug!("Health check passed");
        }

        healthy
    }

    /// Check HTTP endpoint health
    ///
    /// Attempts a simple TCP connect to the configured HTTP endpoint.
    async fn check_http_endpoint(&self) -> bool {
        let endpoint = match &self.config.http_endpoint {
            Some(e) => e,
            None => return true,
        };

        let result = timeout(self.config.check_timeout, async {
            // Try to connect to the endpoint
            match tokio::net::TcpStream::connect(endpoint).await {
                Ok(_) => {
                    debug!("HTTP endpoint {} is reachable", endpoint);
                    true
                }
                Err(e) => {
                    warn!("HTTP endpoint {} unreachable: {}", endpoint, e);
                    false
                }
            }
        })
        .await;

        match result {
            Ok(healthy) => healthy,
            Err(_) => {
                warn!("HTTP health check timed out for {}", endpoint);
                false
            }
        }
    }

    /// Check memory pressure
    ///
    /// Reads /proc/meminfo to check if memory usage is within acceptable limits.
    fn check_memory_pressure(&self) -> bool {
        #[cfg(target_os = "linux")]
        {
            // Read /proc/meminfo
            let meminfo = match std::fs::read_to_string("/proc/meminfo") {
                Ok(content) => content,
                Err(e) => {
                    warn!("Failed to read /proc/meminfo: {}", e);
                    return true; // Don't fail health check if we can't read meminfo
                }
            };

            let mut mem_total: u64 = 0;
            let mut mem_available: u64 = 0;

            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    mem_total = parse_meminfo_value(line);
                } else if line.starts_with("MemAvailable:") {
                    mem_available = parse_meminfo_value(line);
                }
            }

            if mem_total == 0 {
                return true; // Can't determine, assume healthy
            }

            let usage_ratio = 1.0 - (mem_available as f64 / mem_total as f64);

            if usage_ratio > self.config.memory_threshold {
                warn!(
                    "Memory pressure high: {:.1}% used (threshold: {:.1}%)",
                    usage_ratio * 100.0,
                    self.config.memory_threshold * 100.0
                );
                false
            } else {
                debug!(
                    "Memory usage: {:.1}% (threshold: {:.1}%)",
                    usage_ratio * 100.0,
                    self.config.memory_threshold * 100.0
                );
                true
            }
        }

        #[cfg(not(target_os = "linux"))]
        {
            // Can't check memory pressure on non-Linux
            true
        }
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a value from /proc/meminfo (e.g., "MemTotal:       16384000 kB")
#[cfg(target_os = "linux")]
fn parse_meminfo_value(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Run the watchdog task
///
/// Periodically checks health and pings the watchdog only when healthy.
/// If health checks fail, the watchdog won't be pinged and systemd will
/// eventually restart the service after `WatchdogSec` timeout.
///
/// # Arguments
///
/// * `checker` - Health checker instance
///
/// # Example
///
/// ```ignore
/// let checker = HealthChecker::new()
///     .with_http_endpoint("127.0.0.1:3000".to_owned());
///
/// // Spawn as background task
/// tokio::spawn(run_watchdog(checker));
/// ```
pub async fn run_watchdog(checker: HealthChecker) {
    let Some(interval) = SystemdNotifier::watchdog_interval() else {
        debug!("Watchdog not enabled (WATCHDOG_USEC not set)");
        return;
    };

    info!("Watchdog enabled, pinging every {:?}", interval);

    loop {
        tokio::time::sleep(interval).await;

        if checker.is_healthy().await {
            SystemdNotifier::watchdog();
        } else {
            warn!("Health check failed, skipping watchdog ping");
            // Systemd will restart us after WatchdogSec timeout
        }
    }
}

/// Run watchdog with a shutdown signal
///
/// Same as `run_watchdog` but can be stopped via a shutdown receiver.
///
/// # Arguments
///
/// * `checker` - Health checker instance
/// * `shutdown` - Receiver that signals when to stop
pub async fn run_watchdog_with_shutdown(
    checker: HealthChecker,
    mut shutdown: tokio::sync::broadcast::Receiver<()>,
) {
    let Some(interval) = SystemdNotifier::watchdog_interval() else {
        debug!("Watchdog not enabled (WATCHDOG_USEC not set)");
        return;
    };

    info!("Watchdog enabled, pinging every {:?}", interval);

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {
                if checker.is_healthy().await {
                    SystemdNotifier::watchdog();
                } else {
                    warn!("Health check failed, skipping watchdog ping");
                }
            }
            _ = shutdown.recv() => {
                info!("Watchdog task shutting down");
                break;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;

    #[test]
    fn test_health_config_default() {
        let config = HealthConfig::default();
        assert_eq!(config.check_timeout, Duration::from_secs(5));
        assert!(config.http_endpoint.is_none());
        assert!((config.memory_threshold - 0.95).abs() < f64::EPSILON);
    }

    #[test]
    fn test_health_checker_default() {
        let checker = HealthChecker::new();
        assert!(checker.config.http_endpoint.is_none());
    }

    #[test]
    fn test_health_checker_with_http() {
        let checker = HealthChecker::new().with_http_endpoint("127.0.0.1:3000".to_owned());
        assert_eq!(
            checker.config.http_endpoint,
            Some("127.0.0.1:3000".to_owned())
        );
    }

    #[tokio::test]
    async fn test_is_healthy_no_http() {
        // Without HTTP endpoint, should check only memory
        let checker = HealthChecker::new();
        let healthy = checker.is_healthy().await;
        // Should pass unless system is under severe memory pressure
        println!("Health check result: {}", healthy);
    }

    #[test]
    fn test_memory_check() {
        let checker = HealthChecker::new();
        let result = checker.check_memory_pressure();
        // Should return true unless system is under pressure
        println!("Memory check result: {}", result);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_parse_meminfo_value() {
        assert_eq!(parse_meminfo_value("MemTotal:       16384000 kB"), 16384000);
        assert_eq!(parse_meminfo_value("MemAvailable:   8192000 kB"), 8192000);
        assert_eq!(parse_meminfo_value("Invalid line"), 0);
    }
}
