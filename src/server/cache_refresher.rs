//! Background task for periodic cache refresh

use std::sync::Arc;
use std::time::Duration;
use tokio::time;
use tracing::{info, warn, error};
use super::model_cache::ModelCache;

/// Background task that periodically refreshes the model cache
pub struct CacheRefresher {
    model_cache: Arc<ModelCache>,
    refresh_interval: Duration,
}

impl CacheRefresher {
    /// Create a new cache refresher
    pub fn new(model_cache: Arc<ModelCache>, refresh_interval_secs: u64) -> Self {
        Self {
            model_cache,
            refresh_interval: Duration::from_secs(refresh_interval_secs),
        }
    }
    
    /// Start the background refresh task
    pub fn start(self) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            info!("Starting cache refresher with interval: {:?}", self.refresh_interval);
            
            // Initial delay to let server fully start
            time::sleep(Duration::from_secs(60)).await;
            
            let mut interval = time::interval(self.refresh_interval);
            interval.set_missed_tick_behavior(time::MissedTickBehavior::Skip);
            
            loop {
                interval.tick().await;
                
                info!("Running periodic cache refresh...");
                match self.model_cache.refresh_name_cache().await {
                    Ok(_) => {
                        info!("Cache refresh completed successfully");
                    }
                    Err(e) => {
                        error!("Cache refresh failed: {}", e);
                    }
                }
            }
        })
    }
}

/// Configuration for cache refresher
#[derive(Debug, Clone)]
pub struct CacheRefresherConfig {
    /// Enable periodic cache refresh
    pub enabled: bool,
    /// Refresh interval in seconds (default: 5 minutes)
    pub interval_secs: u64,
}

impl Default for CacheRefresherConfig {
    fn default() -> Self {
        Self {
            enabled: false,  // Disabled by default
            interval_secs: 300,  // 5 minutes
        }
    }
}

impl CacheRefresherConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();
        
        if let Ok(enabled) = std::env::var("HYPRSTREAM_CACHE_REFRESH_ENABLED") {
            config.enabled = enabled.to_lowercase() == "true";
        }
        
        if let Ok(interval) = std::env::var("HYPRSTREAM_CACHE_REFRESH_INTERVAL") {
            if let Ok(secs) = interval.parse::<u64>() {
                if secs >= 30 {  // Minimum 30 seconds
                    config.interval_secs = secs;
                } else {
                    warn!("Cache refresh interval too short ({}s), using minimum 30s", secs);
                    config.interval_secs = 30;
                }
            }
        }
        
        if config.enabled {
            info!("Cache refresher enabled with {}s interval", config.interval_secs);
        }
        
        config
    }
}