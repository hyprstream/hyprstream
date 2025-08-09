//! Authentication management for external services
//!
//! This module handles authentication for various ML model registries
//! and services like HuggingFace Hub.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use crate::config::HyprConfig;

/// HuggingFace authentication management
pub struct HfAuth {
    config: HyprConfig,
}

impl HfAuth {
    /// Create a new HuggingFace authentication manager
    pub fn new() -> Result<Self> {
        let config = HyprConfig::load().unwrap_or_default();
        Ok(Self { config })
    }

    /// Get the HuggingFace token file path
    fn hf_token_path(&self) -> PathBuf {
        self.config.config_dir().join("hf_token")
    }

    /// Set the HuggingFace authentication token
    pub async fn set_token(&self, token: &str) -> Result<()> {
        let token_path = self.hf_token_path();
        
        // Ensure config directory exists
        if let Err(e) = self.config.ensure_directories() {
            return Err(anyhow!("Failed to create config directory: {}", e));
        }
        
        tokio::fs::write(&token_path, token.trim()).await
            .map_err(|e| anyhow!("Failed to write HF token to {}: {}", token_path.display(), e))?;
        
        // Set restrictive permissions (readable only by owner)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = tokio::fs::metadata(&token_path).await?.permissions();
            perms.set_mode(0o600); // rw-------
            tokio::fs::set_permissions(&token_path, perms).await?;
        }

        println!("✅ HuggingFace token saved to: {}", token_path.display());
        Ok(())
    }

    /// Get the HuggingFace authentication token
    pub async fn get_token(&self) -> Result<Option<String>> {
        let token_path = self.hf_token_path();
        
        if !token_path.exists() {
            return Ok(None);
        }

        let token = tokio::fs::read_to_string(&token_path).await
            .map_err(|e| anyhow!("Failed to read HF token from {}: {}", token_path.display(), e))?;
        
        Ok(Some(token.trim().to_string()))
    }

    /// Check if authentication token is configured
    pub async fn is_authenticated(&self) -> bool {
        self.get_token().await.map_or(false, |token| token.is_some())
    }

    /// Remove the authentication token
    pub async fn logout(&self) -> Result<()> {
        let token_path = self.hf_token_path();
        
        if token_path.exists() {
            tokio::fs::remove_file(&token_path).await
                .map_err(|e| anyhow!("Failed to remove HF token: {}", e))?;
            println!("✅ Logged out from HuggingFace");
        } else {
            println!("ℹ️  No HuggingFace token found");
        }
        
        Ok(())
    }
}

impl Default for HfAuth {
    fn default() -> Self {
        Self::new().expect("Failed to initialize HF auth")
    }
}