//! Shared constants for the hyprstream application

/// File permissions for security
pub mod permissions {
    /// Standard file permissions for downloaded models (rw-r--r--)
    /// Owner can read/write, others can only read
    #[cfg(unix)]
    pub const MODEL_FILE_PERMISSIONS: u32 = 0o644;
    
    /// Restrictive permissions for auth tokens (rw-------)
    /// Only owner can read/write
    #[cfg(unix)]
    pub const AUTH_TOKEN_PERMISSIONS: u32 = 0o600;
}

