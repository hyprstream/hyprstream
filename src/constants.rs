//! Shared constants for the hyprstream application
//! 
//! Centralizes size limits and other constants to maintain consistency
//! and avoid duplication across the codebase.

/// Maximum size limits for model components
pub mod limits {
    /// Maximum size for a single model file (500GB)
    /// Supports future terabyte-scale models on Optane drives
    pub const MAX_MODEL_FILE_SIZE: u64 = 500 * 1024 * 1024 * 1024;
    
    /// Maximum size for a single tensor (100GB)
    /// Large enough for the biggest weight matrices in current models
    pub const MAX_TENSOR_SIZE: u64 = 100 * 1024 * 1024 * 1024;
    
    /// Maximum size for model weight files (500GB)
    /// Same as MAX_MODEL_FILE_SIZE for consistency
    pub const MAX_WEIGHT_SIZE: u64 = MAX_MODEL_FILE_SIZE;
    
    /// Maximum size for SafeTensors header (100MB)
    /// Headers contain metadata and should never be this large
    pub const MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024;
    
    /// Maximum size for configuration files (100MB)
    /// Config.json files are typically <1MB but allow headroom
    pub const MAX_CONFIG_SIZE: u64 = 100 * 1024 * 1024;
    
    /// Maximum size for tokenizer files (1GB)
    /// Some tokenizers with large vocabularies can be substantial
    pub const MAX_TOKENIZER_SIZE: u64 = 1024 * 1024 * 1024;
    
    /// Minimum size for a valid SafeTensors file (8 bytes for header length)
    pub const MIN_SAFETENSORS_SIZE: usize = 8;
}

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

/// Path validation patterns
pub mod validation {
    /// Characters allowed in model filenames
    pub const ALLOWED_FILENAME_CHARS: &str = "alphanumeric-_.";
    
    /// Path traversal patterns to reject
    pub const PATH_TRAVERSAL_PATTERNS: &[&str] = &[
        "..",
        "./",
        "../",
        "..\\",
        ".\\",
    ];
    
    /// Characters that indicate potential path traversal
    pub const UNSAFE_PATH_CHARS: &[char] = &['\\', '\0'];
}