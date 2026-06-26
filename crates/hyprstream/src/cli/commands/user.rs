//! User management CLI commands

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum UserCommand {
    /// Create a user account (no key required at registration)
    Register {
        /// Username
        username: String,
    },
    /// Remove a registered user
    Remove {
        /// Username
        username: String,
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },
    /// List all registered users
    List,
    /// Manage public keys for a user
    Keys {
        #[command(subcommand)]
        command: UserKeysCommand,
    },
}

#[derive(Debug, Subcommand)]
pub enum UserKeysCommand {
    /// List keys registered for a user
    List {
        /// Username
        username: String,
    },
    /// Import a public key for a user
    Import {
        /// Username
        username: String,
        #[command(subcommand)]
        format: UserKeysImportFormat,
    },
    /// Remove a key by fingerprint (from 'keys list')
    Remove {
        /// Username
        username: String,
        /// Key fingerprint (SHA256:...)
        fingerprint: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum UserKeysImportFormat {
    /// SSH Ed25519 public key (ssh-ed25519 AAAA... line or .pub file)
    #[command(name = "ssh-ed25519")]
    SshEd25519 {
        /// Path to .pub file, or - to read from stdin
        #[arg(short = 'f', long = "file")]
        file: Option<String>,
        /// Inline key data (ssh-ed25519 AAAA...)
        data: Option<String>,
        /// Label for this key (e.g., "laptop", "yubikey")
        #[arg(long)]
        label: Option<String>,
    },
    /// Raw Ed25519 public key (32 bytes, standard base64)
    Raw {
        /// Path to file containing raw base64 key, or - to read from stdin
        #[arg(short = 'f', long = "file")]
        file: Option<String>,
        /// Inline base64 key data
        data: Option<String>,
        /// Label for this key
        #[arg(long)]
        label: Option<String>,
    },
}
