//! User management CLI commands

use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum UserCommand {
    /// Create a user AND enroll a signing key in one step
    ///
    /// Generates (default) or adopts a signing key, installs it as the client's
    /// actual signing key, registers the user record, binds the derived public
    /// key, and grants a role — collapsing the multi-step enrollment that,
    /// done wrong, silently authenticates as `anonymous` (#439).
    Create {
        /// Username to enroll (defaults to the OS user, matching the CLI's `sub`)
        #[arg(default_value_t = os_user_string())]
        username: String,
        /// Role to grant (default `operator`; use `admin` for full access)
        #[arg(long, default_value = "operator")]
        role: String,
        /// Generate a fresh Ed25519 signing key (default)
        #[arg(long, conflicts_with_all = ["key", "ssh"])]
        generate: bool,
        /// Adopt a raw 32-byte Ed25519 seed file as the signing key
        #[arg(long, value_name = "PATH", conflicts_with_all = ["generate", "ssh"])]
        key: Option<String>,
        /// Adopt an OpenSSH Ed25519 private key as the signing key
        /// (passphrase-prompted if encrypted)
        #[arg(long, value_name = "PATH", conflicts_with_all = ["generate", "key"])]
        ssh: Option<String>,
        /// Skip granting the role even if services are running
        #[arg(long)]
        no_role: bool,
    },
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

/// Resolve the OS username exactly as the CLI presents it in `sub`
/// (`$USER` → `$LOGNAME` → `"anonymous"`), so `user create`'s default matches
/// the subject the CLI authenticates as. Kept in lockstep with
/// `sign_challenge.rs::load_user_signing_key`.
fn os_user_string() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "anonymous".to_owned())
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
