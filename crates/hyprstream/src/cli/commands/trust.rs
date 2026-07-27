use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Subcommand)]
pub enum TrustCommand {
    /// Generate an Ed25519 + ML-DSA-65 deployment authority.
    MintDeploymentCa(MintDeploymentCaArgs),
    /// Create a root-authorized, registry-only online signer.
    DelegateRegistrySigner(DelegateRegistrySignerArgs),
    /// Mint the one-hour registry deployment credential.
    MintRegistryJwt(MintRegistryJwtArgs),
    /// Verify deployment artifacts through the production verifier.
    VerifyDeployment(VerifyDeploymentArgs),
    /// Add or replace an authority key through the signed rotation log.
    RotateAuthority(RotateAuthorityArgs),
}

#[derive(Debug, Args)]
pub struct MintDeploymentCaArgs {
    /// Raw 1984-byte public root output (32-byte Ed25519, then 1952-byte ML-DSA-65).
    #[arg(long, default_value = "deployment-ca.hybrid")]
    pub public_ca: PathBuf,

    /// Age-encrypted authority bundle. This file must remain operator-held.
    #[arg(long, default_value = "deployment-ca.age")]
    pub authority_key: PathBuf,

    /// Signed authority rotation log, anchored by the raw public root.
    #[arg(long, default_value = "deployment-authority.log.json")]
    pub authority_log: PathBuf,

    /// Independently provisioned expected authority-log head.
    #[arg(long, default_value = "deployment-authority.head.json")]
    pub authority_checkpoint: PathBuf,

    /// Native age or age-plugin recipient. Repeat to add recovery recipients.
    #[arg(long = "recipient", value_name = "AGE_RECIPIENT")]
    pub recipients: Vec<String>,

    /// age-plugin-yubikey recipient (age1yubikey1...). Repeatable.
    #[arg(long = "yubikey", value_name = "AGE_YUBIKEY_RECIPIENT")]
    pub yubikey_recipients: Vec<String>,

    /// Cloud/PQ-HSM age-plugin recipient. Repeatable.
    #[arg(long = "kms-plugin", value_name = "AGE_PLUGIN_RECIPIENT")]
    pub kms_plugin_recipients: Vec<String>,

    /// Place the Ed25519 leg in this YubiKey PIV slot (firmware 5.7.4+).
    #[arg(long, value_name = "SLOT")]
    pub piv_slot: Option<String>,

    /// Replace existing output files.
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct DelegateRegistrySignerArgs {
    /// Raw 1984-byte public deployment root.
    #[arg(long, default_value = "deployment-ca.hybrid")]
    pub public_ca: PathBuf,

    /// Signed authority rotation log.
    #[arg(long, default_value = "deployment-authority.log.json")]
    pub authority_log: PathBuf,

    /// Independently trusted expected authority-log head.
    #[arg(long, default_value = "deployment-authority.head.json")]
    pub authority_checkpoint: PathBuf,

    /// Age-encrypted root/current authority bundle.
    #[arg(long, default_value = "deployment-ca.age")]
    pub authority_key: PathBuf,

    /// Identity used to unlock the root/current authority. Repeatable.
    #[arg(long = "identity", value_name = "AGE_IDENTITY_FILE")]
    pub identities: Vec<PathBuf>,

    /// YubiKey identity used to unlock the root/current authority. Repeatable.
    #[arg(long = "yubikey-identity", value_name = "AGE_YUBIKEY_IDENTITY_FILE")]
    pub yubikey_identities: Vec<PathBuf>,

    /// Break-glass: use the age-wrapped recovery copy of a PIV Ed25519 key.
    #[arg(long)]
    pub software_recovery: bool,

    /// Recipient for the separately encrypted online signer. Repeatable.
    #[arg(long = "signer-recipient", value_name = "AGE_RECIPIENT")]
    pub signer_recipients: Vec<String>,

    /// Encrypted online signer output.
    #[arg(long, default_value = "registry-delegated-signer.age")]
    pub delegated_key: PathBuf,

    /// Root-authorized delegation output.
    #[arg(long, default_value = "registry-signer.delegation.json")]
    pub delegation: PathBuf,

    /// Delegation lifetime in seconds (default 30 days, maximum one year).
    #[arg(
        long,
        default_value_t = 2_592_000,
        value_parser = clap::value_parser!(u64).range(3600..=31_536_000)
    )]
    pub delegation_ttl_seconds: u64,

    /// Replace existing output files.
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct MintRegistryJwtArgs {
    /// Raw 1984-byte public deployment root.
    #[arg(long, default_value = "deployment-ca.hybrid")]
    pub public_ca: PathBuf,

    /// Age-encrypted root authority bundle (used only with --root).
    #[arg(long, default_value = "deployment-ca.age")]
    pub authority_key: PathBuf,

    /// Age identity file. Repeatable for native or plugin identities.
    #[arg(long = "identity", value_name = "AGE_IDENTITY_FILE")]
    pub identities: Vec<PathBuf>,

    /// age-plugin-yubikey identity file. Repeatable; decryption requires the token.
    #[arg(long = "yubikey-identity", value_name = "AGE_YUBIKEY_IDENTITY_FILE")]
    pub yubikey_identities: Vec<PathBuf>,

    /// Break-glass: use the age-wrapped recovery copy of a PIV Ed25519 key.
    #[arg(long)]
    pub software_recovery: bool,

    /// Common path: decrypt this scoped online signer.
    #[arg(
        long,
        value_name = "DELEGATED_KEY.age",
        conflicts_with = "root",
        required_unless_present = "root"
    )]
    pub via_delegated_signer: Option<PathBuf>,

    /// Delegation authorizing --via-delegated-signer.
    #[arg(
        long,
        value_name = "DELEGATION.json",
        requires = "via_delegated_signer"
    )]
    pub delegation: Option<PathBuf>,

    /// Installed/current public authority log. Required for every credential.
    #[arg(long, default_value = "deployment-authority.log.json")]
    pub authority_log: PathBuf,

    /// Independently trusted expected authority-log head.
    #[arg(long, default_value = "deployment-authority.head.json")]
    pub authority_checkpoint: PathBuf,

    /// Rare/bootstrap path: sign directly with the deployment authority.
    #[arg(long, conflicts_with = "via_delegated_signer")]
    pub root: bool,

    /// Raw 32-byte Ed25519 registry-service public key for the cnf claim.
    #[arg(long, value_name = "PATH")]
    pub registry_public_key: PathBuf,

    /// Credential lifetime in seconds; the profile caps this at one hour.
    #[arg(long, default_value_t = 3600, value_parser = clap::value_parser!(u32).range(1..=3600))]
    pub ttl_seconds: u32,

    /// Registry deployment credential output.
    #[arg(long, default_value = "registry-service.jwt")]
    pub jwt: PathBuf,

    /// Out-of-band cloud-secret publisher manifest (never pass its values to Terraform).
    #[arg(long, default_value = "deployment-trust.contract.json")]
    pub contract: PathBuf,

    /// Replace existing output files.
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct RotateAuthorityArgs {
    /// Raw 1984-byte original public root (does not change during log rotation).
    #[arg(long, default_value = "deployment-ca.hybrid")]
    pub public_ca: PathBuf,

    /// Current signed authority rotation log.
    #[arg(long, default_value = "deployment-authority.log.json")]
    pub authority_log: PathBuf,

    /// Current independently trusted expected authority-log head.
    #[arg(long, default_value = "deployment-authority.head.json")]
    pub authority_checkpoint: PathBuf,

    /// Age-encrypted currently active authority used to sign the rotation.
    #[arg(long, default_value = "deployment-ca.age")]
    pub authority_key: PathBuf,

    /// Identity used to unlock the current authority. Repeatable.
    #[arg(long = "identity", value_name = "AGE_IDENTITY_FILE")]
    pub identities: Vec<PathBuf>,

    /// YubiKey identity used to unlock the current authority. Repeatable.
    #[arg(long = "yubikey-identity", value_name = "AGE_YUBIKEY_IDENTITY_FILE")]
    pub yubikey_identities: Vec<PathBuf>,

    /// Break-glass: use the age-wrapped recovery copy of a PIV Ed25519 key.
    #[arg(long)]
    pub software_recovery: bool,

    /// Recipient ring for the new authority. At least two distinct values.
    #[arg(long = "recipient", value_name = "AGE_RECIPIENT")]
    pub recipients: Vec<String>,

    /// Keep existing active authority keys and add the new key.
    #[arg(long, conflicts_with = "replace")]
    pub add: bool,

    /// Retire existing active authority keys after authorizing the new key.
    #[arg(long, conflicts_with = "add")]
    pub replace: bool,

    /// Encrypted new authority key output.
    #[arg(long, default_value = "deployment-authority-next.age")]
    pub new_authority_key: PathBuf,

    /// Raw 1984-byte new authority public pair (operator record, not the root pin).
    #[arg(long, default_value = "deployment-authority-next.hybrid")]
    pub new_public_key: PathBuf,

    /// Updated signed rotation log output.
    #[arg(long, default_value = "deployment-authority.log.next.json")]
    pub authority_log_out: PathBuf,

    /// Updated independently trusted authority-log head output.
    #[arg(long, default_value = "deployment-authority.head.next.json")]
    pub authority_checkpoint_out: PathBuf,

    /// Replace existing output files.
    #[arg(long)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct VerifyDeploymentArgs {
    /// Raw 1984-byte public deployment root.
    #[arg(long, default_value = "deployment-ca.hybrid")]
    pub public_ca: PathBuf,

    /// Registry deployment credential.
    #[arg(long, default_value = "registry-service.jwt")]
    pub jwt: PathBuf,

    /// Installed/current authority log (required for every credential).
    #[arg(long, default_value = "deployment-authority.log.json")]
    pub authority_log: PathBuf,

    /// Independently trusted expected authority-log head.
    #[arg(long, default_value = "deployment-authority.head.json")]
    pub authority_checkpoint: PathBuf,

    /// Optional contract whose public artifacts must match the files.
    #[arg(long)]
    pub contract: Option<PathBuf>,
}
