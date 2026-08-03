//! Deployment-trust ceremony automation for the setup wizard.
//!
//! Node bootstrap (directories, node root key, per-service keypairs, CA-signed
//! service JWTs) and deployment trust are separate concerns and stay separate:
//! this module is opt-in and additive. Nothing here runs unless the operator
//! asks for the ceremony.
//!
//! What it adds is the decision an operator would otherwise make by reading
//! [`docs/deployment-trust-ceremony.md`]: is a hardware token attached, does its
//! firmware confine an Ed25519 key to the token, and — when nothing is attached
//! — is the operator willing to accept a software root that is only fit for
//! development.
//!
//! Hardware access sits behind [`TokenDetector`] and [`AgeYubikeyPlugin`] so the
//! decision logic can be exercised without a token. The decision logic itself is
//! pure: [`plan_ceremony`], [`select_mode_for_token`], [`validate_break_glass`]
//! and [`ceremony_commands`] take values and return values.

use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::cli::commands::{
    DelegateRegistrySignerArgs, MintDeploymentCaArgs, MintRegistryJwtArgs, TrustCommand,
    VerifyDeploymentArgs,
};

/// First firmware generation that can hold an Ed25519 key in a PIV slot.
pub const PIV_ED25519_MIN_FIRMWARE: FirmwareVersion = FirmwareVersion {
    major: 5,
    minor: 7,
    patch: 4,
};

/// PIV digital-signature slot: PIN is demanded for every signature.
pub const DEFAULT_PIV_SLOT: &str = "9c";

/// Schema of the mode record written beside the ceremony artifacts.
pub const CEREMONY_MODE_SCHEMA: &str = "hyprstream.deployment-trust-ceremony-mode.v1";

/// Human-visible grade of a software root. Deliberately shouty.
pub const DEV_GRADE_LABEL: &str = "DEV-GRADE — software root, no hardware gating, not for production";

/// Grade of a token-gated root.
const HARDWARE_GRADE_LABEL: &str = "hardware-gated deployment root";

/// The delegation lifetime the ceremony requests (30 days).
const DELEGATION_TTL_SECONDS: u64 = 2_592_000;

/// Registry credential lifetime; the profile caps it at one hour anyway.
const REGISTRY_JWT_TTL_SECONDS: u32 = 3600;

/// How long to wait for the token-enumeration tool before giving up.
const DETECTION_TIMEOUT: Duration = Duration::from_secs(10);

// ─────────────────────────────────────────────────────────────────────────────
// Token facts
// ─────────────────────────────────────────────────────────────────────────────

/// A three-part YubiKey firmware version.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FirmwareVersion {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
}

impl FirmwareVersion {
    #[must_use]
    pub const fn new(major: u8, minor: u8, patch: u8) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse `5.7.4`. Anything else is rejected rather than guessed at.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next()?.parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(Self::new(major, minor, patch))
    }

    /// Ed25519-in-PIV, which is what keeps the classical leg off the host.
    #[must_use]
    pub fn supports_piv_ed25519(&self) -> bool {
        *self >= PIV_ED25519_MIN_FIRMWARE
    }
}

impl std::fmt::Display for FirmwareVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// A token the detector reported as attached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DetectedToken {
    pub model: String,
    pub firmware: FirmwareVersion,
    pub serial: String,
}

impl DetectedToken {
    /// One-line label for prompts and summaries.
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "{} (firmware {}, serial {})",
            self.model, self.firmware, self.serial
        )
    }
}

/// Why the wizard could not learn whether a token is attached.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DetectionError {
    /// The enumeration tool is not installed.
    ToolMissing { tool: String },
    /// The tool did not finish within the detection budget.
    Timeout { tool: String, seconds: u64 },
    /// The tool ran but failed, or produced output we will not guess at.
    Failed { tool: String, detail: String },
}

impl std::fmt::Display for DetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToolMissing { tool } => write!(f, "`{tool}` is not installed"),
            Self::Timeout { tool, seconds } => {
                write!(f, "`{tool}` did not answer within {seconds}s")
            }
            Self::Failed { tool, detail } => write!(f, "`{tool}` failed: {detail}"),
        }
    }
}

impl std::error::Error for DetectionError {}

/// Enumerates attached hardware tokens.
///
/// The wizard depends on this trait, never on the tool directly, so the
/// selection logic can be driven from fixed detection results.
pub trait TokenDetector {
    fn detect(&self) -> Result<Vec<DetectedToken>, DetectionError>;
}

/// Creates and exports the PIV-backed age identity on an attached token.
pub trait AgeYubikeyPlugin {
    /// Generate a PIN-always/touch-always identity, returning its recipient.
    fn generate_identity(&self, name: &str) -> Result<String, DetectionError>;
    /// Write the identity stub the trust commands consume.
    fn export_identity_file(&self, out: &Path) -> Result<(), DetectionError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mode selection
// ─────────────────────────────────────────────────────────────────────────────

/// How the deployment root will be protected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CeremonyMode {
    /// Ed25519 leg is generated into a PIV slot and never enters host memory.
    HardwarePiv { token: DetectedToken, slot: String },
    /// Token decrypts the authority bundle; the bundle is briefly in memory.
    HardwareAgeRecipient { token: DetectedToken },
    /// Software recipients only. Fit for development, nothing else.
    SoftwareDevGrade,
}

impl CeremonyMode {
    #[must_use]
    pub fn is_dev_grade(&self) -> bool {
        matches!(self, Self::SoftwareDevGrade)
    }

    /// Stable identifier recorded in the emitted mode record.
    #[must_use]
    pub fn mode_id(&self) -> &'static str {
        match self {
            Self::HardwarePiv { .. } => "yubikey-piv",
            Self::HardwareAgeRecipient { .. } => "yubikey-age-recipient",
            Self::SoftwareDevGrade => "software-dev-grade",
        }
    }

    #[must_use]
    pub fn grade_label(&self) -> &'static str {
        if self.is_dev_grade() {
            DEV_GRADE_LABEL
        } else {
            HARDWARE_GRADE_LABEL
        }
    }

    /// True only when the classical leg is confined to the token.
    #[must_use]
    pub fn ed25519_confined_to_hardware(&self) -> bool {
        matches!(self, Self::HardwarePiv { .. })
    }

    #[must_use]
    pub fn token(&self) -> Option<&DetectedToken> {
        match self {
            Self::HardwarePiv { token, .. } | Self::HardwareAgeRecipient { token } => Some(token),
            Self::SoftwareDevGrade => None,
        }
    }
}

/// What the wizard decided to do about deployment trust.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CeremonyPlan {
    /// Not requested. Node bootstrap stands alone, exactly as before.
    Skipped { reason: String },
    /// A mode was chosen; `rationale` explains the choice in one sentence.
    Proceed { mode: CeremonyMode, rationale: String },
    /// Several tokens are attached and the operator must say which one.
    ChooseToken {
        tokens: Vec<DetectedToken>,
        rationale: String,
    },
    /// The ceremony cannot run as invoked. Never a silent downgrade.
    Blocked { reason: String, remedy: String },
}

/// Everything the plan depends on that does not come from the token itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CeremonyRequest {
    /// Operator asked for the ceremony. False means the wizard does nothing.
    pub enabled: bool,
    /// A human is present and can answer prompts, enter a PIN, and touch.
    pub interactive: bool,
    /// Mint a software root even if a token is attached.
    pub force_software: bool,
    /// Use the token with this serial when several are attached.
    pub serial: Option<String>,
    /// PIV slot for the Ed25519 leg when the firmware supports it.
    pub piv_slot: String,
}

impl Default for CeremonyRequest {
    fn default() -> Self {
        Self {
            enabled: false,
            interactive: true,
            force_software: false,
            serial: None,
            piv_slot: DEFAULT_PIV_SLOT.to_owned(),
        }
    }
}

/// Decide what to do, given a detection result.
///
/// A software root is only ever chosen when no token was found, or when the
/// operator explicitly asked for one. A detection failure blocks rather than
/// degrades: not knowing whether hardware is present is not the same as knowing
/// it is absent.
#[must_use]
pub fn plan_ceremony(
    request: &CeremonyRequest,
    detection: &Result<Vec<DetectedToken>, DetectionError>,
) -> CeremonyPlan {
    if !request.enabled {
        return CeremonyPlan::Skipped {
            reason: "deployment trust not requested; the wizard set up node-local trust only"
                .to_owned(),
        };
    }

    let tokens = match detection {
        Err(error) => {
            if request.force_software {
                return CeremonyPlan::Proceed {
                    mode: CeremonyMode::SoftwareDevGrade,
                    rationale: format!(
                        "token detection failed ({error}) and a software root was explicitly \
                         requested, so the root is {DEV_GRADE_LABEL}"
                    ),
                };
            }
            return CeremonyPlan::Blocked {
                reason: format!("could not tell whether a hardware token is attached: {error}"),
                remedy: "install `ykman` (yubikey-manager) and re-run, or pass \
                         --deployment-trust-software to accept a dev-grade software root"
                    .to_owned(),
            };
        }
        Ok(tokens) => tokens,
    };

    if request.force_software {
        let rationale = if tokens.is_empty() {
            format!("a software root was explicitly requested; the root is {DEV_GRADE_LABEL}")
        } else {
            format!(
                "{} hardware token(s) are attached but a software root was explicitly requested; \
                 the root is {DEV_GRADE_LABEL}",
                tokens.len()
            )
        };
        return CeremonyPlan::Proceed {
            mode: CeremonyMode::SoftwareDevGrade,
            rationale,
        };
    }

    if tokens.is_empty() {
        return CeremonyPlan::Proceed {
            mode: CeremonyMode::SoftwareDevGrade,
            rationale: format!(
                "no hardware token is attached, so the root falls back to software recipients: \
                 {DEV_GRADE_LABEL}"
            ),
        };
    }

    let token = match request.serial.as_deref() {
        Some(serial) => match tokens.iter().find(|token| token.serial == serial) {
            Some(token) => token,
            None => {
                let attached = tokens
                    .iter()
                    .map(|token| token.serial.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                return CeremonyPlan::Blocked {
                    reason: format!("no attached token has serial {serial}"),
                    remedy: format!("attached serials are: {attached}"),
                };
            }
        },
        None => match tokens.split_first() {
            Some((token, [])) => token,
            _ => {
                if request.interactive {
                    return CeremonyPlan::ChooseToken {
                        tokens: tokens.clone(),
                        rationale: format!(
                            "{} hardware tokens are attached; the deployment root must be bound to \
                             exactly one",
                            tokens.len()
                        ),
                    };
                }
                let attached = tokens
                    .iter()
                    .map(|token| token.serial.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                return CeremonyPlan::Blocked {
                    reason: format!("{} hardware tokens are attached", tokens.len()),
                    remedy: format!(
                        "pass --deployment-trust-serial <SERIAL> to pick one (attached: {attached})"
                    ),
                };
            }
        },
    };

    select_mode_for_token(request, token)
}

/// Resolve a chosen token to a mode, or block when no human can drive it.
///
/// Split out so the multiple-token prompt can feed the operator's answer back
/// through exactly the same firmware rule.
#[must_use]
pub fn select_mode_for_token(request: &CeremonyRequest, token: &DetectedToken) -> CeremonyPlan {
    if !request.interactive {
        return CeremonyPlan::Blocked {
            reason: format!(
                "{} is attached, and a hardware ceremony needs a PIN, a physical touch, and a \
                 break-glass passphrase",
                token.label()
            ),
            remedy: "re-run the wizard without --non-interactive, or pass \
                     --deployment-trust-software to mint a dev-grade software root instead"
                .to_owned(),
        };
    }

    if token.firmware.supports_piv_ed25519() {
        CeremonyPlan::Proceed {
            mode: CeremonyMode::HardwarePiv {
                token: token.clone(),
                slot: request.piv_slot.clone(),
            },
            rationale: format!(
                "{} runs firmware {} (>= {PIV_ED25519_MIN_FIRMWARE}), so the Ed25519 leg is \
                 generated into PIV slot {} and never enters host memory",
                token.label(),
                token.firmware,
                request.piv_slot
            ),
        }
    } else {
        CeremonyPlan::Proceed {
            mode: CeremonyMode::HardwareAgeRecipient {
                token: token.clone(),
            },
            rationale: format!(
                "{} runs firmware {}, below the {PIV_ED25519_MIN_FIRMWARE} needed for \
                 Ed25519-in-PIV, so the token acts as the age recipient instead: it gates \
                 decryption of the authority bundle, which is briefly in host memory during the \
                 ceremony",
                token.label(),
                token.firmware
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Break-glass
// ─────────────────────────────────────────────────────────────────────────────

/// The recovery half of the root recipient ring.
///
/// The trust CLI can only check that the two recipients differ; whether the
/// second is protected is decided here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BreakGlass {
    /// A second hardware token. Best: same properties as the primary.
    SecondToken { recipient: String },
    /// A passphrase-encrypted identity file, passphrase held separately.
    PassphraseEncrypted {
        identity_file: PathBuf,
        recipient: String,
    },
    /// A bare identity file. Anyone who can read it holds the root.
    PlaintextIdentity {
        identity_file: PathBuf,
        recipient: String,
    },
}

impl BreakGlass {
    #[must_use]
    pub fn recipient(&self) -> &str {
        match self {
            Self::SecondToken { recipient }
            | Self::PassphraseEncrypted { recipient, .. }
            | Self::PlaintextIdentity { recipient, .. } => recipient,
        }
    }

    #[must_use]
    pub fn kind_id(&self) -> &'static str {
        match self {
            Self::SecondToken { .. } => "second-token",
            Self::PassphraseEncrypted { .. } => "passphrase-encrypted-identity",
            Self::PlaintextIdentity { .. } => "plaintext-identity",
        }
    }

    #[must_use]
    pub fn is_protected(&self) -> bool {
        !matches!(self, Self::PlaintextIdentity { .. })
    }
}

/// Whether an unprotected break-glass is tolerable for the root being minted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaintextPolicy {
    /// Refuse. This is the default for any root worth protecting.
    Refuse,
    /// The operator explicitly overrode the refusal.
    AllowedByOverride,
    /// The root is already dev-grade, so a bare backup adds no exposure.
    AllowedForDevGradeRoot,
}

/// Reject a recipient ring that would leave the root weaker than it looks.
pub fn validate_break_glass(
    break_glass: &BreakGlass,
    primary_recipient: &str,
    policy: PlaintextPolicy,
) -> Result<(), String> {
    let recipient = break_glass.recipient().trim();
    if recipient.is_empty() {
        return Err("break-glass recipient is empty; paste the `age1…` recipient the backup \
                    identity prints"
            .to_owned());
    }
    if !recipient.is_ascii() || recipient.contains(['\n', '\r', '\0']) {
        return Err("break-glass recipient contains characters an age recipient cannot hold; \
                    re-copy it without line breaks"
            .to_owned());
    }
    if !recipient.starts_with("age1") {
        return Err(format!(
            "break-glass value {recipient} is not an age recipient; it must start with `age1`"
        ));
    }
    if recipient == primary_recipient.trim() {
        return Err("break-glass recipient is identical to the primary recipient; the root would \
                    then have a single point of loss, which is exactly what the second recipient \
                    exists to prevent"
            .to_owned());
    }
    if let BreakGlass::SecondToken { .. } = break_glass {
        if !recipient.starts_with("age1yubikey1") {
            return Err(format!(
                "{recipient} is not an age-plugin-yubikey recipient; a second-token break-glass \
                 recipient starts with `age1yubikey1`"
            ));
        }
    }
    if !break_glass.is_protected() && policy == PlaintextPolicy::Refuse {
        return Err("refusing an unencrypted break-glass identity: a root is exactly as strong as \
                    its weakest recipient, and a bare `age-keygen` file hands the deployment root \
                    to anyone who can read it. Use a second token, or a passphrase-encrypted \
                    identity (`age-keygen | age -p`). Pass \
                    --deployment-trust-allow-plaintext-break-glass only for a throwaway root you \
                    destroy immediately afterwards."
            .to_owned());
    }
    Ok(())
}

/// What a candidate break-glass identity file actually is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityFileKind {
    /// age file sealed with a scrypt (passphrase) stanza.
    PassphraseEncrypted,
    /// A bare `AGE-SECRET-KEY-1…` identity.
    PlaintextAgeIdentity,
    /// Encrypted, or unrecognised — protection cannot be confirmed from here.
    Unconfirmed,
}

/// Classify a candidate break-glass file without decrypting it.
///
/// Armored ciphertext reports [`IdentityFileKind::Unconfirmed`] rather than
/// claiming a passphrase we cannot see.
#[must_use]
pub fn classify_identity_file(contents: &str) -> IdentityFileKind {
    if contents.contains("AGE-SECRET-KEY-1") {
        return IdentityFileKind::PlaintextAgeIdentity;
    }
    let mut lines = contents.lines();
    if lines.next().map(str::trim) == Some("age-encryption.org/v1")
        && lines.any(|line| line.starts_with("-> scrypt "))
    {
        return IdentityFileKind::PassphraseEncrypted;
    }
    IdentityFileKind::Unconfirmed
}

// ─────────────────────────────────────────────────────────────────────────────
// Ceremony artifacts and commands
// ─────────────────────────────────────────────────────────────────────────────

/// Absolute paths of every ceremony input and output.
///
/// The trust CLI's bare-filename defaults fail from an unexpected working
/// directory, so the wizard always passes absolute paths.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CeremonyPaths {
    dir: PathBuf,
}

impl CeremonyPaths {
    /// Reject a relative directory up front rather than mid-ceremony.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        if !dir.is_absolute() {
            bail!(
                "ceremony directory {} must be an absolute path; the trust commands resolve \
                 bare filenames against the working directory and fail mid-ceremony",
                dir.display()
            );
        }
        Ok(Self { dir })
    }

    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    #[must_use]
    pub fn public_ca(&self) -> PathBuf {
        self.dir.join("deployment-ca.hybrid")
    }

    #[must_use]
    pub fn authority_key(&self) -> PathBuf {
        self.dir.join("deployment-ca.age")
    }

    #[must_use]
    pub fn authority_log(&self) -> PathBuf {
        self.dir.join("deployment-authority.log.json")
    }

    #[must_use]
    pub fn authority_checkpoint(&self) -> PathBuf {
        self.dir.join("deployment-authority.head.json")
    }

    #[must_use]
    pub fn yubikey_identity(&self) -> PathBuf {
        self.dir.join("yubikey-identity.txt")
    }

    #[must_use]
    pub fn primary_identity(&self) -> PathBuf {
        self.dir.join("root-primary.key")
    }

    #[must_use]
    pub fn break_glass_identity(&self) -> PathBuf {
        self.dir.join("break-glass.key.age")
    }

    #[must_use]
    pub fn online_signer_identity(&self) -> PathBuf {
        self.dir.join("online-signer.key")
    }

    #[must_use]
    pub fn delegated_signer(&self) -> PathBuf {
        self.dir.join("registry-delegated-signer.age")
    }

    #[must_use]
    pub fn delegation(&self) -> PathBuf {
        self.dir.join("registry-signer.delegation.json")
    }

    #[must_use]
    pub fn registry_public_key(&self) -> PathBuf {
        self.dir.join("registry-service.pub")
    }

    #[must_use]
    pub fn registry_jwt(&self) -> PathBuf {
        self.dir.join("registry-service.jwt")
    }

    #[must_use]
    pub fn contract(&self) -> PathBuf {
        self.dir.join("deployment-trust.contract.json")
    }

    #[must_use]
    pub fn mode_record(&self) -> PathBuf {
        self.dir.join("deployment-trust-mode.json")
    }
}

/// The identity that unlocks the authority bundle during the delegation step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnlockIdentity {
    /// An age-plugin-yubikey identity stub: decryption costs a PIN and a touch.
    Yubikey(PathBuf),
    /// A native age identity file.
    Age(PathBuf),
}

/// Every value the four trust invocations need.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CeremonyInputs {
    pub mode: CeremonyMode,
    /// Primary root recipient (`age1yubikey1…` in hardware modes).
    pub primary_recipient: String,
    pub break_glass: BreakGlass,
    pub unlock_identity: UnlockIdentity,
    /// Recipient the unattended deploy can decrypt the online signer with.
    pub signer_recipient: String,
    pub paths: CeremonyPaths,
}

/// Build the ceremony as the exact sequence of trust commands it runs.
///
/// Returning commands rather than running them keeps the mode-to-flag mapping —
/// the part that decides whether the classical leg ever touches host memory —
/// inspectable without a token.
pub fn ceremony_commands(inputs: &CeremonyInputs) -> Result<Vec<TrustCommand>> {
    let policy = if inputs.mode.is_dev_grade() {
        PlaintextPolicy::AllowedForDevGradeRoot
    } else {
        PlaintextPolicy::Refuse
    };
    if let Err(reason) =
        validate_break_glass(&inputs.break_glass, &inputs.primary_recipient, policy)
    {
        bail!("{reason}");
    }

    let (mut recipients, mut yubikey_recipients) = (Vec::new(), Vec::new());
    for recipient in [
        inputs.primary_recipient.trim(),
        inputs.break_glass.recipient().trim(),
    ] {
        if recipient.starts_with("age1yubikey1") {
            yubikey_recipients.push(recipient.to_owned());
        } else {
            recipients.push(recipient.to_owned());
        }
    }

    let piv_slot = match &inputs.mode {
        CeremonyMode::HardwarePiv { slot, .. } => Some(slot.clone()),
        CeremonyMode::HardwareAgeRecipient { .. } | CeremonyMode::SoftwareDevGrade => None,
    };

    let (identities, yubikey_identities) = match &inputs.unlock_identity {
        UnlockIdentity::Yubikey(path) => (Vec::new(), vec![path.clone()]),
        UnlockIdentity::Age(path) => (vec![path.clone()], Vec::new()),
    };

    Ok(vec![
        TrustCommand::MintDeploymentCa(MintDeploymentCaArgs {
            public_ca: inputs.paths.public_ca(),
            authority_key: inputs.paths.authority_key(),
            authority_log: inputs.paths.authority_log(),
            authority_checkpoint: inputs.paths.authority_checkpoint(),
            recipients,
            yubikey_recipients,
            kms_plugin_recipients: Vec::new(),
            piv_slot,
            force: false,
        }),
        TrustCommand::DelegateRegistrySigner(DelegateRegistrySignerArgs {
            public_ca: inputs.paths.public_ca(),
            authority_log: inputs.paths.authority_log(),
            authority_checkpoint: inputs.paths.authority_checkpoint(),
            authority_key: inputs.paths.authority_key(),
            identities: identities.clone(),
            yubikey_identities: yubikey_identities.clone(),
            software_recovery: false,
            signer_recipients: vec![inputs.signer_recipient.trim().to_owned()],
            delegated_key: inputs.paths.delegated_signer(),
            delegation: inputs.paths.delegation(),
            delegation_ttl_seconds: DELEGATION_TTL_SECONDS,
            force: false,
        }),
        TrustCommand::MintRegistryJwt(MintRegistryJwtArgs {
            public_ca: inputs.paths.public_ca(),
            authority_key: inputs.paths.authority_key(),
            identities: vec![inputs.paths.online_signer_identity()],
            yubikey_identities: Vec::new(),
            software_recovery: false,
            via_delegated_signer: Some(inputs.paths.delegated_signer()),
            delegation: Some(inputs.paths.delegation()),
            authority_log: inputs.paths.authority_log(),
            authority_checkpoint: inputs.paths.authority_checkpoint(),
            root: false,
            registry_public_key: inputs.paths.registry_public_key(),
            ttl_seconds: REGISTRY_JWT_TTL_SECONDS,
            jwt: inputs.paths.registry_jwt(),
            contract: inputs.paths.contract(),
            force: false,
        }),
        TrustCommand::VerifyDeployment(VerifyDeploymentArgs {
            public_ca: inputs.paths.public_ca(),
            jwt: inputs.paths.registry_jwt(),
            authority_log: inputs.paths.authority_log(),
            authority_checkpoint: inputs.paths.authority_checkpoint(),
            contract: Some(inputs.paths.contract()),
        }),
    ])
}

/// Machine-readable record of which mode ran, written beside the artifacts.
///
/// A software root is labelled here as well as on screen, so the grade survives
/// after the operator has closed the terminal.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct CeremonyModeRecord {
    pub schema: String,
    pub mode: String,
    pub grade: String,
    pub dev_grade: bool,
    pub ed25519_confined_to_hardware: bool,
    pub token_model: Option<String>,
    pub token_serial: Option<String>,
    pub token_firmware: Option<String>,
    pub piv_slot: Option<String>,
    pub break_glass: String,
    pub break_glass_protected: bool,
    pub rationale: String,
}

/// Summarise a completed selection for the emitted record.
#[must_use]
pub fn mode_record(mode: &CeremonyMode, break_glass: &BreakGlass, rationale: &str) -> CeremonyModeRecord {
    let token = mode.token();
    CeremonyModeRecord {
        schema: CEREMONY_MODE_SCHEMA.to_owned(),
        mode: mode.mode_id().to_owned(),
        grade: mode.grade_label().to_owned(),
        dev_grade: mode.is_dev_grade(),
        ed25519_confined_to_hardware: mode.ed25519_confined_to_hardware(),
        token_model: token.map(|token| token.model.clone()),
        token_serial: token.map(|token| token.serial.clone()),
        token_firmware: token.map(|token| token.firmware.to_string()),
        piv_slot: match mode {
            CeremonyMode::HardwarePiv { slot, .. } => Some(slot.clone()),
            CeremonyMode::HardwareAgeRecipient { .. } | CeremonyMode::SoftwareDevGrade => None,
        },
        break_glass: break_glass.kind_id().to_owned(),
        break_glass_protected: break_glass.is_protected(),
        rationale: rationale.to_owned(),
    }
}

/// External programs a mode needs on `PATH`.
#[must_use]
pub fn required_tools(mode: &CeremonyMode) -> &'static [&'static str] {
    match mode {
        CeremonyMode::HardwarePiv { .. } => &["ykman", "yubico-piv-tool", "age-plugin-yubikey"],
        CeremonyMode::HardwareAgeRecipient { .. } => &["age-plugin-yubikey"],
        CeremonyMode::SoftwareDevGrade => &["age-keygen", "age"],
    }
}

/// Names from `required_tools` that `probe` reports as absent.
///
/// `probe` is a parameter so the check is exercised without touching `PATH`.
#[must_use]
pub fn missing_tools(tools: &[&str], probe: impl Fn(&str) -> bool) -> Vec<String> {
    tools
        .iter()
        .filter(|tool| !probe(tool))
        .map(|tool| (*tool).to_owned())
        .collect()
}

/// True when `tool` resolves through `PATH`.
#[must_use]
pub fn tool_on_path(tool: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(tool).is_file())
}

// ─────────────────────────────────────────────────────────────────────────────
// Output parsing (pure; the commands themselves need hardware)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `ykman list` output.
///
/// A line that names a token but whose firmware cannot be read is an error, not
/// a token silently dropped from the ring: dropping it could turn "hardware is
/// present" into "nothing attached", which selects a software root.
pub fn parse_ykman_list(stdout: &str) -> Result<Vec<DetectedToken>, DetectionError> {
    let mut tokens = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let unreadable = || DetectionError::Failed {
            tool: "ykman".to_owned(),
            detail: format!("could not read model, firmware and serial from `{line}`"),
        };
        let (model, rest) = line.split_once(" (").ok_or_else(unreadable)?;
        let (firmware, rest) = rest.split_once(')').ok_or_else(unreadable)?;
        let firmware = FirmwareVersion::parse(firmware).ok_or_else(unreadable)?;
        let (_, serial) = rest.split_once("Serial: ").ok_or_else(unreadable)?;
        let serial = serial.trim();
        if serial.is_empty() || !serial.chars().all(|c| c.is_ascii_digit()) {
            return Err(unreadable());
        }
        tokens.push(DetectedToken {
            model: model.trim().to_owned(),
            firmware,
            serial: serial.to_owned(),
        });
    }
    Ok(tokens)
}

/// Pull the `age1yubikey1…` recipient out of `age-plugin-yubikey` output.
#[must_use]
pub fn parse_age_plugin_recipient(output: &str) -> Option<String> {
    output
        .split_whitespace()
        .find(|word| word.starts_with("age1yubikey1"))
        .map(str::to_owned)
}

/// Pull the public recipient out of `age-keygen` output.
#[must_use]
pub fn parse_age_keygen_recipient(output: &str) -> Option<String> {
    output
        .split_whitespace()
        .find(|word| word.starts_with("age1") && !word.starts_with("age1yubikey1"))
        .map(str::to_owned)
}

// ─────────────────────────────────────────────────────────────────────────────
// Real hardware backend
// ─────────────────────────────────────────────────────────────────────────────

/// Enumerates tokens with `ykman list`.
///
/// The hardware half of this type cannot be exercised without a token; the
/// parsing and decision it feeds are covered separately.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemTokenDetector;

impl TokenDetector for SystemTokenDetector {
    fn detect(&self) -> Result<Vec<DetectedToken>, DetectionError> {
        if !tool_on_path("ykman") {
            return Err(DetectionError::ToolMissing {
                tool: "ykman".to_owned(),
            });
        }
        let stdout = run_with_timeout("ykman", &["list"], DETECTION_TIMEOUT)?;
        parse_ykman_list(&stdout)
    }
}

/// Drives `age-plugin-yubikey`, which needs a real TTY for PIN entry.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemAgeYubikeyPlugin;

impl AgeYubikeyPlugin for SystemAgeYubikeyPlugin {
    fn generate_identity(&self, name: &str) -> Result<String, DetectionError> {
        if !tool_on_path("age-plugin-yubikey") {
            return Err(DetectionError::ToolMissing {
                tool: "age-plugin-yubikey".to_owned(),
            });
        }
        // Inherited stdio: the plugin refuses a pipe, and the operator has to
        // see the PIN prompt. `always` costs a PIN and a touch per decryption.
        let status = Command::new("age-plugin-yubikey")
            .args(["--generate", "--name"])
            .arg(name)
            .args(["--pin-policy", "always", "--touch-policy", "always"])
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .map_err(|error| DetectionError::Failed {
                tool: "age-plugin-yubikey".to_owned(),
                detail: error.to_string(),
            })?;
        if !status.success() {
            return Err(DetectionError::Failed {
                tool: "age-plugin-yubikey".to_owned(),
                detail: "identity generation did not complete".to_owned(),
            });
        }
        let listed = run_with_timeout("age-plugin-yubikey", &["--list"], DETECTION_TIMEOUT)?;
        parse_age_plugin_recipient(&listed).ok_or_else(|| DetectionError::Failed {
            tool: "age-plugin-yubikey".to_owned(),
            detail: "no `age1yubikey1…` recipient was listed after generation".to_owned(),
        })
    }

    fn export_identity_file(&self, out: &Path) -> Result<(), DetectionError> {
        let identity = run_with_timeout("age-plugin-yubikey", &["--identity"], DETECTION_TIMEOUT)?;
        std::fs::write(out, identity).map_err(|error| DetectionError::Failed {
            tool: "age-plugin-yubikey".to_owned(),
            detail: format!("write identity to {}: {error}", out.display()),
        })
    }
}

/// Run a command, capturing stdout and killing it if it overruns `budget`.
fn run_with_timeout(tool: &str, args: &[&str], budget: Duration) -> Result<String, DetectionError> {
    let mut child = Command::new(tool)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                DetectionError::ToolMissing {
                    tool: tool.to_owned(),
                }
            } else {
                DetectionError::Failed {
                    tool: tool.to_owned(),
                    detail: error.to_string(),
                }
            }
        })?;

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if started.elapsed() >= budget {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(DetectionError::Timeout {
                        tool: tool.to_owned(),
                        seconds: budget.as_secs(),
                    });
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => {
                return Err(DetectionError::Failed {
                    tool: tool.to_owned(),
                    detail: error.to_string(),
                })
            }
        }
    }

    let output = child
        .wait_with_output()
        .map_err(|error| DetectionError::Failed {
            tool: tool.to_owned(),
            detail: error.to_string(),
        })?;
    if !output.status.success() {
        return Err(DetectionError::Failed {
            tool: tool.to_owned(),
            detail: String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Create the ceremony directory, owner-only.
pub fn create_ceremony_dir(dir: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::create_dir_all(dir)
        .with_context(|| format!("create ceremony directory {}", dir.display()))?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("restrict ceremony directory {} to its owner", dir.display()))
}

/// Generate a native age identity at `out`, returning its recipient.
///
/// Used for the online signer, and for the software root's own recipients.
pub fn generate_software_identity(out: &Path) -> Result<String> {
    use std::os::unix::fs::PermissionsExt as _;
    let output = Command::new("age-keygen")
        .arg("-o")
        .arg(out)
        .stdin(Stdio::null())
        .output()
        .context("launch age-keygen (install the `age` tools)")?;
    if !output.status.success() {
        bail!(
            "age-keygen failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    std::fs::set_permissions(out, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("restrict {} to its owner", out.display()))?;
    let printed = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    parse_age_keygen_recipient(&printed)
        .ok_or_else(|| anyhow::anyhow!("age-keygen printed no `age1…` recipient"))
}

/// Generate a passphrase-encrypted break-glass identity, returning its recipient.
///
/// The identity is piped straight into `age -p`, so the private half is never
/// written unencrypted. `age` reads the passphrase from the terminal, which is
/// why this is offered only on an interactive run.
pub fn generate_passphrase_encrypted_identity(out: &Path) -> Result<String> {
    use std::os::unix::fs::PermissionsExt as _;
    let mut keygen = Command::new("age-keygen")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("launch age-keygen (install the `age` tools)")?;
    let keygen_stdout = keygen
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("age-keygen produced no identity stream"))?;
    let seal = Command::new("age")
        .args(["-p", "-o"])
        .arg(out)
        .stdin(Stdio::from(keygen_stdout))
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("launch age (install the `age` tools)")?;
    let keygen = keygen
        .wait_with_output()
        .context("wait for age-keygen to finish")?;
    if !keygen.status.success() {
        bail!(
            "age-keygen failed: {}",
            String::from_utf8_lossy(&keygen.stderr).trim()
        );
    }
    if !seal.success() {
        bail!("age -p did not encrypt the break-glass identity; nothing was written");
    }
    std::fs::set_permissions(out, std::fs::Permissions::from_mode(0o600))
        .with_context(|| format!("restrict {} to its owner", out.display()))?;
    parse_age_keygen_recipient(&String::from_utf8_lossy(&keygen.stderr))
        .ok_or_else(|| anyhow::anyhow!("age-keygen printed no `age1…` recipient"))
}

/// Write the raw 32-byte registry-service public key the JWT mint binds to.
pub fn write_registry_public_key(out: &Path, public_key: &[u8; 32]) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create ceremony directory {}", parent.display()))?;
    }
    std::fs::write(out, public_key)
        .with_context(|| format!("write registry public key to {}", out.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(model: &str, firmware: (u8, u8, u8), serial: &str) -> DetectedToken {
        DetectedToken {
            model: model.to_owned(),
            firmware: FirmwareVersion::new(firmware.0, firmware.1, firmware.2),
            serial: serial.to_owned(),
        }
    }

    fn old_token() -> DetectedToken {
        token("YubiKey 5C NFC", (5, 4, 3), "11111111")
    }

    fn new_token() -> DetectedToken {
        token("YubiKey 5C NFC", (5, 7, 4), "22222222")
    }

    fn enabled() -> CeremonyRequest {
        CeremonyRequest {
            enabled: true,
            ..CeremonyRequest::default()
        }
    }

    /// A detector whose answer is fixed by the test.
    struct FakeDetector(Result<Vec<DetectedToken>, DetectionError>);

    impl TokenDetector for FakeDetector {
        fn detect(&self) -> Result<Vec<DetectedToken>, DetectionError> {
            self.0.clone()
        }
    }

    // ── Firmware ────────────────────────────────────────────────────────────

    #[test]
    fn firmware_parses_three_part_versions_only() {
        assert_eq!(
            FirmwareVersion::parse("5.7.4"),
            Some(FirmwareVersion::new(5, 7, 4))
        );
        assert_eq!(FirmwareVersion::parse("5.7"), None);
        assert_eq!(FirmwareVersion::parse("5.7.4.1"), None);
        assert_eq!(FirmwareVersion::parse("five.seven.four"), None);
    }

    #[test]
    fn piv_ed25519_support_starts_at_the_documented_firmware() {
        assert!(!FirmwareVersion::new(5, 4, 3).supports_piv_ed25519());
        assert!(!FirmwareVersion::new(5, 7, 3).supports_piv_ed25519());
        assert!(FirmwareVersion::new(5, 7, 4).supports_piv_ed25519());
        assert!(FirmwareVersion::new(5, 8, 0).supports_piv_ed25519());
        assert!(FirmwareVersion::new(6, 0, 0).supports_piv_ed25519());
        assert!(!FirmwareVersion::new(4, 9, 9).supports_piv_ed25519());
    }

    // ── Selection ───────────────────────────────────────────────────────────

    #[test]
    fn ceremony_is_skipped_unless_requested() {
        let detection = Ok(vec![new_token()]);
        let plan = plan_ceremony(&CeremonyRequest::default(), &detection);
        assert!(matches!(plan, CeremonyPlan::Skipped { .. }));
    }

    #[test]
    fn no_token_selects_the_dev_grade_software_root() {
        let plan = plan_ceremony(&enabled(), &Ok(Vec::new()));
        let CeremonyPlan::Proceed { mode, rationale } = plan else {
            panic!("expected a software root, got {plan:?}");
        };
        assert_eq!(mode, CeremonyMode::SoftwareDevGrade);
        assert!(mode.is_dev_grade());
        assert!(!mode.ed25519_confined_to_hardware());
        assert!(rationale.contains("no hardware token"));
        assert!(rationale.contains(DEV_GRADE_LABEL));
    }

    #[test]
    fn old_firmware_selects_the_age_recipient_mode() {
        let plan = plan_ceremony(&enabled(), &Ok(vec![old_token()]));
        let CeremonyPlan::Proceed { mode, rationale } = plan else {
            panic!("expected a hardware mode, got {plan:?}");
        };
        assert_eq!(mode, CeremonyMode::HardwareAgeRecipient { token: old_token() });
        assert!(!mode.ed25519_confined_to_hardware());
        assert!(rationale.contains("5.4.3"));
        assert!(rationale.contains("host memory"));
    }

    #[test]
    fn current_firmware_selects_the_piv_slot_mode() {
        let plan = plan_ceremony(&enabled(), &Ok(vec![new_token()]));
        let CeremonyPlan::Proceed { mode, rationale } = plan else {
            panic!("expected a hardware mode, got {plan:?}");
        };
        assert_eq!(
            mode,
            CeremonyMode::HardwarePiv {
                token: new_token(),
                slot: DEFAULT_PIV_SLOT.to_owned(),
            }
        );
        assert!(mode.ed25519_confined_to_hardware());
        assert!(rationale.contains("never enters host memory"));
    }

    #[test]
    fn several_tokens_ask_the_operator_which_one() {
        let plan = plan_ceremony(&enabled(), &Ok(vec![old_token(), new_token()]));
        let CeremonyPlan::ChooseToken { tokens, .. } = plan else {
            panic!("expected an operator choice, got {plan:?}");
        };
        assert_eq!(tokens, vec![old_token(), new_token()]);
    }

    #[test]
    fn a_named_serial_picks_its_token_without_prompting() {
        let request = CeremonyRequest {
            serial: Some(new_token().serial),
            ..enabled()
        };
        let plan = plan_ceremony(&request, &Ok(vec![old_token(), new_token()]));
        let CeremonyPlan::Proceed { mode, .. } = plan else {
            panic!("expected a hardware mode, got {plan:?}");
        };
        assert_eq!(mode.token(), Some(&new_token()));
    }

    #[test]
    fn an_unattached_serial_blocks_and_lists_what_is_attached() {
        let request = CeremonyRequest {
            serial: Some("99999999".to_owned()),
            ..enabled()
        };
        let plan = plan_ceremony(&request, &Ok(vec![old_token(), new_token()]));
        let CeremonyPlan::Blocked { reason, remedy } = plan else {
            panic!("expected a block, got {plan:?}");
        };
        assert!(reason.contains("99999999"));
        assert!(remedy.contains("11111111") && remedy.contains("22222222"));
    }

    #[test]
    fn detection_failure_blocks_rather_than_falling_back_to_software() {
        for error in [
            DetectionError::ToolMissing {
                tool: "ykman".to_owned(),
            },
            DetectionError::Timeout {
                tool: "ykman".to_owned(),
                seconds: 10,
            },
            DetectionError::Failed {
                tool: "ykman".to_owned(),
                detail: "unreadable line".to_owned(),
            },
        ] {
            let plan = plan_ceremony(&enabled(), &Err(error.clone()));
            let CeremonyPlan::Blocked { reason, remedy } = plan else {
                panic!("expected a block for {error:?}");
            };
            assert!(reason.contains("could not tell whether"));
            assert!(remedy.contains("--deployment-trust-software"));
        }
    }

    #[test]
    fn detection_failure_may_be_overridden_into_an_explicit_software_root() {
        let request = CeremonyRequest {
            force_software: true,
            ..enabled()
        };
        let plan = plan_ceremony(
            &request,
            &Err(DetectionError::ToolMissing {
                tool: "ykman".to_owned(),
            }),
        );
        let CeremonyPlan::Proceed { mode, rationale } = plan else {
            panic!("expected an explicit software root, got {plan:?}");
        };
        assert_eq!(mode, CeremonyMode::SoftwareDevGrade);
        assert!(rationale.contains("explicitly requested"));
        assert!(rationale.contains(DEV_GRADE_LABEL));
    }

    #[test]
    fn software_is_never_chosen_over_present_hardware_without_an_explicit_override() {
        // Attached hardware always wins unless the operator overrides it.
        for tokens in [vec![old_token()], vec![new_token()], vec![old_token(), new_token()]] {
            let plan = plan_ceremony(&enabled(), &Ok(tokens.clone()));
            assert!(
                !matches!(
                    plan,
                    CeremonyPlan::Proceed {
                        mode: CeremonyMode::SoftwareDevGrade,
                        ..
                    }
                ),
                "software root silently selected with {tokens:?} attached"
            );
        }
    }

    #[test]
    fn an_explicit_override_says_it_overrode_attached_hardware() {
        let request = CeremonyRequest {
            force_software: true,
            ..enabled()
        };
        let plan = plan_ceremony(&request, &Ok(vec![new_token()]));
        let CeremonyPlan::Proceed { mode, rationale } = plan else {
            panic!("expected a software root, got {plan:?}");
        };
        assert_eq!(mode, CeremonyMode::SoftwareDevGrade);
        assert!(rationale.contains("hardware token(s) are attached"));
    }

    // ── Non-interactive safety ──────────────────────────────────────────────

    #[test]
    fn a_scripted_run_never_reaches_a_prompt() {
        let scripted = CeremonyRequest {
            interactive: false,
            ..enabled()
        };
        for detection in [
            Ok(Vec::new()),
            Ok(vec![old_token()]),
            Ok(vec![new_token()]),
            Ok(vec![old_token(), new_token()]),
            Err(DetectionError::ToolMissing {
                tool: "ykman".to_owned(),
            }),
        ] {
            let plan = plan_ceremony(&scripted, &detection);
            assert!(
                !matches!(plan, CeremonyPlan::ChooseToken { .. }),
                "scripted run would have prompted for {detection:?}"
            );
        }
    }

    #[test]
    fn a_scripted_run_with_hardware_blocks_with_both_ways_forward() {
        let scripted = CeremonyRequest {
            interactive: false,
            ..enabled()
        };
        let CeremonyPlan::Blocked { reason, remedy } =
            plan_ceremony(&scripted, &Ok(vec![new_token()]))
        else {
            panic!("expected a block");
        };
        assert!(reason.contains("physical touch"));
        assert!(remedy.contains("--non-interactive"));
        assert!(remedy.contains("--deployment-trust-software"));
    }

    #[test]
    fn a_scripted_run_with_several_tokens_asks_for_a_serial() {
        let scripted = CeremonyRequest {
            interactive: false,
            ..enabled()
        };
        let CeremonyPlan::Blocked { remedy, .. } =
            plan_ceremony(&scripted, &Ok(vec![old_token(), new_token()]))
        else {
            panic!("expected a block");
        };
        assert!(remedy.contains("--deployment-trust-serial"));
    }

    #[test]
    fn a_scripted_run_without_hardware_still_completes() {
        let scripted = CeremonyRequest {
            interactive: false,
            ..enabled()
        };
        let CeremonyPlan::Proceed { mode, .. } = plan_ceremony(&scripted, &Ok(Vec::new())) else {
            panic!("expected a software root");
        };
        assert!(mode.is_dev_grade());
    }

    #[test]
    fn a_scripted_run_that_did_not_ask_for_trust_is_skipped_even_with_hardware() {
        let scripted = CeremonyRequest {
            interactive: false,
            ..CeremonyRequest::default()
        };
        assert!(matches!(
            plan_ceremony(&scripted, &Ok(vec![new_token()])),
            CeremonyPlan::Skipped { .. }
        ));
    }

    #[test]
    fn a_chosen_token_runs_through_the_same_firmware_rule() {
        let request = enabled();
        let CeremonyPlan::Proceed { mode, .. } = select_mode_for_token(&request, &new_token())
        else {
            panic!("expected a hardware mode");
        };
        assert!(mode.ed25519_confined_to_hardware());
        let CeremonyPlan::Proceed { mode, .. } = select_mode_for_token(&request, &old_token())
        else {
            panic!("expected a hardware mode");
        };
        assert!(!mode.ed25519_confined_to_hardware());
    }

    #[test]
    fn the_detector_seam_feeds_the_plan() {
        let detector = FakeDetector(Ok(vec![new_token()]));
        let plan = plan_ceremony(&enabled(), &detector.detect());
        assert!(matches!(
            plan,
            CeremonyPlan::Proceed {
                mode: CeremonyMode::HardwarePiv { .. },
                ..
            }
        ));
    }

    // ── Break-glass ─────────────────────────────────────────────────────────

    #[test]
    fn a_plaintext_break_glass_is_refused_by_default() {
        let break_glass = BreakGlass::PlaintextIdentity {
            identity_file: PathBuf::from("/tmp/backup.key"),
            recipient: "age1backup".to_owned(),
        };
        let error = validate_break_glass(&break_glass, "age1yubikey1primary", PlaintextPolicy::Refuse)
            .expect_err("plaintext break-glass must be refused");
        assert!(error.contains("weakest recipient"));
        assert!(error.contains("--deployment-trust-allow-plaintext-break-glass"));
    }

    #[test]
    fn a_plaintext_break_glass_is_allowed_only_by_override_or_for_a_dev_root() {
        let break_glass = BreakGlass::PlaintextIdentity {
            identity_file: PathBuf::from("/tmp/backup.key"),
            recipient: "age1backup".to_owned(),
        };
        for policy in [
            PlaintextPolicy::AllowedByOverride,
            PlaintextPolicy::AllowedForDevGradeRoot,
        ] {
            assert!(validate_break_glass(&break_glass, "age1yubikey1primary", policy).is_ok());
        }
    }

    #[test]
    fn a_passphrase_encrypted_break_glass_is_accepted() {
        let break_glass = BreakGlass::PassphraseEncrypted {
            identity_file: PathBuf::from("/tmp/backup.key.age"),
            recipient: "age1backup".to_owned(),
        };
        assert!(validate_break_glass(
            &break_glass,
            "age1yubikey1primary",
            PlaintextPolicy::Refuse
        )
        .is_ok());
    }

    #[test]
    fn a_duplicate_break_glass_recipient_is_refused() {
        let break_glass = BreakGlass::SecondToken {
            recipient: "age1yubikey1primary".to_owned(),
        };
        let error = validate_break_glass(
            &break_glass,
            "age1yubikey1primary",
            PlaintextPolicy::AllowedByOverride,
        )
        .expect_err("a duplicated recipient must be refused");
        assert!(error.contains("single point of loss"));
    }

    #[test]
    fn a_second_token_break_glass_must_be_a_plugin_recipient() {
        let break_glass = BreakGlass::SecondToken {
            recipient: "age1software".to_owned(),
        };
        let error =
            validate_break_glass(&break_glass, "age1yubikey1primary", PlaintextPolicy::Refuse)
                .expect_err("a software recipient is not a second token");
        assert!(error.contains("age1yubikey1"));
    }

    #[test]
    fn malformed_break_glass_recipients_are_refused() {
        for recipient in ["", "   ", "not-a-recipient", "age1good\nage1evil"] {
            let break_glass = BreakGlass::PassphraseEncrypted {
                identity_file: PathBuf::from("/tmp/backup.key.age"),
                recipient: recipient.to_owned(),
            };
            assert!(
                validate_break_glass(&break_glass, "age1yubikey1primary", PlaintextPolicy::Refuse)
                    .is_err(),
                "accepted malformed recipient {recipient:?}"
            );
        }
    }

    #[test]
    fn identity_files_are_classified_without_decryption() {
        assert_eq!(
            classify_identity_file(
                "# created: 2026-01-01\n# public key: age1abc\nAGE-SECRET-KEY-1QQQQ\n"
            ),
            IdentityFileKind::PlaintextAgeIdentity
        );
        assert_eq!(
            classify_identity_file("age-encryption.org/v1\n-> scrypt abcdef 18\nxxxx\n"),
            IdentityFileKind::PassphraseEncrypted
        );
        assert_eq!(
            classify_identity_file("age-encryption.org/v1\n-> X25519 abcdef\nxxxx\n"),
            IdentityFileKind::Unconfirmed
        );
        assert_eq!(
            classify_identity_file("-----BEGIN AGE ENCRYPTED FILE-----\nYWJj\n"),
            IdentityFileKind::Unconfirmed
        );
    }

    // ── Command construction ────────────────────────────────────────────────

    fn paths() -> CeremonyPaths {
        CeremonyPaths::new("/srv/ceremony").expect("absolute path")
    }

    fn inputs(mode: CeremonyMode, primary: &str, break_glass: BreakGlass) -> CeremonyInputs {
        CeremonyInputs {
            mode,
            primary_recipient: primary.to_owned(),
            break_glass,
            unlock_identity: UnlockIdentity::Yubikey(paths().yubikey_identity()),
            signer_recipient: "age1deploy".to_owned(),
            paths: paths(),
        }
    }

    #[test]
    fn a_relative_ceremony_directory_is_refused() {
        assert!(CeremonyPaths::new("ceremony").is_err());
    }

    #[test]
    fn the_piv_mode_sets_the_slot_and_keeps_both_recipients() {
        let commands = ceremony_commands(&inputs(
            CeremonyMode::HardwarePiv {
                token: new_token(),
                slot: DEFAULT_PIV_SLOT.to_owned(),
            },
            "age1yubikey1primary",
            BreakGlass::PassphraseEncrypted {
                identity_file: PathBuf::from("/srv/ceremony/break-glass.key.age"),
                recipient: "age1backup".to_owned(),
            },
        ))
        .expect("commands");
        let TrustCommand::MintDeploymentCa(args) = &commands[0] else {
            panic!("expected the mint step first");
        };
        assert_eq!(args.piv_slot.as_deref(), Some(DEFAULT_PIV_SLOT));
        assert_eq!(args.yubikey_recipients, vec!["age1yubikey1primary"]);
        assert_eq!(args.recipients, vec!["age1backup"]);
        assert!(args.public_ca.is_absolute());
    }

    #[test]
    fn the_age_recipient_mode_leaves_the_piv_slot_unset() {
        let commands = ceremony_commands(&inputs(
            CeremonyMode::HardwareAgeRecipient { token: old_token() },
            "age1yubikey1primary",
            BreakGlass::SecondToken {
                recipient: "age1yubikey1backup".to_owned(),
            },
        ))
        .expect("commands");
        let TrustCommand::MintDeploymentCa(args) = &commands[0] else {
            panic!("expected the mint step first");
        };
        assert_eq!(args.piv_slot, None);
        assert_eq!(
            args.yubikey_recipients,
            vec!["age1yubikey1primary", "age1yubikey1backup"]
        );
        assert!(args.recipients.is_empty());
    }

    #[test]
    fn the_software_mode_uses_native_recipients_and_no_slot() {
        let mut inputs = inputs(
            CeremonyMode::SoftwareDevGrade,
            "age1primary",
            BreakGlass::PlaintextIdentity {
                identity_file: PathBuf::from("/srv/ceremony/break-glass.key"),
                recipient: "age1backup".to_owned(),
            },
        );
        inputs.unlock_identity = UnlockIdentity::Age(paths().primary_identity());
        let commands = ceremony_commands(&inputs).expect("commands");
        let TrustCommand::MintDeploymentCa(args) = &commands[0] else {
            panic!("expected the mint step first");
        };
        assert_eq!(args.piv_slot, None);
        assert!(args.yubikey_recipients.is_empty());
        assert_eq!(args.recipients, vec!["age1primary", "age1backup"]);

        let TrustCommand::DelegateRegistrySigner(args) = &commands[1] else {
            panic!("expected the delegation step second");
        };
        assert!(args.yubikey_identities.is_empty());
        assert_eq!(args.identities, vec![paths().primary_identity()]);
    }

    #[test]
    fn the_ceremony_runs_the_documented_four_steps_in_order() {
        let commands = ceremony_commands(&inputs(
            CeremonyMode::HardwareAgeRecipient { token: old_token() },
            "age1yubikey1primary",
            BreakGlass::SecondToken {
                recipient: "age1yubikey1backup".to_owned(),
            },
        ))
        .expect("commands");
        assert_eq!(commands.len(), 4);
        assert!(matches!(commands[0], TrustCommand::MintDeploymentCa(_)));
        assert!(matches!(
            commands[1],
            TrustCommand::DelegateRegistrySigner(_)
        ));
        assert!(matches!(commands[2], TrustCommand::MintRegistryJwt(_)));
        assert!(matches!(commands[3], TrustCommand::VerifyDeployment(_)));
    }

    #[test]
    fn the_credential_is_minted_through_the_delegated_signer_not_the_root() {
        let commands = ceremony_commands(&inputs(
            CeremonyMode::HardwareAgeRecipient { token: old_token() },
            "age1yubikey1primary",
            BreakGlass::SecondToken {
                recipient: "age1yubikey1backup".to_owned(),
            },
        ))
        .expect("commands");
        let TrustCommand::MintRegistryJwt(args) = &commands[2] else {
            panic!("expected the credential step third");
        };
        assert!(!args.root);
        assert_eq!(
            args.via_delegated_signer.as_deref(),
            Some(paths().delegated_signer().as_path())
        );
        assert!(args.yubikey_identities.is_empty());
        assert_eq!(args.ttl_seconds, REGISTRY_JWT_TTL_SECONDS);
    }

    #[test]
    fn a_hardware_ceremony_refuses_to_build_around_a_plaintext_break_glass() {
        let error = ceremony_commands(&inputs(
            CeremonyMode::HardwarePiv {
                token: new_token(),
                slot: DEFAULT_PIV_SLOT.to_owned(),
            },
            "age1yubikey1primary",
            BreakGlass::PlaintextIdentity {
                identity_file: PathBuf::from("/srv/ceremony/break-glass.key"),
                recipient: "age1backup".to_owned(),
            },
        ))
        .expect_err("a hardware root must not accept a bare backup identity");
        assert!(error.to_string().contains("weakest recipient"));
    }

    // ── Emitted metadata ────────────────────────────────────────────────────

    #[test]
    fn the_mode_record_labels_a_software_root_as_dev_grade() {
        let record = mode_record(
            &CeremonyMode::SoftwareDevGrade,
            &BreakGlass::PlaintextIdentity {
                identity_file: PathBuf::from("/srv/ceremony/break-glass.key"),
                recipient: "age1backup".to_owned(),
            },
            "no hardware token is attached",
        );
        assert!(record.dev_grade);
        assert_eq!(record.grade, DEV_GRADE_LABEL);
        assert!(!record.ed25519_confined_to_hardware);
        assert_eq!(record.token_serial, None);
        assert_eq!(record.piv_slot, None);
        assert!(!record.break_glass_protected);
    }

    #[test]
    fn the_mode_record_keeps_the_token_and_slot_for_a_hardware_root() {
        let record = mode_record(
            &CeremonyMode::HardwarePiv {
                token: new_token(),
                slot: DEFAULT_PIV_SLOT.to_owned(),
            },
            &BreakGlass::SecondToken {
                recipient: "age1yubikey1backup".to_owned(),
            },
            "firmware 5.7.4",
        );
        assert!(!record.dev_grade);
        assert!(record.ed25519_confined_to_hardware);
        assert_eq!(record.token_firmware.as_deref(), Some("5.7.4"));
        assert_eq!(record.token_serial.as_deref(), Some("22222222"));
        assert_eq!(record.piv_slot.as_deref(), Some(DEFAULT_PIV_SLOT));
        assert_eq!(record.schema, CEREMONY_MODE_SCHEMA);
        assert!(record.break_glass_protected);
    }

    #[test]
    fn the_mode_record_round_trips_as_json() {
        let record = mode_record(
            &CeremonyMode::HardwareAgeRecipient { token: old_token() },
            &BreakGlass::SecondToken {
                recipient: "age1yubikey1backup".to_owned(),
            },
            "firmware 5.4.3",
        );
        let json = serde_json::to_string(&record).expect("serialize");
        let parsed: CeremonyModeRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, record);
    }

    // ── Tooling ─────────────────────────────────────────────────────────────

    #[test]
    fn each_mode_declares_the_tools_it_needs() {
        assert!(required_tools(&CeremonyMode::HardwarePiv {
            token: new_token(),
            slot: DEFAULT_PIV_SLOT.to_owned(),
        })
        .contains(&"yubico-piv-tool"));
        assert_eq!(
            required_tools(&CeremonyMode::HardwareAgeRecipient { token: old_token() }),
            &["age-plugin-yubikey"]
        );
        assert!(required_tools(&CeremonyMode::SoftwareDevGrade).contains(&"age-keygen"));
    }

    #[test]
    fn missing_tools_are_reported_through_the_probe_seam() {
        let tools = required_tools(&CeremonyMode::HardwarePiv {
            token: new_token(),
            slot: DEFAULT_PIV_SLOT.to_owned(),
        });
        assert!(missing_tools(tools, |_| true).is_empty());
        assert_eq!(missing_tools(tools, |_| false).len(), tools.len());
        assert_eq!(
            missing_tools(tools, |tool| tool != "yubico-piv-tool"),
            vec!["yubico-piv-tool".to_owned()]
        );
    }

    // ── Output parsing ──────────────────────────────────────────────────────

    #[test]
    fn ykman_output_parses_into_tokens() {
        let listed = "YubiKey 5C NFC (5.4.3) [OTP+FIDO+CCID] Serial: 11111111\n\
                      YubiKey 5 NFC (5.7.4) [OTP+FIDO+CCID] Serial: 22222222\n";
        let tokens = parse_ykman_list(listed).expect("parse");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].model, "YubiKey 5C NFC");
        assert_eq!(tokens[0].firmware, FirmwareVersion::new(5, 4, 3));
        assert_eq!(tokens[1].serial, "22222222");
        assert!(tokens[1].firmware.supports_piv_ed25519());
    }

    #[test]
    fn empty_ykman_output_means_nothing_is_attached() {
        assert!(parse_ykman_list("").expect("parse").is_empty());
        assert!(parse_ykman_list("\n  \n").expect("parse").is_empty());
    }

    #[test]
    fn an_unreadable_ykman_line_is_an_error_not_a_missing_token() {
        for listed in [
            "YubiKey 5C NFC [OTP+FIDO+CCID] Serial: 11111111",
            "YubiKey 5C NFC (5.4) [OTP] Serial: 11111111",
            "YubiKey 5C NFC (5.4.3) [OTP]",
            "YubiKey 5C NFC (5.4.3) [OTP] Serial: none",
        ] {
            assert!(
                parse_ykman_list(listed).is_err(),
                "silently dropped token from {listed:?}"
            );
        }
    }

    #[test]
    fn plugin_and_keygen_recipients_are_told_apart() {
        assert_eq!(
            parse_age_plugin_recipient("Serial: 1, Slot: 1\n  age1yubikey1qabc\n").as_deref(),
            Some("age1yubikey1qabc")
        );
        assert_eq!(parse_age_plugin_recipient("no recipients here"), None);
        assert_eq!(
            parse_age_keygen_recipient("Public key: age1qsoftware").as_deref(),
            Some("age1qsoftware")
        );
        assert_eq!(parse_age_keygen_recipient("Public key: age1yubikey1q"), None);
    }
}
