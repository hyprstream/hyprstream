//! WizardBackend trait and MockWizardBackend for WASI/testing.

// Build-time libtorch metadata (emitted by build.rs).
pub const LIBTORCH_VERSION: &str = env!("LIBTORCH_VERSION");
pub const LIBTORCH_ABI: &str = env!("LIBTORCH_ABI");
pub const LIBTORCH_VARIANT: &str = env!("LIBTORCH_VARIANT");

/// GPU hardware detected on the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuKind {
    /// NVIDIA GPU with driver version and compatible CUDA version.
    Nvidia {
        driver_version: String,
        cuda_compat: String,
    },
    /// AMD GPU with optional ROCm version.
    Amd { rocm_version: Option<String> },
    /// No GPU detected.
    None,
}

/// libtorch variant identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LibtorchVariant {
    Cpu,
    Cuda128,
    Cuda130,
    Rocm71,
}

impl LibtorchVariant {
    /// Human-readable label for display.
    pub fn label(&self) -> &str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda128 => "CUDA 12.8",
            Self::Cuda130 => "CUDA 13.0",
            Self::Rocm71 => "ROCm 7.1",
        }
    }

    /// Short identifier used in paths and config.
    pub fn id(&self) -> &str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda128 => "cuda128",
            Self::Cuda130 => "cuda130",
            Self::Rocm71 => "rocm71",
        }
    }

    /// All available variants.
    pub fn all() -> &'static [LibtorchVariant] {
        &[Self::Cpu, Self::Cuda128, Self::Cuda130, Self::Rocm71]
    }

    /// Parse from the compile-time variant string.
    pub fn from_compiled() -> Self {
        match LIBTORCH_VARIANT {
            "cuda128" => Self::Cuda128,
            "cuda130" => Self::Cuda130,
            "rocm71" => Self::Rocm71,
            _ => Self::Cpu,
        }
    }

    /// Parse from a variant ID string.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "cpu" => Some(Self::Cpu),
            "cuda128" => Some(Self::Cuda128),
            "cuda130" => Some(Self::Cuda130),
            "rocm71" => Some(Self::Rocm71),
            _ => None,
        }
    }
}

impl core::fmt::Display for LibtorchVariant {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.label())
    }
}

/// How hyprstream is currently running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunMode {
    /// Running from AppImage (self-contained, may manage its own backends).
    AppImage,
    /// Standalone binary (cargo install, manual install, etc.).
    BareBinary,
    /// Running from cargo in dev (LIBTORCH set, skip install).
    Development,
}

/// Snapshot of the runtime environment.
#[derive(Debug, Clone)]
pub struct EnvironmentInfo {
    pub run_mode: RunMode,
    pub gpu: GpuKind,
    pub libtorch_path: Option<String>,
    pub current_variant: LibtorchVariant,
    pub recommended_variant: LibtorchVariant,
    pub installed_variant: Option<LibtorchVariant>,
}

/// Action the wizard recommends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallAction {
    /// Nothing to do — skip. Reason displayed to user.
    Skip { reason: String },
    /// Download GPU binary + libtorch variant.
    UpgradeVariant(LibtorchVariant),
    /// Already running the optimal variant.
    AlreadyCurrent,
}

/// Progress during install/download.
#[derive(Debug, Clone)]
pub enum InstallPoll {
    Detecting,
    Downloading { item: String, pct: u8 },
    Extracting { item: String },
    Configuring { step: String },
    Done { summary: String },
    Failed(String),
}

/// Status of an async bootstrap step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapPoll {
    /// Bootstrap still running. Message describes current step.
    InProgress(String),
    /// Bootstrap finished successfully. Steps completed.
    Done(Vec<String>),
    /// Bootstrap failed.
    Failed(String),
}

/// Result of generating a token.
#[derive(Debug, Clone)]
pub struct TokenResult {
    pub token: String,
    pub expires: String,
}

/// Information about a policy template.
#[derive(Debug, Clone)]
pub struct TemplateInfo {
    pub name: String,
    pub description: String,
}

/// Status for polling async operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpStatus {
    InProgress,
    Done,
    Failed(String),
}

/// Backend trait abstracting real hyprstream operations from the TUI.
///
/// `MockWizardBackend` provides canned responses for WASI/browser and testing.
/// The native `BootstrapManager` (in hyprstream) bridges to the real runtime.
pub trait WizardBackend {
    // Install phase
    fn detect_environment(&mut self) -> EnvironmentInfo;
    fn recommend_action(&self, env: &EnvironmentInfo) -> InstallAction;
    fn start_install(&mut self, variant: &LibtorchVariant);
    fn poll_install(&mut self) -> InstallPoll;

    // Bootstrap phase
    fn start_bootstrap(&mut self);
    fn poll_bootstrap(&mut self) -> BootstrapPoll;
    fn has_existing_policy(&self) -> bool;
    fn apply_template(&mut self, name: &str);
    fn add_user(&mut self, username: &str, role: &str);
    fn add_user_custom(&mut self, username: &str, resource: &str, actions: &[String]);
    fn save_policies(&mut self);
    fn generate_token(&mut self, username: &str, duration: &str) -> TokenResult;
    fn start_services(&mut self);
    fn poll_pending(&mut self) -> OpStatus;
    fn local_username(&self) -> String;
    fn templates(&self) -> Vec<TemplateInfo>;
}

/// Mock backend with canned responses for WASI/browser and testing.
pub struct MockWizardBackend {
    bootstrap_tick: u32,
    bootstrap_started: bool,
    service_tick: u32,
    service_started: bool,
    install_tick: u32,
    install_started: bool,
    detect_tick: u32,
    detect_started: bool,
    mock_run_mode: RunMode,
    mock_gpu: GpuKind,
}

impl MockWizardBackend {
    pub fn new() -> Self {
        Self {
            bootstrap_tick: 0,
            bootstrap_started: false,
            service_tick: 0,
            service_started: false,
            install_tick: 0,
            install_started: false,
            detect_tick: 0,
            detect_started: false,
            mock_run_mode: RunMode::BareBinary,
            mock_gpu: GpuKind::Nvidia {
                driver_version: "555.42".to_string(),
                cuda_compat: "cuda130".to_string(),
            },
        }
    }

    /// Create a mock with AppImage run mode (auto-skips install).
    pub fn with_run_mode(mut self, mode: RunMode) -> Self {
        self.mock_run_mode = mode;
        self
    }

    /// Override the mock GPU detection.
    pub fn with_gpu(mut self, gpu: GpuKind) -> Self {
        self.mock_gpu = gpu;
        self
    }
}

impl Default for MockWizardBackend {
    fn default() -> Self {
        Self::new()
    }
}

const BOOTSTRAP_STEPS: &[&str] = &[
    "Checking directories...",
    "Verifying registry...",
    "Loading signing key...",
    "Validating environment...",
];

impl WizardBackend for MockWizardBackend {
    fn detect_environment(&mut self) -> EnvironmentInfo {
        self.detect_started = true;
        self.detect_tick = 0;
        let recommended = match &self.mock_gpu {
            GpuKind::Nvidia { cuda_compat, .. } => match cuda_compat.as_str() {
                "cuda128" => LibtorchVariant::Cuda128,
                "cuda130" => LibtorchVariant::Cuda130,
                _ => LibtorchVariant::Cpu,
            },
            GpuKind::Amd { rocm_version } => {
                if rocm_version.is_some() {
                    LibtorchVariant::Rocm71
                } else {
                    LibtorchVariant::Cpu
                }
            }
            GpuKind::None => LibtorchVariant::Cpu,
        };
        EnvironmentInfo {
            run_mode: self.mock_run_mode.clone(),
            gpu: self.mock_gpu.clone(),
            libtorch_path: None,
            current_variant: LibtorchVariant::from_compiled(),
            recommended_variant: recommended,
            installed_variant: None,
        }
    }

    fn recommend_action(&self, env: &EnvironmentInfo) -> InstallAction {
        if env.run_mode == RunMode::Development {
            return InstallAction::Skip {
                reason: "Development mode — LIBTORCH already configured".to_string(),
            };
        }
        if env.run_mode == RunMode::AppImage {
            return InstallAction::Skip {
                reason: "AppImage manages its own backends".to_string(),
            };
        }
        if env.current_variant == env.recommended_variant {
            return InstallAction::AlreadyCurrent;
        }
        if let Some(ref installed) = env.installed_variant {
            if *installed == env.recommended_variant {
                return InstallAction::AlreadyCurrent;
            }
        }
        InstallAction::UpgradeVariant(env.recommended_variant.clone())
    }

    fn start_install(&mut self, _variant: &LibtorchVariant) {
        self.install_started = true;
        self.install_tick = 0;
    }

    fn poll_install(&mut self) -> InstallPoll {
        if !self.install_started {
            return InstallPoll::Failed("Install not started".to_string());
        }
        self.install_tick += 1;
        match self.install_tick {
            1..=20 => InstallPoll::Downloading {
                item: "libtorch".to_string(),
                pct: ((self.install_tick as u16) * 5).min(100) as u8,
            },
            21 => InstallPoll::Extracting {
                item: "libtorch".to_string(),
            },
            22..=32 => InstallPoll::Downloading {
                item: "hyprstream".to_string(),
                pct: (((self.install_tick - 22) as u16) * 10).min(100) as u8,
            },
            33 => InstallPoll::Configuring {
                step: "Setting up LD_LIBRARY_PATH".to_string(),
            },
            _ => {
                self.install_started = false;
                InstallPoll::Done {
                    summary: "GPU backend installed successfully".to_string(),
                }
            }
        }
    }

    fn start_bootstrap(&mut self) {
        self.bootstrap_started = true;
        self.bootstrap_tick = 0;
    }

    fn poll_bootstrap(&mut self) -> BootstrapPoll {
        if !self.bootstrap_started {
            return BootstrapPoll::InProgress("Not started".to_string());
        }
        self.bootstrap_tick += 1;
        let step_idx = (self.bootstrap_tick / 8) as usize; // ~8 ticks per step at 30fps
        if step_idx >= BOOTSTRAP_STEPS.len() {
            BootstrapPoll::Done(vec![
                "Directories OK".to_string(),
                "Registry OK".to_string(),
                "Signing key OK".to_string(),
                "Environment OK".to_string(),
            ])
        } else {
            BootstrapPoll::InProgress(BOOTSTRAP_STEPS[step_idx].to_string())
        }
    }

    fn has_existing_policy(&self) -> bool {
        false
    }

    fn apply_template(&mut self, _name: &str) {
        // Mock: instant success
    }

    fn add_user(&mut self, _username: &str, _role: &str) {
        // Mock: instant success
    }

    fn add_user_custom(&mut self, _username: &str, _resource: &str, _actions: &[String]) {
        // Mock: instant success
    }

    fn save_policies(&mut self) {
        // Mock: instant success
    }

    fn generate_token(&mut self, username: &str, duration: &str) -> TokenResult {
        TokenResult {
            token: format!("hypr_{username}_mock_token_xxxxxxxxxxxx"),
            expires: match duration {
                "30d" => "2026-04-01".to_string(),
                "1y" => "2027-03-02".to_string(),
                "never" => "never".to_string(),
                _ => "2026-06-01".to_string(),
            },
        }
    }

    fn start_services(&mut self) {
        self.service_started = true;
        self.service_tick = 0;
    }

    fn poll_pending(&mut self) -> OpStatus {
        if !self.service_started {
            return OpStatus::Done;
        }
        self.service_tick += 1;
        if self.service_tick > 15 {
            self.service_started = false;
            OpStatus::Done
        } else {
            OpStatus::InProgress
        }
    }

    fn local_username(&self) -> String {
        "localuser".to_string()
    }

    fn templates(&self) -> Vec<TemplateInfo> {
        vec![
            TemplateInfo {
                name: "local".to_string(),
                description: "Full access for local user (recommended)".to_string(),
            },
            TemplateInfo {
                name: "team".to_string(),
                description: "Multi-user with admin, operator, viewer roles".to_string(),
            },
            TemplateInfo {
                name: "production".to_string(),
                description: "Locked-down production environment".to_string(),
            },
        ]
    }
}
