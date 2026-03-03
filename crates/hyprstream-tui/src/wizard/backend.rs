//! WizardBackend trait and MockWizardBackend for WASI/testing.

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
/// A future `RealWizardBackend` will bridge to the real hyprstream runtime.
pub trait WizardBackend {
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
}

impl MockWizardBackend {
    pub fn new() -> Self {
        Self {
            bootstrap_tick: 0,
            bootstrap_started: false,
            service_tick: 0,
            service_started: false,
        }
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
