use anyhow::{anyhow, Result};
use std::env;

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeType {
    Molt,
    Tcl,
}

impl std::str::FromStr for RuntimeType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "molt" => Ok(RuntimeType::Molt),
            "tcl" => Ok(RuntimeType::Tcl),
            _ => Err(anyhow!(
                "Invalid runtime type '{}'. Valid options: molt, tcl",
                s
            )),
        }
    }
}

impl RuntimeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RuntimeType::Molt => "molt",
            RuntimeType::Tcl => "tcl",
        }
    }

    /// Check if this runtime type is available based on compiled features
    pub fn is_available(&self) -> bool {
        match self {
            RuntimeType::Molt => cfg!(feature = "molt"),
            RuntimeType::Tcl => cfg!(feature = "tcl"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub runtime_type: Option<RuntimeType>,
    pub fallback_enabled: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            runtime_type: None,
            fallback_enabled: true,
        }
    }
}

impl RuntimeConfig {
    /// Get list of available runtimes  
    pub fn available_runtimes() -> Vec<RuntimeType> {
        available_runtimes()
    }

    /// Create runtime config from CLI args and environment variables
    pub fn from_args_and_env(
        cli_runtime: Option<&str>,
        env_runtime: Option<&str>, // Environment variable value
    ) -> Result<Self> {
        let mut config = RuntimeConfig::default();

        // Check environment variable first
        if let Some(env_runtime) = env_runtime {
            config.runtime_type = Some(env_runtime.parse()?);
        }

        // CLI argument overrides environment
        if let Some(cli_runtime) = cli_runtime {
            config.runtime_type = Some(cli_runtime.parse()?);
        }

        Ok(config)
    }
}

/// Trait defining the interface for TCL runtime implementations
pub trait TclRuntime {
    /// Create a new instance of the TCL runtime
    fn new() -> Self
    where
        Self: Sized;

    /// Evaluate a TCL script and return the result
    fn eval(&mut self, script: &str) -> Result<String>;

    /// Set a variable in the TCL runtime
    fn set_var(&mut self, name: &str, value: &str) -> Result<()>;

    /// Get a variable from the TCL runtime
    fn get_var(&self, name: &str) -> Result<String>;

    /// Check if the runtime supports a specific command
    fn has_command(&self, command: &str) -> bool;

    /// Get runtime name for logging/debugging
    fn name(&self) -> &'static str;

    /// Get runtime version
    fn version(&self) -> &'static str;

    /// Get runtime features/capabilities
    fn features(&self) -> Vec<String>;

    /// Check if runtime is safe/sandboxed
    fn is_safe(&self) -> bool;
}

#[cfg(feature = "molt")]
mod molt_runtime;
#[cfg(feature = "molt")]
pub use molt_runtime::MoltRuntime;

#[cfg(feature = "tcl")]
mod tcl_interpreter;
#[cfg(feature = "tcl")]
pub use tcl_interpreter::TclInterpreter;

/// Check if a runtime type is available at compile time
pub fn is_runtime_available(runtime_type: RuntimeType) -> bool {
    runtime_type.is_available()
}

/// Get list of available runtimes
pub fn available_runtimes() -> Vec<RuntimeType> {
    let mut runtimes = Vec::new();
    if cfg!(feature = "molt") {
        runtimes.push(RuntimeType::Molt);
    }
    if cfg!(feature = "tcl") {
        runtimes.push(RuntimeType::Tcl);
    }
    runtimes
}

/// Create runtime with specific configuration
pub fn create_runtime_with_config(config: RuntimeConfig) -> Result<Box<dyn TclRuntime>> {
    if let Some(requested_type) = config.runtime_type {
        // Try to create the requested runtime
        match create_specific_runtime(requested_type.clone()) {
            Ok(runtime) => {
                tracing::info!("Using {} TCL runtime", requested_type.as_str());
                return Ok(runtime);
            }
            Err(e) if config.fallback_enabled => {
                tracing::warn!(
                    "Failed to create requested runtime {:?}: {}. Trying fallback.",
                    requested_type,
                    e
                );
                // Fall through to auto-selection
            }
            Err(e) => return Err(e),
        }
    }

    // Auto-select based on available features (prefer C TCL for full compatibility)
    #[cfg(feature = "tcl")]
    {
        tracing::info!("Auto-selecting official C TCL runtime");
        return Ok(Box::new(TclInterpreter::new()));
    }

    #[cfg(all(feature = "molt", not(feature = "tcl")))]
    {
        tracing::info!("Auto-selecting Molt TCL runtime");
        return Ok(Box::new(MoltRuntime::new()));
    }

    #[cfg(all(not(feature = "molt"), not(feature = "tcl")))]
    {
        return Err(anyhow!(
            "No TCL runtime features enabled. Please build with --features molt or --features tcl"
        ));
    }
}

/// Create a specific runtime type
fn create_specific_runtime(runtime_type: RuntimeType) -> Result<Box<dyn TclRuntime>> {
    match runtime_type {
        RuntimeType::Molt => {
            #[cfg(feature = "molt")]
            {
                Ok(Box::new(MoltRuntime::new()))
            }
            #[cfg(not(feature = "molt"))]
            {
                Err(anyhow!(
                    "Molt runtime not available. Build with --features molt"
                ))
            }
        }
        RuntimeType::Tcl => {
            #[cfg(feature = "tcl")]
            {
                Ok(Box::new(TclInterpreter::new()))
            }
            #[cfg(not(feature = "tcl"))]
            {
                Err(anyhow!(
                    "TCL runtime not available. Build with --features tcl"
                ))
            }
        }
    }
}

/// Get list of available runtime types based on compiled features
pub fn get_available_runtimes() -> Vec<RuntimeType> {
    let mut runtimes = Vec::new();

    #[cfg(feature = "molt")]
    runtimes.push(RuntimeType::Molt);

    #[cfg(feature = "tcl")]
    runtimes.push(RuntimeType::Tcl);

    runtimes
}

/// Create runtime from environment and CLI arguments
pub fn create_runtime_from_env(cli_runtime: Option<&str>) -> Result<Box<dyn TclRuntime>> {
    let mut config = RuntimeConfig::default();

    // Check environment variable first
    if let Ok(env_runtime) = env::var("TCL_MCP_RUNTIME") {
        config.runtime_type = Some(env_runtime.parse()?);
    }

    // CLI argument overrides environment
    if let Some(cli_runtime) = cli_runtime {
        config.runtime_type = Some(cli_runtime.parse()?);
    }

    create_runtime_with_config(config)
}

/// Factory function to create the appropriate runtime based on features (backward compatibility)
pub fn create_runtime() -> Box<dyn TclRuntime> {
    create_runtime_with_config(RuntimeConfig::default()).expect("Failed to create default runtime")
}
