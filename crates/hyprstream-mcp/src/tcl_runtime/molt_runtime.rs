use super::TclRuntime;
use anyhow::{anyhow, Result};
use molt::Interp;

/// Molt TCL interpreter implementation
pub struct MoltRuntime {
    interp: Interp,
}

impl TclRuntime for MoltRuntime {
    fn new() -> Self {
        Self {
            interp: Interp::new(),
        }
    }

    fn eval(&mut self, script: &str) -> Result<String> {
        match self.interp.eval(script) {
            Ok(value) => Ok(value.to_string()),
            Err(error) => Err(anyhow!("Molt execution error: {:?}", error)),
        }
    }

    fn set_var(&mut self, name: &str, value: &str) -> Result<()> {
        match self.interp.set_scalar(name, molt::Value::from(value)) {
            Ok(_) => Ok(()),
            Err(error) => Err(anyhow!("Failed to set variable '{}': {:?}", name, error)),
        }
    }

    fn get_var(&self, name: &str) -> Result<String> {
        match self.interp.scalar(name) {
            Ok(value) => Ok(value.to_string()),
            Err(error) => Err(anyhow!("Failed to get variable '{}': {:?}", name, error)),
        }
    }

    fn has_command(&self, command: &str) -> bool {
        self.interp.has_command(command)
    }

    fn name(&self) -> &'static str {
        "Molt"
    }

    fn version(&self) -> &'static str {
        "0.3.1" // Molt version
    }

    fn features(&self) -> Vec<String> {
        vec![
            "safe_subset".to_string(),
            "memory_safe".to_string(),
            "no_file_io".to_string(),
            "basic_math".to_string(),
            "string_manipulation".to_string(),
            "list_operations".to_string(),
            "control_flow".to_string(),
            "procedures".to_string(),
        ]
    }

    fn is_safe(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_molt_runtime_eval() {
        let mut runtime = MoltRuntime::new();
        let result = runtime.eval("expr {2 + 2}").unwrap();
        assert_eq!(result, "4");
    }

    #[test]
    fn test_molt_runtime_variables() {
        let mut runtime = MoltRuntime::new();
        runtime.set_var("test_var", "hello").unwrap();
        let value = runtime.get_var("test_var").unwrap();
        assert_eq!(value, "hello");
    }

    #[test]
    fn test_molt_runtime_has_command() {
        let runtime = MoltRuntime::new();
        assert!(runtime.has_command("expr"));
        assert!(runtime.has_command("set"));
        assert!(!runtime.has_command("nonexistent_command"));
    }

    #[test]
    fn test_molt_runtime_string_ops() {
        let mut runtime = MoltRuntime::new();
        runtime.set_var("text", "hello").unwrap();
        let result = runtime.eval("string length $text").unwrap();
        assert_eq!(result, "5");
    }
}
