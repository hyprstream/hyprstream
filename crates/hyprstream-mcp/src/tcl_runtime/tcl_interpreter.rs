use super::TclRuntime;
use anyhow::{anyhow, Result};

/// Official TCL interpreter implementation using the tcl crate
#[cfg(feature = "tcl")]
pub struct TclInterpreter {
    interp: tcl::Interpreter,
}

#[cfg(feature = "tcl")]
impl TclRuntime for TclInterpreter {
    fn new() -> Self {
        Self {
            interp: tcl::Interpreter::new().expect("Failed to create TCL interpreter"),
        }
    }

    fn eval(&mut self, script: &str) -> Result<String> {
        match self.interp.eval(script) {
            Ok(result) => Ok(result.to_string()),
            Err(err) => Err(anyhow!("TCL execution error: {}", err)),
        }
    }

    fn set_var(&mut self, name: &str, value: &str) -> Result<()> {
        let _result = self.interp.set(name, value);
        // TCL set always succeeds unless there's a serious error
        Ok(())
    }

    fn get_var(&self, name: &str) -> Result<String> {
        match self.interp.get(name) {
            Ok(value) => Ok(value.to_string()),
            Err(err) => Err(anyhow!("Failed to get variable '{}': {}", name, err)),
        }
    }

    fn has_command(&self, command: &str) -> bool {
        // Check if command exists by trying to get its info
        let check_cmd = format!("info commands {}", command);
        self.interp
            .eval(check_cmd)
            .map(|result| !result.to_string().is_empty())
            .unwrap_or(false)
    }

    fn name(&self) -> &'static str {
        "TCL (Official)"
    }

    fn version(&self) -> &'static str {
        "8.6"
    }

    fn features(&self) -> Vec<String> {
        vec![
            "full_tcl_8_6".to_string(),
            "file_operations".to_string(),
            "networking".to_string(),
            "regex".to_string(),
            "threading".to_string(),
            "packages".to_string(),
            "extensions".to_string(),
            "native_performance".to_string(),
        ]
    }

    fn is_safe(&self) -> bool {
        false // Full TCL has access to file system, exec, etc.
    }
}

#[cfg(not(feature = "tcl"))]
pub struct TclInterpreter;

#[cfg(not(feature = "tcl"))]
impl TclInterpreter {
    pub fn new() -> Self {
        panic!("TCL interpreter not available. Build with --features tcl");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcl_runtime_eval() {
        let mut runtime = TclInterpreter::new();
        let result = runtime.eval("expr {2 + 2}").unwrap();
        assert_eq!(result, "4");
    }

    #[test]
    fn test_tcl_runtime_variables() {
        let mut runtime = TclInterpreter::new();
        runtime.set_var("test_var", "hello").unwrap();
        let value = runtime.get_var("test_var").unwrap();
        assert_eq!(value, "hello");
    }

    #[test]
    fn test_tcl_runtime_has_command() {
        let runtime = TclInterpreter::new();
        assert!(runtime.has_command("expr"));
        assert!(runtime.has_command("set"));
        assert!(!runtime.has_command("nonexistent_command"));
    }

    #[test]
    fn test_tcl_runtime_string_ops() {
        let mut runtime = TclInterpreter::new();
        runtime.set_var("text", "hello").unwrap();
        let result = runtime.eval("string length $text").unwrap();
        assert_eq!(result, "5");
    }
}
