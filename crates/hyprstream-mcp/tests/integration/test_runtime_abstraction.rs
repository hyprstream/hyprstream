use anyhow::Result;
use tcl_mcp_server::tcl_runtime::{TclRuntime, create_runtime};

#[cfg(test)]
mod runtime_abstraction_tests {
    use super::*;

    #[test]
    fn test_runtime_abstraction_eval() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Basic arithmetic
        let result = runtime.eval("expr {10 + 20}")?;
        assert_eq!(result, "30");
        
        // String operations
        let result = runtime.eval("string length \"hello world\"")?;
        assert_eq!(result, "11");
        
        Ok(())
    }
    
    #[test]
    fn test_runtime_abstraction_variables() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Set and get variables
        runtime.set_var("myvar", "test value")?;
        let value = runtime.get_var("myvar")?;
        assert_eq!(value, "test value");
        
        // Use variable in expression
        runtime.set_var("x", "5")?;
        runtime.set_var("y", "3")?;
        let result = runtime.eval("expr {$x * $y}")?;
        assert_eq!(result, "15");
        
        Ok(())
    }
    
    #[test]
    fn test_runtime_abstraction_procedures() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Define a procedure
        runtime.eval("proc multiply {a b} { expr {$a * $b} }")?;
        
        // Call the procedure
        let result = runtime.eval("multiply 4 5")?;
        assert_eq!(result, "20");
        
        Ok(())
    }
    
    #[test]
    fn test_runtime_abstraction_lists() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Create and manipulate lists
        runtime.eval("set fruits {apple banana orange}")?;
        
        let length = runtime.eval("llength $fruits")?;
        assert_eq!(length, "3");
        
        let first = runtime.eval("lindex $fruits 0")?;
        assert_eq!(first, "apple");
        
        Ok(())
    }
    
    #[test]
    fn test_runtime_abstraction_error_handling() {
        let mut runtime = create_runtime();
        
        // Test various error conditions
        assert!(runtime.eval("invalid syntax {").is_err());
        assert!(runtime.get_var("nonexistent").is_err());
        assert!(runtime.eval("undefined_command").is_err());
    }
    
    #[test]
    fn test_runtime_abstraction_has_command() {
        let runtime = create_runtime();
        
        // Standard TCL commands
        assert!(runtime.has_command("set"));
        assert!(runtime.has_command("expr"));
        assert!(runtime.has_command("proc"));
        assert!(runtime.has_command("if"));
        assert!(runtime.has_command("while"));
        assert!(runtime.has_command("foreach"));
        
        // Non-existent command
        assert!(!runtime.has_command("this_definitely_does_not_exist"));
    }
    
    #[test]
    fn test_runtime_abstraction_control_flow() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Test if statement
        runtime.set_var("x", "10")?;
        let result = runtime.eval("if {$x > 5} {expr {$x * 2}} else {expr {$x / 2}}")?;
        assert_eq!(result, "20");
        
        runtime.set_var("x", "4")?;
        let result = runtime.eval("if {$x > 5} {expr {$x * 2}} else {expr {$x / 2}}")?;
        assert_eq!(result, "2");
        
        Ok(())
    }
}