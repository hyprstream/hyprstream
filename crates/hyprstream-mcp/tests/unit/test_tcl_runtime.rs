use anyhow::Result;

#[cfg(test)]
mod tests {
    use super::*;
    use tcl_mcp_server::tcl_runtime::{TclRuntime, create_runtime};

    #[test]
    fn test_runtime_creation() {
        let runtime = create_runtime();
        assert!(!runtime.name().is_empty());
    }

    #[test]
    fn test_runtime_eval_basic() -> Result<()> {
        let mut runtime = create_runtime();
        let result = runtime.eval("expr {2 + 2}")?;
        assert_eq!(result, "4");
        Ok(())
    }

    #[test]
    fn test_runtime_variable_operations() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Set a variable
        runtime.set_var("test_var", "hello world")?;
        
        // Get the variable
        let value = runtime.get_var("test_var")?;
        assert_eq!(value, "hello world");
        
        // Update the variable
        runtime.set_var("test_var", "updated value")?;
        let new_value = runtime.get_var("test_var")?;
        assert_eq!(new_value, "updated value");
        
        Ok(())
    }

    #[test]
    fn test_runtime_has_command() {
        let runtime = create_runtime();
        
        // Common TCL commands
        assert!(runtime.has_command("set"));
        assert!(runtime.has_command("expr"));
        assert!(runtime.has_command("if"));
        assert!(runtime.has_command("proc"));
        
        // Non-existent command
        assert!(!runtime.has_command("this_command_does_not_exist"));
    }

    #[test]
    fn test_runtime_eval_with_variables() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Set variables
        runtime.set_var("a", "10")?;
        runtime.set_var("b", "20")?;
        
        // Use them in expression
        let result = runtime.eval("expr {$a + $b}")?;
        assert_eq!(result, "30");
        
        Ok(())
    }

    #[test]
    fn test_runtime_string_operations() -> Result<()> {
        let mut runtime = create_runtime();
        
        runtime.set_var("text", "hello")?;
        let result = runtime.eval("string length $text")?;
        assert_eq!(result, "5");
        
        let upper = runtime.eval("string toupper $text")?;
        assert_eq!(upper, "HELLO");
        
        Ok(())
    }

    #[test]
    fn test_runtime_list_operations() -> Result<()> {
        let mut runtime = create_runtime();
        
        let result = runtime.eval("list apple banana orange")?;
        assert_eq!(result, "apple banana orange");
        
        runtime.set_var("fruits", "apple banana orange")?;
        let length = runtime.eval("llength $fruits")?;
        assert_eq!(length, "3");
        
        Ok(())
    }

    #[test]
    fn test_runtime_proc_definition() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Define a procedure
        runtime.eval("proc add {a b} { expr {$a + $b} }")?;
        
        // Call the procedure
        let result = runtime.eval("add 5 7")?;
        assert_eq!(result, "12");
        
        Ok(())
    }

    #[test]
    fn test_runtime_error_handling() {
        let mut runtime = create_runtime();
        
        // Invalid syntax
        let result = runtime.eval("expr {2 +");
        assert!(result.is_err());
        
        // Undefined variable
        let result = runtime.get_var("undefined_var");
        assert!(result.is_err());
        
        // Invalid command
        let result = runtime.eval("nonexistent_command arg1 arg2");
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_special_characters() -> Result<()> {
        let mut runtime = create_runtime();
        
        // Test with quotes
        runtime.set_var("quoted", "hello \"world\"")?;
        let value = runtime.get_var("quoted")?;
        assert_eq!(value, "hello \"world\"");
        
        // Test with newlines
        runtime.set_var("multiline", "line1\nline2")?;
        let multiline = runtime.get_var("multiline")?;
        assert_eq!(multiline, "line1\nline2");
        
        Ok(())
    }
}