#[cfg(test)]
mod feature_tests {
    use tcl_mcp_server::tcl_runtime::create_runtime;

    #[test]
    fn test_runtime_name() {
        let runtime = create_runtime();

        #[cfg(feature = "molt")]
        assert_eq!(runtime.name(), "Molt");

        #[cfg(feature = "tcl")]
        assert_eq!(runtime.name(), "TCL (Official)");
    }

    #[test]
    fn test_runtime_features_mutually_exclusive() {
        // This test ensures we have at least one runtime available
        let runtime = create_runtime();
        let name = runtime.name();
        assert!(!name.is_empty());
        assert!(name == "Molt" || name == "TCL (Official)");
    }
}
