use std::str::FromStr;
use tcl_mcp_server::tcl_runtime::{RuntimeConfig, RuntimeType};

#[cfg(test)]
mod runtime_selection_tests {
    use super::*;

    #[test]
    fn test_runtime_type_parsing() {
        assert_eq!(RuntimeType::from_str("molt").unwrap(), RuntimeType::Molt);
        assert_eq!(RuntimeType::from_str("MOLT").unwrap(), RuntimeType::Molt);
        assert_eq!(RuntimeType::from_str("Molt").unwrap(), RuntimeType::Molt);
        assert_eq!(RuntimeType::from_str("tcl").unwrap(), RuntimeType::Tcl);
        assert_eq!(RuntimeType::from_str("TCL").unwrap(), RuntimeType::Tcl);
        assert_eq!(RuntimeType::from_str("Tcl").unwrap(), RuntimeType::Tcl);

        assert!(RuntimeType::from_str("invalid").is_err());
        assert!(RuntimeType::from_str("javascript").is_err());
        assert!(RuntimeType::from_str("").is_err());
    }

    #[test]
    fn test_runtime_type_string_conversion() {
        assert_eq!(RuntimeType::Molt.as_str(), "molt");
        assert_eq!(RuntimeType::Tcl.as_str(), "tcl");
    }

    #[test]
    fn test_runtime_type_availability() {
        #[cfg(feature = "molt")]
        assert!(RuntimeType::Molt.is_available());

        #[cfg(not(feature = "molt"))]
        assert!(!RuntimeType::Molt.is_available());

        #[cfg(feature = "tcl")]
        assert!(RuntimeType::Tcl.is_available());

        #[cfg(not(feature = "tcl"))]
        assert!(!RuntimeType::Tcl.is_available());
    }

    #[test]
    fn test_config_from_args_and_env() {
        // CLI takes precedence over environment
        let config = RuntimeConfig::from_args_and_env(Some("tcl"), Some("molt")).unwrap();
        assert_eq!(config.runtime_type, Some(RuntimeType::Tcl));
        assert!(config.fallback_enabled);

        // Environment used when no CLI
        let config = RuntimeConfig::from_args_and_env(None, Some("tcl")).unwrap();
        assert_eq!(config.runtime_type, Some(RuntimeType::Tcl));

        // Default used when neither specified
        let config = RuntimeConfig::from_args_and_env(None, None).unwrap();
        // Default should be None (auto-select at runtime)
        assert_eq!(config.runtime_type, None);
    }

    #[test]
    fn test_config_invalid_runtime() {
        let result = RuntimeConfig::from_args_and_env(Some("invalid"), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid runtime type"));
    }

    #[test]
    fn test_available_runtimes() {
        let available = RuntimeConfig::available_runtimes();

        #[cfg(feature = "molt")]
        assert!(available.contains(&RuntimeType::Molt));

        #[cfg(not(feature = "molt"))]
        assert!(!available.contains(&RuntimeType::Molt));

        #[cfg(feature = "tcl")]
        assert!(available.contains(&RuntimeType::Tcl));

        #[cfg(not(feature = "tcl"))]
        assert!(!available.contains(&RuntimeType::Tcl));
    }

    #[test]
    fn test_runtime_creation_with_config() {
        // Test creating runtime with available types
        let available = RuntimeConfig::available_runtimes();

        for runtime_type in available {
            let config = RuntimeConfig {
                runtime_type: Some(runtime_type.clone()),
                fallback_enabled: false,
            };

            let result = tcl_mcp_server::tcl_runtime::create_runtime_with_config(config);
            assert!(
                result.is_ok(),
                "Failed to create {} runtime",
                runtime_type.as_str()
            );

            let runtime = result.unwrap();
            assert_eq!(
                runtime
                    .name()
                    .to_lowercase()
                    .contains(runtime_type.as_str()),
                true
            );
        }
    }

    #[test]
    fn test_runtime_creation_fallback() {
        // Test fallback behavior when primary runtime is unavailable
        #[cfg(all(feature = "molt", feature = "tcl"))]
        {
            // If both are available, no fallback should be needed
            let config = RuntimeConfig {
                runtime_type: RuntimeType::Molt,
                fallback_enabled: true,
            };

            let result = tcl_mcp_server::tcl_runtime::create_runtime_with_fallback(&config);
            assert!(result.is_ok());

            let runtime = result.unwrap();
            assert_eq!(runtime.name(), "Molt");
        }
    }

    #[test]
    fn test_runtime_creation_unavailable() {
        // Test error when runtime is not available
        #[cfg(not(feature = "molt"))]
        {
            let config = RuntimeConfig {
                runtime_type: RuntimeType::Molt,
                fallback_enabled: false,
            };

            let result = tcl_mcp_server::tcl_runtime::create_runtime_with_config(config);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("Molt runtime not available"));
        }

        #[cfg(not(feature = "tcl"))]
        {
            let config = RuntimeConfig {
                runtime_type: Some(RuntimeType::Tcl),
                fallback_enabled: false,
            };

            let result = tcl_mcp_server::tcl_runtime::create_runtime_with_config(config);
            assert!(result.is_err());
            // The error should be about TCL runtime not being available
            match result {
                Err(e) => assert!(e.to_string().contains("TCL runtime not available")),
                _ => panic!("Expected error when TCL runtime not available"),
            }
        }
    }

    #[test]
    fn test_runtime_features_and_version() {
        let available = RuntimeConfig::available_runtimes();

        for runtime_type in available {
            let config = RuntimeConfig {
                runtime_type: Some(runtime_type.clone()),
                fallback_enabled: false,
            };

            let runtime = tcl_mcp_server::tcl_runtime::create_runtime_with_config(config).unwrap();

            // All runtimes should have a name
            assert!(!runtime.name().is_empty());

            // Features should be non-empty for both implementations
            assert!(!runtime.features().is_empty());

            match runtime_type {
                RuntimeType::Molt => {
                    assert_eq!(runtime.name(), "Molt");
                    assert!(runtime.features().contains(&"safe_subset".to_string()));
                    assert!(!runtime.version().is_empty());
                }
                RuntimeType::Tcl => {
                    assert_eq!(runtime.name(), "TCL (Official)");
                    assert!(runtime.features().contains(&"full_tcl_8_6".to_string()));
                    // Version might not be available in test environment
                }
            }
        }
    }
}
