use hyprstream_core::cli::commands::config::LoggingConfig;

#[test]
fn test_logging_config() {
    // Test default values
    let default_config = LoggingConfig::default();
    assert_eq!(default_config.verbose, 0);
    assert_eq!(default_config.get_effective_level(), "info");
    assert!(default_config.log_level.is_none());
    assert!(default_config.log_filter.is_none());

    // Test -v flag (debug level)
    let debug_config = LoggingConfig {
        verbose: 1,
        log_level: None,
        log_filter: None,
    };
    assert_eq!(debug_config.get_effective_level(), "debug");

    // Test -vv flag (trace level)
    let trace_config = LoggingConfig {
        verbose: 2,
        log_level: None,
        log_filter: None,
    };
    assert_eq!(trace_config.get_effective_level(), "trace");

    // Test explicit log level
    let explicit_config = LoggingConfig {
        verbose: 0,
        log_level: Some("warn".to_string()),
        log_filter: None,
    };
    assert_eq!(explicit_config.get_effective_level(), "warn");

    // Test that -v/-vv overrides explicit level
    let override_config = LoggingConfig {
        verbose: 2,
        log_level: Some("warn".to_string()),
        log_filter: None,
    };
    assert_eq!(override_config.get_effective_level(), "trace");

    // Test with filter
    let filter_config = LoggingConfig {
        verbose: 1,
        log_level: None,
        log_filter: Some("hyprstream=debug".to_string()),
    };
    assert_eq!(filter_config.get_effective_level(), "debug");
    assert_eq!(filter_config.log_filter.as_deref(), Some("hyprstream=debug"));
}