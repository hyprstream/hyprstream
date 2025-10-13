//! Demo of the explicit worktree strategy system
//!
//! Run with: cargo run --example worktree_strategies_demo --features overlayfs

use git2db::worktree::strategy_enum::{WorktreeStrategyType, WorktreeStrategyBuilder};
use git2db::config_v2::{WorktreeConfigV2, PlatformOverrides};
use std::env;
use colored::*;

fn main() {
    // Initialize logging
    env_logger::init();

    println!("{}", "=== Explicit Worktree Strategy Demo ===".bold().cyan());
    println!();

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let strategy_name = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match strategy_name {
        "help" | "--help" | "-h" => show_help(),
        "list" => list_strategies(),
        "check" => check_availability(),
        "config" => show_config_examples(),
        "migrate" => show_migration(),
        strategy => demo_strategy(strategy),
    }
}

fn show_help() {
    println!("{}", "Usage:".bold());
    println!("  cargo run --example worktree_strategies_demo [command]");
    println!();
    println!("{}", "Commands:".bold());
    println!("  help              Show this help message");
    println!("  list              List all available strategies");
    println!("  check             Check strategy availability on this system");
    println!("  config            Show configuration examples");
    println!("  migrate           Show config migration examples");
    println!("  <strategy>        Demo a specific strategy");
    println!();
    println!("{}", "Strategy Names:".bold());
    println!("  automatic         Auto-select best available");
    println!("  git               Native git worktrees");
    println!("  overlayfs         Linux overlayfs (auto backend)");
    println!("  overlayfs-kernel  Overlayfs with kernel backend");
    println!("  overlayfs-userns  Overlayfs with user namespace");
    println!("  overlayfs-fuse    Overlayfs with FUSE backend");
    println!("  prefer-overlayfs  Try overlayfs, fallback to git");
}

fn list_strategies() {
    println!("{}", "Available Worktree Strategies:".bold().green());
    println!();

    let all_strategies = vec![
        WorktreeStrategyType::Automatic,
        WorktreeStrategyType::Git,
        #[cfg(feature = "overlayfs")]
        WorktreeStrategyType::Overlayfs,
        #[cfg(feature = "overlayfs")]
        WorktreeStrategyType::OverlayfsKernel,
        #[cfg(feature = "overlayfs")]
        WorktreeStrategyType::OverlayfsUserns,
        #[cfg(feature = "overlayfs")]
        WorktreeStrategyType::OverlayfsFuse,
        #[cfg(feature = "overlayfs")]
        WorktreeStrategyType::PreferOverlayfs,
    ];

    for strategy in &all_strategies {
        let available = if strategy.is_available() {
            "✓".green().to_string()
        } else {
            "✗".red().to_string()
        };

        let name = format!("{}", strategy);
        let padded_name = format!("{:20}", name);

        println!(
            "  {} {} - {}",
            available,
            padded_name.bold(),
            strategy.description()
        );

        if !strategy.is_available() {
            if let Some(error) = strategy.availability_error() {
                println!("      {}", format!("Reason: {}", error).red());
            }
        }
    }

    println!();
    println!("{}", "Legend:".bold());
    println!("  {} Available on this system", "✓".green());
    println!("  {} Not available on this system", "✗".red());
}

fn check_availability() {
    println!("{}", "System Capability Check:".bold().green());
    println!();

    // Check platform
    println!("{}", "Platform:".bold());
    println!("  OS: {}", env::consts::OS);
    println!("  Arch: {}", env::consts::ARCH);
    println!();

    // Check feature flags
    println!("{}", "Compile-time features:".bold());
    #[cfg(feature = "overlayfs")]
    println!("  {} overlayfs feature enabled", "✓".green());
    #[cfg(not(feature = "overlayfs"))]
    println!("  {} overlayfs feature disabled", "✗".red());
    println!();

    // Check each backend availability
    #[cfg(all(target_os = "linux", feature = "overlayfs"))]
    {
        use git2db::worktree::overlay_enhanced::{BackendType, BackendSelector};

        println!("{}", "Overlayfs backends:".bold());

        for backend in &[
            BackendType::Kernel,
            BackendType::UserNamespace,
            BackendType::Fuse,
        ] {
            let available = if backend.is_available() {
                "✓".green().to_string()
            } else {
                "✗".red().to_string()
            };

            println!("  {} {:15} - {}", available, backend.name(), backend.description());

            if !backend.is_available() {
                println!("      Requirements:");
                for req in backend.requirements() {
                    println!("        - {}", req);
                }
            }
        }

        println!();
        println!("{}", "Recommended strategy:".bold());

        let selector = BackendSelector::new();
        if let Some(best) = selector.select() {
            println!("  {} ({})", best.name().green(), best.description());
        } else {
            println!("  {} (fallback to git worktrees)", "None available".yellow());
        }
    }

    // Summary
    println!();
    println!("{}", "Summary:".bold());
    let available_count = WorktreeStrategyType::available_strategies().len();
    println!(
        "  {} strategies available on this system",
        available_count.to_string().green()
    );
}

fn show_config_examples() {
    println!("{}", "Configuration Examples:".bold().green());
    println!();

    println!("{}", "1. Simple Git-only Configuration:".bold());
    println!("{}", "```toml".dimmed());
    println!(r#"[worktree]
strategy = "git"
require_strategy = true"#);
    println!("{}", "```".dimmed());
    println!();

    println!("{}", "2. Prefer Overlayfs with Fallback:".bold());
    println!("{}", "```toml".dimmed());
    println!(r#"[worktree]
strategy = "prefer-overlayfs"
require_strategy = false
log_selection = true"#);
    println!("{}", "```".dimmed());
    println!();

    println!("{}", "3. Platform-specific Configuration:".bold());
    println!("{}", "```toml".dimmed());
    println!(r#"[worktree]
strategy = "automatic"

[worktree.platform_overrides]
linux = "overlayfs"
macos = "git"
windows = "git""#);
    println!("{}", "```".dimmed());
    println!();

    println!("{}", "4. Advanced Overlayfs Configuration:".bold());
    println!("{}", "```toml".dimmed());
    println!(r#"[worktree]
strategy = "overlayfs-kernel"
require_strategy = true

[worktree.advanced]
mount_timeout_ms = 10000
retry_mount = true
mount_retries = 5"#);
    println!("{}", "```".dimmed());
}

fn show_migration() {
    println!("{}", "Configuration Migration Guide:".bold().green());
    println!();

    println!("{}", "Old Configuration (Abstract):".bold().red());
    println!("{}", "```toml".dimmed());
    println!(r#"[worktree]
use_overlayfs = true    # Abstract: "use if available"
fallback = true         # Hidden behavior
backend = null          # Implicit selection
log = true"#);
    println!("{}", "```".dimmed());
    println!();

    println!("{}", "New Configuration (Explicit):".bold().green());
    println!("{}", "```toml".dimmed());
    println!(r#"[worktree]
strategy = "prefer-overlayfs"  # Explicit: try overlayfs, fallback to git
require_strategy = false        # Clear behavior
log_selection = true"#);
    println!("{}", "```".dimmed());
    println!();

    println!("{}", "Migration Mapping:".bold());
    println!("  use_overlayfs=true,  fallback=true  → strategy=\"prefer-overlayfs\"");
    println!("  use_overlayfs=true,  fallback=false → strategy=\"overlayfs\"");
    println!("  use_overlayfs=false, fallback=true  → strategy=\"git\"");
    println!("  backend=\"kernel\"                    → strategy=\"overlayfs-kernel\"");
    println!("  backend=\"userns\"                    → strategy=\"overlayfs-userns\"");
    println!("  backend=\"fuse\"                      → strategy=\"overlayfs-fuse\"");
}

fn demo_strategy(name: &str) {
    println!("{}", format!("Demo: {} Strategy", name).bold().green());
    println!();

    // Parse strategy name
    let strategy_type = match WorktreeStrategyType::from_str_lenient(name) {
        Some(s) => s,
        None => {
            println!("{}", format!("Unknown strategy: {}", name).red());
            println!("Run with 'help' to see available strategies");
            return;
        }
    };

    // Show strategy information
    println!("{}", "Strategy Information:".bold());
    println!("  Name: {}", strategy_type.to_string().cyan());
    println!("  Description: {}", strategy_type.description());
    println!(
        "  Available: {}",
        if strategy_type.is_available() {
            "Yes".green()
        } else {
            "No".red()
        }
    );

    if let Some(error) = strategy_type.availability_error() {
        println!("  {}", format!("Error: {}", error).red());
    }
    println!();

    // Try to create the strategy
    println!("{}", "Creating Strategy:".bold());
    match strategy_type.create_strategy() {
        Ok(strategy) => {
            println!("  {} Successfully created", "✓".green());
            println!("  Implementation: {}", strategy.name());

            let caps = strategy.capabilities();
            println!("  Space efficient: {}",
                if caps.space_efficient { "Yes".green() } else { "No".yellow() }
            );
            println!("  Requires privileges: {}",
                if caps.requires_privileges { "Yes".yellow() } else { "No".green() }
            );
            println!("  Performance: {:.1}x", caps.relative_performance);
        }
        Err(e) => {
            println!("  {} Failed to create strategy", "✗".red());
            println!("  Error: {}", e);
        }
    }
    println!();

    // Show configuration example
    println!("{}", "Configuration Example:".bold());
    println!("{}", "```toml".dimmed());
    println!("[worktree]");
    println!("strategy = \"{}\"", strategy_type);

    if matches!(strategy_type, WorktreeStrategyType::Automatic |
                WorktreeStrategyType::Git) {
        println!("# Always available, no special requirements");
    } else {
        println!("require_strategy = false  # Fallback if unavailable");
    }
    println!("{}", "```".dimmed());
}

// Example configuration structure usage
fn _example_config_usage() {
    // Create configuration programmatically
    let mut config = WorktreeConfigV2::default();
    config.strategy = WorktreeStrategyType::PreferOverlayfs;
    config.require_strategy = false;
    config.platform_overrides.linux = Some(WorktreeStrategyType::Overlayfs);
    config.platform_overrides.macos = Some(WorktreeStrategyType::Git);

    // Validate configuration
    match config.validate() {
        Ok(()) => println!("Configuration is valid"),
        Err(e) => println!("Configuration error: {}", e),
    }

    // Get effective strategy for current platform
    let effective = config.effective_strategy();
    println!("Effective strategy: {}", effective);

    // Build strategy with validation
    let builder = WorktreeStrategyBuilder::new()
        .strategy(effective)
        .require_available(config.require_strategy)
        .log_selection(config.log_selection);

    match builder.build() {
        Ok(strategy) => {
            println!("Strategy created: {}", strategy.name());
        }
        Err(e) => {
            println!("Failed to create strategy: {}", e);
        }
    }
}