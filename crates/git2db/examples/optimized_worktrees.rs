//! Examples demonstrating the CORRECT worktree architecture
//!
//! Key principle: Git worktrees are ALWAYS created.
//! Optimizations like overlayfs are optional enhancements underneath.

use git2db::worktree::{
    CoWBackend, CoWConfig, FallbackBehavior, OptimizedGitStrategy, OptimizedWorktreeBuilder,
    StorageOptimization, WorktreeConfig,
};
use std::path::Path;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Example repository and paths
    let repo_path = Path::new("/path/to/repository");
    let worktree_base = Path::new("/tmp/worktrees");

    // ============================================================
    // Example 1: Standard Git Worktree (No Optimization)
    // ============================================================
    info!("Example 1: Standard git worktree without optimization");
    {
        let strategy = OptimizedGitStrategy::standard();
        let worktree = strategy
            .create(repo_path, &worktree_base.join("standard"), "feature-branch")
            .await?;

        info!("Created standard worktree at: {:?}", worktree.path());
        info!("Disk usage: 100% of repository size");
        info!("This is a normal git worktree with full disk usage");

        // Use the worktree...
        // All git commands work normally
    }

    // ============================================================
    // Example 2: Auto-Optimized Git Worktree
    // ============================================================
    info!("\nExample 2: Auto-optimized git worktree");
    {
        // This is the DEFAULT - automatically uses best available optimization
        let strategy = OptimizedGitStrategy::new();
        let worktree = strategy
            .create(repo_path, &worktree_base.join("auto-optimized"), "main")
            .await?;

        let metadata = worktree.metadata();
        info!("Created optimized worktree at: {:?}", worktree.path());
        info!("Strategy: {}", metadata.strategy_name);
        if let Some(saved) = metadata.space_saved_bytes {
            info!("Space saved: {} MB", saved / 1024 / 1024);
        }
        info!("This is STILL a normal git worktree, just more efficient!");
    }

    // ============================================================
    // Example 3: Explicitly Request Copy-on-Write
    // ============================================================
    info!("\nExample 3: Explicit Copy-on-Write optimization");
    {
        let config = WorktreeConfig {
            optimization: StorageOptimization::CopyOnWrite(CoWConfig::default()),
            fallback: FallbackBehavior::Warn,
            log_optimization: true,
        };

        let strategy = OptimizedGitStrategy::with_config(config);
        let worktree = strategy
            .create(repo_path, &worktree_base.join("cow-optimized"), "develop")
            .await?;

        info!("Created CoW-optimized worktree at: {:?}", worktree.path());
        info!("Expected disk savings: ~80%");
        info!("Git commands work exactly the same as always!");
    }

    // ============================================================
    // Example 4: Specific CoW Backend
    // ============================================================
    info!("\nExample 4: Specific overlayfs backend");
    {
        let config = WorktreeConfig {
            optimization: StorageOptimization::CopyOnWrite(CoWConfig {
                backend: CoWBackend::UserNamespace, // Unprivileged
                mount_options: vec![],
                overlay_dir: Some("/tmp/overlay-work".to_string()),
            }),
            fallback: FallbackBehavior::Continue,
            log_optimization: true,
        };

        let strategy = OptimizedGitStrategy::with_config(config);
        let worktree = strategy
            .create(
                repo_path,
                &worktree_base.join("userns-optimized"),
                "feature-x",
            )
            .await?;

        info!("Created worktree with user namespace overlayfs");
        info!("No root privileges required!");
    }

    // ============================================================
    // Example 5: Builder Pattern
    // ============================================================
    info!("\nExample 5: Using the builder pattern");
    {
        let config = OptimizedWorktreeBuilder::new()
            .with_cow() // Enable CoW optimization
            .fallback(FallbackBehavior::Warn) // Warn if not available
            .log_optimization(true) // Log what happens
            .build();

        let strategy = OptimizedGitStrategy::with_config(config);
        let worktree = strategy
            .create(
                repo_path,
                &worktree_base.join("builder-optimized"),
                "bugfix",
            )
            .await?;

        info!("Created worktree using builder pattern");
    }

    // ============================================================
    // Example 6: Fallback Behavior
    // ============================================================
    info!("\nExample 6: Fallback behavior demonstration");
    {
        // This might fail if CoW is not available
        let fail_config = WorktreeConfig {
            optimization: StorageOptimization::CopyOnWrite(CoWConfig {
                backend: CoWBackend::Kernel, // Requires root
                ..Default::default()
            }),
            fallback: FallbackBehavior::Fail, // Fail if not available
            log_optimization: true,
        };

        let strategy = OptimizedGitStrategy::with_config(fail_config);
        match strategy
            .create(repo_path, &worktree_base.join("may-fail"), "test")
            .await
        {
            Ok(worktree) => {
                info!("Created worktree with kernel overlayfs (running as root?)");
            }
            Err(e) => {
                info!("Expected failure when not root: {}", e);

                // Now try with fallback
                let fallback_config = WorktreeConfig {
                    optimization: StorageOptimization::CopyOnWrite(CoWConfig {
                        backend: CoWBackend::Kernel,
                        ..Default::default()
                    }),
                    fallback: FallbackBehavior::Continue, // Continue without optimization
                    log_optimization: true,
                };

                let strategy = OptimizedGitStrategy::with_config(fallback_config);
                let worktree = strategy
                    .create(repo_path, &worktree_base.join("fallback-success"), "test")
                    .await?;

                info!("Fallback successful - created standard git worktree");
            }
        }
    }

    // ============================================================
    // Key Insights Demonstration
    // ============================================================
    info!("\n" + "=".repeat(60).as_str());
    info!("KEY INSIGHTS:");
    info!("1. Every example created a GIT WORKTREE");
    info!("2. Optimizations are OPTIONAL ENHANCEMENTS");
    info!("3. All git commands work IDENTICALLY");
    info!("4. The only difference is DISK USAGE");
    info!("5. Overlayfs does NOT replace git - it optimizes it");
    info!("=".repeat(60).as_str());

    Ok(())
}

/// Demonstrate that optimized worktrees are real git worktrees
async fn demonstrate_git_functionality(worktree_path: &Path) {
    use std::process::Command;

    info!("Demonstrating that this is a real git worktree:");

    // All standard git commands work
    let commands = [
        ("git status", "Check repository status"),
        ("git branch", "Show current branch"),
        ("git log --oneline -5", "Show recent commits"),
        ("git diff", "Show uncommitted changes"),
    ];

    for (cmd, description) in &commands {
        info!("  {} - {}", cmd, description);
        let output = Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .current_dir(worktree_path)
            .output()
            .expect("Failed to execute git command");

        if output.status.success() {
            info!("    ✓ Command succeeded");
        } else {
            error!("    ✗ Command failed");
        }
    }

    info!("All git functionality works normally!");
}

/// Show disk usage comparison
async fn compare_disk_usage(standard_path: &Path, optimized_path: &Path) {
    use std::process::Command;

    let get_size = |path: &Path| -> u64 {
        let output = Command::new("du")
            .arg("-sb")
            .arg(path)
            .output()
            .expect("Failed to get disk usage");

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    };

    let standard_size = get_size(standard_path);
    let optimized_size = get_size(optimized_path);

    let saved = standard_size.saturating_sub(optimized_size);
    let saved_percent = if standard_size > 0 {
        (saved as f64 / standard_size as f64) * 100.0
    } else {
        0.0
    };

    info!("Disk Usage Comparison:");
    info!("  Standard worktree:  {} MB", standard_size / 1024 / 1024);
    info!("  Optimized worktree: {} MB", optimized_size / 1024 / 1024);
    info!(
        "  Space saved:        {} MB ({:.1}%)",
        saved / 1024 / 1024,
        saved_percent
    );
}
