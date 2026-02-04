//! Build script for Hyprstream

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("DOCS_RS").is_ok() {
        return;
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=schema/");
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    // Capture git info for version string
    capture_git_info();

    // Compile Cap'n Proto schemas
    compile_capnp_schemas();

    // If using Python PyTorch or download-libtorch, tch-rs handles libtorch setup
    if env::var("LIBTORCH_USE_PYTORCH").is_ok() || env::var("LIBTORCH").is_err() {
        // tch-rs will handle libtorch setup
        return;
    }

    let libtorch_path = match env::var("LIBTORCH") {
        Ok(path) => path,
        Err(_) => return, // Early return, should not happen due to check above
    };

    // Validate libtorch exists
    let libtorch_dir = Path::new(&libtorch_path);
    if !libtorch_dir.exists() {
        panic!("libtorch directory not found at {libtorch_path}");
    }

    // Configure linking
    println!("cargo:rustc-link-search=native={libtorch_path}/lib");
    println!("cargo:rustc-env=LIBTORCH_STATIC=0");
    println!("cargo:rustc-env=LIBTORCH_BYPASS_VERSION_CHECK=1");
}

fn compile_capnp_schemas() {
    let schema_dir = Path::new("schema");

    // Skip if schema directory doesn't exist
    if !schema_dir.exists() {
        return;
    }

    // Note: common.capnp (identity, envelope) is in hyprstream-rpc crate

    // Compile events schema
    let events_schema = schema_dir.join("events.capnp");
    if events_schema.exists() {
        if let Err(e) = capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&events_schema)
            .run()
        {
            panic!("failed to compile events.capnp: {e}");
        }
    }

    // Compile inference schema
    let inference_schema = schema_dir.join("inference.capnp");
    if inference_schema.exists() {
        if let Err(e) = capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&inference_schema)
            .run()
        {
            panic!("failed to compile inference.capnp: {e}");
        }
    }

    // Compile registry schema
    let registry_schema = schema_dir.join("registry.capnp");
    if registry_schema.exists() {
        if let Err(e) = capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&registry_schema)
            .run()
        {
            panic!("failed to compile registry.capnp: {e}");
        }
    }

    // Compile policy schema
    let policy_schema = schema_dir.join("policy.capnp");
    if policy_schema.exists() {
        if let Err(e) = capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&policy_schema)
            .run()
        {
            panic!("failed to compile policy.capnp: {e}");
        }
    }

    // Compile model schema
    let model_schema = schema_dir.join("model.capnp");
    if model_schema.exists() {
        if let Err(e) = capnpc::CompilerCommand::new()
            .src_prefix("schema")
            .file(&model_schema)
            .run()
        {
            panic!("failed to compile model.capnp: {e}");
        }
    }
}

/// Capture git info and export as environment variables for the build
///
/// Exports:
/// - GIT_SHA: 7-char commit SHA (e.g., "abc1234")
/// - GIT_BRANCH: sanitized branch name (e.g., "main", "feature-auth")
/// - GIT_DIRTY: "true" or "false"
fn capture_git_info() {
    // Get commit SHA (short)
    let sha = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default();

    // Get branch name
    let branch = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .map(|b| if b == "HEAD" { String::new() } else { b }) // Detached HEAD
        .unwrap_or_default();

    // Check if worktree is dirty
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    // Sanitize branch name for filesystem safety
    let sanitized_branch = sanitize_git_ref(&branch);

    // Export individual components as environment variables
    println!("cargo:rustc-env=GIT_SHA={}", sha);
    println!("cargo:rustc-env=GIT_BRANCH={}", sanitized_branch);
    println!("cargo:rustc-env=GIT_DIRTY={}", dirty);

    // Build complete version string
    let cargo_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "unknown".to_owned());
    let build_version = if sha.is_empty() {
        cargo_version
    } else {
        let mut v = format!("{}+", cargo_version);
        if !sanitized_branch.is_empty() {
            v.push_str(&sanitized_branch);
            v.push('.');
        }
        v.push('g');
        v.push_str(&sha);
        if dirty {
            v.push_str(".dirty");
        }
        v
    };
    println!("cargo:rustc-env=BUILD_VERSION={}", build_version);
}

/// Sanitize a git ref name for safe use in filesystem paths
///
/// - Lowercase
/// - Replace `/`, `.`, ` ` with `-`
/// - Remove other special characters
/// - Collapse multiple `-` into one
/// - Trim leading/trailing `-`
/// - Limit to 50 characters
fn sanitize_git_ref(ref_name: &str) -> String {
    let sanitized: String = ref_name
        .to_lowercase()
        .chars()
        .map(|c| match c {
            'a'..='z' | '0'..='9' | '-' | '_' => c,
            _ => '-',
        })
        .collect();

    // Collapse multiple dashes and trim
    let mut result = String::with_capacity(sanitized.len());
    let mut last_was_dash = true; // Start true to trim leading dashes
    for c in sanitized.chars().take(50) {
        if c == '-' {
            if !last_was_dash {
                result.push(c);
                last_was_dash = true;
            }
        } else {
            result.push(c);
            last_was_dash = false;
        }
    }

    // Trim trailing dash
    if result.ends_with('-') {
        result.pop();
    }

    result
}
