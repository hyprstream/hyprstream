use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let profile = env::var("PROFILE").unwrap();

    // Determine the target directory based on profile
    let target_dir = if profile == "release" {
        "target/release"
    } else {
        "target/debug"
    };

    let build_command = if profile == "release" {
        " --release"
    } else {
        ""
    };

    // Create the admin wrapper script content
    let admin_wrapper_content = format!(
        r#"#!/bin/bash
# TCL MCP Server Admin Wrapper
# Automatically enables privileged mode for tool management capabilities

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Path to the main tcl-mcp-server binary
TCL_SERVER="${{SCRIPT_DIR}}/tcl-mcp-server"

# Check if the binary exists
if [ ! -f "$TCL_SERVER" ]; then
    echo "Error: tcl-mcp-server binary not found at: $TCL_SERVER" >&2
    echo "Please run 'cargo build{}' first" >&2
    exit 1
fi

# Execute the server with privileged mode enabled
exec "$TCL_SERVER" --privileged "$@"
"#,
        build_command
    );

    // Create the Molt runtime wrapper scripts (privileged and non-privileged)
    let molt_wrapper_content = format!(
        r#"#!/bin/bash
# Molt MCP Server Wrapper (Non-Privileged)
# Uses the Molt (safe Rust-based) TCL runtime in restricted mode

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Path to the main tcl-mcp-server binary
TCL_SERVER="${{SCRIPT_DIR}}/tcl-mcp-server"

# Check if the binary exists
if [ ! -f "$TCL_SERVER" ]; then
    echo "Error: tcl-mcp-server binary not found at: $TCL_SERVER" >&2
    echo "Please run 'cargo build{}' first" >&2
    exit 1
fi

# Set runtime to Molt and export environment variable as fallback
export TCL_MCP_RUNTIME=molt

# Execute the server with Molt runtime specified (NON-PRIVILEGED)
exec "$TCL_SERVER" --runtime molt "$@"
"#,
        build_command
    );

    let molt_admin_wrapper_content = format!(
        r#"#!/bin/bash
# Molt MCP Server Admin Wrapper (Privileged)
# Uses the Molt (safe Rust-based) TCL runtime with full privileges

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Path to the main tcl-mcp-server binary
TCL_SERVER="${{SCRIPT_DIR}}/tcl-mcp-server"

# Check if the binary exists
if [ ! -f "$TCL_SERVER" ]; then
    echo "Error: tcl-mcp-server binary not found at: $TCL_SERVER" >&2
    echo "Please run 'cargo build{}' first" >&2
    exit 1
fi

# Set runtime to Molt and export environment variable as fallback
export TCL_MCP_RUNTIME=molt

# Execute the server with Molt runtime specified (PRIVILEGED)
exec "$TCL_SERVER" --runtime molt --privileged "$@"
"#,
        build_command
    );

    // Create the TCL runtime wrapper scripts (privileged and non-privileged)
    let tcl_wrapper_content = format!(
        r#"#!/bin/bash
# TCL MCP Server Wrapper (Non-Privileged)
# Uses the official TCL interpreter runtime in restricted mode

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Path to the main tcl-mcp-server binary
TCL_SERVER="${{SCRIPT_DIR}}/tcl-mcp-server"

# Check if the binary exists
if [ ! -f "$TCL_SERVER" ]; then
    echo "Error: tcl-mcp-server binary not found at: $TCL_SERVER" >&2
    echo "Please run 'cargo build{} --features tcl' first" >&2
    exit 1
fi

# Check for TCL system dependencies
if ! command -v tclsh >/dev/null 2>&1; then
    echo "Warning: tclsh not found in PATH. TCL runtime may not work properly." >&2
    echo "Please install TCL development libraries (e.g., tcl-dev, tcl-devel)" >&2
fi

# Set runtime to TCL and export environment variable as fallback  
export TCL_MCP_RUNTIME=tcl

# Execute the server with TCL runtime specified (NON-PRIVILEGED)
exec "$TCL_SERVER" --runtime tcl "$@"
"#,
        build_command
    );

    let tcl_admin_wrapper_content = format!(
        r#"#!/bin/bash
# TCL MCP Server Admin Wrapper (Privileged)
# Uses the official TCL interpreter runtime with full privileges

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Path to the main tcl-mcp-server binary
TCL_SERVER="${{SCRIPT_DIR}}/tcl-mcp-server"

# Check if the binary exists
if [ ! -f "$TCL_SERVER" ]; then
    echo "Error: tcl-mcp-server binary not found at: $TCL_SERVER" >&2
    echo "Please run 'cargo build{} --features tcl' first" >&2
    exit 1
fi

# Check for TCL system dependencies
if ! command -v tclsh >/dev/null 2>&1; then
    echo "Warning: tclsh not found in PATH. TCL runtime may not work properly." >&2
    echo "Please install TCL development libraries (e.g., tcl-dev, tcl-devel)" >&2
fi

# Set runtime to TCL and export environment variable as fallback  
export TCL_MCP_RUNTIME=tcl

# Execute the server with TCL runtime specified (PRIVILEGED)
exec "$TCL_SERVER" --runtime tcl --privileged "$@"
"#,
        build_command
    );

    // Write all wrapper scripts
    let scripts = vec![
        ("tcl-mcp-server-admin", admin_wrapper_content), // Original admin script (uses default runtime)
        ("tcl-mcp-server-ctcl", tcl_wrapper_content),    // TCL runtime, non-privileged
        ("tcl-mcp-server-admin-ctcl", tcl_admin_wrapper_content), // TCL runtime, privileged
        ("tcl-mcp-server-molt", molt_wrapper_content),   // Molt runtime, non-privileged
        ("tcl-mcp-server-admin-molt", molt_admin_wrapper_content), // Molt runtime, privileged
    ];

    // Ensure target directory exists
    fs::create_dir_all(target_dir).expect("Failed to create target directory");

    for (script_name, content) in scripts {
        let script_path = Path::new(target_dir).join(script_name);
        fs::write(&script_path, content).expect(&format!("Failed to write {} script", script_name));

        // Make the script executable on Unix systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&script_path).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&script_path, perms)
                .expect(&format!("Failed to set {} permissions", script_name));
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
}
