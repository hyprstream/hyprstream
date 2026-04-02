//! VFS builtins for the Tcl shell.
//!
//! Each command is a `CommandFunc` registered via `add_context_command`.
//! VFS state is accessed via `interp.context::<ShellContext>(ctx_id)`.
//!
//! All VFS operations go through the channel proxy pattern: each builtin
//! creates a `std::sync::mpsc::sync_channel(1)` for the reply, sends a
//! `VfsRequest` via `send_vfs_request`, and blocks on `reply_rx.recv()`.

use molt::molt_err;
use molt::molt_ok;
use molt::types::*;
use molt::Interp;

use hyprstream_vfs::proxy::{VfsOp, VfsRequest};

use crate::ShellContext;

/// Send a VFS request, handling both tokio-runtime and plain-thread contexts.
///
/// `blocking_send()` panics when called from within a tokio runtime (e.g. the
/// CLI shell's LocalSet). `try_send()` works in any context as long as the
/// channel has capacity (buffer=64, and we send one-at-a-time with sync reply).
pub(crate) fn send_vfs_request(
    tx: &tokio::sync::mpsc::Sender<VfsRequest>,
    req: VfsRequest,
) -> Result<(), ()> {
    tx.try_send(req).map_err(|_| ())
}

/// Register all VFS builtins on the interpreter.
pub fn register_all(interp: &mut Interp, ctx_id: ContextID) {
    interp.add_context_command("cat", cmd_cat, ctx_id);
    interp.add_context_command("ls", cmd_ls, ctx_id);
    interp.add_context_command("write", cmd_write, ctx_id);
    interp.add_context_command("ctl", cmd_ctl, ctx_id);
    interp.add_context_command("json", cmd_json, ctx_id);
    interp.add_context_command("help", cmd_help, ctx_id);
    interp.add_context_command("mount", cmd_mount, ctx_id);
}

/// `cat path [path ...]` — read file contents from the VFS.
fn cmd_cat(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 2, 0, "path ?path ...?")?;

    let mut output = String::new();
    let paths: Vec<String> = argv[1..].iter().map(|v| v.to_string()).collect();
    let ctx = interp.context::<ShellContext>(ctx_id);

    for path in &paths {
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        let req = VfsRequest {
            op: VfsOp::Cat { path: path.clone() },
            reply: reply_tx,
        };
        if send_vfs_request(&ctx.vfs_tx, req).is_err() {
            return molt_err!("VFS proxy gone");
        }
        match reply_rx.recv() {
            Ok(Ok(data)) => {
                output.push_str(&String::from_utf8_lossy(&data));
            }
            Ok(Err(e)) => return molt_err!("{}: {}", path, e),
            Err(_) => return molt_err!("{}: VFS proxy did not respond", path),
        }
    }
    molt_ok!(output)
}

/// `ls [path]` — list directory entries. Defaults to `/`.
fn cmd_ls(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 1, 2, "?path?")?;

    let path = if argv.len() > 1 {
        argv[1].to_string()
    } else {
        "/".to_owned()
    };

    let ctx = interp.context::<ShellContext>(ctx_id);
    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
    let req = VfsRequest {
        op: VfsOp::Ls { path: path.clone() },
        reply: reply_tx,
    };
    if send_vfs_request(&ctx.vfs_tx, req).is_err() {
        return molt_err!("VFS proxy gone");
    }
    match reply_rx.recv() {
        Ok(Ok(data)) => {
            let text = String::from_utf8_lossy(&data);
            molt_ok!(text.into_owned())
        }
        Ok(Err(e)) => molt_err!("{}: {}", path, e),
        Err(_) => molt_err!("{}: VFS proxy did not respond", path),
    }
}

/// `write path data` — write data to a VFS file.
fn cmd_write(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 3, 3, "path data")?;

    let path = argv[1].to_string();
    let data = argv[2].to_string();
    let ctx = interp.context::<ShellContext>(ctx_id);
    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
    let req = VfsRequest {
        op: VfsOp::Echo {
            path: path.clone(),
            data: data.into_bytes(),
        },
        reply: reply_tx,
    };
    if send_vfs_request(&ctx.vfs_tx, req).is_err() {
        return molt_err!("VFS proxy gone");
    }
    match reply_rx.recv() {
        Ok(Ok(_)) => molt_ok!(),
        Ok(Err(e)) => molt_err!("{}: {}", path, e),
        Err(_) => molt_err!("{}: VFS proxy did not respond", path),
    }
}

/// `ctl path command [args...]` — write to a ctl file and read the response.
fn cmd_ctl(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 3, 0, "path command ?args ...?")?;

    let path = argv[1].to_string();
    let cmd_parts: Vec<String> = argv[2..].iter().map(|v| v.to_string()).collect();
    let cmd_str = cmd_parts.join(" ");

    let ctx = interp.context::<ShellContext>(ctx_id);
    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
    let req = VfsRequest {
        op: VfsOp::Ctl {
            path: path.clone(),
            cmd: cmd_str.into_bytes(),
        },
        reply: reply_tx,
    };
    if send_vfs_request(&ctx.vfs_tx, req).is_err() {
        return molt_err!("VFS proxy gone");
    }
    match reply_rx.recv() {
        Ok(Ok(resp)) => {
            let text = String::from_utf8_lossy(&resp);
            molt_ok!(text.into_owned())
        }
        Ok(Err(e)) => molt_err!("{}: {}", path, e),
        Err(_) => molt_err!("{}: VFS proxy did not respond", path),
    }
}

/// `json parse value` — convert a JSON string to a Tcl dict.
///
/// Returns a Tcl dict representation of the JSON object. Nested objects
/// become nested dicts. Arrays become Tcl lists. Use standard `dict get`
/// to extract fields:
///
/// ```tcl
///     set d [json parse $response]
///     dict get $d fieldName
/// ```
///
/// SECURITY: String values are returned unescaped. If the result is used
/// with `eval` or `subst`, bracket contents trigger Tcl command substitution.
/// The attenuated TclShell does not register `eval` or `subst`, so this is
/// safe in normal operation.
fn cmd_json(_interp: &mut Interp, _ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 3, 3, "parse value")?;

    let subcmd = argv[1].to_string();
    if subcmd != "parse" {
        return molt_err!("unknown subcommand \"{}\": must be parse", subcmd);
    }

    let json_str = argv[2].to_string();
    let parsed: serde_json::Value = match serde_json::from_str(&json_str) {
        Ok(v) => v,
        Err(e) => return molt_err!("invalid JSON: {}", e),
    };
    molt_ok!(json_to_tcl(&parsed))
}

/// Convert a serde_json::Value to a Tcl string representation.
///
/// Objects → Tcl dicts (key value key value ...)
/// Arrays  → Tcl lists (value value ...)
/// Strings → raw string value
/// Numbers/bools/null → string representation
fn json_to_tcl(val: &serde_json::Value) -> String {
    match val {
        serde_json::Value::Object(map) => {
            let mut parts = Vec::with_capacity(map.len() * 2);
            for (k, v) in map {
                parts.push(tcl_list_escape(k));
                parts.push(tcl_list_escape(&json_to_tcl(v)));
            }
            parts.join(" ")
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(|v| tcl_list_escape(&json_to_tcl(v))).collect();
            items.join(" ")
        }
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => (if *b { "1" } else { "0" }).to_owned(),
        serde_json::Value::Null => String::new(),
    }
}

/// Escape a string for safe inclusion in a Tcl list.
///
/// If the string contains spaces, braces, backslashes, or is empty,
/// wrap it in braces. Otherwise return as-is.
fn tcl_list_escape(s: &str) -> String {
    if s.is_empty() || s.contains(|c: char| c.is_whitespace() || c == '{' || c == '}' || c == '\\' || c == '"') {
        format!("{{{}}}", s)
    } else {
        s.to_owned()
    }
}

/// `help` — list VFS builtins and `/cmd/` entries.
fn cmd_help(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 1, 1, "")?;

    let mut out = String::new();
    out.push_str("VFS commands:\n");
    out.push_str("  cat <path>           read file contents\n");
    out.push_str("  ls [path]            list directory\n");
    out.push_str("  write <path> <data>  write to file\n");
    out.push_str("  ctl <path> <cmd>     control file (write+read)\n");
    out.push_str("  json parse <str>     convert JSON to Tcl dict\n");
    out.push_str("  mount [prefix]       list mount points\n");
    out.push_str("  help                 this message\n");

    let ctx = interp.context::<ShellContext>(ctx_id);
    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
    let req = VfsRequest {
        op: VfsOp::Ls {
            path: "/bin".to_owned(),
        },
        reply: reply_tx,
    };
    if send_vfs_request(&ctx.vfs_tx, req).is_ok() {
        if let Ok(Ok(data)) = reply_rx.recv() {
            let text = String::from_utf8_lossy(&data);
            if !text.is_empty() {
                out.push_str("\nCommands (/bin/):\n");
                for name in text.split('\n') {
                    if !name.is_empty() {
                        out.push_str(&format!("  {}\n", name.trim_end_matches('/')));
                    }
                }
            }
        }
    }

    molt_ok!(out)
}

/// `mount [prefix]` — list mount points (read-only).
fn cmd_mount(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 1, 2, "?prefix?")?;

    let ctx = interp.context::<ShellContext>(ctx_id);
    let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
    let req = VfsRequest {
        op: VfsOp::MountPrefixes,
        reply: reply_tx,
    };
    if send_vfs_request(&ctx.vfs_tx, req).is_err() {
        return molt_err!("VFS proxy gone");
    }
    let prefixes_str = match reply_rx.recv() {
        Ok(Ok(data)) => String::from_utf8_lossy(&data).into_owned(),
        Ok(Err(e)) => return molt_err!("mount: {}", e),
        Err(_) => return molt_err!("mount: VFS proxy did not respond"),
    };
    let prefixes: Vec<&str> = prefixes_str.lines().collect();

    if argv.len() == 1 {
        molt_ok!(prefixes_str)
    } else {
        let prefix = argv[1].to_string();
        if prefixes.iter().any(|p| *p == prefix) {
            molt_ok!(format!("mounted: {}", prefix))
        } else {
            molt_err!("not mounted: {}", prefix)
        }
    }
}
