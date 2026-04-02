//! VFS builtins for the Tcl shell.
//!
//! Each command is an `AsyncCommandFunc` registered via `add_async_context_command`.
//! VFS operations are called directly on the Namespace — no proxy channel needed.

use molt::molt_err;
use molt::molt_ok;
use molt::types::*;
use molt::Interp;
use std::sync::Arc;

use crate::ShellContext;

/// Register all VFS builtins on the interpreter.
pub fn register_all(interp: &mut Interp, ctx_id: ContextID) {
    interp.add_async_context_command("cat", cmd_cat, ctx_id);
    interp.add_async_context_command("ls", cmd_ls, ctx_id);
    interp.add_async_context_command("write", cmd_write, ctx_id);
    interp.add_async_context_command("ctl", cmd_ctl, ctx_id);
    interp.add_context_command("json", cmd_json, ctx_id);
    interp.add_async_context_command("help", cmd_help, ctx_id);
    interp.add_async_context_command("mount", cmd_mount, ctx_id);
}

/// `cat path [path ...]` — read file contents from the VFS.
fn cmd_cat<'a>(interp: &'a mut Interp, ctx_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        molt::check_args(1, argv, 2, 0, "path ?path ...?")?;

        let paths: Vec<String> = argv[1..].iter().map(|v| v.to_string()).collect();
        let ctx = interp.context::<ShellContext>(ctx_id);
        let namespace = Arc::clone(&ctx.namespace);
        let subject = ctx.subject.clone();

        let mut output = String::new();
        for path in &paths {
            match namespace.cat(path, &subject).await {
                Ok(data) => output.push_str(&String::from_utf8_lossy(&data)),
                Err(e) => return molt_err!("{}: {}", path, e),
            }
        }
        molt_ok!(output)
    })
}

/// `ls [path]` — list directory entries. Defaults to `/`.
fn cmd_ls<'a>(interp: &'a mut Interp, ctx_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        molt::check_args(1, argv, 1, 2, "?path?")?;

        let path = if argv.len() > 1 {
            argv[1].to_string()
        } else {
            "/".to_owned()
        };

        let ctx = interp.context::<ShellContext>(ctx_id);
        let namespace = Arc::clone(&ctx.namespace);
        let subject = ctx.subject.clone();

        match namespace.ls(&path, &subject).await {
            Ok(entries) => {
                let text: String = entries
                    .iter()
                    .map(|e| {
                        if e.is_dir {
                            format!("{}/", e.name)
                        } else {
                            e.name.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                molt_ok!(text)
            }
            Err(e) => molt_err!("{}: {}", path, e),
        }
    })
}

/// `write path data` — write data to a VFS file.
fn cmd_write<'a>(interp: &'a mut Interp, ctx_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        molt::check_args(1, argv, 3, 3, "path data")?;

        let path = argv[1].to_string();
        let data = argv[2].to_string();
        let ctx = interp.context::<ShellContext>(ctx_id);
        let namespace = Arc::clone(&ctx.namespace);
        let subject = ctx.subject.clone();

        match namespace.echo(&path, data.as_bytes(), &subject).await {
            Ok(_) => molt_ok!(),
            Err(e) => molt_err!("{}: {}", path, e),
        }
    })
}

/// `ctl path command [args...]` — write to a ctl file and read the response.
fn cmd_ctl<'a>(interp: &'a mut Interp, ctx_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        molt::check_args(1, argv, 3, 0, "path command ?args ...?")?;

        let path = argv[1].to_string();
        let cmd_parts: Vec<String> = argv[2..].iter().map(|v| v.to_string()).collect();
        let cmd_str = cmd_parts.join(" ");

        let ctx = interp.context::<ShellContext>(ctx_id);
        let namespace = Arc::clone(&ctx.namespace);
        let subject = ctx.subject.clone();

        match namespace.ctl(&path, cmd_str.as_bytes(), &subject).await {
            Ok(resp) => {
                let text = String::from_utf8_lossy(&resp);
                molt_ok!(text.into_owned())
            }
            Err(e) => molt_err!("{}: {}", path, e),
        }
    })
}

/// `json parse value` — convert a JSON string to a Tcl dict.
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
fn tcl_list_escape(s: &str) -> String {
    if s.is_empty() || s.contains(|c: char| c.is_whitespace() || c == '{' || c == '}' || c == '\\' || c == '"') {
        format!("{{{}}}", s)
    } else {
        s.to_owned()
    }
}

/// `help` — list VFS builtins and `/bin/` entries.
fn cmd_help<'a>(interp: &'a mut Interp, ctx_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
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
        let namespace = Arc::clone(&ctx.namespace);
        let subject = ctx.subject.clone();

        if let Ok(entries) = namespace.ls("/bin", &subject).await {
            if !entries.is_empty() {
                out.push_str("\nCommands (/bin/):\n");
                for entry in &entries {
                    let name = entry.name.trim_end_matches('/');
                    if !name.is_empty() {
                        out.push_str(&format!("  {}\n", name));
                    }
                }
            }
        }

        molt_ok!(out)
    })
}

/// `mount [prefix]` — list mount points (read-only).
fn cmd_mount<'a>(interp: &'a mut Interp, ctx_id: ContextID, argv: &'a [Value]) -> BoxFuture<'a, MoltResult> {
    Box::pin(async move {
        molt::check_args(1, argv, 1, 2, "?prefix?")?;

        let ctx = interp.context::<ShellContext>(ctx_id);
        let namespace = Arc::clone(&ctx.namespace);
        let prefixes = namespace.mount_prefixes();
        let prefixes_str = prefixes.join("\n");

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
    })
}
