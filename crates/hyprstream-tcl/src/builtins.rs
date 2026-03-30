//! VFS builtins for the Tcl shell.
//!
//! Each command is a `CommandFunc` registered via `add_context_command`.
//! VFS state is accessed via `interp.context::<ShellContext>(ctx_id)`.

use molt::molt_err;
use molt::molt_ok;
use molt::types::*;
use molt::Interp;

use crate::ShellContext;

/// Register all VFS builtins on the interpreter.
pub fn register_all(interp: &mut Interp, ctx_id: ContextID) {
    interp.add_context_command("cat", cmd_cat, ctx_id);
    interp.add_context_command("ls", cmd_ls, ctx_id);
    interp.add_context_command("echo", cmd_echo, ctx_id);
    interp.add_context_command("ctl", cmd_ctl, ctx_id);
    interp.add_context_command("help", cmd_help, ctx_id);
    interp.add_context_command("mount", cmd_mount, ctx_id);
}

/// `cat path [path ...]` — read file contents from the VFS.
fn cmd_cat(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 2, 0, "path ?path ...?")?;

    let mut output = String::new();
    // Collect paths first, then borrow context — avoids holding &mut Interp across iterations.
    let paths: Vec<String> = argv[1..].iter().map(|v| v.to_string()).collect();
    let ctx = interp.context::<ShellContext>(ctx_id);

    for path in &paths {
        match ctx.rt.block_on(ctx.ns.cat(path, &ctx.subject)) {
            Ok(data) => {
                output.push_str(&String::from_utf8_lossy(&data));
            }
            Err(e) => return molt_err!("{}: {}", path, e),
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
    match ctx.rt.block_on(ctx.ns.ls(&path, &ctx.subject)) {
        Ok(entries) => {
            let lines: Vec<String> = entries
                .iter()
                .map(|e| {
                    if e.is_dir {
                        format!("{}/", e.name)
                    } else {
                        e.name.clone()
                    }
                })
                .collect();
            molt_ok!(lines.join("\n"))
        }
        Err(e) => molt_err!("{}: {}", path, e),
    }
}

/// `echo path data` — write data to a VFS file.
fn cmd_echo(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 3, 3, "path data")?;

    let path = argv[1].to_string();
    let data = argv[2].to_string();
    let ctx = interp.context::<ShellContext>(ctx_id);

    match ctx.rt.block_on(ctx.ns.echo(&path, data.as_bytes(), &ctx.subject)) {
        Ok(()) => molt_ok!(),
        Err(e) => molt_err!("{}: {}", path, e),
    }
}

/// `ctl path command [args...]` — write to a ctl file and read the response.
fn cmd_ctl(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 3, 0, "path command ?args ...?")?;

    let path = argv[1].to_string();
    let cmd_parts: Vec<String> = argv[2..].iter().map(|v| v.to_string()).collect();
    let cmd_str = cmd_parts.join(" ");

    let ctx = interp.context::<ShellContext>(ctx_id);
    match ctx.rt.block_on(ctx.ns.ctl(&path, cmd_str.as_bytes(), &ctx.subject)) {
        Ok(resp) => {
            let text = String::from_utf8_lossy(&resp);
            molt_ok!(text.into_owned())
        }
        Err(e) => molt_err!("{}: {}", path, e),
    }
}

/// `help` — list VFS builtins and `/cmd/` entries.
fn cmd_help(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 1, 1, "")?;

    let mut out = String::new();
    out.push_str("VFS commands:\n");
    out.push_str("  cat <path>           read file contents\n");
    out.push_str("  ls [path]            list directory\n");
    out.push_str("  echo <path> <data>   write to file\n");
    out.push_str("  ctl <path> <cmd>     control file (write+read)\n");
    out.push_str("  mount [prefix]       list mount points\n");
    out.push_str("  help                 this message\n");

    let ctx = interp.context::<ShellContext>(ctx_id);
    if let Ok(entries) = ctx.rt.block_on(ctx.ns.ls("/bin", &ctx.subject)) {
        if !entries.is_empty() {
            out.push_str("\nCommands (/bin/):\n");
            for entry in &entries {
                out.push_str(&format!("  {}\n", entry.name));
            }
        }
    }

    molt_ok!(out)
}

/// `mount [prefix]` — list mount points (read-only).
fn cmd_mount(interp: &mut Interp, ctx_id: ContextID, argv: &[Value]) -> MoltResult {
    molt::check_args(1, argv, 1, 2, "?prefix?")?;

    let ctx = interp.context::<ShellContext>(ctx_id);
    let prefixes = ctx.ns.mount_prefixes();

    if argv.len() == 1 {
        molt_ok!(prefixes.join("\n"))
    } else {
        let prefix = argv[1].to_string();
        if prefixes.iter().any(|p| *p == prefix) {
            molt_ok!(format!("mounted: {}", prefix))
        } else {
            molt_err!("not mounted: {}", prefix)
        }
    }
}
