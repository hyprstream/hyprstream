//! Shared "standard namespace" composition for VFS consumers.
//!
//! Prior to #634, `cli::shell_handlers::handle_shell_tui` and
//! `tui::vfs::build_chat_vfs_namespace` each hand-rolled the same
//! `Namespace::new()` + `bind_mount(...)` recipe: `/srv/model`,
//! `/srv/registry` (+ `/worktree` alias), `/bin` (static command listing +
//! discovered `.tcl` tool scripts), `/env` (session variables), and
//! `/lang/tcl` (the Tcl shell mount, whose driver gets a `fork()` snapshot of
//! the namespace taken *before* `/lang/tcl` itself is mounted so it never
//! observes itself). The two copies had already drifted — the TUI builder
//! was missing `/srv/registry` — so this module is now the single place that
//! recipe is defined. Callers add only what's genuinely consumer-specific
//! (e.g. `/lang/python`, tracked separately in #632) around a call to
//! [`build_standard_namespace`].
//!
//! This is a pure assembly layer: it does not change any Subject/MAC
//! scoping semantics of the individual mounts (`RemoteModelMount`,
//! `RemoteRegistryMount`, `TclMount`) it wires together.

use std::collections::HashMap;
use std::sync::Arc;

use hyprstream_rpc::Subject;
use hyprstream_vfs::Namespace;

use super::fs::{SyntheticNode, SyntheticTree};
use super::generated::model_client::ModelClient;
use super::remote_mount::RemoteModelMount;
use super::remote_registry_mount::RemoteRegistryMount;
use super::RegistryClient;

/// Inputs that legitimately differ between the CLI shell and TUI ChatApp
/// call sites. Everything else about the recipe is fixed by
/// [`build_standard_namespace`].
pub struct StandardNamespaceConfig {
    /// Subject the `/lang/tcl` driver operates as (the caller separately
    /// keeps its own copy of the `Subject` for VFS proxy / RPC use — this
    /// builder only consumes it for `TclMount::spawn`).
    pub subject: Subject,
    /// Already-constructed model service RPC client, mounted at
    /// `/srv/model`.
    pub model_client: ModelClient,
    /// Registry service RPC client. When `Some`, mounts `/srv/registry`
    /// (+ a `/worktree` alias, `BindFlag::After`, same backing mount).
    /// The CLI shell always has a registry client available; the TUI
    /// ChatApp namespace builder currently does not obtain one and passes
    /// `None` — this was already the pre-#634 behavior of each builder, now
    /// made an explicit parameter rather than silently unified.
    pub registry: Option<RegistryClient>,
}

/// Compose the standard VFS namespace shared by the CLI shell and TUI
/// ChatApp namespace builders.
///
/// Returns the un-wrapped `Namespace`; callers `Arc::new` it themselves
/// (some need to add consumer-specific mounts, e.g. `/lang/python`, before
/// sharing it).
pub fn build_standard_namespace(cfg: StandardNamespaceConfig) -> Namespace {
    let StandardNamespaceConfig {
        subject,
        model_client,
        registry,
    } = cfg;

    let mut ns = Namespace::new();

    // Mount the model service's 9P filesystem via RPC proxy.
    // RemoteModelMount translates sync Mount trait calls to async ModelClient
    // RPC requests, so `/srv/model/{model_ref}/status` etc. are served by the
    // model service's SyntheticTree on the other end of the RPC channel.
    let remote_model_mount = RemoteModelMount::new(model_client);
    let _ = ns.mount("/srv/model", Arc::new(remote_model_mount));

    // Mount the registry service's worktree filesystem via RPC proxy, when a
    // registry client is available. RemoteRegistryMount is the registry
    // analogue of RemoteModelMount: it proxies 9P operations to the registry
    // service's `WorktreeClient` (real qids, real filesystem). 2-level scope:
    //   /srv/registry/{repo_name}/{worktree_name}/<...rest...>
    // Per #389 + #391 (Option 1, shared content model): the namespace mirrors
    // the browser namespace's `/srv/registry` mount, both backed at the
    // hyprstream spine. See `STANDARD_NAMESPACE_PATHS` in hyprstream-vfs for
    // the convergence contract.
    if let Some(registry) = registry {
        let remote_registry_mount = RemoteRegistryMount::new(registry);
        let registry_mount: Arc<RemoteRegistryMount> = Arc::new(remote_registry_mount);
        let _ = ns.mount("/srv/registry", registry_mount.clone());
        // `/worktree` is an alias for `/srv/registry` — same backing mount,
        // exposed under the short path for ergonomic worktree access. Both
        // `/worktree/{repo}/{wt}` and `/srv/registry/{repo}/{wt}` resolve to
        // the same worktree content.
        let _ = ns.bind_mount("/worktree", registry_mount, hyprstream_vfs::BindFlag::After);
    }

    // /bin/ — static directory entries for listing purposes.
    // Actual command execution goes through TclShell builtins via VFS proxy,
    // not through these nodes. These exist so `ls /bin` shows available commands.
    let bin_tree = {
        let mut children = HashMap::new();

        children.insert(
            "cat".to_owned(),
            SyntheticNode::ReadFile(Box::new(|| b"usage: cat <path> [path ...]".to_vec())),
        );
        children.insert(
            "ls".to_owned(),
            SyntheticNode::ReadFile(Box::new(|| b"usage: ls [path]".to_vec())),
        );
        children.insert(
            "help".to_owned(),
            SyntheticNode::ReadFile(Box::new(|| {
                let mut out = String::from("VFS commands:\n");
                out.push_str("  cat <path>           read file contents\n");
                out.push_str("  ls [path]            list directory\n");
                out.push_str("  write <path> <data>  write to file\n");
                out.push_str("  ctl <path> <cmd>     control file (write+read)\n");
                out.push_str("  json parse <str>     convert JSON to Tcl dict\n");
                out.push_str("  mount [prefix]       list mount points\n");
                out.push_str("  help                 this message\n");
                out.into_bytes()
            })),
        );
        children.insert(
            "mount".to_owned(),
            SyntheticNode::ReadFile(Box::new(|| b"usage: mount".to_vec())),
        );
        children.insert(
            "write".to_owned(),
            SyntheticNode::ReadFile(Box::new(|| b"usage: write <path> <data>".to_vec())),
        );

        SyntheticTree::new(SyntheticNode::Dir { children })
    };
    let _ = ns.mount("/bin", Arc::new(bin_tree));

    // Discover .tcl tool scripts and bind them into /bin/ (After = fallback).
    let tools_dir = std::env::var("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".config"))
        .join("hyprstream/tools");
    let tools = discover_tools(&tools_dir);
    if !tools.is_empty() {
        let tools_tree = SyntheticTree::new(SyntheticNode::Dir { children: tools });
        let _ = ns.bind_mount("/bin", Arc::new(tools_tree), hyprstream_vfs::BindFlag::After);
    }

    // /env/ — session variables as files (Plan9 per-process /env/).
    let env_store: Arc<parking_lot::RwLock<HashMap<String, String>>> =
        Arc::new(parking_lot::RwLock::new(HashMap::new()));

    let env_tree = {
        let store_list = env_store.clone();
        let store_resolve = env_store.clone();

        SyntheticTree::new(SyntheticNode::DynamicDir {
            list: Box::new(move || {
                store_list
                    .read()
                    .keys()
                    .map(|k| hyprstream_vfs::DirEntry {
                        name: k.clone(),
                        is_dir: false,
                        size: 0,
                        stat: None,
                    })
                    .collect()
            }),
            resolve: Box::new(move |name| {
                let store = store_resolve.clone();
                let key = name.to_owned();
                Some(SyntheticNode::ReadFile(Box::new(move || {
                    store
                        .read()
                        .get(&key)
                        .cloned()
                        .unwrap_or_default()
                        .into_bytes()
                })))
            }),
        })
    };
    let _ = ns.mount("/env", Arc::new(env_tree));

    // /lang/tcl — the driver gets a fork() snapshot of the namespace (taken
    // before /lang/tcl is mounted, so it never observes itself) for
    // `/bin/{cmd}` fallback resolution inside `/lang/tcl/eval`.
    let driver_ns = Arc::new(ns.fork());
    let tcl_mount = Arc::new(hyprstream_workers_tcl::TclMount::spawn(subject, driver_ns));
    let _ = ns.mount("/lang/tcl", tcl_mount);

    // /lang/python needs a wasm Sandbox this builder does not hold; wiring it
    // is tracked in #632.

    ns
}

/// Discover `.tcl` tool scripts in `dir` and turn each into a `SyntheticNode`
/// keyed by file stem, for binding into `/bin` (After = fallback).
pub(crate) fn discover_tools(dir: &std::path::Path) -> HashMap<String, SyntheticNode> {
    let mut tools = HashMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return tools,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("tcl") {
            continue;
        }
        let name = match path.file_stem().and_then(|s| s.to_str()) {
            Some(n) => n.to_owned(),
            None => continue,
        };
        // Reject names with path separators or dots (safety).
        if name.contains('/') || name.contains("..") || name.is_empty() {
            continue;
        }

        let script_path = path.clone();
        tools.insert(
            name,
            SyntheticNode::ReadFile(Box::new(move || {
                // Read script from disk at access time so edits take effect immediately.
                std::fs::read_to_string(&script_path)
                    .unwrap_or_else(|e| format!("# error reading tool script: {e}"))
                    .into_bytes()
            })),
        );
    }

    tools
}
