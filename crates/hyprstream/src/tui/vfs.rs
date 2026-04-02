//! VFS namespace construction for TUI-spawned ChatApps.
//!
//! Builds the same VFS namespace layout as [`crate::cli::shell_handlers`] so that
//! remotely-spawned ChatApps (via TuiService RPC) have full `/bin/`, `/env/`,
//! `/srv/model`, and `/lang/tcl` access.

use std::collections::HashMap;
use std::sync::Arc;

use hyprstream_rpc::Subject;
use hyprstream_vfs::Namespace;

use crate::services::fs::{SyntheticNode, SyntheticTree};

/// Build a VFS namespace suitable for ChatApp usage.
///
/// Returns `(namespace, subject)`. The namespace contains:
/// - `/bin/` — cat, ls, help, mount, echo commands (static entries for listing)
/// - `/env/` — session variable store
/// - `/srv/model` — remote model service mount (if model service is reachable)
/// - `/lang/tcl` — Tcl interpreter state
///
/// Note: Actual VFS command execution goes through TclShell builtins via the
/// VFS proxy channel, not through /bin/ ctl files. The /bin/ entries exist
/// for directory listing so `ls /bin` shows available commands.

/// Spawn a VFS proxy on a **dedicated thread** with its own tokio runtime.
///
/// The TUI service's `RequestLoop` runs on a `current_thread` runtime inside
/// a `LocalSet`. Tasks spawned via `tokio::spawn` on that runtime may not be
/// polled promptly because the LocalSet's event loop dominates. By running
/// the proxy on its own thread, we guarantee responsiveness regardless of
/// the service's runtime state.
pub fn spawn_dedicated_vfs_proxy(
    ns: Arc<Namespace>,
    subject: Subject,
) -> tokio::sync::mpsc::Sender<hyprstream_vfs::proxy::VfsRequest> {
    let (tx, rx) = tokio::sync::mpsc::channel::<hyprstream_vfs::proxy::VfsRequest>(64);
    std::thread::Builder::new()
        .name("vfs-proxy".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("vfs-proxy runtime");
            rt.block_on(vfs_proxy_loop(ns, subject, rx));
        })
        .expect("vfs-proxy thread");
    tx
}

async fn vfs_proxy_loop(
    ns: Arc<Namespace>,
    subject: Subject,
    mut rx: tokio::sync::mpsc::Receiver<hyprstream_vfs::proxy::VfsRequest>,
) {
    use hyprstream_vfs::proxy::VfsOp;

    while let Some(req) = rx.recv().await {
        let result = match req.op {
            VfsOp::Cat { ref path } => ns
                .cat(path, &subject)
                .await
                .map_err(|e| e.to_string()),
            VfsOp::Ls { ref path } => ns
                .ls(path, &subject)
                .await
                .map(|entries| {
                    entries
                        .iter()
                        .map(|e| {
                            if e.is_dir {
                                format!("{}/", e.name)
                            } else {
                                e.name.clone()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                        .into_bytes()
                })
                .map_err(|e| e.to_string()),
            VfsOp::Echo { ref path, ref data } => ns
                .echo(path, data, &subject)
                .await
                .map(|_| Vec::new())
                .map_err(|e| e.to_string()),
            VfsOp::Ctl { ref path, ref cmd } => ns
                .ctl(path, cmd, &subject)
                .await
                .map_err(|e| e.to_string()),
            VfsOp::MountPrefixes => {
                Ok(ns.mount_prefixes().join("\n").into_bytes())
            }
        };
        let _ = req.reply.send(result);
    }
}

pub fn build_chat_vfs_namespace(
    signing_key: &ed25519_dalek::SigningKey,
) -> (Arc<Namespace>, Subject) {
    use hyprstream_rpc::envelope::RequestIdentity;

    let subject = {
        let vk = signing_key.verifying_key();
        let pubkey_hex = hex::encode(vk.as_bytes());
        Subject::new(pubkey_hex)
    };

    let mut ns = Namespace::new();

    // Mount /srv/model via RemoteModelMount.
    let model_client = crate::services::generated::model_client::ModelClient::new(
        signing_key.clone(),
        RequestIdentity::anonymous(),
    );
    let remote_model_mount = crate::services::remote_mount::RemoteModelMount::new(model_client);
    let _ = ns.mount("/srv/model", Arc::new(remote_model_mount));

    // /bin/ — static directory entries for listing purposes.
    // Actual command execution goes through TclShell builtins via VFS proxy,
    // not through these nodes. These exist so `ls /bin` shows available commands.
    let bin_tree = {
        let mut children = HashMap::new();

        children.insert("cat".to_owned(), SyntheticNode::ReadFile(Box::new(|| {
            b"usage: cat <path> [path ...]".to_vec()
        })));
        children.insert("ls".to_owned(), SyntheticNode::ReadFile(Box::new(|| {
            b"usage: ls [path]".to_vec()
        })));
        children.insert("help".to_owned(), SyntheticNode::ReadFile(Box::new(|| {
            let mut out = String::from("VFS commands:\n");
            out.push_str("  cat <path>           read file contents\n");
            out.push_str("  ls [path]            list directory\n");
            out.push_str("  echo <path> <data>   write to file\n");
            out.push_str("  ctl <path> <cmd>     control file (write+read)\n");
            out.push_str("  mount [prefix]       list mount points\n");
            out.push_str("  help                 this message\n");
            out.into_bytes()
        })));
        children.insert("mount".to_owned(), SyntheticNode::ReadFile(Box::new(|| {
            b"usage: mount".to_vec()
        })));
        children.insert("echo".to_owned(), SyntheticNode::ReadFile(Box::new(|| {
            b"usage: echo <path> <data>".to_vec()
        })));

        SyntheticTree::new(SyntheticNode::Dir { children })
    };
    let _ = ns.mount("/bin", Arc::new(bin_tree));

    // Discover .tcl tool scripts and bind them into /bin/ (After = fallback).
    let tools_dir = std::env::var("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".config"))
        .join("hyprstream/tools");
    let tools = crate::cli::shell_handlers::discover_tools(&tools_dir);
    if !tools.is_empty() {
        let tools_tree = SyntheticTree::new(SyntheticNode::Dir { children: tools });
        let _ = ns.bind_mount(
            "/bin",
            Arc::new(tools_tree),
            hyprstream_vfs::BindFlag::After,
        );
    }

    // /env/ — session variables as files.
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

    // /lang/tcl mount.
    let (tcl_mount_tx, _tcl_mount_rx) = hyprstream_tcl::create_mount_channel();
    let tcl_mount = Arc::new(hyprstream_tcl::TclMount::new(tcl_mount_tx));
    let _ = ns.mount("/lang/tcl", tcl_mount);

    (Arc::new(ns), subject)
}
