//! VFS namespace construction for TUI-spawned ChatApps.
//!
//! Builds the same VFS namespace layout as [`crate::cli::shell_handlers`] so that
//! remotely-spawned ChatApps (via TuiService RPC) have full `/bin/`, `/env/`,
//! `/srv/model`, and `/lang/tcl` access.

use std::sync::Arc;

use hyprstream_rpc::Subject;
use hyprstream_vfs::Namespace;

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
///
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
    #[allow(clippy::expect_used)] // Thread/runtime creation failure is unrecoverable
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
) -> anyhow::Result<(Arc<Namespace>, Subject)> {
    let subject = {
        let vk = signing_key.verifying_key();
        let pubkey_hex = hex::encode(vk.as_bytes());
        Subject::new(pubkey_hex)
    };

    // Resolve the model service's verifying key via PolicyClient.
    let policy_vk = hyprstream_service::global_trust_store()
        .resolve_one("policy")
        .ok_or_else(|| anyhow::anyhow!("trust store has no policy key"))?;
    let policy_client = crate::services::PolicyClient::for_service(
        signing_key.clone(),
        policy_vk,
        None,
    )?;
    let model_vk_resp = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(policy_client.resolve_service_key(
            &crate::services::generated::policy_client::ResolveServiceKey {
                service_name: "model".to_owned(),
            },
        ))
    })
    .map_err(|e| anyhow::anyhow!("Failed to resolve model service key: {e}"))?;
    let model_vk = hyprstream_rpc::crypto::VerifyingKey::from_bytes(
        model_vk_resp.verifying_key.as_slice().try_into()
            .map_err(|_| anyhow::anyhow!("Invalid key length"))?,
    ).map_err(|e| anyhow::anyhow!("Invalid key: {e}"))?;

    let model_client = crate::services::generated::model_client::ModelClient::for_service(
        signing_key.clone(),
        model_vk,
        None,
    )?;

    // Compose the standard namespace shared with the CLI shell (see
    // `crate::services::namespace_builder`). This builder does not currently
    // obtain a registry client, so `/srv/registry` and its `/worktree` alias
    // are omitted here (unchanged from pre-#634 behavior).
    let ns = crate::services::build_standard_namespace(crate::services::StandardNamespaceConfig {
        subject: subject.clone(),
        model_client,
        registry: None,
        model_path: "/srv/model".to_owned(),
        registry_path: "/srv/registry".to_owned(),
        worktree_path: "/worktree".to_owned(),
        bin_path: "/bin".to_owned(),
        env_path: "/env".to_owned(),
        tcl_path: "/lang/tcl".to_owned(),
    });

    Ok((Arc::new(ns), subject))
}
