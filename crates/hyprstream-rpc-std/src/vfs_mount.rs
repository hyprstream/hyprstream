//! VFS Mount implementations backed by WASM RpcSession.
//!
//! Each mount translates filesystem operations (walk, read, readdir) into
//! RPC calls via the existing WASM client methods. Uses the same ZMTP/QUIC
//! transport as all other browser RPC.
//!
//! Mount points:
//!   /srv/registry  → RegistryMount (list repos, repo details, branches, worktrees)
//!   /srv/model     → ModelMount (status, load/unload)
//!   /srv/policy    → PolicyMount (check, list scopes)

#![cfg(target_arch = "wasm32")]

use std::sync::Arc;

use async_trait::async_trait;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
use hyprstream_rpc::Subject;
use wasm_bindgen::prelude::*;

use crate::wasm_exports::RpcSession;

// ============================================================================
// Send+Sync wrapper for RpcSession on wasm32
// ============================================================================

/// Wrapper that makes RpcSession usable in Send+Sync contexts.
/// Safe on wasm32 because everything is single-threaded.
struct SessionHandle(Arc<RpcSession>);

// SAFETY: wasm32 is single-threaded — no actual cross-thread sharing occurs.
unsafe impl Send for SessionHandle {}
unsafe impl Sync for SessionHandle {}

impl SessionHandle {
    fn session(&self) -> &RpcSession {
        &self.0
    }
}

impl Clone for SessionHandle {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

// ============================================================================
// Fid state — tracks walked path for deferred RPC calls
// ============================================================================

#[derive(Clone, Debug)]
struct VfsFidState {
    /// Full path components from walk
    path: Vec<String>,
    /// Whether this fid has been opened
    opened: bool,
}

// ============================================================================
// RegistryMount — /srv/registry
// ============================================================================

/// Mount for the registry service.
///
/// Path layout:
///   /srv/registry/              → readdir: list repos by name
///   /srv/registry/{name}        → read: repo JSON (id, url, tracking_ref, worktrees)
///   /srv/registry/{name}/branches → readdir: branch names
///   /srv/registry/{name}/worktrees → readdir: worktree info
///   /srv/registry/{name}/remotes  → readdir: remote info
///   /srv/registry/{name}/status   → read: repo status JSON
pub struct RegistryMount {
    session: SessionHandle,
}

impl RegistryMount {
    pub fn new(session: Arc<RpcSession>) -> Self {
        Self { session: SessionHandle(session) }
    }

    async fn list_repos(&self) -> Result<Vec<serde_json::Value>, MountError> {
        let result = self.session.session().registry_list().await
            .map_err(|e| MountError::Io(format!("{e:?}")))?;
        let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
            .map_err(|e| MountError::Io(e.to_string()))?;

        // unwrap ListResult variant
        if let Some(list) = parsed.get("ListResult") {
            if let Some(arr) = list.as_array() {
                return Ok(arr.clone());
            }
        }
        Ok(Vec::new())
    }

    fn find_repo_id<'a>(repos: &'a [serde_json::Value], name: &str) -> Option<&'a str> {
        repos.iter()
            .find(|r| r.get("name").and_then(|n| n.as_str()) == Some(name))
            .and_then(|r| r.get("id").and_then(|id| id.as_str()))
    }
}

#[async_trait(?Send)]
impl Mount for RegistryMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // Validate path exists (lazily — actual RPC on read)
        Ok(Fid::new(VfsFidState {
            path: components.iter().map(|s| s.to_string()).collect(),
            opened: false,
        }))
    }

    async fn open(&self, fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        if let Some(state) = fid.downcast_mut::<VfsFidState>() {
            state.opened = true;
        }
        Ok(())
    }

    async fn read(&self, fid: &Fid, _offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match state.path.len() {
            0 => Err(MountError::IsDirectory("registry root is a directory".into())),
            1 => {
                // /srv/registry/{name} → get repo details
                let repos = self.list_repos().await?;
                let name = &state.path[0];
                let repo = repos.iter()
                    .find(|r| r.get("name").and_then(|n| n.as_str()) == Some(name))
                    .ok_or_else(|| MountError::NotFound(name.clone()))?;
                Ok(serde_json::to_string_pretty(repo).unwrap_or_default().into_bytes())
            }
            2 => {
                // /srv/registry/{name}/{sub} → read sub-resource
                let repos = self.list_repos().await?;
                let name = &state.path[0];
                let sub = &state.path[1];
                let repo_id = Self::find_repo_id(&repos, name)
                    .ok_or_else(|| MountError::NotFound(name.clone()))?;

                match sub.as_str() {
                    "status" => {
                        let result = self.session.session().registry_repo_status(repo_id).await
                            .map_err(|e| MountError::Io(format!("{e:?}")))?;
                        let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
                            .map_err(|e| MountError::Io(e.to_string()))?;
                        Ok(serde_json::to_string_pretty(&parsed).unwrap_or_default().into_bytes())
                    }
                    _ => Err(MountError::NotFound(format!("{name}/{sub}"))),
                }
            }
            _ => Err(MountError::NotFound(state.path.join("/"))),
        }
    }

    async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        Err(MountError::NotSupported("registry is read-only".into()))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match state.path.len() {
            0 => {
                // /srv/registry/ → list repos
                let repos = self.list_repos().await?;
                Ok(repos.iter().filter_map(|r| {
                    let name = r.get("name")?.as_str()?.to_string();
                    Some(DirEntry { name, is_dir: true, size: 0, stat: None })
                }).collect())
            }
            1 => {
                // /srv/registry/{name}/ → sub-resources
                Ok(vec![
                    DirEntry { name: "branches".into(), is_dir: true, size: 0, stat: None },
                    DirEntry { name: "worktrees".into(), is_dir: true, size: 0, stat: None },
                    DirEntry { name: "remotes".into(), is_dir: true, size: 0, stat: None },
                    DirEntry { name: "status".into(), is_dir: false, size: 0, stat: None },
                ])
            }
            2 => {
                let repos = self.list_repos().await?;
                let name = &state.path[0];
                let sub = &state.path[1];
                let repo_id = Self::find_repo_id(&repos, name)
                    .ok_or_else(|| MountError::NotFound(name.clone()))?;

                match sub.as_str() {
                    "branches" => {
                        let result = self.session.session().registry_repo_list_branches(repo_id).await
                            .map_err(|e| MountError::Io(format!("{e:?}")))?;
                        let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
                            .map_err(|e| MountError::Io(e.to_string()))?;
                        // RepoResult → ListBranches → [string]
                        let branches = parsed.get("RepoResult")
                            .and_then(|r| r.get("ListBranches"))
                            .and_then(|b| b.as_array())
                            .cloned()
                            .unwrap_or_default();
                        Ok(branches.iter().filter_map(|b| {
                            let name = b.as_str()?.to_string();
                            Some(DirEntry { name, is_dir: false, size: 0, stat: None })
                        }).collect())
                    }
                    "worktrees" => {
                        let result = self.session.session().registry_repo_list_worktrees(repo_id).await
                            .map_err(|e| MountError::Io(format!("{e:?}")))?;
                        let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
                            .map_err(|e| MountError::Io(e.to_string()))?;
                        let wts = parsed.get("RepoResult")
                            .and_then(|r| r.get("ListWorktrees"))
                            .and_then(|w| w.as_array())
                            .cloned()
                            .unwrap_or_default();
                        Ok(wts.iter().filter_map(|wt| {
                            let name = wt.get("branchName")?.as_str()?.to_string();
                            Some(DirEntry { name, is_dir: true, size: 0, stat: None })
                        }).collect())
                    }
                    "remotes" => {
                        let result = self.session.session().registry_repo_list_remotes(repo_id).await
                            .map_err(|e| MountError::Io(format!("{e:?}")))?;
                        let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
                            .map_err(|e| MountError::Io(e.to_string()))?;
                        let remotes = parsed.get("RepoResult")
                            .and_then(|r| r.get("ListRemotes"))
                            .and_then(|r| r.as_array())
                            .cloned()
                            .unwrap_or_default();
                        Ok(remotes.iter().filter_map(|rm| {
                            let name = rm.get("name")?.as_str()?.to_string();
                            Some(DirEntry { name, is_dir: false, size: 0, stat: None })
                        }).collect())
                    }
                    _ => Err(MountError::NotFound(format!("{name}/{sub}"))),
                }
            }
            _ => Err(MountError::NotFound(state.path.join("/"))),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let name = state.path.last().map(|s| s.as_str()).unwrap_or("registry");
        let is_dir = state.path.len() != 2 || state.path[1] != "status";
        Ok(Stat {
            qtype: if is_dir { 0x80 } else { 0 },
            size: 0,
            name: name.to_string(),
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {
        // No-op — fid state is self-contained
    }
}

// ============================================================================
// ModelMount — /srv/model
// ============================================================================

/// Mount for the model service.
///
/// Path layout:
///   /srv/model/           → readdir: list loaded models
///   /srv/model/status     → read: all model status JSON
///   /srv/model/health     → read: health check JSON
pub struct ModelMount {
    session: SessionHandle,
}

impl ModelMount {
    pub fn new(session: Arc<RpcSession>) -> Self {
        Self { session: SessionHandle(session) }
    }
}

#[async_trait(?Send)]
impl Mount for ModelMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        Ok(Fid::new(VfsFidState {
            path: components.iter().map(|s| s.to_string()).collect(),
            opened: false,
        }))
    }

    async fn open(&self, fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        if let Some(state) = fid.downcast_mut::<VfsFidState>() {
            state.opened = true;
        }
        Ok(())
    }

    async fn read(&self, fid: &Fid, _offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match state.path.first().map(|s| s.as_str()) {
            Some("status") => {
                let result = self.session.session().model_status(JsValue::from_str("{}")).await
                    .map_err(|e| MountError::Io(format!("{e:?}")))?;
                let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
                    .map_err(|e| MountError::Io(e.to_string()))?;
                Ok(serde_json::to_string_pretty(&parsed).unwrap_or_default().into_bytes())
            }
            Some("health") => {
                let result = self.session.session().model_health_check().await
                    .map_err(|e| MountError::Io(format!("{e:?}")))?;
                let parsed: serde_json::Value = serde_wasm_bindgen::from_value(result)
                    .map_err(|e| MountError::Io(e.to_string()))?;
                Ok(serde_json::to_string_pretty(&parsed).unwrap_or_default().into_bytes())
            }
            None => Err(MountError::IsDirectory("model root is a directory".into())),
            Some(p) => Err(MountError::NotFound(p.into())),
        }
    }

    async fn write(&self, _fid: &Fid, _offset: u64, _data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
        Err(MountError::NotSupported("use ctl for model operations".into()))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        if state.path.is_empty() {
            Ok(vec![
                DirEntry { name: "status".into(), is_dir: false, size: 0, stat: None },
                DirEntry { name: "health".into(), is_dir: false, size: 0, stat: None },
            ])
        } else {
            Err(MountError::NotDirectory(state.path.join("/")))
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid.downcast_ref::<VfsFidState>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let name = state.path.last().map(|s| s.as_str()).unwrap_or("model");
        Ok(Stat {
            qtype: if state.path.is_empty() { 0x80 } else { 0 },
            size: 0,
            name: name.to_string(),
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ============================================================================
// Builder — construct a Namespace with all service mounts
// ============================================================================

/// Build a VFS namespace with mounts backed by RpcSession.
pub fn build_browser_namespace(
    registry_session: Arc<RpcSession>,
    model_session: Arc<RpcSession>,
) -> hyprstream_vfs::Namespace {
    let mut ns = hyprstream_vfs::Namespace::new();

    ns.mount("/srv/registry", Arc::new(RegistryMount::new(registry_session)))
        .expect("mount /srv/registry");
    ns.mount("/srv/model", Arc::new(ModelMount::new(model_session)))
        .expect("mount /srv/model");

    ns
}
