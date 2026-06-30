//! Reusable synthetic-file primitives for hand-built [`Mount`] implementations.
//!
//! `hyprstream-workers-tcl` and `hyprstream-workers-python` (and future
//! language-shell crates) each hand-roll the same two shapes:
//!
//! - a **ctl file**: write triggers an action, the *next* read returns the
//!   latched result (the Plan 9 ctl-file convention — see `eval` in both
//!   crates' mounts).
//! - a **field file**: a scalar read (optionally write) backed by live
//!   interpreter state (`vars/<name>`, `defs/<name>`, `procs/<name>`).
//!
//! This module factors both into reusable, `Send + Sync` primitives so new
//! language/device mounts don't need to re-derive the write→latch→read
//! pattern (or its hazards — see the `unsafe *mut` cast `hyprstream-workers-tcl`
//! used before this module existed) by hand. Modeled after Wanix's
//! `ControlFile` / `FieldFile` devices, adapted to this crate's async
//! [`Mount`] trait and `Box<dyn Any + Send + Sync>` [`Fid`].
//!
//! # Shape
//!
//! [`ControlFile`] and [`FieldFile`] are *descriptors*: long-lived, held by
//! the owning `Mount` (typically behind an `Arc` or directly by value), and
//! they hold the caller-supplied async callback(s). Per-open state — the
//! write→read latch — lives in [`DevFileState`], a small `Fid` payload
//! guarded by a `std::sync::Mutex` (not `parking_lot`, so this module stays
//! wasm-portable; the lock is only ever held synchronously to clone/replace
//! a `Vec<u8>`, never across an `.await`).
//!
//! Both callback shapes take and return owned bytes (`Vec<u8>` in, boxed
//! future of `Vec<u8>`/`Result` out) rather than borrowing — this sidesteps
//! HRTB lifetime issues with `Fn(&[u8]) -> impl Future<..> + '_` trait
//! objects, and mirrors how both existing mounts already shuttle owned
//! `String`/`Vec<u8>` across their command channels.
//!
//! # wasm32 Send-ness
//!
//! [`Mount`] is `#[async_trait(?Send)]` on `wasm32` and `#[async_trait]`
//! (requiring `Send` futures) elsewhere. The callback future type here is
//! gated the same way: `BoxFuture<'static, T>` (`Send`) off wasm32,
//! `LocalBoxFuture<'static, T>` (no `Send` bound) on wasm32. Callers building
//! a [`ControlFile`]/[`FieldFile`] for a native mount must produce `Send`
//! futures (true today for both `hyprstream-workers-tcl` and
//! `hyprstream-workers-python`, which proxy through `tokio::sync` channels).

use parking_lot::Mutex;

use crate::mount::{DirEntry, MountError};

// ─────────────────────────────────────────────────────────────────────────────
// Future type alias (Send off wasm32, ?Send on wasm32 — matches `Mount`)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(target_arch = "wasm32"))]
/// Boxed future returned by device-file callbacks. `Send` off wasm32 to match
/// [`Mount`]'s `#[async_trait]` (non-wasm) bound.
pub type DevFuture<'a, T> = futures::future::BoxFuture<'a, T>;

#[cfg(target_arch = "wasm32")]
/// Boxed future returned by device-file callbacks. No `Send` bound on
/// wasm32, matching [`Mount`]'s `#[async_trait(?Send)]` bound there.
pub type DevFuture<'a, T> = futures::future::LocalBoxFuture<'a, T>;

// ─────────────────────────────────────────────────────────────────────────────
// DevFileState — per-fid write→latch→read buffer
// ─────────────────────────────────────────────────────────────────────────────

/// Per-fid state for a device file: a byte buffer latched by the last write
/// (for [`ControlFile`]) or the last read (for [`FieldFile`] caching).
///
/// Stored as `Fid` payload (`Box<dyn Any + Send + Sync>`); `parking_lot::Mutex`
/// gives interior mutability through the `&Fid` the [`Mount`] trait passes to
/// `read`/`write` without the unsafe `*mut` cast `hyprstream-workers-tcl` used
/// before this module existed.
#[derive(Default)]
pub struct DevFileState {
    latched: Mutex<Vec<u8>>,
}

impl DevFileState {
    /// Fresh, empty state (nothing written/read yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the latched buffer.
    pub fn latch(&self, data: Vec<u8>) {
        *self.latched.lock() = data;
    }

    /// Clone the latched buffer (empty if nothing latched yet).
    pub fn latched(&self) -> Vec<u8> {
        self.latched.lock().clone()
    }
}

/// Slice `data` at `offset`, returning at most the available remainder
/// (9P read semantics: reading past EOF yields an empty result, not an
/// error). `count` is accepted for symmetry with [`Mount::read`] callers
/// but both primitives here return their full latched/rendered buffer from
/// `offset` onward — callers needing chunked reads should slice further.
pub fn read_from_offset(data: &[u8], offset: u64) -> Vec<u8> {
    let start = offset as usize;
    if start >= data.len() {
        Vec::new()
    } else {
        data[start..].to_vec()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ControlFile — write triggers an action, subsequent read returns the result
// ─────────────────────────────────────────────────────────────────────────────

/// A Plan 9-style ctl file: writing bytes invokes a caller-supplied async
/// handler, whose result is latched and served on the next read.
///
/// Generic over the handler closure `F`. Construct with [`ControlFile::new`];
/// the closure receives the written bytes **by value** (`Vec<u8>`, not `&[u8]`)
/// and returns a boxed `'static` future resolving to the response bytes, or a
/// [`MountError`] (rendered as-is — callers that want Wanix-style
/// `"error: ..."` text on the read side should fold that into `F`'s `Ok` arm
/// rather than returning `Err`, exactly as both `TclMount`/`PythonMount`
/// already do for interpreter-level errors).
///
/// Taking owned bytes (rather than `&[u8]`) is a deliberate API choice: a
/// `Fn(&'a [u8]) -> DevFuture<'a, _>` would need to be higher-ranked over
/// `'a` (`for<'a> Fn(&'a [u8]) -> ...`), which closures capturing `move`d
/// state can satisfy in principle but which the compiler frequently fails to
/// infer for boxed closure literals in practice (the closure's `Fn` impl
/// collapses to one concrete lifetime instead of being universally
/// quantified). Owned input avoids the whole class of error and mirrors how
/// both existing mounts already shuttle owned `String`/`Vec<u8>` across their
/// command channels.
///
/// # Fid contract
///
/// Use [`ControlFile::new_fid`] in `Mount::walk` to create the per-open
/// [`Fid`]; call [`ControlFile::handle_write`] from `Mount::write` and
/// [`ControlFile::handle_read`] from `Mount::read`.
pub struct ControlFile<F> {
    handler: F,
}

impl<F> ControlFile<F>
where
    F: Fn(Vec<u8>) -> DevFuture<'static, Result<Vec<u8>, MountError>> + Send + Sync,
{
    /// Build a control file around an async handler.
    pub fn new(handler: F) -> Self {
        Self { handler }
    }

    /// Fresh per-fid latch state for this ctl file (nothing written yet).
    pub fn new_fid(&self) -> DevFileState {
        DevFileState::new()
    }

    /// Handle a `Mount::write`: run the handler over `data`, latch the
    /// result (or an `"error: ..."`-free propagation — the error is returned
    /// directly, NOT latched, matching `Mount::write`'s `Result` surface) into
    /// `state`, and return the byte count written on success.
    pub async fn handle_write(&self, state: &DevFileState, data: &[u8]) -> Result<u32, MountError> {
        let len = data.len();
        let result = (self.handler)(data.to_vec()).await?;
        state.latch(result);
        Ok(len as u32)
    }

    /// Handle a `Mount::read`: serve the latched result (empty if nothing
    /// has been written yet), sliced from `offset`.
    pub fn handle_read(&self, state: &DevFileState, offset: u64) -> Vec<u8> {
        read_from_offset(&state.latched(), offset)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FieldFile — scalar read (and optionally write) backed by live state
// ─────────────────────────────────────────────────────────────────────────────

/// A scalar field file: reading runs a caller-supplied async getter; writing
/// (if a setter is supplied) runs a caller-supplied async setter. Used for
/// `vars/<name>`, `defs/<name>`, `procs/<name>`-style leaves.
///
/// Unlike [`ControlFile`], a `FieldFile` does not require a write before a
/// read — `get` is invoked directly on `Mount::read`. No per-fid latch state
/// is needed (constructed fresh per access), keeping it stateless and cheap
/// to build from a [`DynamicDir`] lookup.
///
/// `S` defaults to [`NoSetter`] (read-only field; [`FieldFile::read_only`]).
/// [`FieldFile::read_write`] takes a real setter closure. Both shapes expose
/// the same `handle_read`/`handle_write` methods — [`NoSetter::handle_write`]
/// always returns [`MountError::NotSupported`], so callers don't need to
/// special-case which constructor was used.
pub struct FieldFile<G, S = NoSetter> {
    get: G,
    set: S,
}

/// A field file's write side: either [`NoSetter`] (read-only) or a real
/// `Fn(Vec<u8>) -> DevFuture<'static, Result<(), MountError>>` closure.
/// Sealed-by-convention — implemented here and via the blanket impl below;
/// not meant to be implemented by downstream crates.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait FieldSetter: Send + Sync {
    /// Apply a write, or reject it (read-only field).
    async fn set(&self, data: Vec<u8>) -> Result<(), MountError>;
}

/// Marker type for a [`FieldFile`] with no setter (read-only field).
pub struct NoSetter;

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl FieldSetter for NoSetter {
    async fn set(&self, _data: Vec<u8>) -> Result<(), MountError> {
        Err(MountError::NotSupported("field is read-only".into()))
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<S> FieldSetter for S
where
    S: Fn(Vec<u8>) -> DevFuture<'static, Result<(), MountError>> + Send + Sync,
{
    async fn set(&self, data: Vec<u8>) -> Result<(), MountError> {
        (self)(data).await
    }
}

impl<G> FieldFile<G, NoSetter>
where
    G: Fn() -> DevFuture<'static, Option<Vec<u8>>> + Send + Sync,
{
    /// Build a read-only field file. `get` returns `None` if the field no
    /// longer exists (e.g. the variable was unset between `walk` and `read`),
    /// surfaced as [`MountError::NotFound`].
    pub fn read_only(get: G) -> Self {
        Self { get, set: NoSetter }
    }
}

impl<G, S> FieldFile<G, S>
where
    G: Fn() -> DevFuture<'static, Option<Vec<u8>>> + Send + Sync,
    S: Fn(Vec<u8>) -> DevFuture<'static, Result<(), MountError>> + Send + Sync,
{
    /// Build a read/write field file.
    pub fn read_write(get: G, set: S) -> Self {
        Self { get, set }
    }
}

impl<G, S> FieldFile<G, S>
where
    G: Fn() -> DevFuture<'static, Option<Vec<u8>>> + Send + Sync,
    S: FieldSetter,
{
    /// Handle a `Mount::read`: run the getter, slice from `offset`.
    pub async fn handle_read(&self, offset: u64, name: &str) -> Result<Vec<u8>, MountError> {
        let data = (self.get)()
            .await
            .ok_or_else(|| MountError::NotFound(name.to_string()))?;
        Ok(read_from_offset(&data, offset))
    }

    /// Handle a `Mount::write`: run the setter (or reject, for [`NoSetter`]).
    pub async fn handle_write(&self, data: &[u8]) -> Result<u32, MountError> {
        self.set.set(data.to_vec()).await?;
        Ok(data.len() as u32)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DynamicDir — a directory of FieldFile-shaped leaves listed by name
// ─────────────────────────────────────────────────────────────────────────────

/// Helper for the `vars/`, `defs/`, `procs/`-style directories both shell
/// mounts expose: a `readdir` that lists live names and a per-name leaf read
/// (and optional write) sourced from the same backing interpreter.
///
/// This is *not* a [`Mount`] itself — it's a pair of async callbacks
/// (`list`, `get`) that a mount's `readdir`/`read` implementations delegate
/// to for a given subtree, keeping the per-mount glue to a `match` on
/// [`Fid`] kind. See `hyprstream-workers-tcl`'s `vars`/`procs` dirs and
/// `hyprstream-workers-python`'s `vars`/`defs` dirs for the call sites.
pub struct DynamicDir<L, G> {
    list: L,
    get: G,
}

impl<L, G> DynamicDir<L, G>
where
    L: Fn() -> DevFuture<'static, Vec<String>> + Send + Sync,
    G: Fn(String) -> DevFuture<'static, Result<Option<Vec<u8>>, MountError>> + Send + Sync,
{
    /// Build a dynamic directory from a name-lister and a per-name getter.
    ///
    /// The getter returns `Result<Option<Vec<u8>>, MountError>` (not just
    /// `Option<Vec<u8>>`) so callers can distinguish a genuine transport/
    /// backend failure (propagated as-is) from "this name doesn't currently
    /// resolve" (`Ok(None)`, surfaced by [`DynamicDir::read`] as
    /// [`MountError::NotFound`]) — e.g. a `vars/<name>` lookup racing an
    /// `unset` is `Ok(None)`, but the interpreter channel being closed is a
    /// distinct `Err`.
    pub fn new(list: L, get: G) -> Self {
        Self { list, get }
    }

    /// `Mount::readdir` for this directory: one flat (non-dir) entry per
    /// live name.
    pub async fn readdir(&self) -> Vec<DirEntry> {
        (self.list)()
            .await
            .into_iter()
            .map(|name| DirEntry {
                name,
                is_dir: false,
                size: 0,
                stat: None,
            })
            .collect()
    }

    /// `Mount::read` for a `<dir>/<name>` leaf: fetch and slice from
    /// `offset`. [`MountError::NotFound`] if the name no longer resolves;
    /// any other error from the getter propagates unchanged.
    pub async fn read(&self, name: &str, offset: u64) -> Result<Vec<u8>, MountError> {
        let data = (self.get)(name.to_string())
            .await?
            .ok_or_else(|| MountError::NotFound(name.to_string()))?;
        Ok(read_from_offset(&data, offset))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — exercise both primitives against a trivial in-memory Mount
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use async_trait::async_trait;
    use hyprstream_rpc::Subject;
    use tokio::sync::Mutex as AsyncMutex;
    use crate::mount::{Fid, Mount, Stat};

    fn subj() -> Subject {
        Subject::new("tester")
    }

    // ── ControlFile standalone tests ───────────────────────────────────────

    #[tokio::test]
    async fn control_file_write_then_read_latches_result() {
        let ctl = ControlFile::new(|data: Vec<u8>| {
            let s = String::from_utf8_lossy(&data).into_owned();
            Box::pin(async move { Ok(format!("echo: {s}").into_bytes()) }) as DevFuture<'static, _>
        });
        let state = ctl.new_fid();

        // Nothing written yet: read returns empty.
        assert!(ctl.handle_read(&state, 0).is_empty());

        let n = ctl.handle_write(&state, b"hello").await.unwrap();
        assert_eq!(n, 5);

        let out = ctl.handle_read(&state, 0);
        assert_eq!(out, b"echo: hello");
    }

    #[tokio::test]
    async fn control_file_offset_slices_latched_result() {
        let ctl = ControlFile::new(|_data: Vec<u8>| {
            Box::pin(async move { Ok(b"0123456789".to_vec()) }) as DevFuture<'static, _>
        });
        let state = ctl.new_fid();
        ctl.handle_write(&state, b"x").await.unwrap();
        assert_eq!(ctl.handle_read(&state, 5), b"56789");
        assert_eq!(ctl.handle_read(&state, 100), b"");
    }

    #[tokio::test]
    async fn control_file_handler_error_propagates_not_latched() {
        let ctl = ControlFile::new(|_data: Vec<u8>| {
            Box::pin(async move { Err(MountError::Io("boom".into())) }) as DevFuture<'static, _>
        });
        let state = ctl.new_fid();
        let res = ctl.handle_write(&state, b"x").await;
        assert!(res.is_err());
        // Nothing latched on error.
        assert!(ctl.handle_read(&state, 0).is_empty());
    }

    #[tokio::test]
    async fn control_file_second_write_replaces_latch() {
        let ctl = ControlFile::new(|data: Vec<u8>| Box::pin(async move { Ok(data) }) as DevFuture<'static, _>);
        let state = ctl.new_fid();
        ctl.handle_write(&state, b"first").await.unwrap();
        assert_eq!(ctl.handle_read(&state, 0), b"first");
        ctl.handle_write(&state, b"second").await.unwrap();
        assert_eq!(ctl.handle_read(&state, 0), b"second");
    }

    // ── FieldFile standalone tests ──────────────────────────────────────────

    #[tokio::test]
    async fn field_file_read_only_get() {
        let field = FieldFile::read_only(|| Box::pin(async { Some(b"42".to_vec()) }) as DevFuture<'static, _>);
        let out = field.handle_read(0, "x").await.unwrap();
        assert_eq!(out, b"42");
    }

    #[tokio::test]
    async fn field_file_read_only_missing_is_not_found() {
        let field = FieldFile::read_only(|| Box::pin(async { None }) as DevFuture<'static, _>);
        let res = field.handle_read(0, "missing").await;
        assert!(matches!(res, Err(MountError::NotFound(_))));
    }

    #[tokio::test]
    async fn field_file_read_only_rejects_write() {
        let field = FieldFile::read_only(|| Box::pin(async { Some(b"v".to_vec()) }) as DevFuture<'static, _>);
        let res = field.handle_write(b"new").await;
        assert!(matches!(res, Err(MountError::NotSupported(_))));
    }

    #[tokio::test]
    async fn field_file_read_write_round_trips_through_shared_state() {
        let store: Arc<AsyncMutex<String>> = Arc::new(AsyncMutex::new("init".to_string()));

        let get_store = store.clone();
        let set_store = store.clone();
        let field = FieldFile::read_write(
            move || {
                let store = get_store.clone();
                Box::pin(async move { Some(store.lock().await.clone().into_bytes()) }) as DevFuture<'static, _>
            },
            move |data: Vec<u8>| {
                let store = set_store.clone();
                Box::pin(async move {
                    *store.lock().await = String::from_utf8_lossy(&data).into_owned();
                    Ok(())
                }) as DevFuture<'static, _>
            },
        );

        assert_eq!(field.handle_read(0, "f").await.unwrap(), b"init");
        let n = field.handle_write(b"updated").await.unwrap();
        assert_eq!(n, 7);
        assert_eq!(field.handle_read(0, "f").await.unwrap(), b"updated");
    }

    // ── DynamicDir standalone tests ─────────────────────────────────────────

    #[tokio::test]
    async fn dynamic_dir_lists_and_reads_names() {
        let dir = DynamicDir::new(
            || Box::pin(async { vec!["a".to_string(), "b".to_string()] }) as DevFuture<'static, _>,
            |name: String| {
                Box::pin(async move {
                    Ok(match name.as_str() {
                        "a" => Some(b"1".to_vec()),
                        "b" => Some(b"2".to_vec()),
                        _ => None,
                    })
                }) as DevFuture<'static, _>
            },
        );

        let entries = dir.readdir().await;
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
        assert!(entries.iter().all(|e| !e.is_dir));

        assert_eq!(dir.read("a", 0).await.unwrap(), b"1");
        assert_eq!(dir.read("b", 0).await.unwrap(), b"2");
        assert!(matches!(dir.read("z", 0).await, Err(MountError::NotFound(_))));
    }

    #[tokio::test]
    async fn dynamic_dir_getter_error_propagates_distinct_from_not_found() {
        // A getter `Err` (e.g. the backing channel is gone) must surface as
        // that error, not be folded into NotFound the way `Ok(None)` is.
        let dir = DynamicDir::new(
            || Box::pin(async { vec!["a".to_string()] }) as DevFuture<'static, _>,
            |_name: String| Box::pin(async { Err(MountError::Io("backend gone".into())) }) as DevFuture<'static, _>,
        );
        assert!(matches!(dir.read("a", 0).await, Err(MountError::Io(_))));
    }

    // ── End-to-end: a trivial Mount built entirely from these primitives ───

    enum DemoFidKind {
        Root,
        Eval,
        VarsDir,
        Var(String),
    }

    struct DemoFid {
        kind: DemoFidKind,
        ctl_state: DevFileState,
    }

    type CtlHandler = Box<dyn Fn(Vec<u8>) -> DevFuture<'static, Result<Vec<u8>, MountError>> + Send + Sync>;
    type ListFn = Box<dyn Fn() -> DevFuture<'static, Vec<String>> + Send + Sync>;
    type GetFn = Box<dyn Fn(String) -> DevFuture<'static, Result<Option<Vec<u8>>, MountError>> + Send + Sync>;

    /// A minimal `Mount` exercising `ControlFile` (an `eval` ctl file that
    /// uppercases its input) and `DynamicDir` (a `vars/` dir over a fixed
    /// in-memory map) — no language interpreter involved, just to prove the
    /// primitives compose into a real `Mount` impl.
    struct DemoMount {
        ctl: ControlFile<CtlHandler>,
        vars: DynamicDir<ListFn, GetFn>,
    }

    fn uppercase_ctl(data: Vec<u8>) -> DevFuture<'static, Result<Vec<u8>, MountError>> {
        let s = String::from_utf8_lossy(&data).to_uppercase();
        Box::pin(async move { Ok(s.into_bytes()) })
    }

    impl DemoMount {
        fn new() -> Self {
            let ctl = ControlFile::new(Box::new(uppercase_ctl) as CtlHandler);

            let vars = DynamicDir::new(
                Box::new(|| {
                    Box::pin(async { vec!["x".to_string(), "y".to_string()] }) as DevFuture<'static, _>
                }) as ListFn,
                Box::new(|name: String| {
                    Box::pin(async move {
                        Ok(match name.as_str() {
                            "x" => Some(b"42".to_vec()),
                            "y" => Some(b"hello".to_vec()),
                            _ => None,
                        })
                    }) as DevFuture<'static, _>
                }) as GetFn,
            );

            Self { ctl, vars }
        }
    }

    #[async_trait]
    impl Mount for DemoMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            let kind = match components {
                [] => DemoFidKind::Root,
                ["eval"] => DemoFidKind::Eval,
                ["vars"] => DemoFidKind::VarsDir,
                ["vars", name] => DemoFidKind::Var((*name).to_owned()),
                _ => return Err(MountError::NotFound(components.join("/"))),
            };
            Ok(Fid::new(DemoFid {
                kind,
                ctl_state: DevFileState::new(),
            }))
        }

        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }

        async fn read(&self, fid: &Fid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<DemoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match &inner.kind {
                DemoFidKind::Eval => Ok(self.ctl.handle_read(&inner.ctl_state, offset)),
                DemoFidKind::Var(name) => self.vars.read(name, offset).await,
                DemoFidKind::Root | DemoFidKind::VarsDir => Err(MountError::IsDirectory("use readdir".into())),
            }
        }

        async fn write(&self, fid: &Fid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            let inner = fid.downcast_ref::<DemoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match &inner.kind {
                DemoFidKind::Eval => self.ctl.handle_write(&inner.ctl_state, data).await,
                _ => Err(MountError::NotSupported("read-only".into())),
            }
        }

        async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<DemoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match &inner.kind {
                DemoFidKind::Root => Ok(vec![
                    DirEntry { name: "eval".into(), is_dir: false, size: 0, stat: None },
                    DirEntry { name: "vars".into(), is_dir: true, size: 0, stat: None },
                ]),
                DemoFidKind::VarsDir => Ok(self.vars.readdir().await),
                _ => Err(MountError::NotDirectory("not a directory".into())),
            }
        }

        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid.downcast_ref::<DemoFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let (name, qtype) = match &inner.kind {
                DemoFidKind::Root => ("demo".to_string(), 0x80),
                DemoFidKind::Eval => ("eval".to_string(), 0),
                DemoFidKind::VarsDir => ("vars".to_string(), 0x80),
                DemoFidKind::Var(n) => (n.clone(), 0),
            };
            Ok(Stat { qtype, size: 0, name, mtime: 0 })
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    #[tokio::test]
    async fn demo_mount_eval_ctl_round_trip() {
        let mount = DemoMount::new();
        let s = subj();
        let mut fid = mount.walk(&["eval"], &s).await.unwrap();
        mount.open(&mut fid, 1, &s).await.unwrap();
        mount.write(&fid, 0, b"hello", &s).await.unwrap();
        let out = mount.read(&fid, 0, 4096, &s).await.unwrap();
        assert_eq!(out, b"HELLO");
    }

    #[tokio::test]
    async fn demo_mount_vars_dir_listing_and_leaves() {
        let mount = DemoMount::new();
        let s = subj();

        let dir_fid = mount.walk(&["vars"], &s).await.unwrap();
        let entries = mount.readdir(&dir_fid, &s).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));

        let x_fid = mount.walk(&["vars", "x"], &s).await.unwrap();
        let out = mount.read(&x_fid, 0, 4096, &s).await.unwrap();
        assert_eq!(out, b"42");

        let missing = mount.walk(&["vars", "z"], &s).await.unwrap();
        assert!(mount.read(&missing, 0, 4096, &s).await.is_err());
    }

    #[tokio::test]
    async fn demo_mount_readdir_root() {
        let mount = DemoMount::new();
        let s = subj();
        let fid = mount.walk(&[], &s).await.unwrap();
        let entries = mount.readdir(&fid, &s).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"eval"));
        assert!(names.contains(&"vars"));
    }
}
