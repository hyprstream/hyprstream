//! `/k8s` VFS projection (K7, #782) — Kubernetes objects as Plan 9 files.
//!
//! This is the **Kubernetes → hyprstream** direction of epic #778's mutual
//! namespace binding: a [`hyprstream_vfs::Mount`] that projects live cluster
//! objects as files, so a Pod becomes a controllable file and `cat`/`echo` from
//! the Tcl/Python shells drive the cluster. It is the `/exec/instances` pattern
//! (epic #608, [`hyprstream-workers`]'s `ExecMount`) applied over a kube-rs
//! watch instead of a `SandboxPool`.
//!
//! ## Layout
//!
//! ```text
//! <root>/pods/<ns>/<name>/status      # read: the object's live `.status` (watch-backed)
//! <root>/pods/<ns>/<name>/ctl         # write: delete|evict (needs a wired actuator + RBAC)
//! <root>/pods/<ns>/<name>/log         # read: log stream (placeholder; live-log follow-up)
//! <root>/pods/<ns>/<name>/watch       # read: current resourceVersion + cache generation
//! <root>/<group>/<kind>/<ns>/<name>/… # generic typed projection (incl. *.hyprstream.io CRDs)
//! ```
//!
//! **Core-group resources appear at the root by plural kind** (`pods`), because
//! the core group has an empty API-group name; **grouped resources nest under
//! their API group** (`models.hyprstream.io/models/…`). A first path component
//! containing a `.` is treated as an API group; otherwise it is a core plural
//! kind — deterministic, matching Kubernetes' own core-vs-grouped split.
//!
//! ## Mount point is a binding decision, not a constant
//!
//! Nothing in this mount hardcodes `/k8s`. All paths are resolved *relative to
//! the mount root*; the assembler binds this wherever it likes
//! ([`hyprstream_vfs::Namespace::bind_mount`]). `/k8s` in the docs above is
//! illustrative only.
//!
//! ## MAC posture (D2 deny-unlabeled floor)
//!
//! Per the ratified interface policy (epic #547, "fill with deny/clamp"), the
//! Kubernetes cluster is a foreign perimeter *outside* our TCB, so:
//!
//! - **Objects with no resolvable label are denied** — [`resolve_label`] returns
//!   `None` and the mount fails closed (`PermissionDenied`), never serving an
//!   "unrestricted" default. This is the [`hyprstream_rpc::auth::mac`]
//!   deny-unlabeled principle applied at the import boundary.
//! - **A declared label is clamped up to the importing boundary's floor** via
//!   [`hyprstream_rpc::auth::mac::import_label`] (D2): a permissive
//!   self-assertion on a k8s object can only make the effective label *more*
//!   restrictive, never less.
//!
//! Like every `Mount` today, this mount **threads the caller [`Subject`] but does
//! not itself authorize** — the MAC reference monitor (S2, #568) is the eventual
//! per-op enforcement point. The deny-unlabeled floor here is the object-label
//! half of that story (the object side is what this mount owns), landed early so
//! an unlabeled cluster object is unreadable through the projection from day one.
//!
//! ## Watch backing & follow-ups
//!
//! [`run_watch`] drives a kube-rs `watcher` into the [`ProjectionCache`], so
//! reads are served from cache with no per-read API-server round trip. The cache
//! carries a monotonic `generation` bumped on every apply/delete for
//! invalidation signalling. Live-cluster watch end-to-end (kind, real RBAC) is a
//! **follow-up** — this crate's tests exercise the mount against fixture objects
//! fed straight into the cache, never a live client.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::{Mutex as PmMutex, RwLock};
use serde_json::Value;

use hyprstream_rpc::auth::mac::{import_label, Assurance, CompartmentSet, Level, SecurityLabel};
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};

const QTDIR: u8 = 0x80;

/// Annotation a projected object may carry to declare its MAC sensitivity level.
///
/// The value is one of `public|internal|confidential|secret` (see [`Level`]).
/// An object without this annotation is **unlabeled** and denied (fail-closed);
/// there is deliberately no permissive default.
pub const LEVEL_ANNOTATION: &str = "mac.hyprstream.io/level";

/// The four per-object files projected under `<…>/<ns>/<name>/`.
const OBJECT_FILES: [&str; 4] = ["status", "ctl", "log", "watch"];

// ─────────────────────────────────────────────────────────────────────────────
// Object identity & cache
// ─────────────────────────────────────────────────────────────────────────────

/// Fully-qualified identity of a projected object.
///
/// `group` is empty for core-group resources (Pods, ConfigMaps…). `kind` is the
/// lowercase *plural* (`pods`, `models`) — the directory name, matching the
/// projection layout.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObjectKey {
    /// API group; empty string for the core group.
    pub group: String,
    /// Lowercase plural kind (`pods`, `models`).
    pub kind: String,
    /// Namespace.
    pub namespace: String,
    /// Object name.
    pub name: String,
}

/// A cached projected object: its JSON plus the watch resourceVersion it came
/// from.
#[derive(Clone, Debug)]
struct CachedObject {
    /// Full object JSON (as delivered by the watch / fed by a fixture).
    json: Value,
    /// The object's `metadata.resourceVersion`, surfaced by the `watch` file.
    resource_version: String,
}

/// Watch-backed store of projected objects.
///
/// Populated by [`run_watch`] in production, or directly by tests via
/// [`ProjectionCache::upsert`]. Every mutation bumps [`generation`], the
/// invalidation signal surfaced through the `watch` file.
///
/// [`generation`]: ProjectionCache::generation
#[derive(Debug, Default)]
pub struct ProjectionCache {
    objects: RwLock<BTreeMap<ObjectKey, CachedObject>>,
    generation: AtomicU64,
}

impl ProjectionCache {
    /// A new, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Current invalidation generation (bumped on every apply/delete).
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Insert or replace an object, bumping the generation.
    ///
    /// `json` is the full object; namespace/name/resourceVersion are read out of
    /// its `metadata`. An object with no `metadata.name` is ignored (nothing to
    /// key on).
    pub fn upsert(&self, group: &str, kind_plural: &str, json: Value) {
        let meta = json.get("metadata");
        let name = meta
            .and_then(|m| m.get("name"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        let Some(name) = name else {
            return;
        };
        // Cluster-scoped objects have no namespace; project them under the
        // reserved pseudo-namespace `_cluster` so the `<ns>` path level is total.
        let namespace = meta
            .and_then(|m| m.get("namespace"))
            .and_then(Value::as_str)
            .unwrap_or("_cluster")
            .to_owned();
        let resource_version = meta
            .and_then(|m| m.get("resourceVersion"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();

        let key = ObjectKey {
            group: group.to_owned(),
            kind: kind_plural.to_owned(),
            namespace,
            name,
        };
        self.objects.write().insert(
            key,
            CachedObject {
                json,
                resource_version,
            },
        );
        self.generation.fetch_add(1, Ordering::SeqCst);
    }

    /// Remove an object by key, bumping the generation if it was present.
    pub fn remove(&self, key: &ObjectKey) {
        if self.objects.write().remove(key).is_some() {
            self.generation.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn get(&self, key: &ObjectKey) -> Option<CachedObject> {
        self.objects.read().get(key).cloned()
    }

    /// Distinct top-level directory entries: the plural kind for core-group
    /// objects, the API group for grouped objects.
    fn top_level(&self) -> Vec<String> {
        let mut seen: Vec<String> = self
            .objects
            .read()
            .keys()
            .map(|k| {
                if k.group.is_empty() {
                    k.kind.clone()
                } else {
                    k.group.clone()
                }
            })
            .collect();
        seen.sort();
        seen.dedup();
        seen
    }

    /// Distinct plural kinds present in `group`.
    fn kinds_in_group(&self, group: &str) -> Vec<String> {
        let mut kinds: Vec<String> = self
            .objects
            .read()
            .keys()
            .filter(|k| k.group == group)
            .map(|k| k.kind.clone())
            .collect();
        kinds.sort();
        kinds.dedup();
        kinds
    }

    /// Distinct namespaces holding objects of `(group, kind)`.
    fn namespaces(&self, group: &str, kind: &str) -> Vec<String> {
        let mut nss: Vec<String> = self
            .objects
            .read()
            .keys()
            .filter(|k| k.group == group && k.kind == kind)
            .map(|k| k.namespace.clone())
            .collect();
        nss.sort();
        nss.dedup();
        nss
    }

    /// Object names of `(group, kind, namespace)`.
    fn names(&self, group: &str, kind: &str, namespace: &str) -> Vec<String> {
        let mut names: Vec<String> = self
            .objects
            .read()
            .keys()
            .filter(|k| k.group == group && k.kind == kind && k.namespace == namespace)
            .map(|k| k.name.clone())
            .collect();
        names.sort();
        names
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MAC object-label resolution (D2 deny-unlabeled floor)
// ─────────────────────────────────────────────────────────────────────────────

/// Parse the [`LEVEL_ANNOTATION`] value into a [`Level`].
fn parse_level(s: &str) -> Option<Level> {
    match s.trim() {
        "public" => Some(Level::Public),
        "internal" => Some(Level::Internal),
        "confidential" => Some(Level::Confidential),
        "secret" => Some(Level::Secret),
        _ => None,
    }
}

/// Resolve a cached object's **effective** security label, applying the D2
/// import clamp against `import_floor`.
///
/// Returns `None` when the object declares no parseable [`LEVEL_ANNOTATION`] —
/// i.e. it is **unlabeled**, which the mount treats as *deny* (fail-closed).
/// There is no permissive default: an unlabeled cluster object is unreadable
/// through the projection.
///
/// A declared level is honored only up to the boundary floor: the k8s cluster
/// is a foreign perimeter, so the object's self-asserted label is a *hint*
/// clamped up via [`import_label`] — `effective = join(import_floor, declared)`,
/// never below the floor. The declared assurance is pinned to
/// [`Assurance::Classical`] because a plain cluster annotation carries no
/// verified post-quantum anchor.
pub fn resolve_label(json: &Value, import_floor: SecurityLabel) -> Option<SecurityLabel> {
    let level = json
        .get("metadata")
        .and_then(|m| m.get("annotations"))
        .and_then(|a| a.get(LEVEL_ANNOTATION))
        .and_then(Value::as_str)
        .and_then(parse_level)?;
    let declared = SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY);
    Some(import_label(import_floor, declared))
}

// ─────────────────────────────────────────────────────────────────────────────
// ctl actuator
// ─────────────────────────────────────────────────────────────────────────────

/// A `ctl` verb: the lifecycle actions the projection exposes.
///
/// `create` is deliberately absent — creation belongs to the k8s backend (K4a)
/// and the operator (K5), under the single-writer rule (#778).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CtlVerb {
    /// Delete the object (`kubectl delete`).
    Delete,
    /// Evict the object (Pod eviction API).
    Evict,
}

impl CtlVerb {
    fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "delete" => Some(Self::Delete),
            "evict" => Some(Self::Evict),
            _ => None,
        }
    }
}

/// The side-effecting half of `ctl`: performs a [`CtlVerb`] against the cluster.
///
/// Kept as a trait so the read-only projection (this issue's core deliverable)
/// carries no live-cluster dependency, and so tests exercise ctl parsing/routing
/// with a fake. A live `kube::Client`-backed implementation (delete/evict via
/// the API) lands with the RBAC-scoped write path; until one is wired, the mount
/// reports ctl as deferred.
#[async_trait]
pub trait CtlActuator: Send + Sync {
    /// Apply `verb` to the object identified by `key`.
    async fn apply(&self, key: &ObjectKey, verb: CtlVerb) -> Result<String, MountError>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fid types
// ─────────────────────────────────────────────────────────────────────────────

/// Which node a fid refers to. Directory nodes carry just enough of the key to
/// enumerate children; object-file nodes carry the full key + which file.
#[derive(Clone, Debug)]
enum Node {
    /// Mount root.
    Root,
    /// An API-group directory (grouped resources only).
    Group { group: String },
    /// A kind directory (`<group>/<kind>` or core `<kind>`).
    Kind { group: String, kind: String },
    /// A namespace directory.
    Namespace {
        group: String,
        kind: String,
        namespace: String,
    },
    /// An object directory `<…>/<ns>/<name>`.
    Object { key: ObjectKey },
    /// A per-object file (`status`/`ctl`/`log`/`watch`).
    File { key: ObjectKey, file: String },
}

/// Fid state.
struct K8sFid {
    node: Node,
    /// Latched result of a `ctl` write, served by the following read on the same
    /// fid. `parking_lot::Mutex` gives safe interior mutability through the
    /// `&Fid` that `Mount::read`/`write` receive (the exact pattern `ExecMount`
    /// uses). Empty for non-`ctl` nodes.
    write_buf: PmMutex<Vec<u8>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// K8sMount
// ─────────────────────────────────────────────────────────────────────────────

/// VFS mount projecting Kubernetes objects as Plan 9 files (K7, #782).
///
/// Construct with [`K8sMount::new`], feed it with [`run_watch`] (production) or
/// [`ProjectionCache::upsert`] (tests), and bind it anywhere in a
/// [`hyprstream_vfs::Namespace`].
pub struct K8sMount {
    cache: Arc<ProjectionCache>,
    /// D2 importing-boundary floor. Every object label is clamped *up* to at
    /// least this; unlabeled objects are denied outright.
    import_floor: SecurityLabel,
    /// Optional live actuator for `ctl` verbs. `None` ⇒ ctl is reported as
    /// deferred (read-only projection).
    actuator: Option<Arc<dyn CtlActuator>>,
}

impl K8sMount {
    /// Create a mount over `cache`, clamping every object label up to
    /// `import_floor` and denying unlabeled objects. No `ctl` actuator (read-only
    /// projection).
    pub fn new(cache: Arc<ProjectionCache>, import_floor: SecurityLabel) -> Self {
        Self {
            cache,
            import_floor,
            actuator: None,
        }
    }

    /// Attach a live `ctl` actuator (enables delete/evict).
    ///
    /// # Precondition
    ///
    /// Do not wire a production Kubernetes client here until the #568
    /// reference-monitor PEP authorizes mutating `ctl` verbs per caller. This
    /// mount currently enforces only the object-label import floor before
    /// dispatching to the actuator; without #568, any fid-holder that can reach
    /// a labeled object's `ctl` file could invoke verbs such as `delete` or
    /// `evict`.
    #[must_use]
    pub fn with_actuator(mut self, actuator: Arc<dyn CtlActuator>) -> Self {
        self.actuator = Some(actuator);
        self
    }

    /// Resolve path components to a [`Node`], validating object existence.
    ///
    /// A first component containing `.` is an API group; otherwise it is a
    /// core-group plural kind.
    fn resolve(&self, components: &[&str]) -> Result<Node, MountError> {
        let is_group = |c: &str| c.contains('.');
        let node = match components {
            [] => Node::Root,
            [group] if is_group(group) => Node::Group {
                group: (*group).to_owned(),
            },
            [kind] => Node::Kind {
                group: String::new(),
                kind: (*kind).to_owned(),
            },
            [group, kind] if is_group(group) => Node::Kind {
                group: (*group).to_owned(),
                kind: (*kind).to_owned(),
            },
            [kind, ns] => Node::Namespace {
                group: String::new(),
                kind: (*kind).to_owned(),
                namespace: (*ns).to_owned(),
            },
            [group, kind, ns] if is_group(group) => Node::Namespace {
                group: (*group).to_owned(),
                kind: (*kind).to_owned(),
                namespace: (*ns).to_owned(),
            },
            [kind, ns, name] => Node::Object {
                key: ObjectKey {
                    group: String::new(),
                    kind: (*kind).to_owned(),
                    namespace: (*ns).to_owned(),
                    name: (*name).to_owned(),
                },
            },
            [group, kind, ns, name] if is_group(group) => Node::Object {
                key: ObjectKey {
                    group: (*group).to_owned(),
                    kind: (*kind).to_owned(),
                    namespace: (*ns).to_owned(),
                    name: (*name).to_owned(),
                },
            },
            [kind, ns, name, file] => Node::File {
                key: ObjectKey {
                    group: String::new(),
                    kind: (*kind).to_owned(),
                    namespace: (*ns).to_owned(),
                    name: (*name).to_owned(),
                },
                file: (*file).to_owned(),
            },
            [group, kind, ns, name, file] if is_group(group) => Node::File {
                key: ObjectKey {
                    group: (*group).to_owned(),
                    kind: (*kind).to_owned(),
                    namespace: (*ns).to_owned(),
                    name: (*name).to_owned(),
                },
                file: (*file).to_owned(),
            },
            _ => return Err(MountError::NotFound(components.join("/"))),
        };

        // Validate object existence for object/file nodes (a walk into an
        // unknown object 404s, like any 9P namespace); validate the file name.
        match &node {
            Node::Object { key } => {
                if self.cache.get(key).is_none() {
                    return Err(MountError::NotFound(components.join("/")));
                }
            }
            Node::File { key, file } => {
                if !OBJECT_FILES.contains(&file.as_str()) {
                    return Err(MountError::NotFound(components.join("/")));
                }
                if self.cache.get(key).is_none() {
                    return Err(MountError::NotFound(components.join("/")));
                }
            }
            _ => {}
        }
        Ok(node)
    }

    /// Load a cached object, enforcing the deny-unlabeled MAC floor.
    ///
    /// Returns the object JSON + resourceVersion once the object both exists and
    /// resolves to a label. An unlabeled object yields `PermissionDenied`
    /// (fail-closed) rather than being served.
    fn load_labeled(&self, key: &ObjectKey) -> Result<CachedObject, MountError> {
        let obj = self
            .cache
            .get(key)
            .ok_or_else(|| MountError::NotFound(object_path(key)))?;
        if resolve_label(&obj.json, self.import_floor).is_none() {
            return Err(MountError::PermissionDenied(format!(
                "{}: object carries no resolvable MAC label — denied (fail-closed)",
                object_path(key)
            )));
        }
        Ok(obj)
    }

    fn render_status(&self, key: &ObjectKey) -> Result<Vec<u8>, MountError> {
        let obj = self.load_labeled(key)?;
        let status = obj.json.get("status").cloned().unwrap_or(Value::Null);
        let text =
            serde_json::to_string_pretty(&status).map_err(|e| MountError::Io(e.to_string()))?;
        Ok(format!("{text}\n").into_bytes())
    }

    fn render_watch(&self, key: &ObjectKey) -> Result<Vec<u8>, MountError> {
        let obj = self.load_labeled(key)?;
        // Snapshot only: current resourceVersion + cache generation. Streaming
        // watch events out of this file is the live-cluster follow-up (#782).
        Ok(format!(
            "resourceVersion={}\ngeneration={}\n",
            obj.resource_version,
            self.cache.generation()
        )
        .into_bytes())
    }

    fn render_log(&self, key: &ObjectKey) -> Result<Vec<u8>, MountError> {
        // Enforce the label floor even for the placeholder.
        let _ = self.load_labeled(key)?;
        Ok(format!(
            "log streaming for {} is not yet wired — live pod-log follow-up (#782)\n",
            object_path(key)
        )
        .into_bytes())
    }

    async fn apply_ctl(&self, key: &ObjectKey, data: &[u8]) -> Vec<u8> {
        // Fail-closed on unlabeled objects here too: a write action against an
        // object we cannot label is denied.
        if let Err(e) = self.load_labeled(key) {
            return format!("error: {e}\n").into_bytes();
        }
        let text = String::from_utf8_lossy(data);
        let Some(verb) = CtlVerb::parse(&text) else {
            return format!("error: unknown ctl verb: {}\n", text.trim()).into_bytes();
        };
        match &self.actuator {
            Some(actuator) => match actuator.apply(key, verb).await {
                Ok(msg) => format!("ok: {msg}\n").into_bytes(),
                Err(e) => format!("error: {e}\n").into_bytes(),
            },
            None => format!(
                "error: ctl {verb:?} requires a live cluster binding — deferred (read-only projection)\n"
            )
            .into_bytes(),
        }
    }
}

/// Render an [`ObjectKey`] back to its projection path (for error messages).
fn object_path(key: &ObjectKey) -> String {
    if key.group.is_empty() {
        format!("{}/{}/{}", key.kind, key.namespace, key.name)
    } else {
        format!("{}/{}/{}/{}", key.group, key.kind, key.namespace, key.name)
    }
}

fn file_entries() -> Vec<DirEntry> {
    OBJECT_FILES
        .iter()
        .map(|name| DirEntry {
            name: (*name).to_owned(),
            is_dir: false,
            size: 0,
            stat: None,
        })
        .collect()
}

fn dir_entries(names: Vec<String>) -> Vec<DirEntry> {
    names
        .into_iter()
        .map(|name| DirEntry {
            name,
            is_dir: true,
            size: 0,
            stat: None,
        })
        .collect()
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Mount for K8sMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // NOTE on `Subject`: threaded per the Mount contract. This mount does
        // not itself authorize (consistent with every Mount today) — the MAC
        // reference monitor (#568) is the eventual per-op enforcement point. The
        // object-label deny-unlabeled floor is applied on read/ctl below.
        let node = self.resolve(components)?;
        Ok(Fid::new(K8sFid {
            node,
            write_buf: PmMutex::new(Vec::new()),
        }))
    }

    async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        _count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let inner = fid
            .downcast_ref::<K8sFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".to_owned()))?;

        let data = match &inner.node {
            Node::File { key, file } => match file.as_str() {
                "status" => self.render_status(key)?,
                "watch" => self.render_watch(key)?,
                "log" => self.render_log(key)?,
                "ctl" => inner.write_buf.lock().clone(),
                other => {
                    return Err(MountError::NotFound(other.to_owned()));
                }
            },
            _ => return Err(MountError::IsDirectory("use readdir".to_owned())),
        };

        let start = offset as usize;
        if start >= data.len() {
            return Ok(Vec::new());
        }
        Ok(data[start..].to_vec())
    }

    async fn write(
        &self,
        fid: &Fid,
        _offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        let inner = fid
            .downcast_ref::<K8sFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".to_owned()))?;

        match &inner.node {
            Node::File { key, file } if file == "ctl" => {
                let response = self.apply_ctl(key, data).await;
                *inner.write_buf.lock() = response;
                Ok(data.len() as u32)
            }
            _ => Err(MountError::NotSupported("read-only".to_owned())),
        }
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<K8sFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".to_owned()))?;

        match &inner.node {
            Node::Root => Ok(dir_entries(self.cache.top_level())),
            Node::Group { group } => Ok(dir_entries(self.cache.kinds_in_group(group))),
            Node::Kind { group, kind } => Ok(dir_entries(self.cache.namespaces(group, kind))),
            Node::Namespace {
                group,
                kind,
                namespace,
            } => Ok(dir_entries(self.cache.names(group, kind, namespace))),
            Node::Object { .. } => Ok(file_entries()),
            Node::File { .. } => Err(MountError::NotDirectory("object file".to_owned())),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<K8sFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".to_owned()))?;

        let (name, qtype) = match &inner.node {
            Node::Root => (String::new(), QTDIR),
            Node::Group { group } => (group.clone(), QTDIR),
            Node::Kind { kind, .. } => (kind.clone(), QTDIR),
            Node::Namespace { namespace, .. } => (namespace.clone(), QTDIR),
            Node::Object { key } => (key.name.clone(), QTDIR),
            Node::File { file, .. } => (file.clone(), 0),
        };
        Ok(Stat::unknown_qid(qtype, 0, name, 0))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Live watch driver (needs the kube client + runtime)
// ─────────────────────────────────────────────────────────────────────────────

/// Drive a kube-rs `watcher` for one resource into `cache`, forever.
///
/// One call per projected GVK; spawn each on its own task. The `watcher`
/// recovers dropped connections internally (relisting from the last
/// resourceVersion), so a transient API-server blip does not need handling here;
/// a hard stream error is propagated to the caller.
///
/// Watch end-to-end against a live cluster (kind) is the follow-up validation
/// path; the mount's own behaviour is covered offline by fixture tests.
pub async fn run_watch(
    client: kube::Client,
    resource: kube::api::ApiResource,
    cache: Arc<ProjectionCache>,
) -> Result<(), kube::runtime::watcher::Error> {
    use futures::{StreamExt, TryStreamExt};
    use kube::api::{Api, DynamicObject};
    use kube::runtime::watcher::{self, Event};

    let group = resource.group.clone();
    let kind_plural = resource.plural.clone();
    let api: Api<DynamicObject> = Api::all_with(client, &resource);
    // `watcher` is both a module (`Config`, `Event`) and a function; call the
    // function via its module path to avoid the name collision.
    let mut stream = watcher::watcher(api, watcher::Config::default()).boxed();

    while let Some(event) = stream.try_next().await? {
        match event {
            Event::Apply(obj) | Event::InitApply(obj) => {
                if let Ok(json) = serde_json::to_value(&obj) {
                    cache.upsert(&group, &kind_plural, json);
                }
            }
            Event::Delete(obj) => {
                let key = ObjectKey {
                    group: group.clone(),
                    kind: kind_plural.clone(),
                    namespace: obj
                        .metadata
                        .namespace
                        .clone()
                        .unwrap_or_else(|| "_cluster".to_owned()),
                    name: obj.metadata.name.clone().unwrap_or_default(),
                };
                cache.remove(&key);
            }
            Event::Init | Event::InitDone => {}
        }
    }
    Ok(())
}

/// Build the [`kube::api::ApiResource`] for core-group Pods — the canonical
/// `pods/<ns>/<name>` branch of the projection.
pub fn pod_api_resource() -> kube::api::ApiResource {
    use kube::core::GroupVersionKind;
    kube::api::ApiResource::from_gvk_with_plural(&GroupVersionKind::gvk("", "v1", "Pod"), "pods")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — fixtures only, never a live cluster.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use serde_json::json;

    fn floor() -> SecurityLabel {
        SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn subject() -> Subject {
        Subject::anonymous()
    }

    /// A labeled Pod fixture (carries the MAC level annotation).
    fn labeled_pod(ns: &str, name: &str, phase: &str) -> Value {
        json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "namespace": ns,
                "resourceVersion": "42",
                "annotations": { LEVEL_ANNOTATION: "confidential" }
            },
            "spec": {},
            "status": { "phase": phase }
        })
    }

    /// A Pod fixture with NO MAC annotation → unlabeled → must be denied.
    fn unlabeled_pod(ns: &str, name: &str) -> Value {
        json!({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": { "name": name, "namespace": ns, "resourceVersion": "7" },
            "spec": {},
            "status": { "phase": "Running" }
        })
    }

    fn mount_with(objects: &[(&str, &str, Value)]) -> K8sMount {
        let cache = Arc::new(ProjectionCache::new());
        for (group, kind, json) in objects {
            cache.upsert(group, kind, json.clone());
        }
        K8sMount::new(cache, floor())
    }

    #[tokio::test]
    async fn root_lists_core_kind_and_group() {
        let mount = mount_with(&[
            ("", "pods", labeled_pod("default", "web", "Running")),
            (
                "models.hyprstream.io",
                "models",
                json!({"metadata": {"name": "qwen", "namespace": "ml",
                    "annotations": {LEVEL_ANNOTATION: "internal"}}}),
            ),
        ]);
        let fid = mount.walk(&[], &subject()).await.unwrap();
        let names: Vec<String> = mount
            .readdir(&fid, &subject())
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        // core kind appears at root; grouped resource nests under its group.
        assert!(names.contains(&"pods".to_owned()), "got {names:?}");
        assert!(
            names.contains(&"models.hyprstream.io".to_owned()),
            "got {names:?}"
        );
    }

    #[tokio::test]
    async fn walk_down_to_pod_and_readdir_files() {
        let mount = mount_with(&[("", "pods", labeled_pod("default", "web", "Running"))]);
        // pods → default → web
        let names: Vec<String> = {
            let fid = mount.walk(&["pods"], &subject()).await.unwrap();
            mount
                .readdir(&fid, &subject())
                .await
                .unwrap()
                .into_iter()
                .map(|e| e.name)
                .collect()
        };
        assert_eq!(names, vec!["default".to_owned()]);

        let obj_fid = mount
            .walk(&["pods", "default", "web"], &subject())
            .await
            .unwrap();
        let files: Vec<String> = mount
            .readdir(&obj_fid, &subject())
            .await
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        for f in ["status", "ctl", "log", "watch"] {
            assert!(files.contains(&f.to_owned()), "missing {f} in {files:?}");
        }
    }

    #[tokio::test]
    async fn status_reflects_cached_object() {
        let mount = mount_with(&[("", "pods", labeled_pod("default", "web", "Running"))]);
        let mut fid = mount
            .walk(&["pods", "default", "web", "status"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let data = mount.read(&fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.contains("Running"), "got: {text}");
    }

    #[tokio::test]
    async fn status_reflects_watch_update() {
        let cache = Arc::new(ProjectionCache::new());
        cache.upsert("", "pods", labeled_pod("default", "web", "Pending"));
        let mount = K8sMount::new(Arc::clone(&cache), floor());

        // Simulate a watch event flipping the phase.
        cache.upsert("", "pods", labeled_pod("default", "web", "Running"));

        let mut fid = mount
            .walk(&["pods", "default", "web", "status"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.contains("Running"), "got: {text}");
        assert!(!text.contains("Pending"), "stale: {text}");
    }

    #[tokio::test]
    async fn watch_file_reports_resource_version_and_generation() {
        let mount = mount_with(&[("", "pods", labeled_pod("default", "web", "Running"))]);
        let mut fid = mount
            .walk(&["pods", "default", "web", "watch"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.contains("resourceVersion=42"), "got: {text}");
        assert!(text.contains("generation="), "got: {text}");
    }

    /// The MAC floor: an object with no resolvable label is denied, never served.
    #[tokio::test]
    async fn unlabeled_object_is_denied_fail_closed() {
        let mount = mount_with(&[("", "pods", unlabeled_pod("default", "secret-pod"))]);
        let mut fid = mount
            .walk(&["pods", "default", "secret-pod", "status"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let result = mount.read(&fid, 0, 4096, &subject()).await;
        assert!(
            matches!(result, Err(MountError::PermissionDenied(_))),
            "unlabeled object must be denied, got: {result:?}"
        );
    }

    /// The D2 import clamp raises a permissive self-assertion up to the floor.
    #[test]
    fn resolve_label_clamps_declared_up_to_floor() {
        // Object declares `public`, but the importing floor is `Internal`.
        let obj = json!({
            "metadata": { "annotations": { LEVEL_ANNOTATION: "public" } }
        });
        let label = resolve_label(&obj, floor()).expect("declared label resolves");
        assert_eq!(
            label.level,
            Level::Internal,
            "declared public must clamp up to the floor"
        );

        // A more-restrictive declaration is honored as-is.
        let obj_secret = json!({
            "metadata": { "annotations": { LEVEL_ANNOTATION: "secret" } }
        });
        let label = resolve_label(&obj_secret, floor()).expect("declared label resolves");
        assert_eq!(label.level, Level::Secret);

        // No annotation → unlabeled → None (deny).
        assert!(resolve_label(&json!({"metadata": {}}), floor()).is_none());
    }

    #[tokio::test]
    async fn walk_unknown_object_not_found() {
        let mount = mount_with(&[("", "pods", labeled_pod("default", "web", "Running"))]);
        // `Fid` is not `Debug`, so map the Ok side away before asserting.
        let result = mount
            .walk(&["pods", "default", "ghost", "status"], &subject())
            .await
            .map(|_| ());
        assert!(
            matches!(result, Err(MountError::NotFound(_))),
            "walk into an unknown object must 404, got: {result:?}"
        );
    }

    #[tokio::test]
    async fn ctl_without_actuator_reports_deferred() {
        let mount = mount_with(&[("", "pods", labeled_pod("default", "web", "Running"))]);
        let mut fid = mount
            .walk(&["pods", "default", "web", "ctl"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        mount.write(&fid, 0, b"delete", &subject()).await.unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.contains("deferred"), "got: {text}");
    }

    #[tokio::test]
    async fn ctl_unknown_verb_rejected() {
        let mount = mount_with(&[("", "pods", labeled_pod("default", "web", "Running"))]);
        let mut fid = mount
            .walk(&["pods", "default", "web", "ctl"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        mount
            .write(&fid, 0, b"frobnicate", &subject())
            .await
            .unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.contains("unknown ctl verb"), "got: {text}");
    }

    /// ctl routes a valid verb to a wired actuator, but only after the label
    /// floor passes.
    #[tokio::test]
    async fn ctl_with_actuator_invokes_delete() {
        use std::sync::atomic::AtomicBool;

        struct FakeActuator {
            deleted: AtomicBool,
        }
        #[async_trait]
        impl CtlActuator for FakeActuator {
            async fn apply(&self, _key: &ObjectKey, verb: CtlVerb) -> Result<String, MountError> {
                assert_eq!(verb, CtlVerb::Delete);
                self.deleted.store(true, Ordering::SeqCst);
                Ok("deleted".to_owned())
            }
        }

        let actuator = Arc::new(FakeActuator {
            deleted: AtomicBool::new(false),
        });
        let cache = Arc::new(ProjectionCache::new());
        cache.upsert("", "pods", labeled_pod("default", "web", "Running"));
        let mount = K8sMount::new(cache, floor())
            .with_actuator(Arc::clone(&actuator) as Arc<dyn CtlActuator>);

        let mut fid = mount
            .walk(&["pods", "default", "web", "ctl"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        mount.write(&fid, 0, b"delete", &subject()).await.unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.starts_with("ok:"), "got: {text}");
        assert!(actuator.deleted.load(Ordering::SeqCst));
    }

    /// ctl against an unlabeled object is denied before the actuator runs.
    #[tokio::test]
    async fn ctl_on_unlabeled_object_denied_before_actuation() {
        use std::sync::atomic::AtomicBool;

        struct PanicActuator {
            called: AtomicBool,
        }
        #[async_trait]
        impl CtlActuator for PanicActuator {
            async fn apply(&self, _key: &ObjectKey, _verb: CtlVerb) -> Result<String, MountError> {
                self.called.store(true, Ordering::SeqCst);
                Ok("should not happen".to_owned())
            }
        }

        let actuator = Arc::new(PanicActuator {
            called: AtomicBool::new(false),
        });
        let cache = Arc::new(ProjectionCache::new());
        cache.upsert("", "pods", unlabeled_pod("default", "web"));
        let mount = K8sMount::new(cache, floor())
            .with_actuator(Arc::clone(&actuator) as Arc<dyn CtlActuator>);

        let mut fid = mount
            .walk(&["pods", "default", "web", "ctl"], &subject())
            .await
            .unwrap();
        mount.open(&mut fid, 2, &subject()).await.unwrap();
        mount.write(&fid, 0, b"delete", &subject()).await.unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.contains("denied"), "got: {text}");
        assert!(
            !actuator.called.load(Ordering::SeqCst),
            "actuator must not run on unlabeled object"
        );
    }

    #[tokio::test]
    async fn generic_grouped_resource_projects() {
        let mount = mount_with(&[(
            "models.hyprstream.io",
            "models",
            json!({
                "metadata": { "name": "qwen", "namespace": "ml", "resourceVersion": "9",
                    "annotations": { LEVEL_ANNOTATION: "internal" } },
                "status": { "stage": "promoted" }
            }),
        )]);
        let mut fid = mount
            .walk(
                &["models.hyprstream.io", "models", "ml", "qwen", "status"],
                &subject(),
            )
            .await
            .unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let text = String::from_utf8(mount.read(&fid, 0, 4096, &subject()).await.unwrap()).unwrap();
        assert!(text.contains("promoted"), "got: {text}");
    }

    #[test]
    fn pod_api_resource_is_core_group() {
        let ar = pod_api_resource();
        assert_eq!(ar.group, "");
        assert_eq!(ar.plural, "pods");
        assert_eq!(ar.kind, "Pod");
    }
}
