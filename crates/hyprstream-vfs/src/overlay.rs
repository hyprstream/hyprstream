//! Rootfs overlay — the **v1** overlay backend (native only).
//!
//! This is the v1 engine of the hyprstream overlay interface: it builds an
//! [`FsMount`](crate::FsMount) backed by `fuse_backend_rs::OverlayFs(lower,
//! upper)`, exposed through the [`FuseFileSystemMount`](crate::FuseFileSystemMount)
//! up-adapter. Copy-up and whiteouts are performed **by `OverlayFs`** — Kata's
//! production overlay — not by us. We deliberately do **not** hand-roll
//! copy-on-write here; a native hyprstream CoW engine (per-tenant writable
//! layers over shared RO model weights, delta/sandbox snapshots) is the backlog
//! ticket #370 and will implement the same `FsMount` trait with zero consumer
//! churn.
//!
//! Layers are parameterised as `BoxedLayer`s so FS-B can later pass a RAFS lower
//! (via `nydus-rafs`) without touching this code. For unit testing — and as the
//! shape FS-B0/FS-B will reuse — [`passthrough_layer`] wraps a host directory
//! (`PassthroughFs`) as a layer.

use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use fuse_backend_rs::api::filesystem::Layer;
use fuse_backend_rs::overlayfs::{config::Config as OverlayConfig, BoxedLayer, OverlayFs};
use fuse_backend_rs::passthrough::{Config as PassthroughConfig, PassthroughFs};

use crate::fuse_adapter::FuseFileSystemMount;
use crate::mount::MountError;

fn map_io(err: io::Error, ctx: &str) -> MountError {
    MountError::Io(format!("{ctx}: {err}"))
}

/// Wrap any `fuse_backend_rs` [`Layer`] (a `FileSystem` that also implements the
/// overlay `Layer` trait) as a [`BoxedLayer`] usable by [`rootfs_overlay`].
///
/// This is the generic seam FS-B uses to plug an **in-process RAFS** lower
/// (`nydus_rafs::fs::Rafs`, which implements `Layer`) into the overlay without
/// this crate depending on `nydus-rafs`: the caller owns the RAFS type, we only
/// require it be a `Layer<Inode = u64, Handle = u64>`. The layer must already be
/// ready to serve (RAFS: `import`ed; PassthroughFs: `import`ed).
pub fn layer_from_fs<L>(fs: L) -> Arc<BoxedLayer>
where
    L: Layer<Inode = u64, Handle = u64> + Send + Sync + 'static,
{
    let layer: Box<dyn Layer<Inode = u64, Handle = u64> + Send + Sync> = Box::new(fs);
    Arc::new(layer)
}

/// Build a `BoxedLayer` from a host directory using `PassthroughFs`.
///
/// `read_only` controls only the cache policy hint; OverlayFs enforces the
/// RO-lower / RW-upper distinction structurally by which slot a layer occupies.
/// This is the test/bootstrap layer constructor; FS-B swaps the lower for RAFS.
pub fn passthrough_layer(root: &Path) -> Result<Arc<BoxedLayer>, MountError> {
    let cfg = PassthroughConfig {
        root_dir: root.to_string_lossy().into_owned(),
        // `do_import = false`: the layer is driven through OverlayFs, which
        // performs the namespace validation itself.
        do_import: false,
        writeback: false,
        xattr: true,
        ..Default::default()
    };
    let fs = PassthroughFs::<()>::new(cfg).map_err(|e| map_io(e, "PassthroughFs::new"))?;
    fs.import().map_err(|e| map_io(e, "PassthroughFs::import"))?;
    Ok(layer_from_fs(fs))
}

/// Build the v1 rootfs overlay as an [`FsMount`].
///
/// `upper` is the single writable layer (copy-ups and whiteouts land here);
/// `lowers` are the read-only layers, highest priority first. The returned
/// [`FuseFileSystemMount`] presents the merged view with full overlay semantics
/// (copy-on-write, whiteout-on-delete, opaque dirs) implemented by `OverlayFs`.
///
/// `work` is the overlay work directory (scratch space for atomic operations).
pub fn rootfs_overlay(
    upper: Arc<BoxedLayer>,
    lowers: Vec<Arc<BoxedLayer>>,
    work: &Path,
) -> Result<FuseFileSystemMount<OverlayFs>, MountError> {
    let cfg = OverlayConfig {
        work: work.to_string_lossy().into_owned(),
        mountpoint: "/".to_owned(),
        // `do_import = true`: OverlayFs builds its inode tree from the layers up
        // front (matches Kata's usage); required before serving.
        do_import: true,
        writeback: true,
        attr_timeout: Duration::from_secs(1),
        entry_timeout: Duration::from_secs(1),
        ..Default::default()
    };

    let overlay =
        OverlayFs::new(Some(upper), lowers, cfg).map_err(|e| map_io(e, "OverlayFs::new"))?;
    overlay
        .import()
        .map_err(|e| map_io(e, "OverlayFs::import"))?;

    // The up-adapter runs `init` for option negotiation; `import` is already done.
    FuseFileSystemMount::new(overlay)
}

/// Convenience: build a passthrough-backed rootfs overlay from host directories.
///
/// Wraps `lower` (RO) and `upper` (RW) host dirs as `PassthroughFs` layers and
/// composes them via [`rootfs_overlay`]. This is the unit-test / bootstrap
/// shape; production rootfs (FS-B) substitutes a RAFS lower.
pub fn passthrough_rootfs_overlay(
    lower: &Path,
    upper: &Path,
    work: &Path,
) -> Result<FuseFileSystemMount<OverlayFs>, MountError> {
    let upper_layer = passthrough_layer(upper)?;
    let lower_layer = passthrough_layer(lower)?;
    rootfs_overlay(upper_layer, vec![lower_layer], work)
}
