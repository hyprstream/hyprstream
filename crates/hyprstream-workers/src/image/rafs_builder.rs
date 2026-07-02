//! In-process RAFS bootstrap builder (FS-B0).
//!
//! Converts the OCI layer tarballs pulled into the CAS (`blobs/`) into a single
//! RAFS bootstrap (`.meta`) that can be consumed by an in-process RAFS
//! `FileSystem` (FS-B, #363) or by `nydusd`.
//!
//! # Why in-process
//!
//! The `nydus-builder` crate (the same code that backs the `nydus-image create`
//! binary) is reachable from the dependency tree, so the whole conversion runs
//! in-process in pure Rust — no external `nydus-image` binary, no shell-out, no
//! silent fallback. If the conversion fails the caller hard-errors.
//!
//! # Approach: layered build
//!
//! OCI layers are ordered lower → upper in the manifest. We build them in that
//! order with [`TarballBuilder`] using [`ConversionType::TargzToRafs`], chaining
//! each layer's build to the previous layer's bootstrap via
//! [`BootstrapManager`]'s parent path. This is the same overlay-aware path that
//! `nydus-image create` uses for multi-layer images:
//!
//! - OCI whiteouts (`.wh.*`) from upper layers correctly delete lower entries.
//! - The builder's `layered_chunk_dict` deduplicates chunks across layers, so a
//!   chunk that already exists in a lower layer is not written again.
//!
//! Data blobs are written content-addressed into `blobs_dir` (named by their
//! RAFS blob hash). The final layer's bootstrap is written to `bootstrap_path`
//! (a `SingleFile`), so `RafsStore::bootstrap_path()` resolves to a real RAFS
//! bootstrap.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use nydus_api::ConfigV2;
use nydus_builder::attributes::Attributes;
use nydus_builder::{
    ArtifactStorage, BlobManager, BootstrapManager, BuildContext, Builder, ConversionType, Features,
    Prefetch, TarballBuilder, WhiteoutSpec,
};
use nydus_rafs::metadata::RafsVersion;
use nydus_utils::{compress, digest};

use crate::error::{Result, WorkerError};

/// RAFS metadata format version emitted by the builder.
///
/// V6 is the EROFS-compatible format used by modern nydusd / Kata virtiofs and
/// is what FS-B / the in-kernel EROFS path expects.
const RAFS_FS_VERSION: RafsVersion = RafsVersion::V6;

/// Build a RAFS bootstrap from an ordered set of pulled OCI layer tarballs.
///
/// * `layer_blob_paths` — absolute paths to the layer blobs already present in
///   the CAS, ordered **lower → upper** (i.e. manifest layer order). The blobs
///   may be gzip-compressed (`*.tar.gz`, the OCI default) or plain tar; the
///   builder auto-detects.
/// * `blobs_dir` — directory the generated RAFS data blobs are written into,
///   content-addressed by their RAFS blob hash.
/// * `bootstrap_path` — destination for the final RAFS bootstrap (`.meta`).
///
/// Returns the path to the produced bootstrap (== `bootstrap_path`). Hard-errors
/// if no layers are supplied or any layer fails to convert; never leaves a
/// partial/missing bootstrap silently.
pub fn build_rafs_bootstrap(
    layer_blob_paths: &[PathBuf],
    blobs_dir: &Path,
    bootstrap_path: &Path,
) -> Result<PathBuf> {
    if layer_blob_paths.is_empty() {
        return Err(WorkerError::RafsError(
            "cannot build RAFS bootstrap: image has no layers".to_owned(),
        ));
    }

    // Shared RAFS configuration. Layered builds require these to be identical
    // across layers (see `RafsSuperConfig::check_compatibility`).
    let config = Arc::new(ConfigV2::default());

    // Each intermediate (and the final) bootstrap is written here. The layered
    // build reads the previous layer's bootstrap as its parent, so we keep a
    // single rolling bootstrap file and chain to it. The final write lands at
    // `bootstrap_path`.
    let mut parent_bootstrap: Option<PathBuf> = None;

    // Intermediate bootstraps for non-final layers live next to the final
    // bootstrap so the parent reader can find them; the final layer overwrites
    // `bootstrap_path` itself.
    let bootstrap_dir = bootstrap_path.parent().ok_or_else(|| {
        WorkerError::RafsError(format!(
            "bootstrap path {} has no parent directory",
            bootstrap_path.display()
        ))
    })?;

    let layer_count = layer_blob_paths.len();
    for (idx, layer_path) in layer_blob_paths.iter().enumerate() {
        let is_final = idx + 1 == layer_count;

        // Final layer writes directly to the requested bootstrap path; earlier
        // layers write to a deterministic intermediate path so the next layer
        // can read it as its parent.
        let out_bootstrap = if is_final {
            bootstrap_path.to_path_buf()
        } else {
            bootstrap_dir.join(format!(
                ".rafs-build-{}-layer{}.meta",
                bootstrap_basename(bootstrap_path),
                idx
            ))
        };

        build_one_layer(
            layer_path,
            blobs_dir,
            &out_bootstrap,
            parent_bootstrap.as_deref(),
            idx,
            config.clone(),
        )?;

        // Clean up the previous intermediate bootstrap (it has been folded into
        // the current one) before advancing.
        if let Some(prev) = parent_bootstrap.take() {
            if prev != out_bootstrap {
                let _ = std::fs::remove_file(&prev);
            }
        }
        parent_bootstrap = Some(out_bootstrap);
    }

    if !bootstrap_path.exists() {
        return Err(WorkerError::RafsError(format!(
            "RAFS build completed but bootstrap {} is missing",
            bootstrap_path.display()
        )));
    }

    Ok(bootstrap_path.to_path_buf())
}

/// Build a single layer into `out_bootstrap`, optionally overlaid on `parent`.
fn build_one_layer(
    layer_path: &Path,
    blobs_dir: &Path,
    out_bootstrap: &Path,
    parent: Option<&Path>,
    layer_idx: usize,
    config: Arc<ConfigV2>,
) -> Result<()> {
    // Data blobs are written into the shared CAS directory, named by their RAFS
    // blob hash (the empty suffix keeps the bare hash as the filename).
    let blob_storage = ArtifactStorage::FileDir((blobs_dir.to_path_buf(), String::new()));
    // The bootstrap is a single explicit file so the caller controls its path.
    let bootstrap_storage = ArtifactStorage::SingleFile(out_bootstrap.to_path_buf());

    let mut ctx = BuildContext::new(
        String::new(), // blob_id: derive from blob hash
        // RAFS v6 (EROFS) requires 4K-aligned chunks; `nydus-image` forces this
        // on for any v6 build that is not TarToTarfs.
        true, // aligned_chunk
        0,    // blob_offset
        compress::Algorithm::Zstd,   // compressor
        digest::Algorithm::Sha256,   // digester
        false,                       // explicit_uidgid (OCI: normalize to 0)
        WhiteoutSpec::Oci,           // OCI .wh.* whiteout semantics
        ConversionType::TargzToRafs, // gzip/tar -> RAFS (+ data blob)
        layer_path.to_path_buf(),    // source_path
        Prefetch::default(),
        Some(blob_storage),
        None,  // external_blob_storage
        false, // blob_inline_meta: keep bootstrap as a standalone .meta file
        Features::new(),
        false, // encrypt
        Attributes::default(),
    );
    ctx.set_fs_version(RAFS_FS_VERSION);
    ctx.set_configuration(config);

    let mut bootstrap_mgr = BootstrapManager::new(
        Some(bootstrap_storage),
        parent.map(|p| p.to_string_lossy().into_owned()),
    );
    let mut blob_mgr = BlobManager::new(digest::Algorithm::Sha256, false);

    let mut builder = TarballBuilder::new(ConversionType::TargzToRafs);
    builder
        .build(&mut ctx, &mut bootstrap_mgr, &mut blob_mgr)
        .map_err(|e| {
            WorkerError::RafsError(format!(
                "failed to build RAFS layer {} from {}: {e:#}",
                layer_idx,
                layer_path.display()
            ))
        })?;

    Ok(())
}

/// List the RAFS data-blob IDs referenced by a bootstrap.
///
/// The IDs are the bare blob-hash filenames the data blobs are stored under in
/// `blobs_dir`. Used by garbage collection so it does not delete the data blobs
/// a live bootstrap depends on (those blobs are not OCI layer digests and so
/// are invisible to the OCI-layer-based GC pass).
pub fn bootstrap_blob_ids(bootstrap_path: &Path) -> Result<Vec<String>> {
    let config = Arc::new(ConfigV2::default());
    let (rs, _reader) = nydus_rafs::metadata::RafsSuper::load_from_file(bootstrap_path, config, false)
        .map_err(|e| {
            WorkerError::RafsError(format!(
                "failed to load RAFS bootstrap {}: {e:#}",
                bootstrap_path.display()
            ))
        })?;
    Ok(rs
        .superblock
        .get_blob_infos()
        .iter()
        .map(|b| b.blob_id())
        .collect())
}

/// Filesystem-safe base name (no extension) used to disambiguate intermediate
/// bootstraps for concurrent pulls of different images.
fn bootstrap_basename(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_owned())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use nydus_rafs::metadata::RafsSuper;

    /// Build a small gzip-compressed tar layer (the OCI default media type)
    /// with the given (path, contents) regular-file entries.
    fn make_gzip_layer(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).unwrap();
        let gz = GzEncoder::new(file, Compression::default());
        let mut tar = tar::Builder::new(gz);
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_entry_type(tar::EntryType::Regular);
            tar.append_data(&mut header, name, *data).unwrap();
        }
        let gz = tar.into_inner().unwrap();
        gz.finish().unwrap();
    }

    #[test]
    fn builds_bootstrap_from_single_layer() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs_dir = tmp.path().join("blobs");
        let bootstrap_dir = tmp.path().join("bootstrap");
        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let layer = tmp.path().join("layer0.tar.gz");
        make_gzip_layer(
            &layer,
            &[
                ("etc/hostname", b"worker\n"),
                ("usr/bin/app", b"#!/bin/sh\necho hi\n"),
            ],
        );

        let bootstrap = bootstrap_dir.join("img.meta");
        let out = build_rafs_bootstrap(&[layer], &blobs_dir, &bootstrap).unwrap();
        assert_eq!(out, bootstrap);

        let meta = std::fs::metadata(&bootstrap).unwrap();
        assert!(meta.len() > 0, "bootstrap must be non-empty");

        // It must parse as a real RAFS bootstrap.
        let config = Arc::new(ConfigV2::default());
        let (rs, _reader) =
            RafsSuper::load_from_file(&bootstrap, config, false).expect("bootstrap must load");
        assert!(rs.meta.inodes_count > 0, "RAFS must have inodes");
    }

    #[test]
    fn builds_bootstrap_from_layered_image() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs_dir = tmp.path().join("blobs");
        let bootstrap_dir = tmp.path().join("bootstrap");
        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let lower = tmp.path().join("layer0.tar.gz");
        make_gzip_layer(
            &lower,
            &[("etc/hostname", b"base\n"), ("etc/config", b"v1\n")],
        );
        let upper = tmp.path().join("layer1.tar.gz");
        // Upper overwrites etc/config and adds a new file.
        make_gzip_layer(
            &upper,
            &[("etc/config", b"v2\n"), ("etc/extra", b"added\n")],
        );

        let bootstrap = bootstrap_dir.join("img.meta");
        let out =
            build_rafs_bootstrap(&[lower, upper], &blobs_dir, &bootstrap).unwrap();
        assert_eq!(out, bootstrap);
        assert!(std::fs::metadata(&bootstrap).unwrap().len() > 0);

        let config = Arc::new(ConfigV2::default());
        let (rs, _reader) =
            RafsSuper::load_from_file(&bootstrap, config, false).expect("bootstrap must load");
        assert!(rs.meta.inodes_count > 0);

        // No intermediate bootstraps should be left behind.
        for entry in std::fs::read_dir(&bootstrap_dir).unwrap() {
            let name = entry.unwrap().file_name();
            let name = name.to_string_lossy();
            assert!(
                !name.starts_with(".rafs-build-"),
                "intermediate bootstrap leaked: {name}"
            );
        }
    }
}
