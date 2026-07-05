# Image Substrate

Status: first serving slice for #789.

`create_image_substrate_router` exposes a read-only HTTP data plane over
`RafsStore` so nydus/containerd nodes can fetch RAFS bootstraps and chunk blobs
from hyprstream-managed CAS storage without using the host-local filesystem
layout directly.

## Routes

All routes are read-only and support `GET`, `HEAD`, and single HTTP byte ranges.

| Route | Body |
| --- | --- |
| `/nydus/v1/bootstraps/{sha256:<hex>}` | RAFS bootstrap bytes from `RafsStore::bootstrap_path` |
| `/nydus/v1/blobs/{algorithm}/{hex}` | Blob bytes from `RafsStore::blob_path` |
| `/v2/{repo...}/blobs/{sha256:<hex>}` | Registry-compatible blob fetch alias |

Anonymous reads are disabled by default. Set
`ImageSubstratePolicy::allow_anonymous_read` only for deployments where
nydus-snapshotter/containerd cannot forward WIT or ticket credentials yet.

## Scope

This is not a custom snapshotter. It serves the existing RAFS artifacts created
by `RafsStore` and leaves ingest, deduplication, garbage collection, and the
RAFS layout decision with the image store. The router is intentionally small so
the eventual operator packaging can mount it behind the service endpoint shape
chosen for nydus-snapshotter.
