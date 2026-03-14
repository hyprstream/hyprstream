//! RPC schema introspection metadata types.
//!
//! Re-exported from `hyprstream-rpc::service::metadata` — the canonical location.
//! This module exists for backward compatibility with code that imports from
//! `hyprstream_service::*`.

pub use hyprstream_rpc::service::metadata::{
    MethodMeta, ParamMeta, SchemaMetadataFn, ScopedSchemaMetadataFn, ScopedClientTreeNode,
};
