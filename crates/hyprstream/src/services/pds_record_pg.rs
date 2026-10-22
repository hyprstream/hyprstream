//! Postgres-backed PDS KV shell (#1257).
//!
//! The implementation moved to `hyprstream_pds::pgsql_kv` so the discovery
//! crate's checkpointed accepted-state authority shares the same audited
//! connection/TLS/pool code. This module re-exports it at the historical
//! path; no divergent copy may exist here.

pub(crate) use hyprstream_pds::pgsql_kv::{prefix_upper_bound, PgKv};
