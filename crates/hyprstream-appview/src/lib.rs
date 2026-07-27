//! Derived AppView inventory for local and federated ATProto identities.
//!
//! This crate is an index, never identity authority. The default build uses the
//! in-memory read model supplied by `hyprstream-pds-service`, avoiding native
//! database dependencies in the demo/CI path. The optional `pglite` feature
//! supplies a PostgreSQL-compatible PGlite adapter for metal deployments.
//! Inventory queries apply the MAC dominance predicate before rows reach the
//! HTTP response.

mod http;
mod inventory;

pub use http::{inventory_router, InventoryViewer};
#[cfg(feature = "pglite")]
pub use inventory::PGliteIdentityInventory;
pub use inventory::{
    DirectoryInventorySource, HostedAccountInventorySource, InventoryIngestor,
    LabeledInventoryEntry, StubDirectoryInventorySource, StubHostedAccountInventorySource,
};
