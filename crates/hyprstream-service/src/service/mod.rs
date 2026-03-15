//! Service orchestration: spawner, factory, manager, metadata.
//!
//! Moved from `hyprstream-rpc::service` to separate orchestration
//! from transport concerns.

pub mod spawner;
pub mod manager;
pub mod factory;
pub mod metadata;

pub use spawner::{InprocManager, Spawnable, SpawnedService};
pub use factory::{get_factory, list_factories, QuicSharedConfig, ServiceClient, ServiceContext, ServiceFactory};
pub use manager::{detect as detect_service_manager, ServiceManager, StandaloneManager};
#[cfg(feature = "systemd")]
pub use manager::SystemdManager;
