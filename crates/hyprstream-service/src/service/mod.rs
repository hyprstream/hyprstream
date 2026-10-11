//! Service orchestration: spawner, factory, manager, metadata.
//!
//! Moved from `hyprstream-rpc::service` to separate orchestration
//! from transport concerns.

pub mod factory;
pub mod manager;
pub mod metadata;
pub mod ordering;
pub mod spawner;
pub mod trust_store;

pub use factory::{
    deployment_data_dir, get_factory, list_factories, NativeAnnouncementPublisher,
    NativeAnnouncementReach, NativeAnnouncementRequest, NativeServiceAnnouncement,
    QuicSharedConfig, ServiceContext, ServiceFactory,
};
#[cfg(feature = "systemd")]
pub use manager::SystemdManager;
pub use manager::{detect as detect_service_manager, ServiceManager, StandaloneManager};
pub use spawner::{InprocManager, Spawnable, SpawnedService};
pub use trust_store::{global_trust_store, Attestation, TrustStore};
