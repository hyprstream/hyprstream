//! Service orchestration: spawner, factory, manager, metadata.
//!
//! Moved from `hyprstream-rpc::service` to separate orchestration
//! from transport concerns.

pub mod spawner;
pub mod manager;
pub mod factory;
pub mod metadata;
pub mod ordering;
pub mod trust_store;

pub use spawner::{InprocManager, Spawnable, SpawnedService};
pub use factory::{
    get_factory, list_factories, DiscoveryBootstrapAuthority, NativeAnnouncementPublisher,
    NativeAnnouncementRequest, NativeServiceAnnouncement, QuicSharedConfig, ServiceContext,
    ServiceFactory,
};
pub use trust_store::{TrustStore, Attestation, global_trust_store};
pub use manager::{detect as detect_service_manager, ServiceManager, StandaloneManager};
#[cfg(feature = "systemd")]
pub use manager::SystemdManager;
