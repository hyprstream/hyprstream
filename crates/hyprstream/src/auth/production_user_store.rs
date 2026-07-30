//! Single production construction boundary for credential/account storage.
//!
//! `ProductionUserStore` is the only account-store handle accepted by OAuth,
//! CLI, and bootstrap wiring. Its public constructor resolves deployment
//! configuration and rejects plaintext-capable backends in a `credential-pds`
//! build before opening any account database.

use super::UserStore;
use crate::config::{CredentialsBackend, CredentialsConfig, HyprConfig};
use anyhow::{Context, Result};
use std::{ops::Deref, path::Path, sync::Arc};

/// Opaque handle proving that an account store passed production admission.
///
/// The inner trait object is intentionally private. Callers cannot manufacture
/// this proof from an arbitrary `UserStore`; they must use [`Self::open`].
///
/// Raw plaintext backends are also not constructible out of crate:
///
/// ```compile_fail
/// use hyprstream_core::auth::RocksDbUserStore;
///
/// let _ = RocksDbUserStore::open(std::path::Path::new("/tmp/plaintext"));
/// ```
///
/// Nor can a downstream crate implement its own plaintext `UserStore`, because
/// the trait's required sealing supertrait is crate-private:
///
/// ```compile_fail
/// struct PlaintextStore;
/// impl hyprstream_core::auth::user_store::private::Sealed for PlaintextStore {}
/// ```
///
/// The admission constructor itself is not an injection point either:
///
/// ```compile_fail
/// # struct PlaintextStore;
/// let _ = hyprstream_core::auth::ProductionUserStore::from_encrypted_backend(
///     PlaintextStore,
/// );
/// ```
///
/// Public OAuth wiring also requires the admitted handle, not an arbitrary
/// trait object:
///
/// ```compile_fail
/// use hyprstream_core::{
///     auth::UserStore,
///     services::oauth::state::OAuthState,
/// };
/// use std::sync::Arc;
///
/// fn inject_plaintext(state: OAuthState, store: Arc<dyn UserStore>) {
///     let _ = state.with_user_store(store);
/// }
/// ```
#[derive(Clone)]
pub struct ProductionUserStore {
    inner: Arc<dyn UserStore>,
}

/// Capability required by every raw backend constructor.
///
/// The type is crate-visible so backend modules can accept it, but its field is
/// private to this module. Production callers therefore cannot open a backend
/// without first passing this module's admission checks.
pub(crate) struct ProductionStorePermit(());

impl ProductionUserStore {
    fn permit() -> ProductionStorePermit {
        ProductionStorePermit(())
    }

    /// Open the deployment-selected account store through the sole production
    /// admission boundary.
    pub async fn open(credentials_dir: &Path) -> Result<Self> {
        let config = HyprConfig::load()
            .map(|config| config.credentials)
            .unwrap_or_default();
        Self::open_with_config(credentials_dir, &config).await
    }

    /// Open from already-resolved configuration. Kept crate-private so public
    /// callers cannot substitute a configuration after deployment admission.
    pub(crate) async fn open_with_config(
        credentials_dir: &Path,
        config: &CredentialsConfig,
    ) -> Result<Self> {
        config.backend.ensure_allowed_for_build()?;

        match config.backend {
            CredentialsBackend::Pglite => {
                #[cfg(feature = "pglite")]
                {
                    let store = super::PgliteUserStore::open_admitted(
                        credentials_dir.join("pglite"),
                        &Self::permit(),
                    )
                    .await
                    .context("open encrypted PGlite credential store")?;
                    Ok(Self::from_encrypted_backend(store))
                }
                #[cfg(not(feature = "pglite"))]
                anyhow::bail!(
                    "credentials.backend = \"pglite\" but this binary lacks the pglite feature"
                )
            }
            CredentialsBackend::Rocksdb => {
                #[cfg(not(feature = "credential-pds"))]
                {
                    let store =
                        super::RocksDbUserStore::open_admitted(credentials_dir, &Self::permit())
                            .context("open legacy RocksDB credential store")?;
                    Ok(Self {
                        inner: Arc::new(store),
                    })
                }
                #[cfg(feature = "credential-pds")]
                unreachable!("credential-pds backend guard admitted RocksDB")
            }
            CredentialsBackend::Valkey => {
                #[cfg(all(not(feature = "credential-pds"), feature = "valkey"))]
                {
                    let store = super::ValkeyUserStore::connect_admitted(
                        &config.valkey.url,
                        &Self::permit(),
                    )
                    .await
                    .context("open legacy Valkey credential store")?;
                    Ok(Self {
                        inner: Arc::new(store),
                    })
                }
                #[cfg(all(not(feature = "credential-pds"), not(feature = "valkey")))]
                {
                    anyhow::bail!(
                        "credentials.backend = \"valkey\" but this binary lacks the valkey feature"
                    )
                }
                #[cfg(feature = "credential-pds")]
                {
                    unreachable!("credential-pds backend guard admitted Valkey")
                }
            }
        }
    }

    /// Open the admitted account store and the independent anonymous-device
    /// store used by OAuth. Legacy non-credential builds that select RocksDB
    /// share one handle so RocksDB's exclusive writer lock is not acquired
    /// twice.
    pub(crate) async fn open_with_device_store(
        credentials_dir: &Path,
        config: &CredentialsConfig,
    ) -> Result<(Self, Option<Arc<dyn super::DeviceStore>>)> {
        config.backend.ensure_allowed_for_build()?;

        #[cfg(not(feature = "credential-pds"))]
        if config.backend == CredentialsBackend::Rocksdb {
            let store = Arc::new(
                super::RocksDbUserStore::open_admitted(credentials_dir, &Self::permit())
                    .context("open legacy RocksDB credential/device store")?,
            );
            return Ok((
                Self {
                    inner: Arc::clone(&store) as Arc<dyn UserStore>,
                },
                Some(store as Arc<dyn super::DeviceStore>),
            ));
        }

        let account_store = Self::open_with_config(credentials_dir, config).await?;
        let device_store = super::RocksDbUserStore::open_admitted(credentials_dir, &Self::permit())
            .map(|store| Arc::new(store) as Arc<dyn super::DeviceStore>)
            .map_err(|error| {
                tracing::warn!("could not open RocksDB device store: {error:#}");
                error
            })
            .ok();
        Ok((account_store, device_store))
    }

    /// Admit a backend whose implementation is structurally marked as
    /// encrypted-at-rest. PGlite uses this today; #1401's Postgres backend must
    /// implement the same crate-private marker when restacked.
    #[allow(dead_code)] // #1401 consumes this marker when Postgres is restacked.
    pub(crate) fn from_encrypted_backend<T>(store: T) -> Self
    where
        T: EncryptedUserStoreBackend + 'static,
    {
        Self {
            inner: Arc::new(store),
        }
    }

    pub(crate) fn clone_inner(&self) -> Arc<dyn UserStore> {
        Arc::clone(&self.inner)
    }

    #[cfg(test)]
    pub(crate) fn for_test(store: Arc<dyn UserStore>) -> Self {
        Self { inner: store }
    }
}

impl Deref for ProductionUserStore {
    type Target = dyn UserStore;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref()
    }
}

/// Crate-private admission marker for encrypted-at-rest account backends.
///
/// Keeping the implementation list here makes adding Postgres an explicit
/// security decision rather than something any `UserStore` implementation
/// inherits automatically.
#[allow(dead_code)] // #1401 implements this when Postgres is restacked.
pub(crate) trait EncryptedUserStoreBackend: UserStore {}

#[cfg(feature = "pglite")]
impl EncryptedUserStoreBackend for super::PgliteUserStore {}

#[cfg(all(test, feature = "credential-pds"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn production_boundary_rejects_plaintext_backends_before_opening_storage() -> Result<()> {
        let credentials_dir = tempfile::tempdir()?;

        for backend in [CredentialsBackend::Rocksdb, CredentialsBackend::Valkey] {
            let config = CredentialsConfig {
                backend,
                ..CredentialsConfig::default()
            };
            let error = match ProductionUserStore::open_with_config(credentials_dir.path(), &config)
                .await
            {
                Ok(_) => panic!("credential-pds admitted a plaintext-capable account backend"),
                Err(error) => error,
            };
            assert!(
                error
                    .to_string()
                    .contains("credential-pds requires credentials.backend = \"pglite\""),
                "unexpected admission error: {error:#}"
            );
        }

        assert!(
            !credentials_dir.path().join("users.db").exists(),
            "rejected backend must not open or create plaintext storage"
        );
        Ok(())
    }
}
