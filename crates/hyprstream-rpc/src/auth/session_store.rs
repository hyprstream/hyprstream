//! Durable (file-backed) session registry — the policy process's canonical
//! session store, mirroring the credential-revocation authority's discipline.
//!
//! A JSONL append log holds one record per registration and one per
//! revocation. Registration validates, appends, and fsyncs BEFORE the session
//! becomes visible; revocation appends its marker BEFORE the in-memory status
//! flips and the verified-subject cache generation is flushed
//! (publication-before-eviction, v16 §3.3). Session identifiers are never
//! reassigned: records are never dropped on load (an expired-but-present
//! record still blocks re-registration), unlike the credential store whose
//! entries expire with their tokens.

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write as _;

#[cfg(not(target_arch = "wasm32"))]
use super::credential::{
    ActiveOrRevoked, InMemorySessionRegistry, SessionKey, SessionRegisterError, SessionRegistry,
    SessionRevokeError, SessionState,
};
#[cfg(not(target_arch = "wasm32"))]
use super::credential::{SessionIdentifier, SessionKind};

/// One JSONL session record. Registration records carry the full state;
/// revocation markers carry only the key and `status: "revoked"`.
#[cfg(not(target_arch = "wasm32"))]
#[derive(serde::Serialize, serde::Deserialize)]
struct SessionRecord {
    iss: String,
    /// "oidc" or "workload" — the disjoint identifier namespaces.
    kind: String,
    id: String,
    status: String, // "active" | "revoked"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tenant: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    created_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    clearance_epoch: Option<u64>,
}

#[cfg(not(target_arch = "wasm32"))]
impl SessionRecord {
    fn from_registration(key: &SessionKey, state: &SessionState) -> Self {
        let (kind, id) = key_parts(key);
        Self {
            iss: key.issuer.clone(),
            kind: kind.to_owned(),
            id: id.to_owned(),
            status: "active".to_owned(),
            subject: Some(state.subject.clone()),
            tenant: Some(state.tenant.clone()),
            created_at: Some(state.created_at),
            expires_at: Some(state.expires_at),
            clearance_epoch: Some(state.clearance_epoch),
        }
    }

    fn revocation_marker(key: &SessionKey) -> Self {
        let (kind, id) = key_parts(key);
        Self {
            iss: key.issuer.clone(),
            kind: kind.to_owned(),
            id: id.to_owned(),
            status: "revoked".to_owned(),
            subject: None,
            tenant: None,
            created_at: None,
            expires_at: None,
            clearance_epoch: None,
        }
    }

    fn key(&self) -> Option<SessionKey> {
        let key = match self.kind.as_str() {
            "oidc" => SessionKey::oidc(self.iss.clone(), self.id.clone()),
            "workload" => SessionKey::workload(self.iss.clone(), self.id.clone()),
            _ => return None,
        };
        key.is_valid().then_some(key)
    }

    /// Decode a registration record into its state. `None` = corrupt (a
    /// registration missing any state field).
    fn registration_state(&self) -> Option<SessionState> {
        let kind = match self.kind.as_str() {
            "oidc" => SessionKind::Interactive,
            "workload" => SessionKind::Workload,
            _ => return None,
        };
        Some(SessionState {
            subject: self.subject.clone()?,
            tenant: self.tenant.clone()?,
            kind,
            created_at: self.created_at?,
            expires_at: self.expires_at?,
            status: ActiveOrRevoked::Active,
            clearance_epoch: self.clearance_epoch?,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn key_parts(key: &SessionKey) -> (&'static str, &str) {
    match &key.id {
        SessionIdentifier::OidcSid(sid) => ("oidc", sid),
        SessionIdentifier::WorkloadSessionId(id) => ("workload", id),
    }
}

/// Durable session registry: an in-memory index in front of a JSONL append
/// file. This is the canonical registry form owned by the policy process.
#[cfg(not(target_arch = "wasm32"))]
pub struct FileBackedSessionRegistry {
    inner: InMemorySessionRegistry,
    file: parking_lot::Mutex<std::fs::File>,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileBackedSessionRegistry {
    /// Open (creating if absent) the durable registry at `path` and load every
    /// record. Records are kept regardless of expiry: an expired record still
    /// blocks re-registration of its identifier (sessions are never
    /// reassigned), and `is_revoked` treats expired as revoked anyway.
    ///
    /// Corruption policy matches the credential store: a malformed COMPLETE
    /// line is a hard error; a malformed or newline-less FINAL fragment is a
    /// torn tail — truncated and logged, with a parseable fragment salvaged
    /// through the normal write path. A freshly created file fsyncs its
    /// directory; creating one where a store should pre-exist warns loudly.
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        let pre_existed = path.exists();
        let file = std::fs::OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(path)?;
        if !pre_existed {
            if let Some(parent) = path.parent() {
                std::fs::File::open(parent)?.sync_all()?;
            }
        }
        let registry = Self {
            inner: InMemorySessionRegistry::new(),
            file: parking_lot::Mutex::new(file.try_clone()?),
        };

        let content = std::fs::read(path)?;
        let complete_len = content
            .iter()
            .rposition(|&b| b == b'\n')
            .map_or(0, |i| i + 1);
        let mut loaded = 0usize;
        for line in content[..complete_len].split(|&b| b == b'\n') {
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let record: SessionRecord = serde_json::from_slice(line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("corrupt session record: {e}"),
                )
            })?;
            registry
                .apply_loaded(record)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            loaded += 1;
        }

        let tail = &content[complete_len..];
        if !tail.is_empty() {
            // Salvage a parseable tail fragment before truncating.
            let salvage = serde_json::from_slice::<SessionRecord>(tail).ok();
            {
                let mut file = registry.file.lock();
                file.set_len(complete_len as u64)?;
                if let Some(record) = salvage {
                    if record.key().is_some() {
                        let mut line = serde_json::to_string(&record).map_err(|e| {
                            std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("torn-tail salvage serialization failed: {e}"),
                            )
                        })?;
                        line.push('\n');
                        file.write_all(line.as_bytes())?;
                        registry
                            .apply_loaded(record)
                            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                        loaded += 1;
                    }
                }
                file.sync_all()?;
            }
            tracing::warn!(
                path = %path.display(),
                "torn session-log tail recovered: truncated to the last complete record"
            );
        }

        if pre_existed {
            tracing::info!(path = %path.display(), loaded, "session registry loaded");
        } else {
            tracing::warn!(
                path = %path.display(),
                "session registry did not exist — created a fresh EMPTY registry; \
                 on a non-fresh deployment every prior session record is lost"
            );
        }
        Ok(registry)
    }

    /// Apply one loaded record. Registrations insert (duplicate registration
    /// of a known key is log corruption — hard error); revocation markers
    /// flip a known key or are skipped when the registration is absent (its
    /// record may itself have been lost to an older torn tail).
    fn apply_loaded(&self, record: SessionRecord) -> Result<(), String> {
        let key = record.key().ok_or("corrupt session record: invalid key")?;
        match record.status.as_str() {
            "active" => {
                let state = record
                    .registration_state()
                    .ok_or("corrupt session record: incomplete registration")?;
                if self.inner.has_key(&key) {
                    return Err(format!(
                        "corrupt session log: duplicate registration for {key:?}"
                    ));
                }
                self.inner.insert_loaded(key, state);
                Ok(())
            }
            "revoked" => {
                self.inner.mark_revoked_loaded(&key);
                Ok(())
            }
            other => Err(format!("corrupt session record: unknown status {other:?}")),
        }
    }

    /// Append one record and fsync. Caller MUST hold the writer lock (the
    /// `file` guard is passed in) so check/append/publish stay atomic.
    fn append_durable(
        &self,
        file: &mut std::fs::File,
        record: &SessionRecord,
    ) -> Result<(), SessionRevokeError> {
        let mut line = serde_json::to_string(record)
            .map_err(|e| SessionRevokeError::new(format!("record serialization failed: {e}")))?;
        line.push('\n');
        file.write_all(line.as_bytes())
            .and_then(|()| file.sync_all())
            .map_err(|e| SessionRevokeError::new(format!("durable append failed: {e}")))
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait::async_trait]
impl SessionRegistry for FileBackedSessionRegistry {
    async fn session_state(&self, key: &SessionKey) -> Option<SessionState> {
        self.inner.session_state(key).await
    }

    async fn register_session(
        &self,
        key: SessionKey,
        state: SessionState,
    ) -> Result<(), SessionRegisterError> {
        // Validate FIRST — an invalid record never reaches the log.
        if !key.is_valid() || state.subject.is_empty() || state.tenant.is_empty() {
            return Err(super::credential::InvalidSessionRecord.into());
        }
        super::credential::validate_key_kind_coherence(&key, &state)?;
        let record = SessionRecord::from_registration(&key, &state);
        {
            // Writer lock: existence check + durable append + memory publish
            // are atomic against concurrent registrations of the same key.
            let mut file = self.file.lock();
            if self.inner.has_key(&key) {
                return Err(super::credential::SessionExists.into());
            }
            self.append_durable(&mut file, &record).map_err(|e| {
                tracing::error!(error = %e, "session registration durable append failed");
                SessionRegisterError::PublicationFailed(super::credential::SessionPublicationFailed)
            })?;
            self.inner.insert_loaded(key, state);
        }
        Ok(())
    }

    async fn revoke_session(&self, key: &SessionKey) -> Result<(), SessionRevokeError> {
        {
            let mut file = self.file.lock();
            if !self.inner.has_key(key) {
                // Revoking a never-registered session is a no-op (nothing to
                // publish); identifiers are random and unguessable, so this
                // cannot be used to pre-empt a future registration.
                return Ok(());
            }
            // Phase 1 — PUBLISH (durable): the marker is appended and fsync'd
            // BEFORE the in-memory status flips.
            self.append_durable(&mut file, &SessionRecord::revocation_marker(key))?;
            self.inner.mark_revoked_loaded(key);
        }
        // Phase 2 — EVICT: flush the verified-subject cache generation, so
        // cached handles derived from credentials carrying this session are
        // invalidated (strictly after publication).
        crate::auth::mac::flush_verified_subject_cache_generation();
        Ok(())
    }

    async fn is_revoked(&self, key: &SessionKey) -> bool {
        self.inner.is_revoked(key).await
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::auth::credential::SessionKind;

    /// Unique tempdir per test without a dev-dependency.
    struct TestDir(std::path::PathBuf);

    impl TestDir {
        fn new(tag: &str) -> Self {
            let unique = format!(
                "hyprstream-session-store-test-{}-{}-{:?}",
                std::process::id(),
                tag,
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            let path = std::env::temp_dir().join(unique);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn active_state(subject: &str, expires_at: i64) -> SessionState {
        SessionState {
            subject: subject.to_owned(),
            tenant: "default".to_owned(),
            kind: SessionKind::Interactive,
            created_at: 1,
            expires_at,
            status: ActiveOrRevoked::Active,
            clearance_epoch: 0,
        }
    }

    #[tokio::test]
    async fn register_revoke_survive_reopen_and_never_reassign() {
        let dir = TestDir::new("durability");
        let path = dir.path().join("sessions.jsonl");
        let now = chrono::Utc::now().timestamp();
        let key = SessionKey::oidc("https://issuer.example", "ses-durable");

        {
            let reg = FileBackedSessionRegistry::open(&path).unwrap();
            reg.register_session(key.clone(), active_state("alice", now + 3600))
                .await
                .unwrap();
            assert!(!reg.is_revoked(&key).await, "active after register");
            reg.revoke_session(&key).await.unwrap();
            assert!(reg.is_revoked(&key).await, "revoked after revoke");
        }

        let reopened = FileBackedSessionRegistry::open(&path).unwrap();
        assert!(
            reopened.is_revoked(&key).await,
            "revocation must survive reopen"
        );
        // Never reassignable: the revoked record still blocks registration.
        let result = reopened
            .register_session(key.clone(), active_state("alice", now + 7200))
            .await;
        assert!(
            matches!(result, Err(SessionRegisterError::Exists(_))),
            "revoked session identifier must never be reassigned, got {result:?}"
        );

        // An expired-but-never-revoked record also blocks re-registration.
        let expired_key = SessionKey::oidc("https://issuer.example", "ses-expired");
        {
            let reg = FileBackedSessionRegistry::open(&path).unwrap();
            reg.register_session(expired_key.clone(), active_state("bob", 1))
                .await
                .unwrap();
        }
        let reopened = FileBackedSessionRegistry::open(&path).unwrap();
        assert!(
            reopened.is_revoked(&expired_key).await,
            "expired session reports revoked"
        );
        let result = reopened
            .register_session(expired_key.clone(), active_state("bob", now + 3600))
            .await;
        assert!(
            matches!(result, Err(SessionRegisterError::Exists(_))),
            "expired session identifier must not be reassigned, got {result:?}"
        );
    }

    #[tokio::test]
    async fn duplicate_register_appends_nothing_and_errors() {
        let dir = TestDir::new("dup");
        let path = dir.path().join("sessions.jsonl");
        let now = chrono::Utc::now().timestamp();
        let key = SessionKey::oidc("https://issuer.example", "ses-dup");

        let reg = FileBackedSessionRegistry::open(&path).unwrap();
        reg.register_session(key.clone(), active_state("alice", now + 3600))
            .await
            .unwrap();
        let result = reg
            .register_session(key.clone(), active_state("alice", now + 3600))
            .await;
        assert!(matches!(result, Err(SessionRegisterError::Exists(_))));
        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            content.matches("ses-dup").count(),
            1,
            "a duplicate registration must not append"
        );
    }

    #[tokio::test]
    async fn torn_tail_recovers_and_midfile_corruption_fails() {
        let dir = TestDir::new("torn");
        let path = dir.path().join("sessions.jsonl");
        let now = chrono::Utc::now().timestamp();
        let key = SessionKey::oidc("https://issuer.example", "ses-torn");

        let complete = {
            let reg = FileBackedSessionRegistry::open(&path).unwrap();
            reg.register_session(key.clone(), active_state("alice", now + 3600))
                .await
                .unwrap();
            std::fs::read(&path).unwrap()
        };
        {
            use std::io::Write as _;
            let mut f = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .unwrap();
            f.write_all(b"{\"iss\":\"https://issuer.example\",\"kind\":\"oid")
                .unwrap();
        }
        let recovered = FileBackedSessionRegistry::open(&path).unwrap();
        assert!(
            !recovered.is_revoked(&key).await,
            "complete records survive torn-tail recovery"
        );
        assert_eq!(
            std::fs::read(&path).unwrap(),
            complete,
            "torn fragment is truncated to the last complete newline"
        );

        // Mid-file corruption still fails closed.
        std::fs::write(
            &path,
            b"{\"iss\":\"x\",\"kind\":\"oidc\",\"id\":\"a\",\"status\":\"active\",\"subject\":\"s\",\"tenant\":\"t\",\"created_at\":1,\"expires_at\":9999999999,\"clearance_epoch\":0}\ngarbage\n",
        )
        .unwrap();
        assert!(
            FileBackedSessionRegistry::open(&path).is_err(),
            "mid-file corruption must fail closed"
        );
    }

    #[tokio::test]
    async fn namespaces_and_validation_hold_through_the_durable_wrapper() {
        let dir = TestDir::new("ns");
        let path = dir.path().join("sessions.jsonl");
        let now = chrono::Utc::now().timestamp();
        let reg = FileBackedSessionRegistry::open(&path).unwrap();

        // Kind coherence is enforced before any log write.
        let wl_key = SessionKey::workload("https://issuer.example", "ses-kind");
        let result = reg
            .register_session(wl_key, active_state("svc", now + 3600))
            .await;
        assert!(
            matches!(result, Err(SessionRegisterError::KindMismatch(_))),
            "workload key + interactive state must fail, got {result:?}"
        );
        assert_eq!(
            std::fs::read_to_string(&path).unwrap().len(),
            0,
            "a rejected registration never reaches the log"
        );

        // OIDC vs workload namespaces stay disjoint on disk.
        let oidc_key = SessionKey::oidc("https://issuer.example", "ses-shared");
        let workload_key = SessionKey::workload("https://issuer.example", "ses-shared");
        let mut wl_state = active_state("svc", now + 3600);
        wl_state.kind = SessionKind::Workload;
        reg.register_session(oidc_key.clone(), active_state("alice", now + 3600))
            .await
            .unwrap();
        reg.register_session(workload_key.clone(), wl_state)
            .await
            .unwrap();
        reg.revoke_session(&oidc_key).await.unwrap();
        assert!(reg.is_revoked(&oidc_key).await);
        assert!(
            !reg.is_revoked(&workload_key).await,
            "workload session with the same id string is unaffected"
        );
        drop(reg);
        let reopened = FileBackedSessionRegistry::open(&path).unwrap();
        assert!(reopened.is_revoked(&oidc_key).await);
        assert!(!reopened.is_revoked(&workload_key).await);
    }

    #[tokio::test]
    async fn unknown_session_revoke_is_a_noop_without_log_write() {
        let dir = TestDir::new("unknown");
        let path = dir.path().join("sessions.jsonl");
        let reg = FileBackedSessionRegistry::open(&path).unwrap();
        let key = SessionKey::oidc("https://issuer.example", "ses-never-registered");
        reg.revoke_session(&key).await.unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap().len(),
            0,
            "revoking an unknown session must not append"
        );
        // And the identifier remains registerable (no pre-emption tombstone).
        let now = chrono::Utc::now().timestamp();
        reg.register_session(key.clone(), active_state("alice", now + 3600))
            .await
            .unwrap();
        assert!(!reg.is_revoked(&key).await);
    }
}
