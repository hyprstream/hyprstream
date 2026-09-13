//! FileAdapter-compatible persistence that waits for the filesystem write.

use casbin::{Adapter, FileAdapter, Filter, Model};
use std::path::PathBuf;
use tokio::io::AsyncWriteExt as _;

/// Casbin's Tokio FileAdapter drops its file after write_all without flushing.
/// Keep its parsing and incremental-operation behavior, but make full saves
/// complete before a caller captures or reopens the policy file.
pub(super) struct PolicyFileAdapter {
    path: PathBuf,
    inner: FileAdapter<PathBuf>,
}

impl PolicyFileAdapter {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            inner: FileAdapter::new(path.clone()),
            path,
        }
    }

    async fn write(&self, bytes: &[u8]) -> casbin::Result<()> {
        let mut file = tokio::fs::File::create(&self.path).await?;
        write_completed(&mut file, bytes).await?;
        Ok(())
    }
}

async fn write_completed(file: &mut tokio::fs::File, bytes: &[u8]) -> std::io::Result<()> {
    file.write_all(bytes).await?;
    // flush waits for Tokio's queued blocking write and propagates its errors.
    // This is completion, not crash durability; publication still owns fsync.
    file.flush().await
}

#[async_trait::async_trait]
impl Adapter for PolicyFileAdapter {
    async fn load_policy(&mut self, model: &mut dyn Model) -> casbin::Result<()> {
        self.inner.load_policy(model).await
    }

    async fn load_filtered_policy<'a>(
        &mut self,
        model: &mut dyn Model,
        filter: Filter<'a>,
    ) -> casbin::Result<()> {
        self.inner.load_filtered_policy(model, filter).await
    }

    async fn save_policy(&mut self, model: &mut dyn Model) -> casbin::Result<()> {
        if self.path.as_os_str().is_empty() {
            return Err(std::io::Error::other("save policy failed, file path is empty").into());
        }
        // Preserve FileAdapter's p/g sections, assertion names, row order,
        // separators and absence of a trailing newline, including g2 domains.
        let policies = model.get_model().get("p").ok_or_else(|| {
            casbin::error::ModelError::P("Missing policy definition in conf file".to_owned())
        })?;
        let mut rows = Vec::new();
        for (name, assertion) in policies
            .iter()
            .chain(model.get_model().get("g").into_iter().flatten())
        {
            for rule in assertion.get_policy() {
                rows.push(format!("{name}, {}", rule.join(", ")));
            }
        }
        self.write(rows.join("\n").as_bytes()).await
    }

    async fn clear_policy(&mut self) -> casbin::Result<()> {
        self.write(&[]).await
    }

    fn is_filtered(&self) -> bool {
        self.inner.is_filtered()
    }

    async fn add_policy(
        &mut self,
        sec: &str,
        ptype: &str,
        rule: Vec<String>,
    ) -> casbin::Result<bool> {
        self.inner.add_policy(sec, ptype, rule).await
    }

    async fn add_policies(
        &mut self,
        sec: &str,
        ptype: &str,
        rules: Vec<Vec<String>>,
    ) -> casbin::Result<bool> {
        self.inner.add_policies(sec, ptype, rules).await
    }

    async fn remove_policy(
        &mut self,
        sec: &str,
        ptype: &str,
        rule: Vec<String>,
    ) -> casbin::Result<bool> {
        self.inner.remove_policy(sec, ptype, rule).await
    }

    async fn remove_policies(
        &mut self,
        sec: &str,
        ptype: &str,
        rules: Vec<Vec<String>>,
    ) -> casbin::Result<bool> {
        self.inner.remove_policies(sec, ptype, rules).await
    }

    async fn remove_filtered_policy(
        &mut self,
        sec: &str,
        ptype: &str,
        index: usize,
        values: Vec<String>,
    ) -> casbin::Result<bool> {
        self.inner
            .remove_filtered_policy(sec, ptype, index, values)
            .await
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn policy_write_waits_for_blocking_io_completion() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .expect("runtime");
        runtime.block_on(async {
            let root = tempfile::tempdir().expect("policy directory");
            let path = root.path().join("policy.csv");
            let mut file = tokio::fs::File::create(&path).await.expect("policy file");
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started_tx.send(()).expect("started");
                release_rx.recv().expect("release");
            });
            started_rx.recv().expect("sole blocking worker occupied");
            let bytes = b"p, service:retained, *, model:*, ttt.writeback, deny";
            let write = write_completed(&mut file, bytes);
            tokio::pin!(write);
            let first_poll = futures::poll!(&mut write);
            let before_release = std::fs::read(&path).expect("read while write is queued");
            // Release before asserting, so a failed old-behavior control cannot
            // strand the runtime's sole blocking worker during teardown.
            release_tx.send(()).expect("release worker");
            blocker.await.expect("worker completed");
            assert!(
                first_poll.is_pending(),
                "save returned before its filesystem write completed"
            );
            assert!(before_release.is_empty(), "write must still be queued");
            write.await.expect("completed policy write");
            assert_eq!(std::fs::read(path).expect("immediate read"), bytes);
        });
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn policy_write_propagates_queued_io_error() {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .open("/dev/full")
            .await
            .expect("open failing write device");
        let error = write_completed(&mut file, b"retained deny")
            .await
            .expect_err("queued filesystem error must propagate");
        assert_eq!(error.raw_os_error(), Some(libc::ENOSPC));
    }

    #[tokio::test]
    async fn policy_adapter_roundtrip_preserves_deny_and_both_groupings() {
        use casbin::{CoreApi as _, DefaultModel, Enforcer, MgmtApi as _};
        let root = tempfile::tempdir().expect("policy directory");
        let path = root.path().join("policy.csv");
        super::super::PolicyManager::new(root.path())
            .await
            .expect("initialize model");
        let model_path = root.path().join("model.conf");
        let model = DefaultModel::from_file(&model_path).await.expect("model");
        let mut enforcer = Enforcer::new(model, casbin::MemoryAdapter::default())
            .await
            .expect("enforcer");
        enforcer
            .add_policy(
                ["retained", "*", "model:*", "ttt.writeback", "deny"]
                    .map(str::to_owned)
                    .to_vec(),
            )
            .await
            .expect("deny");
        enforcer
            .add_grouping_policy(["alice", "reader"].map(str::to_owned).to_vec())
            .await
            .expect("global grouping");
        enforcer
            .add_named_grouping_policy(
                "g2",
                ["bob", "reader", "tenant-a"].map(str::to_owned).to_vec(),
            )
            .await
            .expect("domain grouping");
        let mut adapter = PolicyFileAdapter::new(path.clone());
        adapter
            .save_policy(enforcer.get_mut_model())
            .await
            .expect("save");
        let serialized = std::fs::read_to_string(&path).expect("immediate read");
        assert_eq!(
            serialized
                .lines()
                .collect::<std::collections::BTreeSet<_>>(),
            [
                "p, retained, *, model:*, ttt.writeback, deny",
                "g, alice, reader",
                "g2, bob, reader, tenant-a",
            ]
            .into_iter()
            .collect()
        );
        assert!(!serialized.ends_with('\n'));
        let model = DefaultModel::from_file(&model_path)
            .await
            .expect("reload model");
        let reopened = Enforcer::new(model, PolicyFileAdapter::new(path.clone()))
            .await
            .expect("reopen");
        assert_eq!(reopened.get_policy(), enforcer.get_policy());
        assert_eq!(
            reopened.get_grouping_policy(),
            enforcer.get_grouping_policy()
        );
        assert_eq!(
            reopened.get_named_grouping_policy("g2"),
            enforcer.get_named_grouping_policy("g2")
        );
        adapter.clear_policy().await.expect("clear");
        assert!(std::fs::read(path)
            .expect("immediate clear read")
            .is_empty());
    }
}
