//! Exact selected-model scope ceiling, derived from protected startup config.
use crate::config::model_admission::{ModelOperation, ValidatedModelAdmission};
use anyhow::Result;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FederateModelProfile {
    model_ref: String,
    commit: String,
    repository_name: String,
    operation: ModelOperation,
}
impl FederateModelProfile {
    pub(crate) fn from_admission(admission: &ValidatedModelAdmission) -> Self {
        Self {
            model_ref: admission.model_ref.clone(),
            commit: admission.commit.to_string(),
            repository_name: admission.repository_name.clone(),
            operation: admission.operation,
        }
    }
    pub(crate) fn from_config(config: &crate::config::HyprConfig) -> Result<Option<Self>> {
        Ok(config.selected_model_admission()?.map(Self::from_admission))
    }
    pub(crate) fn model_ref(&self) -> &str {
        &self.model_ref
    }
    pub(crate) fn commit(&self) -> &str {
        &self.commit
    }
    pub(crate) fn repo_name(&self) -> &str {
        &self.repository_name
    }
    pub(crate) fn operation(&self) -> ModelOperation {
        self.operation
    }
    pub(crate) fn scopes(&self) -> BTreeSet<String> {
        BTreeSet::from([
            format!("{}:model:{}", self.operation.action(), self.model_ref),
            "query:model:Status".into(),
            format!("query:registry:{}", self.repository_name),
            "write:model:Load".into(),
        ])
    }
    #[cfg(test)]
    pub(crate) fn qwen_fixture() -> Self {
        Self {
            model_ref: super::REVIEWED_STAGING_MODEL_REF.into(),
            commit: super::REVIEWED_STAGING_MODEL_OID.into(),
            repository_name: super::StagingModelPin::reviewed_repo_name().into(),
            operation: ModelOperation::Generation,
        }
    }
    #[cfg(test)]
    pub(crate) fn clef_fixture() -> Self {
        Self {
            model_ref: super::clef_profile::MODEL_REF.into(),
            commit: super::clef_profile::COMMIT.into(),
            repository_name: "clef-flash-fp8".into(),
            operation: ModelOperation::Decisions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scope_ceiling_tracks_identity_and_closed_operation() {
        for operation in [ModelOperation::Generation, ModelOperation::Decisions] {
            for name in ["first:main", "second:main"] {
                let profile = FederateModelProfile {
                    model_ref: name.into(),
                    commit: "a".repeat(40),
                    repository_name: name.split(':').next().unwrap_or_default().into(),
                    operation,
                };
                let scopes = profile.scopes();
                assert_eq!(scopes.len(), 4);
                assert!(scopes.contains(&format!("{}:model:{name}", operation.action())));
                let other = if operation == ModelOperation::Decisions {
                    "infer"
                } else {
                    "decide"
                };
                assert!(!scopes.contains(&format!("{other}:model:{name}")));
                assert!(!scopes.contains("query:registry:List"));
            }
        }
    }
}
