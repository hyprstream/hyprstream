//! One closed, server-owned Federate model capability. This is a scope ceiling,
//! not a credential, model activation, filesystem permission or browser option.
use super::{
    clef_profile, StagingModelPin, REVIEWED_STAGING_MODEL_OID, REVIEWED_STAGING_MODEL_REF,
};
use anyhow::Result;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FederateModelProfile {
    Qwen,
    Clef,
}
impl FederateModelProfile {
    pub(crate) fn model_ref(self) -> &'static str {
        match self {
            Self::Qwen => REVIEWED_STAGING_MODEL_REF,
            Self::Clef => clef_profile::MODEL_REF,
        }
    }
    pub(crate) fn commit(self) -> &'static str {
        match self {
            Self::Qwen => REVIEWED_STAGING_MODEL_OID,
            Self::Clef => clef_profile::COMMIT,
        }
    }
    pub(crate) fn repo_name(self) -> &'static str {
        match self {
            Self::Qwen => StagingModelPin::reviewed_repo_name(),
            Self::Clef => "clef-flash-fp8",
        }
    }
    pub(crate) fn scopes(self) -> BTreeSet<String> {
        BTreeSet::from([
            format!("infer:model:{}", self.model_ref()),
            "query:model:Status".to_owned(),
            format!("query:registry:{}", self.repo_name()),
            "write:model:Load".to_owned(),
        ])
    }
    /// Policy needs the trusted selector and fixed identity, never the model
    /// payload/cache or inference compute permissions. Model/Registry separately
    /// require the startup-qualified StagingModelPin marker for actual use.
    pub(crate) fn from_env() -> Result<Self> {
        fn optional(name: &str) -> Result<Option<String>> {
            match std::env::var(name) {
                Ok(value) => Ok(Some(value)),
                Err(std::env::VarError::NotPresent) => Ok(None),
                Err(error) => Err(anyhow::anyhow!("read {name}: {error}")),
            }
        }
        let profile =
            optional("HYPRSTREAM_MODEL_ADMISSION_PROFILE")?.unwrap_or_else(|| "default".to_owned());
        let model_ref = optional("HYPRSTREAM_STAGING_MODEL_REF")?;
        let oid = optional("HYPRSTREAM_STAGING_MODEL_OID")?;
        Self::from_settings(&profile, model_ref.as_deref(), oid.as_deref())
    }
    fn from_settings(profile: &str, model_ref: Option<&str>, oid: Option<&str>) -> Result<Self> {
        if profile == clef_profile::PROFILE {
            StagingModelPin::validate_clef_overrides(model_ref, oid)?;
            return Ok(Self::Clef);
        }
        // Preserve existing default/staging configuration validation and Qwen
        // ceiling. An arbitrary custom default pin still gains NO Federate
        // capability: the serving recognizers reject its absent marker.
        let _ = StagingModelPin::from_settings(profile, model_ref, oid)?;
        Ok(Self::Qwen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_scope_selection_preserves_default_and_rejects_cross_profile_overrides() -> Result<()>
    {
        for name in ["default", "staging"] {
            assert_eq!(
                FederateModelProfile::from_settings(name, None, None)?,
                FederateModelProfile::Qwen
            );
        }
        let qwen = FederateModelProfile::Qwen.scopes();
        assert_eq!(
            qwen,
            BTreeSet::from([
                "infer:model:qwen2.5-0.5b-instruct:main".to_owned(),
                "query:model:Status".to_owned(),
                "query:registry:qwen2.5-0.5b-instruct".to_owned(),
                "write:model:Load".to_owned(),
            ])
        );
        let clef = FederateModelProfile::from_settings(clef_profile::PROFILE, None, None)?;
        assert_eq!(clef, FederateModelProfile::Clef);
        assert_eq!(clef.scopes().len(), 4);
        assert!(!clef.scopes().contains("query:registry:List"));
        assert!(!clef
            .scopes()
            .contains("infer:model:qwen2.5-0.5b-instruct:main"));
        for (profile, name, oid) in [
            ("unknown", None, None),
            (
                "staging",
                Some(clef_profile::MODEL_REF),
                Some(clef_profile::COMMIT),
            ),
            (
                clef_profile::PROFILE,
                Some(REVIEWED_STAGING_MODEL_REF),
                Some(REVIEWED_STAGING_MODEL_OID),
            ),
            (
                clef_profile::PROFILE,
                Some("clef-flash-fp8:other"),
                Some(clef_profile::COMMIT),
            ),
            (clef_profile::PROFILE, Some(clef_profile::MODEL_REF), None),
            (clef_profile::PROFILE, None, Some(clef_profile::COMMIT)),
        ] {
            assert!(FederateModelProfile::from_settings(profile, name, oid).is_err());
        }
        assert_eq!(
            FederateModelProfile::from_settings(
                clef_profile::PROFILE,
                Some(clef_profile::MODEL_REF),
                Some(clef_profile::COMMIT)
            )?,
            clef
        );
        Ok(())
    }
}
