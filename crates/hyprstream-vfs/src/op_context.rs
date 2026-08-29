//! Verified attachment context for the direct VFS operation seam.
//!
//! `Mount` still takes `&Subject` for compatibility.  This type is the narrow
//! migration path for a mount that must also bind lifecycle effects to a
//! revocable attachment and operation grant. It cannot be built from a
//! caller-supplied Subject or from a plaintext wire payload: construction
//! consumes an opaque [`hyprstream_rpc::AttachmentOperationGrant`] minted by a
//! trusted policy/capability boundary.

use hyprstream_rpc::{
    AttachmentError, AttachmentId, AttachmentOperation, AttachmentOperationGrant,
    AuthorityGeneration, Subject,
};

use crate::MountError;

/// TCB-created context for one local VFS operation.
#[derive(Clone, Debug)]
pub struct VfsOpContext {
    grant: AttachmentOperationGrant,
}

impl VfsOpContext {
    /// Derive a context from an opaque, policy-issued operation grant.
    ///
    /// A bare verified attachment is deliberately insufficient: identity and
    /// revocation do not authorize a VFS or Worker effect.
    #[must_use]
    pub fn from_attachment_grant(grant: &AttachmentOperationGrant) -> Self {
        Self {
            grant: grant.clone(),
        }
    }

    /// Whether two contexts preserve exactly the same opaque operation grant.
    ///
    /// Trusted transport hand-off boundaries use this to make reattach
    /// idempotent only for the same attachment, generation, subject, and
    /// scope set. It does not authorize an operation by itself.
    #[must_use]
    pub fn same_grant(&self, other: &Self) -> bool {
        self.grant.same_grant(&other.grant)
    }

    /// The local subject view derived at the verified attachment boundary.
    #[must_use]
    pub fn subject(&self) -> &Subject {
        self.grant.subject()
    }

    /// Opaque attachment identifier for correlation/audit records.
    #[must_use]
    pub fn attachment_id(&self) -> &AttachmentId {
        self.grant.attachment_id()
    }

    /// Authority generation captured for this operation.
    #[must_use]
    pub const fn authority_generation(&self) -> AuthorityGeneration {
        self.grant.generation()
    }

    /// Refuse a lifecycle effect if this operation's authority was revoked.
    pub fn ensure_current(&self) -> Result<(), MountError> {
        self.grant.ensure_current().map_err(revoked_mount_error)
    }

    /// Require an operation scope at an actual VFS effect boundary.
    pub fn ensure_operation(&self, operation: AttachmentOperation) -> Result<(), MountError> {
        self.grant.ensure(operation).map_err(revoked_mount_error)
    }
}

fn revoked_mount_error(error: AttachmentError) -> MountError {
    match error {
        AttachmentError::StaleAuthority { .. } => MountError::PermissionDenied(
            "attachment authority was revoked before this VFS effect".to_owned(),
        ),
        // Both cases are an invalid trusted-context state from the mount's
        // perspective. Keep the detailed internal error out of the VFS path.
        AttachmentError::UnverifiedEnvelopeContext
        | AttachmentError::AnonymousSubject
        | AttachmentError::GenerationExhausted { .. }
        | AttachmentError::OperationNotGranted { .. } => {
            MountError::PermissionDenied("invalid verified attachment context".to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_rpc::VerifiedAttachment;

    #[test]
    fn context_uses_attachment_subject_and_fails_after_revocation() {
        let attachment = VerifiedAttachment::for_test_local_root(Subject::new("alice")).unwrap();
        let grant = attachment.for_test_operations(&[AttachmentOperation::TaskRead]);
        let context = VfsOpContext::from_attachment_grant(&grant);

        assert_eq!(context.subject(), &Subject::new("alice"));
        assert_eq!(context.authority_generation().get(), 0);
        assert!(context.ensure_current().is_ok());
        assert!(
            context
                .ensure_operation(AttachmentOperation::TaskRead)
                .is_ok()
        );
        assert!(matches!(
            context.ensure_operation(AttachmentOperation::TaskSpawn),
            Err(MountError::PermissionDenied(_))
        ));

        attachment.revoke().unwrap();
        assert!(matches!(
            context.ensure_current(),
            Err(MountError::PermissionDenied(_))
        ));
    }
}
