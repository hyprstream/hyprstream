//! Verified, revocable attachment state shared by RPC-adjacent subsystems.
//!
//! An attachment is deliberately **not** a wire bearer or a replacement for
//! MAC labels.  It is a process-local TCB object created at a verified request
//! boundary. Consumers receive a scoped operation grant carrying a generation
//! snapshot and must check it immediately before each external effect or
//! continuation.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use rand::RngCore;
use thiserror::Error;

use crate::Subject;
#[cfg(not(target_arch = "wasm32"))]
use crate::service::EnvelopeContext;

/// Opaque identifier for one verified attachment.
///
/// The bytes are generated inside this module and intentionally have no public
/// constructor.  They are correlation material, not an authorization bearer.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AttachmentId([u8; 16]);

impl fmt::Debug for AttachmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AttachmentId")
            .field(&hex::encode(self.0))
            .finish()
    }
}

impl fmt::Display for AttachmentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&hex::encode(self.0))
    }
}

/// Monotonic authority generation associated with an attachment.
///
/// A lease is valid only while its stored generation equals the attachment's
/// current generation.  The inner value remains private so callers cannot
/// manufacture a fresh authority snapshot from a stale one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AuthorityGeneration(u64);

impl AuthorityGeneration {
    /// Numeric representation for correlation and durable audit records.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Operation scopes carried by an attachment grant.
///
/// Identity/revocation and operation permission are intentionally separate:
/// a verified subject is not automatically permitted to allocate or control a
/// Worker task. Production grant issuance is deferred until the dispatch PEP
/// decision and delegated-capability decision can both mint this opaque value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttachmentOperation {
    TaskSpawn,
    TaskAttach,
    TaskSignal,
    TaskRead,
    TaskPublish,
}

impl AttachmentOperation {
    const fn bit(self) -> u8 {
        match self {
            Self::TaskSpawn => 1 << 0,
            Self::TaskAttach => 1 << 1,
            Self::TaskSignal => 1 << 2,
            Self::TaskRead => 1 << 3,
            Self::TaskPublish => 1 << 4,
        }
    }
}

/// Why attachment validation failed.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum AttachmentError {
    /// The context was created by a public compatibility or callback
    /// constructor rather than the verified envelope pipeline.
    #[error("attachment requires an envelope context produced by verified dispatch")]
    UnverifiedEnvelopeContext,
    /// A verified boundary did not yield a concrete principal.
    #[error("attachment requires a non-anonymous verified subject")]
    AnonymousSubject,
    /// The authority generation captured by a lease has been revoked.
    #[error(
        "attachment {attachment} authority generation is stale (expected {expected}, current {current})"
    )]
    StaleAuthority {
        attachment: AttachmentId,
        expected: AuthorityGeneration,
        current: AuthorityGeneration,
    },
    /// Generation exhaustion is refused rather than wrapping to a value that
    /// could accidentally validate an old lease.
    #[error("attachment {attachment} authority generation is exhausted")]
    GenerationExhausted { attachment: AttachmentId },
    /// The attachment is current but has no permit for this operation.
    #[error("attachment operation {operation:?} was not granted")]
    OperationNotGranted { operation: AttachmentOperation },
}

#[derive(Debug)]
struct AttachmentState {
    generation: AtomicU64,
}

/// Process-local identity/revocation attachment created at a verified boundary.
///
/// Its fields are private: callers cannot pair an arbitrary [`Subject`] with
/// an attachment id or generation.  Production construction derives the
/// subject from [`EnvelopeContext`]; the only other constructor is explicitly
/// test-only and is feature-gated for integration fixtures.
#[derive(Clone, Debug)]
pub struct VerifiedAttachment {
    id: AttachmentId,
    subject: Subject,
    state: Arc<AttachmentState>,
}

impl VerifiedAttachment {
    /// Create an attachment from the subject already authenticated by RPC
    /// envelope verification.  Caller-supplied request fields do not enter
    /// this constructor. A public `EnvelopeContext` constructor or a
    /// hand-built `SignedEnvelope` is not sufficient: this requires the
    /// private provenance marker installed by `unwrap_envelope` after
    /// signature/replay verification.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_envelope(context: &EnvelopeContext) -> Result<Self, AttachmentError> {
        let subject = context
            .verified_attachment_subject()
            .ok_or(AttachmentError::UnverifiedEnvelopeContext)?;
        Self::from_verified_subject(subject)
    }

    fn from_verified_subject(subject: Subject) -> Result<Self, AttachmentError> {
        if subject.is_anonymous() {
            return Err(AttachmentError::AnonymousSubject);
        }
        let mut bytes = [0_u8; 16];
        let mut rng = rand::rngs::OsRng;
        rng.fill_bytes(&mut bytes);
        Ok(Self {
            id: AttachmentId(bytes),
            subject,
            state: Arc::new(AttachmentState {
                generation: AtomicU64::new(0),
            }),
        })
    }

    /// Build a trusted local-test root.
    ///
    /// This is intentionally compiled only for test fixture builds.  Production
    /// code must use [`Self::from_envelope`], while the 9P attach bridge will
    /// obtain the same private state from its verified attach boundary in a
    /// follow-up slice.
    #[cfg(any(test, feature = "test-classical-policy"))]
    pub fn for_test_local_root(subject: Subject) -> Result<Self, AttachmentError> {
        Self::from_verified_subject(subject)
    }

    /// Opaque attachment identifier, suitable for correlation but not bearer
    /// authorization.
    #[must_use]
    pub fn id(&self) -> &AttachmentId {
        &self.id
    }

    /// Subject that was derived at the verified construction boundary.
    #[must_use]
    pub fn subject(&self) -> &Subject {
        &self.subject
    }

    /// Capture the generation state held inside an operation grant.
    fn lease(&self) -> AttachmentLease {
        AttachmentLease {
            attachment: self.clone(),
            generation: AuthorityGeneration(self.state.generation.load(Ordering::Acquire)),
        }
    }

    /// Mint an explicit scope grant for a test-only fake vertical slice.
    ///
    /// This is deliberately unavailable in production builds. A production
    /// issuer must require both an admitted dispatch-MAC decision and a
    /// delegated-capability decision; a verified identity alone cannot mint a
    /// task grant.
    #[cfg(any(test, feature = "test-classical-policy"))]
    #[must_use]
    pub fn for_test_operations(
        &self,
        operations: &[AttachmentOperation],
    ) -> AttachmentOperationGrant {
        let scopes = operations
            .iter()
            .fold(0_u8, |scopes, operation| scopes | operation.bit());
        AttachmentOperationGrant {
            lease: self.lease(),
            scopes,
        }
    }

    /// Revoke all outstanding leases and return the new authority generation.
    ///
    /// The caller that owns the authority record chooses when to call this;
    /// consumers cannot advance the generation through a lease.
    pub fn revoke(&self) -> Result<AuthorityGeneration, AttachmentError> {
        let current = self
            .state
            .generation
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |generation| {
                generation.checked_add(1)
            })
            .map_err(|_| AttachmentError::GenerationExhausted {
                attachment: self.id.clone(),
            })?;
        Ok(AuthorityGeneration(current + 1))
    }
}

/// Generation-bound state retained inside an operation grant.
///
/// It preserves the attachment's subject locally, but neither the subject nor
/// generation can be supplied by a normal VFS/RPC caller. It is intentionally
/// not exposed outside this module: only an operation grant carries it onward.
#[derive(Clone, Debug)]
struct AttachmentLease {
    attachment: VerifiedAttachment,
    generation: AuthorityGeneration,
}

impl AttachmentLease {
    /// Attachment identity for correlation.
    #[must_use]
    pub fn attachment_id(&self) -> &AttachmentId {
        self.attachment.id()
    }

    /// Local subject view derived at the verified boundary.
    #[must_use]
    pub fn subject(&self) -> &Subject {
        self.attachment.subject()
    }

    /// Generation captured when the lease was created.
    #[must_use]
    pub const fn generation(&self) -> AuthorityGeneration {
        self.generation
    }

    /// Fail closed unless this lease still names the attachment's current
    /// generation.  Call immediately before every lifecycle effect.
    pub fn ensure_current(&self) -> Result<(), AttachmentError> {
        let current = AuthorityGeneration(self.attachment.state.generation.load(Ordering::Acquire));
        if current == self.generation {
            Ok(())
        } else {
            Err(AttachmentError::StaleAuthority {
                attachment: self.attachment.id.clone(),
                expected: self.generation,
                current,
            })
        }
    }
}

/// Opaque, generation-bound operation permit for an attachment.
///
/// The fields are private and there is intentionally no production
/// constructor. It is the future hand-off point from the dispatch PEP plus a
/// separate delegated-capability decision, rather than a rebranding of a
/// verified subject or MAC label.
#[derive(Clone, Debug)]
pub struct AttachmentOperationGrant {
    lease: AttachmentLease,
    scopes: u8,
}

impl AttachmentOperationGrant {
    /// Attachment identity for correlation/audit records.
    #[must_use]
    pub fn attachment_id(&self) -> &AttachmentId {
        self.lease.attachment_id()
    }

    /// Local subject view retained by the trusted attachment boundary.
    #[must_use]
    pub fn subject(&self) -> &Subject {
        self.lease.subject()
    }

    /// Generation captured when the grant was minted.
    #[must_use]
    pub const fn generation(&self) -> AuthorityGeneration {
        self.lease.generation()
    }

    /// Require both a current generation and the exact operation scope.
    pub fn ensure(&self, operation: AttachmentOperation) -> Result<(), AttachmentError> {
        self.lease.ensure_current()?;
        if self.scopes & operation.bit() == 0 {
            return Err(AttachmentError::OperationNotGranted { operation });
        }
        Ok(())
    }

    /// Recheck revocation without selecting an operation. This is only for
    /// comparison/correlation paths; effect sites must use [`Self::ensure`].
    pub fn ensure_current(&self) -> Result<(), AttachmentError> {
        self.lease.ensure_current()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(target_arch = "wasm32"))]
    use crate::{RequestEnvelope, SignedEnvelope};

    #[test]
    fn local_root_requires_a_principal_and_revokes_existing_operation_grants() {
        assert_eq!(
            VerifiedAttachment::for_test_local_root(Subject::anonymous()).unwrap_err(),
            AttachmentError::AnonymousSubject
        );

        let attachment = VerifiedAttachment::for_test_local_root(Subject::new("alice")).unwrap();
        let grant = attachment.for_test_operations(&[AttachmentOperation::TaskRead]);
        assert!(grant.ensure(AttachmentOperation::TaskRead).is_ok());
        assert_eq!(
            grant.ensure(AttachmentOperation::TaskSpawn),
            Err(AttachmentError::OperationNotGranted {
                operation: AttachmentOperation::TaskSpawn,
            })
        );
        assert_eq!(attachment.revoke().unwrap().get(), 1);
        assert!(matches!(
            grant.ensure(AttachmentOperation::TaskRead),
            Err(AttachmentError::StaleAuthority { .. })
        ));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn attachment_rejects_callback_and_public_signed_envelope_contexts() {
        let callback = EnvelopeContext::from_callback_service(7, "forged-service");
        assert_eq!(
            VerifiedAttachment::from_envelope(&callback).unwrap_err(),
            AttachmentError::UnverifiedEnvelopeContext
        );

        // `SignedEnvelope` fields and the compatibility system constructor are
        // public, so neither is proof that the envelope passed verification.
        let signing_key = crate::SigningKey::from_bytes(&[9_u8; 32]);
        let signed =
            SignedEnvelope::new_signed(RequestEnvelope::anonymous(Vec::new()), &signing_key);
        let public_system = EnvelopeContext::from_verified_as_system(&signed);
        assert_eq!(
            VerifiedAttachment::from_envelope(&public_system).unwrap_err(),
            AttachmentError::UnverifiedEnvelopeContext
        );

        // The crate-private dispatch constructor carries the marker installed
        // only after `unwrap_and_verify` has admitted the envelope.
        let verified_system = EnvelopeContext::from_verified_fixed_signer(&signed);
        assert_eq!(
            VerifiedAttachment::from_envelope(&verified_system)
                .unwrap()
                .subject(),
            &Subject::new("system")
        );
    }
}
