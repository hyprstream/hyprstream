//! **UCAN → Casbin/TE compiler foundation** — S5 / #571 of the native-MAC epic
//! (#547), milestone 1: the vocabulary-INDEPENDENT core.
//!
//! ## The design (epic #547, decided)
//!
//! UCAN is the **authoring / delegation source language**; Casbin/TE is the
//! **compiled enforcement**. An approved UCAN compiles *deterministically and
//! faithfully* into a Casbin/TE bundle that is a **ceiling**. A signed
//! `(UCAN, bundle_hash)` binding ([`approval`]) cryptographically ties the bundle
//! to the reviewed UCAN, so a compiler bug can never produce a bundle that
//! exceeds what was reviewed (design §14 containment backstop). Object labels and
//! the MAC floor come from manifests (S1), **never** from the UCAN — the compiler
//! produces the *grant ceiling*, not object labels.
//!
//! ```text
//!  authoring (control plane)                       enforcement (S4 data plane)
//!  ─────────────────────────                       ───────────────────────────
//!  UCAN delegation chain                            CompiledPolicy { TeMatrix, lattice, generation }
//!     │  verify_signatures (hybrid COSE)               ▲
//!     │  validate_chain (linkage + ATTENUATION ⊆)      │ PolicyLoader verifies ONCE
//!     ▼                                                │ + approval hash-match
//!  validated ceiling ──[emit, M2 seam]──► bundle ──► policy_hash (BLAKE3)
//!     │                                                ▲
//!     └──► SignedApproval{ ucan_cid, generation, ─────┘
//!          bundle_hash }   (hybrid EdDSA+ML-DSA-65)
//! ```
//!
//! ## Module layout
//!
//! - [`capability`] — the typed [`Capability`] `(resource, ability, caveats)` and
//!   the **attenuation (⊆)** relation. Vocabulary-independent; structural.
//! - [`token`] — the typed [`Ucan`] model, CBOR parse, structural validation, and
//!   **hybrid-COSE signature verification** (EdDSA + ML-DSA-65). Minimal model
//!   (we control both ends), aligned with the project crypto — NOT a JWT `ucan`
//!   crate.
//! - [`chain`] — **delegation-chain + ceiling/attenuation validation**. The most
//!   security-critical component: walks the proof chain and rejects any widening,
//!   broken linkage, or window escape. Fail-closed.
//! - [`approval`] — the signed `(UCAN, bundle_hash)` [`approval::ApprovalBinding`],
//!   binding the [`SecurityLabel`](crate::auth::mac) lattice generation too,
//!   signed with hybrid COSE. Its `bundle_hash` is exactly S4's
//!   `CompiledPolicy::policy_hash()` so it drops into the S4 loader's
//!   `PolicyApproval`.
//! - [`seam`] — milestone-2 interfaces left unimplemented: the action-vocabulary
//!   mapping (gated on S3/#582), Casbin/TE bundle emission, and the faithfulness
//!   framework. Scaffolded only.
//!
//! ## Relationship to S4 (#570/#583)
//!
//! S4's PDP (in the `hyprstream` crate, `mac::compiled::CompiledPolicy`) is the
//! consumer of this compiler's output. The emitted bundle is a
//! `hyprstream::mac::TeMatrix` wrapped in a `CompiledPolicy` whose
//! `generation == LatticeVersion::generation()` and whose `policy_hash()` is the
//! `bundle_hash` this module's approval binds. Bundle emission needs the
//! `hyprstream`-crate TE types, so it is a [`seam`] (milestone 2); the
//! milestone-1 core lives here in `hyprstream-rpc` and needs only S1 + crypto.

pub mod approval;
pub mod capability;
pub mod chain;
pub mod seam;
pub mod token;

pub use approval::{ucan_cid, ApprovalBinding, ApprovalError, SignedApproval, APPROVAL_AAD};
pub use capability::{set_attenuates, Ability, Capability, CaveatValue, Caveats, Resource};
pub use chain::{validate, validate_chain, ChainError};
pub use seam::{ActionVocabulary, BundleEmitter, FaithfulnessCheck};
pub use token::{Did, Ucan, UcanError, UcanPayload, UcanVerifier};
