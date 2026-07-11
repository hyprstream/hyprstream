//! INV-2 (ADR #1023): no cleartext-envelope over an untrusted transport carrier.
//!
//! # The invariant
//!
//! ADR #1023 ratifies iroh (and any relay/cross-host QUIC path) as an
//! **untrusted carrier**: the app-layer PQ-hybrid tunnel — HyKEM key agreement
//! (#551) + hybrid-PQC COSE envelopes — is the sole trust and confidentiality
//! root. A `SignedEnvelope` in **cleartext mode** (`encrypted_envelope = None`,
//! see `envelope.rs`) sent over such a carrier silently delegates confidentiality
//! to the carrier's transport TLS, which is:
//!
//! - **iroh:** authenticated only by the peer's *classical* Ed25519 NodeId,
//! - **native-only:** the pinned PQ kx provider does not exist on wasm32 — the
//!   browser path rides the browser's TLS stack (classical), and
//! - **passive/HNDL-only:** an active CRQC adversary voids the channel.
//!
//! INV-2 therefore FORBIDS cleartext `SignedEnvelope`s on iroh-dialed and
//! relay-carried paths. Same-host `inproc` (a function call) and local `UDS`
//! (peer-credential authenticated, never leaves the host) remain legitimate
//! cleartext carriers — the ban must NOT be over-broadened onto them.
//!
//! Enforcement is **structural**: [`guard_cleartext_envelope`] is called on the
//! send path (`RpcClientImpl::sign_envelope`) with the dial-time carrier
//! classification ([`crate::transport::EndpointType::forbids_cleartext_envelope`]).
//!
//! # Current status — GATED on the HyKEM tunnel (#551 / #550)
//!
//! As of this writing the PQ-hybrid tunnel is **NOT wired into the RPC send
//! path**: `RpcClientImpl::sign_envelope` always emits `encrypted_envelope =
//! None`, and the only producer of an encrypted envelope
//! (`SignedEnvelope::new_signed_encrypted_mesh_kem`) is exercised only in a
//! unit test. Consequently, hard-enforcing INV-2 today would make **every** iroh
//! (and cross-host QUIC) RPC fail closed — bricking federation — because nothing
//! can currently produce a compliant encrypted envelope on that path.
//!
//! So enforcement is gated by [`HYKEM_IROH_TUNNEL_WIRED`]. Until that flips
//! `true` (the HyKEM tunnel is made mandatory on iroh/relay send paths), the
//! default mode is [`Inv2Mode::WarnOnly`]: the hole is recorded loudly at the
//! send boundary but the send proceeds so the daemon keeps working. Operators
//! (and tests) can force hard enforcement via `HYPRSTREAM_INV2=enforce` or
//! [`set_inv2_mode`]. When the tunnel lands, flip [`HYKEM_IROH_TUNNEL_WIRED`] to
//! `true` and the default becomes fail-closed [`Inv2Mode::Enforce`] with no
//! caller change — that is the concrete "close the hole" step.

use crate::error::{Result, RpcError};

/// Prerequisite gate: `true` once the app-layer PQ-hybrid (HyKEM #551) tunnel is
/// wired **mandatory** on iroh/relay RPC send paths so those paths can actually
/// produce `encrypted_envelope = Some(..)`. While `false`, hard-enforcing INV-2
/// would fail-close all such traffic, so the default mode is `WarnOnly`.
///
/// Flip this to `true` **only** together with wiring the tunnel into
/// `RpcClientImpl::sign_envelope` (or the transport send boundary). Doing so
/// makes [`default_inv2_mode`] return [`Inv2Mode::Enforce`] — the fail-closed
/// end state INV-2 requires — with no other change.
pub const HYKEM_IROH_TUNNEL_WIRED: bool = false;

/// INV-2 enforcement mode for the send path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inv2Mode {
    /// Fail closed: a cleartext envelope over an untrusted carrier is an error.
    Enforce,
    /// Record the violation loudly but allow the send (interim, tunnel-not-wired).
    WarnOnly,
}

/// The default mode when nothing has been explicitly configured.
///
/// Fail-closed [`Inv2Mode::Enforce`] once [`HYKEM_IROH_TUNNEL_WIRED`] is `true`;
/// [`Inv2Mode::WarnOnly`] until then (see module docs — enforcing before the
/// tunnel exists bricks iroh).
pub const fn default_inv2_mode() -> Inv2Mode {
    if HYKEM_IROH_TUNNEL_WIRED {
        Inv2Mode::Enforce
    } else {
        Inv2Mode::WarnOnly
    }
}

#[cfg(not(target_arch = "wasm32"))]
static INV2_MODE: std::sync::OnceLock<Inv2Mode> = std::sync::OnceLock::new();

/// Name of the operator escape hatch. `enforce` forces fail-closed INV-2 even
/// before the tunnel is wired (federation over iroh will then fail closed, as
/// intended for a hardened deployment); `warn` forces warn-only.
pub const INV2_ENV: &str = "HYPRSTREAM_INV2";

/// Explicitly pin the process-global INV-2 mode. First write wins; returns the
/// value in force (this call's on success, the prior one if already set).
#[cfg(not(target_arch = "wasm32"))]
pub fn set_inv2_mode(mode: Inv2Mode) -> Inv2Mode {
    match INV2_MODE.set(mode) {
        Ok(()) => mode,
        Err(_) => *INV2_MODE.get().unwrap_or(&mode),
    }
}

/// The effective INV-2 mode: explicit install ([`set_inv2_mode`]) wins, else the
/// `HYPRSTREAM_INV2` env var, else [`default_inv2_mode`].
#[cfg(not(target_arch = "wasm32"))]
pub fn inv2_mode() -> Inv2Mode {
    if let Some(m) = INV2_MODE.get() {
        return *m;
    }
    match std::env::var(INV2_ENV).ok().as_deref() {
        Some("enforce") | Some("1") | Some("true") => Inv2Mode::Enforce,
        Some("warn") | Some("0") | Some("false") => Inv2Mode::WarnOnly,
        _ => default_inv2_mode(),
    }
}

/// WASM has no env/process-global config surface here; use the compile-time
/// default (browser path is exactly the untrusted-carrier case INV-2 targets,
/// but the tunnel-not-wired gate applies identically).
#[cfg(target_arch = "wasm32")]
pub fn inv2_mode() -> Inv2Mode {
    default_inv2_mode()
}

/// Pure decision — separated from the global read so it is deterministically
/// testable. Returns whether the send is allowed, and whether to warn.
///
/// `carrier_forbids_cleartext`: the dial-time carrier classification
/// (`EndpointType::forbids_cleartext_envelope`).
/// `envelope_is_encrypted`: `SignedEnvelope::is_encrypted()` (i.e.
/// `encrypted_envelope.is_some()`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inv2Decision {
    /// Compliant (encrypted, or a trusted same-host carrier) — proceed silently.
    Allow,
    /// Violation, but mode is `WarnOnly` — proceed after logging.
    Warn,
    /// Violation and mode is `Enforce` — fail closed.
    Deny,
}

/// The load-bearing decision. A cleartext envelope over an untrusted carrier is
/// a violation; the mode decides warn-vs-deny. Everything else is `Allow`.
pub const fn decide(
    mode: Inv2Mode,
    carrier_forbids_cleartext: bool,
    envelope_is_encrypted: bool,
) -> Inv2Decision {
    if !carrier_forbids_cleartext || envelope_is_encrypted {
        return Inv2Decision::Allow;
    }
    // Violation: cleartext (encrypted_envelope = None) over an untrusted carrier.
    match mode {
        Inv2Mode::Enforce => Inv2Decision::Deny,
        Inv2Mode::WarnOnly => Inv2Decision::Warn,
    }
}

/// Structural INV-2 guard for the send path. Call immediately before handing a
/// serialized `SignedEnvelope` to an untrusted-carrier transport.
///
/// - Trusted carrier (inproc/UDS) or already-encrypted envelope → `Ok(())`.
/// - Cleartext over untrusted carrier + `Enforce` → `Err` (fail closed).
/// - Cleartext over untrusted carrier + `WarnOnly` → `Ok(())` after a loud
///   (rate-limited) warning recording the open hole.
pub fn guard_cleartext_envelope(
    carrier_forbids_cleartext: bool,
    envelope_is_encrypted: bool,
) -> Result<()> {
    match decide(inv2_mode(), carrier_forbids_cleartext, envelope_is_encrypted) {
        Inv2Decision::Allow => Ok(()),
        Inv2Decision::Warn => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        "INV-2 (ADR #1023): sending a CLEARTEXT SignedEnvelope over an \
                         untrusted carrier (iroh/relay/cross-host QUIC). Confidentiality is \
                         delegated to transport TLS (classical, native-only, HNDL). This is \
                         permitted only because the PQ-hybrid HyKEM tunnel (#551) is not yet \
                         wired on this path; set HYPRSTREAM_INV2=enforce to fail closed."
                    );
                });
            }
            Ok(())
        }
        Inv2Decision::Deny => Err(RpcError::InvalidOperation(
            "INV-2 (ADR #1023): refusing to send a cleartext SignedEnvelope over an \
             untrusted carrier (iroh/relay/cross-host QUIC); an encrypted (PQ-hybrid \
             tunnel) envelope is required on this path"
                .to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trusted_carrier_always_allows() {
        // inproc/UDS classify as `carrier_forbids_cleartext = false`.
        assert_eq!(decide(Inv2Mode::Enforce, false, false), Inv2Decision::Allow);
        assert_eq!(decide(Inv2Mode::WarnOnly, false, false), Inv2Decision::Allow);
    }

    #[test]
    fn encrypted_envelope_always_allows() {
        // A compliant PQ-hybrid tunnel envelope passes on any carrier.
        assert_eq!(decide(Inv2Mode::Enforce, true, true), Inv2Decision::Allow);
        assert_eq!(decide(Inv2Mode::WarnOnly, true, true), Inv2Decision::Allow);
    }

    #[test]
    fn cleartext_over_untrusted_carrier_is_denied_when_enforced() {
        // The core INV-2 invariant: cleartext (encrypted=false) over an untrusted
        // carrier (forbids=true) fails closed under Enforce.
        assert_eq!(decide(Inv2Mode::Enforce, true, false), Inv2Decision::Deny);
    }

    #[test]
    fn cleartext_over_untrusted_carrier_warns_when_not_wired() {
        // Interim behavior while the HyKEM tunnel is unwired: warn, don't brick.
        assert_eq!(decide(Inv2Mode::WarnOnly, true, false), Inv2Decision::Warn);
    }

    /// Captures the INV-2 END STATE and its prerequisite. INV-2 requires the
    /// default to be fail-closed [`Inv2Mode::Enforce`]. It currently is NOT,
    /// because the HyKEM tunnel (#551) is not yet wired mandatory on iroh/relay
    /// send paths ([`HYKEM_IROH_TUNNEL_WIRED`] == false) — hard-enforcing now
    /// would fail-close all iroh RPC. This test is `#[ignore]`d until that
    /// prerequisite lands; flipping `HYKEM_IROH_TUNNEL_WIRED` to true (together
    /// with wiring the tunnel) makes it pass and closes the hole by default.
    #[test]
    #[ignore = "BLOCKED on HyKEM tunnel (#551) being wired mandatory on iroh/relay \
                send paths; flip HYKEM_IROH_TUNNEL_WIRED once it is — see ADR #1023 INV-2"]
    fn inv2_default_is_fail_closed_once_tunnel_wired() {
        assert!(
            HYKEM_IROH_TUNNEL_WIRED,
            "HyKEM iroh/relay tunnel must be wired before INV-2 can default to enforce"
        );
        assert_eq!(default_inv2_mode(), Inv2Mode::Enforce);
    }
}
