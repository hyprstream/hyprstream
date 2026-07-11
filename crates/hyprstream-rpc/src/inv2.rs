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
//! # Current status — fail-closed by default
//!
//! The default mode is [`Inv2Mode::Enforce`]. Cleartext envelopes over iroh,
//! relay, cross-host QUIC, and wasm WebTransport are refused before any byte
//! reaches the transport unless a compliant encrypted envelope is present.
//! Operators and tests may explicitly select the temporary warn/dev path via
//! `HYPRSTREAM_INV2=warn` or [`set_inv2_mode`].

use crate::error::{Result, RpcError};

/// Default enforcement gate for INV-2. `true` means the process fails closed on
/// cleartext envelopes over untrusted carriers unless an explicit override
/// selects [`Inv2Mode::WarnOnly`].
///
/// Keep the name for the existing call sites and tests that treat this as the
/// operator-confirmed "fail closed by default" bit for the HyKEM/iroh path.
pub const HYKEM_IROH_TUNNEL_WIRED: bool = true;

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
/// otherwise [`Inv2Mode::WarnOnly`] for explicitly carried development builds.
pub const fn default_inv2_mode() -> Inv2Mode {
    if HYKEM_IROH_TUNNEL_WIRED {
        Inv2Mode::Enforce
    } else {
        Inv2Mode::WarnOnly
    }
}

#[cfg(not(target_arch = "wasm32"))]
static INV2_MODE: std::sync::OnceLock<Inv2Mode> = std::sync::OnceLock::new();

/// Name of the operator escape hatch. `enforce` forces fail-closed INV-2;
/// `warn` forces the temporary warn-only/dev path.
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
/// default. The browser path is exactly the untrusted-carrier case INV-2
/// targets, so it also defaults to fail-closed.
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
    match decide(
        inv2_mode(),
        carrier_forbids_cleartext,
        envelope_is_encrypted,
    ) {
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
                         permitted only because HYPRSTREAM_INV2=warn selected the temporary \
                         warn-only/dev path; unset it or set HYPRSTREAM_INV2=enforce to fail closed."
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
        assert_eq!(
            decide(Inv2Mode::WarnOnly, false, false),
            Inv2Decision::Allow
        );
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

    /// Captures the INV-2 end state: the default is fail-closed
    /// [`Inv2Mode::Enforce`].
    #[test]
    fn inv2_default_is_fail_closed() {
        assert!(
            HYKEM_IROH_TUNNEL_WIRED,
            "INV-2 must default to fail-closed for untrusted carriers"
        );
        assert_eq!(default_inv2_mode(), Inv2Mode::Enforce);
    }
}
