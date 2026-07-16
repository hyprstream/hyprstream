//! Receive-side carrier classification (INV-2, #1042; ADR #1023).
//!
//! A [`CarrierContext`] states, for one accepted connection/request, which
//! class of carrier the bytes arrived on. It is constructed **only at the
//! server's real accept boundary** (the listener/session code in this crate)
//! and threaded through [`super::rpc_session::serve_rpc_connection`] and
//! [`crate::service::dispatch::process_request`] to the envelope-policy
//! chokepoints. It is never derived from request bytes, headers, JWT claims,
//! remote addresses, or caller assertions — whoever constructs a request must
//! not be able to construct its trust class.
//!
//! The context answers exactly one policy question on the receive path: may a
//! cleartext (`encrypted_envelope == None`) request envelope be processed?
//! Untrusted carriers (iroh direct/relay, QUIC including loopback,
//! WebTransport, unknown) must reject cleartext before claims evaluation or
//! handler dispatch; only explicitly trusted same-host carriers (inproc, UDS,
//! systemd-activated UDS) retain their documented cleartext behavior.
//!
//! This mirrors the send side's [`Transport::forbids_cleartext_envelope`]
//! (#1036) so the two halves of INV-2 (#1028) classify identically. The
//! carrier context is **not** an identity: remote NodeId, TLS identity,
//! relay/direct state, and signer equality remain diagnostics — live identity
//! admission is #1027's separate evidence.
//!
//! [`Transport::forbids_cleartext_envelope`]: crate::transport_traits::Transport::forbids_cleartext_envelope

/// Carrier class of an accepted RPC connection, as attested by the accept
/// boundary that constructed it.
///
/// Trusted-local constructors are `pub(crate)`: only this crate's transport
/// accept boundaries (in-memory inproc, UDS/systemd listeners) can mint a
/// trusted context. Untrusted constructors are freely public — classifying a
/// carrier as untrusted is always safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CarrierContext {
    class: CarrierClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CarrierClass {
    /// Same-process in-memory channel: bytes never leave the address space.
    Inproc,
    /// Same-host Unix domain socket accepted by this crate's UDS listener
    /// (including systemd socket activation) — peer-credential authenticated.
    TrustedUds,
    /// iroh connection, direct or relay-carried.
    Iroh,
    /// QUIC session — cross-host or loopback. A loopback address is not
    /// evidence the bytes stay inside the trust boundary (it can terminate at
    /// a proxy/tunnel), so it earns no exemption.
    Quic,
    /// WebTransport session (browser or native `web_transport_quinn`).
    WebTransport,
    /// Unknown or ambiguous origin. Fail-closed: treated as untrusted.
    Unknown,
}

impl CarrierContext {
    /// Same-process in-memory carrier (trusted local). Accept-boundary only.
    pub(crate) fn inproc() -> Self {
        Self {
            class: CarrierClass::Inproc,
        }
    }

    /// Same-host UDS carrier accepted by this crate's listener, including
    /// systemd socket activation (trusted local). Accept-boundary only.
    pub(crate) fn trusted_uds() -> Self {
        Self {
            class: CarrierClass::TrustedUds,
        }
    }

    /// An iroh carrier (direct or relay-carried). Untrusted.
    pub fn iroh() -> Self {
        Self {
            class: CarrierClass::Iroh,
        }
    }

    /// A QUIC carrier — cross-host **or loopback**. Untrusted.
    pub fn quic() -> Self {
        Self {
            class: CarrierClass::Quic,
        }
    }

    /// A WebTransport carrier. Untrusted.
    pub fn web_transport() -> Self {
        Self {
            class: CarrierClass::WebTransport,
        }
    }

    /// Unknown/ambiguous carrier. Untrusted (the fail-closed default any new
    /// accept path should start from until it is deliberately classified).
    pub fn untrusted_unknown() -> Self {
        Self {
            class: CarrierClass::Unknown,
        }
    }

    /// Whether this carrier forbids cleartext request envelopes (INV-2).
    ///
    /// `true` for every network/unknown carrier; `false` only for the
    /// explicitly trusted same-host classes.
    pub fn forbids_cleartext_envelope(&self) -> bool {
        match self.class {
            CarrierClass::Inproc | CarrierClass::TrustedUds => false,
            CarrierClass::Iroh
            | CarrierClass::Quic
            | CarrierClass::WebTransport
            | CarrierClass::Unknown => true,
        }
    }

    /// Short static label for logs/errors. Never used for policy.
    pub fn label(&self) -> &'static str {
        match self.class {
            CarrierClass::Inproc => "inproc",
            CarrierClass::TrustedUds => "uds",
            CarrierClass::Iroh => "iroh",
            CarrierClass::Quic => "quic",
            CarrierClass::WebTransport => "webtransport",
            CarrierClass::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for CarrierContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-2 receive-side classification matrix (#1042): every network or
    /// unknown carrier forbids cleartext; only the explicit same-host classes
    /// permit it.
    #[test]
    fn inv2_receive_carrier_matrix() {
        assert!(CarrierContext::iroh().forbids_cleartext_envelope());
        assert!(CarrierContext::quic().forbids_cleartext_envelope(),);
        assert!(CarrierContext::web_transport().forbids_cleartext_envelope());
        assert!(
            CarrierContext::untrusted_unknown().forbids_cleartext_envelope(),
            "missing/ambiguous context must follow the untrusted branch"
        );
        assert!(!CarrierContext::inproc().forbids_cleartext_envelope());
        assert!(!CarrierContext::trusted_uds().forbids_cleartext_envelope());
    }
}
