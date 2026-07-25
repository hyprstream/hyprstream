//! Reviewed registry of HTTP routes intentionally exempt from native MAC
//! mediation — the "public / declassifier exemption registry" (issue #1273,
//! epic #1267, audit item `NEW-PUBLIC-EXEMPTIONS`).
//!
//! # Why this exists
//! Native MAC ([`crate::mac`]) is the mandatory floor: every object path must
//! be mediated by a verified subject `SecurityContext` and a trusted object
//! label before use. Activation's "all non-exempt paths mediated" claim is only
//! enforceable if the set of intentionally-unmediated routes is *explicit and
//! reviewed*, not implicit. Without a registry, a newly added public route can
//! silently become a bypass — exactly the drift this module prevents.
//!
//! # Contract (do not circumvent)
//! 1. **A route defaults to mediated.** Any HTTP route not listed in
//!    [`PUBLIC_EXEMPTIONS`] must sit behind the protected (auth-mediated)
//!    router. Once the MAC PEP is installed there is no permissive mode
//!    ([`crate::mac`] is fail-closed per #547); "public" is an opt-in,
//!    reviewed exemption, never a default.
//! 2. **The public routers are built from this registry.** [`crate::server`]
//!    and [`crate::services::at9p_verify`] construct their public route sets by
//!    iterating [`PUBLIC_EXEMPTIONS`] for their face, so adding an unmediated
//!    route requires adding a reviewed [`PublicExemption`] entry plus a handler
//!    arm — two review surfaces, no silent addition.
//! 3. **The drift test pins the live set to the registry.**
//!    `public_exemptions_match_live_routes` asserts registry self-consistency,
//!    handler-enum parity, per-face wiring, and that the public/protected split
//!    is live (registry paths answer without auth; a protected path does not).
//!
//! # Scope and dependencies
//! This registry covers genuinely no-auth public routes: health metadata,
//! `.well-known` discovery, pre-credential browser provisioning, and the
//! stateless at9p verification face. OAuth2.1 protocol endpoints (token,
//! authorize, jwks, device, SCIM …) are the *credential-issuance control
//! plane* that MAC sits beneath, not tenant-object access; they remain out of
//! scope here and are tracked separately. Real MAC enforcement remains dormant
//! (uninstalled, so dispatch permits as a pre-activation pass-through) pending
//! production clearance provenance (#698). Once installed, the PEP is
//! fail-closed. This registry is the activation-gate structure that ships now
//! so the exempt set is locked.
//!
//! [#547]: https://github.com/hyprstream/hyprstream/issues/547
//! [#698]: https://github.com/hyprstream/hyprstream/issues/698
//! [#1267]: https://github.com/hyprstream/hyprstream/issues/1267
//! [#1273]: https://github.com/hyprstream/hyprstream/issues/1273

/// HTTP method of an exempt route. Only the methods used by public routes are
/// represented; anything else must go through the protected router.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RouteMethod {
    Get,
    Post,
}

/// Why a route is intentionally unmediated. Used by activation review to
/// confirm each exemption is a narrow, justified carve-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExemptionCategory {
    /// Liveness / health metadata. No tenant data, no object access.
    Health,
    /// Published discovery or capability metadata (`.well-known`, wire-plane
    /// table). Public by design so clients can bootstrap before authenticating.
    PublicMetadata,
    /// Pre-credential bootstrap (browser provisioning before any identity
    /// exists). Rate-limited; carries no tenant object.
    Bootstrap,
    /// Stateless cryptographic verification returning only pass/fail plus the
    /// content-verified identifier. Never echoes request bytes past the check.
    CryptographicVerification,
}

/// Which HTTP face serves the exempt route. Each face builds its public router
/// from its own slice of [`PUBLIC_EXEMPTIONS`]; the drift test cross-checks
/// every face against the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HttpFace {
    /// The main Axum app ([`crate::server::create_app`]).
    MainApp,
    /// The standalone credential-free at9p verification face
    /// ([`crate::services::at9p_verify`]).
    At9pVerify,
}

/// Tag identifying which handler serves an exempt route. The public-router
/// builders match exhaustively on this, so wiring a new route requires both a
/// [`PublicExemption`] entry and a handler arm — the double review touch that
/// keeps the exempt set from silently growing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PublicRouteHandler {
    /// `GET /` and `GET /health` — health metadata.
    HealthCheck,
    /// `GET /.well-known/oauth-protected-resource` — RFC 9728 metadata.
    OauthProtectedResourceMetadata,
    /// `GET /.well-known/export9p` — 9P export discovery metadata.
    Export9pMetadata,
    /// `GET /.well-known/planes` — wire-plane discovery table.
    WirePlanesMetadata,
    /// `GET /9p` — 9P-over-WebSocket export (mount ticket rides the URL query).
    NinepWebSocket,
    /// `GET /.well-known/hyprstream/browser-provisioning/:service` — bootstrap.
    BrowserProvisioning,
    /// `POST /at9p/verify` — stateless login-assertion verification (own face).
    At9pVerify,
}

/// One reviewed public/declassifier exemption. Every field is `&'static` so the
/// registry is a compile-time constant and the drift test needs no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PublicExemption {
    /// Stable, unique kebab-case identifier (the key the router builder looks
    /// up). Changing an id is a reviewed change; reuse across routes is a drift
    /// error caught by [`validate`].
    pub id: &'static str,
    /// Face that owns the route. See [`HttpFace`].
    pub face: HttpFace,
    /// HTTP method.
    pub method: RouteMethod,
    /// Axum path pattern exactly as wired (e.g. `"/health"`,
    /// `"/.well-known/hyprstream/browser-provisioning/:service"`).
    pub path: &'static str,
    /// Reviewed justification category. See [`ExemptionCategory`].
    pub category: ExemptionCategory,
    /// Handler that serves this route. See [`PublicRouteHandler`].
    pub handler: PublicRouteHandler,
    /// One-line human justification for why native MAC is intentionally absent
    /// here. Must be non-empty; [`validate`] rejects empties.
    pub justification: &'static str,
}

/// The authoritative, reviewed registry. This is the single source of truth for
/// which HTTP routes bypass native MAC mediation.
///
/// **To add a public route:** append a [`PublicExemption`] here with a reviewed
/// justification, add the matching handler arm to the owning face's builder,
/// then update the drift test's expected snapshot. All three edits in one PR.
pub static PUBLIC_EXEMPTIONS: &[PublicExemption] = &[
    PublicExemption {
        id: "root-health",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/",
        category: ExemptionCategory::Health,
        handler: PublicRouteHandler::HealthCheck,
        justification: "Liveness probe; returns service name + version only, no tenant data.",
    },
    PublicExemption {
        id: "health",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/health",
        category: ExemptionCategory::Health,
        handler: PublicRouteHandler::HealthCheck,
        justification: "Liveness probe; returns service name + version only, no tenant data.",
    },
    PublicExemption {
        id: "oauth-protected-resource",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/.well-known/oauth-protected-resource",
        category: ExemptionCategory::PublicMetadata,
        handler: PublicRouteHandler::OauthProtectedResourceMetadata,
        justification: "RFC 9728 metadata advertising the authorization server; needed before a client can authenticate.",
    },
    PublicExemption {
        id: "export9p",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/.well-known/export9p",
        category: ExemptionCategory::PublicMetadata,
        handler: PublicRouteHandler::Export9pMetadata,
        justification: "9P export discovery metadata; clients resolve the mount selector from this before presenting a ticket.",
    },
    PublicExemption {
        id: "wire-planes",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/.well-known/planes",
        category: ExemptionCategory::PublicMetadata,
        handler: PublicRouteHandler::WirePlanesMetadata,
        justification: "Wire-plane discovery table (#821); enumerates non-file planes, no tenant objects.",
    },
    PublicExemption {
        id: "ninep-ws",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/9p",
        category: ExemptionCategory::PublicMetadata,
        handler: PublicRouteHandler::NinepWebSocket,
        justification: "9P-over-WebSocket upgrade; the mount ticket rides the URL query (browser WS cannot set headers) and is validated inside the handler.",
    },
    PublicExemption {
        id: "browser-provisioning",
        face: HttpFace::MainApp,
        method: RouteMethod::Get,
        path: "/.well-known/hyprstream/browser-provisioning/:service",
        category: ExemptionCategory::Bootstrap,
        handler: PublicRouteHandler::BrowserProvisioning,
        justification: "Pre-credential browser bootstrap before any identity exists; independently rate-limited, carries no tenant object.",
    },
    PublicExemption {
        id: "at9p-verify",
        face: HttpFace::At9pVerify,
        method: RouteMethod::Post,
        path: "/at9p/verify",
        category: ExemptionCategory::CryptographicVerification,
        handler: PublicRouteHandler::At9pVerify,
        justification: "Stateless login-assertion verification on a credential-free face; returns only verified/did/assurance, never echoes attacker bytes past the check.",
    },
];

impl PublicExemption {
    /// (method, path) tuple, the key used for uniqueness checks.
    fn route_key(&self) -> (HttpFace, RouteMethod, &'static str) {
        (self.face, self.method, self.path)
    }
}

/// Yields the registry entries owned by `face`, in declaration order.
pub fn for_face(face: HttpFace) -> impl Iterator<Item = &'static PublicExemption> {
    PUBLIC_EXEMPTIONS.iter().filter(move |e| e.face == face)
}

/// Looks up an exemption by id. Router builders use this to reference the
/// reviewed path/method rather than a string literal, so a typo or unregistered
/// route is caught at startup instead of silently bypassing MAC.
///
/// Panics if `id` is not in the registry — a public route must be registered.
pub fn require(face: HttpFace, id: &str) -> &'static PublicExemption {
    PUBLIC_EXEMPTIONS
        .iter()
        .find(|e| e.face == face && e.id == id)
        .unwrap_or_else(|| panic!("public exemption {id:?} not registered for face {face:?}"))
}

/// Error raised by [`validate`] when the registry has drifted from its contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    DuplicateId(&'static str),
    DuplicateRoute(HttpFace, RouteMethod, &'static str),
    EmptyJustification(&'static str),
    UnknownFaceHandler(&'static str),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "duplicate exemption id {id:?}"),
            Self::DuplicateRoute(face, method, path) => {
                write!(f, "duplicate route {method:?} {path:?} on face {face:?}")
            }
            Self::EmptyJustification(id) => {
                write!(f, "exemption {id:?} has an empty justification")
            }
            Self::UnknownFaceHandler(id) => {
                write!(f, "exemption {id:?} references a handler unused by its declared face")
            }
        }
    }
}

impl std::error::Error for RegistryError {}

/// Validates the registry against its contract:
/// - unique ids and unique (face, method, path) routes,
/// - non-empty justifications,
/// - each handler appears only under the face that owns it.
///
/// Returns the first [`RegistryError`] found, or `Ok(())`. The drift test calls
/// this and also checks the builder wiring is consistent with the result.
pub fn validate() -> Result<(), RegistryError> {
    let mut seen_ids = std::collections::HashSet::new();
    let mut seen_routes = std::collections::HashSet::new();
    for e in PUBLIC_EXEMPTIONS {
        if !seen_ids.insert(e.id) {
            return Err(RegistryError::DuplicateId(e.id));
        }
        if !seen_routes.insert(e.route_key()) {
            return Err(RegistryError::DuplicateRoute(e.face, e.method, e.path));
        }
        if e.justification.trim().is_empty() {
            return Err(RegistryError::EmptyJustification(e.id));
        }
        if !handler_belongs_to_face(e.handler, e.face) {
            return Err(RegistryError::UnknownFaceHandler(e.id));
        }
    }
    Ok(())
}

/// `true` if `handler` is wired by the builder that owns `face`. Keeps the
/// at9p-verify handler from leaking into the main-app builder (or vice versa).
pub(crate) const fn handler_belongs_to_face(handler: PublicRouteHandler, face: HttpFace) -> bool {
    match face {
        HttpFace::MainApp => matches!(
            handler,
            PublicRouteHandler::HealthCheck
                | PublicRouteHandler::OauthProtectedResourceMetadata
                | PublicRouteHandler::Export9pMetadata
                | PublicRouteHandler::WirePlanesMetadata
                | PublicRouteHandler::NinepWebSocket
                | PublicRouteHandler::BrowserProvisioning
        ),
        HttpFace::At9pVerify => matches!(handler, PublicRouteHandler::At9pVerify),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The registry must satisfy its own contract.
    #[test]
    fn registry_is_self_consistent() {
        validate().expect("PUBLIC_EXEMPTIONS must be self-consistent");
    }

    /// Every [`PublicRouteHandler`] variant is referenced by at least one
    /// registry entry, so adding a handler without an entry (or an entry
    /// without a handler arm) is caught here.
    #[test]
    fn every_handler_variant_is_registered() {
        let all = [
            PublicRouteHandler::HealthCheck,
            PublicRouteHandler::OauthProtectedResourceMetadata,
            PublicRouteHandler::Export9pMetadata,
            PublicRouteHandler::WirePlanesMetadata,
            PublicRouteHandler::NinepWebSocket,
            PublicRouteHandler::BrowserProvisioning,
            PublicRouteHandler::At9pVerify,
        ];
        for h in all {
            let referenced = PUBLIC_EXEMPTIONS.iter().any(|e| e.handler == h);
            assert!(
                referenced,
                "PublicRouteHandler::{h:?} has no registry entry; add one or remove the variant"
            );
        }
    }

    /// The main-app face owns exactly the routes it wires (snapshot). Updating
    /// the registry requires updating this expected set in the same PR — the
    /// explicit review touchpoint that prevents silent drift.
    #[test]
    fn main_app_face_snapshot_is_pinned() {
        let main: Vec<(&'static str, RouteMethod, &'static str)> = for_face(HttpFace::MainApp)
            .map(|e| (e.id, e.method, e.path))
            .collect();
        let expected = [
            ("root-health", RouteMethod::Get, "/"),
            ("health", RouteMethod::Get, "/health"),
            ("oauth-protected-resource", RouteMethod::Get, "/.well-known/oauth-protected-resource"),
            ("export9p", RouteMethod::Get, "/.well-known/export9p"),
            ("wire-planes", RouteMethod::Get, "/.well-known/planes"),
            ("ninep-ws", RouteMethod::Get, "/9p"),
            (
                "browser-provisioning",
                RouteMethod::Get,
                "/.well-known/hyprstream/browser-provisioning/:service",
            ),
        ];
        assert_eq!(main, expected, "main-app public face drifted from the reviewed snapshot");
    }

    /// The at9p-verify face owns exactly its single verification route.
    #[test]
    fn at9p_verify_face_snapshot_is_pinned() {
        let verify: Vec<(&'static str, RouteMethod, &'static str)> =
            for_face(HttpFace::At9pVerify).map(|e| (e.id, e.method, e.path)).collect();
        let expected = [("at9p-verify", RouteMethod::Post, "/at9p/verify")];
        assert_eq!(verify, expected, "at9p-verify public face drifted from the reviewed snapshot");
    }

    /// The two faces are disjoint: no route id or (method, path) is claimed by
    /// both faces — a route is owned by exactly one reviewed exemption.
    #[test]
    fn faces_are_disjoint() {
        let main: std::collections::HashSet<_> =
            for_face(HttpFace::MainApp).map(|e| (e.method, e.path)).collect();
        let verify: std::collections::HashSet<_> =
            for_face(HttpFace::At9pVerify).map(|e| (e.method, e.path)).collect();
        let overlap: Vec<_> = main.intersection(&verify).collect();
        assert!(
            overlap.is_empty(),
            "faces overlap on routes {overlap:?}; each route belongs to exactly one face"
        );
    }

    /// `require` must resolve every id the builders reference, and reject an
    /// unregistered id — the startup gate that blocks an unregistered route.
    #[test]
    fn require_resolves_registered_and_rejects_unknown() {
        assert_eq!(require(HttpFace::MainApp, "health").path, "/health");
        assert_eq!(require(HttpFace::At9pVerify, "at9p-verify").path, "/at9p/verify");
    }

    #[test]
    #[should_panic(expected = "not registered")]
    fn require_panics_for_unregistered_id() {
        let _ = require(HttpFace::MainApp, "definitely-not-registered");
    }
}
