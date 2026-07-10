//! DID URL parser + inert dereference plan (#906, at9p epic #880 **G1**).
//!
//! This module only parses attacker-reproducible selectors and produces a
//! [`DereferencePlan`]. It deliberately does not resolve DID state, decode a
//! carrier address, construct a dial target, or prove possession. Runtime
//! dereferencing must keep these boundaries distinct:
//!
//! 1. verify the immutable genesis GATE;
//! 2. obtain separately accepted current DID state (future #1027 work);
//! 3. select the requested service from that accepted state;
//! 4. decode the service into typed reach;
//! 5. establish live possession before granting authority;
//! 6. attach and walk using this plan's inert selectors.
//!
//! In particular, a `did:key` verification key is not an iroh `NodeId`, an
//! endpoint, or evidence that its holder possesses any carrier.
//!
//! # Grammar (ratified 2026-07-10, fable spike — #906 comment)
//!
//! Recognized query params: **`service | relativeRef | pin`**.
//! - `pin` = a CIDv1 string (base32 `bafy…`, validated via [`crate::cid`]), a
//!   **client-verified** content commitment applied at 9P **READ**. It asserts
//!   **no authority** — a DID URL is an inert, attacker-reproducible selector
//!   (#905 §7/§8); `pin` only lets the reader reject wrong/inconsistent bytes.
//! - `versionId` / `versionTime` → W3C-registered **DID-document-version**
//!   resolution options. A conformant generic dereferencer MUST pass them to the
//!   *resolve* step, which fails (`FEATURE_NOT_SUPPORTED` until capsule-epoch /
//!   KERI resolution exists). We **fail closed at parse time** so a travelling
//!   URL can never be silently mis-resolved, and they are **never** interpreted
//!   as a content pin (that would poison the resolve → `#ns` → `alsoKnownAs`
//!   interop prefix these URLs rely on when carried in `NameRecord.subject`).
//! - `hl` is dropped from the grammar (document-level + doubly redundant for a
//!   self-certifying DID); like any unrecognized param it is silently ignored.
//! - `?service=ns` selects the `#ns` serviceEndpoint by fragment id →
//!   `Tattach(aname="ns")`. Default service is [`DEFAULT_SERVICE`] (`"ns"`).
//! - **path is canonical** over `relativeRef`: when both carry a walk, the URL
//!   path wins and `relativeRef` is normalized away ([`DidUrl::walk_segments`]).
//!
//! # DID URL form
//!
//! Per W3C DID Core + RFC 3986 opaque-path (**no authority** — parses like
//! `mailto:` everywhere, including WHATWG/browser): the DID is the scheme
//! prefix, the path hangs off it natively.
//!
//! ```text
//! did:<method>:<method-specific-id>[/<path>][?<query>][#<fragment>]
//! ```
//! The DID ends at the first `/`, `?`, or `#`. The fragment is a DID-document
//! *internal* reference and is **stripped** — service selection is via
//! `?service=`, never the URL fragment. No `9p://` or `moq://` scheme is
//! accepted (#906 acceptance): only `did:`-prefixed URLs parse.

use anyhow::{anyhow, bail, ensure, Context, Result};

use crate::cid;

/// Default service fragment id (`#ns`) — the `NinePExport` entry a DID URL
/// attaches to when `?service=` is absent (#877 aname default).
pub const DEFAULT_SERVICE: &str = "ns";

// ─── percent-decoding (DID-URL query values) ─────────────────────────────────

/// Percent-decode a DID-URL query value or path segment (RFC 3986 `%XX`).
///
/// Invalid escape sequences are left verbatim, while decoded bytes that are not
/// valid UTF-8 fail closed. Distinct byte strings must never collapse onto the
/// same lossy selector.
pub(crate) fn percent_decode(s: &str) -> Result<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(hi), Some(lo)) = (hi, lo) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).context("percent-decoded DID URL component is not valid UTF-8")
}

/// Strip a single leading `#` from a service-id value (tolerant: `?service=ns`
/// and `?service=%23ns` both select `#ns`).
fn strip_leading_hash(s: &str) -> &str {
    s.strip_prefix('#').unwrap_or(s)
}

// ─── canonical DID-URL type ───────────────────────────────────────────────────

/// A parsed DID URL — the canonical, inert addressing type (#906 G1).
///
/// Carries **zero authority**: a `DidUrl` is a selector. Everything authoritative
/// (identity, assurance, content) is created at the GATE (capsule content) and
/// the admission boundary (subject assurance) during dereferencing — never in
/// the URL itself (#905 §8). The same string reproduced by an attacker grants
/// them nothing; separately accepted state, typed reach, and live possession
/// are still required before any runtime operation can grant authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DidUrl {
    /// The base DID (`did:<method>:<id>`), no path/query/fragment.
    did: String,
    /// Percent-decoded URL path segments (9P walk **selectors**), carried
    /// verbatim. The parser asserts nothing about assurance for these — they are
    /// MAC-checked as selectors by the 9P walk layer, not trusted here.
    path: Vec<String>,
    /// `?service=<id>` — the serviceEndpoint fragment id (without `#`) → the
    /// `aname` for `Tattach`. `None` means "use [`DEFAULT_SERVICE`]".
    service: Option<String>,
    /// `?relativeRef=<path>`. **Path is canonical**: when [`Self::path`] is
    /// non-empty this is redundant and ignored ([`Self::walk_segments`]
    /// normalizes the redundancy away rather than letting both carry the walk).
    relative_ref: Option<Vec<String>>,
    /// `?pin=<cid>` — a validated CIDv1 string (base32), client-verified at READ.
    pin: Option<String>,
}

impl DidUrl {
    /// Parse a DID URL string per the ratified #906 grammar.
    ///
    /// Rejects non-`did:` schemes (`9p://`, `moq://`, `https://`, …). See the
    /// [module docs](self) for the full param-grammar contract; in particular
    /// `versionId` / `versionTime` **fail closed** and `hl` / unknown params are
    /// ignored.
    pub fn parse(url: &str) -> Result<Self> {
        // Must be a DID. Explicitly reject 9p://, moq://, https://, etc.
        // (#906 acceptance: "no 9p:// / moq:// scheme accepted".)
        let rest = url
            .strip_prefix("did:")
            .ok_or_else(|| anyhow!("not a DID URL (must start with 'did:'): {url}"))?;

        // The DID (method + method-specific-id) ends at the first path/query/
        // fragment delimiter (RFC 3986 opaque-path; DID Core §3.2).
        let did_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let method_and_id = &rest[..did_end];
        // A DID needs at least "method:id" (two colon-separated parts after the
        // "did:" scheme). An empty method-specific id is malformed.
        let (method, msi) = method_and_id
            .split_once(':')
            .ok_or_else(|| anyhow!("malformed DID (missing method-specific id): {url}"))?;
        ensure!(
            !method.is_empty() && !msi.is_empty(),
            "malformed DID (empty method or method-specific id): {url}"
        );
        let did = format!("did:{method_and_id}");

        // Tail after the DID: path, ?query, #fragment (in RFC 3986 order).
        let tail = &rest[did_end..];
        // Strip the fragment first — it is a DID-document *internal* reference
        // and never influences dereferencing (service selection is via ?service=).
        let (before_frag, _fragment) = tail.split_once('#').unwrap_or((tail, ""));
        let (path_str, query) = before_frag.split_once('?').unwrap_or((before_frag, ""));

        // Path segments: percent-decoded selectors, carried verbatim. Empty
        // segments (consecutive slashes, leading slash) are dropped — they
        // contribute nothing to a 9P walk. The walk layer is the authority on
        // traversal; the parser does not reject `.`/`..` here.
        let path: Vec<String> = path_str
            .split('/')
            .filter(|s| !s.is_empty())
            .map(percent_decode)
            .collect::<Result<_>>()?;

        let mut service = None;
        let mut relative_ref = None;
        let mut pin = None;
        for pair in query.split('&').filter(|s| !s.is_empty()) {
            let (raw_key, raw_val) = pair.split_once('=').unwrap_or((pair, ""));
            let key = percent_decode(raw_key)?;
            let val = percent_decode(raw_val)?;
            match key.as_str() {
                "service" => {
                    let service_id = strip_leading_hash(&val);
                    ensure!(
                        !service_id.is_empty(),
                        "?service= must contain a fragment id: {url}"
                    );
                    service = Some(service_id.to_owned());
                }
                "relativeRef" => {
                    ensure!(
                        !raw_val.is_empty(),
                        "?relativeRef= must be non-empty: {url}"
                    );
                    // Split before decoding, exactly like the URL path. Decoding
                    // the whole value first would turn `%2F` into a structural
                    // separator here while leaving it inside one path segment.
                    let segments: Vec<String> = raw_val
                        .split('/')
                        .filter(|segment| !segment.is_empty())
                        .map(percent_decode)
                        .collect::<Result<_>>()?;
                    ensure!(
                        !segments.is_empty(),
                        "?relativeRef= must contain a path segment: {url}"
                    );
                    relative_ref = Some(segments);
                }
                "pin" => {
                    ensure!(!val.is_empty(), "?pin= must be non-empty: {url}");
                    // Validate the CID shape now so a malformed pin fails closed
                    // at parse time. The pin asserts NO authority; this only
                    // rejects structurally-invalid content commitments.
                    cid::decode_cid(&val)
                        .with_context(|| format!("?pin= is not a valid CIDv1 string: {val}"))?;
                    pin = Some(val);
                }
                "versionId" | "versionTime" => {
                    // W3C-registered DID-DOCUMENT-version selectors. A conformant
                    // generic dereferencer MUST pass these to the RESOLVE step,
                    // which fails (FEATURE_NOT_SUPPORTED) until capsule-epoch /
                    // KERI resolution exists. NEVER a content pin — interpreting
                    // them as one would poison the resolve → #ns → alsoKnownAs
                    // interop prefix. Fail closed at parse time. (#906 decision.)
                    bail!(
                        "DID URL parameter `{key}` selects a DID-document version, which is not \
                         supported; use `?pin=<cid>` for a client-verified content commitment: {url}"
                    );
                }
                // `hl` is dropped from the grammar (document-level + redundant
                // for a self-certifying DID). Any other unrecognized key is a
                // method-defined / generic param and is silently ignored (DID
                // Core: unknown params must not error a generic dereferencer).
                _ => {}
            }
        }

        Ok(Self {
            did,
            path,
            service,
            relative_ref,
            pin,
        })
    }

    /// The base DID (`did:<method>:<id>`), no path/query/fragment.
    pub fn did(&self) -> &str {
        &self.did
    }

    /// Percent-decoded URL path segments (9P walk selectors), verbatim.
    pub fn path(&self) -> &[String] {
        &self.path
    }

    /// `?service=<id>` if present (without `#`). `None` means "default".
    pub fn service(&self) -> Option<&str> {
        self.service.as_deref()
    }

    /// `?relativeRef=<path>` if present. Path is canonical; see
    /// [`Self::walk_segments`] for the normalized walk.
    pub fn relative_ref(&self) -> Option<&[String]> {
        self.relative_ref.as_deref()
    }

    /// `?pin=<cid>` — a validated CIDv1 string applied (client-verified) at READ.
    pub fn pin(&self) -> Option<&str> {
        self.pin.as_deref()
    }

    /// The selected service fragment id (without `#`), defaulting to
    /// [`DEFAULT_SERVICE`] (`"ns"`). This is the `aname` for `Tattach`.
    pub fn aname(&self) -> &str {
        self.service.as_deref().unwrap_or(DEFAULT_SERVICE)
    }

    /// The effective 9P walk segments. **Path is canonical**: when the URL path
    /// is non-empty it wins and `relativeRef` is ignored (normalized away). When
    /// only `relativeRef` is present, it supplies the walk. Empty when neither
    /// carries a walk (attach the export root).
    ///
    /// This is the single source of truth for "what to `Twalk`" — callers must
    /// not consult [`Self::path`] and [`Self::relative_ref`] independently.
    pub fn walk_segments(&self) -> Vec<&str> {
        if !self.path.is_empty() {
            self.path.iter().map(String::as_str).collect()
        } else if let Some(rel) = &self.relative_ref {
            rel.iter().map(String::as_str).collect()
        } else {
            Vec::new()
        }
    }

    /// Build the dereference plan (aname + walk + pin) from the parsed URL.
    ///
    /// Pure (no resolution). Before using these selectors, a runtime must verify
    /// the genesis GATE, obtain accepted current state, select/decode typed
    /// reach, and establish live possession. This module implements none of
    /// those authority-bearing stages and does not invent #1027 admission.
    pub fn plan(&self) -> DereferencePlan {
        DereferencePlan {
            did: self.did.clone(),
            aname: self.aname().to_owned(),
            walk: self
                .walk_segments()
                .into_iter()
                .map(str::to_owned)
                .collect(),
            pin: self.pin.clone(),
        }
    }
}

// ─── dereference plan + seam ─────────────────────────────────────────────────

/// Inert service/attach/walk inputs for a parsed [`DidUrl`] (#906 G1, #905 §4).
///
/// Produced purely from the URL by [`DidUrl::plan`]. It contains no accepted
/// DID state, transport, identity key, authority subject, or possession proof.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DereferencePlan {
    /// The base DID selector; not proof of accepted state or authority.
    pub did: String,
    /// Requested service/aname selector; not a decoded or admitted endpoint.
    pub aname: String,
    /// Path segments for `Twalk` (selectors; MAC-checked by the 9P layer).
    pub walk: Vec<String>,
    /// Optional content pin — client-verified at READ, asserts **no** authority.
    pub pin: Option<String>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // ─── parser: core shape ───────────────────────────────────────────────

    #[test]
    fn parses_did_with_path_and_service() {
        let u = DidUrl::parse("did:at9p:bafyfake/a/b?service=ns").unwrap();
        assert_eq!(u.did(), "did:at9p:bafyfake");
        assert_eq!(u.path(), &["a".to_owned(), "b".to_owned()]);
        assert_eq!(u.service(), Some("ns"));
        assert_eq!(u.aname(), "ns");
        assert_eq!(u.walk_segments(), vec!["a", "b"]);
        assert!(u.pin().is_none());
    }

    #[test]
    fn default_service_is_ns() {
        let u = DidUrl::parse("did:at9p:bafyfake/x").unwrap();
        assert_eq!(u.service(), None);
        assert_eq!(u.aname(), DEFAULT_SERVICE);
        assert_eq!(u.aname(), "ns");
    }

    #[test]
    fn fragment_is_stripped() {
        // A URL fragment is a DID-doc internal reference; it must not influence
        // selection (service selection is via ?service=).
        let u = DidUrl::parse("did:web:example.com#mesh").unwrap();
        assert_eq!(u.did(), "did:web:example.com");
        assert_eq!(u.service(), None);
        assert_eq!(u.walk_segments(), Vec::<&str>::new());
    }

    #[test]
    fn rejects_non_did_schemes() {
        // #906 acceptance: no 9p:// / moq:// scheme accepted.
        assert!(DidUrl::parse("9p://host:5640/path").is_err());
        assert!(DidUrl::parse("moq://host/stream").is_err());
        assert!(DidUrl::parse("https://example.com/x").is_err());
        assert!(DidUrl::parse("at://did:plc:abc/coll/rkey").is_err());
    }

    #[test]
    fn rejects_malformed_did() {
        assert!(DidUrl::parse("did:").is_err()); // no method/id
        assert!(DidUrl::parse("did:web").is_err()); // no method-specific id
        assert!(DidUrl::parse("did::id").is_err()); // empty method
    }

    #[test]
    fn path_segments_percent_decoded() {
        // Decode only after splitting structural `/` delimiters. `%2F` remains
        // within one inert selector rather than becoming a second segment.
        let u = DidUrl::parse("did:web:example.com/a%2Fb/c%20d").unwrap();
        assert_eq!(u.path(), &["a/b".to_owned(), "c d".to_owned()]);
    }

    // ─── parser: pin (content commitment, client-verified at READ) ─────────

    #[test]
    fn pin_round_trips_when_cid_valid() {
        // A real CIDv1 string (base32). Produced by the rpc cid module so it
        // decodes through the same codec registry the parser validates with.
        let cid_str =
            crate::cid::encode_git_oid("0000000000000000000000000000000000000000").unwrap();
        let url = format!("did:at9p:bafyfake/models/qwen3?pin={cid_str}");
        let u = DidUrl::parse(&url).unwrap();
        assert_eq!(u.pin(), Some(cid_str.as_str()));
        // The plan carries the pin verbatim — READ-time verification is the
        // caller's job, but the commitment round-trips through parse → plan.
        assert_eq!(u.plan().pin.as_deref(), Some(cid_str.as_str()));
    }

    #[test]
    fn pin_rejects_non_cid() {
        // A structurally-invalid pin fails closed at parse time.
        assert!(DidUrl::parse("did:at9p:bafyfake/x?pin=not-a-cid").is_err());
        // base58btc 'z' prefix is for Multikeys, not base32 CIDs.
        assert!(DidUrl::parse("did:at9p:bafyfake/x?pin=z6Mkfoo").is_err());
    }

    #[test]
    fn pin_is_inert_not_authority() {
        // pin asserts no authority: an attacker can reproduce the URL verbatim
        // and it grants nothing. The parser merely carries the CID; it does not
        // mint identity or assurance. (Demonstrated by the plan being selector-
        // only: aname/walk/pin, never a SecurityContext.)
        let cid_str =
            crate::cid::encode_git_oid("1111111111111111111111111111111111111111").unwrap();
        let u = DidUrl::parse(&format!("did:at9p:bafyfake/x?pin={cid_str}&service=ns")).unwrap();
        let plan = u.plan();
        assert_eq!(plan.aname, "ns");
        assert_eq!(plan.walk, vec!["x".to_owned()]);
        assert_eq!(plan.pin.as_deref(), Some(cid_str.as_str()));
    }

    // ─── parser: versionId / versionTime FAIL CLOSED ───────────────────────

    #[test]
    fn version_id_fails_closed_not_treated_as_pin() {
        // versionId is a W3C-registered DID-document-version selector. It MUST
        // fail closed (FEATURE_NOT_SUPPORTED), NEVER be read as a content pin.
        let err = DidUrl::parse("did:at9p:bafyfake/x?versionId=bafyfake").unwrap_err();
        assert!(
            err.to_string().contains("versionId"),
            "must name the offending param: {err}"
        );
        assert!(
            err.to_string().contains("not supported"),
            "must be fail-closed FEATURE_NOT_SUPPORTED: {err}"
        );
    }

    #[test]
    fn version_time_fails_closed() {
        let err =
            DidUrl::parse("did:at9p:bafyfake/x?versionTime=2026-01-01T00:00:00Z").unwrap_err();
        assert!(err.to_string().contains("versionTime"));
    }

    #[test]
    fn version_params_are_never_pins() {
        // Even a versionId value that looks like a CID must not populate pin.
        let res = DidUrl::parse("did:at9p:bafyfake/x?versionId=bafyfakecid");
        assert!(res.is_err());
        // (No DidUrl produced → no pin field can carry it.)
    }

    // ─── parser: hl dropped, unknown params ignored ────────────────────────

    #[test]
    fn hl_is_dropped_from_grammar() {
        // `hl` was removed from the grammar (document-level + redundant for a
        // self-certifying DID). It is silently ignored, not an error, and never
        // populates pin.
        let u = DidUrl::parse("did:at9p:bafyfake/x?hl=bafyfake").unwrap();
        assert!(u.pin().is_none(), "hl must not become a pin");
        assert_eq!(u.walk_segments(), vec!["x"]);
    }

    #[test]
    fn unknown_params_ignored() {
        let u = DidUrl::parse("did:at9p:bafyfake/x?foo=bar&baz=qux&service=ns").unwrap();
        assert_eq!(u.service(), Some("ns"));
        assert!(u.pin().is_none());
    }

    // ─── parser: service → aname mapping ──────────────────────────────────

    #[test]
    fn service_strips_leading_hash_tolerantly() {
        // ?service=%23ns (encoded #) and ?service=ns select the same #ns entry.
        let a = DidUrl::parse("did:at9p:bafyfake/x?service=ns").unwrap();
        let b = DidUrl::parse("did:at9p:bafyfake/x?service=%23ns").unwrap();
        assert_eq!(a.aname(), "ns");
        assert_eq!(b.aname(), "ns");
    }

    #[test]
    fn empty_service_value_rejected() {
        assert!(DidUrl::parse("did:at9p:bafyfake/x?service=").is_err());
        assert!(DidUrl::parse("did:at9p:bafyfake/x?service=%23").is_err());
    }

    #[test]
    fn invalid_utf8_percent_encoding_is_rejected_without_aliasing() {
        assert!(DidUrl::parse("did:at9p:bafyfake/%FF").is_err());
        assert!(DidUrl::parse("did:at9p:bafyfake?relativeRef=%FE").is_err());
        assert!(DidUrl::parse("did:at9p:bafyfake?service=%FF").is_err());

        let replacement = DidUrl::parse("did:at9p:bafyfake/%EF%BF%BD").unwrap();
        assert_eq!(replacement.walk_segments(), vec!["�"]);
    }

    // ─── parser: relativeRef / path normalization (path canonical) ─────────

    #[test]
    fn path_is_canonical_over_relative_ref() {
        // When both carry a walk, path wins and relativeRef is normalized away.
        let u = DidUrl::parse("did:at9p:bafyfake/a/b?relativeRef=c/d").unwrap();
        assert_eq!(u.walk_segments(), vec!["a", "b"], "path must win");
    }

    #[test]
    fn relative_ref_supplies_walk_when_no_path() {
        let u = DidUrl::parse("did:at9p:bafyfake?relativeRef=c/d").unwrap();
        assert!(u.path().is_empty());
        assert_eq!(u.walk_segments(), vec!["c", "d"]);
    }

    #[test]
    fn relative_ref_with_no_segments_is_rejected() {
        assert!(DidUrl::parse("did:at9p:bafyfake?relativeRef=///").is_err());
    }

    #[test]
    fn encoded_slash_has_identical_path_and_relative_ref_semantics() {
        let path = DidUrl::parse("did:at9p:bafyfake/a%2Fb/c").unwrap();
        let relative = DidUrl::parse("did:at9p:bafyfake?relativeRef=a%2Fb/c").unwrap();

        assert_eq!(path.walk_segments(), vec!["a/b", "c"]);
        assert_eq!(relative.walk_segments(), path.walk_segments());
        assert_eq!(
            relative.relative_ref(),
            Some(&["a/b".to_owned(), "c".to_owned()][..])
        );
    }

    #[test]
    fn encoded_slash_cannot_change_precedence() {
        let u =
            DidUrl::parse("did:at9p:bafyfake/canonical%2Fsegment?relativeRef=attacker/controlled")
                .unwrap();
        assert_eq!(u.walk_segments(), vec!["canonical/segment"]);
    }

    #[test]
    fn no_walk_when_neither_present() {
        let u = DidUrl::parse("did:at9p:bafyfake?service=ns").unwrap();
        assert!(u.walk_segments().is_empty());
    }

    // ─── authority separation ──────────────────────────────────────────────

    #[test]
    fn did_key_plan_contains_no_reach_or_possession() {
        let raw = [9u8; 32];
        let did = crate::did_key::ed25519_to_did_key(&raw);
        let url = DidUrl::parse(&format!("{did}/x/y?service=ns")).unwrap();
        let plan = url.plan();

        assert_eq!(plan.did, did);
        assert_eq!(plan.aname, "ns");
        assert_eq!(plan.walk, vec!["x".to_owned(), "y".to_owned()]);
        assert!(plan.pin.is_none());
        // DereferencePlan intentionally has no transport, NodeId, identity key,
        // assurance, or possession field. The key bytes remain inside the DID
        // selector until separately accepted current state and typed reach are
        // supplied by later runtime stages.
    }
}
