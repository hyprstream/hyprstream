//! Causal, live-network proof of the off-site redirect credential-exfiltration
//! fix (issue #1429 Sol P1, revision round 2).
//!
//! This test does **not** mock libgit2. It runs two real local HTTP servers —
//! `A` (the configured/trusted origin, holding a host-scoped token) and `B`
//! (an unrelated "off-site" authority, on a genuinely distinct loopback host
//! so libgit2's own host-based off-site check treats it as off-site) — and
//! drives a real `git2::Remote::fetch()` against `A`, which unconditionally
//! 301-redirects to `B`. It proves, against the actual pinned libgit2 network
//! stack:
//!
//! 1. With the secure default ([`RedirectPolicy::None`]), `B` is **never
//!    contacted** — the redirect is refused at the transport layer, so the
//!    credential callback is never even invoked in a context where it could
//!    leak the token to `B`.
//! 2. With the explicit, insecure opt-in ([`RedirectPolicy::Initial`] —
//!    libgit2's own default), `B` **is** contacted and, per the exact
//!    mechanism Sol's review cited (the credential callback receives the
//!    pre-redirect URL from `A`, so a token scoped to `A` is judged
//!    authorized and handed to the transport, which is now talking to `B`),
//!    the token **is** delivered to `B`. This is not a bug in this test; it
//!    is the documented reason `None` must be the default rather than an
//!    opt-out.
//!
//! Both the send-safe [`CallbackConfig`] clone path and the legacy
//! [`AuthManager`] path are exercised, since both independently need the
//! redirect policy applied to the `git2::FetchOptions` that drives the fetch.

#![cfg(test)]
// Test harness needs: setup calls that should hard-fail the test on error
// (`expect`), and diagnostic visibility into an intentionally-failing fetch
// (`println!`) — both allowed here per the same convention as
// `security_tests.rs` / `send_trait_test.rs`.
#![allow(clippy::expect_used, clippy::print_stdout)]

use git2db::auth::{AuthManager, AuthStrategy};
use git2db::callback_config::{CallbackConfigBuilder, RedirectPolicy};
use parking_lot::Mutex;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

/// Bound to a trusted-forge-shaped token so a captured `Authorization` header
/// unambiguously proves the *token*, not just *some* credential, reached `B`.
const TOKEN: &str = "trusted-forge-secret-token-1429";

/// Two raw-TCP HTTP servers: `A` unconditionally redirects off-site to `B`;
/// `B` challenges for credentials on the first request and captures whatever
/// `Authorization` header arrives on the second.
struct RedirectHarness {
    addr_a: SocketAddr,
    addr_b: SocketAddr,
    b_hits: Arc<AtomicUsize>,
    captured_authorization: Arc<Mutex<Option<String>>>,
}

impl RedirectHarness {
    fn start() -> Self {
        // `A` and `B` are bound to genuinely distinct loopback addresses
        // (127.0.0.1 vs 127.0.0.2 — both route to localhost on Linux), not
        // merely distinct ports on the same address. libgit2's off-site
        // redirect check compares *host*, not port, so a port-only
        // difference is treated as same-site and followed unconditionally
        // regardless of `RemoteRedirect` policy — using distinct hosts is
        // required for this harness to actually exercise the policy.
        let listener_a = TcpListener::bind("127.0.0.1:0").expect("bind A");
        let addr_a = listener_a.local_addr().expect("addr A");
        let listener_b = TcpListener::bind("127.0.0.2:0").expect("bind B");
        let addr_b = listener_b.local_addr().expect("addr B");

        let b_hits = Arc::new(AtomicUsize::new(0));
        let captured_authorization = Arc::new(Mutex::new(None));

        thread::spawn(move || {
            for stream in listener_a.incoming() {
                let Ok(mut stream) = stream else { continue };
                let Some((request_line, _headers)) = read_request(&mut stream) else {
                    continue;
                };
                let Some(path) = request_path(&request_line) else {
                    continue;
                };
                let response = format!(
                    "HTTP/1.1 301 Moved Permanently\r\n\
                     Location: http://{addr_b}{path}\r\n\
                     Connection: close\r\n\
                     Content-Length: 0\r\n\r\n"
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });

        {
            let b_hits = Arc::clone(&b_hits);
            let captured_authorization = Arc::clone(&captured_authorization);
            thread::spawn(move || {
                for stream in listener_b.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    b_hits.fetch_add(1, Ordering::SeqCst);
                    let Some((_request_line, headers)) = read_request(&mut stream) else {
                        continue;
                    };
                    let auth_header = headers.iter().find_map(|h| {
                        let (name, value) = h.split_once(':')?;
                        name.trim()
                            .eq_ignore_ascii_case("authorization")
                            .then(|| value.trim().to_owned())
                    });
                    if let Some(value) = auth_header {
                        *captured_authorization.lock() = Some(value);
                        let _ = stream.write_all(
                            b"HTTP/1.1 200 OK\r\n\
                              Content-Type: application/x-git-upload-pack-advertisement\r\n\
                              Connection: close\r\n\
                              Content-Length: 0\r\n\r\n",
                        );
                    } else {
                        let _ = stream.write_all(
                            b"HTTP/1.1 401 Unauthorized\r\n\
                              WWW-Authenticate: Basic realm=\"git\"\r\n\
                              Connection: close\r\n\
                              Content-Length: 0\r\n\r\n",
                        );
                    }
                }
            });
        }

        Self {
            addr_a,
            addr_b,
            b_hits,
            captured_authorization,
        }
    }

    fn b_hit_count(&self) -> usize {
        self.b_hits.load(Ordering::SeqCst)
    }

    fn captured_authorization(&self) -> Option<String> {
        self.captured_authorization.lock().clone()
    }

    /// The origin URL to fetch from — always points at `A`.
    fn origin_url(&self) -> String {
        format!("http://{}/redirect-repo.git", self.addr_a)
    }

    /// `A`'s canonical authority string, matching what
    /// `git2db::auth::extract_git_authority` would derive from
    /// [`RedirectHarness::origin_url`] — used to bind the host-scoped token.
    fn origin_authority(&self) -> String {
        self.addr_a.to_string()
    }

    /// `B`'s address, for diagnostic assertions that the two servers are
    /// genuinely distinct hosts (not merely distinct ports).
    fn redirect_target_addr(&self) -> SocketAddr {
        self.addr_b
    }
}

/// Read one HTTP request's start-line and headers (stops at the blank line;
/// does not attempt to read a body — none of the requests this harness
/// receives carry one).
fn read_request(stream: &mut TcpStream) -> Option<(String, Vec<String>)> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).ok()? == 0 {
        return None;
    }
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line).ok()?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        headers.push(line.trim_end().to_owned());
    }
    Some((request_line.trim_end().to_owned(), headers))
}

fn request_path(request_line: &str) -> Option<String> {
    request_line.split_whitespace().nth(1).map(str::to_owned)
}

/// Perform a real `git2` fetch against `url` with the given `FetchOptions`.
/// The `Result` is intentionally ignored by callers beyond logging — none of
/// these servers speak the full git smart-HTTP protocol, so the fetch is
/// expected to ultimately fail; what matters is which servers were contacted
/// and with what credentials, observed via the harness.
fn drive_fetch(url: &str, fetch_opts: &mut git2::FetchOptions<'_>) {
    let tmp = TempDir::new().expect("tempdir");
    let repo = git2::Repository::init_bare(tmp.path()).expect("init bare repo");
    let mut remote = repo.remote_anonymous(url).expect("remote_anonymous");
    let result = remote.fetch(
        &["+refs/heads/*:refs/remotes/origin/*"],
        Some(fetch_opts),
        None,
    );
    // Expected to fail (the harness servers are not a real git backend); the
    // assertions of interest are on harness-observed side effects, not this
    // Result. Logged for diagnostic visibility only.
    if let Err(e) = result {
        println!("drive_fetch: fetch returned (expected) error: {e}");
    }
}

// ---------------------------------------------------------------------------
// CallbackConfig (send-safe clone) path
// ---------------------------------------------------------------------------

/// With the secure default (`RedirectPolicy::None`), the off-site redirect to
/// `B` is never followed: `B`'s socket is never contacted, so no credential —
/// scoped or otherwise — can reach it.
#[test]
fn callback_config_default_redirect_policy_blocks_off_site_redirect() {
    let harness = RedirectHarness::start();
    assert_ne!(
        harness.origin_authority(),
        harness.redirect_target_addr().to_string(),
        "sanity: A and B must be genuinely distinct hosts, not merely distinct ports on the \
         same host — libgit2's off-site check is host-based",
    );

    let config = CallbackConfigBuilder::new()
        .auth(AuthStrategy::Token {
            token: TOKEN.to_owned(),
            host: Some(harness.origin_authority()),
        })
        .build();
    assert_eq!(
        config.redirect_policy,
        RedirectPolicy::None,
        "sanity: CallbackConfigBuilder must not silently opt into Initial"
    );

    let mut fetch_opts = git2::FetchOptions::new();
    fetch_opts.remote_callbacks(config.create_callbacks());
    fetch_opts.follow_redirects(config.redirect_policy.to_git2());

    drive_fetch(&harness.origin_url(), &mut fetch_opts);

    assert_eq!(
        harness.b_hit_count(),
        0,
        "RedirectPolicy::None must prevent the off-site redirect target from ever being \
         contacted",
    );
    assert_eq!(
        harness.captured_authorization(),
        None,
        "no credential can leak to a server that was never contacted",
    );
}

/// With the explicit, insecure opt-in (`RedirectPolicy::Initial`, libgit2's
/// own default), the redirect IS followed and — because the credential
/// callback receives the pre-redirect (`A`) URL even though the transport is
/// now talking to `B` — the token bound to `A`'s authority IS delivered to
/// `B`. This documents, against the real pinned libgit2 behavior, exactly the
/// vulnerability `RedirectPolicy::None` is the mandatory default to close.
#[test]
fn callback_config_initial_redirect_policy_leaks_token_to_redirect_target() {
    let harness = RedirectHarness::start();

    let config = CallbackConfigBuilder::new()
        .auth(AuthStrategy::Token {
            token: TOKEN.to_owned(),
            host: Some(harness.origin_authority()),
        })
        .redirect_policy(RedirectPolicy::Initial)
        .build();

    let mut fetch_opts = git2::FetchOptions::new();
    fetch_opts.remote_callbacks(config.create_callbacks());
    fetch_opts.follow_redirects(config.redirect_policy.to_git2());

    drive_fetch(&harness.origin_url(), &mut fetch_opts);

    assert!(
        harness.b_hit_count() >= 1,
        "RedirectPolicy::Initial must follow the off-site redirect to B",
    );
    let captured = harness
        .captured_authorization()
        .expect("B must have observed an Authorization header once challenged");
    let decoded = decode_basic_auth(&captured);
    assert!(
        decoded.ends_with(TOKEN),
        "the token bound to A's authority must have been delivered to B via Basic auth; got \
         decoded credential {decoded:?}",
    );
}

// ---------------------------------------------------------------------------
// Legacy AuthManager (sync) path
// ---------------------------------------------------------------------------

/// The legacy `AuthManager` path defaults `redirect_policy()` to `None`; a
/// caller that plumbs it into `FetchOptions::follow_redirects` (as this test
/// does — `AuthManager` owns no `FetchOptions` of its own) gets the same
/// off-site-redirect protection as the `CallbackConfig` path.
#[test]
fn auth_manager_default_redirect_policy_blocks_off_site_redirect() {
    let harness = RedirectHarness::start();

    let mgr = AuthManager::with_strategies(vec![AuthStrategy::Token {
        token: TOKEN.to_owned(),
        host: Some(harness.origin_authority()),
    }]);
    assert_eq!(mgr.redirect_policy(), RedirectPolicy::None);

    let mut fetch_opts = git2::FetchOptions::new();
    fetch_opts.remote_callbacks(mgr.create_callbacks());
    fetch_opts.follow_redirects(mgr.redirect_policy().to_git2());

    drive_fetch(&harness.origin_url(), &mut fetch_opts);

    assert_eq!(
        harness.b_hit_count(),
        0,
        "AuthManager's default redirect policy must prevent B from ever being contacted",
    );
    assert_eq!(harness.captured_authorization(), None);
}

/// The same leak, reproduced through the legacy `AuthManager` path when a
/// caller explicitly opts into `RedirectPolicy::Initial` — proving the
/// vulnerability (and the fix) is not specific to the `CallbackConfig` type.
#[test]
fn auth_manager_initial_redirect_policy_leaks_token_to_redirect_target() {
    let harness = RedirectHarness::start();

    let mgr = AuthManager::with_strategies(vec![AuthStrategy::Token {
        token: TOKEN.to_owned(),
        host: Some(harness.origin_authority()),
    }])
    .with_redirect_policy(RedirectPolicy::Initial);

    let mut fetch_opts = git2::FetchOptions::new();
    fetch_opts.remote_callbacks(mgr.create_callbacks());
    fetch_opts.follow_redirects(mgr.redirect_policy().to_git2());

    drive_fetch(&harness.origin_url(), &mut fetch_opts);

    assert!(harness.b_hit_count() >= 1);
    let captured = harness
        .captured_authorization()
        .expect("B must have observed an Authorization header once challenged");
    let decoded = decode_basic_auth(&captured);
    assert!(
        decoded.ends_with(TOKEN),
        "got decoded credential {decoded:?}"
    );
}

/// Minimal base64 decoder for the `Basic <base64>` credential libgit2 sends —
/// avoids pulling in a base64 dependency just for this test assertion.
fn decode_basic_auth(header_value: &str) -> String {
    let b64 = header_value
        .strip_prefix("Basic ")
        .unwrap_or(header_value)
        .trim();
    String::from_utf8(base64_decode(b64)).expect("valid utf8 credential")
}

fn base64_decode(input: &str) -> Vec<u8> {
    const TABLE: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lut = [0u8; 256];
    for (i, &c) in TABLE.iter().enumerate() {
        lut[c as usize] = i as u8;
    }
    let clean: Vec<u8> = input.bytes().filter(|&b| b != b'=').collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        let mut buf = [0u8; 4];
        for (i, &b) in chunk.iter().enumerate() {
            buf[i] = lut[b as usize];
        }
        out.push((buf[0] << 2) | (buf[1] >> 4));
        if chunk.len() > 2 {
            out.push((buf[1] << 4) | (buf[2] >> 2));
        }
        if chunk.len() > 3 {
            out.push((buf[2] << 6) | buf[3]);
        }
    }
    out
}
