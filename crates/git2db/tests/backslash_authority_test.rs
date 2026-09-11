//! Causal, live-network proof that the WHATWG-vs-libgit2 backslash authority
//! differential (issue #1429 Kimi-K3 r3 P1) can no longer leak a host-scoped
//! token, now that [`git2db::auth::extract_git_authority`] fails closed on any
//! `\` in the input.
//!
//! This does **not** mock libgit2. It runs two real local HTTP servers on
//! genuinely distinct loopback hosts — `A` (127.0.0.1, the authority the
//! token is bound to) and `B` (127.0.0.2, an unrelated authority) — and
//! drives a real `git2::Remote::fetch()` against a URL of the form
//! `http://<A>\@<B>/repo.git`.
//!
//! Before the fix, the r3 review's compiled probe proved two things against
//! the real pinned stack: the WHATWG parser (`url::Url::parse`, used by the
//! old `extract_git_authority`) resolved this URL's authority to `A` — so the
//! host-scope check passed — while the actual pinned libgit2 1.9.x C parser
//! resolved the *real* connection target to `B` (backslash is an ordinary
//! authority character to libgit2, and userinfo splits at the last `@`).
//! Both credential paths trusted the WHATWG answer and handed the A-bound
//! token to the connection that was actually talking to `B`.
//!
//! After the fix, `extract_git_authority` returns `None` for any
//! backslash-containing input, so the credential callback refuses the
//! authority-scoped token outright (bound-but-unparseable), regardless of
//! which host the connection actually reaches. This test proves, for real:
//! 1. The differential is real and reachable — the raw TCP connection lands
//!    on `B`, not `A`, exactly as the r3 probe found (`B` is always
//!    contacted; `A` never is).
//! 2. `B` is made to challenge for credentials on *every* request (never
//!    accepting), forcing libgit2 through its full retry sequence exactly as
//!    the r3 probe's real end-to-end run did (whose first attempt carried
//!    URL-embedded userinfo, unrelated to the bound token, before later
//!    attempts would have carried the actual secret). Across every request
//!    `B` ever receives, in neither credential path does any `Authorization`
//!    header ever decode to the bound secret token — the callback refuses to
//!    release it once `extract_git_authority` reports the authority as
//!    unparseable.

#![cfg(test)]
#![allow(clippy::expect_used, clippy::print_stdout)]

use git2db::auth::{AuthManager, AuthStrategy, extract_git_authority};
use git2db::callback_config::CallbackConfigBuilder;
use parking_lot::Mutex;
use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use tempfile::TempDir;

const TOKEN: &str = "trusted-forge-secret-token-1429-r3";

/// `A` (the authority the token is bound to) and `B` (the authority the
/// malicious backslash URL actually resolves to at the transport layer). Both
/// count hits; `B` additionally challenges for credentials and captures
/// whatever `Authorization` header (if any) arrives.
struct BackslashHarness {
    addr_a: SocketAddr,
    addr_b: SocketAddr,
    a_hits: Arc<AtomicUsize>,
    b_hits: Arc<AtomicUsize>,
    /// Every `Authorization` header value `B` has ever observed, one entry
    /// per request (across possibly-multiple retry connections). `B` never
    /// grants access, so libgit2 retries through its full credential
    /// resolution sequence rather than stopping after the first attempt.
    captured_authorizations: Arc<Mutex<Vec<String>>>,
}

impl BackslashHarness {
    fn start() -> Self {
        // Genuinely distinct loopback hosts (127.0.0.1 vs 127.0.0.2), not
        // merely distinct ports — mirrors the r3 probe's harness so the
        // "which host did the TCP connection actually reach" assertion below
        // is meaningful.
        let listener_a = TcpListener::bind("127.0.0.1:0").expect("bind A");
        let addr_a = listener_a.local_addr().expect("addr A");
        let listener_b = TcpListener::bind("127.0.0.2:0").expect("bind B");
        let addr_b = listener_b.local_addr().expect("addr B");

        let a_hits = Arc::new(AtomicUsize::new(0));
        {
            let a_hits = Arc::clone(&a_hits);
            thread::spawn(move || {
                for stream in listener_a.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    a_hits.fetch_add(1, Ordering::SeqCst);
                    let _ = read_request(&mut stream);
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
                    );
                }
            });
        }

        let b_hits = Arc::new(AtomicUsize::new(0));
        let captured_authorizations = Arc::new(Mutex::new(Vec::new()));
        {
            let b_hits = Arc::clone(&b_hits);
            let captured_authorizations = Arc::clone(&captured_authorizations);
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
                    if let Some(value) = &auth_header {
                        captured_authorizations.lock().push(value.clone());
                    }
                    // Never grant access — always challenge, so libgit2 keeps
                    // retrying through its full credential resolution
                    // sequence (URL-embedded userinfo first, then the
                    // configured strategies) instead of stopping after
                    // whatever the first attempt happened to carry.
                    let _ = stream.write_all(
                        b"HTTP/1.1 401 Unauthorized\r\n\
                          WWW-Authenticate: Basic realm=\"git\"\r\n\
                          Connection: close\r\n\
                          Content-Length: 0\r\n\r\n",
                    );
                }
            });
        }

        Self {
            addr_a,
            addr_b,
            a_hits,
            b_hits,
            captured_authorizations,
        }
    }

    fn a_hit_count(&self) -> usize {
        self.a_hits.load(Ordering::SeqCst)
    }

    fn b_hit_count(&self) -> usize {
        self.b_hits.load(Ordering::SeqCst)
    }

    fn captured_authorizations(&self) -> Vec<String> {
        self.captured_authorizations.lock().clone()
    }

    /// The legitimate URL to `A` alone — used only to compute the authority
    /// string the token is bound to, exactly as a real caller configuring a
    /// host-scoped token for `A` would.
    fn legitimate_authority(&self) -> String {
        extract_git_authority(&format!("http://{}/repo.git", self.addr_a))
            .expect("A's own URL must parse to a concrete authority")
    }

    /// The malicious URL: `http://<A>\@<B>/repo.git`. Per the r3 probe, the
    /// real libgit2 transport resolves the connection target to `B` (`\` is
    /// an ordinary authority character to libgit2, userinfo splits at the
    /// last `@`), while the pre-fix WHATWG parser resolved the authority to
    /// `A`. This is exactly the differential the fix must close.
    fn malicious_url(&self) -> String {
        format!("http://{}\\@{}/repo.git", self.addr_a, self.addr_b)
    }
}

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

/// Drive a real `git2` fetch against `url` with the given callbacks. None of
/// these servers speak the full git smart-HTTP protocol, so the fetch is
/// expected to ultimately fail; what matters is which servers were contacted
/// and with what credentials, observed via the harness.
fn drive_fetch(url: &str, callbacks: git2::RemoteCallbacks<'_>) {
    let tmp = TempDir::new().expect("tempdir");
    let repo = git2::Repository::init_bare(tmp.path()).expect("init bare repo");
    let mut remote = repo.remote_anonymous(url).expect("remote_anonymous");
    let mut fetch_opts = git2::FetchOptions::new();
    fetch_opts.remote_callbacks(callbacks);
    fetch_opts.follow_redirects(git2::RemoteRedirect::None);
    let result = remote.fetch(
        &["+refs/heads/*:refs/remotes/origin/*"],
        Some(&mut fetch_opts),
        None,
    );
    if let Err(e) = result {
        println!("drive_fetch: fetch returned (expected) error: {e}");
    }
}

/// Sanity: the malicious URL is rejected by `extract_git_authority` itself
/// (unit-level cross-check that this test exercises the same fix the `auth.rs`
/// tests cover, against a URL built from real, live harness addresses).
fn assert_malicious_url_unparseable(url: &str) {
    assert_eq!(
        extract_git_authority(url),
        None,
        "a backslash-containing authority must be rejected outright, not resolved to \
         either parser's differing answer",
    );
}

/// None of the `Authorization` headers `B` observed may decode (as HTTP Basic
/// auth) to a credential ending in the bound secret token. Some entries may
/// legitimately be present — e.g. libgit2's own use of URL-embedded userinfo
/// on the first attempt — but that material originates from the URL string
/// itself (attacker/caller-visible), never from the configured host-scoped
/// secret.
fn assert_token_never_delivered(captured: &[String]) {
    for value in captured {
        let decoded = decode_basic_auth(value);
        assert!(
            !decoded.ends_with(TOKEN),
            "the bound secret token must never be delivered to B across the backslash \
             authority differential; observed credential decoded to {decoded:?}",
        );
    }
}

/// Minimal base64 decoder for the `Basic <base64>` credential libgit2 may
/// send — avoids pulling in a base64 dependency just for this assertion.
fn decode_basic_auth(header_value: &str) -> String {
    let b64 = header_value
        .strip_prefix("Basic ")
        .unwrap_or(header_value)
        .trim();
    String::from_utf8_lossy(&base64_decode(b64)).into_owned()
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

// ---------------------------------------------------------------------------
// CallbackConfig (send-safe clone) path
// ---------------------------------------------------------------------------

#[test]
fn callback_config_refuses_token_across_backslash_authority_differential() {
    let harness = BackslashHarness::start();
    assert_ne!(
        harness.addr_a.to_string(),
        harness.addr_b.to_string(),
        "sanity: A and B must be genuinely distinct hosts"
    );

    let malicious_url = harness.malicious_url();
    assert_malicious_url_unparseable(&malicious_url);

    let config = CallbackConfigBuilder::new()
        .auth(AuthStrategy::Token {
            token: TOKEN.to_owned(),
            host: Some(harness.legitimate_authority()),
        })
        .build();

    drive_fetch(&malicious_url, config.create_callbacks());

    assert!(
        harness.b_hit_count() >= 1,
        "sanity: the real TCP connection must reach B (the r3-proven differential — \
         libgit2 resolves the backslash URL's authority to B, not A), otherwise this test \
         is not exercising the vulnerable path at all",
    );
    assert_token_never_delivered(&harness.captured_authorizations());
    assert_eq!(
        harness.a_hit_count(),
        0,
        "A must never be contacted either — the connection goes straight to B",
    );
}

// ---------------------------------------------------------------------------
// Legacy AuthManager (sync) path
// ---------------------------------------------------------------------------

#[test]
fn auth_manager_refuses_token_across_backslash_authority_differential() {
    let harness = BackslashHarness::start();

    let malicious_url = harness.malicious_url();
    assert_malicious_url_unparseable(&malicious_url);

    let mgr = AuthManager::with_strategies(vec![AuthStrategy::Token {
        token: TOKEN.to_owned(),
        host: Some(harness.legitimate_authority()),
    }]);

    drive_fetch(&malicious_url, mgr.create_callbacks());

    assert!(
        harness.b_hit_count() >= 1,
        "sanity: the real TCP connection must reach B, exercising the vulnerable path",
    );
    assert_token_never_delivered(&harness.captured_authorizations());
}
