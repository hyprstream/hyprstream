//! Pure framing + reach-config parsing for the browser moq subscriber.
//!
//! These are the transport-independent, side-effect-free pieces of the moq
//! worker (`moq_worker.rs`, wasm-only). They live here — NOT gated on
//! `target_arch = "wasm32"` — so they can be unit-tested on the host. The wasm
//! worker re-uses them; the moq subscribe/WebTransport integration itself is
//! exercised by the downstream browser e2e.

use serde::Deserialize;

// DMA frame envelope: `[2: track_len LE][track][4: payload_len LE][payload]`.
//
// A special track name (`__cmd__`) carries JSON control messages main→worker;
// `__ready__`/`__error__` are status frames worker→main.

/// Encode one DMA frame.
///
/// The track-length prefix is a `u16`; a track name longer than `u16::MAX`
/// would truncate the length header and desync [`decode_frame`] for the rest of
/// the stream, so we drop the over-long frame (returning an empty `Vec`) rather
/// than corrupt the channel. Track names are short by construction (moq paths).
pub fn encode_frame(track: &str, payload: &[u8]) -> Vec<u8> {
    let t = track.as_bytes();
    let track_len = match u16::try_from(t.len()) {
        Ok(len) => len,
        Err(_) => {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::error_1(
                &format!("[moq-worker] track name too long ({} bytes), dropping frame", t.len()).into(),
            );
            return Vec::new();
        }
    };
    let mut out = Vec::with_capacity(2 + t.len() + 4 + payload.len());
    out.extend_from_slice(&track_len.to_le_bytes());
    out.extend_from_slice(t);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Decode one DMA frame. Returns `(track_name, payload)`, or `None` if the
/// buffer is malformed (too short, length prefix past the end, or a non-UTF-8
/// track name).
pub fn decode_frame(buf: &[u8]) -> Option<(&str, &[u8])> {
    if buf.len() < 6 {
        return None;
    }
    let track_len = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    if buf.len() < 2 + track_len + 4 {
        return None;
    }
    let track = std::str::from_utf8(&buf[2..2 + track_len]).ok()?;
    let payload_len = u32::from_le_bytes([
        buf[2 + track_len],
        buf[3 + track_len],
        buf[4 + track_len],
        buf[5 + track_len],
    ]) as usize;
    let start = 2 + track_len + 4;
    if buf.len() < start + payload_len {
        return None;
    }
    Some((track, &buf[start..start + payload_len]))
}

/// Network-routable reach the frontend selects (via `selectReach`) and passes to
/// the worker as JSON. The moq broadcast path (which track to subscribe) is NOT
/// here — it arrives later in a `subscribe` command, since one worker can serve
/// several subscriptions on the same endpoint.
///
/// Wire shape: `{ "url": "https://<addr>/moq", "certHashes": ["<base64 SHA-256>", …] }`
/// `certHashes` are acceptable leaf-cert SHA-256 pins for the self-signed mesh
/// (matching the QUIC reach's `certHashes`); absent/empty ⇒ use the browser's
/// system roots (a CA-issued cert).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReachConfig {
    pub url: String,
    pub cert_hashes: Vec<Vec<u8>>,
}

#[derive(Deserialize)]
struct ReachJson {
    #[serde(default)]
    url: String,
    #[serde(default, rename = "certHashes")]
    cert_hashes: Vec<String>,
}

/// Parse the reach JSON the frontend emits into a [`ReachConfig`], base64-decoding
/// the cert hashes.
///
/// A cert pin is a security control: silently dropping an undecodable entry
/// and falling back to system roots would connect unpinned without ever
/// telling the caller. So an individual undecodable entry is an error (not
/// skipped) whenever the list is non-empty — the only silent case is a
/// genuinely empty `certHashes` list, which is the documented "use the
/// browser's system roots" contract.
pub fn parse_reach(reach_json: &str) -> Result<ReachConfig, String> {
    let parsed: ReachJson =
        serde_json::from_str(reach_json).map_err(|e| format!("bad reach_json: {e}"))?;
    use base64::Engine;
    let mut cert_hashes = Vec::with_capacity(parsed.cert_hashes.len());
    for s in &parsed.cert_hashes {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(s)
            .map_err(|e| format!("certHashes entry {s:?} is not valid base64: {e}"))?;
        cert_hashes.push(decoded);
    }
    Ok(ReachConfig {
        url: parsed.url,
        cert_hashes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn frame_round_trips() {
        let f = encode_frame("stream", b"hello-tokens");
        assert_eq!(decode_frame(&f), Some(("stream", &b"hello-tokens"[..])));
    }

    #[test]
    fn frame_round_trips_empty_payload_and_track() {
        let f = encode_frame("", b"");
        assert_eq!(decode_frame(&f), Some(("", &b""[..])));
        let f = encode_frame("__cmd__", b"");
        assert_eq!(decode_frame(&f), Some(("__cmd__", &b""[..])));
    }

    #[test]
    fn frame_handles_binary_payload_with_embedded_lengths() {
        // A payload whose bytes look like length prefixes must not confuse decode.
        let payload: Vec<u8> = (0u8..=255).cycle().take(1000).collect();
        let f = encode_frame("local/streams/deadbeef", &payload);
        let (track, got) = decode_frame(&f).expect("decode");
        assert_eq!(track, "local/streams/deadbeef");
        assert_eq!(got, &payload[..]);
    }

    #[test]
    fn over_long_track_is_dropped_not_corrupted() {
        let huge = "x".repeat(u16::MAX as usize + 1);
        assert!(encode_frame(&huge, b"payload").is_empty());
    }

    #[test]
    fn decode_rejects_malformed() {
        assert_eq!(decode_frame(&[]), None); // empty
        assert_eq!(decode_frame(&[0, 0, 0, 0, 0]), None); // < 6 bytes
        // track_len says 10 but buffer is too short:
        assert_eq!(decode_frame(&[10, 0, b'a', b'b', 0, 0]), None);
        // Valid header but truncated payload:
        let mut f = encode_frame("stream", b"hello");
        f.truncate(f.len() - 2);
        assert_eq!(decode_frame(&f), None);
        // Non-UTF-8 track bytes:
        let bad = [2u8, 0, 0xff, 0xfe, 0, 0, 0, 0];
        assert_eq!(decode_frame(&bad), None);
    }

    #[test]
    fn parse_reach_url_and_cert_hashes() {
        // base64 of the 32-byte hash [0x01; 32] = "AQEB…"; use a known small value.
        let b64 = base64::engine::general_purpose::STANDARD.encode([0xabu8; 32]);
        let json = format!(r#"{{"url":"https://host.tld:4433/moq","certHashes":["{b64}"]}}"#);
        let r = parse_reach(&json).unwrap();
        assert_eq!(r.url, "https://host.tld:4433/moq");
        assert_eq!(r.cert_hashes, vec![vec![0xabu8; 32]]);
    }

    #[test]
    fn parse_reach_defaults() {
        // Missing certHashes ⇒ empty (fall back to system roots) — the only
        // silent case; a *present but undecodable* entry is fatal (below).
        let r = parse_reach(r#"{"url":"https://h/moq"}"#).unwrap();
        assert_eq!(r.url, "https://h/moq");
        assert!(r.cert_hashes.is_empty());
        // Missing url ⇒ empty string (caller treats as unusable).
        let r = parse_reach(r#"{}"#).unwrap();
        assert!(r.url.is_empty());
    }

    #[test]
    fn parse_reach_rejects_invalid_json() {
        assert!(parse_reach("not json").is_err());
    }

    #[test]
    fn parse_reach_rejects_undecodable_cert_hash() {
        // A non-empty certHashes list with an undecodable entry must fail
        // closed, not silently drop the pin and fall back to system roots.
        let r = parse_reach(r#"{"url":"u","certHashes":["!!!not base64!!!"]}"#);
        assert!(r.is_err(), "undecodable cert pin must be rejected, not skipped");
    }
}
