//! The one bounded decode of a signed request body (v16 §5.2).
//!
//! `DecodedRequestBody` is produced exactly once per request by the service's
//! **generated** decoder ([`crate::service::RequestService::decode_request_body`])
//! and then feeds every consumer of the request's method identity and content:
//!
//! - the generated per-method signature policy (`proof::policy`) resolves its
//!   row with [`DecodedRequestBody::leaf_path`];
//! - the dispatch MAC PEP receives the same leaf path as its object
//!   coordinate; and
//! - the handler dispatch reads the request from the same decoded message via
//!   [`DecodedRequestBody::root`] — there is no second
//!   `capnp::serialize::read_message` anywhere between admission and handler.
//!
//! The leaf path is the chain of Cap'n Proto union discriminants from the
//! service root request union down to the executed leaf (v16 §5.1): nested
//! scope unions contribute their discriminants, so a nested method selects a
//! distinct policy row rather than collapsing onto its parent. Human-readable
//! names are review metadata; the numeric discriminants are wire identity.

use capnp::message::ReaderOptions;
use capnp::serialize::OwnedSegments;

/// Reviewed decode caps for the single signed-body decode (v16 §5.2).
///
/// Cap'n Proto's pointer-traversal amplification is bounded by the traversal
/// limit (in 8-byte words); nesting depth bounds recursion. These are
/// deliberately far below `ReaderOptions::default()`'s 64 MiB traversal
/// allowance: an RPC request body is transport-size-capped well below this,
/// so the caps bound decode work without constraining any legitimate request.
pub const BODY_TRAVERSAL_LIMIT_WORDS: u64 = 2 * 1024 * 1024; // 16 MiB of traversal
/// Maximum pointer nesting depth for the signed-body decode.
pub const BODY_NESTING_LIMIT: i32 = 64;

/// The bounded reader options every generated signed-body decoder uses.
pub fn bounded_reader_options() -> ReaderOptions {
    let mut opts = ReaderOptions::new();
    opts.traversal_limit_in_words(Some(BODY_TRAVERSAL_LIMIT_WORDS as usize));
    opts.nesting_limit(BODY_NESTING_LIMIT);
    opts
}

/// A request body decoded exactly once, with its derived full method leaf.
///
/// Constructed by generated service decoders via [`DecodedRequestBody::from_message`],
/// or — only for services with no Cap'n Proto request schema (test echo
/// services, byte-oriented bridges) — via [`DecodedRequestBody::opaque`].
pub struct DecodedRequestBody {
    /// The exact signed body bytes (what the request proof signs).
    bytes: Vec<u8>,
    /// The one decoded message. `None` only for [`DecodedRequestBody::opaque`]
    /// bodies, which cannot serve generated dispatch.
    message: Option<capnp::message::Reader<OwnedSegments>>,
    /// Full union-discriminant chain from the root request union to the
    /// executed leaf. Empty only for opaque bodies.
    leaf_path: Vec<u16>,
}

impl DecodedRequestBody {
    /// Wrap the one decoded message and its derived leaf path.
    ///
    /// Generated decoders call this after the single bounded
    /// `read_message` + leaf traversal. The leaf path must be non-empty: a
    /// request whose root union arm is unknown never reaches this
    /// constructor (the generated decoder returns an error instead).
    pub fn from_message(
        bytes: Vec<u8>,
        message: capnp::message::Reader<OwnedSegments>,
        leaf_path: Vec<u16>,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            !leaf_path.is_empty(),
            "decoded request body derived an empty method leaf path"
        );
        Ok(Self {
            bytes,
            message: Some(message),
            leaf_path,
        })
    }

    /// A body for a service with **no** Cap'n Proto request schema.
    ///
    /// The body carries raw bytes only: there is no decoded message and no
    /// derivable method leaf, so a proof-bearing request to such a service
    /// denies at leaf derivation (v16 §5.2) and generated dispatch cannot be
    /// served from it. This exists for byte-oriented test/bridge services and
    /// is an affirmative per-service choice, never a default.
    pub fn opaque(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            message: None,
            leaf_path: Vec::new(),
        }
    }

    /// The exact signed body bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The full numeric method leaf path, or `None` for an opaque body.
    pub fn leaf_path(&self) -> Option<&[u16]> {
        if self.leaf_path.is_empty() {
            None
        } else {
            Some(&self.leaf_path)
        }
    }

    /// The leaf path in the dotted form generated policy rows are keyed by.
    pub fn leaf_path_string(&self) -> Option<String> {
        self.leaf_path().map(|path| {
            path.iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(".")
        })
    }

    /// Read the typed root of the one decoded message.
    ///
    /// Generated dispatch reads the request through this — pointer traversal
    /// over the already-decoded message, never a second decode of the bytes.
    pub fn root<'a, T: capnp::traits::FromPointerReader<'a>>(&'a self) -> anyhow::Result<T> {
        let message = self.message.as_ref().ok_or_else(|| {
            anyhow::anyhow!("opaque request body has no decoded message to dispatch from")
        })?;
        message.get_root::<T>().map_err(Into::into)
    }
}

impl std::fmt::Debug for DecodedRequestBody {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedRequestBody")
            .field("bytes_len", &self.bytes.len())
            .field("decoded", &self.message.is_some())
            .field("leaf_path", &self.leaf_path)
            .finish()
    }
}
