//! Canonical DID-document artifact for a hyprstream-hosted account.
//!
//! The account document is an immutable input to genesis, not a response
//! assembled by an HTTP handler. [`HostedAccountMint`](crate::HostedAccountMint)
//! builds it from the mint's own account DID and generated ES256 key, then
//! [`SealedHostedDidDocument::cid`] is signed into the genesis `DidOp`.
//! Serving code must return [`SealedHostedDidDocument::as_bytes`] byte-for-byte.

use std::collections::BTreeSet;

use anyhow::{bail, ensure, Context, Result};
use p256::ecdsa::VerifyingKey;
use serde_json::{json, Map, Value};

use crate::cid::Cid;
use crate::did_op::validate_host_form_did_web;
use crate::hosted_account::AllocatedAccountName;

/// Media type for the canonical account DID-document bytes.
pub const DID_DOCUMENT_MEDIA_TYPE: &str = "application/did+json";
/// Relative path at which B3 serves the canonical document.
pub const DID_DOCUMENT_PATH: &str = "/.well-known/did.json";
/// Relative path at which B3 serves the account's operation log.
pub const DID_OPERATION_LOG_PATH: &str = "/.well-known/did-log.json";

const MAX_DID_DOCUMENT_BYTES: usize = 16 * 1024;
const P256_MULTICODEC_PREFIX: [u8; 2] = [0x80, 0x24];
const COMPRESSED_P256_PUBLIC_KEY_LEN: usize = 33;
const DID_CONTEXT: &str = "https://www.w3.org/ns/did/v1";
const MULTIKEY_CONTEXT: &str = "https://w3id.org/security/multikey/v1";

/// A validated, byte-stable hosted-account DID document and its raw CID.
///
/// Fields are private so a caller cannot change a document after its CID has
/// been bound into genesis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SealedHostedDidDocument {
    did: String,
    handle: String,
    pds_endpoint: String,
    atproto_key: Vec<u8>,
    bytes: Vec<u8>,
    cid: Cid,
}

impl SealedHostedDidDocument {
    /// Parse and strictly validate canonical bytes loaded from sealed storage.
    ///
    /// The accepted shape is deliberately exact: one `#atproto` P-256
    /// `Multikey`, one atproto PDS service, one operation-log service, and one
    /// `at://` alias equal to the immutable account host. Unknown fields,
    /// alternate aliases, and non-canonical JSON fail closed.
    pub fn from_canonical_json(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_DID_DOCUMENT_BYTES,
            "hosted DID document exceeds {MAX_DID_DOCUMENT_BYTES} bytes"
        );
        let value: Value =
            serde_json::from_slice(bytes).context("hosted DID document is not valid JSON")?;
        ensure!(
            canonical_json(&value)? == bytes,
            "hosted DID document is not canonical JSON"
        );
        validate_exact_keys(
            &value,
            &[
                "@context",
                "alsoKnownAs",
                "id",
                "service",
                "verificationMethod",
            ],
            "hosted DID document",
        )?;

        let did = required_string(&value, "id", "hosted DID document")?.to_owned();
        validate_host_form_did_web(&did)?;
        let host = did
            .strip_prefix("did:web:")
            .ok_or_else(|| anyhow::anyhow!("hosted DID document id must use did:web"))?;

        let contexts = required_array(&value, "@context", "hosted DID document")?;
        ensure!(
            contexts
                == [
                    Value::String(DID_CONTEXT.to_owned()),
                    Value::String(MULTIKEY_CONTEXT.to_owned()),
                ],
            "hosted DID document has an unsupported @context"
        );

        let aliases = required_array(&value, "alsoKnownAs", "hosted DID document")?;
        let expected_handle = format!("at://{host}");
        ensure!(
            aliases == [Value::String(expected_handle.clone())],
            "hosted DID document must contain exactly the account at:// handle and no other alias"
        );

        let methods = required_array(&value, "verificationMethod", "hosted DID document")?;
        ensure!(
            methods.len() == 1,
            "hosted DID document must contain exactly one verification method"
        );
        let method = methods
            .first()
            .ok_or_else(|| anyhow::anyhow!("hosted DID document omits #atproto"))?;
        validate_exact_keys(
            method,
            &["controller", "id", "publicKeyMultibase", "type"],
            "#atproto verification method",
        )?;
        ensure!(
            required_string(method, "id", "#atproto verification method")?
                == format!("{did}#atproto"),
            "hosted DID document verification method must be the account's #atproto fragment"
        );
        ensure!(
            required_string(method, "controller", "#atproto verification method")? == did,
            "hosted DID document #atproto controller does not match its id"
        );
        ensure!(
            required_string(method, "type", "#atproto verification method")? == "Multikey",
            "hosted DID document #atproto method must use Multikey"
        );
        let atproto_key = decode_p256_multibase(required_string(
            method,
            "publicKeyMultibase",
            "#atproto verification method",
        )?)?;

        let services = required_array(&value, "service", "hosted DID document")?;
        ensure!(
            services.len() == 2,
            "hosted DID document must contain exactly the PDS and operation-log services"
        );
        let pds = services
            .first()
            .ok_or_else(|| anyhow::anyhow!("hosted DID document omits its PDS service"))?;
        validate_service(
            pds,
            &format!("{did}#atproto_pds"),
            "AtprotoPersonalDataServer",
        )?;
        let pds_endpoint =
            required_string(pds, "serviceEndpoint", "atproto PDS service")?.to_owned();
        validate_https_origin(&pds_endpoint)?;

        let operation_log = services.get(1).ok_or_else(|| {
            anyhow::anyhow!("hosted DID document omits its operation-log service")
        })?;
        validate_service(operation_log, &format!("{did}#did-log"), "DidOperationLog")?;
        let operation_log_endpoint = required_string(
            operation_log,
            "serviceEndpoint",
            "DID operation-log service",
        )?;
        ensure!(
            operation_log_endpoint == format!("https://{host}{DID_OPERATION_LOG_PATH}"),
            "hosted DID document operation-log endpoint must be served by its own did:web host"
        );

        Ok(Self {
            did,
            handle: expected_handle,
            pds_endpoint,
            atproto_key,
            cid: Cid::from_raw(bytes),
            bytes: bytes.to_vec(),
        })
    }

    pub(crate) fn seal(
        name: &AllocatedAccountName,
        atproto_key: &VerifyingKey,
        pds_endpoint: &str,
    ) -> Result<Self> {
        validate_https_origin(pds_endpoint)?;
        let did = name.did();
        validate_host_form_did_web(did)?;
        let host = did
            .strip_prefix("did:web:")
            .ok_or_else(|| anyhow::anyhow!("hosted account DID must use did:web"))?;
        let handle = format!("at://{host}");
        let public_key_multibase = encode_p256_multibase(atproto_key);

        let document = json!({
            "@context": [DID_CONTEXT, MULTIKEY_CONTEXT],
            "alsoKnownAs": [handle],
            "id": did,
            "service": [
                {
                    "id": format!("{did}#atproto_pds"),
                    "serviceEndpoint": pds_endpoint,
                    "type": "AtprotoPersonalDataServer",
                },
                {
                    "id": format!("{did}#did-log"),
                    "serviceEndpoint": format!("https://{host}{DID_OPERATION_LOG_PATH}"),
                    "type": "DidOperationLog",
                },
            ],
            "verificationMethod": [{
                "controller": did,
                "id": format!("{did}#atproto"),
                "publicKeyMultibase": public_key_multibase,
                "type": "Multikey",
            }],
        });
        let bytes = canonical_json(&document)?;
        let sealed = Self::from_canonical_json(&bytes)?;
        ensure!(
            sealed.atproto_verifying_key()?.to_encoded_point(true)
                == atproto_key.to_encoded_point(true),
            "sealed DID document changed the mint's #atproto key"
        );
        Ok(sealed)
    }

    /// Immutable account DID named by this artifact.
    #[must_use]
    pub fn did(&self) -> &str {
        &self.did
    }

    /// Account handle, always `at://` plus the host-form DID authority.
    #[must_use]
    pub fn handle(&self) -> &str {
        &self.handle
    }

    /// Origin advertised by the atproto PDS service entry.
    #[must_use]
    pub fn pds_endpoint(&self) -> &str {
        &self.pds_endpoint
    }

    /// Generated account-specific P-256 key carried by `#atproto`.
    pub fn atproto_verifying_key(&self) -> Result<VerifyingKey> {
        VerifyingKey::from_sec1_bytes(&self.atproto_key)
            .context("sealed DID document contains an invalid #atproto P-256 key")
    }

    /// Canonical bytes to place in sealed storage and serve byte-for-byte.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Raw-codec CID over the exact canonical JSON bytes.
    #[must_use]
    pub fn cid(&self) -> Cid {
        self.cid
    }
}

fn encode_p256_multibase(key: &VerifyingKey) -> String {
    let point = key.to_encoded_point(true);
    let mut payload =
        Vec::with_capacity(P256_MULTICODEC_PREFIX.len() + COMPRESSED_P256_PUBLIC_KEY_LEN);
    payload.extend_from_slice(&P256_MULTICODEC_PREFIX);
    payload.extend_from_slice(point.as_bytes());
    format!("z{}", bs58::encode(payload).into_string())
}

fn decode_p256_multibase(encoded: &str) -> Result<Vec<u8>> {
    let payload = encoded
        .strip_prefix('z')
        .ok_or_else(|| anyhow::anyhow!("#atproto publicKeyMultibase must use base58btc"))?;
    let decoded = bs58::decode(payload)
        .into_vec()
        .context("#atproto publicKeyMultibase is not valid base58btc")?;
    ensure!(
        decoded.starts_with(&P256_MULTICODEC_PREFIX),
        "#atproto publicKeyMultibase must use the p256-pub multicodec"
    );
    let key = decoded[P256_MULTICODEC_PREFIX.len()..].to_vec();
    ensure!(
        key.len() == COMPRESSED_P256_PUBLIC_KEY_LEN,
        "#atproto P-256 key must be a compressed SEC1 point"
    );
    VerifyingKey::from_sec1_bytes(&key).context("#atproto P-256 key is invalid")?;
    Ok(key)
}

fn validate_service(value: &Value, id: &str, service_type: &str) -> Result<()> {
    validate_exact_keys(value, &["id", "serviceEndpoint", "type"], service_type)?;
    ensure!(
        required_string(value, "id", service_type)? == id,
        "{service_type} service id is invalid"
    );
    ensure!(
        required_string(value, "type", service_type)? == service_type,
        "{service_type} service type is invalid"
    );
    Ok(())
}

fn validate_https_origin(endpoint: &str) -> Result<()> {
    let parsed = url::Url::parse(endpoint).context("PDS service endpoint is not a valid URL")?;
    ensure!(
        parsed.scheme() == "https",
        "PDS service endpoint must use HTTPS"
    );
    ensure!(
        parsed.host().is_some(),
        "PDS service endpoint must have a host"
    );
    ensure!(
        parsed.username().is_empty() && parsed.password().is_none(),
        "PDS service endpoint must not contain credentials"
    );
    ensure!(
        parsed.path() == "/" && parsed.query().is_none() && parsed.fragment().is_none(),
        "PDS service endpoint must be an origin without path, query, or fragment"
    );
    ensure!(
        endpoint == parsed.origin().ascii_serialization(),
        "PDS service endpoint must use its canonical origin form"
    );
    Ok(())
}

fn validate_exact_keys(value: &Value, expected: &[&str], what: &str) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{what} must be a JSON object"))?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    ensure!(actual == expected, "{what} has missing or unknown fields");
    Ok(())
}

fn required_string<'a>(value: &'a Value, key: &str, what: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("{what} field {key:?} must be a string"))
}

fn required_array<'a>(value: &'a Value, key: &str, what: &str) -> Result<&'a [Value]> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| anyhow::anyhow!("{what} field {key:?} must be an array"))
}

/// Serialize the restricted DID-document value as RFC 8785-compatible JSON.
///
/// These documents contain only objects, arrays, strings, booleans, and null.
/// Rejecting numbers avoids silently implementing only part of JCS numeric
/// normalization if the document profile grows later.
fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    write_canonical_json(value, &mut output)?;
    Ok(output)
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(_) => bail!("hosted DID document canonical JSON does not permit numbers"),
        Value::String(string) => {
            serde_json::to_writer(&mut *output, string)
                .context("failed to serialize hosted DID document string")?;
        }
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(object) => write_canonical_object(object, output)?,
    }
    Ok(())
}

fn write_canonical_object(object: &Map<String, Value>, output: &mut Vec<u8>) -> Result<()> {
    output.push(b'{');
    let mut entries = object.iter().collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(key, _)| *key);
    for (index, (key, value)) in entries.into_iter().enumerate() {
        if index != 0 {
            output.push(b',');
        }
        serde_json::to_writer(&mut *output, key)
            .context("failed to serialize hosted DID document key")?;
        output.push(b':');
        write_canonical_json(value, output)?;
    }
    output.push(b'}');
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use p256::ecdsa::SigningKey;
    use rand::rngs::OsRng;
    use sha2::{Digest, Sha256};

    use super::*;

    fn artifact() -> SealedHostedDidDocument {
        let name = AllocatedAccountName::new("alice", "did:web:alice.acct.example.com").unwrap();
        let key = SigningKey::random(&mut OsRng);
        SealedHostedDidDocument::seal(&name, key.verifying_key(), "https://pds.example.com")
            .unwrap()
    }

    #[test]
    fn account_document_is_exact_atproto_profile_and_roundtrips() {
        let artifact = artifact();
        let document: Value = serde_json::from_slice(artifact.as_bytes()).unwrap();
        assert_eq!(document["id"], "did:web:alice.acct.example.com");
        assert_eq!(
            document["alsoKnownAs"],
            json!(["at://alice.acct.example.com"])
        );
        assert_eq!(document["verificationMethod"].as_array().unwrap().len(), 1);
        assert_eq!(document["verificationMethod"][0]["type"], "Multikey");
        assert_eq!(document["service"].as_array().unwrap().len(), 2);
        assert_eq!(document["service"][0]["type"], "AtprotoPersonalDataServer");
        assert_eq!(document["service"][1]["type"], "DidOperationLog");

        let parsed = SealedHostedDidDocument::from_canonical_json(artifact.as_bytes()).unwrap();
        assert_eq!(parsed, artifact);
        assert_eq!(parsed.cid(), Cid::from_raw(parsed.as_bytes()));
    }

    #[test]
    fn canonical_bytes_are_stable() {
        let name = AllocatedAccountName::new("alice", "did:web:alice.acct.example.com").unwrap();
        let key = SigningKey::from_slice(&[7; 32]).unwrap();
        let artifact =
            SealedHostedDidDocument::seal(&name, key.verifying_key(), "https://pds.example.com")
                .unwrap();
        let digest = hex::encode(Sha256::digest(artifact.as_bytes()));
        assert_eq!(
            digest,
            "f619b0958da09951c69b11f935f4077adf35cf7a6d84f003962955941e058ba5"
        );
    }

    #[test]
    fn account_document_rejects_host_capsule_alias_and_noncanonical_bytes() {
        let artifact = artifact();
        let mut document: Value = serde_json::from_slice(artifact.as_bytes()).unwrap();
        document["alsoKnownAs"] = json!(["did:at9p:bafk-host-capsule"]);
        let tampered = canonical_json(&document).unwrap();
        let error = SealedHostedDidDocument::from_canonical_json(&tampered).unwrap_err();
        assert!(error.to_string().contains("at://"), "{error:#}");

        let pretty = serde_json::to_vec_pretty(&document).unwrap();
        let error = SealedHostedDidDocument::from_canonical_json(&pretty).unwrap_err();
        assert!(error.to_string().contains("canonical"), "{error:#}");
    }

    #[test]
    fn account_document_rejects_wrong_key_shape_and_pds_endpoint() {
        let artifact = artifact();
        let mut document: Value = serde_json::from_slice(artifact.as_bytes()).unwrap();
        document["verificationMethod"][0]["publicKeyMultibase"] = json!("zBadKey");
        let tampered = canonical_json(&document).unwrap();
        assert!(SealedHostedDidDocument::from_canonical_json(&tampered).is_err());

        let name = AllocatedAccountName::new("alice", "did:web:alice.acct.example.com").unwrap();
        let key = SigningKey::random(&mut OsRng);
        for endpoint in [
            "http://pds.example.com",
            "https://pds.example.com/oauth",
            "https://user@pds.example.com",
        ] {
            assert!(
                SealedHostedDidDocument::seal(&name, key.verifying_key(), endpoint).is_err(),
                "{endpoint} must be rejected"
            );
        }
    }
}
