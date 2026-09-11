//! Validation of the two enabled posting schemas and their pinned reference
//! closure. No schema discovery, permissions, network fetches or data rewriting.
use std::collections::BTreeMap;
use std::sync::LazyLock;

use chrono::Datelike as _;
use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation as _;

#[derive(Debug)]
pub(super) enum Error {
    Invalid,
    SchemaUnavailable,
}

type Result<T> = std::result::Result<T, Error>;

macro_rules! schema {
    ($path:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../lexicons/upstream/atproto/",
            $path,
            ".json"
        ))
    };
}

static SCHEMAS: LazyLock<Result<BTreeMap<String, Value>>> = LazyLock::new(|| {
    [
        schema!("app/bsky/feed/post"),
        schema!("app/bsky/actor/profile"),
        schema!("app/bsky/embed/defs"),
        schema!("app/bsky/embed/external"),
        schema!("app/bsky/embed/gallery"),
        schema!("app/bsky/embed/images"),
        schema!("app/bsky/embed/record"),
        schema!("app/bsky/embed/recordWithMedia"),
        schema!("app/bsky/embed/video"),
        schema!("app/bsky/richtext/facet"),
        schema!("com/atproto/label/defs"),
        schema!("com/atproto/repo/strongRef"),
    ]
    .into_iter()
    .map(|text| {
        let value: Value = serde_json::from_str(text).map_err(|_| Error::SchemaUnavailable)?;
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .ok_or(Error::SchemaUnavailable)?
            .to_owned();
        Ok((id, value))
    })
    .collect()
});

pub(super) fn validate(collection: &str, value: &Value) -> Result<()> {
    let schemas = SCHEMAS.as_ref().map_err(|_| Error::SchemaUnavailable)?;
    validate_ref(schemas, collection, value, 0)
}

fn validate_ref(
    schemas: &BTreeMap<String, Value>,
    reference: &str,
    value: &Value,
    depth: usize,
) -> Result<()> {
    let (nsid, name) = reference.split_once('#').unwrap_or((reference, "main"));
    let schema = schemas
        .get(nsid)
        .and_then(|s| s.get("defs"))
        .and_then(|s| s.get(name))
        .ok_or(Error::SchemaUnavailable)?;
    if schema.get("type").and_then(Value::as_str) == Some("record") {
        check(value.get("$type").and_then(Value::as_str) == Some(nsid))?;
        validate_value(
            schemas,
            nsid,
            schema.get("record").ok_or(Error::SchemaUnavailable)?,
            value,
            depth,
        )
    } else {
        validate_value(schemas, nsid, schema, value, depth)
    }
}

fn check(valid: bool) -> Result<()> {
    if valid {
        Ok(())
    } else {
        Err(Error::Invalid)
    }
}

fn bounds(schema: &Value, minimum: &str, maximum: &str, size: usize) -> Result<()> {
    check(
        schema
            .get(minimum)
            .and_then(Value::as_u64)
            .is_none_or(|n| size as u64 >= n)
            && schema
                .get(maximum)
                .and_then(Value::as_u64)
                .is_none_or(|n| size as u64 <= n),
    )
}

fn resolve(nsid: &str, reference: &str) -> String {
    if reference.starts_with('#') {
        format!("{nsid}{reference}")
    } else {
        reference.to_owned()
    }
}

fn validate_value(
    schemas: &BTreeMap<String, Value>,
    nsid: &str,
    schema: &Value,
    value: &Value,
    depth: usize,
) -> Result<()> {
    check(depth <= 128)?;
    check(schema.get("const").is_none_or(|expected| expected == value))?;
    check(
        schema
            .get("enum")
            .and_then(Value::as_array)
            .is_none_or(|allowed| allowed.contains(value)),
    )?;
    match schema
        .get("type")
        .and_then(Value::as_str)
        .ok_or(Error::SchemaUnavailable)?
    {
        "object" => {
            let object = value.as_object().ok_or(Error::Invalid)?;
            if let Some(required) = schema.get("required").and_then(Value::as_array) {
                for key in required {
                    check(object.contains_key(key.as_str().ok_or(Error::SchemaUnavailable)?))?;
                }
            }
            for (key, field) in schema
                .get("properties")
                .and_then(Value::as_object)
                .ok_or(Error::SchemaUnavailable)?
            {
                if let Some(value) = object.get(key) {
                    let nullable = schema
                        .get("nullable")
                        .and_then(Value::as_array)
                        .is_some_and(|fields| fields.contains(&Value::String(key.clone())));
                    if !(nullable && value.is_null()) {
                        validate_value(schemas, nsid, field, value, depth + 1)?;
                    }
                }
            }
            // Lexicon objects are extensible. Unrecognized properties still
            // pass through the existing complete AT data-model/codec checks.
            Ok(())
        }
        "array" => {
            let items = value.as_array().ok_or(Error::Invalid)?;
            bounds(schema, "minLength", "maxLength", items.len())?;
            let item_schema = schema.get("items").ok_or(Error::SchemaUnavailable)?;
            for item in items {
                validate_value(schemas, nsid, item_schema, item, depth + 1)?;
            }
            Ok(())
        }
        "string" => {
            let text = value.as_str().ok_or(Error::Invalid)?;
            bounds(schema, "minLength", "maxLength", text.len())?;
            bounds(
                schema,
                "minGraphemes",
                "maxGraphemes",
                text.graphemes(true).count(),
            )?;
            match schema.get("format").and_then(Value::as_str) {
                Some(format) => check(valid_format(format, text)),
                None => Ok(()),
            }
        }
        "integer" => {
            let n = value.as_i64().ok_or(Error::Invalid)?;
            check(
                schema
                    .get("minimum")
                    .and_then(Value::as_i64)
                    .is_none_or(|min| n >= min)
                    && schema
                        .get("maximum")
                        .and_then(Value::as_i64)
                        .is_none_or(|max| n <= max),
            )
        }
        "ref" => {
            let reference = schema
                .get("ref")
                .and_then(Value::as_str)
                .ok_or(Error::SchemaUnavailable)?;
            validate_ref(schemas, &resolve(nsid, reference), value, depth + 1)
        }
        "union" => {
            let discriminator = value
                .get("$type")
                .and_then(Value::as_str)
                .ok_or(Error::Invalid)?;
            let (id, fragment) = discriminator.split_once('#').unwrap_or((discriminator, ""));
            check(
                valid_nsid(id)
                    && (fragment.is_empty()
                        || (fragment != "main"
                            && fragment
                                .bytes()
                                .all(|b| b.is_ascii_alphanumeric() || b == b'_'))),
            )?;
            let refs = schema
                .get("refs")
                .and_then(Value::as_array)
                .ok_or(Error::SchemaUnavailable)?;
            for reference in refs {
                let reference = resolve(nsid, reference.as_str().ok_or(Error::SchemaUnavailable)?);
                if reference == discriminator {
                    return validate_ref(schemas, &reference, value, depth + 1);
                }
            }
            // Open unions accept future variants, without claiming their schema
            // is known. Closed unions cannot introduce new alternatives.
            check(schema.get("closed").and_then(Value::as_bool) != Some(true))
        }
        "blob" => {
            check(value.get("$type").and_then(Value::as_str) == Some("blob"))?;
            let link = value.get("ref").ok_or(Error::Invalid)?;
            let cid =
                crate::services::public_repo::json_to_dag_cbor(link).map_err(|_| Error::Invalid)?;
            check(
                matches!(cid, hyprstream_pds::dag_cbor::DagCbor::Link(cid) if cid.as_bytes().get(1) == Some(&0x55)),
            )?;
            let size = value
                .get("size")
                .and_then(Value::as_u64)
                .ok_or(Error::Invalid)?;
            check(
                size > 0
                    && schema
                        .get("maxSize")
                        .and_then(Value::as_u64)
                        .is_none_or(|max| size <= max),
            )?;
            let mime = value
                .get("mimeType")
                .and_then(Value::as_str)
                .ok_or(Error::Invalid)?;
            check(
                !mime.is_empty()
                    && schema
                        .get("accept")
                        .and_then(Value::as_array)
                        .is_none_or(|types| {
                            types.iter().any(|t| {
                                t.as_str().is_some_and(|t| {
                                    t == "*/*"
                                        || t == mime
                                        || t.strip_suffix("/*").is_some_and(|prefix| {
                                            mime.strip_prefix(prefix).is_some_and(|rest| {
                                                rest.starts_with('/') && rest.len() > 1
                                            })
                                        })
                                })
                            })
                        }),
            )
        }
        _ => Err(Error::SchemaUnavailable),
    }
}

fn valid_nsid(value: &str) -> bool {
    let Some((authority, name)) = value.rsplit_once('.') else {
        return false;
    };
    value.len() <= 317
        && valid_dns_labels(authority)
        && authority
            .as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphabetic)
        && (1..=63).contains(&name.len())
        && name.as_bytes().first().is_some_and(u8::is_ascii_alphabetic)
        && name.bytes().all(|b| b.is_ascii_alphanumeric())
}

fn valid_handle(value: &str) -> bool {
    valid_dns_labels(value)
        && value
            .rsplit('.')
            .next()
            .is_some_and(|tld| tld.as_bytes().first().is_some_and(u8::is_ascii_alphabetic))
}

fn valid_dns_labels(value: &str) -> bool {
    value.len() <= 253
        && value.contains('.')
        && value.split('.').all(|part| {
            (1..=63).contains(&part.len())
                && !part.starts_with('-')
                && !part.ends_with('-')
                && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
}

fn valid_did(value: &str) -> bool {
    let Some((method, id)) = value.strip_prefix("did:").and_then(|s| s.split_once(':')) else {
        return false;
    };
    value.len() <= 2048
        && !method.is_empty()
        && method.bytes().all(|b| b.is_ascii_lowercase())
        && !id.is_empty()
        && !id.ends_with(':')
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:%-".contains(&b))
}

fn valid_at_uri(value: &str) -> bool {
    let Some(rest) = value.strip_prefix("at://") else {
        return false;
    };
    let mut parts = rest.split('/');
    let authority = parts.next().unwrap_or_default();
    value.len() <= 8192
        && (valid_did(authority) || valid_handle(authority))
        && parts.next().is_none_or(valid_nsid)
        && parts
            .next()
            .is_none_or(|key| hyprstream_pds::atproto_cbor::AtprotoRecordKey::new(key).is_ok())
        && parts.next().is_none()
}

fn valid_format(format: &str, text: &str) -> bool {
    match format {
        "cid" => {
            crate::services::public_repo::json_to_dag_cbor(&serde_json::json!({"$link": text}))
                .is_ok()
        }
        "did" => valid_did(text),
        "at-uri" => valid_at_uri(text),
        "uri" => {
            text.len() <= 8192 && (valid_at_uri(text) || uriparse::URI::try_from(text).is_ok())
        }
        "language" => language_tags::LanguageTag::parse(text).is_ok(),
        "datetime" => {
            static DATETIME: LazyLock<std::result::Result<regex::Regex, regex::Error>> =
                LazyLock::new(|| {
                    regex::Regex::new(
                        r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?(Z|[+-][0-9]{2}:[0-9]{2})$",
                    )
                });
            !text.ends_with("-00:00")
                && DATETIME
                    .as_ref()
                    .is_ok_and(|pattern| pattern.is_match(text))
                && chrono::DateTime::parse_from_rfc3339(text)
                    .is_ok_and(|dt| dt.with_timezone(&chrono::Utc).year() >= 0)
        }
        _ => false,
    }
}
