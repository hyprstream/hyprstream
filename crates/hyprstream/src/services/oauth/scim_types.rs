//! SCIM 2.0 types for JSON serialization.
//!
//! Serde structs for SCIM-compliant request/response bodies.
//! Conforms to RFC 7643 (Core Schema) and RFC 7644 (Protocol).

use serde::{Deserialize, Serialize};

pub const SCIM_API_LIST: &str = "urn:ietf:params:scim:api:messages:2.0:ListResponse";
pub const SCIM_API_ERROR: &str = "urn:ietf:params:scim:api:messages:2.0:Error";
pub const SCIM_SCHEMA_USER: &str = "urn:ietf:params:scim:schemas:core:2.0:User";
pub const SCIM_SCHEMA_LIST: &str = "urn:ietf:params:scim:api:messages:2.0:ListResponse";
pub const SCIM_SCHEMA_EXT_HYPRSTREAM: &str =
    "urn:ietf:params:scim:schemas:extension:hyprstream:1.0";

/// SCIM User resource (RFC 7643 §4.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimUser {
    pub schemas: Vec<String>,
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
    #[serde(rename = "userName")]
    pub user_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emails: Option<Vec<ScimEmail>>,
    pub meta: ScimMeta,
    #[serde(
        rename = "urn:ietf:params:scim:schemas:extension:hyprstream:1.0",
        skip_serializing_if = "Option::is_none"
    )]
    pub hyprstream: Option<ScimHyprstreamExtension>,
}

/// SCIM email sub-attribute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimEmail {
    pub value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary: Option<bool>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub email_type: Option<String>,
}

/// Hyprstream extension — Ed25519 public key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimHyprstreamExtension {
    pub pubkey_base64: String,
}

/// SCIM metadata (RFC 7643 §3.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimMeta {
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// SCIM ListResponse (RFC 7644 §3.4.2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimListResponse {
    pub schemas: Vec<String>,
    #[serde(rename = "totalResults")]
    pub total_results: u32,
    #[serde(rename = "itemsPerPage", skip_serializing_if = "Option::is_none")]
    pub items_per_page: Option<u32>,
    #[serde(rename = "startIndex", skip_serializing_if = "Option::is_none")]
    pub start_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<Vec<ScimUser>>,
}

/// SCIM error response (RFC 7644 §3.12).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimError {
    pub schemas: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub status: String,
    #[serde(rename = "scimType", skip_serializing_if = "Option::is_none")]
    pub scim_type: Option<String>,
}

/// SCIM Service Provider Configuration (RFC 7644 §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimServiceProviderConfig {
    pub schemas: Vec<String>,
    pub patch: ScimFeature,
    pub bulk: ScimFeatureWithMax,
    pub filter: ScimFilterFeature,
    pub change_password: ScimFeature,
    pub sort: ScimFeature,
    pub etag: ScimFeature,
    #[serde(rename = "authenticationSchemes")]
    pub authentication_schemes: Vec<ScimAuthScheme>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimFeature {
    pub supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimFeatureWithMax {
    pub supported: bool,
    #[serde(rename = "maxOperations")]
    pub max_operations: u32,
    #[serde(rename = "maxPayloadSize")]
    pub max_payload_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimFilterFeature {
    pub supported: bool,
    #[serde(rename = "maxResults")]
    pub max_results: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimAuthScheme {
    pub name: String,
    pub description: String,
    #[serde(rename = "specUri")]
    pub spec_uri: String,
    #[serde(rename = "type")]
    pub auth_type: String,
    pub primary: bool,
}

/// SCIM Resource Type (RFC 7644 §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimResourceType {
    pub schemas: Vec<String>,
    pub id: String,
    pub name: String,
    pub endpoint: String,
    pub description: String,
    #[serde(rename = "schema")]
    pub schema_: String,
}

/// SCIM Schema (RFC 7644 §4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimSchema {
    pub schemas: Vec<String>,
    pub id: String,
    pub name: String,
    pub description: String,
    pub attributes: Vec<ScimAttribute>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScimAttribute {
    pub name: String,
    #[serde(rename = "type")]
    pub attr_type: String,
    pub multi_valued: bool,
    pub required: bool,
    pub case_exact: bool,
    pub mutability: String,
    pub returned: String,
    pub uniqueness: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub_attributes: Option<Vec<ScimAttribute>>,
}
