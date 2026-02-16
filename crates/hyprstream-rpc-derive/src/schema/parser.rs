//! Cap'n Proto schema text parser.

use super::types::*;
use crate::util::{to_pascal_case, to_snake_case};

pub fn parse_capnp_schema(text: &str, service_name: &str) -> Option<ParsedSchema> {
    let pascal = to_pascal_case(service_name);
    let request_struct_name = format!("{pascal}Request");
    let response_struct_name = format!("{pascal}Response");

    let structs_parsed = parse_all_structs(text);
    let enums = parse_all_enums(text);

    let has_request = structs_parsed.iter().any(|s| s.name == request_struct_name);
    let has_response = structs_parsed.iter().any(|s| s.name == response_struct_name);
    if !has_request || !has_response {
        return None;
    }

    let request_variants = parse_union_variants(text, &request_struct_name);
    let response_variants = parse_union_variants(text, &response_struct_name);

    if request_variants.is_empty() || response_variants.is_empty() {
        return None;
    }

    // Detect scoped clients from nested union patterns
    let mut scoped_clients = Vec::new();
    for req_variant in &request_variants {
        if matches!(
            req_variant.type_name.as_str(),
            "Void" | "Text" | "Data" | "Bool"
                | "UInt32" | "UInt64" | "Int32" | "Int64" | "Float32" | "Float64"
        ) || req_variant.type_name.starts_with("List(")
        {
            continue;
        }

        if let Some(inner_struct) = structs_parsed
            .iter()
            .find(|s| s.name == req_variant.type_name)
        {
            if inner_struct.has_union {
                let response_variant_name = format!("{}Result", req_variant.name);
                if let Some(resp_variant) = response_variants
                    .iter()
                    .find(|v| v.name == response_variant_name)
                {
                    let inner_req_variants =
                        parse_union_variants(text, &req_variant.type_name);
                    let inner_resp_variants =
                        parse_union_variants(text, &resp_variant.type_name);

                    if !inner_req_variants.is_empty() && !inner_resp_variants.is_empty() {
                        let client_name = if req_variant.type_name.ends_with("Request") {
                            format!(
                                "{}Client",
                                &req_variant.type_name
                                    [..req_variant.type_name.len() - 7]
                            )
                        } else {
                            format!("{}Client", req_variant.type_name)
                        };

                        scoped_clients.push(ScopedClient {
                            factory_name: req_variant.name.clone(),
                            client_name,
                            scope_fields: inner_struct.fields.clone(),
                            inner_request_variants: inner_req_variants,
                            inner_response_variants: inner_resp_variants,
                            capnp_inner_response: to_snake_case(&resp_variant.type_name),
                            nested_clients: Vec::new(),
                            parent_scope_fields: Vec::new(),
                            parent_factory_name: None,
                            parent_capnp_inner_response: None,
                        });
                    }
                }
            }
        }
    }

    // Recursively detect nested scoped clients (3rd level, e.g., Fs within Repository)
    for sc in &mut scoped_clients {
        detect_nested_scoped_clients(text, sc, &structs_parsed);
    }

    let referenced: Vec<StructDef> = structs_parsed
        .into_iter()
        .filter(|s| s.name != request_struct_name && s.name != response_struct_name)
        .collect();

    Some(ParsedSchema {
        request_variants,
        response_variants,
        structs: referenced,
        scoped_clients,
        enums,
    })
}

/// Recursively detect nested scoped clients within a scoped client.
///
/// For each inner request variant, check if it points to a struct with
/// `has_union && !fields.is_empty()` — the same pattern as top-level detection.
/// Detected nested clients are added to `parent.nested_clients` and their
/// variants are removed from `parent.inner_request_variants`.
fn detect_nested_scoped_clients(
    text: &str,
    parent: &mut ScopedClient,
    all_structs: &[StructDef],
) {
    let mut nested = Vec::new();
    let mut nested_factory_names = Vec::new();

    for req_variant in &parent.inner_request_variants {
        // Skip primitives
        if matches!(
            req_variant.type_name.as_str(),
            "Void" | "Text" | "Data" | "Bool"
                | "UInt32" | "UInt64" | "Int32" | "Int64" | "Float32" | "Float64"
        ) || req_variant.type_name.starts_with("List(")
        {
            continue;
        }

        if let Some(inner_struct) = all_structs.iter().find(|s| s.name == req_variant.type_name) {
            if inner_struct.has_union {
                let response_variant_name = format!("{}Result", req_variant.name);
                if let Some(resp_variant) = parent
                    .inner_response_variants
                    .iter()
                    .find(|v| v.name == response_variant_name)
                {
                    let inner_req_variants =
                        parse_union_variants(text, &req_variant.type_name);
                    let inner_resp_variants =
                        parse_union_variants(text, &resp_variant.type_name);

                    if !inner_req_variants.is_empty() && !inner_resp_variants.is_empty() {
                        let client_name = if req_variant.type_name.ends_with("Request") {
                            format!(
                                "{}Client",
                                &req_variant.type_name
                                    [..req_variant.type_name.len() - 7]
                            )
                        } else {
                            format!("{}Client", req_variant.type_name)
                        };

                        nested_factory_names.push(req_variant.name.clone());
                        nested.push(ScopedClient {
                            factory_name: req_variant.name.clone(),
                            client_name,
                            scope_fields: inner_struct.fields.clone(),
                            inner_request_variants: inner_req_variants,
                            inner_response_variants: inner_resp_variants,
                            capnp_inner_response: to_snake_case(&resp_variant.type_name),
                            nested_clients: Vec::new(),
                            parent_scope_fields: parent.scope_fields.clone(),
                            parent_factory_name: Some(parent.factory_name.clone()),
                            parent_capnp_inner_response: Some(parent.capnp_inner_response.clone()),
                        });
                    }
                }
            }
        }
    }

    // Filter out nested-scope variants from parent's inner variants
    if !nested_factory_names.is_empty() {
        parent.inner_request_variants.retain(|v| !nested_factory_names.contains(&v.name));
        parent.inner_response_variants.retain(|v| {
            let result_name = v.name.strip_suffix("Result").unwrap_or(&v.name);
            !nested_factory_names.contains(&result_name.to_string())
        });
    }

    parent.nested_clients = nested;
}

pub fn parse_all_structs(text: &str) -> Vec<StructDef> {
    let mut structs = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("struct ") && line.contains('{') && line.contains('}') {
            // Single-line struct: struct Foo { field @0 :Type; ... }
            let Some(brace_open) = line.find('{') else { i += 1; continue; };
            let Some(brace_close) = line.rfind('}') else { i += 1; continue; };
            let name = line["struct ".len()..brace_open].trim().to_string();
            let body = line[brace_open + 1..brace_close].trim();

            let mut fields = Vec::new();
            let has_union = body.contains("union");

            if !has_union && !body.is_empty() {
                for part in body.split(';') {
                    let part = part.trim();
                    if !part.is_empty() && !part.starts_with('#') {
                        if let Some(field) = parse_field_line(part) {
                            fields.push(field);
                        }
                    }
                }
            }

            structs.push(StructDef { name, fields, has_union, domain_type: None, origin_file: None });
            i += 1;
        } else if line.starts_with("struct ") && line.ends_with('{') {
            // Multi-line struct
            let name = line["struct ".len()..line.len() - 1].trim().to_string();

            let mut fields = Vec::new();
            let mut has_union = false;
            i += 1;

            let mut depth = 1;
            let mut in_union = false;
            while i < lines.len() && depth > 0 {
                let inner = lines[i].trim();
                if inner.contains('{') {
                    depth += 1;
                    if inner.starts_with("union") && depth == 2 {
                        has_union = true;
                    }
                    if inner.starts_with("union") {
                        in_union = true;
                    }
                }
                if inner.contains('}') {
                    depth -= 1;
                    if depth == 1 {
                        in_union = false;
                    }
                }

                if !in_union
                    && depth == 1
                    && !inner.starts_with('#')
                    && !inner.is_empty()
                    && !inner.starts_with('}')
                    && !inner.starts_with("union")
                {
                    if let Some(field) = parse_field_line(inner) {
                        fields.push(field);
                    }
                }
                i += 1;
            }

            structs.push(StructDef {
                name,
                fields,
                has_union,
                domain_type: None,
                origin_file: None,
            });
        } else {
            i += 1;
        }
    }

    structs
}

pub fn parse_all_enums(text: &str) -> Vec<EnumDef> {
    let mut enums = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("enum ") && line.ends_with('{') {
            let name = line["enum ".len()..line.len() - 1].trim().to_string();

            let mut variants = Vec::new();
            i += 1;

            while i < lines.len() {
                let inner = lines[i].trim();
                if inner == "}" {
                    break;
                }
                if !inner.starts_with('#') && !inner.is_empty() && inner.contains('@') {
                    let inner = inner
                        .split('#')
                        .next()
                        .unwrap_or(inner)
                        .trim()
                        .trim_end_matches(';')
                        .trim();
                    if let Some(at_pos) = inner.find('@') {
                        let variant_name = inner[..at_pos].trim().to_string();
                        let ordinal_str = inner[at_pos + 1..].trim();
                        if let Ok(ordinal) = ordinal_str.parse::<u32>() {
                            variants.push((variant_name, ordinal));
                        }
                    }
                }
                i += 1;
            }

            enums.push(EnumDef { name, variants, origin_file: None });
        }
        i += 1;
    }

    enums
}

pub fn parse_field_line(line: &str) -> Option<FieldDef> {
    // Extract inline comment if present
    let (field_part, comment) = if let Some(hash_pos) = line.find('#') {
        let field = &line[..hash_pos];
        let comment = line[hash_pos + 1..].trim().to_string();
        (field, comment)
    } else {
        (line, String::new())
    };

    let mut field_part = field_part.trim().trim_end_matches(';').trim();

    // Extract $fixedSize(N) before stripping annotations
    let fixed_size = if let Some(dollar_pos) = field_part.find("$fixedSize(") {
        let start = dollar_pos + "$fixedSize(".len();
        field_part[start..].find(')').and_then(|p| {
            field_part[start..start + p].trim().parse::<u32>().ok()
        })
    } else {
        None
    };

    // Strip out Cap'n Proto annotations (e.g., $mcpDescription("..."))
    // Annotations are in format: $annotationName("value")
    if let Some(dollar_pos) = field_part.find('$') {
        field_part = field_part[..dollar_pos].trim();
    }

    if field_part.is_empty() {
        return None;
    }

    let at_pos = field_part.find('@')?;
    let colon_pos = field_part.find(':')?;
    if colon_pos <= at_pos {
        return None;
    }

    let name = field_part[..at_pos].trim().to_string();
    let ordinal_str = field_part[at_pos + 1..colon_pos].trim();
    let _ordinal: u32 = ordinal_str.parse().ok()?;
    let type_name = field_part[colon_pos + 1..].trim().to_string();

    Some(FieldDef { name, type_name, description: comment, fixed_size })
}

pub fn parse_union_variants(text: &str, struct_name: &str) -> Vec<UnionVariant> {
    let mut variants = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    // Find the struct
    while i < lines.len() {
        let line = lines[i].trim();
        if line == format!("struct {struct_name} {{") {
            i += 1;
            break;
        }
        i += 1;
    }

    // Find the union block inside this struct
    let mut depth = 1;
    while i < lines.len() && depth > 0 {
        let line = lines[i].trim();

        if line == "union {" {
            i += 1;
            let mut union_depth = 1;
            let mut pending_comment = String::new();
            while i < lines.len() && union_depth > 0 {
                let inner = lines[i].trim();
                if inner.contains('{') {
                    union_depth += 1;
                }
                if inner.contains('}') {
                    union_depth -= 1;
                    if union_depth == 0 {
                        break;
                    }
                }

                // Collect comment lines
                if inner.starts_with('#') {
                    let comment_text = inner[1..].trim();
                    if !pending_comment.is_empty() {
                        pending_comment.push(' ');
                    }
                    pending_comment.push_str(comment_text);
                } else if !inner.is_empty() && inner.contains('@') {
                    // Variant line - use pending comment as description
                    if let Some(mut variant) = parse_variant_line(inner) {
                        variant.description = pending_comment.clone();
                        variants.push(variant);
                    }
                    pending_comment.clear();
                } else if inner.is_empty() {
                    // Blank line resets comment accumulation
                    pending_comment.clear();
                }
                i += 1;
            }
            break;
        }

        if line.contains('}') {
            depth -= 1;
        }
        if line.contains('{') && !line.starts_with("struct") {
            depth += 1;
        }
        i += 1;
    }

    variants
}

pub fn parse_variant_line(line: &str) -> Option<UnionVariant> {
    // Don't extract inline comments here - they're handled by caller
    let line = line.split('#').next().unwrap_or(line);
    let mut line = line.trim().trim_end_matches(';').trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // Strip out Cap'n Proto annotations (e.g., $mcpDescription("..."))
    if let Some(dollar_pos) = line.find('$') {
        line = line[..dollar_pos].trim();
    }

    let at_pos = line.find('@')?;
    let colon_pos = line.find(':')?;

    let name = line[..at_pos].trim().to_string();
    let ordinal_str = line[at_pos + 1..colon_pos].trim();
    let _ordinal: u32 = ordinal_str.parse().ok()?;
    let type_name = line[colon_pos + 1..].trim().to_string();

    Some(UnionVariant { name, type_name, description: String::new(), scope: String::new(), cli_hidden: false })
}

/// Collect all struct names that need Data structs generated.
pub fn collect_list_struct_types(schema: &ParsedSchema) -> Vec<String> {
    let mut types = Vec::new();
    let is_primitive = |name: &str| {
        matches!(
            name,
            "Text"
                | "Data"
                | "Bool"
                | "Void"
                | "UInt8"
                | "UInt16"
                | "UInt32"
                | "UInt64"
                | "Int8"
                | "Int16"
                | "Int32"
                | "Int64"
                | "Float32"
                | "Float64"
        )
    };

    let mut add_type = |type_name: &str| {
        if type_name.starts_with("List(") {
            let inner = &type_name[5..type_name.len() - 1];
            if !is_primitive(inner)
                && !types.contains(&inner.to_string())
                && schema.enums.iter().all(|e| e.name != inner)
            {
                types.push(inner.to_string());
            }
        } else if !is_primitive(type_name)
            && !type_name.starts_with("List(")
            && schema.enums.iter().all(|e| e.name != type_name)
            && schema.structs.iter().any(|s| s.name == type_name)
            && !types.contains(&type_name.to_string())
        {
            types.push(type_name.to_string());
        }
    };

    for v in &schema.response_variants {
        add_type(&v.type_name);
    }
    for sc in &schema.scoped_clients {
        for v in &sc.inner_response_variants {
            add_type(&v.type_name);
        }
    }
    for s in &schema.structs {
        for f in &s.fields {
            add_type(&f.type_name);
        }
    }
    for v in &schema.request_variants {
        add_type(&v.type_name);
        if let Some(s) = schema.structs.iter().find(|s| s.name == v.type_name) {
            for f in &s.fields {
                add_type(&f.type_name);
            }
        }
    }
    for sc in &schema.scoped_clients {
        for v in &sc.inner_request_variants {
            add_type(&v.type_name);
            if let Some(s) = schema.structs.iter().find(|s| s.name == v.type_name) {
                for f in &s.fields {
                    add_type(&f.type_name);
                }
            }
        }
        // Nested scoped clients
        for nested in &sc.nested_clients {
            for v in &nested.inner_response_variants {
                add_type(&v.type_name);
            }
            for v in &nested.inner_request_variants {
                add_type(&v.type_name);
                if let Some(s) = schema.structs.iter().find(|s| s.name == v.type_name) {
                    for f in &s.fields {
                        add_type(&f.type_name);
                    }
                }
            }
        }
    }

    types
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY_SCHEMA: &str = r#"
@0xf1a2b3c4d5e6f708;

struct PolicyRequest {
  id @0 :UInt64;
  union {
    check @1 :PolicyCheck;
    issueToken @2 :IssueToken;
  }
}

struct PolicyCheck {
  subject @0 :Text;
  domain @1 :Text;
  resource @2 :Text;
  operation @3 :Text;
}

struct IssueToken {
  requestedScopes @0 :List(Text);
  ttl @1 :UInt32;
  audience @2 :Text;
  subject @3 :Text;
}

struct PolicyResponse {
  requestId @0 :UInt64;
  union {
    allowed @1 :Bool;
    error @2 :ErrorInfo;
    tokenSuccess @3 :TokenInfo;
  }
}

struct TokenInfo {
  token @0 :Text;
  expiresAt @1 :Int64;
}

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}
"#;

    const ENUM_SCHEMA: &str = r#"
@0xabcdef1234567890;

enum Status {
  active @0;
  inactive @1;
  pending @2;
}

struct ModelRequest {
  id @0 :UInt64;
  union {
    getInfo @1 :Void;
    setStatus @2 :Text;
  }
}

struct ModelResponse {
  requestId @0 :UInt64;
  union {
    getInfoResult @1 :ModelInfo;
    setStatusResult @2 :Void;
  }
}

struct ModelInfo {
  name @0 :Text;
  status @1 :Status;
  tags @2 :List(Text);
}
"#;

    const SCOPED_SCHEMA: &str = r#"
@0x1234567890abcdef;

struct RegistryRequest {
  id @0 :UInt64;
  union {
    list @1 :Void;
    repo @2 :RepositoryRequest;
  }
}

struct RepositoryRequest {
  repoId @0 :Text;
  union {
    clone @1 :CloneParams;
    delete @2 :Void;
  }
}

struct CloneParams {
  url @0 :Text;
  shallow @1 :Bool;
}

struct RegistryResponse {
  requestId @0 :UInt64;
  union {
    listResult @1 :List(Text);
    repoResult @2 :RepositoryResponse;
    error @3 :ErrorInfo;
  }
}

struct RepositoryResponse {
  union {
    cloneResult @0 :Text;
    deleteResult @1 :Void;
  }
}

struct ErrorInfo {
  message @0 :Text;
}
"#;

    #[test]
    fn parse_policy_schema() {
        let schema = parse_capnp_schema(POLICY_SCHEMA, "policy").unwrap();
        assert_eq!(schema.request_variants.len(), 2);
        assert_eq!(schema.request_variants[0].name, "check");
        assert_eq!(schema.request_variants[0].type_name, "PolicyCheck");
        assert_eq!(schema.request_variants[1].name, "issueToken");
        assert_eq!(schema.request_variants[1].type_name, "IssueToken");

        assert_eq!(schema.response_variants.len(), 3);
        assert_eq!(schema.response_variants[0].name, "allowed");
        assert_eq!(schema.response_variants[0].type_name, "Bool");
        assert_eq!(schema.response_variants[1].name, "error");
        assert_eq!(schema.response_variants[2].name, "tokenSuccess");
    }

    #[test]
    fn parse_struct_fields() {
        let schema = parse_capnp_schema(POLICY_SCHEMA, "policy").unwrap();
        let check_struct = schema.structs.iter().find(|s| s.name == "PolicyCheck").unwrap();
        assert_eq!(check_struct.fields.len(), 4);
        assert_eq!(check_struct.fields[0].name, "subject");
        assert_eq!(check_struct.fields[0].type_name, "Text");
        assert_eq!(check_struct.fields[3].name, "operation");

        let issue_struct = schema.structs.iter().find(|s| s.name == "IssueToken").unwrap();
        assert_eq!(issue_struct.fields.len(), 4);
        assert_eq!(issue_struct.fields[0].type_name, "List(Text)");
        assert_eq!(issue_struct.fields[1].type_name, "UInt32");
        assert_eq!(issue_struct.fields[2].type_name, "Text");
        assert_eq!(issue_struct.fields[3].type_name, "Text");
    }

    #[test]
    fn parse_enums() {
        let enums = parse_all_enums(ENUM_SCHEMA);
        assert_eq!(enums.len(), 1);
        assert_eq!(enums[0].name, "Status");
        assert_eq!(enums[0].variants.len(), 3);
        assert_eq!(enums[0].variants[0], ("active".into(), 0));
        assert_eq!(enums[0].variants[1], ("inactive".into(), 1));
        assert_eq!(enums[0].variants[2], ("pending".into(), 2));
    }

    #[test]
    fn parse_schema_with_enums() {
        let schema = parse_capnp_schema(ENUM_SCHEMA, "model").unwrap();
        assert_eq!(schema.enums.len(), 1);
        assert_eq!(schema.enums[0].name, "Status");
        assert_eq!(schema.request_variants.len(), 2);
        assert_eq!(schema.response_variants.len(), 2);

        let info_struct = schema.structs.iter().find(|s| s.name == "ModelInfo").unwrap();
        assert_eq!(info_struct.fields.len(), 3);
        assert_eq!(info_struct.fields[1].type_name, "Status");
        assert_eq!(info_struct.fields[2].type_name, "List(Text)");
    }

    #[test]
    fn parse_scoped_clients() {
        let schema = parse_capnp_schema(SCOPED_SCHEMA, "registry").unwrap();
        assert_eq!(schema.scoped_clients.len(), 1);

        let sc = &schema.scoped_clients[0];
        assert_eq!(sc.factory_name, "repo");
        assert_eq!(sc.client_name, "RepositoryClient");
        assert_eq!(sc.scope_fields.len(), 1);
        assert_eq!(sc.scope_fields[0].name, "repoId");
        assert_eq!(sc.scope_fields[0].type_name, "Text");

        assert_eq!(sc.inner_request_variants.len(), 2);
        assert_eq!(sc.inner_request_variants[0].name, "clone");
        assert_eq!(sc.inner_request_variants[0].type_name, "CloneParams");
        assert_eq!(sc.inner_request_variants[1].name, "delete");
        assert_eq!(sc.inner_request_variants[1].type_name, "Void");

        assert_eq!(sc.inner_response_variants.len(), 2);
        assert_eq!(sc.inner_response_variants[0].name, "cloneResult");
        assert_eq!(sc.inner_response_variants[1].name, "deleteResult");
    }

    #[test]
    fn parse_all_structs_counts() {
        let structs = parse_all_structs(POLICY_SCHEMA);
        // PolicyRequest, PolicyCheck, IssueToken, PolicyResponse, TokenInfo, ErrorInfo
        assert_eq!(structs.len(), 6);
        let names: Vec<&str> = structs.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"PolicyRequest"));
        assert!(names.contains(&"PolicyCheck"));
        assert!(names.contains(&"TokenInfo"));
    }

    #[test]
    fn parse_union_variants_from_struct() {
        let variants = parse_union_variants(POLICY_SCHEMA, "PolicyRequest");
        assert_eq!(variants.len(), 2);
        assert_eq!(variants[0].name, "check");
        assert_eq!(variants[1].name, "issueToken");
    }

    #[test]
    fn parse_variant_line_basic() {
        assert!(parse_variant_line("check @1 :PolicyCheck;").is_some());
        let v = parse_variant_line("check @1 :PolicyCheck;").unwrap();
        assert_eq!(v.name, "check");
        assert_eq!(v.type_name, "PolicyCheck");
    }

    #[test]
    fn parse_field_line_basic() {
        let f = parse_field_line("subject @0 :Text;").unwrap();
        assert_eq!(f.name, "subject");
        assert_eq!(f.type_name, "Text");

        let f = parse_field_line("ttl @1 :UInt32;").unwrap();
        assert_eq!(f.name, "ttl");
        assert_eq!(f.type_name, "UInt32");
    }

    #[test]
    fn parse_field_with_comment() {
        let f = parse_field_line("token @0 :Text; # Signed JWT").unwrap();
        assert_eq!(f.name, "token");
        assert_eq!(f.type_name, "Text");
    }

    #[test]
    fn parse_returns_none_for_missing_schema() {
        let result = parse_capnp_schema("struct Foo { }", "policy");
        assert!(result.is_none());
    }

    #[test]
    fn collect_list_struct_types_finds_referenced() {
        let schema = parse_capnp_schema(ENUM_SCHEMA, "model").unwrap();
        let types = collect_list_struct_types(&schema);
        assert!(types.contains(&"ModelInfo".to_string()));
    }

    const NESTED_SCOPED_SCHEMA: &str = r#"
@0xabcdef0123456789;

struct RegistryRequest {
  id @0 :UInt64;
  union {
    list @1 :Void;
    repo @2 :RepositoryRequest;
  }
}

struct RepositoryRequest {
  repoId @0 :Text;
  union {
    fs @1 :FsRequest;
    delete @2 :Void;
  }
}

struct FsRequest {
  worktree @0 :Text;
  union {
    open @1 :FsOpenRequest;
    stat @2 :FsPathRequest;
  }
}

struct FsOpenRequest {
  path @0 :Text;
}

struct FsPathRequest {
  path @0 :Text;
}

struct RegistryResponse {
  requestId @0 :UInt64;
  union {
    listResult @1 :List(Text);
    repoResult @2 :RepositoryResponse;
    error @3 :ErrorInfo;
  }
}

struct RepositoryResponse {
  union {
    fsResult @0 :FsResponse;
    deleteResult @1 :Void;
  }
}

struct FsResponse {
  union {
    open @0 :Text;
    stat @1 :Text;
  }
}

struct ErrorInfo {
  message @0 :Text;
}
"#;

    #[test]
    fn parse_nested_scoped_clients() {
        let schema = parse_capnp_schema(NESTED_SCOPED_SCHEMA, "registry").unwrap();
        assert_eq!(schema.scoped_clients.len(), 1); // RepositoryRequest

        let repo = &schema.scoped_clients[0];
        assert_eq!(repo.factory_name, "repo");
        assert_eq!(repo.client_name, "RepositoryClient");
        // fs was removed from inner_request_variants (it's now a nested client)
        assert_eq!(repo.inner_request_variants.len(), 1); // only delete
        assert_eq!(repo.inner_request_variants[0].name, "delete");

        // Check nested client
        assert_eq!(repo.nested_clients.len(), 1); // FsRequest
        let fs = &repo.nested_clients[0];
        assert_eq!(fs.factory_name, "fs");
        assert_eq!(fs.client_name, "FsClient");
        assert_eq!(fs.scope_fields.len(), 1);
        assert_eq!(fs.scope_fields[0].name, "worktree");
        assert_eq!(fs.inner_request_variants.len(), 2); // open, stat
        assert_eq!(fs.inner_request_variants[0].name, "open");
        assert_eq!(fs.inner_request_variants[1].name, "stat");

        // Check parent linkage
        assert_eq!(fs.parent_factory_name.as_deref(), Some("repo"));
        assert_eq!(fs.parent_scope_fields.len(), 1);
        assert_eq!(fs.parent_scope_fields[0].name, "repoId");
    }

    #[test]
    fn has_union_flag_set() {
        let structs = parse_all_structs(POLICY_SCHEMA);
        let req = structs.iter().find(|s| s.name == "PolicyRequest").unwrap();
        assert!(req.has_union);
        let check = structs.iter().find(|s| s.name == "PolicyCheck").unwrap();
        assert!(!check.has_union);
    }

    #[test]
    fn parse_single_line_structs() {
        let text = r#"
struct Foo { value @0 :UInt32; }
struct Bar { name @0 :Text; count @1 :UInt64; }
struct MultiLine {
  x @0 :Text;
  y @1 :Bool;
}
"#;
        let structs = parse_all_structs(text);
        assert_eq!(structs.len(), 3);

        let foo = structs.iter().find(|s| s.name == "Foo").unwrap();
        assert_eq!(foo.fields.len(), 1);
        assert_eq!(foo.fields[0].name, "value");
        assert_eq!(foo.fields[0].type_name, "UInt32");
        assert!(!foo.has_union);

        let bar = structs.iter().find(|s| s.name == "Bar").unwrap();
        assert_eq!(bar.fields.len(), 2);
        assert_eq!(bar.fields[0].name, "name");
        assert_eq!(bar.fields[0].type_name, "Text");
        assert_eq!(bar.fields[1].name, "count");
        assert_eq!(bar.fields[1].type_name, "UInt64");

        let multi = structs.iter().find(|s| s.name == "MultiLine").unwrap();
        assert_eq!(multi.fields.len(), 2);
    }
}

/// Merge annotations from metadata JSON into a ParsedSchema.
///
/// The metadata JSON is generated by build.rs using capnp introspection.
/// This function extracts annotation values and merges them into the
/// schema structure that was parsed from text, dispatching by annotation name.
pub fn merge_annotations_from_metadata(
    schema: &mut ParsedSchema,
    metadata_json: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::HashMap;

    #[derive(serde::Deserialize)]
    struct MetadataJson {
        structs: Vec<StructMetadata>,
    }

    #[derive(serde::Deserialize)]
    struct StructMetadata {
        name: String,
        fields: Vec<FieldMetadata>,
    }

    #[derive(serde::Deserialize)]
    struct FieldMetadata {
        name: String,
        #[allow(dead_code)]
        discriminant: u16,
        annotations: Vec<AnnotationMetadata>,
    }

    #[derive(serde::Deserialize)]
    struct AnnotationMetadata {
        #[serde(default)]
        name: String,
        value: serde_json::Value,
    }

    let metadata: MetadataJson = serde_json::from_str(metadata_json)?;

    // Build lookup maps: (struct_name, field_name, annotation_name) -> value
    let mut desc_map: HashMap<(String, String), String> = HashMap::new();
    let mut scope_map: HashMap<(String, String), String> = HashMap::new();
    let mut param_desc_map: HashMap<(String, String), String> = HashMap::new();
    let mut fixed_size_map: HashMap<(String, String), u32> = HashMap::new();

    for struct_meta in &metadata.structs {
        let struct_name = struct_meta.name.split(':').last().unwrap_or(&struct_meta.name).to_string();

        for field_meta in &struct_meta.fields {
            for ann in &field_meta.annotations {
                let key = (struct_name.clone(), field_meta.name.clone());

                // Handle fixedSize (UInt32 value) before string extraction
                if ann.name == "fixedSize" {
                    if let Some(n) = ann.value.as_u64() {
                        fixed_size_map.insert(key, n as u32);
                    }
                    continue;
                }

                let value_str = match &ann.value {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Object(obj) => {
                        // Handle enum ordinal: {"enum_ordinal": N}
                        if let Some(serde_json::Value::Number(n)) = obj.get("enum_ordinal") {
                            n.to_string()
                        } else {
                            continue;
                        }
                    }
                    serde_json::Value::Bool(b) => {
                        if *b { String::new() } else { continue; }
                    }
                    _ => continue,
                };

                match ann.name.as_str() {
                    "mcpDescription" => { desc_map.insert(key, value_str); }
                    "paramDescription" => { param_desc_map.insert(key, value_str); }
                    "mcpScope" => { scope_map.insert(key, value_str); }
                    "" => {
                        // Legacy: no name field — treat as description (backwards compat)
                        if !value_str.is_empty() {
                            desc_map.entry(key).or_insert(value_str);
                        }
                    }
                    _ => {} // cliHidden, deprecated, etc. — not merged into ParsedSchema
                }
            }
        }
    }

    // Helper to merge annotations into a variant
    let merge_variant = |variant: &mut super::types::UnionVariant, struct_name: &str| {
        let key = (struct_name.to_string(), variant.name.clone());
        if let Some(desc) = desc_map.get(&key) {
            variant.description = desc.clone();
        }
        if let Some(scope) = scope_map.get(&key) {
            variant.scope = scope.clone();
        }
    };

    // Merge into request variants
    for variant in &mut schema.request_variants {
        for struct_meta in &metadata.structs {
            let struct_name = struct_meta.name.split(':').last().unwrap_or(&struct_meta.name);
            if struct_name.ends_with("Request") {
                merge_variant(variant, struct_name);
            }
        }
    }

    // Merge into response variants
    for variant in &mut schema.response_variants {
        for struct_meta in &metadata.structs {
            let struct_name = struct_meta.name.split(':').last().unwrap_or(&struct_meta.name);
            if struct_name.ends_with("Response") {
                merge_variant(variant, struct_name);
            }
        }
    }

    // Merge into struct fields (paramDescription takes priority over mcpDescription)
    for struct_def in &mut schema.structs {
        for field in &mut struct_def.fields {
            let key = (struct_def.name.clone(), field.name.clone());
            if let Some(desc) = param_desc_map.get(&key) {
                field.description = desc.clone();
            } else if let Some(desc) = desc_map.get(&key) {
                field.description = desc.clone();
            }
            if let Some(&n) = fixed_size_map.get(&key) {
                field.fixed_size = Some(n);
            }
        }
    }

    // Merge into scoped client variants
    for scoped in &mut schema.scoped_clients {
        let req_name = format!("{}Request", to_pascal_case(&scoped.factory_name));
        let resp_name = format!("{}Response", to_pascal_case(&scoped.factory_name));
        for variant in &mut scoped.inner_request_variants {
            merge_variant(variant, &req_name);
        }
        for variant in &mut scoped.inner_response_variants {
            merge_variant(variant, &resp_name);
        }

        // Nested scoped clients
        for nested in &mut scoped.nested_clients {
            let nested_req = format!("{}Request", to_pascal_case(&nested.factory_name));
            let nested_resp = format!("{}Response", to_pascal_case(&nested.factory_name));
            for variant in &mut nested.inner_request_variants {
                merge_variant(variant, &nested_req);
            }
            for variant in &mut nested.inner_response_variants {
                merge_variant(variant, &nested_resp);
            }
        }
    }

    Ok(())
}
