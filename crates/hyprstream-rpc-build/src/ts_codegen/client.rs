//! Generate TypeScript client classes and scoped client factories.
//!
//! Each service gets a `{Service}Client` class that wraps builder/parser functions
//! and provides typed method calls. Scoped clients (detected from nested union patterns)
//! get their own classes with curried scope fields.
//!
//! Scoped client methods call deterministic builder functions and unwrap responses
//! through the ancestor chain, matching the Rust codegen pattern exactly.

use hyprstream_rpc_build::schema::types::{FieldDef, ParsedSchema, ScopedClient};
use hyprstream_rpc_build::util::{to_camel_case, to_pascal_case};

/// Emit an error-variant check that throws an `Error` with the RPC error message.
///
/// Generates:
/// ```text
/// if ({parsed_var}.variant === 'error') {
///   throw new Error(({parsed_var}.data as { message?: string })?.message ?? 'RPC error');
/// }
/// ```
fn emit_error_throw(out: &mut String, parsed_var: &str) {
    out.push_str(&format!(
        "    if ({parsed_var}.variant === 'error') {{\n\
         \x20     throw new Error(({parsed_var}.data as {{ message?: string }})?.message ?? 'RPC error');\n\
         \x20   }}\n"
    ));
}

use super::{capnp_to_ts_type, is_primitive};

/// Check if any method in the schema returns StreamInfo (indicating streaming support needed).
pub fn has_streaming_methods(schema: &ParsedSchema) -> bool {
    schema
        .response_variants
        .iter()
        .any(|v| v.type_name == "StreamInfo")
        || schema.scoped_clients.iter().any(has_streaming_in_scoped)
}

fn has_streaming_in_scoped(sc: &ScopedClient) -> bool {
    sc.inner_response_variants
        .iter()
        .any(|v| v.type_name == "StreamInfo")
        || sc.nested_clients.iter().any(has_streaming_in_scoped)
}

/// Generate the main client class and scoped client classes.
///
/// Clients take a `WasmRpcClient` (from wasm-bindgen) and call only its
/// service-and-method-bound exports. The method metadata is generated
/// independently from the canonical Cap'n Proto request union.
pub fn generate_client(out: &mut String, service_name: &str, schema: &ParsedSchema) {
    let pascal = to_pascal_case(service_name);

    let has_streaming = has_streaming_methods(schema);

    // Import the exact method-bound WasmRpcClient surface.
    out.push_str(
        "/** Per-call authentication overrides for generated RPC methods. */\n\
         export interface RpcCallOptions {\n\
         \x20 jwt?: string;\n\
         \x20 delegatedBearer?: string;\n\
         }\n\n\
         /** RPC client interface (matches RpcClient from wasm-bindgen). */\n\
         export interface RpcTransport {\n\
         \x20 callForServiceWithMethod(service: string, method: number, bytes: Uint8Array): Promise<Uint8Array>;\n\
         \x20 callWithOptionsForServiceWithMethod(service: string, method: number, bytes: Uint8Array, jwt?: string, delegatedBearer?: string): Promise<Uint8Array>;\n",
    );
    if has_streaming {
        out.push_str(
            "  openStreamForServiceWithMethod(service: string, method: number, bytes: Uint8Array): Promise<StreamHandle>;\n\
             \x20 openStreamWithOptionsForServiceWithMethod(service: string, method: number, bytes: Uint8Array, jwt?: string, delegatedBearer?: string): Promise<StreamHandle>;\n",
        );
    }
    out.push_str(
        "\x20 nextId(): bigint;\n\
         }\n\n",
    );

    // Main client class
    out.push_str(&format!(
        "export class {pascal}Client {{\n\
         \x20 constructor(private readonly client: RpcTransport) {{}}\n\n"
    ));

    // Filter out scoped variants (they become factory methods)
    let scoped_names: Vec<&str> = schema
        .scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Non-scoped methods on the main client
    for variant in &schema.request_variants {
        if scoped_names.contains(&variant.name.as_str()) {
            continue; // handled as factory method
        }

        let method_discriminator = schema
            .request_struct
            .as_ref()
            .and_then(|request| {
                request.fields.iter().find(|field| {
                    field.name == variant.name && field.discriminant_value != 0xFFFF
                })
            })
            .map(|field| field.discriminant_value)
            .expect("generated TypeScript method must have a root union discriminator");

        let is_void = variant.type_name == "Void";
        let is_prim = is_primitive(&variant.type_name) && !is_void;
        let payload_struct = if !is_void && !is_prim {
            schema.structs.iter().find(|s| s.name == variant.type_name)
        } else {
            None
        };

        // Return type from matching response variant
        let resp_variant_name = format!("{}Result", variant.name);
        let resp_variant = schema
            .response_variants
            .iter()
            .find(|v| v.name == resp_variant_name);
        let resp_type_name = resp_variant
            .map(|v| v.type_name.as_str())
            .unwrap_or("unknown");
        let is_streaming = resp_type_name == "StreamInfo";
        let return_type = if is_streaming {
            "StreamHandle".into()
        } else {
            capnp_to_ts_type(resp_type_name)
        };

        // Method signature
        let mut params = Vec::new();
        if let Some(ps) = payload_struct {
            params.push(format!("p: {}", ps.name));
        } else if is_prim {
            params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
        }

        params.push("options?: RpcCallOptions".to_owned());
        out.push_str(&format!(
            "  async {}({}): Promise<{}> {{\n",
            to_camel_case(&variant.name),
            params.join(", "),
            return_type
        ));

        // Build the request
        let builder_name = format!("build{pascal}Request_{}", variant.name);
        if is_void {
            out.push_str(&format!(
                "    const bytes = {builder_name}(this.client.nextId());\n"
            ));
        } else {
            out.push_str(&format!(
                "    const bytes = {builder_name}(this.client.nextId(), p);\n"
            ));
        }

        if is_streaming {
            // Streaming method: use callStreaming and return StreamSubscription
            generate_streaming_body(out, service_name, method_discriminator);
        } else {
            // Call transport and parse response
            emit_bound_unary_call(out, service_name, method_discriminator);
            out.push_str(&format!(
                "    const parsed = parse{pascal}Response(resp);\n"
            ));

            // Error handling
            emit_error_throw(out, "parsed");
            out.push_str(&format!("    return parsed.data as {};\n", return_type));
        }
        out.push_str("  }\n\n");
    }

    // Factory methods for scoped clients
    for sc in &schema.scoped_clients {
        let params: Vec<String> = sc
            .scope_fields
            .iter()
            .map(|f| {
                format!(
                    "{}: {}",
                    to_camel_case(&f.name),
                    capnp_to_ts_type(&f.type_name)
                )
            })
            .collect();
        let args: Vec<String> = sc
            .scope_fields
            .iter()
            .map(|f| to_camel_case(&f.name))
            .collect();

        out.push_str(&format!(
            "  {}({}): {} {{\n",
            to_camel_case(&sc.factory_name),
            params.join(", "),
            sc.client_name
        ));
        out.push_str(&format!(
            "    return new {}(this.client, {});\n",
            sc.client_name,
            args.join(", ")
        ));
        out.push_str("  }\n\n");
    }

    out.push_str("}\n\n");

    // Generate scoped client classes. The authenticated method commitment is
    // the root request union arm that selects the scoped request schema.
    for sc in &schema.scoped_clients {
        let root_method_discriminator = schema
            .request_struct
            .as_ref()
            .and_then(|request| {
                request.fields.iter().find(|field| {
                    field.name == sc.factory_name && field.discriminant_value != 0xFFFF
                })
            })
            .map(|field| field.discriminant_value)
            .expect("generated scoped TypeScript method must have a root union discriminator");
        generate_scoped_client(
            out,
            service_name,
            sc,
            &[],
            schema,
            root_method_discriminator,
        );
    }
}

/// Generate a scoped client class (e.g., RepositoryClient).
///
/// `ancestors` tracks the scoped client chain from outermost to the parent of `sc`.
/// The constructor receives ALL accumulated scope fields from ancestors + own.
/// Methods call deterministic builder functions with the full scope chain.
fn generate_scoped_client(
    out: &mut String,
    service_name: &str,
    sc: &ScopedClient,
    ancestors: &[&ScopedClient],
    schema: &ParsedSchema,
    root_method_discriminator: u16,
) {
    let pascal = to_pascal_case(service_name);

    // All scope fields from ancestors + current
    let all_scope_fields: Vec<&FieldDef> = ancestors
        .iter()
        .flat_map(|a| a.scope_fields.iter())
        .chain(sc.scope_fields.iter())
        .collect();

    // Build the name chain for builder function references (matches builders.rs naming)
    let mut name_chain = String::new();
    for anc in ancestors {
        name_chain.push_str(&format!("_{}", anc.factory_name));
    }
    name_chain.push_str(&format!("_{}", sc.factory_name));

    // Constructor: transport + all accumulated scope fields
    let scope_params: Vec<String> = all_scope_fields
        .iter()
        .map(|f| {
            format!(
                "private readonly {}: {}",
                to_camel_case(&f.name),
                capnp_to_ts_type(&f.type_name)
            )
        })
        .collect();

    out.push_str(&format!("export class {} {{\n", sc.client_name));
    out.push_str("  constructor(\n    private readonly client: RpcTransport,\n");
    for p in &scope_params {
        out.push_str(&format!("    {p},\n"));
    }
    out.push_str("  ) {}\n\n");

    // Nested scoped client factory names (their variants become factories, not methods)
    let nested_names: Vec<&str> = sc
        .nested_clients
        .iter()
        .map(|nc| nc.factory_name.as_str())
        .collect();

    // Chain depth for response unwrapping (ancestors + current scope)
    let chain_depth = ancestors.len() + 1;

    // Methods for each inner variant
    for variant in &sc.inner_request_variants {
        if nested_names.contains(&variant.name.as_str()) {
            continue;
        }

        let is_void = variant.type_name == "Void";
        let is_prim = is_primitive(&variant.type_name) && !is_void;
        let payload_struct = if !is_void && !is_prim {
            schema.structs.iter().find(|s| s.name == variant.type_name)
        } else {
            None
        };

        // Return type from matching inner response variant.
        // Inner response variants use bare names (e.g., "generateStream"),
        // NOT the "Result" suffix used by top-level response variants.
        let resp_variant = sc
            .inner_response_variants
            .iter()
            .find(|v| v.name == variant.name);
        let resp_type_name = resp_variant
            .map(|v| v.type_name.as_str())
            .unwrap_or("unknown");
        let is_streaming = resp_type_name == "StreamInfo";
        let return_type = if is_streaming {
            "StreamHandle".into()
        } else {
            capnp_to_ts_type(resp_type_name)
        };

        let mut params = Vec::new();
        if let Some(ps) = payload_struct {
            params.push(format!("p: {}", ps.name));
        } else if is_prim {
            params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
        }

        params.push("options?: RpcCallOptions".to_owned());
        out.push_str(&format!(
            "  async {}({}): Promise<{}> {{\n",
            to_camel_case(&variant.name),
            params.join(", "),
            return_type
        ));

        // Build the request using the scoped builder function
        // Name matches builders.rs: build{Pascal}Request{name_chain}_{method}
        let builder_name = format!("build{pascal}Request{name_chain}_{}", variant.name);

        // Builder args: transport.nextId() + all scope fields + optional payload
        let mut builder_args = vec!["this.client.nextId()".to_owned()];
        for f in &all_scope_fields {
            builder_args.push(format!("this.{}", to_camel_case(&f.name)));
        }
        if !is_void {
            builder_args.push("p".to_owned());
        }

        out.push_str(&format!(
            "    const bytes = {builder_name}({});\n",
            builder_args.join(", ")
        ));

        if is_streaming {
            // Streaming method: use callStreaming and unwrap through scope chain to get StreamInfo
            generate_streaming_body(out, service_name, root_method_discriminator);
        } else {
            // Call transport and parse outer response
            emit_bound_unary_call(out, service_name, root_method_discriminator);
            out.push_str(&format!(
                "    const parsed = parse{pascal}Response(resp);\n"
            ));

            // Error handling at outer level
            emit_error_throw(out, "parsed");

            // Unwrap through the scoped response chain.
            // The parser returns nested { variant, data: { variant, data: ... } } for scoped responses.
            // Each chain level adds one layer of wrapping.
            let mut current_var = "parsed.data".to_owned();
            for depth in 0..chain_depth {
                let inner_var = format!("_r{depth}");
                out.push_str(&format!(
                    "    const {inner_var} = {current_var} as {{ variant: string; data: unknown }};\n"
                ));

                if depth < chain_depth - 1 {
                    // Intermediate level: just unwrap to next level
                    current_var = format!("{inner_var}.data");
                } else {
                    // Final scope level: check for error and return the method result
                    emit_error_throw(out, &inner_var);
                    out.push_str(&format!(
                        "    return {inner_var}.data as {};\n",
                        return_type
                    ));
                }
            }
        }

        out.push_str("  }\n\n");
    }

    // Nested scoped client factories
    for nc in &sc.nested_clients {
        let params: Vec<String> = nc
            .scope_fields
            .iter()
            .map(|f| {
                format!(
                    "{}: {}",
                    to_camel_case(&f.name),
                    capnp_to_ts_type(&f.type_name)
                )
            })
            .collect();
        let args: Vec<String> = nc
            .scope_fields
            .iter()
            .map(|f| to_camel_case(&f.name))
            .collect();

        // Pass ALL accumulated scope fields through (ancestors + current → child)
        let mut all_args: Vec<String> = all_scope_fields
            .iter()
            .map(|f| format!("this.{}", to_camel_case(&f.name)))
            .collect();
        all_args.extend(args);

        out.push_str(&format!(
            "  {}({}): {} {{\n",
            to_camel_case(&nc.factory_name),
            params.join(", "),
            nc.client_name
        ));
        out.push_str(&format!(
            "    return new {}(this.client, {});\n",
            nc.client_name,
            all_args.join(", ")
        ));
        out.push_str("  }\n\n");
    }

    out.push_str("}\n\n");

    // Recursively generate nested scoped clients with updated ancestors
    let mut new_ancestors: Vec<&ScopedClient> = ancestors.to_vec();
    new_ancestors.push(sc);
    for nc in &sc.nested_clients {
        generate_scoped_client(
            out,
            service_name,
            nc,
            &new_ancestors,
            schema,
            root_method_discriminator,
        );
    }
}

fn emit_bound_unary_call(out: &mut String, service_name: &str, method_discriminator: u16) {
    out.push_str(&format!(
        "    const resp = options\n\
         \x20     ? await this.client.callWithOptionsForServiceWithMethod(\"{service_name}\", {method_discriminator}, bytes, options.jwt, options.delegatedBearer)\n\
         \x20     : await this.client.callForServiceWithMethod(\"{service_name}\", {method_discriminator}, bytes);\n"
    ));
}

fn generate_streaming_body(out: &mut String, service_name: &str, method_discriminator: u16) {
    // All crypto (ECDH, key derivation, HMAC verification) happens in Rust.
    // JS supplies independently generated service/method metadata and receives
    // only the verified StreamHandle.
    out.push_str(&format!(
        "    return options\n\
         \x20     ? await this.client.openStreamWithOptionsForServiceWithMethod(\"{service_name}\", {method_discriminator}, bytes, options.jwt, options.delegatedBearer)\n\
         \x20     : await this.client.openStreamForServiceWithMethod(\"{service_name}\", {method_discriminator}, bytes);\n"
    ));
}

#[cfg(test)]
mod browser_binding_tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use hyprstream_rpc_build::schema::types::{FieldSection, StructDef, UnionVariant};

    fn variant(name: &str, type_name: &str) -> UnionVariant {
        UnionVariant {
            name: name.to_owned(),
            type_name: type_name.to_owned(),
            description: String::new(),
            scope: "invoke".to_owned(),
            scope_exempt: false,
            cli_hidden: false,
            doc_example: String::new(),
            vfs_path: String::new(),
            vfs_kind: String::new(),
            vfs_bulk: false,
            vfs_hidden: false,
            vfs_mac: String::new(),
        }
    }

    fn field(name: &str, discriminant_value: u16) -> FieldDef {
        FieldDef {
            name: name.to_owned(),
            type_name: "Void".to_owned(),
            description: String::new(),
            fixed_size: None,
            optional: false,
            slot_offset: 0,
            section: FieldSection::Data,
            discriminant_value,
            serde_rename: None,
            domain_type: None,
        }
    }

    #[test]
    fn generated_browser_calls_bind_service_method_options_and_streaming() {
        let schema = ParsedSchema {
            request_variants: vec![variant("ping", "Void"), variant("watch", "Void")],
            response_variants: vec![
                variant("pingResult", "Void"),
                variant("watchResult", "StreamInfo"),
            ],
            structs: vec![],
            scoped_clients: vec![],
            enums: vec![],
            request_struct: Some(StructDef {
                name: "Request".to_owned(),
                fields: vec![field("ping", 3), field("watch", 7)],
                has_union: true,
                domain_type: None,
                origin_file: None,
                data_words: 2,
                pointer_words: 0,
                discriminant_count: 2,
                discriminant_offset: 4,
                union_arms: vec![],
            }),
            response_struct: None,
        };
        let mut out = String::new();
        generate_client(&mut out, "model", &schema);
        assert!(
            out.contains("callForServiceWithMethod(\"model\", 3, bytes)"),
            "{out}"
        );
        assert!(
            out.contains("callWithOptionsForServiceWithMethod(\"model\", 3, bytes"),
            "{out}"
        );
        assert!(
            out.contains("openStreamForServiceWithMethod(\"model\", 7, bytes)"),
            "{out}"
        );
        assert!(
            out.contains("openStreamWithOptionsForServiceWithMethod(\"model\", 7, bytes"),
            "{out}"
        );
        assert!(!out.contains("this.client.call(bytes)"), "{out}");
        assert!(!out.contains("this.client.openStream(bytes)"), "{out}");
    }
}
