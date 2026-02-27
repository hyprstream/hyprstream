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

use super::{capnp_to_ts_type, is_primitive};

/// Generate the transport interface, main client class, and scoped client classes.
pub fn generate_client(out: &mut String, service_name: &str, schema: &ParsedSchema) {
    let pascal = to_pascal_case(service_name);

    // Transport interface
    out.push_str(&format!(
        "/** Transport interface for {pascal} RPC calls. */\n\
         export interface {pascal}Transport {{\n\
         \x20 call(bytes: Uint8Array): Promise<Uint8Array>;\n\
         \x20 nextId(): bigint;\n\
         }}\n\n"
    ));

    // Main client class
    out.push_str(&format!(
        "export class {pascal}Client {{\n\
         \x20 constructor(private readonly transport: {pascal}Transport) {{}}\n\n"
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
        let return_type = resp_variant
            .map(|v| capnp_to_ts_type(&v.type_name))
            .unwrap_or_else(|| "unknown".into());

        // Method signature
        let mut params = Vec::new();
        if let Some(ps) = payload_struct {
            params.push(format!("p: {}", ps.name));
        } else if is_prim {
            params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
        }

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
                "    const bytes = {builder_name}(this.transport.nextId());\n"
            ));
        } else {
            out.push_str(&format!(
                "    const bytes = {builder_name}(this.transport.nextId(), p);\n"
            ));
        }

        // Call transport and parse response
        out.push_str("    const resp = await this.transport.call(bytes);\n");
        out.push_str(&format!(
            "    const parsed = parse{pascal}Response(resp);\n"
        ));

        // Error handling
        out.push_str(
            "    if (parsed.variant === 'error') {\n\
             \x20     throw new Error((parsed.data as { message?: string })?.message ?? 'RPC error');\n\
             \x20   }\n",
        );
        out.push_str(&format!(
            "    return parsed.data as {};\n",
            return_type
        ));
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
            "    return new {}(this.transport, {});\n",
            sc.client_name,
            args.join(", ")
        ));
        out.push_str("  }\n\n");
    }

    out.push_str("}\n\n");

    // Generate scoped client classes (with empty ancestor chain initially)
    for sc in &schema.scoped_clients {
        generate_scoped_client(out, service_name, sc, &[], schema);
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
    out.push_str(&format!(
        "  constructor(\n    private readonly transport: {pascal}Transport,\n"
    ));
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
        let return_type = resp_variant
            .map(|v| capnp_to_ts_type(&v.type_name))
            .unwrap_or_else(|| "unknown".into());

        let mut params = Vec::new();
        if let Some(ps) = payload_struct {
            params.push(format!("p: {}", ps.name));
        } else if is_prim {
            params.push(format!("p: {}", capnp_to_ts_type(&variant.type_name)));
        }

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
        let mut builder_args = vec!["this.transport.nextId()".to_string()];
        for f in &all_scope_fields {
            builder_args.push(format!("this.{}", to_camel_case(&f.name)));
        }
        if !is_void {
            builder_args.push("p".to_string());
        }

        out.push_str(&format!(
            "    const bytes = {builder_name}({});\n",
            builder_args.join(", ")
        ));

        // Call transport and parse outer response
        out.push_str("    const resp = await this.transport.call(bytes);\n");
        out.push_str(&format!(
            "    const parsed = parse{pascal}Response(resp);\n"
        ));

        // Error handling at outer level
        out.push_str(
            "    if (parsed.variant === 'error') {\n\
             \x20     throw new Error((parsed.data as { message?: string })?.message ?? 'RPC error');\n\
             \x20   }\n",
        );

        // Unwrap through the scoped response chain.
        // The parser returns nested { variant, data: { variant, data: ... } } for scoped responses.
        // Each chain level adds one layer of wrapping.
        let mut current_var = "parsed.data".to_string();
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
                out.push_str(&format!(
                    "    if ({inner_var}.variant === 'error') {{\n\
                     \x20     throw new Error(({inner_var}.data as {{ message?: string }})?.message ?? 'RPC error');\n\
                     \x20   }}\n"
                ));
                out.push_str(&format!(
                    "    return {inner_var}.data as {};\n",
                    return_type
                ));
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
            "    return new {}(this.transport, {});\n",
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
        generate_scoped_client(out, service_name, nc, &new_ancestors, schema);
    }
}
