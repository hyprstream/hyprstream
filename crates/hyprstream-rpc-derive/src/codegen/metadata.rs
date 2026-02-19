//! Schema metadata and JSON dispatcher generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::resolve::ResolvedSchema;
use crate::schema::types::*;
use crate::util::*;

/// Generate schema metadata functions + JSON dispatchers.
pub fn generate_metadata(service_name: &str, resolved: &ResolvedSchema, types_crate: Option<&syn::Path>) -> TokenStream {
    let pascal = to_pascal_case(service_name);

    let metadata_structs = generate_metadata_structs();
    let schema_metadata = generate_schema_metadata_fn(
        service_name,
        &pascal,
        &resolved.raw.request_variants,
        &resolved.raw.response_variants,
        resolved,
        &resolved.raw.scoped_clients,
    );
    let json_dispatcher = generate_json_dispatcher(
        &pascal,
        &resolved.raw.request_variants,
        &resolved.raw.response_variants,
        resolved,
        &resolved.raw.scoped_clients,
    );
    let scoped_client_tree = generate_scoped_client_tree(&resolved.raw.scoped_clients, types_crate);

    quote! {
        #metadata_structs
        #schema_metadata
        #json_dispatcher
        #scoped_client_tree
    }
}

fn generate_metadata_structs() -> TokenStream {
    quote! {
        pub use hyprstream_rpc::service::metadata::{ParamMeta as ParamSchema, MethodMeta as MethodSchema};
    }
}

fn generate_schema_metadata_fn(
    service_name: &str,
    pascal: &str,
    request_variants: &[UnionVariant],
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    scoped_clients: &[ScopedClient],
) -> TokenStream {
    let scoped_names: Vec<&str> = scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();
    let doc = format!("Schema metadata for the {pascal} service.");

    let method_entries: Vec<TokenStream> = request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .map(|v| generate_method_schema_entry(v, resolved, false, "", response_variants, false))
        .collect();

    let mut scoped_fns = Vec::new();
    collect_scoped_metadata_recursive(service_name, scoped_clients, resolved, &mut scoped_fns);

    quote! {
        #[doc = #doc]
        pub fn schema_metadata() -> (&'static str, &'static [MethodSchema]) {
            static METHODS: &[MethodSchema] = &[
                #(#method_entries,)*
            ];
            (#service_name, METHODS)
        }

        #(#scoped_fns)*
    }
}

fn generate_method_schema_entry(
    v: &UnionVariant,
    resolved: &ResolvedSchema,
    is_scoped: bool,
    scope_field: &str,
    response_variants: &[UnionVariant],
    is_scoped_streaming_check: bool,
) -> TokenStream {
    let method_name = to_snake_case(&v.name);
    let method_desc = &v.description;
    let scope_str = v.scope.as_str();
    let is_streaming = is_streaming_variant(&v.name, response_variants, is_scoped_streaming_check);
    let cli_hidden = v.cli_hidden;
    let ct = CapnpType::classify_primitive(&v.type_name);

    let params = match ct {
        CapnpType::Void => vec![],
        CapnpType::Struct(_) | CapnpType::Unknown(_) => {
            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                sdef.fields
                    .iter()
                    .map(|f| {
                        let fname = to_snake_case(&f.name);
                        let ftype = &f.type_name;
                        let fdesc = &f.description;
                        let is_optional = f.optional || ftype == "Bool";
                        let required = !is_optional;
                        let default_val = if ftype == "Bool" { "false" } else { "" };
                        quote! {
                            ParamSchema { name: #fname, type_name: #ftype, required: #required, description: #fdesc, default_value: #default_val }
                        }
                    })
                    .collect()
            } else {
                vec![]
            }
        }
        _ => {
            let type_str = &v.type_name;
            vec![quote! {
                ParamSchema { name: "value", type_name: #type_str, required: true, description: "", default_value: "" }
            }]
        }
    };

    quote! {
        MethodSchema {
            name: #method_name,
            params: &[#(#params),*],
            is_scoped: #is_scoped,
            scope_field: #scope_field,
            description: #method_desc,
            scope: #scope_str,
            is_streaming: #is_streaming,
            hidden: #cli_hidden,
        }
    }
}

fn generate_scoped_schema_metadata_fn(
    service_name: &str,
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let scope_snake = to_snake_case(&sc.factory_name);
    let fn_name = format_ident!("{}_schema_metadata", scope_snake);
    let scope_pascal = to_pascal_case(&sc.factory_name);
    let doc = format!("Schema metadata for scoped {scope_pascal} methods.");
    let scope_field_name = sc
        .scope_fields
        .first()
        .map(|f| to_snake_case(&f.name))
        .unwrap_or_default();

    let method_entries: Vec<TokenStream> = sc
        .inner_request_variants
        .iter()
        .map(|v| generate_method_schema_entry(v, resolved, true, &scope_field_name, &sc.inner_response_variants, true))
        .collect();

    quote! {
        #[doc = #doc]
        pub fn #fn_name() -> (&'static str, &'static str, &'static [MethodSchema]) {
            static METHODS: &[MethodSchema] = &[
                #(#method_entries,)*
            ];
            (#service_name, #scope_snake, METHODS)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON Dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Check if a request variant's corresponding response is StreamInfo.
fn is_streaming_variant(variant_name: &str, response_variants: &[UnionVariant], is_scoped: bool) -> bool {
    let expected_name = if is_scoped {
        variant_name.to_owned()
    } else {
        format!("{}Result", variant_name)
    };
    response_variants
        .iter()
        .find(|v| v.name == expected_name)
        .map(|v| v.type_name == "StreamInfo")
        .unwrap_or(false)
}

fn generate_json_dispatcher(
    pascal: &str,
    request_variants: &[UnionVariant],
    response_variants: &[UnionVariant],
    resolved: &ResolvedSchema,
    scoped_clients: &[ScopedClient],
) -> TokenStream {
    let client_name = format_ident!("{}Client", pascal);
    let scoped_names: Vec<&str> = scoped_clients
        .iter()
        .map(|sc| sc.factory_name.as_str())
        .collect();

    // Non-streaming dispatch arms for call_method
    let main_match_arms: Vec<TokenStream> = request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .filter(|v| !is_streaming_variant(&v.name, response_variants, false))
        .map(|v| generate_json_method_dispatch_arm(v, response_variants, false, resolved))
        .collect();

    // Streaming dispatch arms for call_streaming_method
    let main_streaming_arms: Vec<TokenStream> = request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .filter(|v| is_streaming_variant(&v.name, response_variants, false))
        .map(|v| generate_json_streaming_dispatch_arm(v, resolved))
        .collect();

    let streaming_method = quote! {
        /// Dispatch a streaming method call by name with JSON arguments and an ephemeral public key.
        /// Returns the StreamInfo as a JSON value.
        #[allow(unused_variables)]
        pub async fn call_streaming_method(
            &self,
            method: &str,
            args: &serde_json::Value,
            ephemeral_pubkey: [u8; 32],
        ) -> anyhow::Result<serde_json::Value> {
            match method {
                #(#main_streaming_arms)*
                _ => anyhow::bail!("Unknown streaming method: {}", method),
            }
        }
    };

    let mut scoped_dispatchers = Vec::new();
    collect_scoped_dispatchers_recursive(scoped_clients, resolved, &mut scoped_dispatchers);

    // Generate call_scoped_method on the top-level client
    let top_scoped_dispatch = generate_call_scoped_method_for_client(scoped_clients);

    // Generate call_scoped_method on scoped clients that have children
    let mut nested_scoped_dispatch = Vec::new();
    collect_call_scoped_method_recursive(scoped_clients, &mut nested_scoped_dispatch);

    quote! {
        impl #client_name {
            /// Dispatch a method call by name with JSON arguments.
            /// Returns the response as a proper JSON value.
            pub async fn call_method(&self, method: &str, args: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
                match method {
                    #(#main_match_arms)*
                    _ => anyhow::bail!("Unknown method: {}", method),
                }
            }

            #streaming_method

            #top_scoped_dispatch
        }

        #(#scoped_dispatchers)*

        #(#nested_scoped_dispatch)*
    }
}

fn generate_json_method_dispatch_arm(
    v: &UnionVariant,
    response_variants: &[UnionVariant],
    is_scoped: bool,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let method_name_str = to_snake_case(&v.name);
    let method_name = format_ident!("{}", method_name_str);
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

    match ct {
        CapnpType::Void => {
            let result_name = if is_scoped {
                v.name.clone()
            } else {
                format!("{}Result", v.name)
            };
            let resp = response_variants.iter().find(|r| r.name == result_name);
            let resp_is_void = resp.is_none_or(|r| r.type_name == "Void");

            if resp_is_void {
                quote! {
                    #method_name_str => {
                        self.#method_name().await?;
                        Ok(serde_json::Value::Null)
                    }
                }
            } else {
                quote! {
                    #method_name_str => {
                        let result = self.#method_name().await?;
                        Ok(serde_json::to_value(&result)?)
                    }
                }
            }
        }
        CapnpType::Text => quote! {
            #method_name_str => {
                let value = args[#method_name_str].as_str().or_else(|| args["value"].as_str()).unwrap_or_default();
                let result = self.#method_name(value).await?;
                Ok(serde_json::to_value(&result)?)
            }
        },
        _ => {
            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                let settable_fields: Vec<&FieldDef> = sdef
                    .fields
                    .iter()
                    .filter(|f| !is_union_only_struct(&f.type_name, resolved))
                    .collect();

                let extractions: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        let fname_str = to_snake_case(&f.name);
                        json_field_extraction_token(&fname, &fname_str, &f.type_name, f.optional, resolved)
                    })
                    .collect();

                let args_list: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        if resolved.resolve_type(&f.type_name).is_by_ref {
                            quote! { &#fname }
                        } else {
                            quote! { #fname }
                        }
                    })
                    .collect();

                quote! {
                    #method_name_str => {
                        #(#extractions)*
                        let result = self.#method_name(#(#args_list),*).await?;
                        Ok(serde_json::to_value(&result)?)
                    }
                }
            } else {
                let err_msg = format!("Method {}: struct type not found", method_name_str);
                quote! {
                    #method_name_str => anyhow::bail!(#err_msg),
                }
            }
        }
    }
}

/// Generate a streaming dispatch arm for call_streaming_method.
fn generate_json_streaming_dispatch_arm(
    v: &UnionVariant,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let method_name_str = to_snake_case(&v.name);
    let method_name = format_ident!("{}", method_name_str);
    let ct = resolved.resolve_type(&v.type_name).capnp_type.clone();

    match ct {
        CapnpType::Void => quote! {
            #method_name_str => {
                let result = self.#method_name(ephemeral_pubkey).await?;
                Ok(serde_json::to_value(&result)?)
            }
        },
        CapnpType::Text => quote! {
            #method_name_str => {
                let value = args[#method_name_str].as_str().or_else(|| args["value"].as_str()).unwrap_or_default();
                let result = self.#method_name(value, ephemeral_pubkey).await?;
                Ok(serde_json::to_value(&result)?)
            }
        },
        _ => {
            if let Some(sdef) = resolved.find_struct(&v.type_name) {
                let settable_fields: Vec<&FieldDef> = sdef
                    .fields
                    .iter()
                    .filter(|f| !is_union_only_struct(&f.type_name, resolved))
                    .collect();

                let extractions: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        let fname_str = to_snake_case(&f.name);
                        json_field_extraction_token(&fname, &fname_str, &f.type_name, f.optional, resolved)
                    })
                    .collect();

                let args_list: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        if resolved.resolve_type(&f.type_name).is_by_ref {
                            quote! { &#fname }
                        } else {
                            quote! { #fname }
                        }
                    })
                    .collect();

                quote! {
                    #method_name_str => {
                        #(#extractions)*
                        let result = self.#method_name(#(#args_list,)* ephemeral_pubkey).await?;
                        Ok(serde_json::to_value(&result)?)
                    }
                }
            } else {
                let err_msg = format!("Streaming method {}: struct type not found", method_name_str);
                quote! {
                    #method_name_str => anyhow::bail!(#err_msg),
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Recursive helpers for scoped metadata and dispatch
// ─────────────────────────────────────────────────────────────────────────────

fn collect_scoped_metadata_recursive(
    service_name: &str,
    clients: &[ScopedClient],
    resolved: &ResolvedSchema,
    out: &mut Vec<TokenStream>,
) {
    for sc in clients {
        out.push(generate_scoped_schema_metadata_fn(service_name, sc, resolved));
        collect_scoped_metadata_recursive(service_name, &sc.nested_clients, resolved, out);
    }
}

fn collect_scoped_dispatchers_recursive(
    clients: &[ScopedClient],
    resolved: &ResolvedSchema,
    out: &mut Vec<TokenStream>,
) {
    for sc in clients {
        out.push(generate_scoped_dispatcher_block(sc, resolved));
        collect_scoped_dispatchers_recursive(&sc.nested_clients, resolved, out);
    }
}

fn generate_scoped_dispatcher_block(
    sc: &ScopedClient,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let scoped_client_name = format_ident!("{}", sc.client_name);
    let scoped_match_arms: Vec<TokenStream> = sc
        .inner_request_variants
        .iter()
        .filter(|v| !is_streaming_variant(&v.name, &sc.inner_response_variants, true))
        .map(|v| generate_json_method_dispatch_arm(v, &sc.inner_response_variants, true, resolved))
        .collect();

    // Streaming dispatch arms
    let scoped_streaming_arms: Vec<TokenStream> = sc
        .inner_request_variants
        .iter()
        .filter(|v| is_streaming_variant(&v.name, &sc.inner_response_variants, true))
        .map(|v| generate_json_streaming_dispatch_arm(v, resolved))
        .collect();

    let scoped_streaming_method = quote! {
        /// Dispatch a scoped streaming method call by name with JSON arguments and an ephemeral public key.
        #[allow(unused_variables)]
        pub async fn call_streaming_method(
            &self,
            method: &str,
            args: &serde_json::Value,
            ephemeral_pubkey: [u8; 32],
        ) -> anyhow::Result<serde_json::Value> {
            match method {
                #(#scoped_streaming_arms)*
                _ => anyhow::bail!("Unknown scoped streaming method: {}", method),
            }
        }
    };

    quote! {
        impl #scoped_client_name {
            /// Dispatch a scoped method call by name with JSON arguments.
            pub async fn call_method(&self, method: &str, args: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
                match method {
                    #(#scoped_match_arms)*
                    _ => anyhow::bail!("Unknown scoped method: {}", method),
                }
            }

            #scoped_streaming_method
        }
    }
}

/// Generate a factory call expression for a scoped client.
/// Handles type conversion for non-string scope fields (e.g., UInt32 → parse from &str).
fn generate_scoped_factory_call(sc: &ScopedClient) -> TokenStream {
    let factory_snake = format_ident!("{}", to_snake_case(&sc.factory_name));
    if sc.scope_fields.is_empty() {
        quote! { self.#factory_snake() }
    } else if sc.scope_fields.len() == 1 {
        // Check if scope field needs parsing from string
        let field = &sc.scope_fields[0];
        match field.type_name.as_str() {
            "UInt8" | "UInt16" | "UInt32" | "UInt64" | "Int8" | "Int16" | "Int32" | "Int64"
            | "Float32" | "Float64" => {
                quote! { self.#factory_snake(scope_id.parse()?) }
            }
            _ => {
                quote! { self.#factory_snake(scope_id) }
            }
        }
    } else {
        // Multiple scope fields — not supported in call_scoped_method yet
        quote! { self.#factory_snake(scope_id) }
    }
}

/// Generate `call_scoped_method` on the top-level client.
fn generate_call_scoped_method_for_client(
    scoped_clients: &[ScopedClient],
) -> TokenStream {
    if scoped_clients.is_empty() {
        return TokenStream::new();
    }

    let match_arms: Vec<TokenStream> = scoped_clients.iter().map(|sc| {
        let scope_name_str = to_snake_case(&sc.factory_name);
        let factory_call = generate_scoped_factory_call(sc);

        quote! {
            #scope_name_str => #factory_call.call_scoped_method(remaining, method, args).await,
        }
    }).collect();

    quote! {
        /// Dispatch a scoped method call through nested scope chain.
        #[allow(unused_variables)]
        pub async fn call_scoped_method(
            &self,
            scopes: &[(&str, &str)],
            method: &str,
            args: &serde_json::Value,
        ) -> anyhow::Result<serde_json::Value> {
            if scopes.is_empty() {
                return self.call_method(method, args).await;
            }
            let (scope_name, scope_id) = scopes[0];
            let remaining = &scopes[1..];
            match scope_name {
                #(#match_arms)*
                _ => anyhow::bail!("Unknown scope: {}", scope_name),
            }
        }
    }
}

/// Recursively generate `call_scoped_method` on all scoped clients.
fn collect_call_scoped_method_recursive(
    clients: &[ScopedClient],
    out: &mut Vec<TokenStream>,
) {
    for sc in clients {
        let client_name = format_ident!("{}", sc.client_name);

        if sc.nested_clients.is_empty() {
            out.push(quote! {
                impl #client_name {
                    /// Dispatch a scoped method call through nested scope chain.
                    pub async fn call_scoped_method(
                        &self,
                        scopes: &[(&str, &str)],
                        method: &str,
                        args: &serde_json::Value,
                    ) -> anyhow::Result<serde_json::Value> {
                        if scopes.is_empty() {
                            return self.call_method(method, args).await;
                        }
                        anyhow::bail!("No nested scopes available for '{}'", scopes[0].0)
                    }
                }
            });
        } else {
            let match_arms: Vec<TokenStream> = sc.nested_clients.iter().map(|nested| {
                let scope_name_str = to_snake_case(&nested.factory_name);
                let factory_call = generate_scoped_factory_call(nested);

                quote! {
                    #scope_name_str => #factory_call.call_scoped_method(remaining, method, args).await,
                }
            }).collect();

            out.push(quote! {
                impl #client_name {
                    /// Dispatch a scoped method call through nested scope chain.
                    #[allow(unused_variables)]
                    pub async fn call_scoped_method(
                        &self,
                        scopes: &[(&str, &str)],
                        method: &str,
                        args: &serde_json::Value,
                    ) -> anyhow::Result<serde_json::Value> {
                        if scopes.is_empty() {
                            return self.call_method(method, args).await;
                        }
                        let (scope_name, scope_id) = scopes[0];
                        let remaining = &scopes[1..];
                        match scope_name {
                            #(#match_arms)*
                            _ => anyhow::bail!("Unknown scope: {}", scope_name),
                        }
                    }
                }
            });
        }
        collect_call_scoped_method_recursive(&sc.nested_clients, out);
    }
}

/// Generate `scoped_client_tree()` metadata function.
fn generate_scoped_client_tree(
    scoped_clients: &[ScopedClient],
    types_crate: Option<&syn::Path>,
) -> TokenStream {
    let mut consts = Vec::new();
    let mut top_level_names = Vec::new();

    for sc in scoped_clients {
        generate_tree_consts(sc, &mut consts, &mut top_level_names, types_crate);
    }

    let top_refs: Vec<syn::Ident> = top_level_names;

    quote! {
        /// Returns the scoped client tree metadata for dynamic CLI/MCP dispatch.
        pub fn scoped_client_tree() -> &'static [hyprstream_rpc::service::metadata::ScopedClientTreeNode] {
            use hyprstream_rpc::service::metadata::ScopedClientTreeNode;
            #(#consts)*
            static TREE: &[ScopedClientTreeNode] = &[#(#top_refs),*];
            TREE
        }
    }
}

/// Recursively generate const nodes for the scoped client tree (bottom-up).
fn generate_tree_consts(
    sc: &ScopedClient,
    consts: &mut Vec<TokenStream>,
    names_out: &mut Vec<syn::Ident>,
    types_crate: Option<&syn::Path>,
) {
    let mut child_names = Vec::new();
    for nested in &sc.nested_clients {
        generate_tree_consts(nested, consts, &mut child_names, types_crate);
    }

    let scope_snake = to_snake_case(&sc.factory_name);
    let const_name = format_ident!("__SCOPE_{}", scope_snake.to_uppercase());
    let scope_field_name = sc.scope_fields.first()
        .map(|f| to_snake_case(&f.name))
        .unwrap_or_default();
    let fn_name = format_ident!("{}_schema_metadata", scope_snake);

    let metadata_fn_path = match types_crate {
        Some(tc) => quote! { #tc::#fn_name },
        None => quote! { #fn_name },
    };

    let nested_refs: Vec<&syn::Ident> = child_names.iter().collect();
    let nested_array = if nested_refs.is_empty() {
        quote! { &[] }
    } else {
        quote! { &[#(#nested_refs),*] }
    };

    consts.push(quote! {
        const #const_name: ScopedClientTreeNode = ScopedClientTreeNode {
            scope_name: #scope_snake,
            scope_field: #scope_field_name,
            metadata_fn: #metadata_fn_path,
            nested: #nested_array,
        };
    });

    names_out.push(const_name);
}

/// Check if a type name refers to a union-only struct (no regular fields).
fn is_union_only_struct(type_name: &str, resolved: &ResolvedSchema) -> bool {
    if let Some(s) = resolved.find_struct(type_name) {
        s.has_union && s.fields.is_empty()
    } else {
        false
    }
}

fn json_field_extraction_token(
    fname: &syn::Ident,
    fname_str: &str,
    type_name: &str,
    optional: bool,
    resolved: &ResolvedSchema,
) -> TokenStream {
    let ct = resolved.resolve_type(type_name).capnp_type.clone();
    match ct {
        CapnpType::Text => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str()).unwrap_or("");
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid string field '{}'", #fname_str))?;
        }},
        CapnpType::Bool => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_bool()).unwrap_or(false);
        },
        CapnpType::UInt8 => if optional { quote! {
            let #fname: u8 = args.get(#fname_str).and_then(|v| v.as_u64())
                .unwrap_or(0).try_into().unwrap_or(0);
        }} else { quote! {
            let #fname: u8 = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u8 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("u8 overflow for field '{}'", #fname_str))?;
        }},
        CapnpType::UInt16 => if optional { quote! {
            let #fname: u16 = args.get(#fname_str).and_then(|v| v.as_u64())
                .unwrap_or(0).try_into().unwrap_or(0);
        }} else { quote! {
            let #fname: u16 = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u16 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("u16 overflow for field '{}'", #fname_str))?;
        }},
        CapnpType::UInt32 => if optional { quote! {
            let #fname: u32 = args.get(#fname_str).and_then(|v| v.as_u64())
                .unwrap_or(0).try_into().unwrap_or(0);
        }} else { quote! {
            let #fname: u32 = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u32 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("u32 overflow for field '{}'", #fname_str))?;
        }},
        CapnpType::UInt64 => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_u64()).unwrap_or(0);
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u64 field '{}'", #fname_str))?;
        }},
        CapnpType::Int8 => if optional { quote! {
            let #fname: i8 = args.get(#fname_str).and_then(|v| v.as_i64())
                .unwrap_or(0).try_into().unwrap_or(0);
        }} else { quote! {
            let #fname: i8 = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i8 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("i8 overflow for field '{}'", #fname_str))?;
        }},
        CapnpType::Int16 => if optional { quote! {
            let #fname: i16 = args.get(#fname_str).and_then(|v| v.as_i64())
                .unwrap_or(0).try_into().unwrap_or(0);
        }} else { quote! {
            let #fname: i16 = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i16 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("i16 overflow for field '{}'", #fname_str))?;
        }},
        CapnpType::Int32 => if optional { quote! {
            let #fname: i32 = args.get(#fname_str).and_then(|v| v.as_i64())
                .unwrap_or(0).try_into().unwrap_or(0);
        }} else { quote! {
            let #fname: i32 = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i32 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("i32 overflow for field '{}'", #fname_str))?;
        }},
        CapnpType::Int64 => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_i64()).unwrap_or(0);
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i64 field '{}'", #fname_str))?;
        }},
        CapnpType::Float32 => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid f32 field '{}'", #fname_str))? as f32;
        }},
        CapnpType::Float64 => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_f64()).unwrap_or(0.0);
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid f64 field '{}'", #fname_str))?;
        }},
        CapnpType::Data => if optional { quote! {
            let #fname: Vec<u8> = args.get(#fname_str).and_then(|v| v.as_str())
                .map(|s| s.as_bytes().to_vec()).unwrap_or_default();
        }} else { quote! {
            let #fname: Vec<u8> = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid data field '{}'", #fname_str))?
                .as_bytes().to_vec();
        }},
        CapnpType::ListText => if optional { quote! {
            let #fname: Vec<String> = args.get(#fname_str).and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
        }} else { quote! {
            let #fname: Vec<String> = args.get(#fname_str).and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid array field '{}'", #fname_str))?
                .iter().map(|v| v.as_str().map(String::from)
                    .ok_or_else(|| anyhow::anyhow!("non-string element in array field '{}'", #fname_str)))
                .collect::<Result<Vec<_>, _>>()?;
        }},
        CapnpType::ListData => if optional { quote! {
            let #fname: Vec<Vec<u8>> = args.get(#fname_str).and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.as_bytes().to_vec())).collect())
                .unwrap_or_default();
        }} else { quote! {
            let #fname: Vec<Vec<u8>> = args.get(#fname_str).and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid array field '{}'", #fname_str))?
                .iter().map(|v| v.as_str().map(|s| s.as_bytes().to_vec())
                    .ok_or_else(|| anyhow::anyhow!("non-string element in array field '{}'", #fname_str)))
                .collect::<Result<Vec<_>, _>>()?;
        }},
        CapnpType::ListPrimitive(_) => {
            let rust_type = rust_type_tokens(&ct.rust_owned_type());
            quote! {
                let #fname: #rust_type = serde_json::from_value(
                    args.get(#fname_str).cloned().unwrap_or(serde_json::Value::Array(vec![]))
                ).unwrap_or_default();
            }
        }
        CapnpType::ListStruct(ref inner) => {
            let data_name = format_ident!("{}", inner);
            quote! {
                let #fname: Vec<#data_name> = serde_json::from_value(
                    args.get(#fname_str).cloned().unwrap_or(serde_json::Value::Array(vec![]))
                ).unwrap_or_default();
            }
        }
        CapnpType::Struct(ref name) => {
            let data_name = format_ident!("{}", name);
            quote! {
                let #fname: #data_name = serde_json::from_value(
                    args.get(#fname_str).cloned().unwrap_or_default()
                ).unwrap_or_default();
            }
        }
        CapnpType::Enum(_) => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str()).unwrap_or("");
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid enum field '{}'", #fname_str))?;
        }},
        _ => if optional { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str()).unwrap_or("");
        }} else { quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid field '{}'", #fname_str))?;
        }},
    }
}
