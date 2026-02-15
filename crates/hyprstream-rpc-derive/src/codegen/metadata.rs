//! Schema metadata and JSON dispatcher generation.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::schema::types::*;
use crate::util::*;

/// Generate schema metadata functions + JSON dispatchers.
pub fn generate_metadata(service_name: &str, schema: &ParsedSchema) -> TokenStream {
    let pascal = to_pascal_case(service_name);

    let metadata_structs = generate_metadata_structs();
    let schema_metadata = generate_schema_metadata_fn(
        service_name,
        &pascal,
        &schema.request_variants,
        &schema.response_variants,
        &schema.structs,
        &schema.scoped_clients,
    );
    let json_dispatcher = generate_json_dispatcher(
        &pascal,
        &schema.request_variants,
        &schema.response_variants,
        &schema.structs,
        &schema.enums,
        &schema.scoped_clients,
    );

    quote! {
        #metadata_structs
        #schema_metadata
        #json_dispatcher
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
    structs: &[StructDef],
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
        .map(|v| generate_method_schema_entry(v, structs, false, "", response_variants, false))
        .collect();

    let mut scoped_fns: Vec<TokenStream> = scoped_clients
        .iter()
        .map(|sc| {
            generate_scoped_schema_metadata_fn(service_name, sc, structs)
        })
        .collect();

    // Generate metadata for nested scoped clients
    for sc in scoped_clients {
        for nested in &sc.nested_clients {
            scoped_fns.push(generate_scoped_schema_metadata_fn(service_name, nested, structs));
        }
    }

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
    structs: &[StructDef],
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
            if let Some(sdef) = structs.iter().find(|s| s.name == v.type_name) {
                sdef.fields
                    .iter()
                    .map(|f| {
                        let fname = to_snake_case(&f.name);
                        let ftype = &f.type_name;
                        let fdesc = &f.description;
                        quote! {
                            ParamSchema { name: #fname, type_name: #ftype, required: true, description: #fdesc }
                        }
                    })
                    .collect()
            } else {
                vec![]
            }
        }
        _ => {
            // All primitives (Text, Data, Bool, numeric types)
            let type_str = &v.type_name;
            vec![quote! {
                ParamSchema { name: "value", type_name: #type_str, required: true, description: "" }
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
    structs: &[StructDef],
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
        .map(|v| generate_method_schema_entry(v, structs, true, &scope_field_name, &sc.inner_response_variants, true))
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
        variant_name.to_string()
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
    structs: &[StructDef],
    enums: &[EnumDef],
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
        .map(|v| generate_json_method_dispatch_arm(v, structs, enums))
        .collect();

    // Streaming dispatch arms for call_streaming_method
    let main_streaming_arms: Vec<TokenStream> = request_variants
        .iter()
        .filter(|v| !scoped_names.contains(&v.name.as_str()))
        .filter(|v| is_streaming_variant(&v.name, response_variants, false))
        .map(|v| generate_json_streaming_dispatch_arm(v, structs, enums))
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

    let mut scoped_dispatchers: Vec<TokenStream> = scoped_clients
        .iter()
        .map(|sc| {
            let scoped_client_name = format_ident!("{}", sc.client_name);
            let scoped_match_arms: Vec<TokenStream> = sc
                .inner_request_variants
                .iter()
                .filter(|v| !is_streaming_variant(&v.name, &sc.inner_response_variants, true))
                .map(|v| generate_json_method_dispatch_arm(v, structs, enums))
                .collect();

            // Streaming dispatch arms for scoped clients
            let scoped_streaming_arms: Vec<TokenStream> = sc
                .inner_request_variants
                .iter()
                .filter(|v| is_streaming_variant(&v.name, &sc.inner_response_variants, true))
                .map(|v| generate_json_streaming_dispatch_arm(v, structs, enums))
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
        })
        .collect();

    // Generate JSON dispatchers for nested scoped clients
    for sc in scoped_clients {
        for nested in &sc.nested_clients {
            let nested_client_name = format_ident!("{}", nested.client_name);
            let nested_match_arms: Vec<TokenStream> = nested
                .inner_request_variants
                .iter()
                .filter(|v| !is_streaming_variant(&v.name, &nested.inner_response_variants, true))
                .map(|v| generate_json_method_dispatch_arm(v, structs, enums))
                .collect();

            scoped_dispatchers.push(quote! {
                impl #nested_client_name {
                    /// Dispatch a nested scoped method call by name with JSON arguments.
                    pub async fn call_method(&self, method: &str, args: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
                        match method {
                            #(#nested_match_arms)*
                            _ => anyhow::bail!("Unknown nested scoped method: {}", method),
                        }
                    }
                }
            });
        }
    }

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
        }

        #(#scoped_dispatchers)*
    }
}

fn generate_json_method_dispatch_arm(
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let method_name_str = to_snake_case(&v.name);
    let method_name = format_ident!("{}", method_name_str);
    let ct = CapnpType::classify(&v.type_name, structs, enums);

    match ct {
        CapnpType::Void => quote! {
            #method_name_str => {
                self.#method_name().await?;
                Ok(serde_json::Value::Null)
            }
        },
        CapnpType::Text => quote! {
            #method_name_str => {
                let value = args[#method_name_str].as_str().or_else(|| args["value"].as_str()).unwrap_or_default();
                let result = self.#method_name(value).await?;
                Ok(serde_json::to_value(&result)?)
            }
        },
        _ => {
            if let Some(sdef) = structs.iter().find(|s| s.name == v.type_name) {
                // Filter fields: skip union-only struct fields (they have no method param)
                let settable_fields: Vec<&FieldDef> = sdef
                    .fields
                    .iter()
                    .filter(|f| !is_union_only_struct(&f.type_name, structs))
                    .collect();

                let extractions: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        let fname_str = to_snake_case(&f.name);
                        json_field_extraction_token(&fname, &fname_str, &f.type_name, structs, enums)
                    })
                    .collect();

                let args_list: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        let fct = CapnpType::classify(&f.type_name, structs, enums);
                        if fct.is_by_ref() {
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
/// Like generate_json_method_dispatch_arm but passes ephemeral_pubkey to the typed method.
fn generate_json_streaming_dispatch_arm(
    v: &UnionVariant,
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let method_name_str = to_snake_case(&v.name);
    let method_name = format_ident!("{}", method_name_str);
    let ct = CapnpType::classify(&v.type_name, structs, enums);

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
            if let Some(sdef) = structs.iter().find(|s| s.name == v.type_name) {
                let settable_fields: Vec<&FieldDef> = sdef
                    .fields
                    .iter()
                    .filter(|f| !is_union_only_struct(&f.type_name, structs))
                    .collect();

                let extractions: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        let fname_str = to_snake_case(&f.name);
                        json_field_extraction_token(&fname, &fname_str, &f.type_name, structs, enums)
                    })
                    .collect();

                let args_list: Vec<TokenStream> = settable_fields
                    .iter()
                    .map(|f| {
                        let fname = format_ident!("{}", to_snake_case(&f.name));
                        let fct = CapnpType::classify(&f.type_name, structs, enums);
                        if fct.is_by_ref() {
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

/// Check if a type name refers to a union-only struct (no regular fields).
fn is_union_only_struct(type_name: &str, structs: &[StructDef]) -> bool {
    if let Some(s) = structs.iter().find(|s| s.name == type_name) {
        s.has_union && s.fields.is_empty()
    } else {
        false
    }
}

fn json_field_extraction_token(
    fname: &syn::Ident,
    fname_str: &str,
    type_name: &str,
    structs: &[StructDef],
    enums: &[EnumDef],
) -> TokenStream {
    let ct = CapnpType::classify(type_name, structs, enums);
    match ct {
        CapnpType::Text => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid string field '{}'", #fname_str))?;
        },
        CapnpType::Bool => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_bool())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid bool field '{}'", #fname_str))?;
        },
        CapnpType::UInt8 => quote! {
            let #fname: u8 = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u8 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("u8 overflow for field '{}'", #fname_str))?;
        },
        CapnpType::UInt16 => quote! {
            let #fname: u16 = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u16 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("u16 overflow for field '{}'", #fname_str))?;
        },
        CapnpType::UInt32 => quote! {
            let #fname: u32 = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u32 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("u32 overflow for field '{}'", #fname_str))?;
        },
        CapnpType::UInt64 => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid u64 field '{}'", #fname_str))?;
        },
        CapnpType::Int8 => quote! {
            let #fname: i8 = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i8 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("i8 overflow for field '{}'", #fname_str))?;
        },
        CapnpType::Int16 => quote! {
            let #fname: i16 = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i16 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("i16 overflow for field '{}'", #fname_str))?;
        },
        CapnpType::Int32 => quote! {
            let #fname: i32 = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i32 field '{}'", #fname_str))?
                .try_into().map_err(|_| anyhow::anyhow!("i32 overflow for field '{}'", #fname_str))?;
        },
        CapnpType::Int64 => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_i64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid i64 field '{}'", #fname_str))?;
        },
        CapnpType::Float32 => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid f32 field '{}'", #fname_str))? as f32;
        },
        CapnpType::Float64 => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid f64 field '{}'", #fname_str))?;
        },
        CapnpType::Data => quote! {
            let #fname: Vec<u8> = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid data field '{}'", #fname_str))?
                .as_bytes().to_vec();
        },
        CapnpType::ListText => quote! {
            let #fname: Vec<String> = args.get(#fname_str).and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid array field '{}'", #fname_str))?
                .iter().map(|v| v.as_str().map(String::from)
                    .ok_or_else(|| anyhow::anyhow!("non-string element in array field '{}'", #fname_str)))
                .collect::<Result<Vec<_>, _>>()?;
        },
        CapnpType::ListData => quote! {
            let #fname: Vec<Vec<u8>> = args.get(#fname_str).and_then(|v| v.as_array())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid array field '{}'", #fname_str))?
                .iter().map(|v| v.as_str().map(|s| s.as_bytes().to_vec())
                    .ok_or_else(|| anyhow::anyhow!("non-string element in array field '{}'", #fname_str)))
                .collect::<Result<Vec<_>, _>>()?;
        },
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
        CapnpType::Enum(_) => {
            // Enum params are passed as &str to the client method
            quote! {
                let #fname = args.get(#fname_str).and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("missing or invalid enum field '{}'", #fname_str))?;
            }
        }
        _ => quote! {
            let #fname = args.get(#fname_str).and_then(|v| v.as_str())
                .ok_or_else(|| anyhow::anyhow!("missing or invalid field '{}'", #fname_str))?;
        },
    }
}
