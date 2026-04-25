//! Type definitions for parsed Cap'n Proto schemas.
//!
//! Re-exported from `hyprstream_rpc_build::schema::types` — the canonical definitions
//! live in the build crate so they can be shared between the proc-macro and the
//! TypeScript codegen binary.

pub use hyprstream_rpc_build::schema::types::*;

/// Collect all struct names that need Data structs generated.
pub fn collect_list_struct_types(schema: &ParsedSchema) -> Vec<String> {
    let mut types = Vec::new();

    let add_type = |type_name: &str, types: &mut Vec<String>| {
        if type_name.starts_with("List(") {
            let inner = &type_name[5..type_name.len() - 1];
            if !is_primitive_capnp_type(inner)
                && !types.contains(&inner.to_owned())
                && schema.enums.iter().all(|e| e.name != inner)
            {
                types.push(inner.to_owned());
            }
        } else if !is_primitive_capnp_type(type_name)
            && !type_name.starts_with("List(")
            && schema.enums.iter().all(|e| e.name != type_name)
            && schema.structs.iter().any(|s| s.name == type_name)
            && !types.contains(&type_name.to_owned())
        {
            types.push(type_name.to_owned());
        }
    };

    for v in &schema.response_variants {
        add_type(&v.type_name, &mut types);
    }
    for sc in &schema.scoped_clients {
        for v in &sc.inner_response_variants {
            add_type(&v.type_name, &mut types);
        }
    }
    for s in &schema.structs {
        for f in s.non_union_fields() {
            add_type(&f.type_name, &mut types);
        }
    }
    for v in &schema.request_variants {
        add_type(&v.type_name, &mut types);
        if let Some(s) = schema.structs.iter().find(|s| s.name == v.type_name) {
            for f in s.non_union_fields() {
                add_type(&f.type_name, &mut types);
            }
        }
    }
    for sc in &schema.scoped_clients {
        collect_from_scoped(sc, schema, &mut types);
    }

    types
}

fn collect_from_scoped(sc: &ScopedClient, schema: &ParsedSchema, types: &mut Vec<String>) {
    let add_type = |type_name: &str, types: &mut Vec<String>| {
        if type_name.starts_with("List(") {
            let inner = &type_name[5..type_name.len() - 1];
            if !is_primitive_capnp_type(inner)
                && !types.contains(&inner.to_owned())
                && schema.enums.iter().all(|e| e.name != inner)
            {
                types.push(inner.to_owned());
            }
        } else if !is_primitive_capnp_type(type_name)
            && !type_name.starts_with("List(")
            && schema.enums.iter().all(|e| e.name != type_name)
            && schema.structs.iter().any(|s| s.name == type_name)
            && !types.contains(&type_name.to_owned())
        {
            types.push(type_name.to_owned());
        }
    };
    for v in &sc.inner_response_variants {
        add_type(&v.type_name, types);
    }
    for v in &sc.inner_request_variants {
        add_type(&v.type_name, types);
        if let Some(s) = schema.structs.iter().find(|s| s.name == v.type_name) {
            for f in s.non_union_fields() {
                add_type(&f.type_name, types);
            }
        }
    }
    for nested in &sc.nested_clients {
        collect_from_scoped(nested, schema, types);
    }
}
