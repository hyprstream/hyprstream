use proc_macro2::TokenStream;
use proc_macro_error2::abort;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;

use super::input::*;

struct Common<'a> {
    original_name: &'a proc_macro2::Ident,
    visibility: &'a syn::Visibility,
    field_members: Vec<syn::Member>,
    field_idents: Vec<syn::Ident>,
    skipped_field_names: Vec<syn::Member>,
    field_indices: Vec<syn::LitInt>,
    field_types: Vec<&'a syn::Type>,
    field_names: Vec<String>,
}

impl<'a> From<&'a DeriveStruct> for Common<'a> {
    fn from(input: &'a DeriveStruct) -> Self {
        let original_name = &input.common.name;
        let visibility = &input.common.visibility;

        let (skipped_fields, fields): (Vec<_>, Vec<_>) = input.fields.iter().partition(|field| field.skip);
        if fields.is_empty() {
            abort!(
                original_name.span(),
                "Expected struct to have at least one field"
            );
        }

        let field_members = fields
            .iter()
            .enumerate()
            .map(|(id, field)| {
                field
                    .syn
                    .ident
                    .as_ref()
                    .cloned()
                    .map_or_else(|| syn::Member::Unnamed(id.into()), syn::Member::Named)
            })
            .collect::<Vec<_>>();

        let field_idents = field_members
            .iter()
            .map(|f| match f {
                // `Member` doesn't impl `IdentFragment` in a way that preserves the "r#" prefix stripping of `Ident`, so we go one level inside.
                syn::Member::Named(ident) => format_ident!("field_{}", ident),
                syn::Member::Unnamed(index) => format_ident!("field_{}", index),
            })
            .collect::<Vec<_>>();

        let skipped_field_names = skipped_fields
            .iter()
            .enumerate()
            .map(|(id, field)| {
                field
                    .syn
                    .ident
                    .as_ref()
                    .cloned()
                    .map_or_else(|| syn::Member::Unnamed(id.into()), syn::Member::Named)
            })
            .collect::<Vec<_>>();

        let field_indices = field_members
            .iter()
            .enumerate()
            .map(|(idx, _ident)| syn::LitInt::new(&format!("{idx}"), proc_macro2::Span::call_site()))
            .collect::<Vec<_>>();

        let field_types: Vec<&syn::Type> = fields
            .iter()
            .map(|field| match &field.field_type {
                syn::Type::Path(_) => &field.field_type,
                syn::Type::Array(_) => &field.field_type,
                syn::Type::Reference(_) => &field.field_type,
                _ => panic!("Only `Path`, `Array`, `Reference` types are supported atm"),
            })
            .collect::<Vec<&syn::Type>>();

        let field_names = fields
            .iter()
            .enumerate()
            .map(
                |(id, field)| match (field.field_name.as_ref(), field.syn.ident.as_ref()) {
                    (Some(name), _) => name.to_owned(),                         // override enabled
                    (_, Some(ident)) => format_ident!("{}", ident).to_string(), // no override, named field
                    (_, None) => format!("field_{id}"),                         // no override, unnamed field
                },
            )
            .collect::<Vec<_>>();

        Self {
            original_name,
            visibility,
            field_members,
            field_idents,
            skipped_field_names,
            field_indices,
            field_types,
            field_names,
        }
    }
}

pub fn expand_field(input: DeriveStruct) -> TokenStream {
    let Common {
        original_name,
        field_types,
        field_names,
        ..
    } = (&input).into();

    let arrow_schema_impl = if input.fields.len() == 1 && input.is_transparent {
        quote! {}
    } else {
        quote! {
          impl #original_name {
            pub fn arrow_schema() -> arrow::datatypes::Schema {
                arrow::datatypes::Schema::new(vec![
                    #(
                        <#field_types as arrow_convert::field::ArrowField>::field(#field_names),
                    )*
                ])
            }
          }
        }
    };

    let data_type_impl = {
        if input.fields.len() == 1 && input.is_transparent {
            // Special case for single-field (tuple) structs
            let field = &input.fields[0];
            let ty = &field.field_type;
            quote! (
                <#ty as arrow_convert::field::ArrowField>::data_type()
            )
        } else {
            quote!(arrow::datatypes::DataType::Struct(Self::arrow_schema().fields))
        }
    };

    quote!(
        #arrow_schema_impl

        impl arrow_convert::field::ArrowField for #original_name {
            type Type = Self;

            fn data_type() -> arrow::datatypes::DataType {
                #data_type_impl
            }
        }

        arrow_convert::arrow_enable_vec_for_type!(#original_name);
    )
}

pub fn expand_serialize(input: DeriveStruct) -> TokenStream {
    let Common {
        original_name,
        visibility,
        field_members,
        field_idents,
        field_types,
        ..
    } = (&input).into();

    let mutable_array_name = &input.common.mutable_array_name();
    let mutable_field_array_types = field_types
        .iter()
        .map(|field_type| quote_spanned!( field_type.span() => <#field_type as arrow_convert::serialize::ArrowSerialize>::ArrayBuilderType))
        .collect::<Vec<TokenStream>>();

    let array_decl = quote! {
        #[derive(Debug)]
        #visibility struct #mutable_array_name {
            #(
                #field_idents: #mutable_field_array_types,
            )*
            data_type: arrow::datatypes::DataType,
            validity: Option<arrow::array::BooleanBufferBuilder>,
        }
    };

    let array_impl = quote! {
        impl #mutable_array_name {
            pub fn new() -> Self {
                Self {
                    #(#field_idents: <#field_types as arrow_convert::serialize::ArrowSerialize>::new_array(),)*
                    data_type: <#original_name as arrow_convert::field::ArrowField>::data_type(),
                    validity: None,
                }
            }

            fn init_validity(&mut self) {
                let length = <Self as arrow::array::ArrayBuilder>::len(self);
                let mut validity = arrow::array::BooleanBufferBuilder::new(length);
                validity.append_n(length - 1, true);
                validity.append(false);
                self.validity = Some(validity);
            }

            fn data_type(&self) -> &arrow::datatypes::DataType {
                &self.data_type
            }

            fn append_null(&mut self) {
                self.try_push(None::<&#original_name>).unwrap();
            }

            fn validity(&self) -> Option<&arrow::array::BooleanBufferBuilder> {
                self.validity.as_ref()
            }

            fn try_push(&mut self, item: Option<impl std::borrow::Borrow<#original_name>>) -> arrow::error::Result<()> {
                use arrow::array::ArrayBuilder;
                use std::borrow::Borrow;

                match item {
                    Some(i) =>  {
                        let i = i.borrow() as &#original_name;
                        #(
                            <#field_types as arrow_convert::serialize::ArrowSerialize>::arrow_serialize(i.#field_members.borrow(), &mut self.#field_idents)?;
                        )*;
                        match &mut self.validity {
                            Some(validity) => validity.append(true),
                            None => {}
                        }
                    },
                    None => {
                        <Self as arrow_convert::serialize::PushNull>::push_null(self);
                    }
                }
                Ok(())
            }

            fn try_extend<'a, I: IntoIterator<Item = Option<&'a #original_name>>>(&mut self, iter: I) -> arrow::error::Result<()> {
                for i in iter {
                    self.try_push(i)?;
                }
                Ok(())
            }
        }
    };

    let array_default_impl = quote! {
        impl Default for #mutable_array_name {
            fn default() -> Self {
                Self::new()
            }
        }
    };

    let array_push_null_impl = quote! {
        impl arrow_convert::serialize::PushNull for #mutable_array_name {
            fn push_null(&mut self) {
                use arrow::array::ArrayBuilder;
                use arrow_convert::serialize::{ArrowSerialize, PushNull};
                use std::borrow::Borrow;

                #(
                    // #mutable_field_array_types::append_null(&mut self.#field_idents);
                    <<#field_types as ArrowSerialize>::ArrayBuilderType as PushNull>::push_null(&mut self.#field_idents);
                    // self.#field_idents.append_null();
                )*;
                match &mut self.validity {
                    Some(validity) => validity.append(false),
                    None => {
                        self.init_validity();
                    }
                }
            }
        }
    };

    let first_ident = &field_idents[0];

    let array_mutable_array_impl = quote! {
        impl arrow::array::ArrayBuilder for #mutable_array_name {
            fn len(&self) -> usize {
                self.#first_ident.len()
            }

            fn finish(&mut self) -> arrow::array::ArrayRef {
                let values = vec![#(
                    <#mutable_field_array_types as arrow::array::ArrayBuilder>::finish(&mut self.#field_idents),
                )*];

                let arrow::datatypes::DataType::Struct(fields) =
                  <#original_name as arrow_convert::field::ArrowField>::data_type()
                  .clone() else {
                    panic!("datatype is not struct")
                  };

                std::sync::Arc::new(arrow::array::StructArray::new(
                    fields,
                    values,
                    std::mem::take(&mut self.validity).map(|mut x| x.finish().into()),
                ))
            }

            fn finish_cloned(&self) -> arrow::array::ArrayRef {
                let values = vec![#(
                    <#mutable_field_array_types as arrow::array::ArrayBuilder>::finish_cloned(&self.#field_idents),
                )*];

                let arrow::datatypes::DataType::Struct(fields) =
                  <#original_name as arrow_convert::field::ArrowField>::data_type()
                  .clone() else {
                    panic!("datatype is not struct")
                  };

                std::sync::Arc::new(arrow::array::StructArray::new(
                    fields,
                    values,
                    self.validity.as_ref().map(|x| x.finish_cloned().into())
                ))
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
                self
            }

            fn into_box_any(self: Box<Self>) -> Box<dyn std::any::Any> {
                self
            }
        }
    };

    // Special case for single-field (tuple) structs.
    if input.fields.len() == 1 && input.is_transparent {
        let first_type = &field_types[0];
        let first_field = &field_members[0];
        // Everything delegates to first field.
        quote! {
            impl arrow_convert::serialize::ArrowSerialize for #original_name {
                type ArrayBuilderType = <#first_type as arrow_convert::serialize::ArrowSerialize>::ArrayBuilderType;

                #[inline]
                fn new_array() -> Self::ArrayBuilderType {
                    <#first_type as arrow_convert::serialize::ArrowSerialize>::new_array()
                }

                #[inline]
                fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
                    <#first_type as arrow_convert::serialize::ArrowSerialize>::arrow_serialize(&v.#first_field, array)
                }
            }
        }
    } else {
        let field_arrow_serialize_impl = quote! {
            impl arrow_convert::serialize::ArrowSerialize for #original_name {
                type ArrayBuilderType = #mutable_array_name;

                #[inline]
                fn new_array() -> Self::ArrayBuilderType {
                    Self::ArrayBuilderType::default()
                }

                #[inline]
                fn arrow_serialize(v: &Self, array: &mut Self::ArrayBuilderType) -> arrow::error::Result<()> {
                    array.try_push(Some(v))
                }
            }
        };
        TokenStream::from_iter([
            array_decl,
            array_impl,
            array_default_impl,
            array_push_null_impl,
            array_mutable_array_impl,
            field_arrow_serialize_impl,
        ])
    }
}

pub fn expand_deserialize(input: DeriveStruct) -> TokenStream {
    let Common {
        original_name,
        visibility,
        field_members,
        field_idents,
        skipped_field_names,
        field_indices,
        field_types,
        ..
    } = (&input).into();

    let array_name = &input.common.array_name();
    let iterator_name = &input.common.iterator_name();
    let is_tuple_struct = matches!(field_members[0], syn::Member::Unnamed(_));

    let array_decl = quote! {
        #visibility struct #array_name
        {}
    };

    let array_impl = quote! {
        impl arrow_convert::deserialize::ArrowArray for #array_name
        {
            type BaseArrayType = arrow::array::StructArray;

            #[inline]
            fn iter_from_array_ref<'a>(b: &'a dyn arrow::array::Array)  -> <Self as arrow_convert::deserialize::ArrowArrayIterable>::Iter<'a>
            {
                use core::ops::Deref;
                use arrow::array::Array;

                let arr = b.as_any().downcast_ref::<arrow::array::StructArray>().unwrap();
                let values = arr.columns();
                let validity = arr.nulls();
                // for now do a straight comp
                #iterator_name {
                    #(
                        #field_idents: <<#field_types as arrow_convert::deserialize::ArrowDeserialize>::ArrayType as arrow_convert::deserialize::ArrowArray>::iter_from_array_ref(values[#field_indices].deref()),
                    )*
                    has_validity: validity.as_ref().is_some(),
                    validity_iter: validity.as_ref().map(|x| x.iter()).unwrap_or_else(|| arrow::util::bit_iterator::BitIterator::new(&[], 0, 0))
                }
            }
        }
    };

    let array_iterable_impl = quote! {
        impl arrow_convert::deserialize::ArrowArrayIterable for #array_name
        {
            type Item<'a> = Option<#original_name>;
            type Iter<'a> = #iterator_name<'a>;

            fn iter(&self) -> Self::Iter<'_> {
                unimplemented!("Use iter_from_array_ref");
            }
        }
    };

    let iterator_decl = quote! {
        #visibility struct #iterator_name<'a> {
            #(
                #field_idents: <<#field_types as arrow_convert::deserialize::ArrowDeserialize>::ArrayType as arrow_convert::deserialize::ArrowArrayIterable>::Iter<'a>,
            )*
            validity_iter: arrow::util::bit_iterator::BitIterator<'a>,
            has_validity: bool
        }
    };

    let struct_inst: syn::Pat = if is_tuple_struct {
        // If the fields are unnamed, we create a tuple-struct
        syn::parse_quote! {
            #original_name (
                #(<#field_types as arrow_convert::deserialize::ArrowDeserialize>::arrow_deserialize_internal(#field_idents),)*
            )
        }
    } else {
        syn::parse_quote! {
            #original_name {
                #(#field_members: <#field_types as arrow_convert::deserialize::ArrowDeserialize>::arrow_deserialize_internal(#field_idents),)*
                #(#skipped_field_names: std::default::Default::default(),)*
            }
        }
    };

    let iterator_impl = quote! {
        impl<'a> #iterator_name<'a> {
            #[inline]
            fn return_next(&mut self) -> Option<#original_name> {
                if let (#(
                    Some(#field_idents),
                )*) = (
                    #(self.#field_idents.next(),)*
                )
                { Some(#struct_inst) }
                else { None }
            }

            #[inline]
            fn consume_next(&mut self) {
                #(let _ = self.#field_idents.next();)*
            }
        }
    };

    let iterator_iterator_impl = quote! {
        impl<'a> Iterator for #iterator_name<'a> {
            type Item = Option<#original_name>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                if !self.has_validity {
                    self.return_next().map(|y| Some(y))
                }
                else {
                    let is_valid = self.validity_iter.next();
                    is_valid.map(|x| if x { self.return_next() } else { self.consume_next(); None })
                }
            }
        }
    };

    // Special case for single-field (tuple) structs.
    if input.fields.len() == 1 && input.is_transparent {
        let first_type = &field_types[0];

        let deser_body_mapper = if is_tuple_struct {
            quote! { #original_name }
        } else {
            let first_name = &field_members[0];
            quote! { |v| #original_name { #first_name: v } }
        };

        // Everything delegates to first field.
        quote! {
            impl arrow_convert::deserialize::ArrowDeserialize for #original_name {
                type ArrayType = <#first_type as arrow_convert::deserialize::ArrowDeserialize>::ArrayType;

                #[inline]
                fn arrow_deserialize<'a>(v: <Self::ArrayType as arrow_convert::deserialize::ArrowArrayIterable>::Item<'a>) -> Option<Self> {
                    <#first_type as arrow_convert::deserialize::ArrowDeserialize>::arrow_deserialize(v).map(#deser_body_mapper)
                }
            }
        }
    } else {
        let field_arrow_deserialize_impl = quote! {
            impl arrow_convert::deserialize::ArrowDeserialize for #original_name {
                type ArrayType = #array_name;

                #[inline]
                fn arrow_deserialize<'a>(v: Option<Self>) -> Option<Self> {
                    v
                }
            }
        };

        TokenStream::from_iter([
            array_decl,
            array_impl,
            array_iterable_impl,
            iterator_decl,
            iterator_impl,
            iterator_iterator_impl,
            field_arrow_deserialize_impl,
        ])
    }
}
