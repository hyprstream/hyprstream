pub use root::*;
#[no_implicit_prelude]
mod root {
    pub mod org {
        pub mod apache {
            pub mod arrow {
                pub mod flatbuf {
                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Footer {
                        pub version: self::MetadataVersion,
                        pub schema:
                            ::core::option::Option<::planus::alloc::boxed::Box<self::Schema>>,
                        pub dictionaries:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Block>>,
                        pub record_batches:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Block>>,
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                    }

                    impl Footer {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            version: impl ::planus::WriteAsDefault<
                                self::MetadataVersion,
                                self::MetadataVersion,
                            >,
                            schema: impl ::planus::WriteAsOptional<::planus::Offset<self::Schema>>,
                            dictionaries: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::Block]>,
                            >,
                            record_batches: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[self::Block]>,
                            >,
                            custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_version =
                                version.prepare(builder, &self::MetadataVersion::V1);

                            let prepared_schema = schema.prepare(builder);

                            let prepared_dictionaries = dictionaries.prepare(builder);

                            let prepared_record_batches = record_batches.prepare(builder);

                            let prepared_custom_metadata = custom_metadata.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<12, 18>::new(builder);

                            if prepared_version.is_some() {
                                table_writer.calculate_size::<self::MetadataVersion>(2);
                            }
                            if prepared_schema.is_some() {
                                table_writer.calculate_size::<::planus::Offset<self::Schema>>(4);
                            }
                            if prepared_dictionaries.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[self::Block]>>(6);
                            }
                            if prepared_record_batches.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[self::Block]>>(8);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(10);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_schema) =
                                    prepared_schema
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_schema);
                                }
                                if let ::core::option::Option::Some(prepared_dictionaries) =
                                    prepared_dictionaries
                                {
                                    table_writer.write::<_, _, 4>(2, &prepared_dictionaries);
                                }
                                if let ::core::option::Option::Some(prepared_record_batches) =
                                    prepared_record_batches
                                {
                                    table_writer.write::<_, _, 4>(3, &prepared_record_batches);
                                }
                                if let ::core::option::Option::Some(prepared_custom_metadata) =
                                    prepared_custom_metadata
                                {
                                    table_writer.write::<_, _, 4>(4, &prepared_custom_metadata);
                                }
                                if let ::core::option::Option::Some(prepared_version) =
                                    prepared_version
                                {
                                    table_writer.write::<_, _, 2>(0, &prepared_version);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Footer>> for Footer {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Footer>> for Footer {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Footer>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Footer> for Footer {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Footer> {
                            Footer::create(
                                builder,
                                &self.version,
                                &self.schema,
                                &self.dictionaries,
                                &self.record_batches,
                                &self.custom_metadata,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct FooterRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FooterRef<'a> {
                        pub fn version(&self) -> ::planus::Result<self::MetadataVersion> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Footer", "version")?
                                    .unwrap_or(self::MetadataVersion::V1),
                            )
                        }

                        pub fn schema(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::SchemaRef<'a>>>
                        {
                            self.0.access(1, "Footer", "schema")
                        }

                        pub fn dictionaries(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::BlockRef<'a>>>,
                        > {
                            self.0.access(2, "Footer", "dictionaries")
                        }

                        pub fn record_batches(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::BlockRef<'a>>>,
                        > {
                            self.0.access(3, "Footer", "record_batches")
                        }

                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(4, "Footer", "custom_metadata")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FooterRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FooterRef");
                            f.field("version", &self.version());
                            if let ::core::option::Option::Some(schema) = self.schema().transpose()
                            {
                                f.field("schema", &schema);
                            }
                            if let ::core::option::Option::Some(dictionaries) =
                                self.dictionaries().transpose()
                            {
                                f.field("dictionaries", &dictionaries);
                            }
                            if let ::core::option::Option::Some(record_batches) =
                                self.record_batches().transpose()
                            {
                                f.field("record_batches", &record_batches);
                            }
                            if let ::core::option::Option::Some(custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &custom_metadata);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FooterRef<'a>> for Footer {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FooterRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                version: ::core::convert::TryInto::try_into(value.version()?)?,
                                schema: if let ::core::option::Option::Some(schema) =
                                    value.schema()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(schema)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                dictionaries: if let ::core::option::Option::Some(dictionaries) =
                                    value.dictionaries()?
                                {
                                    ::core::option::Option::Some(dictionaries.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                record_batches: if let ::core::option::Option::Some(
                                    record_batches,
                                ) = value.record_batches()?
                                {
                                    ::core::option::Option::Some(record_batches.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FooterRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FooterRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FooterRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Footer>> for Footer {
                        type Value = ::planus::Offset<Footer>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Footer>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FooterRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[FooterRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    pub struct Block {
                        pub offset: i64,
                        pub meta_data_length: i32,
                        pub body_length: i64,
                    }

                    impl ::planus::Primitive for Block {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 24;
                    }

                    #[allow(clippy::identity_op)]
                    impl ::planus::WriteAsPrimitive<Block> for Block {
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            let (cur, cursor) = cursor.split::<8, 16>();
                            self.offset.write(cur, buffer_position - 0);
                            let (cur, cursor) = cursor.split::<4, 12>();
                            self.meta_data_length.write(cur, buffer_position - 8);
                            let cursor = cursor.write::<4, 8>([0; 4]);
                            let (cur, cursor) = cursor.split::<8, 0>();
                            self.body_length.write(cur, buffer_position - 16);
                            cursor.finish([]);
                        }
                    }

                    impl ::planus::WriteAs<Block> for Block {
                        type Prepared = Self;
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }
                    }

                    impl ::planus::WriteAsOptional<Block> for Block {
                        type Prepared = Self;
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Self> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct BlockRef<'a>(::planus::ArrayWithStartOffset<'a, 24>);

                    impl<'a> BlockRef<'a> {
                        pub fn offset(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(0).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }

                        pub fn meta_data_length(&self) -> i32 {
                            let buffer = self.0.advance_as_array::<4>(8).unwrap();

                            i32::from_le_bytes(*buffer.as_array())
                        }

                        pub fn body_length(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(16).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }
                    }

                    impl<'a> ::core::fmt::Debug for BlockRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BlockRef");
                            f.field("offset", &self.offset());
                            f.field("meta_data_length", &self.meta_data_length());
                            f.field("body_length", &self.body_length());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BlockRef<'a>> for Block {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: BlockRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Block {
                                offset: value.offset(),
                                meta_data_length: value.meta_data_length(),
                                body_length: value.body_length(),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BlockRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let buffer = buffer.advance_as_array::<24>(offset)?;
                            ::core::result::Result::Ok(Self(buffer))
                        }
                    }

                    impl<'a> ::planus::VectorRead<'a> for BlockRef<'a> {
                        const STRIDE: usize = 24;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> Self {
                            Self(buffer.unchecked_advance_as_array(offset))
                        }
                    }

                    impl ::planus::VectorWrite<Block> for Block {
                        const STRIDE: usize = 24;

                        type Value = Block;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Block],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 24];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (24 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum MetadataVersion {
                        V1 = 0,
                        V2 = 1,
                        V3 = 2,
                        V4 = 3,
                        V5 = 4,
                    }

                    impl ::core::convert::TryFrom<i16> for MetadataVersion {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(MetadataVersion::V1),
                                1 => ::core::result::Result::Ok(MetadataVersion::V2),
                                2 => ::core::result::Result::Ok(MetadataVersion::V3),
                                3 => ::core::result::Result::Ok(MetadataVersion::V4),
                                4 => ::core::result::Result::Ok(MetadataVersion::V5),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<MetadataVersion> for i16 {
                        fn from(value: MetadataVersion) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for MetadataVersion {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<MetadataVersion> for MetadataVersion {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<MetadataVersion> for MetadataVersion {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> MetadataVersion {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<MetadataVersion, MetadataVersion> for MetadataVersion {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &MetadataVersion,
                        ) -> ::core::option::Option<MetadataVersion> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<MetadataVersion> for MetadataVersion {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<MetadataVersion> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for MetadataVersion {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for MetadataVersion {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "MetadataVersion",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<MetadataVersion> for MetadataVersion {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i64)]
                    pub enum Feature {
                        Unused = 0,
                        DictionaryReplacement = 1,
                        CompressedBody = 2,
                    }

                    impl ::core::convert::TryFrom<i64> for Feature {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i64,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(Feature::Unused),
                                1 => ::core::result::Result::Ok(Feature::DictionaryReplacement),
                                2 => ::core::result::Result::Ok(Feature::CompressedBody),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<Feature> for i64 {
                        fn from(value: Feature) -> Self {
                            value as i64
                        }
                    }

                    impl ::planus::Primitive for Feature {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 8;
                    }

                    impl ::planus::WriteAsPrimitive<Feature> for Feature {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i64).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<Feature> for Feature {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Feature {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<Feature, Feature> for Feature {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &Feature,
                        ) -> ::core::option::Option<Feature> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<Feature> for Feature {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Feature> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for Feature {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i64 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for Feature {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 8;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i64 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "Feature",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<Feature> for Feature {
                        const STRIDE: usize = 8;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 8];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (8 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Null {}

                    impl Null {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Null>> for Null {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Null>> for Null {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Null>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Null> for Null {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Null> {
                            Null::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct NullRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> NullRef<'a> {}

                    impl<'a> ::core::fmt::Debug for NullRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("NullRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<NullRef<'a>> for Null {
                        type Error = ::planus::Error;

                        fn try_from(_value: NullRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for NullRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for NullRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[NullRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Null>> for Null {
                        type Value = ::planus::Offset<Null>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Null>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for NullRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[NullRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Struct {}

                    impl Struct {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Struct>> for Struct {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Struct>> for Struct {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Struct>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Struct> for Struct {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Struct> {
                            Struct::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct StructRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> StructRef<'a> {}

                    impl<'a> ::core::fmt::Debug for StructRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("StructRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<StructRef<'a>> for Struct {
                        type Error = ::planus::Error;

                        fn try_from(_value: StructRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for StructRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for StructRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[StructRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Struct>> for Struct {
                        type Value = ::planus::Offset<Struct>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Struct>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for StructRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[StructRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct List {}

                    impl List {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<List>> for List {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<List>> for List {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<List>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<List> for List {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<List> {
                            List::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct ListRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> ListRef<'a> {}

                    impl<'a> ::core::fmt::Debug for ListRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("ListRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<ListRef<'a>> for List {
                        type Error = ::planus::Error;

                        fn try_from(_value: ListRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for ListRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for ListRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[ListRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<List>> for List {
                        type Value = ::planus::Offset<List>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<List>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for ListRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[ListRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct LargeList {}

                    impl LargeList {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeList>> for LargeList {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeList>> for LargeList {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeList>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeList> for LargeList {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeList> {
                            LargeList::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct LargeListRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeListRef<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeListRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeListRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeListRef<'a>> for LargeList {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeListRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeListRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeListRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeListRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<LargeList>> for LargeList {
                        type Value = ::planus::Offset<LargeList>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeList>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeListRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[LargeListRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct ListView {}

                    impl ListView {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<ListView>> for ListView {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<ListView>> for ListView {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<ListView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<ListView> for ListView {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<ListView> {
                            ListView::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct ListViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> ListViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for ListViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("ListViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<ListViewRef<'a>> for ListView {
                        type Error = ::planus::Error;

                        fn try_from(_value: ListViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for ListViewRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for ListViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[ListViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<ListView>> for ListView {
                        type Value = ::planus::Offset<ListView>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<ListView>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for ListViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[ListViewRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct LargeListView {}

                    impl LargeListView {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeListView>> for LargeListView {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeListView>> for LargeListView {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeListView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeListView> for LargeListView {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeListView> {
                            LargeListView::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct LargeListViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeListViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeListViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeListViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeListViewRef<'a>> for LargeListView {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeListViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeListViewRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeListViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeListViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<LargeListView>> for LargeListView {
                        type Value = ::planus::Offset<LargeListView>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeListView>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeListViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeListViewRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct FixedSizeList {
                        pub list_size: i32,
                    }

                    impl FixedSizeList {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            list_size: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_list_size = list_size.prepare(builder, &0);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 4>::new(builder);

                            if prepared_list_size.is_some() {
                                table_writer.calculate_size::<i32>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_list_size) =
                                    prepared_list_size
                                {
                                    table_writer.write::<_, _, 4>(0, &prepared_list_size);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<FixedSizeList>> for FixedSizeList {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<FixedSizeList>> for FixedSizeList {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FixedSizeList>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<FixedSizeList> for FixedSizeList {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeList> {
                            FixedSizeList::create(builder, &self.list_size)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct FixedSizeListRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FixedSizeListRef<'a> {
                        pub fn list_size(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "FixedSizeList", "list_size")?.unwrap_or(0),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FixedSizeListRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FixedSizeListRef");
                            f.field("list_size", &self.list_size());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FixedSizeListRef<'a>> for FixedSizeList {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FixedSizeListRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                list_size: ::core::convert::TryInto::try_into(value.list_size()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FixedSizeListRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FixedSizeListRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeListRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<FixedSizeList>> for FixedSizeList {
                        type Value = ::planus::Offset<FixedSizeList>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<FixedSizeList>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FixedSizeListRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeListRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Map {
                        pub keys_sorted: bool,
                    }

                    impl Map {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            keys_sorted: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_keys_sorted = keys_sorted.prepare(builder, &false);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 1>::new(builder);

                            if prepared_keys_sorted.is_some() {
                                table_writer.calculate_size::<bool>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_keys_sorted) =
                                    prepared_keys_sorted
                                {
                                    table_writer.write::<_, _, 1>(0, &prepared_keys_sorted);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Map>> for Map {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Map>> for Map {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Map>> {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Map> for Map {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Map> {
                            Map::create(builder, &self.keys_sorted)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct MapRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> MapRef<'a> {
                        pub fn keys_sorted(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "Map", "keys_sorted")?.unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for MapRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("MapRef");
                            f.field("keys_sorted", &self.keys_sorted());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<MapRef<'a>> for Map {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: MapRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                keys_sorted: ::core::convert::TryInto::try_into(
                                    value.keys_sorted()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for MapRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for MapRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[MapRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Map>> for Map {
                        type Value = ::planus::Offset<Map>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Map>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for MapRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[MapRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum UnionMode {
                        Sparse = 0,
                        Dense = 1,
                    }

                    impl ::core::convert::TryFrom<i16> for UnionMode {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(UnionMode::Sparse),
                                1 => ::core::result::Result::Ok(UnionMode::Dense),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<UnionMode> for i16 {
                        fn from(value: UnionMode) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for UnionMode {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<UnionMode> for UnionMode {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<UnionMode> for UnionMode {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> UnionMode {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<UnionMode, UnionMode> for UnionMode {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &UnionMode,
                        ) -> ::core::option::Option<UnionMode> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<UnionMode> for UnionMode {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<UnionMode> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for UnionMode {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for UnionMode {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "UnionMode",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<UnionMode> for UnionMode {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Union {
                        pub mode: self::UnionMode,
                        pub type_ids: ::core::option::Option<::planus::alloc::vec::Vec<i32>>,
                    }

                    impl Union {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            mode: impl ::planus::WriteAsDefault<self::UnionMode, self::UnionMode>,
                            type_ids: impl ::planus::WriteAsOptional<::planus::Offset<[i32]>>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_mode = mode.prepare(builder, &self::UnionMode::Sparse);

                            let prepared_type_ids = type_ids.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 6>::new(builder);

                            if prepared_mode.is_some() {
                                table_writer.calculate_size::<self::UnionMode>(2);
                            }
                            if prepared_type_ids.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[i32]>>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_type_ids) =
                                    prepared_type_ids
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_type_ids);
                                }
                                if let ::core::option::Option::Some(prepared_mode) = prepared_mode {
                                    table_writer.write::<_, _, 2>(0, &prepared_mode);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Union>> for Union {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Union>> for Union {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Union>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Union> for Union {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Union> {
                            Union::create(builder, &self.mode, &self.type_ids)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct UnionRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> UnionRef<'a> {
                        pub fn mode(&self) -> ::planus::Result<self::UnionMode> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Union", "mode")?
                                    .unwrap_or(self::UnionMode::Sparse),
                            )
                        }

                        pub fn type_ids(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i32>>>
                        {
                            self.0.access(1, "Union", "type_ids")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for UnionRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("UnionRef");
                            f.field("mode", &self.mode());
                            if let ::core::option::Option::Some(type_ids) =
                                self.type_ids().transpose()
                            {
                                f.field("type_ids", &type_ids);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<UnionRef<'a>> for Union {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: UnionRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                mode: ::core::convert::TryInto::try_into(value.mode()?)?,
                                type_ids: if let ::core::option::Option::Some(type_ids) =
                                    value.type_ids()?
                                {
                                    ::core::option::Option::Some(type_ids.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for UnionRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for UnionRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[UnionRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Union>> for Union {
                        type Value = ::planus::Offset<Union>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Union>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for UnionRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[UnionRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Int {
                        pub bit_width: i32,
                        pub is_signed: bool,
                    }

                    impl Int {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            bit_width: impl ::planus::WriteAsDefault<i32, i32>,
                            is_signed: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_bit_width = bit_width.prepare(builder, &0);

                            let prepared_is_signed = is_signed.prepare(builder, &false);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 5>::new(builder);

                            if prepared_bit_width.is_some() {
                                table_writer.calculate_size::<i32>(2);
                            }
                            if prepared_is_signed.is_some() {
                                table_writer.calculate_size::<bool>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_bit_width) =
                                    prepared_bit_width
                                {
                                    table_writer.write::<_, _, 4>(0, &prepared_bit_width);
                                }
                                if let ::core::option::Option::Some(prepared_is_signed) =
                                    prepared_is_signed
                                {
                                    table_writer.write::<_, _, 1>(1, &prepared_is_signed);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Int>> for Int {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Int>> for Int {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Int>> {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Int> for Int {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Int> {
                            Int::create(builder, &self.bit_width, &self.is_signed)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct IntRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> IntRef<'a> {
                        pub fn bit_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "Int", "bit_width")?.unwrap_or(0),
                            )
                        }

                        pub fn is_signed(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Int", "is_signed")?.unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for IntRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("IntRef");
                            f.field("bit_width", &self.bit_width());
                            f.field("is_signed", &self.is_signed());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<IntRef<'a>> for Int {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: IntRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                bit_width: ::core::convert::TryInto::try_into(value.bit_width()?)?,
                                is_signed: ::core::convert::TryInto::try_into(value.is_signed()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for IntRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for IntRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[IntRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Int>> for Int {
                        type Value = ::planus::Offset<Int>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Int>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for IntRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[IntRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum Precision {
                        Half = 0,
                        Single = 1,
                        Double = 2,
                    }

                    impl ::core::convert::TryFrom<i16> for Precision {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(Precision::Half),
                                1 => ::core::result::Result::Ok(Precision::Single),
                                2 => ::core::result::Result::Ok(Precision::Double),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<Precision> for i16 {
                        fn from(value: Precision) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for Precision {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<Precision> for Precision {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<Precision> for Precision {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Precision {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<Precision, Precision> for Precision {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &Precision,
                        ) -> ::core::option::Option<Precision> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<Precision> for Precision {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Precision> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for Precision {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for Precision {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "Precision",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<Precision> for Precision {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct FloatingPoint {
                        pub precision: self::Precision,
                    }

                    impl FloatingPoint {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            precision: impl ::planus::WriteAsDefault<self::Precision, self::Precision>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_precision =
                                precision.prepare(builder, &self::Precision::Half);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 2>::new(builder);

                            if prepared_precision.is_some() {
                                table_writer.calculate_size::<self::Precision>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_precision) =
                                    prepared_precision
                                {
                                    table_writer.write::<_, _, 2>(0, &prepared_precision);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<FloatingPoint>> for FloatingPoint {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<FloatingPoint>> for FloatingPoint {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FloatingPoint>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<FloatingPoint> for FloatingPoint {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FloatingPoint> {
                            FloatingPoint::create(builder, &self.precision)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct FloatingPointRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FloatingPointRef<'a> {
                        pub fn precision(&self) -> ::planus::Result<self::Precision> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "FloatingPoint", "precision")?
                                    .unwrap_or(self::Precision::Half),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FloatingPointRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FloatingPointRef");
                            f.field("precision", &self.precision());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FloatingPointRef<'a>> for FloatingPoint {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FloatingPointRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                precision: ::core::convert::TryInto::try_into(value.precision()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FloatingPointRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FloatingPointRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FloatingPointRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<FloatingPoint>> for FloatingPoint {
                        type Value = ::planus::Offset<FloatingPoint>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<FloatingPoint>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FloatingPointRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FloatingPointRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Utf8 {}

                    impl Utf8 {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Utf8>> for Utf8 {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Utf8>> for Utf8 {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Utf8>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Utf8> for Utf8 {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8> {
                            Utf8::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct Utf8Ref<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> Utf8Ref<'a> {}

                    impl<'a> ::core::fmt::Debug for Utf8Ref<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("Utf8Ref");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<Utf8Ref<'a>> for Utf8 {
                        type Error = ::planus::Error;

                        fn try_from(_value: Utf8Ref<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for Utf8Ref<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for Utf8Ref<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[Utf8Ref]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Utf8>> for Utf8 {
                        type Value = ::planus::Offset<Utf8>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Utf8>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for Utf8Ref<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[Utf8Ref]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Binary {}

                    impl Binary {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Binary>> for Binary {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Binary>> for Binary {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Binary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Binary> for Binary {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Binary> {
                            Binary::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct BinaryRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BinaryRef<'a> {}

                    impl<'a> ::core::fmt::Debug for BinaryRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BinaryRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BinaryRef<'a>> for Binary {
                        type Error = ::planus::Error;

                        fn try_from(_value: BinaryRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BinaryRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BinaryRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BinaryRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Binary>> for Binary {
                        type Value = ::planus::Offset<Binary>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Binary>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BinaryRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[BinaryRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct LargeUtf8 {}

                    impl LargeUtf8 {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeUtf8>> for LargeUtf8 {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeUtf8>> for LargeUtf8 {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeUtf8>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeUtf8> for LargeUtf8 {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeUtf8> {
                            LargeUtf8::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct LargeUtf8Ref<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeUtf8Ref<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeUtf8Ref<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeUtf8Ref");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeUtf8Ref<'a>> for LargeUtf8 {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeUtf8Ref<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeUtf8Ref<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeUtf8Ref<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeUtf8Ref]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<LargeUtf8>> for LargeUtf8 {
                        type Value = ::planus::Offset<LargeUtf8>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeUtf8>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeUtf8Ref<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[LargeUtf8Ref]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct LargeBinary {}

                    impl LargeBinary {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<LargeBinary>> for LargeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<LargeBinary>> for LargeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<LargeBinary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<LargeBinary> for LargeBinary {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<LargeBinary> {
                            LargeBinary::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct LargeBinaryRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> LargeBinaryRef<'a> {}

                    impl<'a> ::core::fmt::Debug for LargeBinaryRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("LargeBinaryRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<LargeBinaryRef<'a>> for LargeBinary {
                        type Error = ::planus::Error;

                        fn try_from(_value: LargeBinaryRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for LargeBinaryRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for LargeBinaryRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeBinaryRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<LargeBinary>> for LargeBinary {
                        type Value = ::planus::Offset<LargeBinary>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<LargeBinary>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for LargeBinaryRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[LargeBinaryRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Utf8View {}

                    impl Utf8View {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Utf8View>> for Utf8View {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Utf8View>> for Utf8View {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Utf8View>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Utf8View> for Utf8View {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Utf8View> {
                            Utf8View::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct Utf8ViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> Utf8ViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for Utf8ViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("Utf8ViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<Utf8ViewRef<'a>> for Utf8View {
                        type Error = ::planus::Error;

                        fn try_from(_value: Utf8ViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for Utf8ViewRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for Utf8ViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[Utf8ViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Utf8View>> for Utf8View {
                        type Value = ::planus::Offset<Utf8View>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Utf8View>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for Utf8ViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[Utf8ViewRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct BinaryView {}

                    impl BinaryView {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<BinaryView>> for BinaryView {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<BinaryView>> for BinaryView {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<BinaryView>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<BinaryView> for BinaryView {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BinaryView> {
                            BinaryView::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct BinaryViewRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BinaryViewRef<'a> {}

                    impl<'a> ::core::fmt::Debug for BinaryViewRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BinaryViewRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BinaryViewRef<'a>> for BinaryView {
                        type Error = ::planus::Error;

                        fn try_from(_value: BinaryViewRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BinaryViewRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BinaryViewRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BinaryViewRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<BinaryView>> for BinaryView {
                        type Value = ::planus::Offset<BinaryView>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<BinaryView>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BinaryViewRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[BinaryViewRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct FixedSizeBinary {
                        pub byte_width: i32,
                    }

                    impl FixedSizeBinary {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            byte_width: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_byte_width = byte_width.prepare(builder, &0);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 4>::new(builder);

                            if prepared_byte_width.is_some() {
                                table_writer.calculate_size::<i32>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_byte_width) =
                                    prepared_byte_width
                                {
                                    table_writer.write::<_, _, 4>(0, &prepared_byte_width);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<FixedSizeBinary>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<FixedSizeBinary> for FixedSizeBinary {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<FixedSizeBinary> {
                            FixedSizeBinary::create(builder, &self.byte_width)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct FixedSizeBinaryRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FixedSizeBinaryRef<'a> {
                        pub fn byte_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "FixedSizeBinary", "byte_width")?
                                    .unwrap_or(0),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FixedSizeBinaryRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FixedSizeBinaryRef");
                            f.field("byte_width", &self.byte_width());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FixedSizeBinaryRef<'a>> for FixedSizeBinary {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FixedSizeBinaryRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                byte_width: ::core::convert::TryInto::try_into(
                                    value.byte_width()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FixedSizeBinaryRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FixedSizeBinaryRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeBinaryRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<FixedSizeBinary>> for FixedSizeBinary {
                        type Value = ::planus::Offset<FixedSizeBinary>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<FixedSizeBinary>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FixedSizeBinaryRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FixedSizeBinaryRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Bool {}

                    impl Bool {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Bool>> for Bool {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Bool>> for Bool {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Bool>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Bool> for Bool {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Bool> {
                            Bool::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct BoolRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BoolRef<'a> {}

                    impl<'a> ::core::fmt::Debug for BoolRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BoolRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BoolRef<'a>> for Bool {
                        type Error = ::planus::Error;

                        fn try_from(_value: BoolRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BoolRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BoolRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BoolRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Bool>> for Bool {
                        type Value = ::planus::Offset<Bool>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Bool>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BoolRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[BoolRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct RunEndEncoded {}

                    impl RunEndEncoded {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(builder: &mut ::planus::Builder) -> ::planus::Offset<Self> {
                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 0>::new(builder);

                            table_writer.finish_calculating();

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<RunEndEncoded>> for RunEndEncoded {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<RunEndEncoded>> for RunEndEncoded {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<RunEndEncoded>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<RunEndEncoded> for RunEndEncoded {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RunEndEncoded> {
                            RunEndEncoded::create(builder)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct RunEndEncodedRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> RunEndEncodedRef<'a> {}

                    impl<'a> ::core::fmt::Debug for RunEndEncodedRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("RunEndEncodedRef");

                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<RunEndEncodedRef<'a>> for RunEndEncoded {
                        type Error = ::planus::Error;

                        fn try_from(_value: RunEndEncodedRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {})
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for RunEndEncodedRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for RunEndEncodedRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RunEndEncodedRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<RunEndEncoded>> for RunEndEncoded {
                        type Value = ::planus::Offset<RunEndEncoded>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<RunEndEncoded>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for RunEndEncodedRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RunEndEncodedRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Decimal {
                        pub precision: i32,
                        pub scale: i32,
                        pub bit_width: i32,
                    }

                    impl Decimal {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            precision: impl ::planus::WriteAsDefault<i32, i32>,
                            scale: impl ::planus::WriteAsDefault<i32, i32>,
                            bit_width: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_precision = precision.prepare(builder, &0);

                            let prepared_scale = scale.prepare(builder, &0);

                            let prepared_bit_width = bit_width.prepare(builder, &128);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<8, 12>::new(builder);

                            if prepared_precision.is_some() {
                                table_writer.calculate_size::<i32>(2);
                            }
                            if prepared_scale.is_some() {
                                table_writer.calculate_size::<i32>(4);
                            }
                            if prepared_bit_width.is_some() {
                                table_writer.calculate_size::<i32>(6);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_precision) =
                                    prepared_precision
                                {
                                    table_writer.write::<_, _, 4>(0, &prepared_precision);
                                }
                                if let ::core::option::Option::Some(prepared_scale) = prepared_scale
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_scale);
                                }
                                if let ::core::option::Option::Some(prepared_bit_width) =
                                    prepared_bit_width
                                {
                                    table_writer.write::<_, _, 4>(2, &prepared_bit_width);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Decimal>> for Decimal {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Decimal>> for Decimal {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Decimal>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Decimal> for Decimal {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Decimal> {
                            Decimal::create(builder, &self.precision, &self.scale, &self.bit_width)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct DecimalRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DecimalRef<'a> {
                        pub fn precision(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "Decimal", "precision")?.unwrap_or(0),
                            )
                        }

                        pub fn scale(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Decimal", "scale")?.unwrap_or(0),
                            )
                        }

                        pub fn bit_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(2, "Decimal", "bit_width")?.unwrap_or(128),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DecimalRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DecimalRef");
                            f.field("precision", &self.precision());
                            f.field("scale", &self.scale());
                            f.field("bit_width", &self.bit_width());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DecimalRef<'a>> for Decimal {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DecimalRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                precision: ::core::convert::TryInto::try_into(value.precision()?)?,
                                scale: ::core::convert::TryInto::try_into(value.scale()?)?,
                                bit_width: ::core::convert::TryInto::try_into(value.bit_width()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DecimalRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DecimalRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DecimalRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Decimal>> for Decimal {
                        type Value = ::planus::Offset<Decimal>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Decimal>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DecimalRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[DecimalRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum DateUnit {
                        Day = 0,
                        Millisecond = 1,
                    }

                    impl ::core::convert::TryFrom<i16> for DateUnit {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(DateUnit::Day),
                                1 => ::core::result::Result::Ok(DateUnit::Millisecond),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<DateUnit> for i16 {
                        fn from(value: DateUnit) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for DateUnit {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<DateUnit> for DateUnit {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<DateUnit> for DateUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> DateUnit {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<DateUnit, DateUnit> for DateUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &DateUnit,
                        ) -> ::core::option::Option<DateUnit> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<DateUnit> for DateUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<DateUnit> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for DateUnit {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for DateUnit {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "DateUnit",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<DateUnit> for DateUnit {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Date {
                        pub unit: self::DateUnit,
                    }

                    impl Date {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            unit: impl ::planus::WriteAsDefault<self::DateUnit, self::DateUnit>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit = unit.prepare(builder, &self::DateUnit::Millisecond);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 2>::new(builder);

                            if prepared_unit.is_some() {
                                table_writer.calculate_size::<self::DateUnit>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_unit) = prepared_unit {
                                    table_writer.write::<_, _, 2>(0, &prepared_unit);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Date>> for Date {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Date>> for Date {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Date>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Date> for Date {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Date> {
                            Date::create(builder, &self.unit)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct DateRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DateRef<'a> {
                        pub fn unit(&self) -> ::planus::Result<self::DateUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Date", "unit")?
                                    .unwrap_or(self::DateUnit::Millisecond),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DateRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DateRef");
                            f.field("unit", &self.unit());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DateRef<'a>> for Date {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DateRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DateRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DateRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DateRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Date>> for Date {
                        type Value = ::planus::Offset<Date>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Date>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DateRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[DateRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum TimeUnit {
                        Second = 0,
                        Millisecond = 1,
                        Microsecond = 2,
                        Nanosecond = 3,
                    }

                    impl ::core::convert::TryFrom<i16> for TimeUnit {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(TimeUnit::Second),
                                1 => ::core::result::Result::Ok(TimeUnit::Millisecond),
                                2 => ::core::result::Result::Ok(TimeUnit::Microsecond),
                                3 => ::core::result::Result::Ok(TimeUnit::Nanosecond),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<TimeUnit> for i16 {
                        fn from(value: TimeUnit) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for TimeUnit {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<TimeUnit> for TimeUnit {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<TimeUnit> for TimeUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> TimeUnit {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<TimeUnit, TimeUnit> for TimeUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &TimeUnit,
                        ) -> ::core::option::Option<TimeUnit> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<TimeUnit> for TimeUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<TimeUnit> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for TimeUnit {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for TimeUnit {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "TimeUnit",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<TimeUnit> for TimeUnit {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Time {
                        pub unit: self::TimeUnit,
                        pub bit_width: i32,
                    }

                    impl Time {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            unit: impl ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            bit_width: impl ::planus::WriteAsDefault<i32, i32>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit = unit.prepare(builder, &self::TimeUnit::Millisecond);

                            let prepared_bit_width = bit_width.prepare(builder, &32);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 6>::new(builder);

                            if prepared_unit.is_some() {
                                table_writer.calculate_size::<self::TimeUnit>(2);
                            }
                            if prepared_bit_width.is_some() {
                                table_writer.calculate_size::<i32>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_bit_width) =
                                    prepared_bit_width
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_bit_width);
                                }
                                if let ::core::option::Option::Some(prepared_unit) = prepared_unit {
                                    table_writer.write::<_, _, 2>(0, &prepared_unit);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Time>> for Time {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Time>> for Time {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Time>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Time> for Time {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Time> {
                            Time::create(builder, &self.unit, &self.bit_width)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct TimeRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TimeRef<'a> {
                        pub fn unit(&self) -> ::planus::Result<self::TimeUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Time", "unit")?
                                    .unwrap_or(self::TimeUnit::Millisecond),
                            )
                        }

                        pub fn bit_width(&self) -> ::planus::Result<i32> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Time", "bit_width")?.unwrap_or(32),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TimeRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TimeRef");
                            f.field("unit", &self.unit());
                            f.field("bit_width", &self.bit_width());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TimeRef<'a>> for Time {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TimeRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                                bit_width: ::core::convert::TryInto::try_into(value.bit_width()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TimeRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TimeRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TimeRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Time>> for Time {
                        type Value = ::planus::Offset<Time>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Time>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TimeRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TimeRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Timestamp {
                        pub unit: self::TimeUnit,
                        pub timezone: ::core::option::Option<::planus::alloc::string::String>,
                    }

                    impl Timestamp {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            unit: impl ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                            timezone: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit = unit.prepare(builder, &self::TimeUnit::Second);

                            let prepared_timezone = timezone.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 6>::new(builder);

                            if prepared_unit.is_some() {
                                table_writer.calculate_size::<self::TimeUnit>(2);
                            }
                            if prepared_timezone.is_some() {
                                table_writer.calculate_size::<::planus::Offset<str>>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_timezone) =
                                    prepared_timezone
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_timezone);
                                }
                                if let ::core::option::Option::Some(prepared_unit) = prepared_unit {
                                    table_writer.write::<_, _, 2>(0, &prepared_unit);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Timestamp>> for Timestamp {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Timestamp>> for Timestamp {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Timestamp>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Timestamp> for Timestamp {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Timestamp> {
                            Timestamp::create(builder, &self.unit, &self.timezone)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct TimestampRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TimestampRef<'a> {
                        pub fn unit(&self) -> ::planus::Result<self::TimeUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Timestamp", "unit")?
                                    .unwrap_or(self::TimeUnit::Second),
                            )
                        }

                        pub fn timezone(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(1, "Timestamp", "timezone")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TimestampRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TimestampRef");
                            f.field("unit", &self.unit());
                            if let ::core::option::Option::Some(timezone) =
                                self.timezone().transpose()
                            {
                                f.field("timezone", &timezone);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TimestampRef<'a>> for Timestamp {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TimestampRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                                timezone: if let ::core::option::Option::Some(timezone) =
                                    value.timezone()?
                                {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(timezone)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TimestampRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TimestampRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TimestampRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Timestamp>> for Timestamp {
                        type Value = ::planus::Offset<Timestamp>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Timestamp>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TimestampRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TimestampRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum IntervalUnit {
                        YearMonth = 0,
                        DayTime = 1,
                        MonthDayNano = 2,
                    }

                    impl ::core::convert::TryFrom<i16> for IntervalUnit {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(IntervalUnit::YearMonth),
                                1 => ::core::result::Result::Ok(IntervalUnit::DayTime),
                                2 => ::core::result::Result::Ok(IntervalUnit::MonthDayNano),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<IntervalUnit> for i16 {
                        fn from(value: IntervalUnit) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for IntervalUnit {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<IntervalUnit> for IntervalUnit {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<IntervalUnit> for IntervalUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> IntervalUnit {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<IntervalUnit, IntervalUnit> for IntervalUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &IntervalUnit,
                        ) -> ::core::option::Option<IntervalUnit> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<IntervalUnit> for IntervalUnit {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<IntervalUnit> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for IntervalUnit {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for IntervalUnit {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "IntervalUnit",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<IntervalUnit> for IntervalUnit {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Interval {
                        pub unit: self::IntervalUnit,
                    }

                    impl Interval {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            unit: impl ::planus::WriteAsDefault<self::IntervalUnit, self::IntervalUnit>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit =
                                unit.prepare(builder, &self::IntervalUnit::YearMonth);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 2>::new(builder);

                            if prepared_unit.is_some() {
                                table_writer.calculate_size::<self::IntervalUnit>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_unit) = prepared_unit {
                                    table_writer.write::<_, _, 2>(0, &prepared_unit);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Interval>> for Interval {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Interval>> for Interval {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Interval>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Interval> for Interval {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Interval> {
                            Interval::create(builder, &self.unit)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct IntervalRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> IntervalRef<'a> {
                        pub fn unit(&self) -> ::planus::Result<self::IntervalUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Interval", "unit")?
                                    .unwrap_or(self::IntervalUnit::YearMonth),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for IntervalRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("IntervalRef");
                            f.field("unit", &self.unit());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<IntervalRef<'a>> for Interval {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: IntervalRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for IntervalRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for IntervalRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[IntervalRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Interval>> for Interval {
                        type Value = ::planus::Offset<Interval>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Interval>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for IntervalRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[IntervalRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Duration {
                        pub unit: self::TimeUnit,
                    }

                    impl Duration {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            unit: impl ::planus::WriteAsDefault<self::TimeUnit, self::TimeUnit>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_unit = unit.prepare(builder, &self::TimeUnit::Millisecond);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<4, 2>::new(builder);

                            if prepared_unit.is_some() {
                                table_writer.calculate_size::<self::TimeUnit>(2);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_unit) = prepared_unit {
                                    table_writer.write::<_, _, 2>(0, &prepared_unit);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Duration>> for Duration {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Duration>> for Duration {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Duration>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Duration> for Duration {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Duration> {
                            Duration::create(builder, &self.unit)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct DurationRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DurationRef<'a> {
                        pub fn unit(&self) -> ::planus::Result<self::TimeUnit> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Duration", "unit")?
                                    .unwrap_or(self::TimeUnit::Millisecond),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DurationRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DurationRef");
                            f.field("unit", &self.unit());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DurationRef<'a>> for Duration {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DurationRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                unit: ::core::convert::TryInto::try_into(value.unit()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DurationRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DurationRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DurationRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Duration>> for Duration {
                        type Value = ::planus::Offset<Duration>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Duration>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DurationRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[DurationRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub enum Type {
                        Null(::planus::alloc::boxed::Box<self::Null>),
                        Int(::planus::alloc::boxed::Box<self::Int>),
                        FloatingPoint(::planus::alloc::boxed::Box<self::FloatingPoint>),
                        Binary(::planus::alloc::boxed::Box<self::Binary>),
                        Utf8(::planus::alloc::boxed::Box<self::Utf8>),
                        Bool(::planus::alloc::boxed::Box<self::Bool>),
                        Decimal(::planus::alloc::boxed::Box<self::Decimal>),
                        Date(::planus::alloc::boxed::Box<self::Date>),
                        Time(::planus::alloc::boxed::Box<self::Time>),
                        Timestamp(::planus::alloc::boxed::Box<self::Timestamp>),
                        Interval(::planus::alloc::boxed::Box<self::Interval>),
                        List(::planus::alloc::boxed::Box<self::List>),
                        Struct(::planus::alloc::boxed::Box<self::Struct>),
                        Union(::planus::alloc::boxed::Box<self::Union>),
                        FixedSizeBinary(::planus::alloc::boxed::Box<self::FixedSizeBinary>),
                        FixedSizeList(::planus::alloc::boxed::Box<self::FixedSizeList>),
                        Map(::planus::alloc::boxed::Box<self::Map>),
                        Duration(::planus::alloc::boxed::Box<self::Duration>),
                        LargeBinary(::planus::alloc::boxed::Box<self::LargeBinary>),
                        LargeUtf8(::planus::alloc::boxed::Box<self::LargeUtf8>),
                        LargeList(::planus::alloc::boxed::Box<self::LargeList>),
                        RunEndEncoded(::planus::alloc::boxed::Box<self::RunEndEncoded>),
                        BinaryView(::planus::alloc::boxed::Box<self::BinaryView>),
                        Utf8View(::planus::alloc::boxed::Box<self::Utf8View>),
                        ListView(::planus::alloc::boxed::Box<self::ListView>),
                        LargeListView(::planus::alloc::boxed::Box<self::LargeListView>),
                    }

                    impl Type {
                        pub fn create_null(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Null>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
                        }

                        pub fn create_int(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Int>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
                        }

                        pub fn create_floating_point(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::FloatingPoint>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
                        }

                        pub fn create_binary(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Binary>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(4, value.prepare(builder).downcast())
                        }

                        pub fn create_utf8(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Utf8>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(5, value.prepare(builder).downcast())
                        }

                        pub fn create_bool(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Bool>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(6, value.prepare(builder).downcast())
                        }

                        pub fn create_decimal(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Decimal>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(7, value.prepare(builder).downcast())
                        }

                        pub fn create_date(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Date>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(8, value.prepare(builder).downcast())
                        }

                        pub fn create_time(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Time>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(9, value.prepare(builder).downcast())
                        }

                        pub fn create_timestamp(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Timestamp>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(10, value.prepare(builder).downcast())
                        }

                        pub fn create_interval(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Interval>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(11, value.prepare(builder).downcast())
                        }

                        pub fn create_list(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::List>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(12, value.prepare(builder).downcast())
                        }

                        pub fn create_struct(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Struct>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(13, value.prepare(builder).downcast())
                        }

                        pub fn create_union(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Union>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(14, value.prepare(builder).downcast())
                        }

                        pub fn create_fixed_size_binary(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::FixedSizeBinary>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(15, value.prepare(builder).downcast())
                        }

                        pub fn create_fixed_size_list(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::FixedSizeList>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(16, value.prepare(builder).downcast())
                        }

                        pub fn create_map(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Map>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(17, value.prepare(builder).downcast())
                        }

                        pub fn create_duration(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Duration>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(18, value.prepare(builder).downcast())
                        }

                        pub fn create_large_binary(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeBinary>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(19, value.prepare(builder).downcast())
                        }

                        pub fn create_large_utf8(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeUtf8>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(20, value.prepare(builder).downcast())
                        }

                        pub fn create_large_list(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeList>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(21, value.prepare(builder).downcast())
                        }

                        pub fn create_run_end_encoded(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::RunEndEncoded>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(22, value.prepare(builder).downcast())
                        }

                        pub fn create_binary_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::BinaryView>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(23, value.prepare(builder).downcast())
                        }

                        pub fn create_utf8_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Utf8View>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(24, value.prepare(builder).downcast())
                        }

                        pub fn create_list_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::ListView>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(25, value.prepare(builder).downcast())
                        }

                        pub fn create_large_list_view(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::LargeListView>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(26, value.prepare(builder).downcast())
                        }
                    }

                    impl ::planus::WriteAsUnion<Type> for Type {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Self> {
                            match self {
                                Self::Null(value) => Self::create_null(builder, value),
                                Self::Int(value) => Self::create_int(builder, value),
                                Self::FloatingPoint(value) => {
                                    Self::create_floating_point(builder, value)
                                }
                                Self::Binary(value) => Self::create_binary(builder, value),
                                Self::Utf8(value) => Self::create_utf8(builder, value),
                                Self::Bool(value) => Self::create_bool(builder, value),
                                Self::Decimal(value) => Self::create_decimal(builder, value),
                                Self::Date(value) => Self::create_date(builder, value),
                                Self::Time(value) => Self::create_time(builder, value),
                                Self::Timestamp(value) => Self::create_timestamp(builder, value),
                                Self::Interval(value) => Self::create_interval(builder, value),
                                Self::List(value) => Self::create_list(builder, value),
                                Self::Struct(value) => Self::create_struct(builder, value),
                                Self::Union(value) => Self::create_union(builder, value),
                                Self::FixedSizeBinary(value) => {
                                    Self::create_fixed_size_binary(builder, value)
                                }
                                Self::FixedSizeList(value) => {
                                    Self::create_fixed_size_list(builder, value)
                                }
                                Self::Map(value) => Self::create_map(builder, value),
                                Self::Duration(value) => Self::create_duration(builder, value),
                                Self::LargeBinary(value) => {
                                    Self::create_large_binary(builder, value)
                                }
                                Self::LargeUtf8(value) => Self::create_large_utf8(builder, value),
                                Self::LargeList(value) => Self::create_large_list(builder, value),
                                Self::RunEndEncoded(value) => {
                                    Self::create_run_end_encoded(builder, value)
                                }
                                Self::BinaryView(value) => Self::create_binary_view(builder, value),
                                Self::Utf8View(value) => Self::create_utf8_view(builder, value),
                                Self::ListView(value) => Self::create_list_view(builder, value),
                                Self::LargeListView(value) => {
                                    Self::create_large_list_view(builder, value)
                                }
                            }
                        }
                    }

                    impl ::planus::WriteAsOptionalUnion<Type> for Type {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Self>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    #[derive(Copy, Clone, Debug)]
                    pub enum TypeRef<'a> {
                        Null(self::NullRef<'a>),
                        Int(self::IntRef<'a>),
                        FloatingPoint(self::FloatingPointRef<'a>),
                        Binary(self::BinaryRef<'a>),
                        Utf8(self::Utf8Ref<'a>),
                        Bool(self::BoolRef<'a>),
                        Decimal(self::DecimalRef<'a>),
                        Date(self::DateRef<'a>),
                        Time(self::TimeRef<'a>),
                        Timestamp(self::TimestampRef<'a>),
                        Interval(self::IntervalRef<'a>),
                        List(self::ListRef<'a>),
                        Struct(self::StructRef<'a>),
                        Union(self::UnionRef<'a>),
                        FixedSizeBinary(self::FixedSizeBinaryRef<'a>),
                        FixedSizeList(self::FixedSizeListRef<'a>),
                        Map(self::MapRef<'a>),
                        Duration(self::DurationRef<'a>),
                        LargeBinary(self::LargeBinaryRef<'a>),
                        LargeUtf8(self::LargeUtf8Ref<'a>),
                        LargeList(self::LargeListRef<'a>),
                        RunEndEncoded(self::RunEndEncodedRef<'a>),
                        BinaryView(self::BinaryViewRef<'a>),
                        Utf8View(self::Utf8ViewRef<'a>),
                        ListView(self::ListViewRef<'a>),
                        LargeListView(self::LargeListViewRef<'a>),
                    }

                    impl<'a> ::core::convert::TryFrom<TypeRef<'a>> for Type {
                        type Error = ::planus::Error;

                        fn try_from(value: TypeRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(match value {
                                TypeRef::Null(value) => {
                                    Type::Null(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Int(value) => Type::Int(::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryFrom::try_from(value)?,
                                )),

                                TypeRef::FloatingPoint(value) => {
                                    Type::FloatingPoint(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Binary(value) => {
                                    Type::Binary(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Utf8(value) => {
                                    Type::Utf8(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Bool(value) => {
                                    Type::Bool(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Decimal(value) => {
                                    Type::Decimal(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Date(value) => {
                                    Type::Date(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Time(value) => {
                                    Type::Time(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Timestamp(value) => {
                                    Type::Timestamp(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Interval(value) => {
                                    Type::Interval(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::List(value) => {
                                    Type::List(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Struct(value) => {
                                    Type::Struct(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Union(value) => {
                                    Type::Union(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::FixedSizeBinary(value) => {
                                    Type::FixedSizeBinary(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::FixedSizeList(value) => {
                                    Type::FixedSizeList(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Map(value) => Type::Map(::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryFrom::try_from(value)?,
                                )),

                                TypeRef::Duration(value) => {
                                    Type::Duration(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeBinary(value) => {
                                    Type::LargeBinary(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeUtf8(value) => {
                                    Type::LargeUtf8(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeList(value) => {
                                    Type::LargeList(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::RunEndEncoded(value) => {
                                    Type::RunEndEncoded(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::BinaryView(value) => {
                                    Type::BinaryView(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::Utf8View(value) => {
                                    Type::Utf8View(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::ListView(value) => {
                                    Type::ListView(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                TypeRef::LargeListView(value) => {
                                    Type::LargeListView(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }
                            })
                        }
                    }

                    impl<'a> ::planus::TableReadUnion<'a> for TypeRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            field_offset: usize,
                            tag: u8,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            match tag {
                                1 => ::core::result::Result::Ok(Self::Null(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                2 => ::core::result::Result::Ok(Self::Int(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                3 => ::core::result::Result::Ok(Self::FloatingPoint(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                4 => ::core::result::Result::Ok(Self::Binary(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                5 => ::core::result::Result::Ok(Self::Utf8(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                6 => ::core::result::Result::Ok(Self::Bool(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                7 => ::core::result::Result::Ok(Self::Decimal(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                8 => ::core::result::Result::Ok(Self::Date(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                9 => ::core::result::Result::Ok(Self::Time(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                10 => ::core::result::Result::Ok(Self::Timestamp(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                11 => ::core::result::Result::Ok(Self::Interval(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                12 => ::core::result::Result::Ok(Self::List(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                13 => ::core::result::Result::Ok(Self::Struct(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                14 => ::core::result::Result::Ok(Self::Union(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                15 => ::core::result::Result::Ok(Self::FixedSizeBinary(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                16 => ::core::result::Result::Ok(Self::FixedSizeList(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                17 => ::core::result::Result::Ok(Self::Map(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                18 => ::core::result::Result::Ok(Self::Duration(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                19 => ::core::result::Result::Ok(Self::LargeBinary(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                20 => ::core::result::Result::Ok(Self::LargeUtf8(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                21 => ::core::result::Result::Ok(Self::LargeList(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                22 => ::core::result::Result::Ok(Self::RunEndEncoded(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                23 => ::core::result::Result::Ok(Self::BinaryView(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                24 => ::core::result::Result::Ok(Self::Utf8View(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                25 => ::core::result::Result::Ok(Self::ListView(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                26 => ::core::result::Result::Ok(Self::LargeListView(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                _ => ::core::result::Result::Err(
                                    ::planus::errors::ErrorKind::UnknownUnionTag { tag },
                                ),
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct KeyValue {
                        pub key: ::core::option::Option<::planus::alloc::string::String>,
                        pub value: ::core::option::Option<::planus::alloc::string::String>,
                    }

                    impl KeyValue {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            key: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                            value: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_key = key.prepare(builder);

                            let prepared_value = value.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 8>::new(builder);

                            if prepared_key.is_some() {
                                table_writer.calculate_size::<::planus::Offset<str>>(2);
                            }
                            if prepared_value.is_some() {
                                table_writer.calculate_size::<::planus::Offset<str>>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_key) = prepared_key {
                                    table_writer.write::<_, _, 4>(0, &prepared_key);
                                }
                                if let ::core::option::Option::Some(prepared_value) = prepared_value
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_value);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<KeyValue>> for KeyValue {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<KeyValue>> for KeyValue {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<KeyValue>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<KeyValue> for KeyValue {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<KeyValue> {
                            KeyValue::create(builder, &self.key, &self.value)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct KeyValueRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> KeyValueRef<'a> {
                        pub fn key(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(0, "KeyValue", "key")
                        }

                        pub fn value(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(1, "KeyValue", "value")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for KeyValueRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("KeyValueRef");
                            if let ::core::option::Option::Some(key) = self.key().transpose() {
                                f.field("key", &key);
                            }
                            if let ::core::option::Option::Some(value) = self.value().transpose() {
                                f.field("value", &value);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<KeyValueRef<'a>> for KeyValue {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: KeyValueRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                key: if let ::core::option::Option::Some(key) = value.key()? {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(key)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                                value: if let ::core::option::Option::Some(value) = value.value()? {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(value)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for KeyValueRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for KeyValueRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[KeyValueRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<KeyValue>> for KeyValue {
                        type Value = ::planus::Offset<KeyValue>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<KeyValue>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for KeyValueRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[KeyValueRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum DictionaryKind {
                        DenseArray = 0,
                    }

                    impl ::core::convert::TryFrom<i16> for DictionaryKind {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(DictionaryKind::DenseArray),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<DictionaryKind> for i16 {
                        fn from(value: DictionaryKind) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for DictionaryKind {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<DictionaryKind> for DictionaryKind {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<DictionaryKind> for DictionaryKind {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> DictionaryKind {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<DictionaryKind, DictionaryKind> for DictionaryKind {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &DictionaryKind,
                        ) -> ::core::option::Option<DictionaryKind> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<DictionaryKind> for DictionaryKind {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<DictionaryKind> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for DictionaryKind {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for DictionaryKind {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "DictionaryKind",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<DictionaryKind> for DictionaryKind {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct DictionaryEncoding {
                        pub id: i64,
                        pub index_type:
                            ::core::option::Option<::planus::alloc::boxed::Box<self::Int>>,
                        pub is_ordered: bool,
                        pub dictionary_kind: self::DictionaryKind,
                    }

                    impl DictionaryEncoding {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            id: impl ::planus::WriteAsDefault<i64, i64>,
                            index_type: impl ::planus::WriteAsOptional<::planus::Offset<self::Int>>,
                            is_ordered: impl ::planus::WriteAsDefault<bool, bool>,
                            dictionary_kind: impl ::planus::WriteAsDefault<
                                self::DictionaryKind,
                                self::DictionaryKind,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_id = id.prepare(builder, &0);

                            let prepared_index_type = index_type.prepare(builder);

                            let prepared_is_ordered = is_ordered.prepare(builder, &false);

                            let prepared_dictionary_kind =
                                dictionary_kind.prepare(builder, &self::DictionaryKind::DenseArray);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<10, 15>::new(builder);

                            if prepared_id.is_some() {
                                table_writer.calculate_size::<i64>(2);
                            }
                            if prepared_index_type.is_some() {
                                table_writer.calculate_size::<::planus::Offset<self::Int>>(4);
                            }
                            if prepared_is_ordered.is_some() {
                                table_writer.calculate_size::<bool>(6);
                            }
                            if prepared_dictionary_kind.is_some() {
                                table_writer.calculate_size::<self::DictionaryKind>(8);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_id) = prepared_id {
                                    table_writer.write::<_, _, 8>(0, &prepared_id);
                                }
                                if let ::core::option::Option::Some(prepared_index_type) =
                                    prepared_index_type
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_index_type);
                                }
                                if let ::core::option::Option::Some(prepared_dictionary_kind) =
                                    prepared_dictionary_kind
                                {
                                    table_writer.write::<_, _, 2>(3, &prepared_dictionary_kind);
                                }
                                if let ::core::option::Option::Some(prepared_is_ordered) =
                                    prepared_is_ordered
                                {
                                    table_writer.write::<_, _, 1>(2, &prepared_is_ordered);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<DictionaryEncoding>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<DictionaryEncoding> for DictionaryEncoding {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryEncoding> {
                            DictionaryEncoding::create(
                                builder,
                                &self.id,
                                &self.index_type,
                                &self.is_ordered,
                                &self.dictionary_kind,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct DictionaryEncodingRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DictionaryEncodingRef<'a> {
                        pub fn id(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "DictionaryEncoding", "id")?.unwrap_or(0),
                            )
                        }

                        pub fn index_type(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::IntRef<'a>>>
                        {
                            self.0.access(1, "DictionaryEncoding", "index_type")
                        }

                        pub fn is_ordered(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(2, "DictionaryEncoding", "is_ordered")?
                                    .unwrap_or(false),
                            )
                        }

                        pub fn dictionary_kind(&self) -> ::planus::Result<self::DictionaryKind> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(3, "DictionaryEncoding", "dictionary_kind")?
                                    .unwrap_or(self::DictionaryKind::DenseArray),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DictionaryEncodingRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DictionaryEncodingRef");
                            f.field("id", &self.id());
                            if let ::core::option::Option::Some(index_type) =
                                self.index_type().transpose()
                            {
                                f.field("index_type", &index_type);
                            }
                            f.field("is_ordered", &self.is_ordered());
                            f.field("dictionary_kind", &self.dictionary_kind());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DictionaryEncodingRef<'a>> for DictionaryEncoding {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DictionaryEncodingRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                id: ::core::convert::TryInto::try_into(value.id()?)?,
                                index_type: if let ::core::option::Option::Some(index_type) =
                                    value.index_type()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(index_type)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                is_ordered: ::core::convert::TryInto::try_into(
                                    value.is_ordered()?,
                                )?,
                                dictionary_kind: ::core::convert::TryInto::try_into(
                                    value.dictionary_kind()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DictionaryEncodingRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DictionaryEncodingRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryEncodingRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<DictionaryEncoding>> for DictionaryEncoding {
                        type Value = ::planus::Offset<DictionaryEncoding>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<DictionaryEncoding>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DictionaryEncodingRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryEncodingRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Field {
                        pub name: ::core::option::Option<::planus::alloc::string::String>,
                        pub nullable: bool,
                        pub type_: ::core::option::Option<self::Type>,
                        pub dictionary: ::core::option::Option<
                            ::planus::alloc::boxed::Box<self::DictionaryEncoding>,
                        >,
                        pub children:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Field>>,
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                    }

                    impl Field {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            name: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                            nullable: impl ::planus::WriteAsDefault<bool, bool>,
                            type_: impl ::planus::WriteAsOptionalUnion<self::Type>,
                            dictionary: impl ::planus::WriteAsOptional<
                                ::planus::Offset<self::DictionaryEncoding>,
                            >,
                            children: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_name = name.prepare(builder);

                            let prepared_nullable = nullable.prepare(builder, &false);

                            let prepared_type_ = type_.prepare(builder);

                            let prepared_dictionary = dictionary.prepare(builder);

                            let prepared_children = children.prepare(builder);

                            let prepared_custom_metadata = custom_metadata.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<16, 22>::new(builder);

                            if prepared_name.is_some() {
                                table_writer.calculate_size::<::planus::Offset<str>>(2);
                            }
                            if prepared_nullable.is_some() {
                                table_writer.calculate_size::<bool>(4);
                            }
                            if prepared_type_.is_some() {
                                table_writer.calculate_size::<u8>(6);
                                table_writer.calculate_size::<::planus::Offset<self::Type>>(8);
                            }
                            if prepared_dictionary.is_some() {
                                table_writer
                                    .calculate_size::<::planus::Offset<self::DictionaryEncoding>>(
                                        10,
                                    );
                            }
                            if prepared_children.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::Field>]>>(12);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(14);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_name) = prepared_name {
                                    table_writer.write::<_, _, 4>(0, &prepared_name);
                                }
                                if let ::core::option::Option::Some(prepared_type_) = prepared_type_
                                {
                                    table_writer.write::<_, _, 4>(3, &prepared_type_.offset());
                                }
                                if let ::core::option::Option::Some(prepared_dictionary) =
                                    prepared_dictionary
                                {
                                    table_writer.write::<_, _, 4>(4, &prepared_dictionary);
                                }
                                if let ::core::option::Option::Some(prepared_children) =
                                    prepared_children
                                {
                                    table_writer.write::<_, _, 4>(5, &prepared_children);
                                }
                                if let ::core::option::Option::Some(prepared_custom_metadata) =
                                    prepared_custom_metadata
                                {
                                    table_writer.write::<_, _, 4>(6, &prepared_custom_metadata);
                                }
                                if let ::core::option::Option::Some(prepared_nullable) =
                                    prepared_nullable
                                {
                                    table_writer.write::<_, _, 1>(1, &prepared_nullable);
                                }
                                if let ::core::option::Option::Some(prepared_type_) = prepared_type_
                                {
                                    table_writer.write::<_, _, 1>(2, &prepared_type_.tag());
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Field>> for Field {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Field>> for Field {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Field>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Field> for Field {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Field> {
                            Field::create(
                                builder,
                                &self.name,
                                &self.nullable,
                                &self.type_,
                                &self.dictionary,
                                &self.children,
                                &self.custom_metadata,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct FieldRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> FieldRef<'a> {
                        pub fn name(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(0, "Field", "name")
                        }

                        pub fn nullable(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0.access(1, "Field", "nullable")?.unwrap_or(false),
                            )
                        }

                        pub fn type_(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::TypeRef<'a>>>
                        {
                            self.0.access_union(2, "Field", "type_")
                        }

                        pub fn dictionary(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::DictionaryEncodingRef<'a>>>
                        {
                            self.0.access(4, "Field", "dictionary")
                        }

                        pub fn children(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::FieldRef<'a>>>,
                            >,
                        > {
                            self.0.access(5, "Field", "children")
                        }

                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(6, "Field", "custom_metadata")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FieldRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FieldRef");
                            if let ::core::option::Option::Some(name) = self.name().transpose() {
                                f.field("name", &name);
                            }
                            f.field("nullable", &self.nullable());
                            if let ::core::option::Option::Some(type_) = self.type_().transpose() {
                                f.field("type_", &type_);
                            }
                            if let ::core::option::Option::Some(dictionary) =
                                self.dictionary().transpose()
                            {
                                f.field("dictionary", &dictionary);
                            }
                            if let ::core::option::Option::Some(children) =
                                self.children().transpose()
                            {
                                f.field("children", &children);
                            }
                            if let ::core::option::Option::Some(custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &custom_metadata);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FieldRef<'a>> for Field {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FieldRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                name: if let ::core::option::Option::Some(name) = value.name()? {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(name)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                                nullable: ::core::convert::TryInto::try_into(value.nullable()?)?,
                                type_: if let ::core::option::Option::Some(type_) = value.type_()? {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(type_)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                                dictionary: if let ::core::option::Option::Some(dictionary) =
                                    value.dictionary()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(dictionary)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                children: if let ::core::option::Option::Some(children) =
                                    value.children()?
                                {
                                    ::core::option::Option::Some(children.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FieldRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for FieldRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[FieldRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Field>> for Field {
                        type Value = ::planus::Offset<Field>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Field>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for FieldRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[FieldRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum Endianness {
                        Little = 0,
                        Big = 1,
                    }

                    impl ::core::convert::TryFrom<i16> for Endianness {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(Endianness::Little),
                                1 => ::core::result::Result::Ok(Endianness::Big),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<Endianness> for i16 {
                        fn from(value: Endianness) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for Endianness {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<Endianness> for Endianness {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<Endianness> for Endianness {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Endianness {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<Endianness, Endianness> for Endianness {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &Endianness,
                        ) -> ::core::option::Option<Endianness> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<Endianness> for Endianness {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Endianness> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for Endianness {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for Endianness {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "Endianness",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<Endianness> for Endianness {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    pub struct Buffer {
                        pub offset: i64,
                        pub length: i64,
                    }

                    impl ::planus::Primitive for Buffer {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 16;
                    }

                    #[allow(clippy::identity_op)]
                    impl ::planus::WriteAsPrimitive<Buffer> for Buffer {
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            let (cur, cursor) = cursor.split::<8, 8>();
                            self.offset.write(cur, buffer_position - 0);
                            let (cur, cursor) = cursor.split::<8, 0>();
                            self.length.write(cur, buffer_position - 8);
                            cursor.finish([]);
                        }
                    }

                    impl ::planus::WriteAs<Buffer> for Buffer {
                        type Prepared = Self;
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }
                    }

                    impl ::planus::WriteAsOptional<Buffer> for Buffer {
                        type Prepared = Self;
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Self> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct BufferRef<'a>(::planus::ArrayWithStartOffset<'a, 16>);

                    impl<'a> BufferRef<'a> {
                        pub fn offset(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(0).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }

                        pub fn length(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(8).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }
                    }

                    impl<'a> ::core::fmt::Debug for BufferRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BufferRef");
                            f.field("offset", &self.offset());
                            f.field("length", &self.length());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BufferRef<'a>> for Buffer {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: BufferRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Buffer {
                                offset: value.offset(),
                                length: value.length(),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BufferRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let buffer = buffer.advance_as_array::<16>(offset)?;
                            ::core::result::Result::Ok(Self(buffer))
                        }
                    }

                    impl<'a> ::planus::VectorRead<'a> for BufferRef<'a> {
                        const STRIDE: usize = 16;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> Self {
                            Self(buffer.unchecked_advance_as_array(offset))
                        }
                    }

                    impl ::planus::VectorWrite<Buffer> for Buffer {
                        const STRIDE: usize = 16;

                        type Value = Buffer;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Buffer],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 16];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (16 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Schema {
                        pub endianness: self::Endianness,
                        pub fields: ::core::option::Option<::planus::alloc::vec::Vec<self::Field>>,
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                        pub features:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Feature>>,
                    }

                    impl Schema {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            endianness: impl ::planus::WriteAsDefault<
                                self::Endianness,
                                self::Endianness,
                            >,
                            fields: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::Field>]>,
                            >,
                            custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                            features: impl ::planus::WriteAsOptional<::planus::Offset<[self::Feature]>>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_endianness =
                                endianness.prepare(builder, &self::Endianness::Little);

                            let prepared_fields = fields.prepare(builder);

                            let prepared_custom_metadata = custom_metadata.prepare(builder);

                            let prepared_features = features.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<10, 14>::new(builder);

                            if prepared_endianness.is_some() {
                                table_writer.calculate_size::<self::Endianness>(2);
                            }
                            if prepared_fields.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::Field>]>>(4);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(6);
                            }
                            if prepared_features.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[self::Feature]>>(8);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_fields) =
                                    prepared_fields
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_fields);
                                }
                                if let ::core::option::Option::Some(prepared_custom_metadata) =
                                    prepared_custom_metadata
                                {
                                    table_writer.write::<_, _, 4>(2, &prepared_custom_metadata);
                                }
                                if let ::core::option::Option::Some(prepared_features) =
                                    prepared_features
                                {
                                    table_writer.write::<_, _, 4>(3, &prepared_features);
                                }
                                if let ::core::option::Option::Some(prepared_endianness) =
                                    prepared_endianness
                                {
                                    table_writer.write::<_, _, 2>(0, &prepared_endianness);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Schema>> for Schema {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Schema>> for Schema {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Schema>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Schema> for Schema {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Schema> {
                            Schema::create(
                                builder,
                                &self.endianness,
                                &self.fields,
                                &self.custom_metadata,
                                &self.features,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct SchemaRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SchemaRef<'a> {
                        pub fn endianness(&self) -> ::planus::Result<self::Endianness> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Schema", "endianness")?
                                    .unwrap_or(self::Endianness::Little),
                            )
                        }

                        pub fn fields(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::FieldRef<'a>>>,
                            >,
                        > {
                            self.0.access(1, "Schema", "fields")
                        }

                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(2, "Schema", "custom_metadata")
                        }

                        pub fn features(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<
                                    'a,
                                    ::core::result::Result<
                                        self::Feature,
                                        ::planus::errors::UnknownEnumTag,
                                    >,
                                >,
                            >,
                        > {
                            self.0.access(3, "Schema", "features")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SchemaRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SchemaRef");
                            f.field("endianness", &self.endianness());
                            if let ::core::option::Option::Some(fields) = self.fields().transpose()
                            {
                                f.field("fields", &fields);
                            }
                            if let ::core::option::Option::Some(custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &custom_metadata);
                            }
                            if let ::core::option::Option::Some(features) =
                                self.features().transpose()
                            {
                                f.field("features", &features);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SchemaRef<'a>> for Schema {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SchemaRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                endianness: ::core::convert::TryInto::try_into(
                                    value.endianness()?,
                                )?,
                                fields: if let ::core::option::Option::Some(fields) =
                                    value.fields()?
                                {
                                    ::core::option::Option::Some(fields.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                features: if let ::core::option::Option::Some(features) =
                                    value.features()?
                                {
                                    ::core::option::Option::Some(features.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SchemaRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SchemaRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SchemaRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Schema>> for Schema {
                        type Value = ::planus::Offset<Schema>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Schema>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SchemaRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[SchemaRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    pub struct FieldNode {
                        pub length: i64,
                        pub null_count: i64,
                    }

                    impl ::planus::Primitive for FieldNode {
                        const ALIGNMENT: usize = 8;
                        const SIZE: usize = 16;
                    }

                    #[allow(clippy::identity_op)]
                    impl ::planus::WriteAsPrimitive<FieldNode> for FieldNode {
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            let (cur, cursor) = cursor.split::<8, 8>();
                            self.length.write(cur, buffer_position - 0);
                            let (cur, cursor) = cursor.split::<8, 0>();
                            self.null_count.write(cur, buffer_position - 8);
                            cursor.finish([]);
                        }
                    }

                    impl ::planus::WriteAs<FieldNode> for FieldNode {
                        type Prepared = Self;
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self {
                            *self
                        }
                    }

                    impl ::planus::WriteAsOptional<FieldNode> for FieldNode {
                        type Prepared = Self;
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<Self> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct FieldNodeRef<'a>(::planus::ArrayWithStartOffset<'a, 16>);

                    impl<'a> FieldNodeRef<'a> {
                        pub fn length(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(0).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }

                        pub fn null_count(&self) -> i64 {
                            let buffer = self.0.advance_as_array::<8>(8).unwrap();

                            i64::from_le_bytes(*buffer.as_array())
                        }
                    }

                    impl<'a> ::core::fmt::Debug for FieldNodeRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("FieldNodeRef");
                            f.field("length", &self.length());
                            f.field("null_count", &self.null_count());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<FieldNodeRef<'a>> for FieldNode {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: FieldNodeRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(FieldNode {
                                length: value.length(),
                                null_count: value.null_count(),
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for FieldNodeRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let buffer = buffer.advance_as_array::<16>(offset)?;
                            ::core::result::Result::Ok(Self(buffer))
                        }
                    }

                    impl<'a> ::planus::VectorRead<'a> for FieldNodeRef<'a> {
                        const STRIDE: usize = 16;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> Self {
                            Self(buffer.unchecked_advance_as_array(offset))
                        }
                    }

                    impl ::planus::VectorWrite<FieldNode> for FieldNode {
                        const STRIDE: usize = 16;

                        type Value = FieldNode;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[FieldNode],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 16];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (16 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i8)]
                    pub enum CompressionType {
                        Lz4Frame = 0,
                        Zstd = 1,
                    }

                    impl ::core::convert::TryFrom<i8> for CompressionType {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i8,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(CompressionType::Lz4Frame),
                                1 => ::core::result::Result::Ok(CompressionType::Zstd),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<CompressionType> for i8 {
                        fn from(value: CompressionType) -> Self {
                            value as i8
                        }
                    }

                    impl ::planus::Primitive for CompressionType {
                        const ALIGNMENT: usize = 1;
                        const SIZE: usize = 1;
                    }

                    impl ::planus::WriteAsPrimitive<CompressionType> for CompressionType {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i8).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<CompressionType> for CompressionType {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(&self, _builder: &mut ::planus::Builder) -> CompressionType {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<CompressionType, CompressionType> for CompressionType {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &CompressionType,
                        ) -> ::core::option::Option<CompressionType> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<CompressionType> for CompressionType {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<CompressionType> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for CompressionType {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i8 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for CompressionType {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 1;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i8 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "CompressionType",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<CompressionType> for CompressionType {
                        const STRIDE: usize = 1;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - i as u32,
                                );
                            }
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i8)]
                    pub enum BodyCompressionMethod {
                        Buffer = 0,
                    }

                    impl ::core::convert::TryFrom<i8> for BodyCompressionMethod {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i8,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(BodyCompressionMethod::Buffer),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<BodyCompressionMethod> for i8 {
                        fn from(value: BodyCompressionMethod) -> Self {
                            value as i8
                        }
                    }

                    impl ::planus::Primitive for BodyCompressionMethod {
                        const ALIGNMENT: usize = 1;
                        const SIZE: usize = 1;
                    }

                    impl ::planus::WriteAsPrimitive<BodyCompressionMethod> for BodyCompressionMethod {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i8).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<BodyCompressionMethod> for BodyCompressionMethod {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> BodyCompressionMethod {
                            *self
                        }
                    }

                    impl ::planus::WriteAsDefault<BodyCompressionMethod, BodyCompressionMethod>
                        for BodyCompressionMethod
                    {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &BodyCompressionMethod,
                        ) -> ::core::option::Option<BodyCompressionMethod> {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<BodyCompressionMethod> for BodyCompressionMethod {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<BodyCompressionMethod> {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for BodyCompressionMethod {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i8 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for BodyCompressionMethod {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 1;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i8 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "BodyCompressionMethod",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<BodyCompressionMethod> for BodyCompressionMethod {
                        const STRIDE: usize = 1;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 1];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - i as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct BodyCompression {
                        pub codec: self::CompressionType,
                        pub method: self::BodyCompressionMethod,
                    }

                    impl BodyCompression {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            codec: impl ::planus::WriteAsDefault<
                                self::CompressionType,
                                self::CompressionType,
                            >,
                            method: impl ::planus::WriteAsDefault<
                                self::BodyCompressionMethod,
                                self::BodyCompressionMethod,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_codec =
                                codec.prepare(builder, &self::CompressionType::Lz4Frame);

                            let prepared_method =
                                method.prepare(builder, &self::BodyCompressionMethod::Buffer);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 2>::new(builder);

                            if prepared_codec.is_some() {
                                table_writer.calculate_size::<self::CompressionType>(2);
                            }
                            if prepared_method.is_some() {
                                table_writer.calculate_size::<self::BodyCompressionMethod>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_codec) = prepared_codec
                                {
                                    table_writer.write::<_, _, 1>(0, &prepared_codec);
                                }
                                if let ::core::option::Option::Some(prepared_method) =
                                    prepared_method
                                {
                                    table_writer.write::<_, _, 1>(1, &prepared_method);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<BodyCompression>> for BodyCompression {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<BodyCompression>> for BodyCompression {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<BodyCompression>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<BodyCompression> for BodyCompression {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<BodyCompression> {
                            BodyCompression::create(builder, &self.codec, &self.method)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct BodyCompressionRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> BodyCompressionRef<'a> {
                        pub fn codec(&self) -> ::planus::Result<self::CompressionType> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "BodyCompression", "codec")?
                                    .unwrap_or(self::CompressionType::Lz4Frame),
                            )
                        }

                        pub fn method(&self) -> ::planus::Result<self::BodyCompressionMethod> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(1, "BodyCompression", "method")?
                                    .unwrap_or(self::BodyCompressionMethod::Buffer),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for BodyCompressionRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("BodyCompressionRef");
                            f.field("codec", &self.codec());
                            f.field("method", &self.method());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<BodyCompressionRef<'a>> for BodyCompression {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: BodyCompressionRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                codec: ::core::convert::TryInto::try_into(value.codec()?)?,
                                method: ::core::convert::TryInto::try_into(value.method()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for BodyCompressionRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for BodyCompressionRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BodyCompressionRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<BodyCompression>> for BodyCompression {
                        type Value = ::planus::Offset<BodyCompression>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<BodyCompression>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for BodyCompressionRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[BodyCompressionRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct RecordBatch {
                        pub length: i64,
                        pub nodes:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::FieldNode>>,
                        pub buffers:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::Buffer>>,
                        pub compression: ::core::option::Option<
                            ::planus::alloc::boxed::Box<self::BodyCompression>,
                        >,
                        pub variadic_buffer_counts:
                            ::core::option::Option<::planus::alloc::vec::Vec<i64>>,
                    }

                    impl RecordBatch {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            length: impl ::planus::WriteAsDefault<i64, i64>,
                            nodes: impl ::planus::WriteAsOptional<::planus::Offset<[self::FieldNode]>>,
                            buffers: impl ::planus::WriteAsOptional<::planus::Offset<[self::Buffer]>>,
                            compression: impl ::planus::WriteAsOptional<
                                ::planus::Offset<self::BodyCompression>,
                            >,
                            variadic_buffer_counts: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[i64]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_length = length.prepare(builder, &0);

                            let prepared_nodes = nodes.prepare(builder);

                            let prepared_buffers = buffers.prepare(builder);

                            let prepared_compression = compression.prepare(builder);

                            let prepared_variadic_buffer_counts =
                                variadic_buffer_counts.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<12, 24>::new(builder);

                            if prepared_length.is_some() {
                                table_writer.calculate_size::<i64>(2);
                            }
                            if prepared_nodes.is_some() {
                                table_writer
                                    .calculate_size::<::planus::Offset<[self::FieldNode]>>(4);
                            }
                            if prepared_buffers.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[self::Buffer]>>(6);
                            }
                            if prepared_compression.is_some() {
                                table_writer
                                    .calculate_size::<::planus::Offset<self::BodyCompression>>(8);
                            }
                            if prepared_variadic_buffer_counts.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[i64]>>(10);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_length) =
                                    prepared_length
                                {
                                    table_writer.write::<_, _, 8>(0, &prepared_length);
                                }
                                if let ::core::option::Option::Some(prepared_nodes) = prepared_nodes
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_nodes);
                                }
                                if let ::core::option::Option::Some(prepared_buffers) =
                                    prepared_buffers
                                {
                                    table_writer.write::<_, _, 4>(2, &prepared_buffers);
                                }
                                if let ::core::option::Option::Some(prepared_compression) =
                                    prepared_compression
                                {
                                    table_writer.write::<_, _, 4>(3, &prepared_compression);
                                }
                                if let ::core::option::Option::Some(
                                    prepared_variadic_buffer_counts,
                                ) = prepared_variadic_buffer_counts
                                {
                                    table_writer
                                        .write::<_, _, 4>(4, &prepared_variadic_buffer_counts);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<RecordBatch>> for RecordBatch {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<RecordBatch>> for RecordBatch {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<RecordBatch>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<RecordBatch> for RecordBatch {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<RecordBatch> {
                            RecordBatch::create(
                                builder,
                                &self.length,
                                &self.nodes,
                                &self.buffers,
                                &self.compression,
                                &self.variadic_buffer_counts,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct RecordBatchRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> RecordBatchRef<'a> {
                        pub fn length(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "RecordBatch", "length")?.unwrap_or(0),
                            )
                        }

                        pub fn nodes(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::FieldNodeRef<'a>>>,
                        > {
                            self.0.access(1, "RecordBatch", "nodes")
                        }

                        pub fn buffers(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<::planus::Vector<'a, self::BufferRef<'a>>>,
                        > {
                            self.0.access(2, "RecordBatch", "buffers")
                        }

                        pub fn compression(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::BodyCompressionRef<'a>>>
                        {
                            self.0.access(3, "RecordBatch", "compression")
                        }

                        pub fn variadic_buffer_counts(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i64>>>
                        {
                            self.0.access(4, "RecordBatch", "variadic_buffer_counts")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for RecordBatchRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("RecordBatchRef");
                            f.field("length", &self.length());
                            if let ::core::option::Option::Some(nodes) = self.nodes().transpose() {
                                f.field("nodes", &nodes);
                            }
                            if let ::core::option::Option::Some(buffers) =
                                self.buffers().transpose()
                            {
                                f.field("buffers", &buffers);
                            }
                            if let ::core::option::Option::Some(compression) =
                                self.compression().transpose()
                            {
                                f.field("compression", &compression);
                            }
                            if let ::core::option::Option::Some(variadic_buffer_counts) =
                                self.variadic_buffer_counts().transpose()
                            {
                                f.field("variadic_buffer_counts", &variadic_buffer_counts);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<RecordBatchRef<'a>> for RecordBatch {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: RecordBatchRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                length: ::core::convert::TryInto::try_into(value.length()?)?,
                                nodes: if let ::core::option::Option::Some(nodes) = value.nodes()? {
                                    ::core::option::Option::Some(nodes.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                buffers: if let ::core::option::Option::Some(buffers) =
                                    value.buffers()?
                                {
                                    ::core::option::Option::Some(buffers.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                compression: if let ::core::option::Option::Some(compression) =
                                    value.compression()?
                                {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(compression)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                variadic_buffer_counts: if let ::core::option::Option::Some(
                                    variadic_buffer_counts,
                                ) = value.variadic_buffer_counts()?
                                {
                                    ::core::option::Option::Some(variadic_buffer_counts.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for RecordBatchRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for RecordBatchRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RecordBatchRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<RecordBatch>> for RecordBatch {
                        type Value = ::planus::Offset<RecordBatch>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<RecordBatch>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for RecordBatchRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[RecordBatchRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct DictionaryBatch {
                        pub id: i64,
                        pub data:
                            ::core::option::Option<::planus::alloc::boxed::Box<self::RecordBatch>>,
                        pub is_delta: bool,
                    }

                    impl DictionaryBatch {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            id: impl ::planus::WriteAsDefault<i64, i64>,
                            data: impl ::planus::WriteAsOptional<::planus::Offset<self::RecordBatch>>,
                            is_delta: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_id = id.prepare(builder, &0);

                            let prepared_data = data.prepare(builder);

                            let prepared_is_delta = is_delta.prepare(builder, &false);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<8, 13>::new(builder);

                            if prepared_id.is_some() {
                                table_writer.calculate_size::<i64>(2);
                            }
                            if prepared_data.is_some() {
                                table_writer
                                    .calculate_size::<::planus::Offset<self::RecordBatch>>(4);
                            }
                            if prepared_is_delta.is_some() {
                                table_writer.calculate_size::<bool>(6);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_id) = prepared_id {
                                    table_writer.write::<_, _, 8>(0, &prepared_id);
                                }
                                if let ::core::option::Option::Some(prepared_data) = prepared_data {
                                    table_writer.write::<_, _, 4>(1, &prepared_data);
                                }
                                if let ::core::option::Option::Some(prepared_is_delta) =
                                    prepared_is_delta
                                {
                                    table_writer.write::<_, _, 1>(2, &prepared_is_delta);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<DictionaryBatch>> for DictionaryBatch {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<DictionaryBatch>> for DictionaryBatch {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<DictionaryBatch>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<DictionaryBatch> for DictionaryBatch {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<DictionaryBatch> {
                            DictionaryBatch::create(builder, &self.id, &self.data, &self.is_delta)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct DictionaryBatchRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> DictionaryBatchRef<'a> {
                        pub fn id(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "DictionaryBatch", "id")?.unwrap_or(0),
                            )
                        }

                        pub fn data(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::RecordBatchRef<'a>>>
                        {
                            self.0.access(1, "DictionaryBatch", "data")
                        }

                        pub fn is_delta(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(2, "DictionaryBatch", "is_delta")?
                                    .unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for DictionaryBatchRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("DictionaryBatchRef");
                            f.field("id", &self.id());
                            if let ::core::option::Option::Some(data) = self.data().transpose() {
                                f.field("data", &data);
                            }
                            f.field("is_delta", &self.is_delta());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<DictionaryBatchRef<'a>> for DictionaryBatch {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: DictionaryBatchRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                id: ::core::convert::TryInto::try_into(value.id()?)?,
                                data: if let ::core::option::Option::Some(data) = value.data()? {
                                    ::core::option::Option::Some(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryInto::try_into(data)?,
                                    ))
                                } else {
                                    ::core::option::Option::None
                                },
                                is_delta: ::core::convert::TryInto::try_into(value.is_delta()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for DictionaryBatchRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for DictionaryBatchRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryBatchRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<DictionaryBatch>> for DictionaryBatch {
                        type Value = ::planus::Offset<DictionaryBatch>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<DictionaryBatch>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for DictionaryBatchRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[DictionaryBatchRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub enum MessageHeader {
                        Schema(::planus::alloc::boxed::Box<self::Schema>),
                        DictionaryBatch(::planus::alloc::boxed::Box<self::DictionaryBatch>),
                        RecordBatch(::planus::alloc::boxed::Box<self::RecordBatch>),
                        Tensor(::planus::alloc::boxed::Box<self::Tensor>),
                        SparseTensor(::planus::alloc::boxed::Box<self::SparseTensor>),
                    }

                    impl MessageHeader {
                        pub fn create_schema(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Schema>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
                        }

                        pub fn create_dictionary_batch(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::DictionaryBatch>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
                        }

                        pub fn create_record_batch(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::RecordBatch>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
                        }

                        pub fn create_tensor(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::Tensor>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(4, value.prepare(builder).downcast())
                        }

                        pub fn create_sparse_tensor(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseTensor>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(5, value.prepare(builder).downcast())
                        }
                    }

                    impl ::planus::WriteAsUnion<MessageHeader> for MessageHeader {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Self> {
                            match self {
                                Self::Schema(value) => Self::create_schema(builder, value),
                                Self::DictionaryBatch(value) => {
                                    Self::create_dictionary_batch(builder, value)
                                }
                                Self::RecordBatch(value) => {
                                    Self::create_record_batch(builder, value)
                                }
                                Self::Tensor(value) => Self::create_tensor(builder, value),
                                Self::SparseTensor(value) => {
                                    Self::create_sparse_tensor(builder, value)
                                }
                            }
                        }
                    }

                    impl ::planus::WriteAsOptionalUnion<MessageHeader> for MessageHeader {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Self>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    #[derive(Copy, Clone, Debug)]
                    pub enum MessageHeaderRef<'a> {
                        Schema(self::SchemaRef<'a>),
                        DictionaryBatch(self::DictionaryBatchRef<'a>),
                        RecordBatch(self::RecordBatchRef<'a>),
                        Tensor(self::TensorRef<'a>),
                        SparseTensor(self::SparseTensorRef<'a>),
                    }

                    impl<'a> ::core::convert::TryFrom<MessageHeaderRef<'a>> for MessageHeader {
                        type Error = ::planus::Error;

                        fn try_from(value: MessageHeaderRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(match value {
                                MessageHeaderRef::Schema(value) => {
                                    MessageHeader::Schema(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::DictionaryBatch(value) => {
                                    MessageHeader::DictionaryBatch(
                                        ::planus::alloc::boxed::Box::new(
                                            ::core::convert::TryFrom::try_from(value)?,
                                        ),
                                    )
                                }

                                MessageHeaderRef::RecordBatch(value) => {
                                    MessageHeader::RecordBatch(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::Tensor(value) => {
                                    MessageHeader::Tensor(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }

                                MessageHeaderRef::SparseTensor(value) => {
                                    MessageHeader::SparseTensor(::planus::alloc::boxed::Box::new(
                                        ::core::convert::TryFrom::try_from(value)?,
                                    ))
                                }
                            })
                        }
                    }

                    impl<'a> ::planus::TableReadUnion<'a> for MessageHeaderRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            field_offset: usize,
                            tag: u8,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            match tag {
                                1 => ::core::result::Result::Ok(Self::Schema(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                2 => ::core::result::Result::Ok(Self::DictionaryBatch(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                3 => ::core::result::Result::Ok(Self::RecordBatch(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                4 => ::core::result::Result::Ok(Self::Tensor(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                5 => ::core::result::Result::Ok(Self::SparseTensor(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                _ => ::core::result::Result::Err(
                                    ::planus::errors::ErrorKind::UnknownUnionTag { tag },
                                ),
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Message {
                        pub version: self::MetadataVersion,
                        pub header: ::core::option::Option<self::MessageHeader>,
                        pub body_length: i64,
                        pub custom_metadata:
                            ::core::option::Option<::planus::alloc::vec::Vec<self::KeyValue>>,
                    }

                    impl Message {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            version: impl ::planus::WriteAsDefault<
                                self::MetadataVersion,
                                self::MetadataVersion,
                            >,
                            header: impl ::planus::WriteAsOptionalUnion<self::MessageHeader>,
                            body_length: impl ::planus::WriteAsDefault<i64, i64>,
                            custom_metadata: impl ::planus::WriteAsOptional<
                                ::planus::Offset<[::planus::Offset<self::KeyValue>]>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_version =
                                version.prepare(builder, &self::MetadataVersion::V1);

                            let prepared_header = header.prepare(builder);

                            let prepared_body_length = body_length.prepare(builder, &0);

                            let prepared_custom_metadata = custom_metadata.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<12, 19>::new(builder);

                            if prepared_version.is_some() {
                                table_writer.calculate_size::<self::MetadataVersion>(2);
                            }
                            if prepared_header.is_some() {
                                table_writer.calculate_size::<u8>(4);
                                table_writer
                                    .calculate_size::<::planus::Offset<self::MessageHeader>>(6);
                            }
                            if prepared_body_length.is_some() {
                                table_writer.calculate_size::<i64>(8);
                            }
                            if prepared_custom_metadata.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::KeyValue>]>>(10);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_body_length) =
                                    prepared_body_length
                                {
                                    table_writer.write::<_, _, 8>(3, &prepared_body_length);
                                }
                                if let ::core::option::Option::Some(prepared_header) =
                                    prepared_header
                                {
                                    table_writer.write::<_, _, 4>(2, &prepared_header.offset());
                                }
                                if let ::core::option::Option::Some(prepared_custom_metadata) =
                                    prepared_custom_metadata
                                {
                                    table_writer.write::<_, _, 4>(4, &prepared_custom_metadata);
                                }
                                if let ::core::option::Option::Some(prepared_version) =
                                    prepared_version
                                {
                                    table_writer.write::<_, _, 2>(0, &prepared_version);
                                }
                                if let ::core::option::Option::Some(prepared_header) =
                                    prepared_header
                                {
                                    table_writer.write::<_, _, 1>(1, &prepared_header.tag());
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Message>> for Message {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Message>> for Message {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Message>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Message> for Message {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Message> {
                            Message::create(
                                builder,
                                &self.version,
                                &self.header,
                                &self.body_length,
                                &self.custom_metadata,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct MessageRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> MessageRef<'a> {
                        pub fn version(&self) -> ::planus::Result<self::MetadataVersion> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "Message", "version")?
                                    .unwrap_or(self::MetadataVersion::V1),
                            )
                        }

                        pub fn header(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<self::MessageHeaderRef<'a>>>
                        {
                            self.0.access_union(1, "Message", "header")
                        }

                        pub fn body_length(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(3, "Message", "body_length")?.unwrap_or(0),
                            )
                        }

                        pub fn custom_metadata(
                            &self,
                        ) -> ::planus::Result<
                            ::core::option::Option<
                                ::planus::Vector<'a, ::planus::Result<self::KeyValueRef<'a>>>,
                            >,
                        > {
                            self.0.access(4, "Message", "custom_metadata")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for MessageRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("MessageRef");
                            f.field("version", &self.version());
                            if let ::core::option::Option::Some(header) = self.header().transpose()
                            {
                                f.field("header", &header);
                            }
                            f.field("body_length", &self.body_length());
                            if let ::core::option::Option::Some(custom_metadata) =
                                self.custom_metadata().transpose()
                            {
                                f.field("custom_metadata", &custom_metadata);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<MessageRef<'a>> for Message {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: MessageRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                version: ::core::convert::TryInto::try_into(value.version()?)?,
                                header: if let ::core::option::Option::Some(header) =
                                    value.header()?
                                {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(header)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                                body_length: ::core::convert::TryInto::try_into(
                                    value.body_length()?,
                                )?,
                                custom_metadata: if let ::core::option::Option::Some(
                                    custom_metadata,
                                ) = value.custom_metadata()?
                                {
                                    ::core::option::Option::Some(custom_metadata.to_vec_result()?)
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for MessageRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for MessageRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[MessageRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Message>> for Message {
                        type Value = ::planus::Offset<Message>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Message>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for MessageRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[MessageRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct SparseTensorIndexCoo {
                        pub indices_type: ::planus::alloc::boxed::Box<self::Int>,
                        pub indices_strides: ::core::option::Option<::planus::alloc::vec::Vec<i64>>,
                        pub indices_buffer: self::Buffer,
                        pub is_canonical: bool,
                    }

                    impl SparseTensorIndexCoo {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            indices_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            indices_strides: impl ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            indices_buffer: impl ::planus::WriteAs<self::Buffer>,
                            is_canonical: impl ::planus::WriteAsDefault<bool, bool>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_indices_type = indices_type.prepare(builder);

                            let prepared_indices_strides = indices_strides.prepare(builder);

                            let prepared_indices_buffer = indices_buffer.prepare(builder);

                            let prepared_is_canonical = is_canonical.prepare(builder, &false);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<10, 25>::new(builder);

                            table_writer.calculate_size::<::planus::Offset<self::Int>>(2);
                            if prepared_indices_strides.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[i64]>>(4);
                            }
                            table_writer.calculate_size::<self::Buffer>(6);
                            if prepared_is_canonical.is_some() {
                                table_writer.calculate_size::<bool>(8);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                table_writer.write::<_, _, 16>(2, &prepared_indices_buffer);
                                table_writer.write::<_, _, 4>(0, &prepared_indices_type);
                                if let ::core::option::Option::Some(prepared_indices_strides) =
                                    prepared_indices_strides
                                {
                                    table_writer.write::<_, _, 4>(1, &prepared_indices_strides);
                                }
                                if let ::core::option::Option::Some(prepared_is_canonical) =
                                    prepared_is_canonical
                                {
                                    table_writer.write::<_, _, 1>(3, &prepared_is_canonical);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensorIndexCoo>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseTensorIndexCoo> for SparseTensorIndexCoo {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCoo> {
                            SparseTensorIndexCoo::create(
                                builder,
                                &self.indices_type,
                                &self.indices_strides,
                                &self.indices_buffer,
                                &self.is_canonical,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct SparseTensorIndexCooRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseTensorIndexCooRef<'a> {
                        pub fn indices_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(0, "SparseTensorIndexCoo", "indices_type")
                        }

                        pub fn indices_strides(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i64>>>
                        {
                            self.0.access(1, "SparseTensorIndexCoo", "indices_strides")
                        }

                        pub fn indices_buffer(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0
                                .access_required(2, "SparseTensorIndexCoo", "indices_buffer")
                        }

                        pub fn is_canonical(&self) -> ::planus::Result<bool> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(3, "SparseTensorIndexCoo", "is_canonical")?
                                    .unwrap_or(false),
                            )
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseTensorIndexCooRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseTensorIndexCooRef");
                            f.field("indices_type", &self.indices_type());
                            if let ::core::option::Option::Some(indices_strides) =
                                self.indices_strides().transpose()
                            {
                                f.field("indices_strides", &indices_strides);
                            }
                            f.field("indices_buffer", &self.indices_buffer());
                            f.field("is_canonical", &self.is_canonical());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorIndexCooRef<'a>> for SparseTensorIndexCoo {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseTensorIndexCooRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                indices_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indices_type()?)?,
                                ),
                                indices_strides: if let ::core::option::Option::Some(
                                    indices_strides,
                                ) = value.indices_strides()?
                                {
                                    ::core::option::Option::Some(indices_strides.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                indices_buffer: ::core::convert::TryInto::try_into(
                                    value.indices_buffer()?,
                                )?,
                                is_canonical: ::core::convert::TryInto::try_into(
                                    value.is_canonical()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseTensorIndexCooRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseTensorIndexCooRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCooRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<SparseTensorIndexCoo>> for SparseTensorIndexCoo {
                        type Value = ::planus::Offset<SparseTensorIndexCoo>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseTensorIndexCoo>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseTensorIndexCooRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCooRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(
                        Copy, Clone, Debug, PartialEq, Eq, ::serde::Serialize, ::serde::Deserialize,
                    )]
                    #[repr(i16)]
                    pub enum SparseMatrixCompressedAxis {
                        Row = 0,
                        Column = 1,
                    }

                    impl ::core::convert::TryFrom<i16> for SparseMatrixCompressedAxis {
                        type Error = ::planus::errors::UnknownEnumTagKind;
                        fn try_from(
                            value: i16,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTagKind>
                        {
                            #[allow(clippy::match_single_binding)]
                            match value {
                                0 => ::core::result::Result::Ok(SparseMatrixCompressedAxis::Row),
                                1 => ::core::result::Result::Ok(SparseMatrixCompressedAxis::Column),

                                _ => ::core::result::Result::Err(
                                    ::planus::errors::UnknownEnumTagKind { tag: value as i128 },
                                ),
                            }
                        }
                    }

                    impl ::core::convert::From<SparseMatrixCompressedAxis> for i16 {
                        fn from(value: SparseMatrixCompressedAxis) -> Self {
                            value as i16
                        }
                    }

                    impl ::planus::Primitive for SparseMatrixCompressedAxis {
                        const ALIGNMENT: usize = 2;
                        const SIZE: usize = 2;
                    }

                    impl ::planus::WriteAsPrimitive<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        #[inline]
                        fn write<const N: usize>(
                            &self,
                            cursor: ::planus::Cursor<'_, N>,
                            buffer_position: u32,
                        ) {
                            (*self as i16).write(cursor, buffer_position);
                        }
                    }

                    impl ::planus::WriteAs<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> SparseMatrixCompressedAxis {
                            *self
                        }
                    }

                    impl
                        ::planus::WriteAsDefault<
                            SparseMatrixCompressedAxis,
                            SparseMatrixCompressedAxis,
                        > for SparseMatrixCompressedAxis
                    {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                            default: &SparseMatrixCompressedAxis,
                        ) -> ::core::option::Option<SparseMatrixCompressedAxis>
                        {
                            if self == default {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(*self)
                            }
                        }
                    }

                    impl ::planus::WriteAsOptional<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        type Prepared = Self;

                        #[inline]
                        fn prepare(
                            &self,
                            _builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<SparseMatrixCompressedAxis>
                        {
                            ::core::option::Option::Some(*self)
                        }
                    }

                    impl<'buf> ::planus::TableRead<'buf> for SparseMatrixCompressedAxis {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            let n: i16 = ::planus::TableRead::from_buffer(buffer, offset)?;
                            ::core::result::Result::Ok(::core::convert::TryInto::try_into(n)?)
                        }
                    }

                    impl<'buf> ::planus::VectorReadInner<'buf> for SparseMatrixCompressedAxis {
                        type Error = ::planus::errors::UnknownEnumTag;
                        const STRIDE: usize = 2;
                        #[inline]
                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'buf>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::UnknownEnumTag>
                        {
                            let value = <i16 as ::planus::VectorRead>::from_buffer(buffer, offset);
                            let value: ::core::result::Result<Self, _> =
                                ::core::convert::TryInto::try_into(value);
                            value.map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "SparseMatrixCompressedAxis",
                                    "VectorRead::from_buffer",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl<'buf> ::planus::VectorWrite<SparseMatrixCompressedAxis> for SparseMatrixCompressedAxis {
                        const STRIDE: usize = 2;

                        type Value = Self;

                        fn prepare(&self, _builder: &mut ::planus::Builder) -> Self::Value {
                            *self
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[Self],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 2];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (2 * i) as u32,
                                );
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct SparseMatrixIndexCsx {
                        pub compressed_axis: self::SparseMatrixCompressedAxis,
                        pub indptr_type: ::planus::alloc::boxed::Box<self::Int>,
                        pub indptr_buffer: self::Buffer,
                        pub indices_type: ::planus::alloc::boxed::Box<self::Int>,
                        pub indices_buffer: self::Buffer,
                    }

                    impl SparseMatrixIndexCsx {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            compressed_axis: impl ::planus::WriteAsDefault<
                                self::SparseMatrixCompressedAxis,
                                self::SparseMatrixCompressedAxis,
                            >,
                            indptr_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            indptr_buffer: impl ::planus::WriteAs<self::Buffer>,
                            indices_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            indices_buffer: impl ::planus::WriteAs<self::Buffer>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_compressed_axis = compressed_axis
                                .prepare(builder, &self::SparseMatrixCompressedAxis::Row);

                            let prepared_indptr_type = indptr_type.prepare(builder);

                            let prepared_indptr_buffer = indptr_buffer.prepare(builder);

                            let prepared_indices_type = indices_type.prepare(builder);

                            let prepared_indices_buffer = indices_buffer.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<12, 42>::new(builder);

                            if prepared_compressed_axis.is_some() {
                                table_writer.calculate_size::<self::SparseMatrixCompressedAxis>(2);
                            }
                            table_writer.calculate_size::<::planus::Offset<self::Int>>(4);
                            table_writer.calculate_size::<self::Buffer>(6);
                            table_writer.calculate_size::<::planus::Offset<self::Int>>(8);
                            table_writer.calculate_size::<self::Buffer>(10);

                            table_writer.finish_calculating();

                            unsafe {
                                table_writer.write::<_, _, 16>(2, &prepared_indptr_buffer);
                                table_writer.write::<_, _, 16>(4, &prepared_indices_buffer);
                                table_writer.write::<_, _, 4>(1, &prepared_indptr_type);
                                table_writer.write::<_, _, 4>(3, &prepared_indices_type);
                                if let ::core::option::Option::Some(prepared_compressed_axis) =
                                    prepared_compressed_axis
                                {
                                    table_writer.write::<_, _, 2>(0, &prepared_compressed_axis);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseMatrixIndexCsx>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseMatrixIndexCsx> for SparseMatrixIndexCsx {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseMatrixIndexCsx> {
                            SparseMatrixIndexCsx::create(
                                builder,
                                &self.compressed_axis,
                                &self.indptr_type,
                                &self.indptr_buffer,
                                &self.indices_type,
                                &self.indices_buffer,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct SparseMatrixIndexCsxRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseMatrixIndexCsxRef<'a> {
                        pub fn compressed_axis(
                            &self,
                        ) -> ::planus::Result<self::SparseMatrixCompressedAxis>
                        {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(0, "SparseMatrixIndexCsx", "compressed_axis")?
                                    .unwrap_or(self::SparseMatrixCompressedAxis::Row),
                            )
                        }

                        pub fn indptr_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(1, "SparseMatrixIndexCsx", "indptr_type")
                        }

                        pub fn indptr_buffer(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0
                                .access_required(2, "SparseMatrixIndexCsx", "indptr_buffer")
                        }

                        pub fn indices_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(3, "SparseMatrixIndexCsx", "indices_type")
                        }

                        pub fn indices_buffer(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0
                                .access_required(4, "SparseMatrixIndexCsx", "indices_buffer")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseMatrixIndexCsxRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseMatrixIndexCsxRef");
                            f.field("compressed_axis", &self.compressed_axis());
                            f.field("indptr_type", &self.indptr_type());
                            f.field("indptr_buffer", &self.indptr_buffer());
                            f.field("indices_type", &self.indices_type());
                            f.field("indices_buffer", &self.indices_buffer());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseMatrixIndexCsxRef<'a>> for SparseMatrixIndexCsx {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseMatrixIndexCsxRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                compressed_axis: ::core::convert::TryInto::try_into(
                                    value.compressed_axis()?,
                                )?,
                                indptr_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indptr_type()?)?,
                                ),
                                indptr_buffer: ::core::convert::TryInto::try_into(
                                    value.indptr_buffer()?,
                                )?,
                                indices_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indices_type()?)?,
                                ),
                                indices_buffer: ::core::convert::TryInto::try_into(
                                    value.indices_buffer()?,
                                )?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseMatrixIndexCsxRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseMatrixIndexCsxRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseMatrixIndexCsxRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<SparseMatrixIndexCsx>> for SparseMatrixIndexCsx {
                        type Value = ::planus::Offset<SparseMatrixIndexCsx>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseMatrixIndexCsx>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseMatrixIndexCsxRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseMatrixIndexCsxRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct SparseTensorIndexCsf {
                        pub indptr_type: ::planus::alloc::boxed::Box<self::Int>,
                        pub indptr_buffers: ::planus::alloc::vec::Vec<self::Buffer>,
                        pub indices_type: ::planus::alloc::boxed::Box<self::Int>,
                        pub indices_buffers: ::planus::alloc::vec::Vec<self::Buffer>,
                        pub axis_order: ::planus::alloc::vec::Vec<i32>,
                    }

                    impl SparseTensorIndexCsf {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            indptr_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            indptr_buffers: impl ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            indices_type: impl ::planus::WriteAs<::planus::Offset<self::Int>>,
                            indices_buffers: impl ::planus::WriteAs<::planus::Offset<[self::Buffer]>>,
                            axis_order: impl ::planus::WriteAs<::planus::Offset<[i32]>>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_indptr_type = indptr_type.prepare(builder);

                            let prepared_indptr_buffers = indptr_buffers.prepare(builder);

                            let prepared_indices_type = indices_type.prepare(builder);

                            let prepared_indices_buffers = indices_buffers.prepare(builder);

                            let prepared_axis_order = axis_order.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<12, 20>::new(builder);

                            table_writer.calculate_size::<::planus::Offset<self::Int>>(2);
                            table_writer.calculate_size::<::planus::Offset<[self::Buffer]>>(4);
                            table_writer.calculate_size::<::planus::Offset<self::Int>>(6);
                            table_writer.calculate_size::<::planus::Offset<[self::Buffer]>>(8);
                            table_writer.calculate_size::<::planus::Offset<[i32]>>(10);

                            table_writer.finish_calculating();

                            unsafe {
                                table_writer.write::<_, _, 4>(0, &prepared_indptr_type);
                                table_writer.write::<_, _, 4>(1, &prepared_indptr_buffers);
                                table_writer.write::<_, _, 4>(2, &prepared_indices_type);
                                table_writer.write::<_, _, 4>(3, &prepared_indices_buffers);
                                table_writer.write::<_, _, 4>(4, &prepared_axis_order);
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensorIndexCsf>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseTensorIndexCsf> for SparseTensorIndexCsf {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensorIndexCsf> {
                            SparseTensorIndexCsf::create(
                                builder,
                                &self.indptr_type,
                                &self.indptr_buffers,
                                &self.indices_type,
                                &self.indices_buffers,
                                &self.axis_order,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct SparseTensorIndexCsfRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseTensorIndexCsfRef<'a> {
                        pub fn indptr_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(0, "SparseTensorIndexCsf", "indptr_type")
                        }

                        pub fn indptr_buffers(
                            &self,
                        ) -> ::planus::Result<::planus::Vector<'a, self::BufferRef<'a>>>
                        {
                            self.0
                                .access_required(1, "SparseTensorIndexCsf", "indptr_buffers")
                        }

                        pub fn indices_type(&self) -> ::planus::Result<self::IntRef<'a>> {
                            self.0
                                .access_required(2, "SparseTensorIndexCsf", "indices_type")
                        }

                        pub fn indices_buffers(
                            &self,
                        ) -> ::planus::Result<::planus::Vector<'a, self::BufferRef<'a>>>
                        {
                            self.0
                                .access_required(3, "SparseTensorIndexCsf", "indices_buffers")
                        }

                        pub fn axis_order(&self) -> ::planus::Result<::planus::Vector<'a, i32>> {
                            self.0
                                .access_required(4, "SparseTensorIndexCsf", "axis_order")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseTensorIndexCsfRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseTensorIndexCsfRef");
                            f.field("indptr_type", &self.indptr_type());
                            f.field("indptr_buffers", &self.indptr_buffers());
                            f.field("indices_type", &self.indices_type());
                            f.field("indices_buffers", &self.indices_buffers());
                            f.field("axis_order", &self.axis_order());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorIndexCsfRef<'a>> for SparseTensorIndexCsf {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseTensorIndexCsfRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                indptr_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indptr_type()?)?,
                                ),
                                indptr_buffers: value.indptr_buffers()?.to_vec()?,
                                indices_type: ::planus::alloc::boxed::Box::new(
                                    ::core::convert::TryInto::try_into(value.indices_type()?)?,
                                ),
                                indices_buffers: value.indices_buffers()?.to_vec()?,
                                axis_order: value.axis_order()?.to_vec()?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseTensorIndexCsfRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseTensorIndexCsfRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCsfRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<SparseTensorIndexCsf>> for SparseTensorIndexCsf {
                        type Value = ::planus::Offset<SparseTensorIndexCsf>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseTensorIndexCsf>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseTensorIndexCsfRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorIndexCsfRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub enum SparseTensorIndex {
                        SparseTensorIndexCoo(
                            ::planus::alloc::boxed::Box<self::SparseTensorIndexCoo>,
                        ),
                        SparseMatrixIndexCsx(
                            ::planus::alloc::boxed::Box<self::SparseMatrixIndexCsx>,
                        ),
                        SparseTensorIndexCsf(
                            ::planus::alloc::boxed::Box<self::SparseTensorIndexCsf>,
                        ),
                    }

                    impl SparseTensorIndex {
                        pub fn create_sparse_tensor_index_coo(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseTensorIndexCoo>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(1, value.prepare(builder).downcast())
                        }

                        pub fn create_sparse_matrix_index_csx(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseMatrixIndexCsx>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(2, value.prepare(builder).downcast())
                        }

                        pub fn create_sparse_tensor_index_csf(
                            builder: &mut ::planus::Builder,
                            value: impl ::planus::WriteAsOffset<self::SparseTensorIndexCsf>,
                        ) -> ::planus::UnionOffset<Self> {
                            ::planus::UnionOffset::new(3, value.prepare(builder).downcast())
                        }
                    }

                    impl ::planus::WriteAsUnion<SparseTensorIndex> for SparseTensorIndex {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::UnionOffset<Self> {
                            match self {
                                Self::SparseTensorIndexCoo(value) => {
                                    Self::create_sparse_tensor_index_coo(builder, value)
                                }
                                Self::SparseMatrixIndexCsx(value) => {
                                    Self::create_sparse_matrix_index_csx(builder, value)
                                }
                                Self::SparseTensorIndexCsf(value) => {
                                    Self::create_sparse_tensor_index_csf(builder, value)
                                }
                            }
                        }
                    }

                    impl ::planus::WriteAsOptionalUnion<SparseTensorIndex> for SparseTensorIndex {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::UnionOffset<Self>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsUnion::prepare(
                                self, builder,
                            ))
                        }
                    }

                    #[derive(Copy, Clone, Debug)]
                    pub enum SparseTensorIndexRef<'a> {
                        SparseTensorIndexCoo(self::SparseTensorIndexCooRef<'a>),
                        SparseMatrixIndexCsx(self::SparseMatrixIndexCsxRef<'a>),
                        SparseTensorIndexCsf(self::SparseTensorIndexCsfRef<'a>),
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorIndexRef<'a>> for SparseTensorIndex {
                        type Error = ::planus::Error;

                        fn try_from(value: SparseTensorIndexRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(match value {
                                SparseTensorIndexRef::SparseTensorIndexCoo(value) => {
                                    SparseTensorIndex::SparseTensorIndexCoo(
                                        ::planus::alloc::boxed::Box::new(
                                            ::core::convert::TryFrom::try_from(value)?,
                                        ),
                                    )
                                }

                                SparseTensorIndexRef::SparseMatrixIndexCsx(value) => {
                                    SparseTensorIndex::SparseMatrixIndexCsx(
                                        ::planus::alloc::boxed::Box::new(
                                            ::core::convert::TryFrom::try_from(value)?,
                                        ),
                                    )
                                }

                                SparseTensorIndexRef::SparseTensorIndexCsf(value) => {
                                    SparseTensorIndex::SparseTensorIndexCsf(
                                        ::planus::alloc::boxed::Box::new(
                                            ::core::convert::TryFrom::try_from(value)?,
                                        ),
                                    )
                                }
                            })
                        }
                    }

                    impl<'a> ::planus::TableReadUnion<'a> for SparseTensorIndexRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            field_offset: usize,
                            tag: u8,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            match tag {
                                1 => ::core::result::Result::Ok(Self::SparseTensorIndexCoo(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                2 => ::core::result::Result::Ok(Self::SparseMatrixIndexCsx(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                3 => ::core::result::Result::Ok(Self::SparseTensorIndexCsf(
                                    ::planus::TableRead::from_buffer(buffer, field_offset)?,
                                )),
                                _ => ::core::result::Result::Err(
                                    ::planus::errors::ErrorKind::UnknownUnionTag { tag },
                                ),
                            }
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct SparseTensor {
                        pub type_: self::Type,
                        pub shape: ::planus::alloc::vec::Vec<self::TensorDim>,
                        pub non_zero_length: i64,
                        pub sparse_index: self::SparseTensorIndex,
                        pub data: self::Buffer,
                    }

                    impl SparseTensor {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            type_: impl ::planus::WriteAsUnion<self::Type>,
                            shape: impl ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            non_zero_length: impl ::planus::WriteAsDefault<i64, i64>,
                            sparse_index: impl ::planus::WriteAsUnion<self::SparseTensorIndex>,
                            data: impl ::planus::WriteAs<self::Buffer>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_type_ = type_.prepare(builder);

                            let prepared_shape = shape.prepare(builder);

                            let prepared_non_zero_length = non_zero_length.prepare(builder, &0);

                            let prepared_sparse_index = sparse_index.prepare(builder);

                            let prepared_data = data.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<16, 38>::new(builder);

                            table_writer.calculate_size::<u8>(2);
                            table_writer.calculate_size::<::planus::Offset<self::Type>>(4);
                            table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::TensorDim>]>>(6);
                            if prepared_non_zero_length.is_some() {
                                table_writer.calculate_size::<i64>(8);
                            }
                            table_writer.calculate_size::<u8>(10);
                            table_writer
                                .calculate_size::<::planus::Offset<self::SparseTensorIndex>>(12);
                            table_writer.calculate_size::<self::Buffer>(14);

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_non_zero_length) =
                                    prepared_non_zero_length
                                {
                                    table_writer.write::<_, _, 8>(3, &prepared_non_zero_length);
                                }
                                table_writer.write::<_, _, 16>(6, &prepared_data);
                                table_writer.write::<_, _, 4>(1, &prepared_type_.offset());
                                table_writer.write::<_, _, 4>(2, &prepared_shape);
                                table_writer.write::<_, _, 4>(5, &prepared_sparse_index.offset());
                                table_writer.write::<_, _, 1>(0, &prepared_type_.tag());
                                table_writer.write::<_, _, 1>(4, &prepared_sparse_index.tag());
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<SparseTensor>> for SparseTensor {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<SparseTensor>> for SparseTensor {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<SparseTensor>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<SparseTensor> for SparseTensor {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<SparseTensor> {
                            SparseTensor::create(
                                builder,
                                &self.type_,
                                &self.shape,
                                &self.non_zero_length,
                                &self.sparse_index,
                                &self.data,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct SparseTensorRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> SparseTensorRef<'a> {
                        pub fn type_(&self) -> ::planus::Result<self::TypeRef<'a>> {
                            self.0.access_union_required(0, "SparseTensor", "type_")
                        }

                        pub fn shape(
                            &self,
                        ) -> ::planus::Result<
                            ::planus::Vector<'a, ::planus::Result<self::TensorDimRef<'a>>>,
                        > {
                            self.0.access_required(2, "SparseTensor", "shape")
                        }

                        pub fn non_zero_length(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0
                                    .access(3, "SparseTensor", "non_zero_length")?
                                    .unwrap_or(0),
                            )
                        }

                        pub fn sparse_index(
                            &self,
                        ) -> ::planus::Result<self::SparseTensorIndexRef<'a>>
                        {
                            self.0
                                .access_union_required(4, "SparseTensor", "sparse_index")
                        }

                        pub fn data(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0.access_required(6, "SparseTensor", "data")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for SparseTensorRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("SparseTensorRef");
                            f.field("type_", &self.type_());
                            f.field("shape", &self.shape());
                            f.field("non_zero_length", &self.non_zero_length());
                            f.field("sparse_index", &self.sparse_index());
                            f.field("data", &self.data());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<SparseTensorRef<'a>> for SparseTensor {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: SparseTensorRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                type_: ::core::convert::TryInto::try_into(value.type_()?)?,
                                shape: value.shape()?.to_vec_result()?,
                                non_zero_length: ::core::convert::TryInto::try_into(
                                    value.non_zero_length()?,
                                )?,
                                sparse_index: ::core::convert::TryInto::try_into(
                                    value.sparse_index()?,
                                )?,
                                data: ::core::convert::TryInto::try_into(value.data()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for SparseTensorRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for SparseTensorRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<SparseTensor>> for SparseTensor {
                        type Value = ::planus::Offset<SparseTensor>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<SparseTensor>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for SparseTensorRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[SparseTensorRef]",
                                    "read_as_root",
                                    0,
                                )
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct TensorDim {
                        pub size: i64,
                        pub name: ::core::option::Option<::planus::alloc::string::String>,
                    }

                    impl TensorDim {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            size: impl ::planus::WriteAsDefault<i64, i64>,
                            name: impl ::planus::WriteAsOptional<
                                ::planus::Offset<::core::primitive::str>,
                            >,
                        ) -> ::planus::Offset<Self> {
                            let prepared_size = size.prepare(builder, &0);

                            let prepared_name = name.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<6, 12>::new(builder);

                            if prepared_size.is_some() {
                                table_writer.calculate_size::<i64>(2);
                            }
                            if prepared_name.is_some() {
                                table_writer.calculate_size::<::planus::Offset<str>>(4);
                            }

                            table_writer.finish_calculating();

                            unsafe {
                                if let ::core::option::Option::Some(prepared_size) = prepared_size {
                                    table_writer.write::<_, _, 8>(0, &prepared_size);
                                }
                                if let ::core::option::Option::Some(prepared_name) = prepared_name {
                                    table_writer.write::<_, _, 4>(1, &prepared_name);
                                }
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<TensorDim>> for TensorDim {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<TensorDim>> for TensorDim {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<TensorDim>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<TensorDim> for TensorDim {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<TensorDim> {
                            TensorDim::create(builder, &self.size, &self.name)
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct TensorDimRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TensorDimRef<'a> {
                        pub fn size(&self) -> ::planus::Result<i64> {
                            ::core::result::Result::Ok(
                                self.0.access(0, "TensorDim", "size")?.unwrap_or(0),
                            )
                        }

                        pub fn name(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<&'a ::core::primitive::str>>
                        {
                            self.0.access(1, "TensorDim", "name")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TensorDimRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TensorDimRef");
                            f.field("size", &self.size());
                            if let ::core::option::Option::Some(name) = self.name().transpose() {
                                f.field("name", &name);
                            }
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TensorDimRef<'a>> for TensorDim {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TensorDimRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                size: ::core::convert::TryInto::try_into(value.size()?)?,
                                name: if let ::core::option::Option::Some(name) = value.name()? {
                                    ::core::option::Option::Some(
                                        ::core::convert::TryInto::try_into(name)?,
                                    )
                                } else {
                                    ::core::option::Option::None
                                },
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TensorDimRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TensorDimRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TensorDimRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<TensorDim>> for TensorDim {
                        type Value = ::planus::Offset<TensorDim>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<TensorDim>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TensorDimRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TensorDimRef]", "read_as_root", 0)
                            })
                        }
                    }

                    #[derive(Clone, Debug, PartialEq, ::serde::Serialize, ::serde::Deserialize)]
                    pub struct Tensor {
                        pub type_: self::Type,
                        pub shape: ::planus::alloc::vec::Vec<self::TensorDim>,
                        pub strides: ::core::option::Option<::planus::alloc::vec::Vec<i64>>,
                        pub data: self::Buffer,
                    }

                    impl Tensor {
                        #[allow(clippy::too_many_arguments)]
                        pub fn create(
                            builder: &mut ::planus::Builder,
                            type_: impl ::planus::WriteAsUnion<self::Type>,
                            shape: impl ::planus::WriteAs<
                                ::planus::Offset<[::planus::Offset<self::TensorDim>]>,
                            >,
                            strides: impl ::planus::WriteAsOptional<::planus::Offset<[i64]>>,
                            data: impl ::planus::WriteAs<self::Buffer>,
                        ) -> ::planus::Offset<Self> {
                            let prepared_type_ = type_.prepare(builder);

                            let prepared_shape = shape.prepare(builder);

                            let prepared_strides = strides.prepare(builder);

                            let prepared_data = data.prepare(builder);

                            let mut table_writer =
                                ::planus::table_writer::TableWriter::<12, 29>::new(builder);

                            table_writer.calculate_size::<u8>(2);
                            table_writer.calculate_size::<::planus::Offset<self::Type>>(4);
                            table_writer.calculate_size::<::planus::Offset<[::planus::Offset<self::TensorDim>]>>(6);
                            if prepared_strides.is_some() {
                                table_writer.calculate_size::<::planus::Offset<[i64]>>(8);
                            }
                            table_writer.calculate_size::<self::Buffer>(10);

                            table_writer.finish_calculating();

                            unsafe {
                                table_writer.write::<_, _, 16>(4, &prepared_data);
                                table_writer.write::<_, _, 4>(1, &prepared_type_.offset());
                                table_writer.write::<_, _, 4>(2, &prepared_shape);
                                if let ::core::option::Option::Some(prepared_strides) =
                                    prepared_strides
                                {
                                    table_writer.write::<_, _, 4>(3, &prepared_strides);
                                }
                                table_writer.write::<_, _, 1>(0, &prepared_type_.tag());
                            }

                            table_writer.finish()
                        }
                    }

                    impl ::planus::WriteAs<::planus::Offset<Tensor>> for Tensor {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor> {
                            ::planus::WriteAsOffset::prepare(self, builder)
                        }
                    }

                    impl ::planus::WriteAsOptional<::planus::Offset<Tensor>> for Tensor {
                        type Prepared = ::planus::Offset<Self>;

                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::core::option::Option<::planus::Offset<Tensor>>
                        {
                            ::core::option::Option::Some(::planus::WriteAsOffset::prepare(
                                self, builder,
                            ))
                        }
                    }

                    impl ::planus::WriteAsOffset<Tensor> for Tensor {
                        fn prepare(
                            &self,
                            builder: &mut ::planus::Builder,
                        ) -> ::planus::Offset<Tensor> {
                            Tensor::create(
                                builder,
                                &self.type_,
                                &self.shape,
                                &self.strides,
                                &self.data,
                            )
                        }
                    }

                    #[derive(Copy, Clone)]
                    pub struct TensorRef<'a>(::planus::table_reader::Table<'a>);

                    impl<'a> TensorRef<'a> {
                        pub fn type_(&self) -> ::planus::Result<self::TypeRef<'a>> {
                            self.0.access_union_required(0, "Tensor", "type_")
                        }

                        pub fn shape(
                            &self,
                        ) -> ::planus::Result<
                            ::planus::Vector<'a, ::planus::Result<self::TensorDimRef<'a>>>,
                        > {
                            self.0.access_required(2, "Tensor", "shape")
                        }

                        pub fn strides(
                            &self,
                        ) -> ::planus::Result<::core::option::Option<::planus::Vector<'a, i64>>>
                        {
                            self.0.access(3, "Tensor", "strides")
                        }

                        pub fn data(&self) -> ::planus::Result<self::BufferRef<'a>> {
                            self.0.access_required(4, "Tensor", "data")
                        }
                    }

                    impl<'a> ::core::fmt::Debug for TensorRef<'a> {
                        fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                            let mut f = f.debug_struct("TensorRef");
                            f.field("type_", &self.type_());
                            f.field("shape", &self.shape());
                            if let ::core::option::Option::Some(strides) =
                                self.strides().transpose()
                            {
                                f.field("strides", &strides);
                            }
                            f.field("data", &self.data());
                            f.finish()
                        }
                    }

                    impl<'a> ::core::convert::TryFrom<TensorRef<'a>> for Tensor {
                        type Error = ::planus::Error;

                        #[allow(unreachable_code)]
                        fn try_from(value: TensorRef<'a>) -> ::planus::Result<Self> {
                            ::core::result::Result::Ok(Self {
                                type_: ::core::convert::TryInto::try_into(value.type_()?)?,
                                shape: value.shape()?.to_vec_result()?,
                                strides: if let ::core::option::Option::Some(strides) =
                                    value.strides()?
                                {
                                    ::core::option::Option::Some(strides.to_vec()?)
                                } else {
                                    ::core::option::Option::None
                                },
                                data: ::core::convert::TryInto::try_into(value.data()?)?,
                            })
                        }
                    }

                    impl<'a> ::planus::TableRead<'a> for TensorRef<'a> {
                        fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::core::result::Result<Self, ::planus::errors::ErrorKind>
                        {
                            ::core::result::Result::Ok(Self(
                                ::planus::table_reader::Table::from_buffer(buffer, offset)?,
                            ))
                        }
                    }

                    impl<'a> ::planus::VectorReadInner<'a> for TensorRef<'a> {
                        type Error = ::planus::Error;
                        const STRIDE: usize = 4;

                        unsafe fn from_buffer(
                            buffer: ::planus::SliceWithStartOffset<'a>,
                            offset: usize,
                        ) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(buffer, offset).map_err(|error_kind| {
                                error_kind.with_error_location(
                                    "[TensorRef]",
                                    "get",
                                    buffer.offset_from_start,
                                )
                            })
                        }
                    }

                    impl ::planus::VectorWrite<::planus::Offset<Tensor>> for Tensor {
                        type Value = ::planus::Offset<Tensor>;
                        const STRIDE: usize = 4;
                        fn prepare(&self, builder: &mut ::planus::Builder) -> Self::Value {
                            ::planus::WriteAs::prepare(self, builder)
                        }

                        #[inline]
                        unsafe fn write_values(
                            values: &[::planus::Offset<Tensor>],
                            bytes: *mut ::core::mem::MaybeUninit<u8>,
                            buffer_position: u32,
                        ) {
                            let bytes = bytes as *mut [::core::mem::MaybeUninit<u8>; 4];
                            for (i, v) in ::core::iter::Iterator::enumerate(values.iter()) {
                                ::planus::WriteAsPrimitive::write(
                                    v,
                                    ::planus::Cursor::new(&mut *bytes.add(i)),
                                    buffer_position - (Self::STRIDE * i) as u32,
                                );
                            }
                        }
                    }

                    impl<'a> ::planus::ReadAsRoot<'a> for TensorRef<'a> {
                        fn read_as_root(slice: &'a [u8]) -> ::planus::Result<Self> {
                            ::planus::TableRead::from_buffer(
                                ::planus::SliceWithStartOffset {
                                    buffer: slice,
                                    offset_from_start: 0,
                                },
                                0,
                            )
                            .map_err(|error_kind| {
                                error_kind.with_error_location("[TensorRef]", "read_as_root", 0)
                            })
                        }
                    }
                }
            }
        }
    }
}
