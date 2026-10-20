//! Allocation preflight over GENERATED Arrow IPC tables, before owned schema
//! conversion or array construction. Wire bytes are not an allocation budget:
//! aliases clone strings/fields, and a zero-buffer Null child can materialize a
//! logical bitmap while a Struct/dictionary is validated.
//!
//! Independent source-fixed limits cover schema ownership, array/object
//! ownership and potential logical bitmaps, each <= the 16MiB payload ceiling.
//! This is NOT a 16MiB process-memory claim: wire/body copies, aligned buffers,
//! concat outputs and the previously bounded codec workspaces also coexist.
//! Counters charge occurrences, NEVER unique FlatBuffer addresses.
use arrow::array::{ArrayData, ArrayRef, RecordBatch};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field};
use arrow::ipc::{self, Field as IpcField, Schema as IpcSchema, Type};
use std::mem::size_of;
use tonic::Status;

const MAX_SCHEMA_DEPTH: usize = 64;
// Arc's strong/weak reference-count words precede its contained allocation.
const ARC_HEADER: usize = 2 * size_of::<usize>();

fn invalid(message: &str) -> Status {
    Status::invalid_argument(format!("decision payload IPC metadata: {message}"))
}
fn add(counter: &mut usize, bytes: usize, limit: usize, label: &str) -> Result<(), Status> {
    *counter = counter.checked_add(bytes).ok_or_else(|| invalid(label))?;
    if *counter > limit {
        return Err(invalid(label));
    }
    Ok(())
}
fn mul(a: usize, b: usize) -> Result<usize, Status> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("allocation size overflow"))
}
fn vector_capacity(n: usize) -> Result<usize, Status> {
    if n == 0 {
        return Ok(0);
    }
    // Vec incremental growth: reserve space for the initial four elements and
    // next power-of-two growth, even where exact-size iterators allocate less.
    n.max(4)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("vector capacity overflow"))
}

/// Stream-scoped (not per-message) ownership counters. No decoded Arrow schema
/// or payload bytes are retained in this proof state.
pub(super) struct MetadataBudget<'a> {
    limit: usize,
    schema_bytes: usize,
    array_bytes: usize,
    bitmap_bytes: usize,
    // Largest owned field subtree is an upper bound on ANY datatype clone.
    // Decoder's builder/ArrayData/concrete ArrayRef and ancestor conversions
    // may clone datatypes. Charging full owned subtrees deliberately includes
    // names/maps which actual Arc-backed datatype clones merely share.
    largest_field: usize,
    max_depth: usize,
    record_views: usize,
    schema: Option<IpcSchema<'a>>,
    dictionary_fields: std::collections::HashMap<i64, IpcField<'a>>,
    representation: super::decision_ipc_representation::RepresentationBudget<'a>,
}
impl<'a> MetadataBudget<'a> {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit: limit.min(super::decision_batch::MAX_PAYLOAD_BYTES),
            schema_bytes: 0,
            array_bytes: 0,
            bitmap_bytes: 0,
            largest_field: 0,
            max_depth: 0,
            record_views: 0,
            schema: None,
            dictionary_fields: std::collections::HashMap::new(),
            representation: super::decision_ipc_representation::RepresentationBudget::new(limit),
        }
    }
    #[cfg(test)]
    pub(super) fn reduce_representation_limit(&mut self, limit: usize) {
        self.representation.reduce_representation_limit(limit);
    }
    pub(super) fn message(&mut self) -> Result<(), Status> {
        // The framing walker retains Vec<IpcMessageBounds>; four slots per
        // message cover initial Vec capacity as well as later growth.
        self.array_charge(4 * size_of::<super::decision_batch::IpcMessageBounds>())
    }
    fn schema_charge(&mut self, bytes: usize) -> Result<(), Status> {
        add(
            &mut self.schema_bytes,
            bytes,
            self.limit,
            "owned schema allocation budget exceeded",
        )
    }
    fn array_charge(&mut self, bytes: usize) -> Result<(), Status> {
        add(
            &mut self.array_bytes,
            bytes,
            self.limit,
            "array metadata allocation budget exceeded",
        )
    }
    fn bitmap_charge(&mut self, length: usize) -> Result<(), Status> {
        let bytes = length
            .checked_add(7)
            .ok_or_else(|| invalid("logical bitmap size overflow"))?
            / 8;
        add(
            &mut self.bitmap_bytes,
            bytes,
            self.limit,
            "logical bitmap allocation budget exceeded",
        )
    }
    fn fields_container(&mut self, count: usize) -> Result<(), Status> {
        // convert.rs builds Vec<Field>, then Fields::from_iter creates one
        // Arc<Field> per element and an Arc<[FieldRef]> (fields.rs267-287).
        // Both transient Vec backing and final Arc slice storage are charged.
        let slots = vector_capacity(count)?;
        self.schema_charge(mul(slots, size_of::<Field>())?)?;
        self.schema_charge(mul(slots, size_of::<std::sync::Arc<Field>>())?)?;
        self.schema_charge(ARC_HEADER)?;
        Ok(())
    }
    fn key_value(&mut self, key: Option<&str>, value: Option<&str>) -> Result<(), Status> {
        // Charge every entry and both strings even when HashMap overwrites a
        // duplicate. A one-entry map's four buckets dominate amortized table
        // storage: four (String,String) slots, bucket control bytes and the
        // pinned hashbrown implementation's <=16-byte SIMD control tail.
        self.schema_charge(4 * (size_of::<(String, String)>() + 1) + size_of::<u128>())?;
        self.schema_charge(key.map_or(0, str::len))?;
        self.schema_charge(value.map_or(0, str::len))
    }
    pub(super) fn schema(&mut self, schema: IpcSchema<'a>) -> Result<(), Status> {
        let fields = schema
            .fields()
            .ok_or_else(|| invalid("schema fields missing"))?;
        self.fields_container(fields.len())?;
        self.schema_charge(size_of::<arrow::datatypes::Schema>() + ARC_HEADER)?;
        if let Some(metadata) = schema.custom_metadata() {
            for kv in metadata {
                self.key_value(kv.key(), kv.value())?;
            }
        }
        if !matches!(
            schema.endianness(),
            ipc::Endianness::Little | ipc::Endianness::Big
        ) {
            return Err(invalid("invalid schema endianness"));
        }
        self.record_views = 0;
        self.dictionary_fields.clear();
        self.schema = Some(schema);
        for field in fields {
            self.field(field, 0, schema.endianness())?;
        }
        self.representation
            .schema(self.largest_field, self.max_depth);
        for field in fields {
            self.index_dictionaries(field);
            self.record_views = self
                .record_views
                .checked_add(view_fields(field, true)?)
                .ok_or_else(|| invalid("view field count overflow"))?;
        }
        Ok(())
    }
    fn field(
        &mut self,
        field: IpcField<'a>,
        depth: usize,
        endian: ipc::Endianness,
    ) -> Result<(), Status> {
        if depth >= MAX_SCHEMA_DEPTH {
            return Err(invalid("schema nesting budget exceeded"));
        }
        self.max_depth = self.max_depth.max(depth + 1);
        let before = self.schema_bytes;
        let name = field.name().ok_or_else(|| invalid("field name missing"))?;
        self.schema_charge(name.len())?;
        self.schema_charge(size_of::<Field>() + ARC_HEADER + size_of::<DataType>())?;
        if let Some(metadata) = field.custom_metadata() {
            for kv in metadata {
                self.key_value(kv.key(), kv.value())?;
            }
        }
        if let Some(dictionary) = field.dictionary() {
            let index = dictionary
                .indexType()
                .ok_or_else(|| invalid("dictionary index type missing"))?;
            if !matches!(index.bitWidth(), 8 | 16 | 32 | 64) {
                return Err(invalid("invalid dictionary index width"));
            }
            // Dictionary(index,value) owns two Box<DataType> allocations.
            self.schema_charge(2 * size_of::<DataType>())?;
        }
        self.field_geometry(field, endian)?;
        if let Some(children) = field.children() {
            self.fields_container(children.len())?;
            for child in children {
                self.field(child, depth + 1, endian)?;
            }
        }
        // Only index a dictionary AFTER its entire subtree was charged and
        // validated; never traverse an aliased subtree before its budget.
        if let Some(dictionary) = field.dictionary() {
            self.schema_charge(4 * (size_of::<(i64, IpcField<'a>)>() + 1) + size_of::<u128>())?;
            let _ = dictionary; // indexing waits until the whole schema is charged
        }
        self.largest_field = self.largest_field.max(self.schema_bytes - before);
        Ok(())
    }
    fn index_dictionaries(&mut self, field: IpcField<'a>) {
        if let Some(dictionary) = field.dictionary() {
            self.dictionary_fields
                .entry(dictionary.id())
                .or_insert(field);
        }
        match field.type_type() {
            Type::Struct_
            | Type::Union
            | Type::List
            | Type::LargeList
            | Type::FixedSizeList
            | Type::Map => {
                if let Some(children) = field.children() {
                    for child in children {
                        self.index_dictionaries(child);
                    }
                }
            }
            // Arrow Field::_fields discovers dictionaries only under the
            // values field of RunEndEncoded (run-end indices are plain Int).
            Type::RunEndEncoded => {
                if let Some(children) = field.children() {
                    self.index_dictionaries(children.get(1));
                }
            }
            _ => {}
        }
    }
    fn field_geometry(
        &mut self,
        field: IpcField<'_>,
        endian: ipc::Endianness,
    ) -> Result<(), Status> {
        let children = field.children().map_or(0, |v| v.len());
        let require_arity = |n| {
            if children == n {
                Ok(())
            } else {
                Err(invalid("invalid nested field arity"))
            }
        };
        let valid_time = |unit| {
            matches!(
                unit,
                ipc::TimeUnit::SECOND
                    | ipc::TimeUnit::MILLISECOND
                    | ipc::TimeUnit::MICROSECOND
                    | ipc::TimeUnit::NANOSECOND
            )
        };
        match field.type_type() {
            Type::Null
            | Type::Bool
            | Type::Binary
            | Type::LargeBinary
            | Type::BinaryView
            | Type::Utf8
            | Type::LargeUtf8
            | Type::Utf8View
            | Type::Struct_ => {}
            Type::Int => {
                let x = field
                    .type_as_int()
                    .ok_or_else(|| invalid("integer type missing"))?;
                if !matches!(x.bitWidth(), 8 | 16 | 32 | 64) {
                    return Err(invalid("invalid integer width"));
                }
            }
            Type::FloatingPoint => {
                let x = field
                    .type_as_floating_point()
                    .ok_or_else(|| invalid("float type missing"))?;
                if !matches!(
                    x.precision(),
                    ipc::Precision::HALF | ipc::Precision::SINGLE | ipc::Precision::DOUBLE
                ) {
                    return Err(invalid("invalid float precision"));
                }
            }
            Type::FixedSizeBinary => {
                let x = field
                    .type_as_fixed_size_binary()
                    .ok_or_else(|| invalid("binary type missing"))?;
                if x.byteWidth() < 0 {
                    return Err(invalid("negative fixed binary width"));
                }
            }
            Type::Date => {
                let x = field
                    .type_as_date()
                    .ok_or_else(|| invalid("date type missing"))?;
                if !matches!(x.unit(), ipc::DateUnit::DAY | ipc::DateUnit::MILLISECOND) {
                    return Err(invalid("invalid date unit"));
                }
            }
            Type::Time => {
                let x = field
                    .type_as_time()
                    .ok_or_else(|| invalid("time type missing"))?;
                if !matches!(
                    (x.bitWidth(), x.unit()),
                    (32, ipc::TimeUnit::SECOND | ipc::TimeUnit::MILLISECOND)
                        | (64, ipc::TimeUnit::MICROSECOND | ipc::TimeUnit::NANOSECOND)
                ) {
                    return Err(invalid("invalid time width/unit"));
                }
            }
            Type::Timestamp => {
                let x = field
                    .type_as_timestamp()
                    .ok_or_else(|| invalid("timestamp type missing"))?;
                if !valid_time(x.unit()) {
                    return Err(invalid("invalid timestamp unit"));
                }
                self.schema_charge(x.timezone().map_or(0, str::len) + ARC_HEADER)?;
            }
            Type::Duration => {
                let x = field
                    .type_as_duration()
                    .ok_or_else(|| invalid("duration type missing"))?;
                if !valid_time(x.unit()) {
                    return Err(invalid("invalid duration unit"));
                }
            }
            Type::Interval => {
                let x = field
                    .type_as_interval()
                    .ok_or_else(|| invalid("interval type missing"))?;
                if !matches!(
                    x.unit(),
                    ipc::IntervalUnit::YEAR_MONTH
                        | ipc::IntervalUnit::DAY_TIME
                        | ipc::IntervalUnit::MONTH_DAY_NANO
                ) {
                    return Err(invalid("invalid interval unit"));
                }
            }
            Type::List | Type::LargeList => require_arity(1)?,
            Type::FixedSizeList => {
                require_arity(1)?;
                let x = field
                    .type_as_fixed_size_list()
                    .ok_or_else(|| invalid("list type missing"))?;
                if x.listSize() < 0 {
                    return Err(invalid("negative fixed list size"));
                }
            }
            Type::RunEndEncoded => {
                require_arity(2)?;
                let children = field
                    .children()
                    .ok_or_else(|| invalid("run-end children missing"))?;
                let run_ends = children.get(0);
                let index = run_ends
                    .type_as_int()
                    .ok_or_else(|| invalid("invalid run-end index type"))?;
                if run_ends.dictionary().is_some()
                    || !index.is_signed()
                    || !matches!(index.bitWidth(), 16 | 32 | 64)
                {
                    return Err(invalid("invalid run-end index geometry"));
                }
            }
            Type::Map => {
                require_arity(1)?;
                field
                    .type_as_map()
                    .ok_or_else(|| invalid("map type missing"))?;
                let entries = field
                    .children()
                    .ok_or_else(|| invalid("map entries missing"))?
                    .get(0);
                if entries.dictionary().is_some()
                    || entries.type_type() != Type::Struct_
                    || entries.children().map_or(0, |v| v.len()) != 2
                {
                    return Err(invalid("invalid map entry geometry"));
                }
            }
            Type::Decimal => {
                let x = field
                    .type_as_decimal()
                    .ok_or_else(|| invalid("decimal type missing"))?;
                if endian == ipc::Endianness::Big
                    || !matches!(x.bitWidth(), 32 | 64 | 128 | 256)
                    || u8::try_from(x.precision()).is_err()
                    || i8::try_from(x.scale()).is_err()
                {
                    return Err(invalid("invalid/unsupported Arrow56 decimal geometry"));
                }
            }
            Type::Union => {
                let x = field
                    .type_as_union()
                    .ok_or_else(|| invalid("union type missing"))?;
                if !matches!(x.mode(), ipc::UnionMode::Dense | ipc::UnionMode::Sparse) {
                    return Err(invalid("invalid union mode"));
                }
                if let Some(ids) = x.typeIds() {
                    if ids.len() != children {
                        return Err(invalid("union type ID/child count mismatch"));
                    }
                    let mut seen = [false; 128];
                    for id in ids {
                        let id =
                            usize::try_from(id).map_err(|_| invalid("negative union type ID"))?;
                        if id >= seen.len() || seen[id] {
                            return Err(invalid("invalid/duplicate union type ID"));
                        }
                        seen[id] = true;
                    }
                    self.schema_charge(mul(ids.len(), size_of::<i8>())?)?;
                } else if children >= 128 {
                    return Err(invalid(
                        "implicit union IDs exceed Arrow56 conversion range",
                    ));
                }
            }
            _ => return Err(invalid("unsupported Arrow56 field type")),
        }
        Ok(())
    }
    pub(super) fn record(
        &mut self,
        batch: &ipc::RecordBatch<'_>,
        dictionary: Option<i64>,
    ) -> Result<(), Status> {
        if batch.length() < 0 {
            return Err(invalid("negative record/dictionary length"));
        }
        let nodes = batch
            .nodes()
            .ok_or_else(|| invalid("field nodes missing"))?;
        let buffers = batch
            .buffers()
            .ok_or_else(|| invalid("body buffers missing"))?;
        self.array_charge(
            4 * size_of::<RecordBatch>()
                + 4 * (size_of::<(i64, ArrayRef)>() + 1)
                + size_of::<u128>(),
        )?;
        // Decoder's transient child-data vectors, final ArrayRef vectors and
        // each concrete Arc array are distinct ownership. Per-node four slots
        // cover Vec's initial capacity even for one-child containers.
        let array_object = size_of::<arrow::array::StructArray>()
            .max(size_of::<arrow::array::StringViewArray>())
            .max(size_of::<
                arrow::array::DictionaryArray<arrow::datatypes::Int64Type>,
            >())
            .max(size_of::<arrow::array::FixedSizeListArray>())
            .max(size_of::<arrow::array::UnionArray>())
            .max(size_of::<arrow::array::RunArray<arrow::datatypes::Int64Type>>())
            .max(size_of::<ArrayData>())
            .max(size_of::<arrow::array::LargeListArray>())
            .max(size_of::<arrow::array::LargeStringArray>())
            .max(size_of::<arrow::array::LargeBinaryArray>())
            .max(size_of::<arrow::array::MapArray>())
            .max(size_of::<arrow::array::BooleanArray>())
            .max(size_of::<arrow::array::NullArray>())
            .max(size_of::<
                arrow::array::PrimitiveArray<arrow::datatypes::Decimal256Type>,
            >());
        let per_node =
            4 * (size_of::<ArrayData>() + size_of::<ArrayRef>()) + ARC_HEADER + array_object;
        // Parent ArrayData::to_data/clone may retain a descriptor subtree at
        // EACH ancestor. Depth is measured from this stream's actual schema,
        // with three additional live stages: reader Vec, ArrayData builder,
        // concrete ArrayRef construction. This is an explicit ownership bound,
        // not a guessed process-memory multiplier.
        let stages = self
            .max_depth
            .checked_add(3)
            .ok_or_else(|| invalid("allocation stage overflow"))?;
        self.array_charge(mul(mul(nodes.len(), per_node)?, stages)?)?;
        self.array_charge(mul(mul(nodes.len(), self.largest_field)?, stages)?)?;
        self.array_charge(mul(
            mul(vector_capacity(buffers.len())?, size_of::<Buffer>())?,
            stages,
        )?)?;
        // The codec verifier returns one actual decoded length per buffer;
        // this independent temporary Vec is allocated only AFTER this charge.
        self.array_charge(mul(vector_capacity(buffers.len())?, size_of::<usize>())?)?;
        for node in nodes {
            let len = usize::try_from(node.length())
                .map_err(|_| invalid("negative field node length"))?;
            let nulls = usize::try_from(node.null_count())
                .map_err(|_| invalid("negative field node null count"))?;
            if nulls > len {
                return Err(invalid("field node null count exceeds length"));
            }
            // Includes dictionary/child lengths >4096. This is a bitmap memory
            // budget, never a record-row cap; Null logical_nulls may allocate it
            // even when no encoded null buffer exists.
            self.bitmap_charge(len)?;
        }
        let expected_views = match dictionary {
            Some(id) => view_fields(
                *self
                    .dictionary_fields
                    .get(&id)
                    .ok_or_else(|| invalid("unknown dictionary ID"))?,
                false,
            )?,
            None => self.record_views,
        };
        if batch.variadicBufferCounts().map_or(0, |v| v.len()) != expected_views {
            return Err(invalid(
                "variadic count vector does not match schema view fields",
            ));
        }
        if let Some(counts) = batch.variadicBufferCounts() {
            self.array_charge(mul(vector_capacity(counts.len())?, size_of::<i64>())?)?;
            let mut consumed = 0usize;
            for count in counts {
                let with_fixed = count
                    .checked_add(2)
                    .ok_or_else(|| invalid("variadic buffer count overflow"))?;
                if count < 0 {
                    return Err(invalid("negative variadic buffer count"));
                }
                let n = usize::try_from(with_fixed)
                    .map_err(|_| invalid("variadic buffer count overflow"))?;
                consumed = consumed
                    .checked_add(n)
                    .ok_or_else(|| invalid("variadic buffer count overflow"))?;
                if consumed > buffers.len() {
                    return Err(invalid(
                        "variadic buffer counts exceed actual buffer vector",
                    ));
                }
            }
        }
        Ok(())
    }
    /// Match the canonical reader's node/buffer consumption WITHOUT constructing
    /// arrays or copying buffer contents. Lengths came from the actual bounded
    /// codec verifier, not the untrusted compressed-length declarations.
    pub(super) fn layout(
        &mut self,
        batch: ipc::RecordBatch<'_>,
        dictionary: Option<(i64, bool)>,
        version: ipc::MetadataVersion,
        lengths: &[usize],
    ) -> Result<(), Status> {
        let mut cursor = LayoutCursor {
            batch,
            version,
            lengths,
            node: 0,
            buffer: 0,
            variadic: 0,
        };
        if let Some((id, _)) = dictionary {
            let field = self
                .dictionary_fields
                .get(&id)
                .ok_or_else(|| invalid("unknown dictionary ID"))?;
            // Arrow creates one synthetic field with the dictionary's VALUE
            // type; only this root dictionary wrapper is removed.
            cursor.field(*field, false, 0)?;
        } else {
            let fields = self
                .schema
                .ok_or_else(|| invalid("record before schema"))?
                .fields()
                .ok_or_else(|| invalid("schema fields missing"))?;
            for field in fields {
                cursor.field(field, true, 0)?;
            }
        }
        let nodes = batch
            .nodes()
            .ok_or_else(|| invalid("field nodes missing"))?;
        let counts = batch.variadicBufferCounts().map_or(0, |v| v.len());
        if cursor.node != nodes.len() || cursor.buffer != lengths.len() || cursor.variadic != counts
        {
            return Err(invalid("node/buffer count does not match schema"));
        }
        if let Some((id, delta)) = dictionary {
            self.representation.dictionary(
                *self
                    .dictionary_fields
                    .get(&id)
                    .ok_or_else(|| invalid("unknown dictionary ID"))?,
                batch,
                lengths,
                version,
                id,
                delta,
            )?;
        } else {
            let fields = self
                .schema
                .ok_or_else(|| invalid("record before schema"))?
                .fields()
                .ok_or_else(|| invalid("schema fields missing"))?;
            self.representation
                .record(fields.iter(), batch, lengths, version)?;
        }
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Result<(), Status> {
        self.representation.finish()
    }
}

/// Borrowed, constant-space traversal state. Recursion is bounded by the
/// already-charged schema depth; it retains no decoded schema or body copies.
struct LayoutCursor<'a, 'b> {
    batch: ipc::RecordBatch<'a>,
    version: ipc::MetadataVersion,
    lengths: &'b [usize],
    node: usize,
    buffer: usize,
    variadic: usize,
}
impl LayoutCursor<'_, '_> {
    fn node(&mut self) -> Result<(usize, usize), Status> {
        let nodes = self
            .batch
            .nodes()
            .ok_or_else(|| invalid("field nodes missing"))?;
        if self.node >= nodes.len() {
            return Err(invalid("schema requires more field nodes"));
        }
        let node = nodes.get(self.node);
        self.node += 1;
        let len =
            usize::try_from(node.length()).map_err(|_| invalid("negative field node length"))?;
        let nulls = usize::try_from(node.null_count())
            .map_err(|_| invalid("negative field node null count"))?;
        if nulls > len {
            return Err(invalid("field node null count exceeds length"));
        }
        Ok((len, nulls))
    }
    fn buffer(&mut self, minimum: usize) -> Result<(), Status> {
        let len = *self
            .lengths
            .get(self.buffer)
            .ok_or_else(|| invalid("schema requires more body buffers"))?;
        self.buffer += 1;
        if len < minimum {
            return Err(invalid("body buffer shorter than schema requires"));
        }
        Ok(())
    }
    fn validity(&mut self, len: usize, nulls: usize) -> Result<(), Status> {
        let minimum = if nulls == 0 {
            0
        } else {
            len.checked_add(7)
                .ok_or_else(|| invalid("validity length overflow"))?
                / 8
        };
        self.buffer(minimum)
    }
    fn field(
        &mut self,
        field: IpcField<'_>,
        dictionary_wrapper: bool,
        depth: usize,
    ) -> Result<(), Status> {
        if depth >= MAX_SCHEMA_DEPTH {
            return Err(invalid("schema nesting budget exceeded"));
        }
        let (len, nulls) = self.node()?;
        if dictionary_wrapper && field.dictionary().is_some() {
            self.validity(len, nulls)?;
            return self.buffer(0); // indices; ArrayData validates index width/content
        }
        match field.type_type() {
            Type::Null => {
                if len != nulls {
                    return Err(invalid("Null field length/null count mismatch"));
                }
            }
            Type::Utf8View | Type::BinaryView => {
                let counts = self
                    .batch
                    .variadicBufferCounts()
                    .ok_or_else(|| invalid("view variadic count missing"))?;
                if self.variadic >= counts.len() {
                    return Err(invalid("view variadic count missing"));
                }
                let count = usize::try_from(counts.get(self.variadic))
                    .map_err(|_| invalid("negative variadic buffer count"))?;
                self.variadic += 1;
                self.validity(len, nulls)?;
                self.buffer(mul(len, 16)?)?;
                for _ in 0..count {
                    self.buffer(0)?;
                }
            }
            Type::Utf8 | Type::LargeUtf8 | Type::Binary | Type::LargeBinary => {
                self.validity(len, nulls)?;
                self.buffer(0)?; // checked offsets/content remain the canonical decoder's job
                self.buffer(0)?;
            }
            Type::List | Type::LargeList | Type::Map => {
                self.validity(len, nulls)?;
                self.buffer(0)?;
                self.children(field, depth)?;
            }
            Type::FixedSizeList | Type::Struct_ => {
                self.validity(len, nulls)?;
                self.children(field, depth)?;
            }
            Type::RunEndEncoded => self.children(field, depth)?,
            Type::Union => {
                // Arrow56 consumes the legacy validity slot before V5 even
                // though it does not construct a Union validity bitmap.
                if self.version < ipc::MetadataVersion::V5 {
                    self.validity(len, nulls)?;
                }
                self.buffer(len)?;
                let union = field
                    .type_as_union()
                    .ok_or_else(|| invalid("union type missing"))?;
                if union.mode() == ipc::UnionMode::Dense {
                    self.buffer(mul(len, 4)?)?;
                }
                self.children(field, depth)?;
            }
            _ => {
                self.validity(len, nulls)?;
                self.buffer(0)?;
            }
        }
        Ok(())
    }
    fn children(&mut self, field: IpcField<'_>, depth: usize) -> Result<(), Status> {
        if let Some(children) = field.children() {
            for child in children {
                self.field(child, true, depth + 1)?;
            }
        }
        Ok(())
    }
}

// Canonical reader consumes one variadic entry for every actual View array.
// Dictionary indices do not consume entries; their separately decoded value
// subtree does. This is a typed shape count, not a replacement array decoder.
fn view_fields(field: IpcField<'_>, skip_dictionary: bool) -> Result<usize, Status> {
    fn visit(field: IpcField<'_>, skip_dictionary: bool, depth: usize) -> Result<usize, Status> {
        if depth >= MAX_SCHEMA_DEPTH {
            return Err(invalid("schema nesting budget exceeded"));
        }
        if skip_dictionary && field.dictionary().is_some() {
            return Ok(0);
        }
        let mut count = usize::from(matches!(
            field.type_type(),
            Type::Utf8View | Type::BinaryView
        ));
        if matches!(
            field.type_type(),
            Type::List
                | Type::LargeList
                | Type::FixedSizeList
                | Type::Map
                | Type::Struct_
                | Type::RunEndEncoded
                | Type::Union
        ) {
            if let Some(children) = field.children() {
                for child in children {
                    count = count
                        .checked_add(visit(child, true, depth + 1)?)
                        .ok_or_else(|| invalid("view field count overflow"))?;
                }
            }
        }
        Ok(count)
    }
    visit(field, skip_dictionary, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::ipc::gen::Message::{FieldNode, Message, MessageArgs, RecordBatchArgs};
    use flatbuffers::FlatBufferBuilder;

    fn schema_bytes(references: usize, nested: bool, shared_metadata: bool) -> Vec<u8> {
        let mut b = FlatBufferBuilder::new();
        let name = b.create_string(&"field".repeat(128));
        let key = b.create_string("shared-key");
        let value = b.create_string(&"value".repeat(128));
        let kv = ipc::KeyValue::create(
            &mut b,
            &ipc::KeyValueArgs {
                key: Some(key),
                value: Some(value),
            },
        );
        let metadata = b.create_vector(&[kv, kv]);
        let null_type = ipc::Null::create(&mut b, &ipc::NullArgs {});
        let leaf = ipc::Field::create(
            &mut b,
            &ipc::FieldArgs {
                name: Some(name),
                nullable: true,
                type_type: Type::Null,
                type_: Some(null_type.as_union_value()),
                custom_metadata: shared_metadata.then_some(metadata),
                ..Default::default()
            },
        );
        let field = if nested {
            let children = b.create_vector(&[leaf, leaf]);
            let struct_type = ipc::Struct_::create(&mut b, &ipc::Struct_Args {});
            ipc::Field::create(
                &mut b,
                &ipc::FieldArgs {
                    name: Some(name),
                    nullable: true,
                    type_type: Type::Struct_,
                    type_: Some(struct_type.as_union_value()),
                    children: Some(children),
                    ..Default::default()
                },
            )
        } else {
            leaf
        };
        let fields = b.create_vector(&vec![field; references]);
        let schema = ipc::Schema::create(
            &mut b,
            &ipc::SchemaArgs {
                fields: Some(fields),
                ..Default::default()
            },
        );
        let message = Message::create(
            &mut b,
            &MessageArgs {
                header_type: ipc::MessageHeader::Schema,
                header: Some(schema.as_union_value()),
                ..Default::default()
            },
        );
        b.finish(message, None);
        b.finished_data().to_vec()
    }
    fn visit_schema(bytes: &[u8], limit: usize) -> Result<MetadataBudget<'_>, Status> {
        let message =
            ipc::root_as_message(bytes).map_err(|_| invalid("fixture metadata invalid"))?;
        let schema = message
            .header_as_schema()
            .ok_or_else(|| invalid("fixture schema missing"))?;
        let mut budget = MetadataBudget::new(limit);
        budget.schema(schema)?;
        Ok(budget)
    }
    fn record_bytes(nodes: &[FieldNode], counts: &[i64], buffer_count: usize) -> Vec<u8> {
        let mut b = FlatBufferBuilder::new();
        let nodes = b.create_vector(nodes);
        let buffers = b.create_vector(&vec![ipc::Buffer::new(0, 0); buffer_count]);
        let counts = b.create_vector(counts);
        let rb = ipc::RecordBatch::create(
            &mut b,
            &RecordBatchArgs {
                length: 1,
                nodes: Some(nodes),
                buffers: Some(buffers),
                variadicBufferCounts: Some(counts),
                ..Default::default()
            },
        );
        let msg = Message::create(
            &mut b,
            &MessageArgs {
                header_type: ipc::MessageHeader::RecordBatch,
                header: Some(rb.as_union_value()),
                ..Default::default()
            },
        );
        b.finish(msg, None);
        b.finished_data().to_vec()
    }
    fn visit_record(bytes: &[u8], limit: usize) -> Result<MetadataBudget<'_>, Status> {
        let msg = ipc::root_as_message(bytes).map_err(|_| invalid("fixture metadata invalid"))?;
        let rb = msg
            .header_as_record_batch()
            .ok_or_else(|| invalid("fixture record missing"))?;
        let mut budget = MetadataBudget::new(limit);
        // Standalone count-arithmetic fixture: full schema association is
        // independently exercised by the real View IPC roundtrip below.
        budget.record_views = rb.variadicBufferCounts().map_or(0, |v| v.len());
        budget.record(&rb, None)?;
        Ok(budget)
    }
    #[test]
    fn shared_schema_occurrences_and_owned_budget_boundary() -> Result<(), Status> {
        for (nested, metadata) in [(false, false), (false, true), (true, false), (true, true)] {
            let small = schema_bytes(1, nested, metadata);
            let aliased = schema_bytes(64, nested, metadata);
            assert!(
                aliased.len() < small.len() + 512,
                "wire uses repeated offsets, not copied strings"
            );
            let small_cost = visit_schema(&small, 1024 * 1024)?.schema_bytes;
            let error = visit_schema(&aliased, small_cost * 2)
                .err()
                .ok_or_else(|| invalid("alias expansion must reject"))?;
            assert!(error.message().contains("owned schema"));
            let cost = visit_schema(&small, 1024 * 1024)?.schema_bytes;
            visit_schema(&small, cost)?;
            assert!(visit_schema(&small, cost - 1).is_err());
        }
        Ok(())
    }
    #[test]
    fn signed_node_geometry_and_implied_bitmap_fail_before_decode() -> Result<(), Status> {
        for (len, nulls) in [(-1, 0), (1, -1), (1, 2)] {
            let bytes = record_bytes(&[FieldNode::new(len, nulls)], &[], 0);
            assert!(visit_record(&bytes, 64 * 1024).is_err());
        }
        // These are exactly the Struct/Null nodes which can materialize a
        // logical Null bitmap during dictionary value validation. No body
        // buffers or decoder is invoked; low test budget avoids huge fixtures.
        let bytes = record_bytes(
            &[
                FieldNode::new(1_000_000, 0),
                FieldNode::new(1_000_000, 1_000_000),
            ],
            &[],
            1,
        );
        let error = visit_record(&bytes, 64 * 1024)
            .err()
            .ok_or_else(|| invalid("implied bitmap must reject"))?;
        assert!(error.message().contains("logical bitmap"));
        let ordinary = record_bytes(&[FieldNode::new(5000, 0)], &[], 0);
        visit_record(&ordinary, 64 * 1024)?; // dictionary/child >4096 is legitimate
        Ok(())
    }
    #[test]
    fn variadic_counts_checked_before_arrow_vector_arithmetic() -> Result<(), Status> {
        for count in [-1, i64::MAX, i64::MAX - 1, 1] {
            let bytes = record_bytes(&[FieldNode::new(1, 0)], &[count], 2);
            assert!(visit_record(&bytes, 64 * 1024).is_err());
        }
        let bytes = record_bytes(&[FieldNode::new(1, 0)], &[1], 3);
        visit_record(&bytes, 64 * 1024)?;
        Ok(())
    }
    #[test]
    fn schema_geometry_and_stray_variadic_entries_reject_before_owned_conversion(
    ) -> Result<(), Status> {
        let mut b = FlatBufferBuilder::new();
        let name = b.create_string("invalid-width");
        let int = ipc::Int::create(
            &mut b,
            &ipc::IntArgs {
                bitWidth: 17,
                is_signed: true,
            },
        );
        let field = ipc::Field::create(
            &mut b,
            &ipc::FieldArgs {
                name: Some(name),
                type_type: Type::Int,
                type_: Some(int.as_union_value()),
                ..Default::default()
            },
        );
        let fields = b.create_vector(&[field]);
        let schema = ipc::Schema::create(
            &mut b,
            &ipc::SchemaArgs {
                fields: Some(fields),
                ..Default::default()
            },
        );
        let msg = Message::create(
            &mut b,
            &MessageArgs {
                header_type: ipc::MessageHeader::Schema,
                header: Some(schema.as_union_value()),
                ..Default::default()
            },
        );
        b.finish(msg, None);
        let error = visit_schema(b.finished_data(), 64 * 1024)
            .err()
            .ok_or_else(|| invalid("invalid integer width must reject"))?;
        assert!(error.message().contains("invalid integer width"));
        let bytes = record_bytes(&[FieldNode::new(1, 0)], &[0], 2);
        let msg = ipc::root_as_message(&bytes).map_err(|_| invalid("fixture metadata invalid"))?;
        let batch = msg
            .header_as_record_batch()
            .ok_or_else(|| invalid("fixture batch missing"))?;
        let mut budget = MetadataBudget::new(64 * 1024); // Schema has zero View fields.
        let error = budget
            .record(&batch, None)
            .err()
            .ok_or_else(|| invalid("stray variadic entry must reject"))?;
        assert!(error.message().contains("schema view fields"));
        Ok(())
    }
    #[test]
    fn real_view_and_large_dictionary_roundtrip_preserve_valid_types() -> Result<(), Status> {
        use arrow::array::{builder::StringDictionaryBuilder, Array, StringViewArray};
        use arrow::datatypes::{Int32Type, Schema};
        use std::sync::Arc;
        let view = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "view",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(StringViewArray::from_iter_values(["alpha", "λ"]))],
        )
        .map_err(|e| invalid(&e.to_string()))?;
        let bytes = super::super::decision_batch::encode_batch_ipc(&view)?;
        let restored =
            super::super::decision_batch::decode_batch_ipc_bounded(&bytes, 16 * 1024 * 1024)?;
        assert_eq!(restored, view);
        let mut builder = StringDictionaryBuilder::<Int32Type>::new();
        for n in 0..5000 {
            builder
                .append(n.to_string())
                .map_err(|e| invalid(&e.to_string()))?;
        }
        let dictionary = builder.finish().slice(0, 1);
        let schema = Schema::new(vec![Field::new(
            "dictionary",
            dictionary.data_type().clone(),
            false,
        )]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(dictionary)])
            .map_err(|e| invalid(&e.to_string()))?;
        let bytes = super::super::decision_batch::encode_batch_ipc(&batch)?;
        let restored =
            super::super::decision_batch::decode_batch_ipc_bounded(&bytes, 16 * 1024 * 1024)?;
        assert_eq!(restored, batch);
        Ok(())
    }
    /// Tiny real IPC streams. The scanner consumes the generated schema and
    /// batch tables, and the positive controls use the canonical Arrow reader.
    fn geometry_stream(
        kind: &str,
        version: ipc::MetadataVersion,
        nodes: &[FieldNode],
        payloads: &[Vec<u8>],
        compressed: bool,
    ) -> Result<Vec<u8>, Status> {
        use arrow::ipc::writer::{write_message, EncodedData, IpcWriteOptions};
        // This handcrafted fixture pads body buffers to8 bytes. Arrow56's
        // default writer requires64-byte body alignment; select the supported
        // 8-byte layout explicitly for BOTH messages and preserve the chosen
        // V4/V5 metadata version. No production reader option is changed.
        let options =
            IpcWriteOptions::try_new(8, false, version).map_err(|e| invalid(&e.to_string()))?;
        let mut out = Vec::new();
        let mut b = FlatBufferBuilder::new();
        let name = b.create_string("value");
        let field = if kind == "int" {
            let ty = ipc::Int::create(
                &mut b,
                &ipc::IntArgs {
                    bitWidth: 32,
                    is_signed: false,
                },
            );
            ipc::Field::create(
                &mut b,
                &ipc::FieldArgs {
                    name: Some(name),
                    nullable: true,
                    type_type: Type::Int,
                    type_: Some(ty.as_union_value()),
                    ..Default::default()
                },
            )
        } else {
            let null = ipc::Null::create(&mut b, &ipc::NullArgs {});
            let child = ipc::Field::create(
                &mut b,
                &ipc::FieldArgs {
                    name: Some(name),
                    nullable: true,
                    type_type: Type::Null,
                    type_: Some(null.as_union_value()),
                    ..Default::default()
                },
            );
            let children = b.create_vector(&[child]);
            let ids = b.create_vector(&[0i32]);
            let ty = ipc::Union::create(
                &mut b,
                &ipc::UnionArgs {
                    mode: if kind == "dense" {
                        ipc::UnionMode::Dense
                    } else {
                        ipc::UnionMode::Sparse
                    },
                    typeIds: Some(ids),
                },
            );
            ipc::Field::create(
                &mut b,
                &ipc::FieldArgs {
                    name: Some(name),
                    nullable: false,
                    type_type: Type::Union,
                    type_: Some(ty.as_union_value()),
                    children: Some(children),
                    ..Default::default()
                },
            )
        };
        let fields = b.create_vector(&[field]);
        let schema = ipc::Schema::create(
            &mut b,
            &ipc::SchemaArgs {
                fields: Some(fields),
                ..Default::default()
            },
        );
        let msg = Message::create(
            &mut b,
            &MessageArgs {
                version,
                header_type: ipc::MessageHeader::Schema,
                header: Some(schema.as_union_value()),
                ..Default::default()
            },
        );
        b.finish(msg, None);
        write_message(
            &mut out,
            EncodedData {
                ipc_message: b.finished_data().to_vec(),
                arrow_data: Vec::new(),
            },
            &options,
        )
        .map_err(|e| invalid(&e.to_string()))?;
        let mut body = Vec::new();
        let mut buffers = Vec::new();
        for payload in payloads {
            let offset = body.len();
            body.extend_from_slice(payload);
            buffers.push(ipc::Buffer::new(offset as i64, payload.len() as i64));
            while body.len() % 8 != 0 {
                body.push(0);
            }
        }
        let mut b = FlatBufferBuilder::new();
        let nodes = b.create_vector(nodes);
        let buffers = b.create_vector(&buffers);
        let compression = if compressed {
            Some(ipc::BodyCompression::create(
                &mut b,
                &ipc::BodyCompressionArgs {
                    codec: ipc::CompressionType::LZ4_FRAME,
                    ..Default::default()
                },
            ))
        } else {
            None
        };
        let batch = ipc::RecordBatch::create(
            &mut b,
            &RecordBatchArgs {
                length: 1,
                nodes: Some(nodes),
                buffers: Some(buffers),
                compression,
                ..Default::default()
            },
        );
        let msg = Message::create(
            &mut b,
            &MessageArgs {
                version,
                header_type: ipc::MessageHeader::RecordBatch,
                header: Some(batch.as_union_value()),
                bodyLength: body.len() as i64,
                ..Default::default()
            },
        );
        b.finish(msg, None);
        write_message(
            &mut out,
            EncodedData {
                ipc_message: b.finished_data().to_vec(),
                arrow_data: body,
            },
            &options,
        )
        .map_err(|e| invalid(&e.to_string()))?;
        out.extend_from_slice(&[255, 255, 255, 255, 0, 0, 0, 0]);
        Ok(out)
    }
    fn scan_fixture(bytes: &[u8]) -> Result<(), Status> {
        super::super::decision_batch::scan_ipc_bounds(bytes, 16 * 1024 * 1024)
    }
    #[test]
    fn fixed_buffer_node_cardinality_and_validity_reject_before_arrow() -> Result<(), Status> {
        let node = [FieldNode::new(1, 0)];
        let payloads = vec![Vec::new(), 7u32.to_le_bytes().to_vec()];
        let valid = geometry_stream("int", ipc::MetadataVersion::V5, &node, &payloads, false)?;
        let decoded =
            super::super::decision_batch::decode_batch_ipc_bounded(&valid, 16 * 1024 * 1024)?;
        let value = decoded
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .ok_or_else(|| invalid("expected UInt32 control"))?;
        assert_eq!(value.value(0), 7);
        for count in [0, 1] {
            let bytes = geometry_stream(
                "int",
                ipc::MetadataVersion::V5,
                &node,
                &payloads[..count],
                false,
            )?;
            let error = scan_fixture(&bytes)
                .err()
                .ok_or_else(|| invalid("missing buffer accepted"))?;
            assert!(error.message().contains("requires more body buffers"));
        }
        for nodes in [Vec::new(), vec![FieldNode::new(1, 0), FieldNode::new(1, 0)]] {
            let bytes = geometry_stream("int", ipc::MetadataVersion::V5, &nodes, &payloads, false)?;
            assert!(scan_fixture(&bytes).is_err());
        }
        let mut extra = payloads.clone();
        extra.push(Vec::new());
        let bytes = geometry_stream("int", ipc::MetadataVersion::V5, &node, &extra, false)?;
        assert!(scan_fixture(&bytes).is_err());
        let bytes = geometry_stream(
            "int",
            ipc::MetadataVersion::V5,
            &[FieldNode::new(1, 1)],
            &payloads,
            false,
        )?;
        let error = scan_fixture(&bytes)
            .err()
            .ok_or_else(|| invalid("missing validity accepted"))?;
        assert!(error.message().contains("shorter than schema"));
        let with_null = geometry_stream(
            "int",
            ipc::MetadataVersion::V5,
            &[FieldNode::new(1, 1)],
            &[vec![0], vec![0; 4]],
            false,
        )?;
        let decoded =
            super::super::decision_batch::decode_batch_ipc_bounded(&with_null, 16 * 1024 * 1024)?;
        assert_eq!(decoded.column(0).null_count(), 1);
        Ok(())
    }
    #[test]
    fn union_fixed_lengths_and_legacy_validity_match_reader() -> Result<(), Status> {
        let nodes = [FieldNode::new(1, 0), FieldNode::new(1, 1)];
        for version in [ipc::MetadataVersion::V4, ipc::MetadataVersion::V5] {
            for kind in ["sparse", "dense"] {
                let mut buffers = if version < ipc::MetadataVersion::V5 {
                    vec![Vec::new()]
                } else {
                    Vec::new()
                };
                let tag = buffers.len();
                buffers.push(vec![0]);
                if kind == "dense" {
                    buffers.push(0i32.to_le_bytes().to_vec());
                }
                let bytes = geometry_stream(kind, version, &nodes, &buffers, false)?;
                let decoded = super::super::decision_batch::decode_batch_ipc_bounded(
                    &bytes,
                    16 * 1024 * 1024,
                )?;
                assert_eq!(decoded.num_rows(), 1);
                let mut short = buffers.clone();
                short[tag].clear();
                let bytes = geometry_stream(kind, version, &nodes, &short, false)?;
                assert!(scan_fixture(&bytes).is_err());
                if kind == "dense" {
                    let mut short = buffers.clone();
                    short[tag + 1].truncate(3);
                    let bytes = geometry_stream(kind, version, &nodes, &short, false)?;
                    assert!(scan_fixture(&bytes).is_err());
                }
                if version < ipc::MetadataVersion::V5 {
                    let bytes = geometry_stream(kind, version, &nodes, &buffers[1..], false)?;
                    assert!(
                        scan_fixture(&bytes).is_err(),
                        "legacy validity slot is consumed"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn layout_uses_actual_decompressed_length_not_encoded_length() -> Result<(), Status> {
        let nodes = [FieldNode::new(1, 0), FieldNode::new(1, 1)];
        // Arrow's zero-prefix rule returns an empty buffer even though the
        // encoded body slice is long enough for a Union type ID.
        let mut encoded = 0i64.to_le_bytes().to_vec();
        encoded.extend_from_slice(&[0; 32]);
        let bytes = geometry_stream("sparse", ipc::MetadataVersion::V5, &nodes, &[encoded], true)?;
        let error = scan_fixture(&bytes)
            .err()
            .ok_or_else(|| invalid("encoded-length substitution accepted"))?;
        assert!(error.message().contains("shorter than schema"));
        // -1 means the bounded one-byte payload is stored verbatim: the same
        // codec path and schema now have one verified decoded type-ID byte.
        let mut encoded = (-1i64).to_le_bytes().to_vec();
        encoded.push(0);
        let bytes = geometry_stream("sparse", ipc::MetadataVersion::V5, &nodes, &[encoded], true)?;
        let decoded =
            super::super::decision_batch::decode_batch_ipc_bounded(&bytes, 16 * 1024 * 1024)?;
        assert_eq!(decoded.num_rows(), 1);
        Ok(())
    }
}
