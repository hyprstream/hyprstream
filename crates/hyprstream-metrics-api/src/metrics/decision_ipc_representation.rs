//! Pre-Arrow representation admission for dictionary evolution and concat.
//!
//! Incoming bytes are not a bound on retained snapshots or concat outputs.
//! This module records immutable, borrowed shape plans at each decode event.
//! No payload is decoded or copied here. The production component ceilings
//! are source-fixed; these are separate ownership/scratch bounds, not RSS.
use arrow::array::{ArrayData, ArrayRef};
use arrow::buffer::Buffer;
use arrow::datatypes::DataType;
use arrow::ipc::{self, Field, Type};
use std::collections::HashMap;
use std::mem::size_of;
use tonic::Status;

const DEPTH: usize = 64;
// Every schema field can contribute a dictionary wrapper AND its values
// array. The schema depth64 contract therefore admits128 representation
// nodes; temporal references cannot bypass this finite typed bound.
const CAPTURE_DEPTH: usize = 2 * DEPTH;
fn invalid(label: &str) -> Status {
    Status::invalid_argument(format!("decision payload IPC representation: {label}"))
}
fn plus(a: usize, b: usize) -> Result<usize, Status> {
    a.checked_add(b).ok_or_else(|| invalid("size overflow"))
}
fn times(a: usize, b: usize) -> Result<usize, Status> {
    a.checked_mul(b).ok_or_else(|| invalid("size overflow"))
}
fn bits(n: usize) -> Result<usize, Status> {
    Ok(plus(n, 7)? / 8)
}
fn slots(n: usize) -> Result<usize, Status> {
    if n == 0 {
        return Ok(0);
    }
    n.max(4)
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("vector capacity overflow"))
}
fn aligned(n: usize) -> Result<usize, Status> {
    times(plus(n, 63)? / 64, 64)
}
// Let P = ceil_pow2(round64(max(initial, final requirement))). Initial
// storage is <=P. Before any growth old<required<=P, and Arrow reserves
// new=max(round64(required),2*old)<=2P. A moving realloc can therefore hold
// old+new<=3P, even when the initial reserve is not a power of two. If new
// overshoots the final requirement it cannot grow again; otherwise the same
// bound applies at every subsequent growth. No-growth storage is only <=P.
// Exact-size Vec builders are also covered. This is allocated byte capacity,
// not an allocator/RSS measurement; all arithmetic, including3P, is checked.
fn storage(n: usize) -> Result<usize, Status> {
    if n == 0 {
        return Ok(0);
    }
    times(
        aligned(n)?
            .checked_next_power_of_two()
            .ok_or_else(|| invalid("buffer capacity overflow"))?,
        3,
    )
}
fn add(counter: &mut usize, n: usize, limit: usize, label: &str) -> Result<(), Status> {
    *counter = plus(*counter, n)?;
    if *counter > limit {
        return Err(invalid(label));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Usage {
    node: usize,
    copies: usize,
    // None means a full ordered range. Some(n) permits arbitrary repeated
    // selections, notably dense-union offsets into variable-width children.
    selected: Option<usize>,
}
impl Usage {
    fn whole(node: usize) -> Self {
        Self {
            node,
            copies: 1,
            selected: None,
        }
    }
}
struct Shape<'a> {
    field: Field<'a>,
    dictionary: bool,
    len: usize,
    data_bytes: usize,
    buffer_count: usize,
    children: Vec<usize>,
    // For a dictionary wrapper this is the values generation captured NOW,
    // never a dictionary ID to resolve later.
    values: Option<usize>,
    logical_null_bytes: usize,
    clone_bytes: usize,
    capture_depth: usize,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Public,
    Mutable,
}

pub(super) struct RepresentationBudget<'a> {
    limit: usize,
    metadata_limit: usize,
    materialized: usize,
    metadata: usize,
    scratch_peak: usize,
    largest_field: usize,
    schema_depth: usize,
    arena: Vec<Shape<'a>>,
    current: HashMap<i64, usize>,
    records: Vec<Vec<usize>>,
}
impl<'a> RepresentationBudget<'a> {
    pub(super) fn new(limit: usize) -> Self {
        Self {
            limit: limit.min(super::decision_batch::MAX_PAYLOAD_BYTES),
            metadata_limit: limit.min(super::decision_batch::MAX_PAYLOAD_BYTES),
            materialized: 0,
            metadata: 0,
            scratch_peak: 0,
            largest_field: 0,
            schema_depth: 0,
            arena: Vec::new(),
            current: HashMap::new(),
            records: Vec::new(),
        }
    }
    #[cfg(test)]
    pub(super) fn reduce_representation_limit(&mut self, limit: usize) {
        self.limit = self.limit.min(limit);
    }
    pub(super) fn schema(&mut self, largest_field: usize, depth: usize) {
        self.largest_field = self.largest_field.max(largest_field);
        self.schema_depth = self.schema_depth.max(depth);
    }
    fn metadata(&mut self, bytes: usize) -> Result<(), Status> {
        add(
            &mut self.metadata,
            bytes,
            self.metadata_limit,
            "owned metadata budget exceeded",
        )
    }
    fn vector<T>(&mut self, count: usize) -> Result<(), Status> {
        self.metadata(times(slots(count)?, size_of::<T>())?)
    }
    fn shape(&mut self, mut shape: Shape<'a>) -> Result<usize, Status> {
        // Cache DAG aggregates once. Rewalking captured shared subtrees at
        // every record would itself allow exponential preflight work.
        shape.logical_null_bytes = storage(bits(shape.len)?)?;
        shape.clone_bytes = plus(size_of::<ArrayData>(), self.largest_field)?;
        shape.clone_bytes = plus(
            shape.clone_bytes,
            times(slots(shape.buffer_count)?, size_of::<Buffer>())?,
        )?;
        shape.capture_depth = 1;
        for child in shape.children.iter().copied().chain(shape.values) {
            let node = &self.arena[child];
            shape.logical_null_bytes = plus(shape.logical_null_bytes, node.logical_null_bytes)?;
            shape.clone_bytes = plus(shape.clone_bytes, node.clone_bytes)?;
            shape.capture_depth = shape.capture_depth.max(plus(node.capture_depth, 1)?);
        }
        if shape.capture_depth > CAPTURE_DEPTH {
            return Err(invalid("captured dictionary nesting exceeded"));
        }
        self.scratch(shape.logical_null_bytes)?;
        self.metadata(times(4, size_of::<Shape<'a>>())?)?;
        let id = self.arena.len();
        self.arena.push(shape);
        Ok(id)
    }
    fn buffer(&mut self, initial: usize, output: usize) -> Result<(), Status> {
        add(
            &mut self.materialized,
            storage(initial.max(output))?,
            self.limit,
            "materialized buffer budget exceeded",
        )
    }
    fn scratch(&mut self, bytes: usize) -> Result<(), Status> {
        self.scratch_peak = self.scratch_peak.max(bytes);
        if self.scratch_peak > self.limit {
            return Err(invalid("scratch budget exceeded"));
        }
        Ok(())
    }
    fn output_objects(&mut self, inputs: usize, buffers: usize) -> Result<(), Status> {
        // Incoming-node accounting in MetadataBudget does not cover synthetic
        // concat nodes. At every schema ancestor Arrow can clone ArrayData
        // descriptors, owned datatypes and buffer vectors. Three live stages
        // are extraction/MutableArrayData, freeze/ArrayData and concrete array.
        let stages = plus(self.schema_depth, 3)?;
        let per = plus(
            times(4, plus(size_of::<ArrayData>(), size_of::<ArrayRef>())?)?,
            plus(size_of::<DataType>(), self.largest_field)?,
        )?;
        self.metadata(times(stages, per)?)?;
        self.vector::<Buffer>(times(stages, buffers)?)?;
        // Source refs, ArrayData clones, per-source boxed extend-values and
        // extend-null closures, lengths and offsets. Six vectors correspond
        // to those six independent source-indexed collections in transform.
        let slot = plus(
            size_of::<ArrayData>(),
            plus(4 * size_of::<usize>(), self.largest_field)?,
        )?;
        self.metadata(times(times(6, slots(inputs)?)?, slot)?)?;
        Ok(())
    }
    fn rows(&self, usage: Usage) -> Result<usize, Status> {
        times(self.arena[usage.node].len_for(usage), usage.copies)
    }
    pub(super) fn record(
        &mut self,
        fields: impl Iterator<Item = Field<'a>>,
        batch: ipc::RecordBatch<'_>,
        lengths: &[usize],
        version: ipc::MetadataVersion,
    ) -> Result<(), Status> {
        let mut cursor = Cursor {
            batch,
            lengths,
            version,
            node: 0,
            buffer: 0,
            variadic: 0,
        };
        let count = fields
            .size_hint()
            .1
            .ok_or_else(|| invalid("field count missing"))?;
        self.vector::<usize>(count)?;
        let mut roots = Vec::with_capacity(count);
        for field in fields {
            roots.push(self.decode(field, true, &mut cursor, 0)?);
        }
        self.metadata(4 * size_of::<Vec<usize>>())?;
        self.records.push(roots);
        Ok(())
    }
    pub(super) fn dictionary(
        &mut self,
        field: Field<'a>,
        batch: ipc::RecordBatch<'_>,
        lengths: &[usize],
        version: ipc::MetadataVersion,
        id: i64,
        delta: bool,
    ) -> Result<(), Status> {
        let mut cursor = Cursor {
            batch,
            lengths,
            version,
            node: 0,
            buffer: 0,
            variadic: 0,
        };
        let incoming = self.decode(field, false, &mut cursor, 0)?;
        let result = if delta {
            let prior = *self
                .current
                .get(&id)
                .ok_or_else(|| invalid("delta without dictionary"))?;
            self.concat(
                &[Usage::whole(prior), Usage::whole(incoming)],
                Mode::Public,
                None,
                0,
                0,
            )?
        } else {
            incoming
        };
        if !self.current.contains_key(&id) {
            self.metadata(4 * (size_of::<(i64, usize)>() + 1) + size_of::<u128>())?;
        }
        self.current.insert(id, result);
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Result<(), Status> {
        if self.records.len() < 2 {
            return Ok(());
        }
        let columns = self.records.first().map_or(0, Vec::len);
        for col in 0..columns {
            self.vector::<Usage>(self.records.len())?;
            let inputs = self
                .records
                .iter()
                .map(|roots| {
                    roots
                        .get(col)
                        .copied()
                        .map(Usage::whole)
                        .ok_or_else(|| invalid("record schema changed"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.concat(&inputs, Mode::Public, None, 0, 0)?;
        }
        Ok(())
    }
    fn decode(
        &mut self,
        field: Field<'a>,
        wrapper: bool,
        cursor: &mut Cursor<'_, '_>,
        depth: usize,
    ) -> Result<usize, Status> {
        if depth >= DEPTH {
            return Err(invalid("representation nesting exceeded"));
        }
        let len = cursor.next_node()?;
        let dictionary = wrapper && field.dictionary().is_some();
        let mut buffers = 0usize;
        let mut data_bytes = 0usize;
        let mut children = Vec::new();
        let mut values = None;
        if dictionary {
            cursor.skip()?; // validity
            data_bytes = cursor.take()?;
            buffers = 1;
            let id = field
                .dictionary()
                .ok_or_else(|| invalid("dictionary missing"))?
                .id();
            values = Some(
                *self
                    .current
                    .get(&id)
                    .ok_or_else(|| invalid("dictionary reference before definition"))?,
            );
        } else {
            macro_rules! consume {
                ($n:expr) => {{
                    let count = $n;
                    data_bytes = plus(data_bytes, cursor.data(count)?)?;
                    buffers = plus(buffers, count)?;
                }};
            }
            match field.type_type() {
                Type::Null | Type::RunEndEncoded => {}
                Type::Union => {
                    if cursor.version < ipc::MetadataVersion::V5 {
                        cursor.skip()?;
                    }
                    let dense = field
                        .type_as_union()
                        .ok_or_else(|| invalid("union missing"))?
                        .mode()
                        == ipc::UnionMode::Dense;
                    consume!(if dense { 2 } else { 1 });
                }
                Type::Utf8View | Type::BinaryView => {
                    let n = cursor.next_variadic()?;
                    cursor.skip()?;
                    consume!(plus(n, 1)?);
                }
                Type::Utf8 | Type::Binary | Type::LargeUtf8 | Type::LargeBinary => {
                    cursor.skip()?;
                    consume!(2);
                }
                Type::Struct_ | Type::FixedSizeList => {
                    cursor.skip()?;
                }
                _ => {
                    cursor.skip()?;
                    consume!(1);
                }
            }
            if matches!(
                field.type_type(),
                Type::List
                    | Type::LargeList
                    | Type::Map
                    | Type::FixedSizeList
                    | Type::Struct_
                    | Type::Union
                    | Type::RunEndEncoded
            ) {
                if let Some(fields) = field.children() {
                    self.vector::<usize>(fields.len())?;
                    children = Vec::with_capacity(fields.len());
                    for child in fields {
                        children.push(self.decode(child, true, cursor, depth + 1)?);
                    }
                }
            }
        }
        // Incoming byte capacity was already bounded by the codec verifier.
        // Account potential logical-null scratch through captured dictionary
        // descendants too, even where no physical validity buffer exists.
        let root = self.shape(Shape {
            field,
            dictionary,
            len,
            data_bytes,
            buffer_count: buffers,
            children,
            values,
            logical_null_bytes: 0,
            clone_bytes: 0,
            capture_depth: 0,
        })?;
        let scratch = self.logical_nulls(root, 0)?;
        self.scratch(scratch)?;
        Ok(root)
    }
    fn logical_nulls(&self, root: usize, _depth: usize) -> Result<usize, Status> {
        Ok(self.arena[root].logical_null_bytes)
    }
    fn child_inputs(
        &mut self,
        inputs: &[Usage],
        child: usize,
        kind: ChildSelection,
    ) -> Result<Vec<Usage>, Status> {
        self.vector::<Usage>(inputs.len())?;
        inputs
            .iter()
            .map(|u| {
                let s = &self.arena[u.node];
                let node = *s
                    .children
                    .get(child)
                    .ok_or_else(|| invalid("child shape missing"))?;
                let (copies, selected) = match kind {
                    ChildSelection::Same => (u.copies, u.selected),
                    ChildSelection::List => match u.selected {
                        None => (u.copies, None),
                        Some(n) => (
                            times(u.copies, n.max(1))?,
                            if n == 0 { Some(0) } else { None },
                        ),
                    },
                    ChildSelection::Fixed(n) => {
                        (u.copies, u.selected.map(|rows| times(rows, n)).transpose()?)
                    }
                    ChildSelection::Dense => (u.copies, Some(s.len_for(*u))),
                };
                Ok(Usage {
                    node,
                    copies,
                    selected,
                })
            })
            .collect()
    }
    fn concat(
        &mut self,
        inputs: &[Usage],
        mode: Mode,
        inherited: Option<usize>,
        ancestor_scratch: usize,
        depth: usize,
    ) -> Result<usize, Status> {
        if depth >= CAPTURE_DEPTH {
            return Err(invalid("concat nesting exceeded"));
        }
        let first = *inputs
            .first()
            .ok_or_else(|| invalid("concat inputs missing"))?;
        let input_count = inputs.iter().try_fold(0usize, |n, u| plus(n, u.copies))?;
        // Public concat's single-input slice shares all payloads. Inherited
        // MutableArrayData construction must still run for one source array.
        if mode == Mode::Public && input_count == 1 && first.selected.is_none() {
            return Ok(first.node);
        }
        let len = inputs
            .iter()
            .try_fold(0usize, |n, u| plus(n, self.rows(*u)?))?;
        let initial = inherited.unwrap_or(len);
        let field = self.arena[first.node].field;
        let dictionary = self.arena[first.node].dictionary;
        let kind = field.type_type();
        let data_bytes = inputs.iter().try_fold(0usize, |n, u| {
            let s = &self.arena[u.node];
            let repeats = times(u.copies, u.selected.unwrap_or(1))?;
            plus(n, times(s.data_bytes, repeats)?)
        })?;
        let buffer_count = inputs.iter().try_fold(0usize, |n, u| {
            plus(n, times(self.arena[u.node].buffer_count, u.copies)?)
        })?;
        self.output_objects(input_count, buffer_count)?;
        let cloned = inputs.iter().try_fold(0usize, |n, u| {
            plus(n, times(u.copies, self.arena[u.node].clone_bytes)?)
        })?;
        // ArrayData extraction plus transform/freeze intermediates clone the
        // whole captured descriptor tree, including shared dictionary values.
        // Payload buffers remain shared; these are descriptor bytes only.
        self.metadata(times(6, cloned)?)?;
        let bitmap = storage(bits(initial.max(len))?)?;
        let local_scratch = plus(ancestor_scratch, bitmap)?;
        self.scratch(local_scratch)?;
        // Conservatively reserve output validity even for all-valid inputs.
        self.buffer(bits(initial)?, bits(len)?)?;
        let mut children = Vec::new();
        let mut values = None;
        let output_data;
        let mut output_buffers = buffer_count;
        if dictionary {
            let width = dictionary_width(field)?;
            self.buffer(times(initial, width)?, times(len, width)?)?;
            output_data = times(len, width)?;
            output_buffers = 1;
            self.vector::<Usage>(inputs.len())?;
            let value_inputs = inputs
                .iter()
                .map(|u| {
                    Ok(Usage {
                        node: self.arena[u.node]
                            .values
                            .ok_or_else(|| invalid("dictionary values missing"))?,
                        copies: u.copies,
                        selected: None,
                    })
                })
                .collect::<Result<Vec<_>, Status>>()?;
            let same = value_inputs.iter().all(|u| u.node == value_inputs[0].node);
            let merge_possible = mode == Mode::Public
                && !same
                && merge_supported(self.arena[value_inputs[0].node].field);
            if !merge_possible {
                // Pinned fallback uses an infallible expect after checking
                // that each dictionary's exclusive end fits the key type.
                // Reject that unsupported concat geometry before Arrow can
                // panic. Public merge can deduplicate beyond this bound and
                // must not receive this fallback-only restriction.
                let end = if same {
                    self.arena[value_inputs[0].node].len
                } else {
                    value_inputs.iter().try_fold(0usize, |n, u| {
                        plus(n, times(self.arena[u.node].len, u.copies)?)
                    })?
                };
                let index = field
                    .dictionary()
                    .and_then(|d| d.indexType())
                    .ok_or_else(|| invalid("dictionary index missing"))?;
                let bits = u32::try_from(index.bitWidth())
                    .map_err(|_| invalid("dictionary index width"))?;
                let max = if index.is_signed() {
                    (1u128 << (bits - 1)) - 1
                } else {
                    (1u128 << bits) - 1
                };
                if end as u128 > max {
                    return Err(invalid("fallback dictionary key range exceeded"));
                }
            }
            if same {
                values = Some(value_inputs[0].node);
            } else {
                // MutableArrayData concatenates EVERY occurrence, including
                // repeated references to a generation when another differs.
                values = Some(self.concat(
                    &value_inputs,
                    Mode::Mutable,
                    None,
                    local_scratch,
                    depth + 1,
                )?);
                if mode == Mode::Public && merge_supported(self.arena[value_inputs[0].node].field) {
                    self.merge_scratch(&value_inputs, len, width, local_scratch)?;
                    // Primitive/byte merge outputs cannot exceed the complete
                    // fallback values envelope; key output is already paid.
                }
            }
        } else {
            match kind {
                Type::Null => {
                    output_data = 0;
                    output_buffers = 0;
                }
                Type::Bool => {
                    self.buffer(bits(initial)?, bits(len)?)?;
                    output_data = bits(len)?;
                    output_buffers = 1;
                }
                Type::Utf8 | Type::Binary | Type::LargeUtf8 | Type::LargeBinary => {
                    let width = if matches!(kind, Type::LargeUtf8 | Type::LargeBinary) {
                        8
                    } else {
                        4
                    };
                    self.buffer(
                        times(plus(initial, 1)?, width)?,
                        times(plus(len, 1)?, width)?,
                    )?;
                    // In Array capacities Arrow initially reserves one byte
                    // per row; public specialized concat reserves exact data.
                    self.buffer(
                        if mode == Mode::Mutable {
                            initial
                        } else {
                            data_bytes
                        },
                        data_bytes,
                    )?;
                    output_data = plus(data_bytes, times(plus(len, 1)?, width)?)?;
                    output_buffers = 2;
                }
                Type::Utf8View | Type::BinaryView => {
                    self.buffer(times(initial, 16)?, times(len, 16)?)?;
                    // backing data are retained, descriptor/remapping tables
                    // are new. Charge every buffer reference occurrence.
                    self.vector::<Buffer>(buffer_count)?;
                    self.vector::<(usize, usize)>(times(2, buffer_count)?)?;
                    output_data = plus(data_bytes, times(len, 16)?)?;
                }
                Type::List | Type::LargeList | Type::Map => {
                    let width = if kind == Type::LargeList { 8 } else { 4 };
                    self.buffer(
                        times(plus(initial, 1)?, width)?,
                        times(plus(len, 1)?, width)?,
                    )?;
                    let uses = self.child_inputs(inputs, 0, ChildSelection::List)?;
                    let public = mode == Mode::Public && kind != Type::Map;
                    let child = self.concat(
                        &uses,
                        if public { Mode::Public } else { Mode::Mutable },
                        if public { None } else { Some(initial) },
                        local_scratch,
                        depth + 1,
                    )?;
                    self.vector::<usize>(1)?;
                    children.push(child);
                    output_data = times(plus(len, 1)?, width)?;
                    output_buffers = 1;
                }
                Type::FixedSizeList => {
                    let count = usize::try_from(
                        field
                            .type_as_fixed_size_list()
                            .ok_or_else(|| invalid("fixed list missing"))?
                            .listSize(),
                    )
                    .map_err(|_| invalid("fixed list width"))?;
                    let uses = self.child_inputs(inputs, 0, ChildSelection::Fixed(count))?;
                    // The fallback's nested get_capacity special cases may
                    // choose a smaller/exact byte capacity. This Array bound
                    // reserves the propagated parent * list_size, plus the
                    // child's complete output, covering either alternative.
                    let child = self.concat(
                        &uses,
                        Mode::Mutable,
                        Some(times(initial, count)?),
                        local_scratch,
                        depth + 1,
                    )?;
                    self.vector::<usize>(1)?;
                    children.push(child);
                    output_data = 0;
                    output_buffers = 0;
                }
                Type::Struct_ | Type::Union | Type::RunEndEncoded => {
                    let count = self.arena[first.node].children.len();
                    self.vector::<usize>(count)?;
                    children = Vec::with_capacity(count);
                    let dense = kind == Type::Union
                        && field
                            .type_as_union()
                            .ok_or_else(|| invalid("union missing"))?
                            .mode()
                            == ipc::UnionMode::Dense;
                    if kind == Type::Union {
                        self.buffer(initial, len)?;
                        if dense {
                            self.buffer(times(initial, 4)?, times(len, 4)?)?;
                        }
                    }
                    if kind == Type::RunEndEncoded {
                        // Fallback run transforms create temporary run-end byte
                        // Vecs; repeated selections may create one per run.
                        self.scratch(plus(local_scratch, storage(times(len, 8)?)?)?)?;
                    }
                    for child in 0..count {
                        let selection = if dense {
                            ChildSelection::Dense
                        } else if kind == Type::RunEndEncoded {
                            ChildSelection::List
                        } else {
                            ChildSelection::Same
                        };
                        let uses = self.child_inputs(inputs, child, selection)?;
                        let public = mode == Mode::Public && kind != Type::Union;
                        let next = self.concat(
                            &uses,
                            if public { Mode::Public } else { Mode::Mutable },
                            if public { None } else { Some(initial) },
                            local_scratch,
                            depth + 1,
                        )?;
                        children.push(next);
                    }
                    output_data = if kind == Type::Union {
                        times(len, if dense { 5 } else { 1 })?
                    } else {
                        0
                    };
                    output_buffers = if kind == Type::Union {
                        if dense {
                            2
                        } else {
                            1
                        }
                    } else {
                        0
                    };
                }
                _ => {
                    let width = fixed_width(field)?;
                    self.buffer(times(initial, width)?, times(len, width)?)?;
                    output_data = times(len, width)?;
                    output_buffers = 1;
                }
            }
        }
        let result = self.shape(Shape {
            field,
            dictionary,
            len,
            data_bytes: output_data,
            buffer_count: output_buffers,
            children,
            values,
            logical_null_bytes: 0,
            clone_bytes: 0,
            capture_depth: 0,
        })?;
        self.scratch(plus(local_scratch, self.logical_nulls(result, 0)?)?)?;
        Ok(result)
    }
    fn merge_scratch(
        &mut self,
        inputs: &[Usage],
        rows: usize,
        key_width: usize,
        ancestors: usize,
    ) -> Result<(), Status> {
        let entries = inputs.iter().try_fold(0usize, |n, u| {
            plus(n, times(self.arena[u.node].len, u.copies)?)
        })?;
        let arrays = inputs.iter().try_fold(0usize, |n, u| plus(n, u.copies))?;
        // Pinned Interner uses (u64::MAX >> leading_zeros(capacity+128))+1:
        // the STRICT next power of two, even if capacity+128 is a power of2.
        // Actual selected capacity is <= total full values cardinality.
        let buckets = interner_buckets(entries)?;
        let bucket = size_of::<Option<(Option<&[u8]>, usize)>>();
        let masked = size_of::<(usize, Option<&[u8]>)>();
        let mut bytes = ancestors;
        bytes = plus(bytes, storage(times(buckets, bucket)?)?)?;
        bytes = plus(bytes, storage(times(entries, masked)?)?)?;
        bytes = plus(
            bytes,
            storage(times(entries, size_of::<(usize, usize)>())?)?,
        )?;
        bytes = plus(bytes, storage(times(entries, key_width)?)?)?;
        bytes = plus(bytes, storage(bits(entries)?)?)?;
        bytes = plus(bytes, storage(bits(rows)?)?)?;
        bytes = plus(
            bytes,
            storage(times(arrays, 6 * (size_of::<Vec<usize>>() + 64))?)?,
        )?;
        self.scratch(bytes)
    }
}
impl Shape<'_> {
    fn len_for(&self, usage: Usage) -> usize {
        usage.selected.unwrap_or(self.len)
    }
}
#[derive(Clone, Copy)]
enum ChildSelection {
    Same,
    List,
    Fixed(usize),
    Dense,
}

struct Cursor<'a, 'b> {
    batch: ipc::RecordBatch<'a>,
    lengths: &'b [usize],
    version: ipc::MetadataVersion,
    node: usize,
    buffer: usize,
    variadic: usize,
}
impl Cursor<'_, '_> {
    fn next_node(&mut self) -> Result<usize, Status> {
        let nodes = self.batch.nodes().ok_or_else(|| invalid("nodes missing"))?;
        if self.node >= nodes.len() {
            return Err(invalid("node missing"));
        }
        let len =
            usize::try_from(nodes.get(self.node).length()).map_err(|_| invalid("node length"))?;
        self.node = plus(self.node, 1)?;
        Ok(len)
    }
    fn take(&mut self) -> Result<usize, Status> {
        let n = *self
            .lengths
            .get(self.buffer)
            .ok_or_else(|| invalid("buffer missing"))?;
        self.buffer = plus(self.buffer, 1)?;
        Ok(n)
    }
    fn skip(&mut self) -> Result<(), Status> {
        self.take().map(|_| ())
    }
    fn data(&mut self, n: usize) -> Result<usize, Status> {
        let mut bytes = 0;
        for _ in 0..n {
            bytes = plus(bytes, self.take()?)?;
        }
        Ok(bytes)
    }
    fn next_variadic(&mut self) -> Result<usize, Status> {
        let v = self
            .batch
            .variadicBufferCounts()
            .ok_or_else(|| invalid("variadic missing"))?;
        if self.variadic >= v.len() {
            return Err(invalid("variadic missing"));
        }
        let n = usize::try_from(v.get(self.variadic)).map_err(|_| invalid("variadic count"))?;
        self.variadic = plus(self.variadic, 1)?;
        Ok(n)
    }
}
fn interner_buckets(entries: usize) -> Result<usize, Status> {
    plus(entries, 129)?
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("interner size overflow"))
}
fn dictionary_width(field: Field<'_>) -> Result<usize, Status> {
    let d = field
        .dictionary()
        .ok_or_else(|| invalid("dictionary metadata missing"))?;
    let i = d
        .indexType()
        .ok_or_else(|| invalid("dictionary index missing"))?;
    usize::try_from(i.bitWidth() / 8).map_err(|_| invalid("dictionary index width"))
}
fn merge_supported(field: Field<'_>) -> bool {
    matches!(
        field.type_type(),
        Type::Int
            | Type::FloatingPoint
            | Type::Decimal
            | Type::Date
            | Type::Time
            | Type::Timestamp
            | Type::Duration
            | Type::Interval
            | Type::Utf8
            | Type::LargeUtf8
            | Type::Binary
            | Type::LargeBinary
    )
}
fn fixed_width(field: Field<'_>) -> Result<usize, Status> {
    let width = match field.type_type() {
        Type::Int => field.type_as_int().map(|v| v.bitWidth() / 8),
        Type::FloatingPoint => field.type_as_floating_point().map(|v| match v.precision() {
            ipc::Precision::HALF => 2,
            ipc::Precision::SINGLE => 4,
            _ => 8,
        }),
        Type::Decimal => field.type_as_decimal().map(|v| v.bitWidth() / 8),
        Type::FixedSizeBinary => field.type_as_fixed_size_binary().map(|v| v.byteWidth()),
        Type::Date => field
            .type_as_date()
            .map(|v| if v.unit() == ipc::DateUnit::DAY { 4 } else { 8 }),
        Type::Time => field.type_as_time().map(|v| v.bitWidth() / 8),
        Type::Timestamp | Type::Duration => Some(8),
        Type::Interval => field.type_as_interval().map(|v| match v.unit() {
            ipc::IntervalUnit::YEAR_MONTH => 4,
            ipc::IntervalUnit::DAY_TIME => 8,
            _ => 16,
        }),
        _ => None,
    }
    .ok_or_else(|| invalid("fixed width missing"))?;
    usize::try_from(width).map_err(|_| invalid("negative fixed width"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        DictionaryArray, FixedSizeBinaryArray, Int32Array, RecordBatch, UInt8Array,
    };
    use arrow::datatypes::{DataType, Field as ArrowField, Int32Type, Schema};
    use arrow::ipc::writer::{DictionaryHandling, IpcWriteOptions, StreamWriter};
    use std::sync::Arc;

    fn dict_batch(values: ArrayRef) -> RecordBatch {
        let datatype = DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(values.data_type().clone()),
        );
        let schema = Arc::new(Schema::new(vec![ArrowField::new("extra", datatype, false)]));
        let array = DictionaryArray::<Int32Type>::new(Int32Array::from(vec![0]), values);
        RecordBatch::try_new(schema, vec![Arc::new(array)])
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"))
    }
    fn delta_stream(base: usize, epochs: usize) -> Vec<u8> {
        let first = dict_batch(Arc::new(UInt8Array::from(vec![0u8; base])));
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new_with_options(
            &mut bytes,
            &first.schema(),
            IpcWriteOptions::default().with_dictionary_handling(DictionaryHandling::Delta),
        )
        .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        writer
            .write(&first)
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        for n in 1..epochs {
            writer
                .write(&dict_batch(Arc::new(UInt8Array::from(vec![0u8; base + n]))))
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        }
        writer
            .finish()
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        drop(writer);
        bytes
    }
    fn binary_stream(width: usize, records: usize, replace: bool) -> Vec<u8> {
        binary_stream_rows(width, records, replace, false)
    }
    fn binary_stream_rows(width: usize, records: usize, replace: bool, empty: bool) -> Vec<u8> {
        let a = vec![1u8; width];
        let b = vec![2u8; width];
        let a = Arc::new(
            FixedSizeBinaryArray::try_from_iter([a.as_slice()].into_iter())
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
        ) as ArrayRef;
        let b = Arc::new(
            FixedSizeBinaryArray::try_from_iter([b.as_slice()].into_iter())
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
        ) as ArrayRef;
        let first = dict_batch(a.clone());
        let first = if empty { first.slice(0, 0) } else { first };
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &first.schema())
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        writer
            .write(&first)
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        for _ in 1..records {
            let batch = dict_batch(if replace { b.clone() } else { a.clone() });
            let batch = if empty { batch.slice(0, 0) } else { batch };
            writer
                .write(&batch)
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        }
        writer
            .finish()
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        drop(writer);
        bytes
    }
    fn scan(bytes: &[u8], limit: usize) -> Result<(), Status> {
        super::super::decision_batch::walk_ipc_bounds_with_budget(
            bytes,
            super::super::decision_batch::MAX_PAYLOAD_BYTES,
            |budget| budget.reduce_representation_limit(limit),
        )
        .map(|_| ())
    }
    fn positive(bytes: &[u8], limit: usize) {
        scan(bytes, limit).unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        let actual = super::super::decision_batch::decode_batch_ipc_bounded(
            bytes,
            super::super::decision_batch::MAX_PAYLOAD_BYTES,
        )
        .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        let canonical = super::super::decision_batch::decode_batch_ipc(bytes)
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        assert_eq!(actual, canonical);
    }
    #[test]
    fn representation_rejects_delta_snapshot_materialization_before_decode() {
        let bytes = delta_stream(8192, 16);
        assert!(bytes.len() < 32 * 1024);
        let err = scan(&bytes, 64 * 1024)
            .err()
            .unwrap_or_else(|| panic!("predecode guard unexpectedly accepted fixture"));
        assert!(
            err.message().contains("IPC representation: materialized"),
            "{err}"
        );
        positive(&delta_stream(1024, 2), 1024 * 1024);
    }
    #[test]
    fn representation_mixed_epochs_charge_repeated_values_but_static_epoch_shares() {
        let bytes = binary_stream(2048, 32, true);
        assert!(bytes.len() < 32 * 1024);
        let err = scan(&bytes, 32 * 1024)
            .err()
            .unwrap_or_else(|| panic!("predecode guard unexpectedly accepted fixture"));
        assert!(
            err.message().contains("IPC representation: materialized"),
            "{err}"
        );
        positive(&binary_stream(2048, 2, true), 32 * 1024);
        positive(&binary_stream(2048, 32, false), 32 * 1024);
        let bytes = binary_stream_rows(2048, 32, true, true);
        let error = scan(&bytes, 32 * 1024)
            .err()
            .unwrap_or_else(|| panic!("empty records still carry mixed dictionary generations"));
        assert!(
            error.message().contains("IPC representation: materialized"),
            "{error}"
        );
        positive(&binary_stream_rows(2048, 32, false, true), 32 * 1024);
    }
    fn plain_stream(batch: &RecordBatch, records: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut writer = StreamWriter::try_new(&mut bytes, &batch.schema())
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        for _ in 0..records {
            writer
                .write(batch)
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        }
        writer
            .finish()
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        drop(writer);
        bytes
    }
    fn plain_batch(array: ArrayRef) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![ArrowField::new(
            "extra",
            array.data_type().clone(),
            true,
        )]));
        RecordBatch::try_new(schema, vec![array])
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"))
    }
    #[test]
    fn representation_dense_union_repeated_selection_and_empty_child_reserve() {
        use arrow::array::{Array, ListArray, StringArray, UnionArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::UnionFields;
        let children = UnionFields::new(
            vec![0],
            vec![ArrowField::new("text", DataType::Utf8, false)],
        );
        let text = "x".repeat(2048);
        let union = UnionArray::try_new(
            children,
            vec![0i8; 16].into(),
            Some(vec![0i32; 16].into()),
            vec![Arc::new(StringArray::from(vec![text.as_str()]))],
        )
        .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        let batch = plain_batch(Arc::new(union));
        let bytes = plain_stream(&batch, 2);
        assert!(bytes.len() < 16 * 1024);
        let err = scan(&bytes, 32 * 1024)
            .err()
            .unwrap_or_else(|| panic!("predecode guard unexpectedly accepted fixture"));
        assert!(
            err.message().contains("IPC representation: materialized"),
            "{err}"
        );
        positive(&bytes, 1024 * 1024);

        // FixedSizeBinary has NO incoming data, yet MutableArrayData's Map
        // fallback passes its parent reserve through the entry Struct into
        // this wide zero-length child. Two empty records retain that path.
        let values = Arc::new(FixedSizeBinaryArray::new(
            2048,
            Buffer::from(Vec::<u8>::new()),
            None,
        )) as ArrayRef;
        let list = ListArray::try_new(
            Arc::new(ArrowField::new("item", values.data_type().clone(), true)),
            OffsetBuffer::new(vec![0i32; 17].into()),
            values,
            None,
        )
        .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        let list = Arc::new(list) as ArrayRef;
        // List itself uses optimized public concat; wrap in FixedSizeList to
        // force MutableArrayData construction of its List/FSB descendants.
        let field = Arc::new(ArrowField::new("list", list.data_type().clone(), true));
        let array = arrow::array::FixedSizeListArray::try_new(field, 1, list, None)
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        let bytes = plain_stream(&plain_batch(Arc::new(array)), 2);
        let err = scan(&bytes, 32 * 1024)
            .err()
            .unwrap_or_else(|| panic!("predecode guard unexpectedly accepted fixture"));
        assert!(
            err.message().contains("IPC representation: materialized"),
            "{err}"
        );
        positive(&bytes, 1024 * 1024);
    }
    #[test]
    fn representation_views_nulls_alias_columns_zero_rows_and_static_many_records() {
        use arrow::array::{Array, NullArray, StringViewArray};
        positive(
            &plain_stream(
                &plain_batch(Arc::new(StringViewArray::from(vec![
                    "a value longer than twelve bytes",
                    "short",
                ]))),
                3,
            ),
            1024 * 1024,
        );
        positive(
            &plain_stream(&plain_batch(Arc::new(NullArray::new(100))), 3),
            1024 * 1024,
        );
        let values = Arc::new(
            FixedSizeBinaryArray::try_from_iter([&[1u8; 128][..]].into_iter())
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
        ) as ArrayRef;
        let dict = Arc::new(DictionaryArray::<Int32Type>::new(
            Int32Array::from(vec![0]),
            values,
        ));
        let field = Arc::new(ArrowField::new("same", dict.data_type().clone(), false));
        let schema = Arc::new(Schema::new(vec![field.clone(), field]));
        let batch = RecordBatch::try_new(schema, vec![dict.clone(), dict])
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        positive(&plain_stream(&batch, 3), 1024 * 1024);
        positive(&plain_stream(&batch.slice(0, 0), 3), 1024 * 1024);
        positive(
            &binary_stream(2048, 1000, false),
            super::super::decision_batch::MAX_PAYLOAD_BYTES,
        );
    }
    fn encoded_messages(
        bytes: &[u8],
    ) -> Vec<(
        ipc::MessageHeader,
        Option<i64>,
        arrow::ipc::writer::EncodedData,
    )> {
        super::super::decision_batch::walk_ipc_bounds_with_budget(
            bytes,
            super::super::decision_batch::MAX_PAYLOAD_BYTES,
            |_| {},
        )
        .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"))
        .into_iter()
        .map(|bounds| {
            let metadata = &bytes[bounds.metadata.0..bounds.metadata.1];
            let message = ipc::root_as_message(metadata)
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
            let id = message.header_as_dictionary_batch().map(|d| d.id());
            (
                message.header_type(),
                id,
                arrow::ipc::writer::EncodedData {
                    ipc_message: metadata.to_vec(),
                    arrow_data: bytes[bounds.body.0..bounds.body.1].to_vec(),
                },
            )
        })
        .collect()
    }
    #[test]
    fn representation_nested_dictionary_captures_old_generation() {
        use arrow::array::{Array, StructArray};
        fn nested(value: u8) -> RecordBatch {
            let bytes = vec![value; 1024];
            let value = Arc::new(
                FixedSizeBinaryArray::try_from_iter([bytes.as_slice()].into_iter())
                    .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
            ) as ArrayRef;
            let inner = Arc::new(DictionaryArray::<Int32Type>::new(
                Int32Array::from(vec![0]),
                value,
            )) as ArrayRef;
            let fields = vec![Arc::new(ArrowField::new(
                "inner",
                inner.data_type().clone(),
                false,
            ))]
            .into();
            let outer = Arc::new(
                StructArray::try_new(fields, vec![inner], None)
                    .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
            ) as ArrayRef;
            dict_batch(outer)
        }
        let a = plain_stream(&nested(1), 1);
        let b = plain_stream(&nested(2), 1);
        let ma = encoded_messages(&a);
        let mb = encoded_messages(&b);
        let ids = ma.iter().filter_map(|(_, id, _)| *id).collect::<Vec<_>>();
        assert_eq!(ids.len(), 2);
        let (inner, outer) = (ids[0], ids[1]);
        let mut bytes = Vec::new();
        let options = IpcWriteOptions::default();
        for (_, _, m) in encoded_messages(&a) {
            arrow::ipc::writer::write_message(&mut bytes, m, &options)
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        }
        // Change the inner dictionary without replacing the cached outer.
        for (_, id, m) in encoded_messages(&b) {
            if id == Some(inner) {
                arrow::ipc::writer::write_message(&mut bytes, m, &options)
                    .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
            }
        }
        for (kind, _, m) in encoded_messages(&a) {
            if kind == ipc::MessageHeader::RecordBatch {
                arrow::ipc::writer::write_message(&mut bytes, m, &options)
                    .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
            }
        }
        for (kind, id, m) in mb {
            if id == Some(outer) || kind == ipc::MessageHeader::RecordBatch {
                arrow::ipc::writer::write_message(&mut bytes, m, &options)
                    .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
            }
        }
        bytes.extend_from_slice(&[255, 255, 255, 255, 0, 0, 0, 0]);
        positive(&bytes, 1024 * 1024);
        let batch = super::super::decision_batch::decode_batch_ipc(&bytes)
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        let outer = batch
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap_or_else(|| panic!("canonical fixture array type mismatch"));
        let fields = outer
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap_or_else(|| panic!("canonical fixture array type mismatch"));
        let inner = fields
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap_or_else(|| panic!("canonical fixture array type mismatch"));
        let values = inner
            .values()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap_or_else(|| panic!("canonical fixture array type mismatch"));
        let observed = (0..outer.len())
            .map(|row| {
                let outer_key = outer.keys().value(row) as usize;
                let inner_key = inner.keys().value(outer_key) as usize;
                values.value(inner_key)[0]
            })
            .collect::<Vec<_>>();
        assert_eq!(observed, vec![1, 1, 2]);
        let err = scan(&bytes, 4096)
            .err()
            .unwrap_or_else(|| panic!("predecode guard unexpectedly accepted fixture"));
        assert!(
            err.message().contains("IPC representation: materialized"),
            "{err}"
        );
    }
    #[test]
    fn representation_dictionary_merge_scratch_and_key_overflow_alternatives() {
        let bytes = delta_stream(1024, 2);
        let error = scan(&bytes, 64 * 1024)
            .err()
            .unwrap_or_else(|| panic!("merge scratch must exceed the tiny budget"));
        assert!(
            error
                .message()
                .contains("IPC representation: scratch budget"),
            "{error}"
        );
        positive(&bytes, 1024 * 1024);
        // Public dictionary merge can deduplicate >127 full input values to
        // two referenced values for Int8 keys. A fallback-only key-end guard
        // must not accidentally reject this valid optimized path.
        use arrow::array::Int8Array;
        use arrow::datatypes::Int8Type;
        let make = |value: u8| {
            let array = DictionaryArray::<Int8Type>::new(
                Int8Array::from(vec![0]),
                Arc::new(UInt8Array::from(vec![value; 200])),
            );
            plain_batch(Arc::new(array))
        };
        let a = make(0);
        let b = make(1);
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, &a.schema())
                .unwrap_or_else(|e| panic!("fixture writer: {e}"));
            writer
                .write(&a)
                .unwrap_or_else(|e| panic!("fixture write: {e}"));
            writer
                .write(&b)
                .unwrap_or_else(|e| panic!("fixture write: {e}"));
            writer
                .finish()
                .unwrap_or_else(|e| panic!("fixture finish: {e}"));
        }
        positive(&bytes, 1024 * 1024);
    }
    #[test]
    fn representation_supported_nested_and_run_types_match_canonical() {
        use arrow::array::{
            Array, BooleanArray, FixedSizeListArray, LargeListArray, ListArray, RunArray,
            StructArray, UnionArray,
        };
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{Int16Type, UnionFields};
        let values = Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef;
        let field = Arc::new(ArrowField::new("item", DataType::Int32, true));
        let list = ListArray::try_new(
            field.clone(),
            OffsetBuffer::new(vec![0i32, 1, 4].into()),
            values.clone(),
            None,
        )
        .unwrap_or_else(|e| panic!("fixture list: {e}"));
        let large = LargeListArray::try_new(
            field.clone(),
            OffsetBuffer::new(vec![0i64, 2, 4].into()),
            values.clone(),
            None,
        )
        .unwrap_or_else(|e| panic!("fixture list: {e}"));
        let fixed = FixedSizeListArray::try_new(field, 2, values, None)
            .unwrap_or_else(|e| panic!("fixture fixed list: {e}"));
        let fields = vec![
            Arc::new(ArrowField::new("list", list.data_type().clone(), true)),
            Arc::new(ArrowField::new("large", large.data_type().clone(), true)),
            Arc::new(ArrowField::new("fixed", fixed.data_type().clone(), true)),
        ]
        .into();
        let record = StructArray::try_new(
            fields,
            vec![Arc::new(list), Arc::new(large), Arc::new(fixed)],
            None,
        )
        .unwrap_or_else(|e| panic!("fixture struct: {e}"));
        positive(
            &plain_stream(&plain_batch(Arc::new(record)), 3),
            1024 * 1024,
        );
        let mut map = arrow::array::builder::MapBuilder::new(
            None,
            arrow::array::builder::StringBuilder::new(),
            arrow::array::builder::Int32Builder::new(),
        );
        map.keys().append_value("key");
        map.values().append_value(3);
        map.append(true)
            .unwrap_or_else(|e| panic!("fixture map: {e}"));
        positive(
            &plain_stream(&plain_batch(Arc::new(map.finish())), 3),
            1024 * 1024,
        );
        let run_ends = arrow::array::Int16Array::from(vec![2i16, 5]);
        let run = RunArray::<Int16Type>::try_new(&run_ends, &Int32Array::from(vec![1, 2]))
            .unwrap_or_else(|e| panic!("fixture runs: {e}"));
        positive(&plain_stream(&plain_batch(Arc::new(run)), 3), 1024 * 1024);
        positive(
            &plain_stream(
                &plain_batch(Arc::new(BooleanArray::from(vec![Some(true), None]))),
                3,
            ),
            1024 * 1024,
        );
        let fields = UnionFields::new(vec![0], vec![ArrowField::new("n", DataType::Int32, false)]);
        let union = UnionArray::try_new(
            fields,
            vec![0i8, 0].into(),
            None,
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )
        .unwrap_or_else(|e| panic!("fixture sparse union: {e}"));
        positive(&plain_stream(&plain_batch(Arc::new(union)), 3), 1024 * 1024);
    }
    #[test]
    fn representation_non_power_of_two_growth_matches_nested_canonical_geometry() {
        use arrow::array::{Array, FixedSizeListArray, ListArray};
        use arrow::buffer::OffsetBuffer;
        fn batch(parents: usize, leaves: usize) -> RecordBatch {
            let values = Arc::new(FixedSizeBinaryArray::new(
                64,
                Buffer::from(vec![7u8; leaves * 64]),
                None,
            )) as ArrayRef;
            let mut offsets = vec![leaves as i32; parents + 1];
            offsets[0] = 0;
            let list = ListArray::try_new(
                Arc::new(ArrowField::new("leaf", values.data_type().clone(), false)),
                OffsetBuffer::new(offsets.into()),
                values,
                None,
            )
            .unwrap_or_else(|e| panic!("canonical list fixture: {e}"));
            let field = Arc::new(ArrowField::new("list", list.data_type().clone(), false));
            let outer = FixedSizeListArray::try_new(field, 1, Arc::new(list), None)
                .unwrap_or_else(|e| panic!("canonical fixed-list fixture: {e}"));
            plain_batch(Arc::new(outer))
        }
        //63 parents,64 leaves: fallback propagates63*64=4032 initial
        // leaf bytes, then the second record grows the requirement to4096.
        let a = batch(62, 63);
        let b = batch(1, 1);
        let mut bytes = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut bytes, &a.schema())
                .unwrap_or_else(|e| panic!("canonical writer: {e}"));
            writer
                .write(&a)
                .unwrap_or_else(|e| panic!("canonical write: {e}"));
            writer
                .write(&b)
                .unwrap_or_else(|e| panic!("canonical write: {e}"));
            writer
                .finish()
                .unwrap_or_else(|e| panic!("canonical finish: {e}"));
        }
        assert!(bytes.len() < 16 * 1024, "safe tiny wire fixture");
        let error = scan(&bytes, 12 * 1024)
            .err()
            .unwrap_or_else(|| panic!("moving growth must exceed private representation budget"));
        assert!(
            error
                .message()
                .contains("IPC representation: materialized buffer budget"),
            "{error}"
        );
        positive(&bytes, 64 * 1024);
        let canonical = super::super::decision_batch::decode_batch_ipc(&bytes)
            .unwrap_or_else(|e| panic!("canonical decode: {e}"));
        assert_eq!(canonical.num_rows(), 63);
        let outer = canonical
            .column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap_or_else(|| panic!("expected fixed list"));
        let list = outer
            .values()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap_or_else(|| panic!("expected list"));
        let leaf = list
            .values()
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap_or_else(|| panic!("expected fixed binary"));
        assert_eq!(leaf.len(), 64);
        // Actual final native capacity is observed. The possible old+new
        // coexistence is calculated from source, NOT measured by this test.
        assert_eq!(leaf.values().capacity(), 8064);
        let permitted_moving_capacity = 4032 + 8064;
        assert_eq!(permitted_moving_capacity, 12096);
        assert!(
            2 * 4096 < permitted_moving_capacity,
            "old2P envelope undercounts"
        );
        assert_eq!(
            storage(4096).unwrap_or_else(|e| panic!("storage: {e}")),
            12288
        );
        assert!(
            storage(4096).unwrap_or_else(|e| panic!("storage: {e}")) >= permitted_moving_capacity
        );
    }
    #[test]
    fn representation_transient_capacity_covers_non_power_of_two_growth_sequences() {
        // Model the PINNED DEPENDENCY's reserve recurrence, without allocating
        // payloads. Different append sizes exercise repeated growth, jumps,
        // alignment crossings and initially non-power-of-two capacities.
        for initial in [0usize, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 4032] {
            for step in [1usize, 7, 65, 257] {
                let final_requirement = 4096usize;
                let bound = storage(initial.max(final_requirement))
                    .unwrap_or_else(|e| panic!("bound: {e}"));
                let mut capacity =
                    aligned(initial).unwrap_or_else(|e| panic!("initial alignment: {e}"));
                let mut required = 0usize;
                while required < final_requirement {
                    required = (required + step).min(final_requirement);
                    if required > capacity {
                        let old = capacity;
                        capacity = aligned(required)
                            .unwrap_or_else(|e| panic!("growth alignment: {e}"))
                            .max(old * 2);
                        assert!(old+capacity<=bound,"initial={initial},step={step},required={required},old={old},new={capacity},bound={bound}");
                    }
                }
                assert!(capacity <= bound);
            }
        }
        assert_eq!(storage(193).unwrap_or_else(|e| panic!("storage: {e}")), 768);
        assert!(storage(193).unwrap_or_else(|e| panic!("storage: {e}")) >= 192 + 384);
        assert_eq!(storage(0).unwrap_or_else(|e| panic!("zero: {e}")), 0);
        let largest_fitting_power = 1usize << (usize::BITS - 2);
        assert_eq!(
            storage(largest_fitting_power).unwrap_or_else(|e| panic!("fitting bound: {e}")),
            largest_fitting_power * 3
        );
        // On either32- or64-bit targets, ceilpow2 itself fits here but the
        // final3P multiply must reject. It must not wrap the component charge.
        assert!(storage(1usize << (usize::BITS - 1)).is_err());
    }
    #[test]
    fn representation_checked_capacity_arithmetic() {
        assert!(plus(usize::MAX, 1).is_err());
        assert!(times(usize::MAX, 2).is_err());
        assert!(storage(usize::MAX).is_err());
        assert!(bits(usize::MAX).is_err());
        assert!(slots(usize::MAX).is_err());
        assert_eq!(
            interner_buckets(0)
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
            256
        );
        assert_eq!(
            interner_buckets(128)
                .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}")),
            512
        );
        assert!(interner_buckets(usize::MAX).is_err());
        let mut b = RepresentationBudget::new(192);
        b.buffer(64, 64)
            .unwrap_or_else(|error| panic!("canonical fixture failed: {error:?}"));
        assert_eq!(b.materialized, 192);
        let mut b = RepresentationBudget::new(191);
        assert!(b.buffer(64, 64).is_err());
    }
}
