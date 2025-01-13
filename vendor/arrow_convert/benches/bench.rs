use arrow::{
    array::ArrayRef,
    buffer::{Buffer, ScalarBuffer},
};
use arrow_convert::{
    deserialize::TryIntoCollection, serialize::TryIntoArrow, ArrowDeserialize, ArrowField, ArrowSerialize,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Arrow stores U8 arrays as `arrow::array::BinaryArray`
#[derive(ArrowField, ArrowSerialize, ArrowDeserialize)]
#[arrow_field(transparent)]
pub struct BufU8Struct(Buffer);

// Arrow stores other arrows as `arrow::array::ListArray`
#[derive(ArrowField, ArrowSerialize, ArrowDeserialize)]
#[arrow_field(transparent)]
pub struct BufU32Struct(ScalarBuffer<u32>);

// Arrow stores U8 arrows as `arrow::array::BinaryArray`
#[derive(ArrowField, ArrowSerialize, ArrowDeserialize)]
#[arrow_field(transparent)]
pub struct VecU8Struct(Vec<u8>);

// Arrow stores other arrows as `arrow::array::ListArray`
#[derive(ArrowField, ArrowSerialize, ArrowDeserialize)]
#[arrow_field(transparent)]
pub struct VecU32Struct(Vec<u32>);

pub fn bench_buffer_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialize");
    for size in [1, 10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("BufferU8", size), size, |b, &size| {
            let data = [BufU8Struct((0..size as u8).collect())];
            b.iter(|| {
                let _: ArrayRef = TryIntoArrow::try_into_arrow(black_box(&data)).unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("VecU8", size), size, |b, &size| {
            let data = [VecU8Struct((0..size as u8).collect())];
            b.iter(|| {
                let _: ArrayRef = TryIntoArrow::try_into_arrow(black_box(&data)).unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("BufferU32", size), size, |b, &size| {
            let data = [BufU32Struct((0..size as u32).collect())];
            b.iter(|| {
                let _: ArrayRef = TryIntoArrow::try_into_arrow(black_box(&data)).unwrap();
            });
        });
        group.bench_with_input(BenchmarkId::new("VecU32", size), size, |b, &size| {
            let data = [VecU32Struct((0..size as u32).collect())];
            b.iter(|| {
                let _: ArrayRef = TryIntoArrow::try_into_arrow(black_box(&data)).unwrap();
            });
        });
    }
}
pub fn bench_buffer_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("deserialize");
    for size in [1, 10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("BufferU8", size), size, |b, &size| {
            let data: ArrayRef = [BufU8Struct((0..size as u8).collect())].try_into_arrow().unwrap();
            b.iter_batched(
                || data.clone(),
                |data| {
                    let _: Vec<BufU8Struct> = TryIntoCollection::try_into_collection(black_box(data)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("VecU8", size), size, |b, &size| {
            let data: ArrayRef = [VecU8Struct((0..size as u8).collect())].try_into_arrow().unwrap();
            b.iter_batched(
                || data.clone(),
                |data| {
                    let _: Vec<VecU8Struct> = TryIntoCollection::try_into_collection(black_box(data)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("BufferU32", size), size, |b, &size| {
            let data: ArrayRef = [BufU32Struct((0..size as u32).collect())].try_into_arrow().unwrap();
            b.iter_batched(
                || data.clone(),
                |data| {
                    let _: Vec<BufU32Struct> = TryIntoCollection::try_into_collection(black_box(data)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            )
        });
        group.bench_with_input(BenchmarkId::new("VecU32", size), size, |b, &size| {
            let data: ArrayRef = [VecU32Struct((0..size as u32).collect())].try_into_arrow().unwrap();
            b.iter_batched(
                || data.clone(),
                |data| {
                    let _: Vec<VecU32Struct> = TryIntoCollection::try_into_collection(black_box(data)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
}

criterion_group!(benches, bench_buffer_serialize, bench_buffer_deserialize);
criterion_main!(benches);
