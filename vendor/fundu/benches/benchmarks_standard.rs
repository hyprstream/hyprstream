// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fundu::DurationParser;
use fundu::TimeUnit::*;

fn criterion_config() -> Criterion {
    Criterion::default()
        .nresamples(500_000)
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
}

fn get_parsing_speed_inputs() -> Vec<(String, String)> {
    vec![
        ("single digit".to_string(), "1".to_string()),
        ("mixed digits 7".to_string(), "1234567.1234567".to_string()),
        (
            "mixed digits 8".to_string(),
            "12345678.12345678".to_string(),
        ),
        (
            "mixed digits 9".to_string(),
            "123456789.123456789".to_string(),
        ),
        (
            "large input".to_string(),
            format!("{}.{}e-1022", "1".repeat(1022), "1".repeat(1022)),
        ),
    ]
}

fn benchmark_initialization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("standard duration parser initialization");
    group.bench_function("without time units", |b| {
        b.iter(DurationParser::without_time_units)
    });
    group.bench_function("default time units", |b| b.iter(DurationParser::new));
    let input = &[
        NanoSecond,
        MicroSecond,
        MilliSecond,
        Second,
        Minute,
        Hour,
        Day,
        Week,
        Month,
        Year,
    ];
    group.bench_with_input("custom time units", input, |b, input| {
        b.iter(|| DurationParser::with_time_units(input))
    });
    group.bench_function("all time units", |b| {
        b.iter(DurationParser::with_all_time_units)
    });
    group.finish();
}

fn benchmark_parsing(criterion: &mut Criterion) {
    let inputs = get_parsing_speed_inputs();
    let parser = DurationParser::with_all_time_units();
    let mut group = criterion.benchmark_group("parsing speed");
    for (parameter, input) in inputs {
        group.bench_with_input(
            BenchmarkId::new("input without time units", parameter),
            &input,
            |b, input| b.iter(|| black_box(&parser).parse(input).unwrap()),
        );
    }
    group.finish();
}

fn benchmark_parsing_infinity(criterion: &mut Criterion) {
    let inputs = vec!["inf", "infinity"];
    let parser = DurationParser::with_all_time_units();
    let mut group = criterion.benchmark_group("parsing speed infinity");
    for input in inputs {
        group.bench_with_input(input, input, |b, input| {
            b.iter(|| black_box(&parser).parse(input).unwrap())
        });
    }
    group.finish();
}

fn benchmark_parsing_with_time_units(criterion: &mut Criterion) {
    let inputs = [(NanoSecond, "ns"), (Second, "s"), (Year, "y")];
    let mut parser = DurationParser::with_all_time_units();
    parser.number_is_optional(true);
    let mut group = criterion.benchmark_group("parsing speed time units");
    for (unit, input) in inputs {
        parser.default_unit(unit);
        group.bench_with_input(
            BenchmarkId::new(format!("input without time unit (default = {unit:?})"), "1"),
            "1",
            |b, input| b.iter(|| black_box(&parser).parse(input).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new(format!("input without number (default = {unit:?})"), input),
            input,
            |b, input| b.iter(|| black_box(&parser).parse(input).unwrap()),
        );
        let input = format!("1{input}");
        group.bench_with_input(
            BenchmarkId::new(
                format!("input with time units (default = {unit:?})"),
                &input,
            ),
            &input,
            |b, input| b.iter(|| black_box(&parser).parse(input).unwrap()),
        );
    }
    group.finish();
}

fn reference_benchmark(criterion: &mut Criterion) {
    let inputs = get_parsing_speed_inputs();
    let mut group = criterion.benchmark_group("reference speed");
    for (parameter, input) in inputs {
        group.bench_with_input(
            BenchmarkId::new("reference function", parameter),
            &input,
            |b, input| b.iter(|| Duration::from_secs_f64(input.parse().unwrap())),
        );
    }
    group.finish();
}

criterion_group!(
    name = initialization;
    config = criterion_config();
    targets = benchmark_initialization
);
criterion_group!(
    name = parsing;
    config = criterion_config();
    targets = benchmark_parsing
);
criterion_group!(
    name = parsing_infinity;
    config = criterion_config();
    targets = benchmark_parsing_infinity
);
criterion_group!(
    name = parsing_time_units;
    config = criterion_config();
    targets = benchmark_parsing_with_time_units
);
criterion_group!(
    name = reference;
    config = criterion_config();
    targets = reference_benchmark
);
criterion_main!(
    initialization,
    parsing,
    parsing_infinity,
    reference,
    parsing_time_units
);
