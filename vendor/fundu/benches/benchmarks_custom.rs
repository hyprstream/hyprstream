// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fundu::{CustomDurationParser, DEFAULT_ALL_TIME_UNITS, SYSTEMD_TIME_UNITS};

fn criterion_config() -> Criterion {
    Criterion::default()
        .nresamples(500_000)
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
}

fn benchmark_initialization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("custom duration parser initialization");
    group.bench_function("without time units", |b| b.iter(CustomDurationParser::new));
    group.bench_with_input(
        "default time units",
        &DEFAULT_ALL_TIME_UNITS,
        |b, time_units| b.iter(|| CustomDurationParser::with_time_units(time_units)),
    );
    group.bench_with_input(
        "systemd time units",
        &SYSTEMD_TIME_UNITS,
        |b, time_units| b.iter(|| CustomDurationParser::with_time_units(time_units)),
    );
    group.finish();
}

fn benchmark_parsing(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("custom duration parser parsing speed time units");
    let default_parser = CustomDurationParser::with_time_units(&DEFAULT_ALL_TIME_UNITS);
    let systemd_parser = CustomDurationParser::with_time_units(&SYSTEMD_TIME_UNITS);
    for input in ["1", "1s", "1ns", "1y"] {
        group.bench_with_input(
            BenchmarkId::new("default time units", input),
            input,
            |b, input| b.iter(|| black_box(&default_parser).parse(input).unwrap()),
        );
        group.bench_with_input(
            BenchmarkId::new("systemd time units", input),
            input,
            |b, input| b.iter(|| black_box(&systemd_parser).parse(input).unwrap()),
        );
    }
    let input = "1years";
    group.bench_with_input(
        BenchmarkId::new("systemd time units", input),
        input,
        |b, input| b.iter(|| black_box(&systemd_parser).parse(input).unwrap()),
    );
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

criterion_main!(initialization, parsing);
