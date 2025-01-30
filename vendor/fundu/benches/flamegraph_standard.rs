// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! Flamegraphs for the standard module and the DurationParser

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fundu::TimeUnit::*;
use fundu::{DurationParser, DurationParserBuilder};
use pprof::criterion::{Output, PProfProfiler};
use pprof::flamegraph::Options as FlamegraphOptions;

fn flamegraph_initialization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("standard duration parser initialization");
    group.bench_function("without time units", |b| {
        b.iter(|| DurationParser::without_time_units())
    });
    group.bench_function("default time units", |b| b.iter(|| DurationParser::new()));
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
        b.iter(|| DurationParser::with_all_time_units())
    });
    group.finish();
}

fn flamegraph_parsing(criterion: &mut Criterion) {
    let parser = DurationParserBuilder::new()
        .all_time_units()
        .number_is_optional()
        .build();
    let mut group = criterion.benchmark_group("standard duration parser parsing");
    for &input in &[
        "1",
        "1s",
        "s",
        "1ns",
        "ns",
        "1y",
        "y",
        "1234567.1234567",
        "12345678.12345678",
        &format!("{}.{}e-1022", "1".repeat(1022), "1".repeat(1022)),
    ] {
        group.bench_with_input(
            BenchmarkId::new("all default time units", input),
            &(&parser, input),
            |b, (parser, input)| b.iter(|| parser.parse(input).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(
    name = initialization;
    config = Criterion::default().with_profiler(PProfProfiler::new(1_000_000, Output::Flamegraph({
        let mut options = FlamegraphOptions::default();
        options.title = "Flame graph for standard duration parser".to_string();
        options.subtitle = Some("Initialization".to_string());
        Some(options)
    })));
    targets = flamegraph_initialization
);
criterion_group!(
    name = parsing;
    config = Criterion::default().with_profiler(PProfProfiler::new(1_000_000, Output::Flamegraph({
        let mut options = FlamegraphOptions::default();
        options.title = "Flame graph for standard duration parser".to_string();
        options.subtitle = Some("Parsing".to_string());
        Some(options)
    })));
    targets = flamegraph_parsing
);
criterion_main!(initialization, parsing);
