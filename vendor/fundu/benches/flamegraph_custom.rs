// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! Flamegraphs for the custom module and the CustomDurationParser

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use fundu::{CustomDurationParser, DEFAULT_ALL_TIME_UNITS, SYSTEMD_TIME_UNITS};
use pprof::criterion::{Output, PProfProfiler};
use pprof::flamegraph::Options as FlamegraphOptions;

fn flamegraph_initialization(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("custom duration parser initialization");
    group.bench_function("without time units", |b| {
        b.iter(|| CustomDurationParser::new())
    });
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

fn flamegraph_parsing(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("custom duration parser parsing");
    for &input in &["1", "1ns", "1y", "1years"] {
        group.bench_with_input(
            BenchmarkId::new("systemd time units", input),
            &(
                CustomDurationParser::with_time_units(&SYSTEMD_TIME_UNITS),
                input,
            ),
            |b, (parser, input)| b.iter(|| parser.parse(input)),
        );
    }

    group.finish();
}

criterion_group!(
    name = initialization;
    config = Criterion::default().with_profiler(PProfProfiler::new(1_000_000, Output::Flamegraph({
        let mut options = FlamegraphOptions::default();
        options.title = "Flame graph for custom duration parser".to_string();
        options.subtitle = Some("Initialization".to_string());
        Some(options)
    })));
    targets = flamegraph_initialization
);
criterion_group!(
    name = parsing;
    config = Criterion::default().with_profiler(PProfProfiler::new(1_000_000, Output::Flamegraph({
        let mut options = FlamegraphOptions::default();
        options.title = "Flame graph for custom duration parser".to_string();
        options.subtitle = Some("Parsing".to_string());
        Some(options)
    })));
    targets = flamegraph_parsing
);
criterion_main!(initialization, parsing);
