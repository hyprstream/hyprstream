// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
use std::hint::black_box;

use fundu::{CustomDurationParser, DEFAULT_ALL_TIME_UNITS, SYSTEMD_TIME_UNITS};
use iai_callgrind::{
    library_benchmark, library_benchmark_group, main, EventKind, FlamegraphConfig,
    LibraryBenchmarkConfig, RegressionConfig,
};

#[library_benchmark]
fn default_time_units<'a>() -> CustomDurationParser<'a> {
    CustomDurationParser::with_time_units(black_box(&DEFAULT_ALL_TIME_UNITS))
}

#[library_benchmark]
fn systemd_time_units<'a>() -> CustomDurationParser<'a> {
    CustomDurationParser::with_time_units(black_box(&SYSTEMD_TIME_UNITS))
}

library_benchmark_group!(
    name = initialization;
    benchmarks = default_time_units, systemd_time_units
);

main!(
    config = LibraryBenchmarkConfig::default()
        .flamegraph(FlamegraphConfig::default())
        .regression(RegressionConfig::default().limits([(EventKind::Ir, 5.0)]));
    library_benchmark_groups = initialization
);
