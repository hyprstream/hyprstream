// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
use std::hint::black_box;

use fundu::TimeUnit::*;
use fundu::{Duration, DurationParser, DurationParserBuilder, TimeUnit};
use iai_callgrind::{
    library_benchmark, library_benchmark_group, main, EventKind, FlamegraphConfig,
    LibraryBenchmarkConfig, RegressionConfig,
};

#[library_benchmark]
#[bench::small(DurationParser::without_time_units(), "1")]
#[bench::mixed_7(DurationParser::without_time_units(), "1234567.1234567")]
#[bench::mixed_8(DurationParser::without_time_units(), "12345678.12345678")]
#[bench::large(DurationParser::without_time_units(), &format!("{0}.{0}e-1022", "1".repeat(1022)))]
fn without_time_units(parser: DurationParser, input: &str) -> Duration {
    black_box(parser.parse(input)).unwrap()
}

fn setup_parser_with_default<'a>(default_unit: TimeUnit) -> DurationParser<'a> {
    DurationParserBuilder::new()
        .all_time_units()
        .default_unit(default_unit)
        .number_is_optional()
        .build()
}

#[library_benchmark]
#[bench::one_when_default_is_nano_second(setup_parser_with_default(NanoSecond), "1")]
#[bench::one_nano_second(setup_parser_with_default(NanoSecond), "1ns")]
#[bench::nano_second(setup_parser_with_default(NanoSecond), "ns")]
#[bench::one_when_default_is_second(setup_parser_with_default(Second), "1")]
#[bench::one_second(setup_parser_with_default(Second), "1s")]
#[bench::second(setup_parser_with_default(Second), "s")]
#[bench::one_when_default_is_year(setup_parser_with_default(Year), "1")]
#[bench::one_year(setup_parser_with_default(Year), "1y")]
#[bench::year(setup_parser_with_default(Year), "y")]
fn with_time_units(parser: DurationParser, input: &str) -> Duration {
    black_box(parser.parse(input)).unwrap()
}

library_benchmark_group!(name = parsing_speed; benchmarks = without_time_units, with_time_units);

main!(
    config = LibraryBenchmarkConfig::default()
        .flamegraph(
            FlamegraphConfig::default()
        )
        .regression(
            RegressionConfig::default().limits([(EventKind::Ir, 5.0)])
        );
    library_benchmark_groups = parsing_speed
);
