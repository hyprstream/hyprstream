// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! A simple calculator to calculate an infinite amount of durations collected from user provided
//! input arguments to seconds. The default time unit is set to nano seconds and parsing an exponent
//! is disabled. The default month and year calculation uses the Julian Year. For demonstration
//! purposes, we use the sidereal year and month. For fun, there's also the `fortnight` as
//! additional custom time unit defined.
//!
//! To reduce run-time costs a little bit, as much as possible is defined as global const.
//!
//! Here's some sample output of this example:
//!
//! ```text
//! cargo run --example custom --features custom --no-default-features -- 10 100ns 1.09y 1.42mon 1week '1     fortnight' 1e20 -1.2 '1 µs'
//! A simple calculator to calculate the input duration to seconds (Default time unit is the nano second):
//!
//!                Input|              Result in seconds
//! --------------------|-------------------------------
//!                   10|                    0.000000010
//!                100ns|                    0.000000100
//!                1.09y|             34398382.959360000
//!              1.42mon|              3352039.944768000
//!                1week|               604800.000000000
//!      1     fortnight|              1209600.000000000
//! Error parsing '1e20': Syntax error: No exponent allowed at column 1
//! Error parsing '-1.2': Number was negative
//!                 1 µs|                    0.000001000
//! ```

use clap::{command, Arg};
use fundu::TimeUnit::*;
use fundu::{CustomDurationParser, CustomTimeUnit, Multiplier};

const CUSTOM_TIME_UNITS: [CustomTimeUnit; 11] = [
    CustomTimeUnit::with_default(NanoSecond, &["ns", "nano", "nanos"]),
    CustomTimeUnit::with_default(MicroSecond, &["µs", "Ms", "micro", "micros"]),
    CustomTimeUnit::with_default(MilliSecond, &["ms", "milli", "millis"]),
    CustomTimeUnit::with_default(Second, &["s", "sec", "secs", "second", "seconds"]),
    CustomTimeUnit::with_default(Minute, &["m", "min", "mins", "minutes"]),
    CustomTimeUnit::with_default(Hour, &["h", "hr", "hrs", "hour", "hours"]),
    CustomTimeUnit::with_default(Day, &["d", "day", "days"]),
    CustomTimeUnit::with_default(Week, &["w", "week", "weeks"]),
    // The fortnight (=2 weeks)
    CustomTimeUnit::new(
        Week,
        &["f", "fortnight", "fortnights"],
        Some(Multiplier(2, 0)),
    ),
    // The sidereal month
    CustomTimeUnit::new(
        Second,
        &["M", "mon", "month", "months"],
        Some(Multiplier(236_059_151, -2)),
    ),
    // The sidereal year
    CustomTimeUnit::new(
        Second,
        &["y", "yr", "year", "years"],
        Some(Multiplier(3_155_814_976, -2)),
    ),
];

fn main() {
    let matches = command!()
        .allow_negative_numbers(true)
        .arg(Arg::new("DURATION").action(clap::ArgAction::Append))
        .get_matches();

    let parser = CustomDurationParser::builder()
        .time_units(&CUSTOM_TIME_UNITS)
        .default_unit(NanoSecond)
        .disable_exponent()
        .allow_time_unit_delimiter()
        .inner_delimiter(|byte| matches!(byte, b'\t' | b'\n' | b'\r' | b' '))
        .build();

    // The headline
    println!(
        "A simple calculator to calculate the input duration to seconds (Default time unit is the \
         nano second):\n"
    );
    // The table header
    println!("{:>20}|{:>31}", "Input", "Result in seconds");
    // The table separator from the content
    println!("{}|{}", "-".repeat(20), "-".repeat(31));

    // Now let's parse and print the output
    for input in matches
        .get_many("DURATION")
        .expect("At least one argument must be present")
        .cloned()
        .collect::<Vec<String>>()
    {
        match parser.parse(input.trim()) {
            Ok(duration) => {
                let duration: std::time::Duration = duration.try_into().unwrap();
                println!(
                    "{:>20}|{:21}.{:09}",
                    &input,
                    duration.as_secs(),
                    duration.subsec_nanos()
                )
            }
            Err(error) => eprintln!("Error parsing '{}': {}", &input, error),
        }
    }
}
