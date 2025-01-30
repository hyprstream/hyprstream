// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! fundu's Duration converted to std::time::Duration, chrono::Duration and time::Duration

use clap::{command, Arg};
use fundu::{DurationParser, SaturatingInto};

const PARSER: DurationParser = DurationParser::builder()
    .all_time_units()
    .allow_negative()
    .build();

fn main() {
    let matches = command!()
        .about("Conversion of fundu's duration to other durations")
        .allow_negative_numbers(true)
        .allow_hyphen_values(true)
        .arg(
            Arg::new("DURATION")
                .action(clap::ArgAction::Set)
                .help("A duration"),
        )
        .get_matches();

    let input: &String = matches
        .get_one("DURATION")
        .expect("One argument must be present");

    match PARSER.parse(input.trim()) {
        Ok(duration) => {
            println!("Input was: '{}'", input);
            println!("fundu::Duration: {:?}", duration);
            println!("fundu::Duration as human readable string: {}\n", duration);

            match std::time::Duration::try_from(duration) {
                Ok(duration) => {
                    println!("Conversion to std::time::Duration: {:?}", duration)
                }
                Err(error) => println!(
                    "Error converting to std::time::Duration: The original error message was: '{}'",
                    error
                ),
            };
            println!(
                "Saturating conversion with SaturatingInto::<std::time::Duration>: {:?}\n",
                SaturatingInto::<std::time::Duration>::saturating_into(duration)
            );

            match chrono::Duration::try_from(duration) {
                Ok(duration) => {
                    println!("Conversion to chrono::Duration: {:?}", duration)
                }
                Err(error) => println!(
                    "Error converting to chrono::Duration: The original error message was: '{}'",
                    error
                ),
            };
            println!(
                "Saturating conversion with SaturatingInto::<chrono::Duration>: {:?}\n",
                SaturatingInto::<chrono::Duration>::saturating_into(duration)
            );

            match time::Duration::try_from(duration) {
                Ok(duration) => {
                    println!("Conversion to time::Duration: {:?}", duration)
                }
                Err(error) => println!(
                    "Error converting to time::Duration: The original error message was: '{}'",
                    error
                ),
            };
            println!(
                "Saturating conversion with SaturatingInto::<time::Duration>: {:?}",
                SaturatingInto::<time::Duration>::saturating_into(duration)
            );
        }
        Err(error) => println!("Error parsing duration '{}': {}", input, error),
    }
}
