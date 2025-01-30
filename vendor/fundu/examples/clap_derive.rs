// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! Command line application with clap_derive which prints the sum of two durations if two
//! positional arguments were given or else just prints the parsed duration from the first
//! positional argument.

use clap::Parser;
use fundu::DurationParser;

#[derive(Parser)]
#[clap(author, version, about, long_about = None, allow_negative_numbers = true)]
struct Args {
    #[clap(value_name = "DURATION1")]
    duration_1: String,

    #[clap(value_name = "DURATION2")]
    duration_2: Option<String>,
}

fn main() {
    let args = &Args::parse();
    let parser = DurationParser::new();

    match args.duration_2.as_deref() {
        Some(arg2) => {
            let sum = parser
                .parse(&args.duration_1)
                .unwrap()
                .saturating_add(parser.parse(arg2).unwrap()); //
            println!("The sum of the two durations: {sum:?}");
        }
        None => {
            let duration = parser.parse(&args.duration_1).unwrap();
            println!("The duration is: {duration:?}");
        }
    }
}
