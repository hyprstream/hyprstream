// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use clap::{arg, command};
use fundu::DurationParser;

fn main() {
    let matches = command!()
        .arg(arg!([DURATION1] "A duration"))
        .arg(
            arg!([DURATION2] "An optional duration to sum with the first duration").required(false),
        )
        .get_matches();
    let parser = DurationParser::new();

    if let Some(arg2) = matches.get_one::<String>("DURATION2") {
        let sum = parser
            .parse(matches.get_one::<String>("DURATION1").unwrap())
            .unwrap()
            .saturating_add(parser.parse(arg2).unwrap());
        println!("The sum of the two durations: {sum:?}");
    } else {
        let duration = parser
            .parse(matches.get_one::<String>("DURATION1").unwrap())
            .unwrap();
        println!("The duration is: {duration:?}");
    }
}
