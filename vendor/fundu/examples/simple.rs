// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use fundu::{Duration, DurationParser, TimeUnit};

fn main() {
    let input = "100y";
    let duration = DurationParser::with_time_units(&[TimeUnit::Year])
        .parse(input)
        .unwrap();

    assert_eq!(duration, Duration::positive(3_155_760_000, 0))
}
