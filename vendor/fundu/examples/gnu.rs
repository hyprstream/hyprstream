// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! A gnu relative time parser as specified here
//! `https://www.gnu.org/software/coreutils/manual/html_node/Relative-items-in-date-strings.html`.

use clap::{command, Arg};
use fundu::TimeUnit::*;
use fundu::{
    CustomDurationParserBuilder, CustomTimeUnit, Duration, Multiplier, SaturatingInto, TimeKeyword,
};

const GNU_TIME_UNITS: [CustomTimeUnit<'static>; 8] = [
    CustomTimeUnit::with_default(Second, &["sec", "secs", "second", "seconds"]),
    CustomTimeUnit::with_default(Minute, &["min", "mins", "minute", "minutes"]),
    CustomTimeUnit::with_default(Hour, &["hour", "hours"]),
    CustomTimeUnit::with_default(Day, &["day", "days"]),
    CustomTimeUnit::with_default(Week, &["week", "weeks"]),
    CustomTimeUnit::new(Week, &["fortnight", "fortnights"], Some(Multiplier(2, 0))),
    CustomTimeUnit::with_default(Month, &["month", "months"]),
    CustomTimeUnit::with_default(Year, &["year", "years"]),
];

const GNU_KEYWORDS: [TimeKeyword<'static>; 3] = [
    TimeKeyword::new(Day, &["yesterday"], Some(Multiplier(-1, 0))),
    TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))),
    TimeKeyword::new(Day, &["now", "today"], Some(Multiplier(0, 0))),
];

// Whitespace as defined in the POSIX locale (and used by gnu). In contrast to rust's definition of
// whitespace the POSIX definition includes the VERTICAL TAB:
// SPACE, HORIZONTAL TAB, LINE FEED, FORM FEED, CARRIAGE RETURN, VERTICAL TAB
const DELIMITER: fn(u8) -> bool =
    |byte| matches!(byte, b' ' | b'\t' | b'\n' | b'\x0c' | b'\r' | b'\x0b');

const PARSER_BUILDER: CustomDurationParserBuilder = CustomDurationParserBuilder::new()
    .allow_ago()
    .allow_time_unit_delimiter()
    .allow_negative()
    .disable_exponent()
    .disable_fraction()
    .disable_infinity()
    .number_is_optional()
    .inner_delimiter(DELIMITER)
    .outer_delimiter(DELIMITER)
    .parse_multiple(None);

fn make_plural(time: u64, singular: &str) -> String {
    if time > 1 {
        format!("{}s", singular)
    } else {
        singular.to_string()
    }
}

/// Create a human readable string like `100years 2hours 1min 40secs` from a `Duration`
fn make_human(duration: Duration) -> String {
    const YEAR: u64 = Year.multiplier().0.unsigned_abs();
    const MONTH: u64 = Month.multiplier().0.unsigned_abs();
    const WEEK: u64 = Week.multiplier().0.unsigned_abs();
    const DAY: u64 = Day.multiplier().0.unsigned_abs();
    const HOUR: u64 = Hour.multiplier().0.unsigned_abs();
    const MINUTE: u64 = Minute.multiplier().0.unsigned_abs();

    if duration.is_zero() {
        return "0sec".to_string();
    }

    let std_duration: std::time::Duration = duration.abs().saturating_into();

    let mut result = Vec::with_capacity(10);
    let mut secs = std_duration.as_secs();
    if secs > 0 {
        if secs >= YEAR {
            let years = secs / YEAR;
            result.push(format!("{}{}", years, make_plural(years, "year")));
            secs %= YEAR;
        }
        if secs >= MONTH {
            let months = secs / MONTH;
            result.push(format!("{}{}", months, make_plural(months, "month")));
            secs %= MONTH;
        }
        if secs >= WEEK {
            let weeks = secs / WEEK;
            result.push(format!("{}{}", weeks, make_plural(weeks, "week")));
            secs %= WEEK;
        }
        if secs >= DAY {
            let days = secs / DAY;
            result.push(format!("{}{}", days, make_plural(days, "day")));
            secs %= DAY;
        }
        if secs >= HOUR {
            let hours = secs / HOUR;
            result.push(format!("{}{}", hours, make_plural(hours, "hour")));
            secs %= HOUR;
        }
        if secs >= MINUTE {
            let minutes = secs / MINUTE;
            result.push(format!("{}{}", minutes, make_plural(minutes, "min")));
            secs %= MINUTE;
        }
        if secs >= 1 {
            result.push(format!("{}{}", secs, make_plural(secs, "sec")));
        }
    }

    if duration.is_negative() {
        format!("-{}", &result.join(" -"))
    } else {
        result.join(" ")
    }
}

fn main() {
    let matches = command!()
        .about(
            "A gnu relative time parser as specified in `info '(coreutils) Relative items in \
             date'`.",
        )
        .allow_negative_numbers(true)
        .allow_hyphen_values(true)
        .arg(
            Arg::new("GNU_RELATIVE_TIME")
                .action(clap::ArgAction::Set)
                .help(
                    "A relative time as specified in `info '(coreutils) Relative items in date \
                     strings'`",
                ),
        )
        .get_matches();

    let parser = PARSER_BUILDER
        .time_units(&GNU_TIME_UNITS)
        .keywords(&GNU_KEYWORDS)
        .build();

    let input: &String = matches
        .get_one("GNU_RELATIVE_TIME")
        .expect("One argument must be present");
    match parser.parse(input.trim()) {
        Ok(duration) => {
            let std_duration: std::time::Duration = duration.abs().saturating_into();
            println!("{:>8}: {}", "Original", input);
            println!(
                "{:>8}: {}",
                "Seconds",
                if duration.is_negative() {
                    format!("-{}", std_duration.as_secs())
                } else {
                    std_duration.as_secs().to_string()
                }
            );
            println!("{:>8}: {}", "Human", make_human(duration));
        }
        Err(error) => eprintln!("Failed to parse relative time '{}': {}", &input, error),
    }
}
