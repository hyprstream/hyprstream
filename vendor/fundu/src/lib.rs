// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! # Overview
//!
//! Parse a rust string into a [`crate::Duration`].
//!
//! `fundu` is a configurable, precise and blazingly fast duration parser
//!
//! * with the flexibility to customize [`TimeUnit`]s, or even create own time units with a
//!   [`CustomTimeUnit`] (the `custom` feature is needed)
//! * without floating point calculations. What you put in is what you get out.
//! * with sound limit handling. Infinity and numbers larger than [`Duration::MAX`] evaluate to
//!   [`Duration::MAX`]. Numbers `x` with `abs(x) < 1e-18` evaluate to [`Duration::ZERO`].
//! * with many options to customize the number format and other aspects of the parsing process like
//!   parsing negative durations
//! * and with meaningful error messages
//!
//! # Fundu's Duration
//!
//! This [`crate::Duration`] is returned by the parser of this library and can be converted to a
//! [`std::time::Duration`] and if the feature is activated into a [`time::Duration`] respectively
//! [`chrono::Duration`]. This crates duration is a superset of the aforementioned durations ranging
//! from `-Duration::MAX` to `+Duration::MAX` with `Duration::MAX` having `u64::MAX` seconds and
//! `999_999_999` nano seconds. Converting to fundu's duration from any of the above durations with
//! `From` or `Into` is lossless. Converting from [`crate::Duration`] to any of the other durations
//! can overflow or can't be negative, so conversions must be done with `TryFrom` or `TryInto`.
//! Additionally, fundu's duration implements [`SaturatingInto`] for the above durations, so
//! conversions saturate at the maximum or minimum of these durations.
//!
//! # Features
//!
//! ## `standard`
//!
//! The `standard` feature exposes the [`DurationParser`] and [`DurationParserBuilder`] structs with
//! the a predefined small set of time units. The set of time units can be customized but the
//! `identifier` for each [`TimeUnit`] is fixed.
//!
//! ## `custom`
//!
//! The `custom` feature provides a [`CustomDurationParser`] and [`CustomDurationParserBuilder`]
//! with fully customizable identifiers for each [`TimeUnit`]. With the [`CustomDurationParser`]
//! it is also possible to define completely new time units, a [`CustomTimeUnit`].
//!
//! ## `base`
//!
//! The `base` feature exports the basic [`Parser`] and the [`Config`] on which the `standard` and
//! `custom` features are built. It may lack the convenience of the other features but provides
//! greater freedom. To be able to use this [`Parser`] an implementation of [`TimeUnitsLike`] is
//! needed for time units and optionally for time keywords. Optionally, [`NumbersLike`]
//! implementations are supported, too. For fixed sets of time units and time keywords this is
//! usually a simple and straightforward process. See the documentation of [`Parser`],
//! [`TimeUnitsLike`] and [`NumbersLike`] for examples.
//!
//! ## `chrono` and `time`
//!
//! The `chrono` feature activates methods of [`Duration`] to convert from and to a
//! [`chrono::Duration`]. The `time` feature activates methods of [`Duration`] to convert from and
//! to a [`time::Duration`]. Both of these durations allow negative durations. Parsing negative
//! numbers can be enabled with [`DurationParser::allow_negative`] or
//! [`CustomDurationParser::allow_negative`] independently of the `chrono` or `time` feature.
//!
//! ## `serde`
//!
//! Some structs and enums can be serialized and deserialized with `serde` if the feature is
//! activated.
//!
//! # Configuration and Format
//!
//! The `standard` parser can be configured to accept strings with a default set of time units
//! [`DurationParser::new`], with all time units [`DurationParser::with_all_time_units`] or without
//! [`DurationParser::without_time_units`]. A custom set of time units is also possible with
//! [`DurationParser::with_time_units`]. All these parsers accept strings such as
//!
//! * `1.41`
//! * `42`
//! * `2e-8`, `2e+8` (or likewise `2.0e8`)
//! * `.5` or likewise `0.5`
//! * `3.` or likewise `3.0`
//! * `inf`, `+inf`, `infinity`, `+infinity`
//!
//! All alphabetic characters are matched case-insensitive, so `InFINity` or `2E8` are valid input
//! strings. Additionally, depending on the chosen set of time units one of the following time
//! units (the first column) is accepted.
//!
//! | [`TimeUnit`]    | default id | is default time unit
//! | --------------- | ----------:|:--------------------:
//! | Nanosecond  |         ns | yes
//! | Microsecond |         Ms | yes
//! | Millisecond |         ms | yes
//! | Second      |          s | yes
//! | Minute      |          m | yes
//! | Hour        |          h | yes
//! | Day         |          d | yes
//! | Week        |          w | yes
//! | Month       |          M |  no
//! | Year        |          y |  no
//!
//! If no time unit is given and not specified otherwise with [`DurationParser::default_unit`]
//! then `s` (= [`Second`]) is assumed. Some accepted strings with time units
//!
//! * `31.2s`
//! * `200000Ms`
//! * `3.14e8w`
//! * ...
//!
//! Per default there is no whitespace allowed between the number and the [`TimeUnit`], but this
//! behavior can be changed with [`DurationParser::allow_time_unit_delimiter`].
//!
//! # Format specification
//!
//! The `TimeUnit`s and every `Char` is case-sensitive, all other alphabetic characters are
//! case-insensitive
//!
//! ```text
//! Durations ::= Duration [ DurationStartingWithDigit
//!             | ( Delimiter+ ( Duration | Conjunction ))
//!             ]* ;
//! Conjunction ::= ConjunctionWord (( Delimiter+ Duration ) | DurationStartingWithDigit ) ;
//! ConjunctionWord ::= Char+ ;
//! Duration ::= Sign? ( 'inf' | 'infinity'
//!             | TimeKeyword
//!             | Number [ TimeUnit [ Delimiter+ 'ago' ]]
//!             ) ;
//! DurationStartingWithDigit ::=
//!             ( Digit+ | Digit+ '.' Digit* ) Exp? [ TimeUnit [ Delimiter+ 'ago' ]] ;
//! TimeUnit ::= ns | Ms | ms | s | m | h | d | w | M | y | CustomTimeUnit ;
//! CustomTimeUnit ::= Char+ ;
//! TimeKeyword ::= Char+ ;
//! Number   ::= ( Digits Exp? ) | Exp ;
//! Digits   ::= Digit+ | Digit+ '.' Digit* | Digit* '.' Digit+
//! Exp      ::= 'e' Sign? Digit+ ;
//! Sign     ::= [+-] ;
//! Digit    ::= [0-9] ;
//! Char     ::= ? a valid UTF-8 character ? ;
//! Delimiter ::= ? a closure with the signature u8 -> bool ? ;
//! ```
//!
//! Special cases which are not displayed in the specification:
//!
//! * Parsing multiple `Durations` must be enabled with `parse_multiple`. The [`Delimiter`] and
//!   `ConjunctionWords` can also be defined with the `parse_multiple` method. Multiple `Durations`
//!   are summed up following the saturation rule below
//! * A negative [`Duration`] (`Sign` == `-`), including negative infinity is not allowed as long as
//!   the `allow_negative` option is not enabled. For exceptions see the next point.
//! * Numbers `x` (positive and negative) close to `0` (`abs(x) < 1e-18`) are treated as `0`
//! * Positive infinity and numbers exceeding [`Duration::MAX`] saturate at [`Duration::MAX`]. If
//!   the `allow_negative` option is enabled, negative infinity and numbers falling below
//!   [`Duration::MIN`] saturate at [`Duration::MIN`].
//! * The exponent must be in the range `-32768 <= Exp <= 32767`
//! * If `allow_time_unit_delimiter` is set then any [`Delimiter`] is allowed between the `Number`
//!   and `TimeUnit`.
//! * If `number_is_optional` is enabled then the `Number` is optional but the `TimeUnit` must be
//!   present instead.
//! * The `ago` keyword must be enabled in the parser with `allow_ago`
//! * [`TimeKeyword`] is a `custom` feature which must be enabled by adding a [`TimeKeyword`] to the
//!   [`CustomDurationParser`]
//! * [`CustomTimeUnit`] is a `custom` feature which lets you define own time units
//!
//! # Examples
//!
//! If only the default configuration is required once, the [`parse_duration`] method can be used.
//!
//! ```rust
//! use std::time::Duration;
//!
//! use fundu::parse_duration;
//!
//! let input = "1.0e2s";
//! assert_eq!(parse_duration(input).unwrap(), Duration::new(100, 0));
//! ```
//!
//! When a customization of the accepted [`TimeUnit`]s is required, then [`DurationParser`] can be
//! used.
//!
//! ```rust
//! use fundu::{Duration, DurationParser};
//!
//! let input = "3m";
//! assert_eq!(
//!     DurationParser::with_all_time_units().parse(input).unwrap(),
//!     Duration::positive(180, 0)
//! );
//! ```
//!
//! When no time units are configured, seconds is assumed.
//!
//! ```rust
//! use fundu::{Duration, DurationParser};
//!
//! let input = "1.0e2";
//! assert_eq!(
//!     DurationParser::without_time_units().parse(input).unwrap(),
//!     Duration::positive(100, 0)
//! );
//! ```
//!
//! However, the following will return an error because `y` (Years) is not a default time unit:
//!
//! ```rust
//! use fundu::DurationParser;
//!
//! let input = "3y";
//! assert!(DurationParser::new().parse(input).is_err());
//! ```
//!
//! The parser is reusable and the set of time units is fully customizable
//!
//! ```rust
//! use fundu::TimeUnit::*;
//! use fundu::{Duration, DurationParser};
//!
//! let parser = DurationParser::with_time_units(&[NanoSecond, Minute, Hour]);
//!
//! assert_eq!(parser.parse("9e3ns").unwrap(), Duration::positive(0, 9000));
//! assert_eq!(parser.parse("10m").unwrap(), Duration::positive(600, 0));
//! assert_eq!(parser.parse("1.1h").unwrap(), Duration::positive(3960, 0));
//! assert_eq!(parser.parse("7").unwrap(), Duration::positive(7, 0));
//! ```
//!
//! Setting the default time unit (if no time unit is given in the input string) to something
//! different than seconds is also easily possible
//!
//! ```rust
//! use fundu::TimeUnit::*;
//! use fundu::{Duration, DurationParser};
//!
//! assert_eq!(
//!     DurationParser::without_time_units()
//!         .default_unit(MilliSecond)
//!         .parse("1000")
//!         .unwrap(),
//!     Duration::positive(1, 0)
//! );
//! ```
//!
//! The identifiers for time units can be fully customized with any number of valid
//! [utf-8](https://en.wikipedia.org/wiki/UTF-8) sequences if the `custom` feature is activated:
//!
//! ```rust
//! use fundu::TimeUnit::*;
//! use fundu::{CustomTimeUnit, CustomDurationParser, Duration};
//!
//! let parser = CustomDurationParser::with_time_units(&[
//!     CustomTimeUnit::with_default(MilliSecond, &["χιλιοστό του δευτερολέπτου"]),
//!     CustomTimeUnit::with_default(Second, &["s", "secs"]),
//!     CustomTimeUnit::with_default(Hour, &["⏳"]),
//! ]);
//!
//! assert_eq!(parser.parse(".3χιλιοστό του δευτερολέπτου"), Ok(Duration::positive(0, 300_000)));
//! assert_eq!(parser.parse("1e3secs"), Ok(Duration::positive(1000, 0)));
//! assert_eq!(parser.parse("1.1⏳"), Ok(Duration::positive(3960, 0)));
//! ```
//!
//! The `custom` feature can be used to customize a lot more. See the documentation of the exported
//! items of the `custom` feature (like [`CustomTimeUnit`], [`TimeKeyword`]) for more information.
//!
//! Also, `fundu` tries to give informative error messages
//!
//! ```rust
//! use fundu::DurationParser;
//!
//! assert_eq!(
//!     DurationParser::without_time_units()
//!         .parse("1y")
//!         .unwrap_err()
//!         .to_string(),
//!     "Time unit error: No time units allowed but found: 'y' at column 1"
//! );
//! ```
//!
//! The number format can be easily adjusted to your needs. For example to allow numbers being
//! optional, allow some ascii whitespace between the number and the time unit and restrict the
//! number format to whole numbers, without fractional part and an exponent:
//!
//! ```rust
//! use fundu::TimeUnit::*;
//! use fundu::{Duration, DurationParser, ParseError};
//!
//! const PARSER: DurationParser = DurationParser::builder()
//!     .time_units(&[NanoSecond])
//!     .allow_time_unit_delimiter()
//!     .number_is_optional()
//!     .disable_fraction()
//!     .disable_exponent()
//!     .build();
//!
//! assert_eq!(PARSER.parse("ns").unwrap(), Duration::positive(0, 1));
//! assert_eq!(
//!     PARSER.parse("1000\t\n\r ns").unwrap(),
//!     Duration::positive(0, 1000)
//! );
//!
//! assert_eq!(
//!     PARSER.parse("1.0ns").unwrap_err(),
//!     ParseError::Syntax(1, "No fraction allowed".to_string())
//! );
//! assert_eq!(
//!     PARSER.parse("1e9ns").unwrap_err(),
//!     ParseError::Syntax(1, "No exponent allowed".to_string())
//! );
//! ```
//!
//! [`time::Duration`]: <https://docs.rs/time/latest/time/struct.Duration.html>
//! [`chrono::Duration`]: <https://docs.rs/chono/latest/chrono/struct.Duration.html>
//! [`Duration::MAX`]: [`std::Duration::MAX`]
//! [`Duration::ZERO`]: [`std::Duration::ZERO`]
//! [`NanoSecond`]: [`TimeUnit::NanoSecond`]
//! [`MicroSecond`]: [`TimeUnit::MicroSecond`]
//! [`MilliSecond`]: [`TimeUnit::MilliSecond`]
//! [`Second`]: [`TimeUnit::Second`]
//! [`Minute`]: [`TimeUnit::Minute`]
//! [`Hour`]: [`TimeUnit::Hour`]
//! [`Day`]: [`TimeUnit::Day`]
//! [`Week`]: [`TimeUnit::Week`]
//! [`Month`]: [`TimeUnit::Month`]
//! [`Year`]: [`TimeUnit::Year`]

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc(test(attr(warn(unused))))]
#![doc(test(attr(allow(unused_extern_crates))))]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::else_if_without_else)]
#![warn(clippy::fn_to_numeric_cast_any)]
#![warn(clippy::get_unwrap)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::mixed_read_write_in_expression)]
#![warn(clippy::partial_pub_fields)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::todo)]
#![warn(clippy::try_err)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unneeded_field_pattern)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "custom")]
mod custom;
#[cfg(feature = "standard")]
mod standard;

#[cfg(feature = "custom")]
pub use custom::{
    builder::CustomDurationParserBuilder,
    parser::CustomDurationParser,
    time_units::{
        CustomTimeUnit, TimeKeyword, DEFAULT_ALL_TIME_UNITS, DEFAULT_TIME_UNITS, SYSTEMD_TIME_UNITS,
    },
    Numeral,
};
pub use fundu_core::config::Delimiter;
pub use fundu_core::error::{ParseError, TryFromDurationError};
pub use fundu_core::time::{
    Duration, Multiplier, SaturatingInto, TimeUnit, DEFAULT_ID_DAY, DEFAULT_ID_HOUR,
    DEFAULT_ID_MICRO_SECOND, DEFAULT_ID_MILLI_SECOND, DEFAULT_ID_MINUTE, DEFAULT_ID_MONTH,
    DEFAULT_ID_NANO_SECOND, DEFAULT_ID_SECOND, DEFAULT_ID_WEEK, DEFAULT_ID_YEAR,
};
#[cfg(test)]
pub use rstest_reuse;
#[cfg(feature = "standard")]
pub use standard::{
    builder::DurationParserBuilder, parser::parse_duration, parser::DurationParser,
};
#[cfg(feature = "base")]
pub use {
    fundu_core::config::{Config, ConfigBuilder, NumbersLike},
    fundu_core::parse::Parser,
    fundu_core::time::TimeUnitsLike,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        #[cfg(feature = "custom")]
        {
            assert_send::<CustomDurationParserBuilder>();
            assert_send::<CustomDurationParser>();
            assert_send::<CustomTimeUnit>();
            assert_send::<TimeKeyword>();
        }

        #[cfg(feature = "standard")]
        {
            assert_send::<DurationParserBuilder>();
            assert_send::<DurationParser>();
        }
    }

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        #[cfg(feature = "custom")]
        {
            assert_sync::<CustomDurationParserBuilder>();
            assert_sync::<CustomDurationParser>();
            assert_sync::<CustomTimeUnit>();
            assert_sync::<TimeKeyword>();
        }
        #[cfg(feature = "standard")]
        {
            assert_sync::<DurationParserBuilder>();
            assert_sync::<DurationParser>();
        }
    }
}
