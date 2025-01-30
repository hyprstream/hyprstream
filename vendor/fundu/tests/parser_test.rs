// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
//
// cspell: ignore infi anda manµ someµ tomorro

use std::time::Duration as StdDuration;

use fundu::TimeUnit::*;
use fundu::{
    parse_duration, CustomDurationParser, CustomDurationParserBuilder, CustomTimeUnit, Duration,
    DurationParser, Multiplier, ParseError, TimeKeyword, TimeUnit, SYSTEMD_TIME_UNITS,
};
use rstest::rstest;

const YEAR: u64 = 60 * 60 * 24 * 365 + 60 * 60 * 24 / 4; // 365 days + day/4
const MONTH: u64 = YEAR / 12;

#[rstest]
#[case::empty_string("")]
#[case::leading_whitespace("  1")]
#[case::trailing_whitespace("1   ")]
#[case::trailing_invalid_character("1.1e1msNOPE")]
#[case::only_whitespace("  \t\n")]
#[case::only_point(".")]
#[case::only_sign("+")]
#[case::only_exponent("e-10")]
#[case::sign_with_exponent("-e1")]
#[case::sign_with_point_and_exponent("-.e1")]
#[case::negative_seconds("-1")]
#[case::negative_seconds_with_fraction("-1.0")]
#[case::negative_nano_seconds("-0.000000001")]
#[case::exponent_with_only_plus("2E+")]
#[case::negative_number_high_exponent("-3E75")]
#[case::negative_number_barely_not_zero("-1.e-18")]
#[case::negative_number_1("-.00000000000019999999e-4")]
#[case::negative_number_2("-.00000003e-2")]
#[case::negative_large_input_with_high_exponent("-088888888888888888888888818288833333333333333333333333333333333333333333333388888888888888888880000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000060000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001616564957e-1027")]
fn test_parse_duration_with_illegal_argument_then_error(#[case] source: &str) {
    let parser = DurationParser::builder().default_time_units().build();
    assert!(parser.parse(source).is_err());

    let parser = DurationParser::builder()
        .default_time_units()
        .parse_multiple(Some(&["and"]))
        .build();
    assert!(parser.parse(source).is_err());
}

#[rstest]
#[case::simple_number("1", Duration::positive(1, 0))]
#[case::trailing_zeros("10.010000000", Duration::positive(10, 10_000_000))]
#[case::simple_zero("0", Duration::ZERO)]
#[case::many_zeros(&"0".repeat(2000), Duration::ZERO)]
#[case::many_leading_zeros(&format!("{}1", "0".repeat(2000)), Duration::positive(1, 0))]
#[case::zero_point_zero("0.0", Duration::ZERO)]
#[case::point_zero(".0", Duration::ZERO)]
#[case::zero_point("0.", Duration::ZERO)]
#[case::one_with_fraction_number("1.1", Duration::positive(1, 100_000_000))]
#[case::leading_zero_max_nanos("0.999999999", Duration::positive(0, 999_999_999))]
#[case::leading_number_max_nanos("1.999999999", Duration::positive(1, 999_999_999))]
#[case::simple_number("1234.123456789", Duration::positive(1234, 123_456_789))]
#[case::max_seconds(&u64::MAX.to_string(), Duration::positive(u64::MAX, 0))]
#[case::leading_zeros("000000100", Duration::positive(100, 0))]
#[case::leading_zeros_with_fraction("00000010.0", Duration::positive(10, 0))]
#[case::negative_number_negative_exponent_below_attos("-9.99999999999E-19", Duration::ZERO)]
#[case::negative_large_input_with_high_exponent_is_zero("-.000000000000000000000000000000000000011888888888888888888888888888880000000000003333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333888888888888888888888888888888888800003333333333333333333333333333333333333333333333333333333333333333333333333338888888888888888833333333333333333333333333333333333333333333333338888888888888833333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333338888888888888888888888888888888888888888888888885888888000000000000333333333333333333333333333338883333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333888888888888888888888888888888888888888888888888888888800000000000033333333333333333000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011414703819657838596", Duration::ZERO)]
#[case::f64_max(&format!("{}", f64::MAX), Duration::MAX)]
fn test_parse_duration_when_simple_arguments_are_valid(
    #[case] source: &str,
    #[case] expected: Duration,
) {
    let parser = DurationParser::builder().default_time_units().build();
    assert_eq!(parser.parse(source), Ok(expected));

    let parser = DurationParser::builder()
        .default_time_units()
        .parse_multiple(Some(&["and"])) // cov:excl-line
        .build();
    assert_eq!(parser.parse(source), Ok(expected));
}

#[rstest]
#[case::seconds_overflow_when_negative_and_positive_exponent(&format!("-{}.11e1", u64::MAX as u128 * 100), Duration::MIN)]
#[case::minimum_exponent_with_minimum_time_unit(&format!("1{}.0e{}ns", "0".repeat((i16::MAX as usize) + 1), i16::MIN), Duration::positive(0, 1))]
#[case::seconds_overflow_when_positive_and_negative_exponent(&format!("{}e-1", u64::MAX as u128 * 100), Duration::MAX)]
#[case::seconds_overflow_when_positive_and_positive_exponent(&format!("{}.11e1", u64::MAX), Duration::MAX)]
#[case::seconds_overflow_when_negative_and_negative_exponent(&format!("-{}e-1", u64::MAX as u128 * 100), Duration::MIN)]
#[case::minus_sign_whole_to_fract("1.00000001e-1", Duration::positive(0, 100_000_001))]
#[case::zero("1.1e0", Duration::positive(1, 100_000_000))]
#[case::point_and_then_exponent("1.e0", Duration::positive(1, 0))]
#[case::negative_zero("1.1e-0", Duration::positive(1, 100_000_000))]
#[case::simple("1.09e1", Duration::positive(10, 900_000_000))]
#[case::simple_big_e("1.09E1", Duration::positive(10, 900_000_000))]
#[case::two_e4("2e4", Duration::positive(20000, 0))]
#[case::lower_than_nanos_min("0.0000000001e1", Duration::positive(0, 1))]
#[case::higher_than_seconds_max(&format!("{}9.999999999e-1", u64::MAX), Duration::MAX)]
#[case::plus_sign("0.1000000001e+1", Duration::positive(1, 1))]
#[case::minus_sign_zero_to_fract("10.00000001e-1", Duration::positive(1, 1))]
#[case::exponent_then_nineteen_zeros_in_fraction("1.0e-20", Duration::ZERO)]
#[case::no_overflow_error_low("1.0e-32768", Duration::ZERO)]
#[case::no_overflow_error_high("1.0e+32767", Duration::MAX)]
#[case::maximum_exponent(&format!("0.{}9e+{}", "0".repeat(i16::MAX as usize), i16::MAX), Duration::positive(0, 900_000_000))]
#[case::maximum_exponent_barely_not_zero(&format!(".{}1e{}", "0".repeat((i16::MAX as usize) + 8), i16::MAX), Duration::positive(0, 1))]
#[case::maximum_exponent_barely_not_zero_with_time_unit(&format!(".{}1e{}y", "0".repeat((i16::MAX as usize) + 15), i16::MAX), Duration::positive(0, 3))]
#[case::maximum_exponent_with_maximum_time_unit(&format!("0.{}9e+{}y", "0".repeat(i16::MAX as usize), i16::MAX), Duration::positive(28401840, 0))]
#[case::minimum_exponent(&format!("1{}.0e{}", "0".repeat(i16::MIN.unsigned_abs() as usize), i16::MIN), Duration::positive(1, 0))]
#[case::minimum_exponent_barely_not_max_duration(&format!("1{}.0e{}", "0".repeat((i16::MIN.unsigned_abs() as usize) + 19), i16::MIN), Duration::positive(10_000_000_000_000_000_000, 0))]
#[case::minimum_exponent_barely_not_max_duration_with_time_unit(&format!("1{}.0e{}ns", "0".repeat((i16::MIN.unsigned_abs() as usize) + 28), i16::MIN), Duration::positive(10_000_000_000_000_000_000, 0))]
fn test_parse_duration_when_arguments_contain_exponent(
    #[case] source: &str,
    #[case] expected: Duration,
) {
    let duration = DurationParser::with_all_time_units()
        .allow_negative(true)
        .parse(source)
        .unwrap();
    assert_eq!(duration, expected);
}

#[rstest]
#[case::exponent_without_mantissa("e1")]
#[case::no_number("1e")]
#[case::no_number_but_time_unit("6eMs")]
#[case::invalid_number("1e+F")]
#[case::exponent_overflow_error_high("1e32768")]
#[case::exponent_overflow_error_low("1e-32769")]
#[case::exponent_parse_i16_overflow_error(&format!("1e{}", i16::MIN as i32 - 1))]
fn test_parse_duration_when_arguments_with_illegal_exponent_then_error(#[case] source: &str) {
    let parser = DurationParser::builder().all_time_units().build();
    let result = parser.parse(source);
    assert!(result.is_err());
}

#[rstest]
#[case::no_rounding("1.99999999999999999", StdDuration::new(1, 999_999_999))]
#[case::high_value_no_swallow_fract(&format!("{}.1", u64::MAX), StdDuration::new(u64::MAX, 100_000_000) )]
fn test_parse_duration_when_precision_of_float_would_be_insufficient_then_still_parse_exact(
    #[case] source: &str,
    #[case] expected: StdDuration,
) {
    let duration = parse_duration(source).unwrap();
    assert_eq!(duration, expected);
}

#[rstest]
#[case::lower_than_min_nanos("1.0000000001", StdDuration::new(1, 0))]
#[case::max_digits_of_nanos("1.99999999999", StdDuration::new(1, 999_999_999))]
#[case::higher_than_max_seconds(&format!("{}", u64::MAX as u128 + 1), StdDuration::MAX)]
#[case::higher_than_max_seconds_with_fraction(&format!("{}.0", u64::MAX as u128 + 1), StdDuration::MAX)]
fn test_parse_duration_when_arguments_are_capped_then_max_duration_or_min_nanos(
    #[case] source: &str,
    #[case] expected: StdDuration,
) {
    let duration = parse_duration(source).unwrap();
    assert_eq!(duration, expected);
}

#[rstest]
#[case::plus_zero("+0", StdDuration::ZERO)]
#[case::plus_zero_with_fraction("+0.0", StdDuration::ZERO)]
#[case::minus_zero("-0", StdDuration::ZERO)]
#[case::minus_zero_with_fraction("-0.0", StdDuration::ZERO)]
#[case::plus_one_with_fraction("+1.0", StdDuration::new(1, 0))]
fn test_parse_duration_when_arguments_have_a_sign(
    #[case] source: &str,
    #[case] expected: StdDuration,
) {
    let duration = parse_duration(source).unwrap();
    assert_eq!(duration, expected);
}

#[rstest]
#[case::infinity_short("inf")]
#[case::infinity_short_with_sign("+inf")]
#[case::infinity_short_case_insensitive("iNf")]
#[case::infinity_long("infinity")]
#[case::infinity_long_with_sign("+infinity")]
#[case::infinity_long_case_insensitive("InfiNitY")]
fn test_parse_duration_when_arguments_are_infinity_values(#[case] source: &str) {
    let duration = parse_duration(source).unwrap();
    assert_eq!(duration, StdDuration::MAX);
}

#[rstest]
#[case::negative_infinity_short("-inf", ParseError::NegativeNumber)]
#[case::negative_infinity_long("-infinity", ParseError::NegativeNumber)]
#[case::infinity_long_trailing_invalid("infinityINVALID", ParseError::Syntax(8, "Expected end of input but found: 'INVALID'".to_string()))]
#[case::incomplete_infinity("infin", ParseError::Syntax(5, "Error parsing infinity: Premature end of input".to_string()))]
#[case::infinity_with_number("inf1.0", ParseError::Syntax(3, "Error parsing infinity: Invalid character '1'".to_string()))]
fn test_parse_duration_when_arguments_are_illegal_infinity_values_then_error(
    #[case] source: &str,
    #[case] expected: ParseError,
) {
    assert_eq!(parse_duration(source), Err(expected));
}

#[rstest]
#[case::max_attos_no_u64_overflow(&format!("0.{}y", "9".repeat(100)), Duration::positive(31557599, 999_999_999))]
#[case::seconds("1s", Duration::positive(1, 0))]
#[case::milli_seconds("1ms", Duration::positive(0, 1_000_000))]
#[case::milli_seconds("1000ms", Duration::positive(1, 0))]
#[case::micro_seconds("1Ms", Duration::positive(0, 1_000))]
#[case::micro_seconds("1e-3Ms", Duration::positive(0, 1))]
#[case::nano_seconds_time_unit("1ns", Duration::positive(0, 1))]
#[case::minutes_fraction("0.1m", Duration::positive(6, 0))]
#[case::minutes_negative_exponent("100.0e-3m", Duration::positive(6, 0))]
#[case::minutes_underflow("0.0000000001m", Duration::positive(0, 6))]
#[case::hours_underflow("0.000000000001h", Duration::positive(0, 3))]
#[case::years_underflow("0.0000000000000001y", Duration::positive(0, 3))]
#[case::minutes_overflow(&format!("{}m", u64::MAX), Duration::MAX)]
#[case::time_unit_overflow_with_fraction(&format!("{}.00001m", u64::MAX), Duration::MAX)]
#[case::negative_time_unit_overflow_with_fraction(&format!("-{}.00001m", u64::MAX), Duration::MIN)]
#[case::time_unit_overflow_with_overflowing_fraction(&format!("{}.1d", u64::MAX), Duration::MAX)]
#[case::negative_time_unit_overflow_with_overflowing_fraction(&format!("-{}.1d", u64::MAX), Duration::MIN)]
fn test_parse_duration_when_time_units_are_given(#[case] source: &str, #[case] expected: Duration) {
    let duration = DurationParser::with_all_time_units()
        .allow_negative(true)
        .parse(source)
        .unwrap();
    assert_eq!(duration, expected);
}

#[rstest]
#[case::seconds("1s", vec![TimeUnit::Second])]
#[case::hour("1h", vec![TimeUnit::Hour])]
fn test_parser_when_time_units(#[case] source: &str, #[case] time_units: Vec<TimeUnit>) {
    _ = DurationParser::with_time_units(&time_units)
        .parse(source)
        .unwrap();
}

#[rstest]
#[case::minute_short("1s", vec![TimeUnit::Minute])]
fn test_parser_when_time_units_are_not_present_then_error(
    #[case] source: &str,
    #[case] time_units: Vec<TimeUnit>,
) {
    assert!(
        DurationParser::with_time_units(time_units.as_slice())
            .parse(source)
            .is_err()
    );
}

#[rstest]
#[case::empty("", ParseError::Empty)]
#[case::only_space(" ", ParseError::InvalidInput(" ".to_string()))]
#[case::space_before_number(" 123", ParseError::InvalidInput(" 123".to_string()))]
#[case::space_at_end_of_input("123 ns ", ParseError::TimeUnit(4, "Invalid time unit: 'ns '".to_string()))]
#[case::other_whitespace("123\tns", ParseError::TimeUnit(3, "Invalid time unit: '\tns'".to_string()))]
fn test_parser_when_allow_time_unit_delimiter_then_error(
    #[case] input: &str,
    #[case] expected: ParseError,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .allow_time_unit_delimiter(true)
            .set_inner_delimiter(|byte| byte == b' ')
            .parse(input)
            .unwrap_err(),
        expected
    );
}

#[rstest]
#[case::without_delimiter("123ns", |b : u8| b.is_ascii_whitespace(),  Duration::positive(0, 123))]
#[case::all_rust_whitespace("123 \t\n\x0C\rns", |b : u8| b.is_ascii_whitespace(),  Duration::positive(0, 123))]
fn test_parser_when_allow_time_unit_delimiter(
    #[case] input: &str,
    #[case] delimiter: fundu::Delimiter,
    #[case] expected: Duration,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .allow_time_unit_delimiter(true)
            .set_inner_delimiter(delimiter)
            .parse(input)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::without_spaces("123ns", Duration::positive(0, 123))]
#[case::single_space("123 ns", Duration::positive(0, 123))]
#[case::multiple_spaces("123      ns", Duration::positive(0, 123))]
fn test_parser_when_allow_delimiter_then_ok(#[case] input: &str, #[case] expected: Duration) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .allow_time_unit_delimiter(true)
            .set_inner_delimiter(|byte| byte == b' ')
            .parse(input),
        Ok(expected)
    );
}

#[rstest]
#[case::nano_seconds("ns", Ok(Duration::positive(0, 1)))]
#[case::exponent_without_mantissa("e1", Err(ParseError::InvalidInput("e1".to_string())))]
#[case::just_point(".", Err(ParseError::Syntax(0, "Either the whole number part or the fraction must be present".to_string())))]
fn test_parser_when_number_is_optional(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .number_is_optional(true)
            .parse(input),
        expected
    );
}

#[rstest]
#[case::starts_with_delimiter("\rd", ParseError::InvalidInput("\rd".to_string()))]
fn test_parser_when_number_is_optional_and_allow_delimiter_then_error(
    #[case] input: &str,
    #[case] expected: ParseError,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .number_is_optional(true)
            .allow_time_unit_delimiter(true)
            .parse(input),
        Err(expected)
    );
}

#[rstest]
#[case::whole_with_just_point("1.", Err(ParseError::Syntax(1, "No fraction allowed".to_string())))]
#[case::fract_with_just_point(".1", Err(ParseError::Syntax(0, "No fraction allowed".to_string())))]
#[case::just_point(".", Err(ParseError::Syntax(0, "No fraction allowed".to_string())))]
fn test_parser_when_disable_fraction(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .disable_fraction(true)
            .parse(input),
        expected
    );
}

#[rstest]
#[case::whole_with_exponent("1e0", Err(ParseError::Syntax(1, "No exponent allowed".to_string())))]
#[case::fract_with_exponent("0.1e0", Err(ParseError::Syntax(3, "No exponent allowed".to_string())))]
#[case::exponent_without_number("1e", Err(ParseError::Syntax(1, "No exponent allowed".to_string())))]
fn test_parser_when_disable_exponent(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .disable_exponent(true)
            .parse(input),
        expected
    );
}

#[rstest]
#[case::whole_with_exponent("i", Err(ParseError::InvalidInput("i".to_string())))]
#[case::whole_with_exponent("in", Err(ParseError::InvalidInput("in".to_string())))]
#[case::whole_with_exponent("inf", Err(ParseError::InvalidInput("inf".to_string())))]
#[case::whole_with_exponent("+inf", Err(ParseError::InvalidInput("inf".to_string())))]
#[case::whole_with_exponent("-inf", Err(ParseError::InvalidInput("inf".to_string())))]
#[case::whole_with_exponent("infi", Err(ParseError::InvalidInput("infi".to_string())))]
#[case::whole_with_exponent("infinity", Err(ParseError::InvalidInput("infinity".to_string())))]
fn test_parser_when_disable_infinity(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    assert_eq!(
        DurationParser::with_all_time_units()
            .disable_infinity(true)
            .parse(input),
        expected
    );
}

#[rstest]
#[case::minute_short("1s", TimeUnit::Minute)]
fn test_parser_when_custom_time_unit_then_error(#[case] source: &str, #[case] time_unit: TimeUnit) {
    assert!(
        DurationParser::with_time_units(&[time_unit])
            .parse(source)
            .is_err()
    );
}

#[rstest]
#[case::time_unit_error_when_no_time_units(
    DurationParser::without_time_units(),
    "1y",
    ParseError::TimeUnit(1, "No time units allowed but found: 'y'".to_string()),
    "Time unit error: No time units allowed but found: 'y' at column 1"
)]
#[case::time_unit_error_when_all_time_units(
    DurationParser::with_all_time_units(),
    "1years",
    ParseError::TimeUnit(1, "Invalid time unit: 'years'".to_string()),
    "Time unit error: Invalid time unit: 'years' at column 1"
)]
#[case::negative_exponent_overflow_error(
    DurationParser::new(),
    "1e-32769",
    ParseError::NegativeExponentOverflow,
    "Negative exponent overflow: Minimum is -32768"
)]
#[case::positive_exponent_overflow_error(
    DurationParser::new(),
    "1e+32768",
    ParseError::PositiveExponentOverflow,
    "Positive exponent overflow: Maximum is +32767"
)]
#[case::negative_number_error(
    DurationParser::new(),
    "-1",
    ParseError::NegativeNumber,
    "Number was negative"
)]
#[case::negative_number_when_infinity_error(
    DurationParser::new(),
    "-inf",
    ParseError::NegativeNumber,
    "Number was negative"
)]
fn test_parse_error_messages(
    #[case] parser: DurationParser,
    #[case] input: &str,
    #[case] expected_error: ParseError,
    #[case] expected_string: &str,
) {
    assert_eq!(parser.parse(input), Err(expected_error));
    assert_eq!(
        parser.parse(input).unwrap_err().to_string(),
        expected_string
    );
}

#[rstest]
#[case::nano_second(NanoSecond, Duration::positive(0, 1))]
#[case::micro_second(MicroSecond, Duration::positive(0, 1_000))]
#[case::milli_second(MilliSecond, Duration::positive(0, 1_000_000))]
#[case::second(Second, Duration::positive(1, 0))]
#[case::minute(Minute, Duration::positive(60, 0))]
#[case::hour(Hour, Duration::positive(60 * 60, 0))]
#[case::day(Day, Duration::positive(60 * 60 * 24, 0))]
#[case::week(Week, Duration::positive(60 * 60 * 24 * 7, 0))]
#[case::month(Month, Duration::positive(MONTH, 0))]
#[case::year(Year, Duration::positive(YEAR, 0))]
fn test_parser_setting_default_time_unit(#[case] time_unit: TimeUnit, #[case] expected: Duration) {
    assert_eq!(
        DurationParser::without_time_units()
            .default_unit(time_unit)
            .parse("1")
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::zero_zero("0 0", Duration::ZERO)]
#[case::zero_conjunction_zero("0 and 0", Duration::ZERO)]
#[case::many_whitespace("0 \t\n\r  0", Duration::ZERO)]
#[case::zero_one("0 1", Duration::positive(1, 0))]
#[case::zero_one_conjunction("0 and 1", Duration::positive(1, 0))]
#[case::one_zero("1 0", Duration::positive(1, 0))]
#[case::one_zero_conjunction("1 and 0", Duration::positive(1, 0))]
#[case::one_one("1 1", Duration::positive(2, 0))]
#[case::one_one_conjunction("1 and 1", Duration::positive(2, 0))]
#[case::one_one("1 1", Duration::positive(2, 0))]
#[case::one_one_with_sign_as_delimiter("1+1", Duration::positive(2, 0))]
#[case::two_with_time_units("1ns 1ns", Duration::positive(0, 2))]
#[case::two_with_time_units_with_sign("1ns+1ns", Duration::positive(0, 2))]
#[case::two_with_time_units_conjunction("1ns and 1ns", Duration::positive(0, 2))]
#[case::two_with_time_units_conjunction_and_sign("1ns and+1ns", Duration::positive(0, 2))]
#[case::two_with_time_units_without_delimiter("1ns1ns", Duration::positive(0, 2))]
#[case::two_with_fraction_exponent_time_units(
    "1.123e9ns 1.987e9ns",
    Duration::positive(3, 110_000_000)
)]
#[case::two_with_fraction_exponent_time_units_conjunction(
    "1.123e9ns and 1.987e9ns",
    Duration::positive(3, 110_000_000)
)]
#[case::two_when_saturating(&format!("{0}s {0}s", u64::MAX), Duration::MAX)]
#[case::multiple_mixed("1ns 1.001Ms1e1ms 9 .9 3m6h", Duration::positive(21789, 910_001_002))]
#[case::multiple_mixed_with_sign_as_delimiter(
    "1ns+1.001Ms-1e1+9-.9+3m6h",
    Duration::positive(21778, 100_001_002)
)]
#[case::multiple_mixed_with_conjunction(
    "1ns and 1.001Ms and1e1ms and 9 .9 and 3m6h",
    Duration::positive(21789, 910_001_002)
)]
#[case::single_infinity_short("inf", Duration::MAX)]
#[case::single_infinity_long("infinity", Duration::MAX)]
#[case::multiple_infinity_short("inf inf", Duration::MAX)]
#[case::multiple_infinity_long("infinity infinity", Duration::MAX)]
#[case::multiple_infinity_mixed_short_then_long("inf infinity", Duration::MAX)]
#[case::multiple_infinity_mixed_long_then_short("infinity inf", Duration::MAX)]
#[case::multiple_infinity_short_conjunction("inf and inf", Duration::MAX)]
#[case::multiple_infinity_long_conjunction("infinity and infinity", Duration::MAX)]
fn test_parser_when_setting_parse_multiple(#[case] input: &str, #[case] expected: Duration) {
    let parser = DurationParser::builder()
        .all_time_units()
        .parse_multiple(Some(&["and"]))
        .allow_negative()
        .build();
    assert_eq!(parser.parse(input), Ok(expected))
}

#[rstest]
#[case::empty("", ParseError::Empty)]
#[case::only_conjunction("and", ParseError::InvalidInput("and".to_string()))]
#[case::only_whitespace(" \t\n", ParseError::InvalidInput(" \t\n".to_owned()))]
#[case::just_point(".", ParseError::Syntax(0, "Either the whole number part or the fraction must be present".to_string()))]
#[case::just_time_unit("ns", ParseError::InvalidInput("ns".to_string()))]
#[case::valid_then_invalid("1 a", ParseError::InvalidInput("a".to_string()))]
#[case::end_with_space("1 1 ", ParseError::Syntax(3, "Input may not end with a delimiter".to_string()))]
#[case::end_with_conjunction("1 and", ParseError::Syntax(2, "Input may not end with a conjunction but found: 'and'".to_string()))]
#[case::end_with_wrong_conjunction("1 anda", ParseError::Syntax(5, "A conjunction must be separated by a delimiter, sign or digit but found: 'a'".to_string()))]
#[case::end_with_conjunction_and_delimiter("1 and ", ParseError::Syntax(5, "Input may not end with a delimiter".to_string()))]
#[case::valid_time_unit_end_with_delimiter("1s ", ParseError::Syntax(2, "Input may not end with a delimiter".to_string()))]
#[case::two_end_with_conjunction("1 1 and", ParseError::Syntax(4, "Input may not end with a conjunction but found: 'and'".to_string()))]
#[case::invalid_then_valid("a 1", ParseError::InvalidInput("a 1".to_string()))]
#[case::multiple_invalid("a a", ParseError::InvalidInput("a a".to_string()))]
#[case::infinity_then_space("inf ", ParseError::Syntax(3, "Input may not end with a delimiter".to_string()))]
#[case::infinity_then_conjunction("inf and", ParseError::Syntax(4, "Input may not end with a conjunction but found: 'and'".to_string()))]
#[case::infinity_short_then_number("inf1", ParseError::Syntax(3, "Error parsing infinity: Invalid character '1'".to_string()))]
#[case::infinity_long_then_number("infinity1", ParseError::Syntax(8, "Error parsing infinity: Expected a delimiter but found '1'".to_string()))]
#[case::premature_end_parsing_infinity("infi", ParseError::Syntax(0, "Error parsing infinity: 'infi' is an invalid identifier for infinity".to_string()))]
fn test_parser_when_setting_parse_multiple_then_error(
    #[case] input: &str,
    #[case] expected: ParseError,
) {
    let parser = DurationParser::builder()
        .all_time_units()
        .parse_multiple(Some(&["and"]))
        .build();
    assert_eq!(parser.parse(input), Err(expected))
}

#[rstest]
#[case::numbers_without_time_units("1 ns 1 s", Ok(Duration::positive(1, 1)))]
#[case::sign_without_number("1s + 1s", Err(ParseError::InvalidInput(" 1s".to_owned())))]
fn test_parser_when_parse_multiple_number_is_optional_allow_delimiter(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    let parser = DurationParser::builder()
        .all_time_units()
        .parse_multiple(Some(&["and"]))
        .number_is_optional()
        .allow_time_unit_delimiter()
        .build();
    assert_eq!(parser.parse(input), expected)
}

#[rstest]
#[case::just_sign(
    "+",
    Err(ParseError::Syntax(1, "Unexpected end of input. Sign without a number".to_owned()))
)]
#[case::just_sign_end_with_delimiter(
    "+   ",
    Err(ParseError::Syntax(1, "Input may not end with a delimiter".to_owned()))
)]
#[case::sign_without_delimiter("+1", Ok(Duration::positive(1, 0)))]
#[case::sign_with_delimiter("+    1", Ok(Duration::positive(1, 0)))]
#[case::multiple_sign_with_delimiter("+2 - 1", Ok(Duration::positive(1, 0)))]
#[case::multiple_sign_end_with_sign(
    "+2 - 1 +",
    Err(ParseError::Syntax(8, "Unexpected end of input. Sign without a number".to_owned()))
)]
#[case::multiple_sign_end_with_delimiter(
    "+2 - 1 + ",
    Err(ParseError::Syntax(8, "Input may not end with a delimiter".to_owned()))
)]
fn test_parser_when_parse_multiple_with_sign_delimiter(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    let parser = DurationParser::builder()
        .all_time_units()
        .parse_multiple(Some(&["and"]))
        .allow_time_unit_delimiter()
        .allow_sign_delimiter()
        .allow_negative()
        .build();
    assert_eq!(parser.parse(input), expected)
}

#[test]
fn test_parser_when_parse_multiple_number_is_optional_not_allow_delimiter() {
    let parser = DurationParser::builder()
        .all_time_units()
        .parse_multiple(Some(&["and"]))
        .number_is_optional()
        .build();
    assert_eq!(parser.parse("1 ns 1 s"), Ok(Duration::positive(3, 1)))
}

#[test]
fn test_parser_when_parse_multiple_with_invalid_delimiter() {
    let delimiter = |byte: u8| byte == 0xb5;
    let parser = CustomDurationParser::builder()
        .time_unit(CustomTimeUnit::with_default(MicroSecond, &["µ"]))
        .parse_multiple(None)
        .outer_delimiter(delimiter)
        .build();

    // The delimiter will split the multibyte µ and produces invalid utf-8
    // µ = 0xc2 0xb5
    assert_eq!(
        parser.parse("1µ"),
        Err(ParseError::InvalidInput("µ".to_string()))
    )
}

#[test]
fn test_parser_when_allow_ago_with_invalid_delimiter() {
    let delimiter = |byte: u8| byte == 0xb5;
    let parser = CustomDurationParser::builder()
        .time_unit(CustomTimeUnit::with_default(MicroSecond, &["µ"]))
        .allow_ago()
        .inner_delimiter(delimiter)
        .build();

    // The delimiter will split the multibyte µ and produces invalid utf-8
    // µ = 0xc2 0xb5
    assert!(parser.parse("1µ ago").is_err());
}

#[test]
fn test_parser_parse_multiple_and_keywords_with_invalid_delimiter() {
    let delimiter = |byte: u8| byte == 0xb5;
    let parser = CustomDurationParser::builder()
        .time_unit(CustomTimeUnit::with_default(MicroSecond, &["µ"]))
        .keyword(TimeKeyword::new(MicroSecond, &["manµ"], None))
        .parse_multiple(None)
        .outer_delimiter(delimiter)
        .build();

    // The delimiter will split the multibyte µ and produces invalid utf-8
    // µ = 0xc2 0xb5
    assert_eq!(
        parser.parse("someµ"),
        Err(ParseError::InvalidInput("someµ".to_string()))
    )
}

#[rstest]
#[case::only_numbers("1 1", Ok(Duration::positive(2, 0)))]
#[case::with_time_units("1ns 1ns", Err(ParseError::InvalidInput("ns 1ns".to_string())))]
#[case::number_then_with_time_unit("1 1ns", Err(ParseError::InvalidInput("ns".to_string())))]
#[case::ago_without_time_unit("1 ago", Err(ParseError::InvalidInput("ago".to_string())))]
fn test_parser_when_parse_multiple_without_time_units(
    #[case] input: &str,
    #[case] expected: Result<Duration, ParseError>,
) {
    let parser = CustomDurationParser::builder()
        .parse_multiple(Some(&["and"]))
        .allow_ago()
        .build();
    assert_eq!(parser.parse(input), expected);
}

#[test]
fn test_parser_when_parse_multiple_without_time_units_default_unit_is_not_seconds() {
    let parser = DurationParser::builder()
        .parse_multiple(Some(&["and"]))
        .default_unit(NanoSecond)
        .build();
    assert_eq!(parser.parse("1 1"), Ok(Duration::positive(0, 2)));
}

#[rstest]
#[case::nano_second(&["ns", "nsec"], Duration::positive(0, 1))]
#[case::micro_second(&["us", "µs", "usec"], Duration::positive(0, 1000))]
#[case::milli_second(&["ms", "msec"], Duration::positive(0, 1_000_000))]
#[case::second(&["s", "sec", "second", "seconds"], Duration::positive(1, 0))]
#[case::minute(&["m", "min", "minute", "minutes"], Duration::positive(60, 0))]
#[case::hour(&["h", "hr", "hour", "hours"], Duration::positive(60 * 60, 0))]
#[case::day(&["d", "day", "days"], Duration::positive(60 * 60 * 24, 0))]
#[case::week(&["w", "week", "weeks"], Duration::positive(60 * 60 * 24 * 7, 0))]
#[case::month(&["M", "month", "months"], Duration::positive(MONTH, 0))]
#[case::year(&["y", "year", "years"], Duration::positive(YEAR, 0))]
fn test_custom_duration_parser_parse_when_systemd_time_units(
    #[case] inputs: &[&str],
    #[case] expected: Duration,
) {
    let parser = CustomDurationParser::with_time_units(&SYSTEMD_TIME_UNITS);
    for input in inputs {
        assert_eq!(parser.parse(&format!("1{input}")), Ok(expected));
    }
}

#[rstest]
#[case::negative_zero("-0", Duration::ZERO)]
#[case::negative_barely_not_zero("-0.000000001", Duration::negative(0, 1))]
#[case::negative_barely_zero("-0.0000000001", Duration::negative(0, 0))] // FIXME: should be Duration::ZERO
#[case::positive_zero("+0", Duration::ZERO)]
#[case::positive_barely_not_zero("0.000000001", Duration::positive(0, 1))]
#[case::positive_barely_zero("0.0000000001", Duration::ZERO)]
#[case::zero_without_sign("0", Duration::ZERO)]
#[case::negative_one("-1", Duration::negative(1, 0))]
#[case::negative_one_with_fraction("-1.2", Duration::negative(1, 200_000_000))]
#[case::negative_one_with_exponent("-1e-9", Duration::negative(0, 1))]
#[case::negative_min_seconds(&format!("-{}", u64::MAX), Duration::negative(u64::MAX, 0))]
#[case::negative_min_seconds_low_nanos(&format!("-{}.000000001", u64::MAX), Duration::negative(u64::MAX, 1))]
#[case::negative_min_seconds_and_nanos(&format!("-{}.999999999", u64::MAX), Duration::MIN)]
#[case::negative_min_nanos_10_digits(&format!("-{}.9999999999", u64::MAX), Duration::MIN)]
#[case::negative_seconds_barely_saturate(&format!("-{}", u64::MAX as u128 + 1), Duration::MIN)]
#[case::negative_some_mixed_number(
    "-1122334455.123456789e-4",
    Duration::negative(112233, 445512345)
)]
#[case::negative_years("-1.000000001y", Duration::negative(YEAR, YEAR as u32))]
#[case::negative_high_value_saturate(&format!("-{}.{}e1000", "1".repeat(1000), "1".repeat(1000)), Duration::MIN)]
#[case::positive_one("1", Duration::positive(1, 0))]
#[case::positive_max_seconds(&format!("{}", u64::MAX), Duration::positive(u64::MAX, 0))]
#[case::positive_max_seconds_low_nanos(&format!("{}.000000001", u64::MAX), Duration::positive(u64::MAX, 1))]
#[case::positive_max_seconds_and_nanos(&format!("{}.999999999", u64::MAX), Duration::MAX)]
#[case::positive_seconds_barely_saturate(&format!("{}", u64::MAX as u128 + 1), Duration::MAX)]
#[case::positive_seconds_and_nanos_barely_saturate(&format!("{}.000000001", u64::MAX as u128 + 1), Duration::MAX)]
#[case::positive_some_mixed_number(
    "1122334455.123456789e-4",
    Duration::positive(112233, 445512345)
)]
#[case::positive_years("1.000000001y", Duration::positive(YEAR, YEAR as u32))]
#[case::positive_high_value_saturate(&format!("{}.{}e1000", "1".repeat(1000), "1".repeat(1000)), Duration::MAX)]
#[case::time_unit_causes_saturate_negative(&format!("-{}y", u64::MAX), Duration::MIN)]
fn test_parse_negative(#[case] source: &str, #[case] expected: Duration) {
    let actual = DurationParser::with_all_time_units()
        .allow_negative(true)
        .parse(source)
        .unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[case::simple_zero("0", Duration::ZERO)]
#[case::two_zero("0 0", Duration::ZERO)]
#[case::two_one("1 1", Duration::positive(2, 0))]
#[case::negative_and_positive_then_zero("-1 1", Duration::ZERO)]
#[case::two_negative("-1 -1", Duration::negative(2, 0))]
#[case::two_negative_conjunction("-1 and -1", Duration::negative(2, 0))]
#[case::two_negative_with_time_units("-1ns -1s", Duration::negative(1, 1))]
#[case::two_negative_with_time_units_conjunction("-1ns and -1s", Duration::negative(1, 1))]
#[case::negative_and_positive_with_time_units("1ns -1s", Duration::negative(0, 999_999_999))]
#[case::two_negative_mixed("-1.1ms -1e-9s", Duration::negative(0, 1_100_001))]
#[case::two_negative_saturate_negative(&format!("-{}s -1s", u64::MAX), Duration::MIN)]
#[case::two_positive_saturate_positive(&format!("{}s 1s", u64::MAX), Duration::MAX)]
#[case::negative_infinity_short("-inf", Duration::MIN)]
#[case::two_negative_infinity_short_then_saturate("-inf -inf", Duration::MIN)]
#[case::negative_infinity_long("-infinity", Duration::MIN)]
#[case::two_negative_infinity_long_then_saturate("-infinity -infinity", Duration::MIN)]
#[case::two_negative_infinity_long_conjunction_then_saturate(
    "-infinity and -infinity",
    Duration::MIN
)]
#[case::two_positive_infinity_long_then_saturate_max("infinity infinity", Duration::MAX)]
#[case::negative_infinity_and_positive_infinity_no_error("-inf +inf", Duration::ZERO)]
#[case::negative_infinity_and_positive_infinity_conjunction("-inf and +inf", Duration::ZERO)]
fn test_parse_negative_when_multiple(#[case] input: &str, #[case] expected: Duration) {
    assert_eq!(
        DurationParser::builder()
            .all_time_units()
            .parse_multiple(Some(&["and"]))
            .allow_negative()
            .build()
            .parse(input)
            .unwrap(),
        expected
    );
}

#[rstest]
#[case::i_without_number("i", "i")]
#[case::big_i_without_number("I", "I")]
#[case::simple_plus_i("i", "+i")]
#[case::simple_i_one("i", "1i")]
#[case::simple_i_with_fraction("i", "1.0i")]
#[case::simple_i_with_exponent("i", "1.0e0i")]
#[case::simple_inf("inf", "inf")]
#[case::infi("infi", "infi")]
#[case::infinity("infinity", "infinity")]
#[case::one_infinity("infinity", "1infinity")]
fn test_custom_parser_when_disable_infinity_then_no_problems_with_infinity_like_ids(
    #[case] id: &str,
    #[case] input: &str,
) {
    let ids = &[id];
    let parser = CustomDurationParserBuilder::new()
        .disable_infinity()
        .number_is_optional()
        .time_units(&[CustomTimeUnit::with_default(NanoSecond, ids)])
        .build();
    assert_eq!(parser.parse(input), Ok(Duration::positive(0, 1)));
}

#[rstest]
#[case::positive_zero("0s", Duration::ZERO)]
#[case::negative_zero("-0s", Duration::ZERO)]
#[case::positive_second("1s", Duration::negative(1, 0))]
#[case::negative_second("-1s", Duration::positive(1, 0))]
#[case::positive_two_seconds("2s", Duration::negative(2, 0))]
#[case::negative_two_seconds("-2s", Duration::positive(2, 0))]
#[case::positive_fraction("0.1s", Duration::negative(0, 100_000_000))]
#[case::negative_fraction("-0.1s", Duration::positive(0, 100_000_000))]
#[case::positive_overflow(&format!("{}s", u128::MAX), Duration::MIN)]
#[case::negative_overflow(&format!("-{}s", u128::MAX), Duration::MAX)]
fn test_custom_parser_when_negative_multiplier(#[case] input: &str, #[case] expected: Duration) {
    let parser = CustomDurationParserBuilder::new()
        .time_unit(CustomTimeUnit::new(Second, &["s"], Some(Multiplier(-1, 0))))
        .allow_negative()
        .build();
    let actual = parser.parse(input).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[case::positive_zero("0s")]
#[case::negative_zero("-0s")]
#[case::one("1s")]
#[case::some("1234567.89s")]
#[case::max(&format!("{}s", u64::MAX))]
#[case::min(&format!("-{}s", u64::MAX))]
fn test_custom_parser_when_multiplier_is_zero(#[case] input: &str) {
    let parser = CustomDurationParserBuilder::new()
        .time_unit(CustomTimeUnit::new(Second, &["s"], Some(Multiplier(0, 0))))
        .allow_negative()
        .build();
    assert_eq!(parser.parse(input), Ok(Duration::ZERO));
}

#[rstest]
#[case::yesterday("yesterday", Duration::negative(60 * 60 * 24, 0))]
#[case::negative_yesterday("-yesterday", Duration::positive(60 * 60 * 24, 0))]
#[case::positive_yesterday("+yesterday", Duration::negative(60 * 60 * 24, 0))]
#[case::tomorrow("tomorrow", Duration::positive(60 * 60 * 24, 0))]
#[case::negative_tomorrow("-tomorrow", Duration::negative(60 * 60 * 24, 0))]
#[case::today("today", Duration::ZERO)]
#[case::negative_today("-today", Duration::ZERO)]
fn test_custom_parser_with_keywords(#[case] input: &str, #[case] expected: Duration) {
    let parser = CustomDurationParserBuilder::new()
        .keyword(TimeKeyword::new(
            Day,
            &["yesterday"],
            Some(Multiplier(-1, 0)),
        ))
        .keyword(TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))))
        .keyword(TimeKeyword::new(Day, &["today"], Some(Multiplier(0, 0))))
        .time_units(&[
            CustomTimeUnit::with_default(NanoSecond, &["ns"]),
            CustomTimeUnit::with_default(Second, &["s", "second"]),
        ])
        .allow_negative()
        .build();
    let actual = parser.parse(input).unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[case::leading_space(" tomorrow", ParseError::InvalidInput(" tomorrow".to_string()))]
#[case::trailing_space("tomorrow ", ParseError::InvalidInput("tomorrow ".to_string()))]
#[case::incomplete_keyword("tomorro", ParseError::InvalidInput("tomorro".to_string()))]
#[case::number("1tomorrow", ParseError::TimeUnit(1, "Invalid time unit: 'tomorrow'".to_string()))]
#[case::number_with_space("1 tomorrow", ParseError::TimeUnit(1, "Invalid time unit: ' tomorrow'".to_string()))]
fn test_custom_parser_with_keywords_then_error(#[case] input: &str, #[case] expected: ParseError) {
    let parser = CustomDurationParserBuilder::new()
        .keyword(TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))))
        .time_units(&[
            CustomTimeUnit::with_default(NanoSecond, &["ns"]),
            CustomTimeUnit::with_default(Second, &["s", "second"]),
        ])
        .build();
    assert_eq!(parser.parse(input), Err(expected));
}

#[rstest]
#[case::single_tomorrow("tomorrow", Duration::positive(60 * 60 * 24, 0))]
#[case::tomorrow_then_number("tomorrow1.1", Duration::positive(60 * 60 * 24 + 1, 100_000_000))]
#[case::two_tomorrow("tomorrow tomorrow", Duration::positive(60 * 60 * 24 * 2, 0))]
#[case::yesterday_tomorrow("yesterday tomorrow", Duration::ZERO)]
#[case::number_tomorrow("1 tomorrow", Duration::positive(60 * 60 * 24 + 1, 0))]
#[case::fraction_tomorrow(".1 tomorrow", Duration::positive(60 * 60 * 24, 100_000_000))]
#[case::number_conjunction_keyword(".1 and tomorrow", Duration::positive(60 * 60 * 24, 100_000_000))]
#[case::keyword_conjunction_number("tomorrow and .1", Duration::positive(60 * 60 * 24, 100_000_000))]
#[case::keyword_conjunction_no_delimiter_number("tomorrow and1.1", Duration::positive(60 * 60 * 24 + 1, 100_000_000))]
fn test_custom_parser_with_keywords_when_parse_multiple(
    #[case] input: &str,
    #[case] expected: Duration,
) {
    let parser = CustomDurationParserBuilder::new()
        .keyword(TimeKeyword::new(
            Day,
            &["yesterday"],
            Some(Multiplier(-1, 0)),
        ))
        .keyword(TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))))
        .time_units(&[
            CustomTimeUnit::with_default(NanoSecond, &["ns"]),
            CustomTimeUnit::with_default(Second, &["s", "second"]),
        ])
        .parse_multiple(Some(&["and"]))
        .allow_negative()
        .build();
    assert_eq!(parser.parse(input), Ok(expected));
}

#[rstest]
#[case::nano_second("ns", Duration::positive(0, 1))]
#[case::two_times_nano_second("ns ns", Duration::positive(0, 2))]
#[case::two_times_nano_second_with_conjunction("ns and ns", Duration::positive(0, 2))]
#[case::tomorrow_nano_second("tomorrow ns", Duration::positive(60 * 60 * 24, 1))]
#[case::tomorrow_conjunction_nano_second("tomorrow and ns", Duration::positive(60 * 60 * 24, 1))]
#[case::nano_second_tomorrow("ns tomorrow", Duration::positive(60 * 60 * 24, 1))]
#[case::nano_second_conjunction_tomorrow("ns and tomorrow", Duration::positive(60 * 60 * 24, 1))]
#[case::one_tomorrow_nano_second("1 tomorrow ns", Duration::positive(60 * 60 * 24 + 1, 1))]
#[case::one_tomorrow_nano_second_with_conjunctions("1 and tomorrow and ns", Duration::positive(60 * 60 * 24 + 1, 1))]
fn test_custom_parser_with_keywords_when_parse_multiple_and_number_is_optional(
    #[case] input: &str,
    #[case] expected: Duration,
) {
    let parser = CustomDurationParserBuilder::new()
        .keyword(TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))))
        .time_units(&[
            CustomTimeUnit::with_default(NanoSecond, &["ns"]),
            CustomTimeUnit::with_default(Second, &["s", "second"]),
        ])
        .parse_multiple(Some(&["and"]))
        .number_is_optional()
        .build();
    assert_eq!(parser.parse(input), Ok(expected));
}

#[rstest]
#[case::zero(Multiplier(0, 0), Duration::ZERO)]
#[case::max_exponent(Multiplier(1, i16::MAX), Duration::MAX)]
#[case::min_exponent(Multiplier(1, i16::MIN), Duration::ZERO)]
#[case::high_exponent(Multiplier(1, 19), Duration::positive(10_000_000_000_000_000_000, 0))]
#[case::low_exponent(Multiplier(1, -19), Duration::ZERO)]
#[case::barely_lower_than_nano(Multiplier(1, -10), Duration::ZERO)]
#[case::high_exponent_then_saturate_max(Multiplier(1, 20), Duration::MAX)]
#[case::high_exponent_then_saturate_min(Multiplier(-1, 20), Duration::MIN)]
#[case::low_exponent_then_zero(Multiplier(1, -18), Duration::ZERO)]
#[case::some_positive_multiplier(
    Multiplier(1234567, 8),
    Duration::positive(123_456_700_000_000, 0)
)]
#[case::some_negative_multiplier(
    Multiplier(-1234567, 8),
    Duration::negative(123_456_700_000_000, 0)
)]
fn test_custom_parser_multiplier_when_number_is_optional_and_time_unit_without_number(
    #[case] multiplier: Multiplier,
    #[case] expected: Duration,
) {
    let parser = CustomDurationParserBuilder::new()
        .time_unit(CustomTimeUnit::new(Second, &["my"], Some(multiplier)))
        .number_is_optional()
        .allow_negative()
        .build();

    assert_eq!(parser.parse("my"), Ok(expected));
}

#[test]
fn test_custom_parser_with_negative_keyword_when_not_allow_negative_then_error() {
    let parser = CustomDurationParserBuilder::new()
        .keyword(TimeKeyword::new(
            Day,
            &["yesterday"],
            Some(Multiplier(-1, 0)),
        ))
        .build();
    assert_eq!(parser.parse("yesterday"), Err(ParseError::NegativeNumber));
}

#[rstest]
#[case::one("1", Duration::positive(1, 0))]
#[case::second("1s", Duration::positive(1, 0))]
#[case::seconds_ago("1s ago", Duration::negative(1, 0))]
#[case::seconds_ago_big("1s AGO", Duration::negative(1, 0))]
#[case::seconds_ago_mixed("1s aGO", Duration::negative(1, 0))]
#[case::negative_seconds_ago("-1s ago", Duration::positive(1, 0))]
#[case::day_ago("1day ago", Duration::negative(60 * 60 * 24, 0))]
#[case::with_delimiter_between_number_and_time_unit("1 s ago", Duration::negative(1, 0))]
fn test_custom_parser_with_allow_ago(#[case] input: &str, #[case] expected: Duration) {
    let parser = CustomDurationParserBuilder::new()
        .time_units(&SYSTEMD_TIME_UNITS)
        .allow_time_unit_delimiter()
        .allow_ago()
        .allow_negative()
        .build();
    assert_eq!(parser.parse(input), Ok(expected));
}

#[rstest]
#[case::ago_without_time_unit("1 ago", ParseError::TimeUnit(2, "Invalid time unit: 'ago'".to_string()))]
#[case::ago_as_time_unit("1 ago", ParseError::TimeUnit(2, "Invalid time unit: 'ago'".to_string()))]
#[case::just_ago("ago", ParseError::InvalidInput("ago".to_string()))]
#[case::incomplete_ago("1s ag", ParseError::TimeUnit(1, "Invalid time unit: 's ag'".to_string()))]
#[case::one_second_twice("1s 1s", ParseError::TimeUnit(1, "Invalid time unit: 's 1s'".to_string()))]
fn test_custom_parser_with_allow_ago_then_error(#[case] input: &str, #[case] expected: ParseError) {
    let parser = CustomDurationParserBuilder::new()
        .time_units(&SYSTEMD_TIME_UNITS)
        .allow_time_unit_delimiter()
        .allow_ago()
        .allow_negative()
        .build();
    assert_eq!(parser.parse(input), Err(expected));
}

#[rstest]
#[case::seconds_without_ago("1s", Duration::positive(1, 0))]
#[case::seconds("1s ago", Duration::negative(1, 0))]
#[case::twice_seconds("1s ago 1s ago", Duration::negative(2, 0))]
#[case::seconds_and_day("1s ago 1day ago", Duration::negative(60 * 60 * 24 + 1, 0))]
#[case::two_without_delimiter_between_multiple("1s ago1s ago", Duration::negative(2, 0))]
#[case::two_with_delimiter_between_number_and_time_unit(
    "1 s ago 1 day ago",
    Duration::negative(60 * 60 * 24 + 1, 0)
)]
#[case::two_with_conjunction_between_number_and_time_unit(
    "1 s ago and 1 day ago",
    Duration::negative(60 * 60 * 24 + 1, 0)
)]
#[case::without_ago_conjunction_and_with_ago(
    "1 s and 1 day ago",
    Duration::negative(60 * 60 * 24 - 1, 0)
)]
#[case::without_ago_and_with_ago(
    "1 s 1 day ago",
    Duration::negative(60 * 60 * 24 - 1, 0)
)]
#[case::with_ago_and_without_ago(
    "1s ago 1 day",
    Duration::positive(60 * 60 * 24 - 1, 0)
)]
#[case::with_ago_conjunction_without_ago(
    "1s ago and 1 day",
    Duration::positive(60 * 60 * 24 - 1, 0)
)]
fn test_custom_parser_with_allow_ago_when_parse_multiple(
    #[case] input: &str,
    #[case] expected: Duration,
) {
    let parser = CustomDurationParserBuilder::new()
        .time_units(&SYSTEMD_TIME_UNITS)
        .allow_time_unit_delimiter()
        .allow_ago()
        .allow_negative()
        .parse_multiple(Some(&["and"]))
        .build();
    assert_eq!(parser.parse(input), Ok(expected));
}
