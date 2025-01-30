// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use fundu_core::config::{Config, Delimiter};
use fundu_core::parse::Parser;
use fundu_core::time::TimeUnit;

use super::time_units::TimeUnits;
use crate::DurationParser;

#[derive(Debug, PartialEq, Eq)]
enum TimeUnitsChoice<'a> {
    Default,
    All,
    None,
    Custom(&'a [TimeUnit]),
}

/// An ergonomic builder for a [`DurationParser`].
///
/// The [`DurationParserBuilder`] is more ergonomic in some use cases than using [`DurationParser`]
/// directly, especially when using the `DurationParser` for parsing multiple inputs. This builder
/// can also be used in `const` context, so it's possible to create a configured [`DurationParser`]
/// at compilation time.
///
/// # Examples
///
/// ```rust
/// use fundu::TimeUnit::*;
/// use fundu::{Duration, DurationParser, DurationParserBuilder};
///
/// let parser = DurationParserBuilder::new()
///     .all_time_units()
///     .default_unit(MicroSecond)
///     .allow_time_unit_delimiter()
///     .build();
///
/// assert_eq!(parser.parse("1   ns").unwrap(), Duration::positive(0, 1));
/// assert_eq!(parser.parse("1").unwrap(), Duration::positive(0, 1_000));
///
/// // instead of
///
/// let mut parser = DurationParser::with_all_time_units();
/// parser
///     .default_unit(MicroSecond)
///     .allow_time_unit_delimiter(true);
///
/// assert_eq!(parser.parse("1    ns").unwrap(), Duration::positive(0, 1));
/// assert_eq!(parser.parse("1").unwrap(), Duration::positive(0, 1_000));
/// ```
///
/// The builder in `const` context.
///
/// ```rust
/// use fundu::TimeUnit::*;
/// use fundu::{Duration, DurationParser, DurationParserBuilder};
///
/// const PARSER: DurationParser = DurationParserBuilder::new()
///     .time_units(&[Second, Minute, Hour, Day])
///     .allow_negative()
///     .parse_multiple(None)
///     .build();
/// assert_eq!(PARSER.parse("1h").unwrap(), Duration::positive(60 * 60, 0));
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct DurationParserBuilder<'a> {
    time_units_choice: TimeUnitsChoice<'a>,
    config: Config<'a>,
}

impl<'a> Default for DurationParserBuilder<'a> {
    /// Construct a new [`DurationParserBuilder`] without any time units.
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> DurationParserBuilder<'a> {
    /// Construct a new reusable [`DurationParserBuilder`].
    ///
    /// This method is the same like invoking [`DurationParserBuilder::default`]. Per default
    /// there are no time units configured in the builder. Use one of
    ///
    /// * [`DurationParserBuilder::default_time_units`]
    /// * [`DurationParserBuilder::all_time_units`]
    /// * [`DurationParserBuilder::time_units`]
    ///
    /// to add time units.
    ///
    /// # Examples
    #[cfg_attr(miri, doc = "```rust,ignore")]
    #[cfg_attr(not(miri), doc = "```rust")]
    /// use fundu::{DurationParser, DurationParserBuilder};
    ///
    /// assert_eq!(
    ///     DurationParserBuilder::new().build(),
    ///     DurationParser::without_time_units()
    /// );
    /// ```
    pub const fn new() -> Self {
        Self {
            time_units_choice: TimeUnitsChoice::None,
            config: Config::new(),
        }
    }

    /// Configure [`DurationParserBuilder`] to build the [`DurationParser`] with default time
    /// units.
    ///
    /// Setting the time units with this method overwrites any previous choices with
    ///
    /// * [`DurationParserBuilder::all_time_units`]
    /// * [`DurationParserBuilder::time_units`]
    ///
    /// The default time units with their identifiers are:
    ///
    /// | [`TimeUnit`]    | default id
    /// | --------------- | ----------:
    /// | Nanosecond  |         ns
    /// | Microsecond |         Ms
    /// | Millisecond |         ms
    /// | Second      |          s
    /// | Minute      |          m
    /// | Hour        |          h
    /// | Day         |          d
    /// | Week        |          w
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::DurationParserBuilder;
    /// use fundu::TimeUnit::*;
    ///
    /// assert_eq!(
    ///     DurationParserBuilder::new()
    ///         .default_time_units()
    ///         .build()
    ///         .get_current_time_units(),
    ///     vec![
    ///         NanoSecond,
    ///         MicroSecond,
    ///         MilliSecond,
    ///         Second,
    ///         Minute,
    ///         Hour,
    ///         Day,
    ///         Week
    ///     ]
    /// );
    /// ```
    pub const fn default_time_units(mut self) -> Self {
        self.time_units_choice = TimeUnitsChoice::Default;
        self
    }

    /// Configure [`DurationParserBuilder`] to build the [`DurationParser`] with all time units.
    ///
    /// Setting the time units with this method overwrites any previous choices with
    ///
    /// * [`DurationParserBuilder::default_time_units`]
    /// * [`DurationParserBuilder::time_units`]
    ///
    /// The time units with their identifiers are:
    ///
    /// | [`TimeUnit`]    | default id
    /// | --------------- | ----------:
    /// | Nanosecond  |         ns
    /// | Microsecond |         Ms
    /// | Millisecond |         ms
    /// | Second      |          s
    /// | Minute      |          m
    /// | Hour        |          h
    /// | Day         |          d
    /// | Week        |          w
    /// | Month       |          M
    /// | Year        |          y
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::DurationParserBuilder;
    /// use fundu::TimeUnit::*;
    ///
    /// assert_eq!(
    ///     DurationParserBuilder::new()
    ///         .all_time_units()
    ///         .build()
    ///         .get_current_time_units(),
    ///     vec![
    ///         NanoSecond,
    ///         MicroSecond,
    ///         MilliSecond,
    ///         Second,
    ///         Minute,
    ///         Hour,
    ///         Day,
    ///         Week,
    ///         Month,
    ///         Year
    ///     ]
    /// );
    /// ```
    pub const fn all_time_units(mut self) -> Self {
        self.time_units_choice = TimeUnitsChoice::All;
        self
    }

    /// Configure the [`DurationParserBuilder`] to build the [`DurationParser`] with a custom set
    /// of time units.
    ///
    /// Setting the time units with this method overwrites any previous choices with
    ///
    /// * [`DurationParserBuilder::default_time_units`]
    /// * [`DurationParserBuilder::all_time_units`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::DurationParserBuilder;
    /// use fundu::TimeUnit::*;
    ///
    /// assert_eq!(
    ///     DurationParserBuilder::new()
    ///         .time_units(&[NanoSecond, Second, Year])
    ///         .build()
    ///         .get_current_time_units(),
    ///     vec![NanoSecond, Second, Year]
    /// );
    /// ```
    pub const fn time_units(mut self, time_units: &'a [TimeUnit]) -> Self {
        self.time_units_choice = TimeUnitsChoice::Custom(time_units);
        self
    }

    /// Set the default time unit to something different than [`TimeUnit::Second`]
    ///
    /// See also [`DurationParser::default_unit`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// assert_eq!(
    ///     DurationParserBuilder::new()
    ///         .all_time_units()
    ///         .default_unit(NanoSecond)
    ///         .build()
    ///         .parse("42")
    ///         .unwrap(),
    ///     Duration::positive(0, 42)
    /// );
    /// ```
    pub const fn default_unit(mut self, time_unit: TimeUnit) -> Self {
        self.config.default_unit = time_unit;
        self
    }

    /// Allow one or more delimiters between the number and the [`TimeUnit`].
    ///
    /// See also [`DurationParser::allow_time_unit_delimiter`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .default_time_units()
    ///     .allow_time_unit_delimiter()
    ///     .build();
    ///
    /// assert_eq!(
    ///     parser.parse("123 \t\n\x0C\rns"),
    ///     Ok(Duration::positive(0, 123))
    /// );
    /// assert_eq!(parser.parse("123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123   ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub const fn allow_time_unit_delimiter(mut self) -> Self {
        self.config.allow_time_unit_delimiter = true;
        self
    }

    /// Allow one or more delimiters between the leading sign and the number.
    ///
    /// See also [`DurationParser::allow_sign_delimiter`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .default_time_units()
    ///     .allow_sign_delimiter()
    ///     .build();
    ///
    /// assert_eq!(parser.parse("+123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("+\t\n 123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("+ 123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("+     123ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub const fn allow_sign_delimiter(mut self) -> Self {
        self.config.allow_sign_delimiter = true;
        self
    }

    /// If set, parsing negative durations is possible
    ///
    /// See also [`DurationParser::allow_negative`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new().allow_negative().build();
    ///
    /// assert_eq!(parser.parse("-123"), Ok(Duration::negative(123, 0)));
    /// assert_eq!(parser.parse("-1.23e-7"), Ok(Duration::negative(0, 123)));
    /// ```
    pub const fn allow_negative(mut self) -> Self {
        self.config.allow_negative = true;
        self
    }

    /// The `ago` keyword can follow a time unit separated by the `inner_delimiter` to denote a
    /// negative duration.
    ///
    /// See also [`DurationParser::allow_ago`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .all_time_units()
    ///     .allow_ago()
    ///     .build();
    ///
    /// assert_eq!(parser.parse("123ns ago"), Ok(Duration::negative(0, 123)));
    /// assert_eq!(parser.parse("-123ns ago"), Ok(Duration::positive(0, 123)));
    /// ```
    ///
    /// And some illegal usages of `ago`
    ///
    /// ```rust
    /// use fundu::{DurationParserBuilder, ParseError};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .all_time_units()
    ///     .allow_ago()
    ///     .build();
    ///
    /// // Error because no time unit was specified
    /// assert_eq!(
    ///     parser.parse("123 ago"),
    ///     Err(ParseError::TimeUnit(
    ///         3,
    ///         "Invalid time unit: ' ago'".to_string()
    ///     ))
    /// );
    ///
    /// // Error because ago was specified multiple times
    /// assert_eq!(
    ///     parser.parse("123ns ago ago"),
    ///     Err(ParseError::Syntax(
    ///         9,
    ///         "Expected end of input but found: ' ago'".to_string()
    ///     ))
    /// );
    /// ```
    pub const fn allow_ago(mut self) -> Self {
        self.config.allow_ago = true;
        self.config.allow_negative = true;
        self
    }

    /// Disable parsing an exponent.
    ///
    /// See also [`DurationParser::disable_exponent`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{DurationParserBuilder, ParseError};
    ///
    /// assert_eq!(
    ///     DurationParserBuilder::new()
    ///         .default_time_units()
    ///         .disable_exponent()
    ///         .build()
    ///         .parse("123e+1"),
    ///     Err(ParseError::Syntax(3, "No exponent allowed".to_string()))
    /// );
    /// ```
    pub const fn disable_exponent(mut self) -> Self {
        self.config.disable_exponent = true;
        self
    }

    /// Disable parsing a fraction in the source string.
    ///
    /// See also [`DurationParser::disable_fraction`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder, ParseError};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .default_time_units()
    ///     .disable_fraction()
    ///     .build();
    ///
    /// assert_eq!(
    ///     parser.parse("123.456"),
    ///     Err(ParseError::Syntax(3, "No fraction allowed".to_string()))
    /// );
    ///
    /// assert_eq!(
    ///     parser.parse("123e-2"),
    ///     Ok(Duration::positive(1, 230_000_000))
    /// );
    /// assert_eq!(parser.parse("123ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub const fn disable_fraction(mut self) -> Self {
        self.config.disable_fraction = true;
        self
    }

    /// Disable parsing infinity values
    ///
    /// See also [`DurationParser::disable_infinity`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{DurationParserBuilder, ParseError};
    ///
    /// let parser = DurationParserBuilder::new().disable_infinity().build();
    ///
    /// assert_eq!(
    ///     parser.parse("inf"),
    ///     Err(ParseError::InvalidInput("inf".to_owned()))
    /// );
    /// assert_eq!(
    ///     parser.parse("infinity"),
    ///     Err(ParseError::InvalidInput("infinity".to_owned()))
    /// );
    /// assert_eq!(
    ///     parser.parse("+inf"),
    ///     Err(ParseError::InvalidInput("inf".to_owned()))
    /// );
    /// ```
    pub const fn disable_infinity(mut self) -> Self {
        self.config.disable_infinity = true;
        self
    }

    /// This setting makes a number in the source string optional.
    ///
    /// See also [`DurationParser::number_is_optional`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .default_time_units()
    ///     .number_is_optional()
    ///     .build();
    ///
    /// assert_eq!(parser.parse("ns"), Ok(Duration::positive(0, 1)));
    /// assert_eq!(parser.parse("+ns"), Ok(Duration::positive(0, 1)));
    /// ```
    pub const fn number_is_optional(mut self) -> Self {
        self.config.number_is_optional = true;
        self
    }

    /// Parse possibly multiple durations and sum them up.
    ///
    /// See also [`DurationParser::parse_multiple`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .default_time_units()
    ///     .parse_multiple(Some(&["and"]))
    ///     .build();
    ///
    /// assert_eq!(
    ///     parser.parse("1.5h 2e+2ns"),
    ///     Ok(Duration::positive(5400, 200))
    /// );
    /// assert_eq!(
    ///     parser.parse("55s500ms"),
    ///     Ok(Duration::positive(55, 500_000_000))
    /// );
    /// assert_eq!(parser.parse("1m and 1ns"), Ok(Duration::positive(60, 1)));
    /// assert_eq!(
    ///     parser.parse("1.   .1"),
    ///     Ok(Duration::positive(1, 100_000_000))
    /// );
    /// assert_eq!(parser.parse("2h"), Ok(Duration::positive(2 * 60 * 60, 0)));
    /// assert_eq!(
    ///     parser.parse("300ms20s 5d"),
    ///     Ok(Duration::positive(5 * 60 * 60 * 24 + 20, 300_000_000))
    /// );
    /// ```
    pub const fn parse_multiple(mut self, conjunctions: Option<&'a [&'a str]>) -> Self {
        self.config.allow_multiple = true;
        self.config.conjunctions = conjunctions;
        self
    }

    /// Set the inner [`Delimiter`] to something different then the default
    /// [`u8::is_ascii_whitespace`]
    ///
    /// See also [`DurationParser::set_inner_delimiter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .all_time_units()
    ///     .allow_ago()
    ///     .inner_delimiter(|byte| byte == b'#')
    ///     .build();
    ///
    /// assert_eq!(parser.parse("1.5h#ago"), Ok(Duration::negative(5400, 0)));
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .all_time_units()
    ///     .allow_sign_delimiter()
    ///     .inner_delimiter(|byte| byte == b'#')
    ///     .build();
    ///
    /// assert_eq!(parser.parse("+##1.5h"), Ok(Duration::positive(5400, 0)));
    /// ```
    pub const fn inner_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.config.inner_delimiter = delimiter;
        self
    }

    /// Set the outer [`Delimiter`] to something different then the default
    /// [`u8::is_ascii_whitespace`]
    ///
    /// See also [`DurationParser::set_outer_delimiter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .all_time_units()
    ///     .parse_multiple(None)
    ///     .outer_delimiter(|byte| byte == b';')
    ///     .build();
    ///
    /// assert_eq!(
    ///     parser.parse("1.5h;2e+2ns"),
    ///     Ok(Duration::positive(5400, 200))
    /// );
    ///
    /// let parser = DurationParserBuilder::new()
    ///     .all_time_units()
    ///     .parse_multiple(Some(&["and"]))
    ///     .outer_delimiter(|byte| byte == b';')
    ///     .build();
    ///
    /// assert_eq!(
    ///     parser.parse("1.5h;and;2e+2ns"),
    ///     Ok(Duration::positive(5400, 200))
    /// );
    /// ```
    pub const fn outer_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.config.outer_delimiter = delimiter;
        self
    }

    /// Finally, build the [`DurationParser`] from this builder.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParserBuilder};
    ///
    /// let parser = DurationParserBuilder::new().default_time_units().build();
    /// for input in &["1m", "60s"] {
    ///     assert_eq!(parser.parse(input).unwrap(), Duration::positive(60, 0))
    /// }
    /// ```
    pub const fn build(self) -> DurationParser<'a> {
        let parser = Parser::with_config(self.config);

        match self.time_units_choice {
            TimeUnitsChoice::Default => DurationParser {
                time_units: TimeUnits::with_default_time_units(),
                inner: parser,
            },
            TimeUnitsChoice::All => DurationParser {
                time_units: TimeUnits::with_all_time_units(),
                inner: parser,
            },
            TimeUnitsChoice::None => DurationParser {
                time_units: TimeUnits::new(),
                inner: parser,
            },
            TimeUnitsChoice::Custom(time_units) => DurationParser {
                time_units: TimeUnits::with_time_units(time_units),
                inner: parser,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use fundu_core::config::Config;
    use rstest::rstest;

    use super::*;
    use crate::TimeUnit::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_new() {
        let builder = DurationParserBuilder::new();
        assert_eq!(builder.config, Config::new());
        assert_eq!(builder.time_units_choice, TimeUnitsChoice::None);
    }

    #[test]
    fn test_duration_parser_builder_when_default_time_units() {
        let builder = DurationParserBuilder::new().default_time_units();
        assert_eq!(builder.time_units_choice, TimeUnitsChoice::Default);
    }

    #[test]
    fn test_duration_parser_builder_when_all_time_units() {
        let builder = DurationParserBuilder::new().all_time_units();
        assert_eq!(builder.time_units_choice, TimeUnitsChoice::All);
    }

    #[test]
    fn test_duration_parser_builder_when_custom_time_units() {
        let builder = DurationParserBuilder::new().time_units(&[MicroSecond, Hour, Week, Year]);
        assert_eq!(
            builder.time_units_choice,
            TimeUnitsChoice::Custom(&[MicroSecond, Hour, Week, Year])
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_default_unit() {
        let mut expected = Config::new();
        expected.default_unit = MicroSecond;

        let builder = DurationParserBuilder::new().default_unit(MicroSecond);

        assert_eq!(builder.config, expected);
    }

    #[test]
    fn test_duration_parser_builder_when_allow_delimiter() {
        let builder = DurationParserBuilder::new().allow_time_unit_delimiter();
        assert!(builder.config.allow_time_unit_delimiter);
    }

    #[test]
    fn test_duration_parser_builder_when_allow_ago() {
        let builder = DurationParserBuilder::new().allow_ago();
        assert!(builder.config.allow_ago);
        assert!(builder.config.allow_negative);
    }

    #[test]
    fn test_duration_parser_builder_when_allow_sign_delimiter() {
        let builder = DurationParserBuilder::new().allow_sign_delimiter();
        assert!(builder.config.allow_sign_delimiter);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_disable_fraction() {
        let mut expected = Config::new();
        expected.disable_fraction = true;

        let builder = DurationParserBuilder::new().disable_fraction();

        assert_eq!(builder.config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_disable_exponent() {
        let mut expected = Config::new();
        expected.disable_exponent = true;

        let builder = DurationParserBuilder::new().disable_exponent();

        assert_eq!(builder.config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_disable_infinity() {
        let mut expected = Config::new();
        expected.disable_infinity = true;

        let builder = DurationParserBuilder::new().disable_infinity();

        assert_eq!(builder.config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_number_is_optional() {
        let mut expected = Config::new();
        expected.number_is_optional = true;

        let builder = DurationParserBuilder::new().number_is_optional();
        assert_eq!(builder.config, expected);
    }

    #[test]
    fn test_duration_parser_builder_when_parse_multiple() {
        let builder = DurationParserBuilder::new().parse_multiple(None);

        assert!(builder.config.allow_multiple);
    }

    #[rstest]
    #[case::default_time_units(TimeUnitsChoice::Default, DurationParser::new())]
    #[case::all_time_units(TimeUnitsChoice::All, DurationParser::with_all_time_units())]
    #[case::no_time_units(TimeUnitsChoice::None, DurationParser::without_time_units())]
    #[case::custom_time_units(
            TimeUnitsChoice::Custom(&[NanoSecond, Minute]),
            DurationParser::with_time_units(&[NanoSecond, Minute])
    )]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_build(
        #[case] choice: TimeUnitsChoice,
        #[case] expected: DurationParser,
    ) {
        let mut builder = DurationParserBuilder::new();
        builder.time_units_choice = choice;

        assert_eq!(builder.build(), expected);
    }

    #[test]
    fn test_duration_parser_builder_set_inner_delimiter() {
        let builder = DurationParserBuilder::new().inner_delimiter(|byte| byte == b'a');
        assert!((builder.config.inner_delimiter)(b'a'));
        assert!(!(builder.config.inner_delimiter)(b' '));
    }

    #[test]
    fn test_duration_parser_builder_set_outer_delimiter() {
        let builder = DurationParserBuilder::new().outer_delimiter(|byte| byte == b'a');
        assert!((builder.config.outer_delimiter)(b'a'));
        assert!(!(builder.config.outer_delimiter)(b' '));
    }
}
