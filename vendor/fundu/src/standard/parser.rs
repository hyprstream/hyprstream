// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use std::time::Duration as StdDuration;

use fundu_core::config::Delimiter;
use fundu_core::parse::Parser;
use fundu_core::time::Duration as FunduDuration;

use super::time_units::TimeUnits;
use crate::{DurationParserBuilder, ParseError, TimeUnit};

/// A parser with a customizable set of [`TimeUnit`]s with default identifiers.
///
/// See also the [module level documentation](crate) for more details and more information about
/// the format.
///
/// # Examples
///
/// A parser with the default set of time units
///
/// ```rust
/// use fundu::{Duration, DurationParser};
///
/// let parser = DurationParser::new();
/// assert_eq!(parser.parse("42Ms").unwrap(), Duration::positive(0, 42_000));
/// ```
///
/// The parser is reusable and the set of time units is fully customizable
///
/// ```rust
/// use fundu::{Duration, DurationParser, TimeUnit::*};
//
/// let parser = DurationParser::with_time_units(&[NanoSecond, Minute, Hour]);
/// for (input, expected) in &[
///     ("9e3ns", Duration::positive(0, 9000)),
///     ("10m", Duration::positive(600, 0)),
///     ("1.1h", Duration::positive(3960, 0)),
///     ("7", Duration::positive(7, 0)),
/// ] {
///     assert_eq!(parser.parse(input).unwrap(), *expected);
/// }
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct DurationParser<'a> {
    pub(super) time_units: TimeUnits,
    pub(super) inner: Parser<'a>,
}

impl<'a> DurationParser<'a> {
    /// Construct the parser with the default set of [`TimeUnit`]s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{Duration, DurationParser};
    ///
    /// assert_eq!(
    ///     DurationParser::new().parse("1").unwrap(),
    ///     Duration::positive(1, 0)
    /// );
    /// assert_eq!(
    ///     DurationParser::new().parse("1s").unwrap(),
    ///     Duration::positive(1, 0)
    /// );
    /// assert_eq!(
    ///     DurationParser::new().parse("42.0e9ns").unwrap(),
    ///     Duration::positive(42, 0)
    /// );
    ///
    /// assert_eq!(
    ///     DurationParser::new().get_current_time_units(),
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
    pub const fn new() -> Self {
        Self {
            time_units: TimeUnits::with_default_time_units(),
            inner: Parser::new(),
        }
    }

    /// Initialize the parser with a custom set of [`TimeUnit`]s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{Duration, DurationParser};
    ///
    /// assert_eq!(
    ///     DurationParser::with_time_units(&[NanoSecond, Hour, Week])
    ///         .parse("1.5w")
    ///         .unwrap(),
    ///     Duration::positive(60 * 60 * 24 * 7 + 60 * 60 * 24 * 7 / 2, 0)
    /// );
    /// ```
    pub const fn with_time_units(time_units: &[TimeUnit]) -> Self {
        Self {
            time_units: TimeUnits::with_time_units(time_units),
            inner: Parser::new(),
        }
    }

    /// Return a new parser without [`TimeUnit`]s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// assert_eq!(
    ///     DurationParser::without_time_units().parse("33.33").unwrap(),
    ///     Duration::positive(33, 330_000_000)
    /// );
    ///
    /// assert_eq!(
    ///     DurationParser::without_time_units().get_current_time_units(),
    ///     vec![]
    /// );
    /// ```
    pub const fn without_time_units() -> Self {
        Self {
            time_units: TimeUnits::new(),
            inner: Parser::new(),
        }
    }

    /// Construct a parser with all available [`TimeUnit`]s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::DurationParser;
    /// use fundu::TimeUnit::*;
    ///
    /// assert_eq!(
    ///     DurationParser::with_all_time_units().get_current_time_units(),
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
    pub const fn with_all_time_units() -> Self {
        Self {
            time_units: TimeUnits::with_all_time_units(),
            inner: Parser::new(),
        }
    }

    /// Use the [`DurationParserBuilder`] to construct a [`DurationParser`].
    ///
    /// The [`DurationParserBuilder`] is more ergonomic in some use cases than using
    /// [`DurationParser`] directly. Using this method is the same like invoking
    /// [`DurationParserBuilder::default`].
    ///
    /// See [`DurationParserBuilder`] for more details.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{Duration, DurationParser};
    ///
    /// let parser = DurationParser::builder()
    ///     .all_time_units()
    ///     .default_unit(MicroSecond)
    ///     .allow_time_unit_delimiter()
    ///     .build();
    ///
    /// assert_eq!(parser.parse("1 \t\nns").unwrap(), Duration::positive(0, 1));
    /// assert_eq!(parser.parse("1").unwrap(), Duration::positive(0, 1_000));
    ///
    /// // instead of
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser
    ///     .default_unit(MicroSecond)
    ///     .allow_time_unit_delimiter(true);
    ///
    /// assert_eq!(parser.parse("1 ns").unwrap(), Duration::positive(0, 1));
    /// assert_eq!(parser.parse("1").unwrap(), Duration::positive(0, 1_000));
    /// ```
    pub const fn builder() -> DurationParserBuilder<'a> {
        DurationParserBuilder::new()
    }

    /// Parse the `source` string into a [`crate::Duration`].
    ///
    /// See the [module-level documentation](crate) for more information on the format.
    ///
    /// # Errors
    ///
    /// If parsing into a [`crate::Duration`] fails returns a [`ParseError`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// assert_eq!(
    ///     DurationParser::new().parse("1.2e-1s").unwrap(),
    ///     Duration::positive(0, 120_000_000),
    /// );
    /// ```
    #[inline]
    pub fn parse(&self, source: &str) -> Result<FunduDuration, ParseError> {
        self.inner.parse(source, &self.time_units, None, None)
    }

    /// Set the default [`TimeUnit`] to `unit`.
    ///
    /// The default time unit is applied when no time unit was given in the input string. If the
    /// default time unit is not set with this method the parser defaults to [`TimeUnit::Second`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{Duration, DurationParser};
    ///
    /// assert_eq!(
    ///     DurationParser::with_all_time_units()
    ///         .default_unit(NanoSecond)
    ///         .parse("42")
    ///         .unwrap(),
    ///     Duration::positive(0, 42)
    /// );
    /// ```
    pub fn default_unit(&mut self, time_unit: TimeUnit) -> &mut Self {
        self.inner.config.default_unit = time_unit;
        self
    }

    /// If `Some`, allow one or more inner [`Delimiter`] between the number and the [`TimeUnit`].
    ///
    /// Per default no delimiter is allowed between the number and the [`TimeUnit`]. As
    /// usual the default time unit is assumed if no time unit was present. The delimiter can be
    /// changed with [`DurationParser::set_inner_delimiter`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser, ParseError};
    ///
    /// let mut parser = DurationParser::new();
    /// assert_eq!(
    ///     parser.parse("123 ns"),
    ///     Err(ParseError::TimeUnit(
    ///         3,
    ///         "Invalid time unit: ' ns'".to_string()
    ///     ))
    /// );
    ///
    /// parser.allow_time_unit_delimiter(true);
    /// assert_eq!(parser.parse("123 ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123    ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123ns"), Ok(Duration::positive(0, 123)));
    ///
    /// parser.allow_time_unit_delimiter(true);
    /// assert_eq!(parser.parse("123\r\nns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123\t\n\r ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub fn allow_time_unit_delimiter(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_time_unit_delimiter = value;
        self
    }

    /// Allow one or more delimiters between the leading sign and the number.
    ///
    /// A [`Delimiter`] is defined as closure taking a byte and returning true if the delimiter
    /// matched. If set to `None` (the default) no delimiter is allowed between the sign and the
    /// number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.allow_sign_delimiter(true);
    ///
    /// assert_eq!(parser.parse("+123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("+\t\n 123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("+ 123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("+     123ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub fn allow_sign_delimiter(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_sign_delimiter = value;
        self
    }

    /// If true, then parsing negative durations is possible
    ///
    /// Without setting this option the parser returns [`ParseError::NegativeNumber`] if it
    /// encounters a negative number.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.allow_negative(true);
    ///
    /// assert_eq!(parser.parse("-123"), Ok(Duration::negative(123, 0)));
    /// assert_eq!(parser.parse("-1.23e-7"), Ok(Duration::negative(0, 123)));
    /// ```
    pub fn allow_negative(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_negative = value;
        self
    }

    /// If true, the `ago` keyword can follow a time unit and the `inner_delimiter` to denote a
    /// negative duration
    ///
    /// The `ago` keyword is allowed in the source string after a time unit and only if a time unit
    /// was encountered. The time unit and `ago` must be delimited by the `inner_delimiter`. Note
    /// that setting this option automatically sets [`DurationParser::allow_negative`] to true.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser.allow_ago(true);
    ///
    /// assert_eq!(parser.parse("123ns ago"), Ok(Duration::negative(0, 123)));
    /// assert_eq!(parser.parse("-123ns ago"), Ok(Duration::positive(0, 123)));
    /// ```
    ///
    /// And some illegal usages of `ago`
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{DurationParser, Multiplier, ParseError, TimeKeyword};
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser.allow_ago(true);
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
    pub fn allow_ago(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_ago = value;
        if value {
            self.inner.config.allow_negative = true;
        }
        self
    }

    /// If true, disable parsing an exponent.
    ///
    /// If an exponent is encountered in the input string and this setting is active this results
    /// in an [`ParseError::Syntax`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{DurationParser, ParseError};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.disable_exponent(true);
    /// assert_eq!(
    ///     parser.parse("123e+1"),
    ///     Err(ParseError::Syntax(3, "No exponent allowed".to_string()))
    /// );
    /// ```
    pub fn disable_exponent(&mut self, value: bool) -> &mut Self {
        self.inner.config.disable_exponent = value;
        self
    }

    /// If true, disable parsing a fraction in the source string.
    ///
    /// This setting will disable parsing a fraction and a point delimiter will cause an error
    /// [`ParseError::Syntax`]. It does not prevent [`crate::Duration`]s from being smaller than
    /// seconds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser, ParseError};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.disable_fraction(true);
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
    ///
    /// assert_eq!(parser.parse("123ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub fn disable_fraction(&mut self, value: bool) -> &mut Self {
        self.inner.config.disable_fraction = value;
        self
    }

    /// If true, disable parsing infinity
    ///
    /// This setting will disable parsing infinity values like (`inf` or `infinity`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{DurationParser, ParseError};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.disable_infinity(true);
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
    pub fn disable_infinity(&mut self, value: bool) -> &mut Self {
        self.inner.config.disable_infinity = value;
        self
    }

    /// If true, this setting makes a number in the source string optional.
    ///
    /// If no number is present, then `1` is assumed. If a number is present then it must still
    /// consist of either a whole part and/or fraction part, if not disabled with
    /// [`DurationParser::disable_fraction`]. The exponent is also part of the number and needs a
    /// mantissa.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.number_is_optional(true);
    ///
    /// assert_eq!(parser.parse("ns"), Ok(Duration::positive(0, 1)));
    /// assert_eq!(parser.parse("+ns"), Ok(Duration::positive(0, 1)));
    /// ```
    pub fn number_is_optional(&mut self, value: bool) -> &mut Self {
        self.inner.config.number_is_optional = value;
        self
    }

    /// If set to some [`Delimiter`], parse possibly multiple durations and sum them up.
    ///
    /// If [`Delimiter`] is set to `None`, this functionality is disabled. The [`Delimiter`] may or
    /// may not occur to separate the durations. If the delimiter does not occur the next duration
    /// is recognized by a leading digit. It's also possible to use complete words, like `and` as
    /// conjunctions between durations. Note that the [`Delimiter`] is still needed to be set if
    /// conjunctions are used in order to separate the conjunctions from durations.
    ///
    /// Like a single duration, the summed durations saturate at [`crate::Duration::MAX`].
    /// Parsing multiple durations is short-circuiting and parsing stops after the first
    /// [`ParseError`] was encountered. Note that parsing doesn't stop when reaching
    /// [`crate::Duration::MAX`], so any [`ParseError`]s later in the input string are still
    /// reported.
    ///
    /// # Usage together with number format customizations
    ///
    /// The number format and other aspects can be customized as usual via the methods within this
    /// struct and have the known effect. However, there is a notable constellation which has an
    /// effect on how durations are parsed:
    ///
    /// If [`DurationParser::allow_time_unit_delimiter`] is set to some delimiter, the [`Delimiter`]
    /// of this method and the [`Delimiter`] of the `allow_time_unit_delimiter` method can be equal
    /// either in parts or in a whole without having side-effects on each other. But, if
    /// simultaneously [`DurationParser::number_is_optional`] is set to true, then the resulting
    /// [`crate::Duration`] will differ:
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.parse_multiple(true, None).number_is_optional(true);
    ///
    /// // Here, the parser parses `1`, `s`, `1` and then `ns` separately
    /// assert_eq!(parser.parse("1 s 1 ns"), Ok(Duration::positive(3, 1)));
    ///
    /// // Here, the parser parses `1 s` and then `1 ns`.
    /// parser.allow_time_unit_delimiter(true);
    /// assert_eq!(parser.parse("1 s 1 ns"), Ok(Duration::positive(1, 1)));
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::new();
    /// parser.parse_multiple(true, Some(&["and"]));
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
    pub fn parse_multiple(
        &mut self,
        value: bool,
        conjunctions: Option<&'a [&'a str]>,
    ) -> &mut Self {
        self.inner.config.allow_multiple = value;
        self.inner.config.conjunctions = conjunctions;
        self
    }

    /// Set the inner [`Delimiter`] to something different then the default
    /// [`u8::is_ascii_whitespace`]
    ///
    /// Where the inner delimiter occurs, depends on other options:
    /// * [`DurationParser::allow_sign_delimiter`]: Between the sign and the number
    /// * [`DurationParser::allow_time_unit_delimiter`]: Between the number and the time unit
    /// * [`DurationParser::allow_ago`]: Between the time unit and the `ago` keyword
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser
    ///     .allow_ago(true)
    ///     .set_inner_delimiter(|byte| byte == b'#');
    ///
    /// assert_eq!(parser.parse("1.5h#ago"), Ok(Duration::negative(5400, 0)));
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser
    ///     .allow_sign_delimiter(true)
    ///     .set_inner_delimiter(|byte| byte == b'#');
    ///
    /// assert_eq!(parser.parse("+##1.5h"), Ok(Duration::positive(5400, 0)));
    /// ```
    pub fn set_inner_delimiter(&mut self, delimiter: Delimiter) -> &mut Self {
        self.inner.config.inner_delimiter = delimiter;
        self
    }

    /// Set the outer [`Delimiter`] to something different then the default
    /// [`u8::is_ascii_whitespace`]
    ///
    /// The outer delimiter is used to separate multiple durations like in `1second 1minute` and is
    /// therefore used only if [`DurationParser::parse_multiple`] is set. If `conjunctions` are set,
    /// this delimiter also separates the conjunction from the durations (like in `1second and
    /// 1minute`)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{Duration, DurationParser};
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser
    ///     .parse_multiple(true, None)
    ///     .set_outer_delimiter(|byte| byte == b';');
    ///
    /// assert_eq!(
    ///     parser.parse("1.5h;2e+2ns"),
    ///     Ok(Duration::positive(5400, 200))
    /// );
    ///
    /// let mut parser = DurationParser::with_all_time_units();
    /// parser
    ///     .parse_multiple(true, Some(&["and"]))
    ///     .set_outer_delimiter(|byte| byte == b';');
    ///
    /// assert_eq!(
    ///     parser.parse("1.5h;and;2e+2ns"),
    ///     Ok(Duration::positive(5400, 200))
    /// );
    /// ```
    pub fn set_outer_delimiter(&mut self, delimiter: Delimiter) -> &mut Self {
        self.inner.config.outer_delimiter = delimiter;
        self
    }

    /// Return the currently defined set of [`TimeUnit`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::DurationParser;
    /// use fundu::TimeUnit::*;
    ///
    /// let parser = DurationParser::without_time_units();
    /// assert_eq!(parser.get_current_time_units(), vec![]);
    ///
    /// assert_eq!(
    ///     DurationParser::with_time_units(&[NanoSecond]).get_current_time_units(),
    ///     vec![NanoSecond]
    /// );
    /// ```
    pub fn get_current_time_units(&self) -> Vec<TimeUnit> {
        self.time_units.get_time_units()
    }
}

impl<'a> Default for DurationParser<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a string into a [`std::time::Duration`] by accepting a `string` similar to floating
/// point with the default set of time units.
///
/// This method is providing a simple onetime parser. In contrast to [`DurationParser::parse`] it
/// returns a [`std::time::Duration`]. It is generally a better idea to use the [`DurationParser`]
/// builder if multiple inputs with the same set of time units need to be parsed or a customization
/// of the time format is needed.
///
/// See also the [module level documentation](crate) for more details and more information about
/// the format.
///
/// # Errors
///
/// This function returns a [`ParseError`] when parsing of the input `string` failed.
///
/// # Examples
///
/// ```rust
/// use std::time::Duration;
///
/// use fundu::{parse_duration, ParseError};
///
/// let duration = parse_duration("+1.09e1").unwrap();
/// assert_eq!(duration, Duration::new(10, 900_000_000));
///
/// assert_eq!(
///     parse_duration("Not a number"),
///     Err(ParseError::InvalidInput("Not a number".to_owned()))
/// );
/// ```
#[allow(clippy::missing_panics_doc)]
pub fn parse_duration(string: &str) -> Result<StdDuration, ParseError> {
    DurationParser::new()
        .parse(string)
        // unwrap is safe here because negative durations aren't allowed in the default
        // configuration of the DurationParser
        .map(|d| d.try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use fundu_core::config::Config;
    use fundu_core::time::TimeUnit::*;

    use super::*;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_init_when_default() {
        let config = Config::new();
        let parser = DurationParser::default();
        assert_eq!(parser.inner.config, config);
        assert_eq!(
            parser.get_current_time_units(),
            vec![
                NanoSecond,
                MicroSecond,
                MilliSecond,
                Second,
                Minute,
                Hour,
                Day,
                Week
            ]
        );
    }

    #[test]
    fn test_duration_parser_setting_allow_delimiter() {
        let mut parser = DurationParser::new();
        parser.allow_time_unit_delimiter(true);
        assert!(parser.inner.config.allow_time_unit_delimiter);
    }

    #[test]
    fn test_duration_parser_setting_allow_sign_delimiter() {
        let mut parser = DurationParser::new();
        parser.allow_sign_delimiter(true);
        assert!(parser.inner.config.allow_sign_delimiter);
    }

    #[test]
    fn test_duration_parser_setting_allow_ago() {
        let mut parser = DurationParser::new();
        parser.allow_ago(true);
        assert!(parser.inner.config.allow_ago);
        assert!(parser.inner.config.allow_negative);
    }

    #[test]
    fn test_duration_parser_setting_allow_ago_when_false() {
        let mut parser = DurationParser::new();
        parser.allow_ago(false);
        assert!(!parser.inner.config.allow_ago);
        assert!(!parser.inner.config.allow_negative);
    }

    #[test]
    fn test_duration_parser_setting_disable_infinity() {
        let mut parser = DurationParser::new();
        parser.disable_infinity(true);

        assert!(parser.inner.config.disable_infinity);
    }

    #[test]
    fn test_duration_parser_setting_parse_multiple() {
        let mut parser = DurationParser::new();
        parser.parse_multiple(true, None);

        assert!(parser.inner.config.allow_multiple);
        assert!(parser.inner.config.conjunctions.is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_when_builder() {
        assert_eq!(DurationParser::builder(), DurationParserBuilder::new());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_duration_parser_builder_when_default() {
        assert_eq!(
            DurationParserBuilder::default(),
            DurationParserBuilder::new()
        );
    }

    #[test]
    fn test_duration_parser_set_inner_delimiter() {
        let mut parser = DurationParser::new();
        parser.set_inner_delimiter(|byte| byte == b'a');
        assert!((parser.inner.config.inner_delimiter)(b'a'));
        assert!(!(parser.inner.config.inner_delimiter)(b' '));
    }

    #[test]
    fn test_duration_parser_set_outer_delimiter() {
        let mut parser = DurationParser::new();
        parser.set_outer_delimiter(|byte| byte == b'a');
        assert!((parser.inner.config.outer_delimiter)(b'a'));
        assert!(!(parser.inner.config.outer_delimiter)(b' '));
    }
}
