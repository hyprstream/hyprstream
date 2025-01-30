// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use fundu_core::config::Delimiter;
use fundu_core::parse::Parser;
use fundu_core::time::{Duration, Multiplier, TimeUnitsLike};

use super::builder::CustomDurationParserBuilder;
use super::time_units::{CustomTimeUnit, CustomTimeUnits, TimeKeyword};
use super::Numerals;
use crate::{Numeral, ParseError, TimeUnit};

/// A parser with a customizable set of [`TimeUnit`]s and customizable identifiers.
///
/// See also [`CustomDurationParser::with_time_units`].
///
/// # Problems
///
/// It's possible to choose identifiers very freely as long as they are valid utf-8. However, some
/// identifiers interact badly with the parser and may lead to problems if they start with:
///
/// * `e` or `E` which is also indicating an exponent. If [`CustomDurationParser::disable_exponent`]
///   is set to true this problem does not occur.
/// * `inf` and in consequence `infinity` case-insensitive. These are reserved words as long as
///   [`CustomDurationParser::disable_infinity`] isn't set to true.
/// * ascii digits from `0` to `9`
/// * `.` which is also indicating a fraction. If [`CustomDurationParser::disable_fraction`] is set
///   to true, this problem does not occur
/// * `+`, `-` which are in use for signs.
/// * whitespace characters
#[derive(Debug, PartialEq, Eq)]
pub struct CustomDurationParser<'a> {
    pub(super) time_units: CustomTimeUnits<'a>,
    pub(super) keywords: CustomTimeUnits<'a>,
    pub(super) numerals: Numerals<'a>,
    pub(super) inner: Parser<'a>,
}

impl<'a> CustomDurationParser<'a> {
    /// Create a new empty [`CustomDurationParser`] without any time units.
    ///
    /// There's also [`CustomDurationParser::with_time_units`] which initializes the parser with a
    /// set of [`CustomTimeUnit`]s.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, Duration};
    ///
    /// assert_eq!(
    ///     CustomDurationParser::new().parse("100.0").unwrap(),
    ///     Duration::positive(100, 0)
    /// );
    /// ```
    pub fn new() -> Self {
        Self {
            time_units: CustomTimeUnits::new(),
            keywords: CustomTimeUnits::new(),
            inner: Parser::new(),
            numerals: Numerals::new(),
        }
    }

    /// Create a new [`CustomDurationParser`] with an initial set of [`CustomTimeUnit`]s.
    ///
    /// Not all time units need to be defined, so if there is no intention to include a specific
    /// [`TimeUnit`] just leave it out of the `units`. Be aware, that this method does not check the
    /// validity of identifiers, so besides the need to be a valid `utf-8` sequence there are no
    /// other hard limitations but see also the `Problems` section in [`CustomDurationParser`].
    /// There is also no check for duplicate `ids`, however [`CustomTimeUnit`]s with empty `ids` are
    /// ignored. Note the ids for time units are case sensitive.
    ///
    /// You may find it helpful to start with a pre-defined custom sets of [`TimeUnit`]:
    /// * [`crate::SYSTEMD_TIME_UNITS`]: This is the set of time units as specified in the [`systemd.time`](https://www.man7.org/linux/man-pages/man7/systemd.time.7.html)
    ///   documentation
    /// * [`crate::DEFAULT_TIME_UNITS`]: This is the complete set of time units with their default
    ///   ids as used the standard crate by [`crate::DurationParser`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier};
    ///
    /// let parser = CustomDurationParser::with_time_units(&[
    ///     CustomTimeUnit::with_default(Second, &["s"]),
    ///     CustomTimeUnit::with_default(Minute, &["Min"]),
    ///     CustomTimeUnit::with_default(Hour, &["ώρα"]),
    /// ]);
    /// assert_eq!(
    ///     parser.get_time_unit_by_id("s"),
    ///     Some((Second, Multiplier(1, 0)))
    /// );
    /// assert_eq!(
    ///     parser.get_time_unit_by_id("Min"),
    ///     Some((Minute, Multiplier(1, 0)))
    /// );
    /// assert_eq!(
    ///     parser.get_time_unit_by_id("ώρα"),
    ///     Some((Hour, Multiplier(1, 0)))
    /// );
    ///
    /// assert!(parser.parse("42.0min").is_err()); // Note the small letter `m` instead of `M`
    ///
    /// assert_eq!(
    ///     parser.parse("42e-1ώρα").unwrap(),
    ///     Duration::positive(15120, 0)
    /// );
    /// ```
    pub fn with_time_units(units: &[CustomTimeUnit<'a>]) -> Self {
        Self {
            time_units: CustomTimeUnits::with_time_units(units),
            keywords: CustomTimeUnits::new(),
            inner: Parser::new(),
            numerals: Numerals::new(),
        }
    }

    /// Use the [`CustomDurationParserBuilder`] to construct a [`CustomDurationParser`].
    ///
    /// The [`CustomDurationParserBuilder`] is more ergonomic in some use cases than using
    /// [`CustomDurationParser`] directly. Using this method is the same like invoking
    /// [`CustomDurationParserBuilder::default`].
    ///
    /// See [`CustomDurationParserBuilder`] for more details.
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
    /// assert_eq!(parser.parse("1 ns").unwrap(), Duration::positive(0, 1));
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
    pub const fn builder() -> CustomDurationParserBuilder<'a> {
        CustomDurationParserBuilder::new()
    }

    /// Add a [`CustomTimeUnit`] to the current set of time units.
    ///
    /// [`CustomTimeUnit`]s have a base [`TimeUnit`] and a [`Multiplier`] in addition to their
    /// identifiers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier};
    ///
    /// let mut parser = CustomDurationParser::new();
    /// parser
    ///     .time_unit(CustomTimeUnit::new(
    ///         Week,
    ///         &["fortnight", "fortnights"],
    ///         Some(Multiplier(2, 0)),
    ///     ))
    ///     .time_unit(CustomTimeUnit::new(
    ///         Second,
    ///         &["kilosecond", "kiloseconds", "kilos"],
    ///         Some(Multiplier(1000, 0)),
    ///     ))
    ///     .time_unit(CustomTimeUnit::new(
    ///         Second,
    ///         &["shakes"],
    ///         Some(Multiplier(1, -8)),
    ///     ));
    /// assert_eq!(
    ///     parser.parse("1fortnights").unwrap(),
    ///     Duration::positive(2 * 7 * 24 * 60 * 60, 0),
    /// );
    /// ```
    ///
    /// The `base_unit` is only used to calculate the final duration and does not need to be
    /// unique in the set of time units. It's even possible to define an own time unit for
    /// example for a definition of a [`TimeUnit::Year`] either in addition or as a
    /// replacement of the year definition of this crate (Julian Year = `365.25` days).
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier, DEFAULT_TIME_UNITS};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
    ///
    /// // The common year is usually defined as 365 days instead of the Julian Year with `365.25`
    /// // days.
    ///
    /// parser.time_unit(CustomTimeUnit::new(
    ///     Day,
    ///     &["y", "year", "years"],
    ///     Some(Multiplier(365, 0)),
    /// ));
    /// assert_eq!(
    ///     parser.parse("1year").unwrap(),
    ///     Duration::positive(365 * 24 * 60 * 60, 0),
    /// );
    /// ```
    pub fn time_unit(&mut self, time_unit: CustomTimeUnit<'a>) -> &mut Self {
        self.time_units.add_custom_time_unit(time_unit);
        self
    }

    /// Add multiple [`CustomTimeUnit`]s at once.
    ///
    /// See also [`CustomDurationParser::time_unit`]
    ///
    /// # Example
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier};
    ///
    /// const CUSTOM_TIME_UNITS: [CustomTimeUnit; 2] = [
    ///     CustomTimeUnit::new(Week, &["fortnight", "fortnights"], Some(Multiplier(2, 0))),
    ///     CustomTimeUnit::new(Second, &["shake", "shakes"], Some(Multiplier(1, -8))),
    /// ];
    ///
    /// let mut parser = CustomDurationParser::new();
    /// parser.time_units(&CUSTOM_TIME_UNITS);
    ///
    /// assert_eq!(
    ///     parser.parse("1fortnight").unwrap(),
    ///     Duration::positive(2 * 7 * 24 * 60 * 60, 0),
    /// );
    /// ```
    pub fn time_units(&mut self, time_units: &[CustomTimeUnit<'a>]) -> &mut Self {
        for time_unit in time_units {
            self.time_unit(*time_unit);
        }
        self
    }

    /// Add a [`TimeKeyword`] to the current set of keywords
    ///
    /// [`TimeKeyword`]s are almost the same like [`CustomTimeUnit`]s and can be defined in the same
    /// way like [`CustomTimeUnit`]s. However, they are parsed differently and don't accept a number
    /// in front of them.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{
    ///     CustomDurationParser, CustomTimeUnit, Duration, Multiplier, ParseError, TimeKeyword,
    /// };
    ///
    /// let mut parser =
    ///     CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(NanoSecond, &["ns"])]);
    /// parser.keyword(TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))));
    ///
    /// assert_eq!(parser.parse("123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(
    ///     parser.parse("tomorrow"),
    ///     Ok(Duration::positive(60 * 60 * 24, 0))
    /// );
    ///
    /// // but not
    /// assert_eq!(
    ///     parser.parse("123tomorrow"),
    ///     Err(ParseError::TimeUnit(
    ///         3,
    ///         "Invalid time unit: 'tomorrow'".to_string()
    ///     ))
    /// );
    /// ```
    pub fn keyword(&mut self, keyword: TimeKeyword<'a>) -> &mut Self {
        self.keywords
            .add_custom_time_unit(keyword.to_custom_time_unit());
        self
    }

    /// Add multiple [`TimeKeyword`]s to the current set of keywords
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, Duration, Multiplier, TimeKeyword};
    ///
    /// let mut parser = CustomDurationParser::new();
    /// parser.allow_negative(true).keywords(&[
    ///     TimeKeyword::new(Day, &["yesterday"], Some(Multiplier(-1, 0))),
    ///     TimeKeyword::new(Day, &["tomorrow"], Some(Multiplier(1, 0))),
    /// ]);
    ///
    /// assert_eq!(
    ///     parser.parse("yesterday"),
    ///     Ok(Duration::negative(60 * 60 * 24, 0))
    /// );
    /// assert_eq!(
    ///     parser.parse("tomorrow"),
    ///     Ok(Duration::positive(60 * 60 * 24, 0))
    /// );
    /// ```
    pub fn keywords(&mut self, keywords: &[TimeKeyword<'a>]) -> &mut Self {
        for keyword in keywords {
            self.keyword(*keyword);
        }
        self
    }

    /// Add a [`Numeral`] to the current set of numerals
    ///
    /// `Numerals` need at least one [`CustomTimeUnit`] to be defined.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier, Numeral};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(
    ///     NanoSecond,
    ///     &["nano", "nanos"],
    /// )]);
    /// parser.numeral(Numeral::new(&["one", "next"], Multiplier(1, 0)));
    ///
    /// assert_eq!(parser.parse("next nano"), Ok(Duration::positive(0, 1)));
    /// assert_eq!(parser.parse("one nano"), Ok(Duration::positive(0, 1)));
    /// ```
    pub fn numeral(&mut self, numeral: Numeral<'a>) -> &mut Self {
        self.numerals.data.push(numeral);
        self
    }

    /// Add one or more [`Numeral`]s to the current set of numerals
    ///
    /// Note that numerals need at least one [`CustomTimeUnit`] to be defined.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier, Numeral};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(
    ///     NanoSecond,
    ///     &["nano", "nanos"],
    /// )]);
    /// parser
    ///     .allow_negative(true) // needed for the numeral `last`
    ///     .numerals(&[
    ///         Numeral::new(&["last"], Multiplier(-1, 0)),
    ///         Numeral::new(&["this"], Multiplier(0, 0)),
    ///         Numeral::new(&["one", "next"], Multiplier(1, 0)),
    ///         Numeral::new(&["two"], Multiplier(2, 0)),
    ///     ]);
    ///
    /// assert_eq!(parser.parse("last nano"), Ok(Duration::negative(0, 1)));
    /// assert_eq!(parser.parse("this nano"), Ok(Duration::negative(0, 0)));
    /// assert_eq!(parser.parse("next nano"), Ok(Duration::positive(0, 1)));
    /// assert_eq!(parser.parse("two nanos"), Ok(Duration::positive(0, 2)));
    /// ```
    pub fn numerals(&mut self, numerals: &[Numeral<'a>]) -> &mut Self {
        for numeral in numerals {
            self.numeral(*numeral);
        }
        self
    }

    /// Parse the `source` string into a [`crate::Duration`].
    ///
    /// See the [module level documentation](crate) for more information on the format.
    ///
    /// # Errors
    ///
    /// If parsing to a [`crate::Duration`] fails, a [`ParseError`] is returned
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, Duration};
    ///
    /// assert_eq!(
    ///     CustomDurationParser::new().parse("1.2e-1").unwrap(),
    ///     Duration::positive(0, 120_000_000),
    /// );
    /// ```
    #[inline]
    pub fn parse(&self, source: &str) -> Result<Duration, ParseError> {
        self.inner.parse(
            source,
            &self.time_units,
            (!self.keywords.is_empty()).then_some(&self.keywords),
            (!self.numerals.is_empty()).then_some(&self.numerals),
        )
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
    /// use fundu::{CustomDurationParser, Duration};
    ///
    /// assert_eq!(
    ///     CustomDurationParser::new()
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

    /// If `Some`, allow one or more [`Delimiter`] between the number and the [`TimeUnit`].
    ///
    /// See also [`crate::DurationParser::allow_time_unit_delimiter`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, ParseError};
    ///
    /// let mut parser =
    ///     CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(NanoSecond, &["ns"])]);
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
    /// assert_eq!(parser.parse("123     ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123ns"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123\t\n\r ns"), Ok(Duration::positive(0, 123)));
    /// ```
    pub fn allow_time_unit_delimiter(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_time_unit_delimiter = value;
        self
    }

    /// Allow one or more delimiters between the leading sign and the number.
    ///
    /// See also [`crate::DurationParser::allow_sign_delimiter`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration};
    ///
    /// let mut parser =
    ///     CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(NanoSecond, &["ns"])]);
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

    /// If true, parsing negative durations is possible
    ///
    /// This setting must be enabled if a [`CustomTimeUnit`] has a negative [`Multiplier`], a
    /// [`CustomDurationParser::keyword`] is negative or the [`CustomDurationParser::allow_ago`]
    /// setting is enabled. If `allow_negative` is disabled and a negative time unit, keyword etc.
    /// is encountered a [`ParseError::NegativeNumber`] is returned.
    ///
    /// See also [`crate::DurationParser::allow_negative`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier, TimeKeyword};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&[
    ///     CustomTimeUnit::with_default(NanoSecond, &["ns"]),
    ///     CustomTimeUnit::new(Second, &["neg"], Some(Multiplier(-1, 0))),
    /// ]);
    /// parser.allow_negative(true).keyword(TimeKeyword::new(
    ///     Day,
    ///     &["yesterday"],
    ///     Some(Multiplier(-1, 0)),
    /// ));
    ///
    /// assert_eq!(parser.parse("-123ns"), Ok(Duration::negative(0, 123)));
    /// assert_eq!(parser.parse("1.23e-7neg"), Ok(Duration::negative(0, 123)));
    /// assert_eq!(
    ///     parser.parse("yesterday"),
    ///     Ok(Duration::negative(60 * 60 * 24, 0))
    /// );
    /// ```
    pub fn allow_negative(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_negative = value;
        self
    }

    /// If the delimiter is set, the `ago` keyword can follow a time unit and a [`Delimiter`] to
    /// denote a negative duration
    ///
    /// The `ago` keyword is allowed in the source string after a time unit and only if a time unit
    /// was encountered. The time unit and `ago` must be delimited by the specified delimiter. In
    /// contrast to [`CustomTimeUnit`]'s, an encountered [`TimeKeyword`] doesn't allow for the `ago`
    /// keyword. Note that setting this option automatically sets
    /// [`CustomDurationParser::allow_negative`] to true.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, Multiplier, TimeKeyword};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&[
    ///     CustomTimeUnit::with_default(NanoSecond, &["ns"]),
    ///     CustomTimeUnit::new(Second, &["neg"], Some(Multiplier(-1, 0))),
    /// ]);
    /// parser.allow_ago(true).keyword(TimeKeyword::new(
    ///     Day,
    ///     &["yesterday"],
    ///     Some(Multiplier(-1, 0)),
    /// ));
    ///
    /// assert_eq!(parser.parse("123ns ago"), Ok(Duration::negative(0, 123)));
    /// assert_eq!(parser.parse("-123ns ago"), Ok(Duration::positive(0, 123)));
    /// assert_eq!(parser.parse("123neg ago"), Ok(Duration::positive(123, 0)));
    /// ```
    ///
    /// And some illegal usages of `ago`
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Multiplier, ParseError, TimeKeyword};
    ///
    /// let mut parser =
    ///     CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(NanoSecond, &["ns"])]);
    /// parser.allow_ago(true).keyword(TimeKeyword::new(
    ///     Day,
    ///     &["yesterday"],
    ///     Some(Multiplier(-1, 0)),
    /// ));
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
    ///
    /// // Error because `yesterday` is a [`TimeKeyword`]
    /// assert_eq!(
    ///     parser.parse("yesterday ago"),
    ///     Err(ParseError::InvalidInput("yesterday ago".to_string()))
    /// );
    /// ```
    pub fn allow_ago(&mut self, value: bool) -> &mut Self {
        self.inner.config.allow_ago = value;
        if value {
            self.inner.config.allow_negative = true;
        }
        self
    }

    /// Disable parsing the exponent.
    ///
    /// See also [`crate::DurationParser::disable_exponent`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, ParseError, DEFAULT_TIME_UNITS};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
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

    /// Disable parsing a fraction in the source string.
    ///
    /// See also [`crate::DurationParser::disable_fraction`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Duration, ParseError};
    ///
    /// let mut parser =
    ///     CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(NanoSecond, &["ns"])]);
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
    /// See also [`crate::DurationParser::disable_infinity`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, ParseError};
    ///
    /// let mut parser = CustomDurationParser::new();
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

    /// This setting makes a number in the source string optional.
    ///
    /// See also [`crate::DurationParser::number_is_optional`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, Duration, DEFAULT_TIME_UNITS};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
    /// parser.number_is_optional(true);
    ///
    /// assert_eq!(parser.parse("ns"), Ok(Duration::positive(0, 1)));
    /// assert_eq!(parser.parse(".001e-6s"), Ok(Duration::positive(0, 1)));
    /// assert_eq!(parser.parse("+ns"), Ok(Duration::positive(0, 1)));
    /// ```
    pub fn number_is_optional(&mut self, value: bool) -> &mut Self {
        self.inner.config.number_is_optional = value;
        self
    }

    /// If set to some [`Delimiter`], parse possibly multiple durations and sum them up.
    ///
    /// See also [`crate::DurationParser::parse_multiple`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, Duration, DEFAULT_TIME_UNITS};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
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
    /// * [`CustomDurationParser::allow_sign_delimiter`]: Between the sign and the number
    /// * [`CustomDurationParser::allow_time_unit_delimiter`]: Between the number and the time unit
    /// * [`CustomDurationParser::allow_ago`]: Between the time unit and the `ago` keyword
    /// * If [`NumbersLike`] numerals are used, between the numeral and the time unit
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, Duration, DEFAULT_TIME_UNITS};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
    /// parser
    ///     .allow_ago(true)
    ///     .set_inner_delimiter(|byte| byte == b'#');
    ///
    /// assert_eq!(parser.parse("1.5h#ago"), Ok(Duration::negative(5400, 0)));
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
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
    /// therefore used only if [`CustomDurationParser::parse_multiple`] is set. If
    /// `conjunctions` are set, this delimiter also separates the conjunction from the durations
    /// (like in `1second and 1minute`)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::{CustomDurationParser, Duration, DEFAULT_TIME_UNITS};
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
    /// parser
    ///     .parse_multiple(true, None)
    ///     .set_outer_delimiter(|byte| byte == b';');
    ///
    /// assert_eq!(
    ///     parser.parse("1.5h;2e+2ns"),
    ///     Ok(Duration::positive(5400, 200))
    /// );
    ///
    /// let mut parser = CustomDurationParser::with_time_units(&DEFAULT_TIME_UNITS);
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

    /// Try to find the [`TimeUnit`] with it's associate [`Multiplier`] by id
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::TimeUnit::*;
    /// use fundu::{CustomDurationParser, CustomTimeUnit, Multiplier};
    ///
    /// let parser = CustomDurationParser::with_time_units(&[
    ///     CustomTimeUnit::with_default(NanoSecond, &["ns"]),
    ///     CustomTimeUnit::with_default(MicroSecond, &["Ms"]),
    /// ]);
    ///
    /// assert_eq!(parser.get_time_unit_by_id("does_not_exist"), None);
    ///
    /// for (time_unit, id) in &[(NanoSecond, "ns"), (MicroSecond, "Ms")] {
    ///     assert_eq!(
    ///         parser.get_time_unit_by_id(id),
    ///         Some((*time_unit, Multiplier(1, 0)))
    ///     );
    /// }
    /// ```
    pub fn get_time_unit_by_id(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
        self.time_units.get(identifier)
    }

    /// Return true if there are haven't been any time units added, yet.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu::CustomDurationParser;
    ///
    /// let parser = CustomDurationParser::new();
    /// assert!(parser.is_empty())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.time_units.is_empty()
    }
}

impl<'a> Default for CustomDurationParser<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use fundu_core::config::Config;
    use fundu_core::time::Duration;
    use rstest::rstest;

    use super::*;
    use crate::custom::builder::CustomDurationParserBuilder;
    use crate::custom::time_units::DEFAULT_ALL_TIME_UNITS;
    use crate::TimeUnit::*;

    const YEAR: u64 = 60 * 60 * 24 * 365 + 60 * 60 * 24 / 4;
    const MONTH: u64 = YEAR / 12;

    #[test]
    fn test_custom_duration_parser_init_new() {
        let parser = CustomDurationParser::new();
        assert_eq!(parser.inner.config.default_unit, Second);
        assert!(parser.time_units.is_empty());
        assert_eq!(parser.parse("1.0"), Ok(Duration::positive(1, 0)));
        assert_eq!(
            parser.parse("1.0s"),
            Err(ParseError::TimeUnit(
                3,
                "No time units allowed but found: 's'".to_owned()
            ))
        );
    }

    #[test]
    fn test_custom_duration_parser_init_with_time_units() {
        let parser = CustomDurationParser::with_time_units(&DEFAULT_ALL_TIME_UNITS);
        assert_eq!(parser.inner.config.default_unit, Second);
        for unit in DEFAULT_ALL_TIME_UNITS {
            let CustomTimeUnit {
                base_unit,
                identifiers,
                ..
            } = unit;
            for id in identifiers {
                assert_eq!(
                    parser.get_time_unit_by_id(id),
                    Some((base_unit, Multiplier::default()))
                );
            }
        }
        assert_eq!(parser.parse("1.0"), Ok(Duration::positive(1, 0)));
    }

    #[test]
    fn test_custom_duration_parser_init_default() {
        let parser = CustomDurationParser::default();
        assert!(parser.time_units.is_empty());
    }

    #[test]
    fn test_custom_duration_parser_when_add_custom_time_units() {
        let ids = ["century", "centuries"];
        let mut parser = CustomDurationParser::new();
        parser.time_unit(CustomTimeUnit::new(Year, &ids, Some(Multiplier(100, 0))));
        for id in ids {
            assert_eq!(
                parser.get_time_unit_by_id(id),
                Some((Year, Multiplier(100, 0)))
            );
        }
    }

    #[test]
    fn test_custom_duration_parser_when_setting_default_time_unit() {
        let mut parser = CustomDurationParser::new();
        parser.default_unit(NanoSecond);

        assert_eq!(parser.inner.config.default_unit, NanoSecond);
        assert_eq!(parser.parse("1"), Ok(Duration::positive(0, 1)));
    }

    #[rstest]
    #[case::nano_second("1ns", Duration::positive(0, 1))]
    #[case::micro_second("1Ms", Duration::positive(0, 1000))]
    #[case::milli_second("1ms", Duration::positive(0, 1_000_000))]
    #[case::second("1s", Duration::positive(1, 0))]
    #[case::minute("1m", Duration::positive(60, 0))]
    #[case::hour("1h", Duration::positive(60 * 60, 0))]
    #[case::day("1d", Duration::positive(60 * 60 * 24, 0))]
    #[case::week("1w", Duration::positive(60 * 60 * 24 * 7, 0))]
    #[case::month("1M", Duration::positive(MONTH, 0))]
    #[case::year("1y", Duration::positive(YEAR, 0))]
    fn test_custom_duration_parser_parse_when_default_time_units(
        #[case] input: &str,
        #[case] expected: Duration,
    ) {
        let parser = CustomDurationParser::with_time_units(&DEFAULT_ALL_TIME_UNITS);
        assert_eq!(parser.parse(input), Ok(expected));
    }

    #[test]
    fn test_custom_duration_parser_parse_when_non_ascii() {
        let parser = CustomDurationParser::with_time_units(&[CustomTimeUnit::with_default(
            MilliSecond,
            &["мілісекунда"],
        )]);
        assert_eq!(
            parser.parse("1мілісекунда"),
            Ok(Duration::positive(0, 1_000_000))
        );
    }

    #[test]
    fn test_custom_duration_parser_setting_allow_spaces() {
        let mut parser = CustomDurationParser::new();
        parser.allow_time_unit_delimiter(true);
        assert!(parser.inner.config.allow_time_unit_delimiter);
    }

    #[test]
    fn test_custom_duration_parser_setting_allow_sign_delimiter() {
        let mut parser = CustomDurationParser::new();
        parser.allow_sign_delimiter(true);
        assert!(parser.inner.config.allow_sign_delimiter);
    }

    #[test]
    fn test_custom_duration_parser_setting_disable_fraction() {
        let mut parser = CustomDurationParser::new();
        parser.disable_fraction(true);
        assert!(parser.inner.config.disable_fraction);
    }

    #[test]
    fn test_custom_duration_parser_setting_disable_exponent() {
        let mut parser = CustomDurationParser::new();
        parser.disable_exponent(true);
        assert!(parser.inner.config.disable_exponent);
    }

    #[test]
    fn test_custom_duration_parser_setting_disable_infinity() {
        let mut parser = CustomDurationParser::new();
        parser.disable_infinity(true);
        assert!(parser.inner.config.disable_infinity);
    }

    #[test]
    fn test_custom_duration_parser_setting_number_is_optional() {
        let mut parser = CustomDurationParser::new();
        parser.number_is_optional(true);
        assert!(parser.inner.config.number_is_optional);
    }

    #[test]
    fn test_custom_duration_parser_setting_parse_multiple() {
        let mut parser = CustomDurationParser::new();
        parser.parse_multiple(true, None);

        assert!(parser.inner.config.allow_multiple);
        assert!(parser.inner.config.conjunctions.is_none());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_custom_duration_parser_method_builder() {
        assert_eq!(
            CustomDurationParser::builder(),
            CustomDurationParserBuilder::new()
        );
    }

    #[test]
    fn test_custom_duration_parser_when_adding_custom_time_units() {
        let time_units = [
            CustomTimeUnit::new(Second, &["sec"], Some(Multiplier(1, 0))),
            CustomTimeUnit::new(Second, &["secs"], Some(Multiplier(2, 0))),
        ];
        let mut custom = CustomDurationParser::new();
        custom.time_units(&time_units);
        assert_eq!(
            custom.get_time_unit_by_id("sec"),
            Some((Second, Multiplier(1, 0)))
        );
        assert_eq!(
            custom.get_time_unit_by_id("secs"),
            Some((Second, Multiplier(2, 0)))
        );
    }

    #[test]
    fn test_custom_duration_parser_when_calling_custom_time_units_with_empty_collection() {
        let mut custom = CustomDurationParser::new();
        assert!(custom.is_empty());

        custom.time_units(&[]);
        assert!(custom.is_empty());
    }

    #[test]
    fn test_custom_duration_parser_when_keyword() {
        let mut custom = CustomDurationParser::new();
        custom.keyword(TimeKeyword::new(Second, &["sec"], Some(Multiplier(2, 0))));
        assert_eq!(
            custom.keywords.get("sec").unwrap(),
            (Second, Multiplier(2, 0))
        );
    }

    #[test]
    fn test_custom_duration_parser_when_keywords() {
        let mut custom = CustomDurationParser::new();
        custom.keywords(&[
            TimeKeyword::new(Second, &["sec"], Some(Multiplier(1, 0))),
            TimeKeyword::new(Second, &["secs"], Some(Multiplier(2, 0))),
        ]);
        assert_eq!(
            custom.keywords.get("sec").unwrap(),
            (Second, Multiplier(1, 0))
        );
        assert_eq!(
            custom.keywords.get("secs").unwrap(),
            (Second, Multiplier(2, 0))
        );
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_custom_duration_parser_allow_negative() {
        let mut expected = Config::new();
        expected.allow_negative = true;

        let mut parser = CustomDurationParser::new();
        parser.allow_negative(true);

        assert_eq!(parser.inner.config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_custom_duration_parser_allow_ago() {
        let mut expected = Config::new();
        expected.allow_ago = true;
        expected.allow_negative = true;

        let mut parser = CustomDurationParser::new();
        parser.allow_ago(true);

        assert_eq!(parser.inner.config, expected);
    }

    #[test]
    fn test_custom_duration_parser_when_numeral() {
        let mut parser = CustomDurationParser::new();
        parser.numeral(Numeral::new(&["some"], Multiplier(1, 0)));

        assert!(!parser.numerals.is_empty());
        assert_eq!(
            parser.numerals.data,
            vec![Numeral::new(&["some"], Multiplier(1, 0))]
        );
    }

    #[test]
    fn test_custom_duration_parser_when_numerals() {
        let mut parser = CustomDurationParser::new();
        parser.numerals(&[
            Numeral::new(&["some"], Multiplier(1, 0)),
            Numeral::new(&["other"], Multiplier(2, 0)),
        ]);

        assert!(!parser.numerals.is_empty());
        assert_eq!(
            parser.numerals.data,
            vec![
                Numeral::new(&["some"], Multiplier(1, 0)),
                Numeral::new(&["other"], Multiplier(2, 0)),
            ]
        );
    }

    #[test]
    fn test_custom_duration_parser_when_set_inner_delimiter() {
        let mut parser = CustomDurationParser::new();
        parser.set_inner_delimiter(|byte| byte == b' ');

        assert!((parser.inner.config.inner_delimiter)(b' '));
        assert!(!(parser.inner.config.inner_delimiter)(b'\n'));
    }

    #[test]
    fn test_custom_duration_parser_when_set_outer_delimiter() {
        let mut parser = CustomDurationParser::new();
        parser.set_outer_delimiter(|byte| byte == b' ');

        assert!((parser.inner.config.outer_delimiter)(b' '));
        assert!(!(parser.inner.config.outer_delimiter)(b'\n'));
    }
}
