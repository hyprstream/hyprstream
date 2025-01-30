// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! Provide the [`Config`], [`ConfigBuilder`] and other structures used to adjust the parsing
//! process

use crate::time::{Multiplier, TimeUnit, DEFAULT_TIME_UNIT};

pub(crate) const DEFAULT_CONFIG: Config = Config::new();

/// An ascii delimiter defined as closure.
///
/// The [`Delimiter`] is a type alias for a closure taking a `u8` byte and returning a `bool`.
///
/// # Problems
///
/// The parser relies on the property that this delimiter is defined to match ascii characters. The
/// delimiter takes a `u8` as input, but matching any non-ascii (`0x80 - 0xff`) bytes may lead to a
/// [`crate::error::ParseError`] or panic if the input string contains multi-byte utf-8 characters.
///
/// # Examples
///
/// ```rust
/// use fundu_core::config::Delimiter;
///
/// fn is_delimiter(delimiter: Delimiter, byte: u8) -> bool {
///     delimiter(byte)
/// }
///
/// assert!(is_delimiter(
///     |byte| matches!(byte, b' ' | b'\n' | b'\t'),
///     b' '
/// ));
/// assert!(!is_delimiter(
///     |byte| matches!(byte, b' ' | b'\n' | b'\t'),
///     b'\r'
/// ));
/// assert!(is_delimiter(|byte| byte.is_ascii_whitespace(), b'\r'));
/// ```
pub type Delimiter = fn(u8) -> bool;

/// [`NumbersLike`] strings can occur where usually a number would occur in the source string
///
/// `NumbersLike` words or strings express a number as a word like `one` or `next` instead of `1` or
/// `half` instead of `0.5` but also `last` instead of `-1` etc. This symbolic number is defined
/// as a [`Multiplier`]. In contrast to numbers, `NumbersLike` without a time unit are considered an
/// error and therefore have to be followed by a time unit.
///
/// # Examples
///
/// ```rust
/// use fundu_core::config::NumbersLike;
/// use fundu_core::time::Multiplier;
///
/// struct Numerals {}
/// impl NumbersLike for Numerals {
///     fn get(&self, input: &str) -> Option<Multiplier> {
///         match input {
///             "one" | "next" => Some(Multiplier(1, 0)),
///             "this" => Some(Multiplier(0, 0)),
///             "last" => Some(Multiplier(-1, 0)),
///             "half" => Some(Multiplier(5, -1)),
///             _ => None,
///         }
///     }
/// }
///
/// let numerals = Numerals {};
/// assert_eq!(numerals.get("one"), Some(Multiplier(1, 0)));
/// ```
pub trait NumbersLike {
    fn get(&self, input: &str) -> Option<Multiplier>;
}

/// The structure containing all options for the [`crate::parse::Parser`]
///
/// This struct is highly likely to change often, so it is not possible to create the `Config` with
/// `Config { ... }` outside of this crate. Instead, use [`Config::new`], [`Config::default`] or the
/// [`ConfigBuilder`] to create a new `Config`.
///
/// # Examples
///
/// Here's a little bit more involved example to see the effects of the configuration in action. To
/// keep the example small, [`crate::time::TimeUnitsLike`] is implemented in such a way that no time
/// units are allowed in the input string. The final `Config` uses [`TimeUnit::MilliSecond`] as
/// default time unit instead of [`TimeUnit::Second`] and allows negative durations.
///
/// ```rust
/// use fundu_core::config::{Config, ConfigBuilder};
/// use fundu_core::parse::Parser;
/// use fundu_core::time::{Duration, Multiplier, TimeUnit, TimeUnitsLike};
///
/// struct TimeUnits {}
/// impl TimeUnitsLike for TimeUnits {
///     fn is_empty(&self) -> bool {
///         true
///     }
///
///     fn get(&self, string: &str) -> Option<(TimeUnit, Multiplier)> {
///         None
///     }
/// }
///
/// const TIME_UNITS: TimeUnits = TimeUnits {};
///
/// const CONFIG: Config = ConfigBuilder::new()
///     .default_unit(TimeUnit::MilliSecond)
///     .allow_negative()
///     .build();
///
/// const PARSER: Parser = Parser::with_config(CONFIG);
///
/// assert_eq!(
///     PARSER.parse("1000", &TIME_UNITS, None, None),
///     Ok(Duration::positive(1, 0))
/// );
/// assert_eq!(
///     PARSER.parse("-1", &TIME_UNITS, None, None),
///     Ok(Duration::negative(0, 1_000_000))
/// );
/// ```
#[derive(Debug, PartialEq, Eq, Clone)]
#[allow(clippy::struct_excessive_bools)]
#[non_exhaustive]
pub struct Config<'a> {
    /// The [`TimeUnit`] the parser applies if no time unit was given (Default: `TimeUnit::Second`)
    ///
    /// A configuration with `TimeUnit::MilliSecond` would parse a string without time units like
    /// `"1000"` to a [`crate::time::Duration`] with `Duration::positive(1, 0)` which is worth `1`
    /// second.
    pub default_unit: TimeUnit,

    /// The default [`Multiplier`] used internally to make custom time units possible
    ///
    /// This field is only used internally and it is not recommended to change this setting!
    pub default_multiplier: Multiplier,

    /// Disable parsing an exponent (Default: `false`)
    ///
    /// The exponent in the input string may start with an `e` or `E` followed by an optional sign
    /// character and a mandatory number like in `"1e+20"`, `"2E-3"` ... Occurrences of an exponent
    /// in strings like `"1e20"`,`"10E2"`, `"3.4e-10"` lead to a [`crate::error::ParseError`].
    pub disable_exponent: bool,

    /// Disable parsing a fraction (Default: `false`)
    ///
    /// The fraction in the input string starts with a point character `.` followed by an optional
    /// number like in `"1.234"` but also `"1."` as long as there is a number present before the
    /// point. Occurrences of a fraction like in `"1.234"` when a fraction is not allowed by this
    /// setting lead to a [`crate::error::ParseError`].
    pub disable_fraction: bool,

    /// Disable parsing infinity (Default: `false`)
    ///
    /// An infinity in the input string is either `"inf"` or `"infinity"` case insensitive
    /// optionally preceded by a sign character. Infinity values evaluate to the maximum possible
    /// duration or if negative to the maximum negative value of the duration.
    pub disable_infinity: bool,

    /// Make a number in the input string optional (Default: `false`)
    ///
    /// Usually, a time unit needs a number like in `"1second"`. With this setting set to `true` a
    /// time unit can occur without number like `"second"` and a number with value `1` is assumed.
    pub number_is_optional: bool,

    /// If `true`, this setting allows multiple `durations` in the input (Default: `false`)
    ///
    /// Multiple durations may be delimited by the [`Config::outer_delimiter`]. A subsequent
    /// duration is also recognized if it starts with a digit or a sign character.
    pub allow_multiple: bool,

    /// If parsing multiple durations, allow conjunctions in addition to the [`Delimiter`]
    /// (Default: `None`)
    ///
    /// Conjunctions are words like `"and"`, `"or"` but also single characters like `","` or ";".
    /// So, a string like `"3seconds and 1second"` would parse to a `Duration::positive(4, 0)` and
    /// `"1second, 2seconds"` would parse to a `Duration::positive(3, 0)`. Unlike a [`Delimiter`],
    /// conjunctions can occur only once between two durations.
    pub conjunctions: Option<&'a [&'a str]>,

    /// Allow parsing negative durations (Default: `false`)
    ///
    /// Negative durations usually start with a `-` sign like in `-1second` which would evaluate to
    /// a `Duration::negative(1, 0)`. But it can also be indicated by the `ago` keyword if the
    /// `allow_ago` settings is set to `true`.
    pub allow_negative: bool,

    /// This delimiter may occur within a duration
    ///
    /// The occurrences of this delimiter depend on other configuration settings:
    /// * If `allow_sign_delimiter` is set, this delimiter may separate the sign from the number
    /// * If `allow_time_unit_delimiter` is set, this delimiter may separate the number from the
    ///   time unit
    /// * If `allow_ago` is set, this delimiter has to separate the time unit from the `ago`
    ///   keyword
    ///
    /// If parsing with [`NumbersLike`] numerals, then this delimiter has to separate the numeral
    /// from the time unit.
    pub inner_delimiter: Delimiter,

    /// This delimiter may occur to separate multiple durations and [`Config::conjunctions`]
    ///
    /// This delimiter is used only if `allow_multiple` is set.
    pub outer_delimiter: Delimiter,

    /// Allow the ago keyword to indicate a negative duration (Default: false)
    ///
    /// The `ago` keyword must be delimited by the [`Config::inner_delimiter`] from the time unit.
    /// IMPORTANT: If `allow_ago` is set to `true` it's also necessary to set `allow_negative` to
    /// `true` explicitly.
    ///
    /// The `ago` keyword can only occur in conjunction with a time unit like in `"1second ago"`
    /// which would result in a `Duration::negative(1, 0)`. `"1 ago"` and `"1ago"` would result in
    /// a [`crate::error::ParseError`].
    pub allow_ago: bool,

    /// Allow the [`Config::inner_delimiter`] between the sign and a number, time keyword ...
    /// (Default: `false`)
    ///
    /// For example, setting the `inner_delimiter` to `|byte| matches!(byte, b' ' | b'\n')` would
    /// parse strings like `"+1ms"`, `"- 1ms"`, `"+   yesterday"`, `"+\n4e2000years"` ...
    pub allow_sign_delimiter: bool,

    /// Allow a [`Delimiter`] between the number and time unit (Default: `false`)
    ///
    /// This setting does not enforce the [`Config::inner_delimiter`], so time units directly
    /// following the number are still parsed without error. A delimiter may occur multiple times.
    ///
    /// For example, setting this option with an `inner_delimiter` of `|byte| matches!(byte, b' ' |
    /// b'\n')` would parse strings like `"1ms"`, `"1 ms"`, `"3.2 minutes"`, `"4e2000 \n years"`
    /// ...
    pub allow_time_unit_delimiter: bool,
}

impl<'a> Default for Config<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Config<'a> {
    /// Create a new default configuration
    ///
    /// Please see the documentation of the fields of this `struct` for more information and their
    /// default values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::Config;
    /// use fundu_core::time::{Multiplier, TimeUnit};
    ///
    /// const DEFAULT_CONFIG: Config = Config::new();
    ///
    /// assert_eq!(DEFAULT_CONFIG.default_unit, TimeUnit::Second);
    /// assert_eq!(DEFAULT_CONFIG.allow_time_unit_delimiter, false);
    /// assert_eq!(DEFAULT_CONFIG.default_multiplier, Multiplier(1, 0));
    /// assert_eq!(DEFAULT_CONFIG.disable_exponent, false);
    /// assert_eq!(DEFAULT_CONFIG.disable_fraction, false);
    /// assert_eq!(DEFAULT_CONFIG.number_is_optional, false);
    /// assert_eq!(DEFAULT_CONFIG.disable_infinity, false);
    /// assert_eq!(DEFAULT_CONFIG.allow_multiple, false);
    /// assert_eq!(DEFAULT_CONFIG.conjunctions, None);
    /// assert_eq!(DEFAULT_CONFIG.allow_negative, false);
    /// assert_eq!(DEFAULT_CONFIG.allow_ago, false);
    /// ```
    pub const fn new() -> Self {
        Self {
            allow_time_unit_delimiter: false,
            default_unit: DEFAULT_TIME_UNIT,
            default_multiplier: Multiplier(1, 0),
            disable_exponent: false,
            disable_fraction: false,
            number_is_optional: false,
            disable_infinity: false,
            allow_multiple: false,
            conjunctions: None,
            allow_negative: false,
            allow_ago: false,
            allow_sign_delimiter: false,
            inner_delimiter: |byte| byte.is_ascii_whitespace(),
            outer_delimiter: |byte| byte.is_ascii_whitespace(),
        }
    }

    /// Convenience method to use the [`ConfigBuilder`] to build this `Config`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    /// use fundu_core::time::TimeUnit;
    ///
    /// let config = Config::builder()
    ///     .disable_infinity()
    ///     .allow_negative()
    ///     .build();
    ///
    /// assert_eq!(config.disable_infinity, true);
    /// assert_eq!(config.allow_negative, true);
    /// ```
    pub const fn builder() -> ConfigBuilder<'a> {
        ConfigBuilder::new()
    }
}

/// A builder to create a [`Config`]
///
/// The `ConfigBuilder` starts with the default `Config` and can create the `Config` with the
/// [`ConfigBuilder::build`] method as const at compile time.
///
/// # Examples
///
/// ```rust
/// use fundu_core::config::{Config, ConfigBuilder};
/// use fundu_core::time::TimeUnit;
///
/// const CONFIG: Config = ConfigBuilder::new()
///     .default_unit(TimeUnit::MilliSecond)
///     .disable_fraction()
///     .build();
///
/// assert_eq!(CONFIG.default_unit, TimeUnit::MilliSecond);
/// assert_eq!(CONFIG.disable_fraction, true);
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct ConfigBuilder<'a> {
    config: Config<'a>,
}

impl<'a> ConfigBuilder<'a> {
    /// Create a new `ConfigBuilder` with the default [`Config`] as base.
    ///
    /// Note the `ConfigBuilder` can build the [`Config`] as const at compile time if needed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::ConfigBuilder;
    ///
    /// let config = ConfigBuilder::new().allow_negative().build();
    ///
    /// assert_eq!(config.allow_negative, true);
    /// ```
    pub const fn new() -> Self {
        Self {
            config: Config::new(),
        }
    }

    /// Build the [`Config`] with the configuration of this builder
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().disable_fraction().build();
    ///
    /// assert_eq!(CONFIG.disable_fraction, true);
    /// ```
    pub const fn build(self) -> Config<'a> {
        self.config
    }

    /// Allow a [`Delimiter`] between the number and time unit (Default: `None`)
    ///
    /// See also the documentation of [`Config::allow_time_unit_delimiter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().allow_time_unit_delimiter().build();
    ///
    /// assert!(CONFIG.allow_time_unit_delimiter);
    /// ```
    ///
    /// or with another whitespace definition for the `inner_delimiter`
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new()
    ///     .allow_time_unit_delimiter()
    ///     .inner_delimiter(|byte| byte == b' ')
    ///     .build();
    ///
    /// assert!(CONFIG.allow_time_unit_delimiter);
    /// assert!((CONFIG.inner_delimiter)(b' '));
    /// ```
    pub const fn allow_time_unit_delimiter(mut self) -> Self {
        self.config.allow_time_unit_delimiter = true;
        self
    }

    /// The [`TimeUnit`] the parser applies if no time unit was given (Default: `TimeUnit::Second`)
    ///
    /// See also the documentation of [`Config::default_unit`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    /// use fundu_core::time::TimeUnit;
    ///
    /// const CONFIG: Config = ConfigBuilder::new()
    ///     .default_unit(TimeUnit::MilliSecond)
    ///     .build();
    ///
    /// assert_eq!(CONFIG.default_unit, TimeUnit::MilliSecond);
    /// ```
    pub const fn default_unit(mut self, time_unit: TimeUnit) -> Self {
        self.config.default_unit = time_unit;
        self
    }

    /// Disable parsing an exponent (Default: `false`)
    ///
    /// See also the documentation of [`Config::disable_exponent`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().disable_exponent().build();
    ///
    /// assert_eq!(CONFIG.disable_exponent, true);
    /// ```
    pub const fn disable_exponent(mut self) -> Self {
        self.config.disable_exponent = true;
        self
    }

    /// Disable parsing a fraction (Default: `false`)
    ///
    /// See also the documentation of [`Config::disable_fraction`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().disable_fraction().build();
    ///
    /// assert_eq!(CONFIG.disable_fraction, true);
    /// ```
    pub const fn disable_fraction(mut self) -> Self {
        self.config.disable_fraction = true;
        self
    }

    /// Disable parsing infinity (Default: `false`)
    ///
    /// See also the documentation of [`Config::disable_infinity`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().disable_infinity().build();
    ///
    /// assert_eq!(CONFIG.disable_infinity, true);
    /// ```
    pub const fn disable_infinity(mut self) -> Self {
        self.config.disable_infinity = true;
        self
    }

    /// Make a number in the input string optional (Default: `false`)
    ///
    /// See also the documentation of [`Config::number_is_optional`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().number_is_optional().build();
    ///
    /// assert_eq!(CONFIG.number_is_optional, true);
    /// ```
    pub const fn number_is_optional(mut self) -> Self {
        self.config.number_is_optional = true;
        self
    }

    /// Allow parsing negative durations (Default: `false`)
    ///
    /// See also the documentation of [`Config::allow_negative`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().allow_negative().build();
    ///
    /// assert_eq!(CONFIG.allow_negative, true);
    /// ```
    pub const fn allow_negative(mut self) -> Self {
        self.config.allow_negative = true;
        self
    }

    /// This setting allows multiple `durations` in the source string (Default: `None`)
    ///
    /// See also the documentation of [`Config::allow_multiple`] and
    /// [`Config::conjunctions`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder, Delimiter};
    ///
    /// const CONJUNCTIONS: &[&str] = &["and", ",", "also"];
    /// const CONFIG: Config = ConfigBuilder::new()
    ///     .parse_multiple(Some(CONJUNCTIONS))
    ///     .build();
    ///
    /// assert_eq!(CONFIG.conjunctions, Some(CONJUNCTIONS));
    /// ```
    pub const fn parse_multiple(mut self, conjunctions: Option<&'a [&'a str]>) -> Self {
        self.config.allow_multiple = true;
        self.config.conjunctions = conjunctions;
        self
    }

    /// Allow the ago keyword delimited by a [`Delimiter`] to indicate a negative duration
    /// (Default: `None`)
    ///
    /// See also the documentation of [`Config::allow_ago`]. Setting `allow_ago` with this
    /// method also enables parsing negative durations like with [`ConfigBuilder::allow_negative`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().allow_ago().build();
    ///
    /// assert!(CONFIG.allow_ago);
    /// assert!(CONFIG.allow_negative);
    /// ```
    pub const fn allow_ago(mut self) -> Self {
        self.config.allow_ago = true;
        self.config.allow_negative = true;
        self
    }

    /// Allow a [`Delimiter`] between the sign and a number, time keyword ... (Default: `None`)
    ///
    /// See also the documentation of [`Config::allow_sign_delimiter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new().allow_sign_delimiter().build();
    ///
    /// assert!(CONFIG.allow_sign_delimiter);
    /// ```
    pub const fn allow_sign_delimiter(mut self) -> Self {
        self.config.allow_sign_delimiter = true;
        self
    }

    /// Set the inner [`Delimiter`] to something different then the default
    /// [`u8::is_ascii_whitespace`]
    ///
    /// Where the inner delimiter occurs, depends on other options:
    /// * [`ConfigBuilder::allow_sign_delimiter`]: Between the sign and the number
    /// * [`ConfigBuilder::allow_time_unit_delimiter`]: Between the number and the time unit
    /// * [`ConfigBuilder::allow_ago`]: Between the time unit and the `ago` keyword
    /// * If [`NumbersLike`] numerals are used, between the numeral and the time unit
    ///
    /// See also the documentation of [`Config::inner_delimiter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new()
    ///     .inner_delimiter(|byte| byte == b'#')
    ///     .build();
    ///
    /// assert!((CONFIG.inner_delimiter)(b'#'));
    /// ```
    pub const fn inner_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.config.inner_delimiter = delimiter;
        self
    }

    /// Set the outer [`Delimiter`] to something different then the default
    /// [`u8::is_ascii_whitespace`]
    ///
    /// The outer delimiter is used to separate multiple durations like in `1second 1minute` and is
    /// therefore used only if [`Config::allow_multiple`] is set to true. If
    /// `conjunctions` are set, this delimiter also separates the conjunction from the durations
    /// (like in `1second and 1minute`)
    ///
    /// See also the documentation of [`Config::outer_delimiter`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::config::{Config, ConfigBuilder};
    ///
    /// const CONFIG: Config = ConfigBuilder::new()
    ///     .outer_delimiter(|byte| byte == b'#')
    ///     .build();
    ///
    /// assert!((CONFIG.outer_delimiter)(b'#'));
    /// ```
    pub const fn outer_delimiter(mut self, delimiter: Delimiter) -> Self {
        self.config.outer_delimiter = delimiter;
        self
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    pub fn test_time_unit() -> TimeUnit {
        TimeUnit::MilliSecond
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_default_for_config() {
        assert_eq!(Config::default(), Config::new());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_default_for_config_builder() {
        assert_eq!(ConfigBuilder::new().build(), Config::new());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_method_builder() {
        assert_eq!(Config::builder().build(), Config::new());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_allow_delimiter() {
        let config = ConfigBuilder::new().allow_time_unit_delimiter().build();

        let mut expected = Config::new();
        expected.allow_time_unit_delimiter = true;

        assert_eq!(config, expected);
    }

    #[rstest]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_default_unit(test_time_unit: TimeUnit) {
        let config = ConfigBuilder::new().default_unit(test_time_unit).build();

        let mut expected = Config::new();
        expected.default_unit = test_time_unit;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_disable_exponent() {
        let config = ConfigBuilder::new().disable_exponent().build();

        let mut expected = Config::new();
        expected.disable_exponent = true;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_disable_fraction() {
        let config = ConfigBuilder::new().disable_fraction().build();

        let mut expected = Config::new();
        expected.disable_fraction = true;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_number_is_optional() {
        let config = ConfigBuilder::new().number_is_optional().build();

        let mut expected = Config::new();
        expected.number_is_optional = true;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_disable_infinity() {
        let config = ConfigBuilder::new().disable_infinity().build();

        let mut expected = Config::new();
        expected.disable_infinity = true;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_allow_negative() {
        let config = ConfigBuilder::new().allow_negative().build();

        let mut expected = Config::new();
        expected.allow_negative = true;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_parse_multiple_when_no_conjunctions() {
        let config = ConfigBuilder::new().parse_multiple(None).build();

        let mut expected = Config::new();
        expected.allow_multiple = true;
        expected.conjunctions = None;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_parse_multiple_when_conjunctions() {
        let conjunctions = &["and", ","];
        let config = ConfigBuilder::new()
            .parse_multiple(Some(conjunctions))
            .build();

        let mut expected = Config::new();
        expected.allow_multiple = true;
        expected.conjunctions = Some(conjunctions);

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_allow_ago() {
        let config = ConfigBuilder::new().allow_ago().build();

        let mut expected = Config::new();
        expected.allow_ago = true;
        expected.allow_negative = true;

        assert_eq!(config, expected);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_config_builder_allow_sign_delimiter() {
        let config = ConfigBuilder::new().allow_sign_delimiter().build();

        let mut expected = Config::new();
        expected.allow_sign_delimiter = true;

        assert_eq!(config, expected);
    }
}
