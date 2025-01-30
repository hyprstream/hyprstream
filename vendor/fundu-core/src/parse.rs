// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! This module is the working horse of the parser. Public interfaces to the parser are located in
//! the main library `lib.rs`.

use std::cmp::Ordering::{Equal, Greater, Less};
use std::str::Utf8Error;
use std::time::Duration as StdDuration;

use crate::config::{Config, Delimiter, NumbersLike, DEFAULT_CONFIG};
use crate::error::ParseError;
use crate::time::{Duration, Multiplier, TimeUnit, TimeUnitsLike};
use crate::util::POW10;

pub const ATTOS_PER_SEC: u64 = 1_000_000_000_000_000_000;
pub const ATTOS_PER_SEC_U128: u128 = ATTOS_PER_SEC as u128;
pub const ATTOS_PER_NANO: u64 = 1_000_000_000;
pub const ATTOS_PER_NANO_U128: u128 = ATTOS_PER_NANO as u128;
pub const NANOS_PER_SEC: u64 = 1_000_000_000;
pub const NANOS_PER_SEC_U128: u128 = NANOS_PER_SEC as u128;
pub const ATTOS_MAX: u64 = 999_999_999_999_999_999;
pub const SECONDS_MAX: u64 = u64::MAX;
pub const SECONDS_AND_ATTOS_MAX: (u64, u64) = (SECONDS_MAX, ATTOS_MAX);

/// The core duration parser to parse strings into a [`crate::time::Duration`]
///
/// To be able to use the [`Parser::parse`] method an implementation of the
/// [`crate::time::TimeUnitsLike`] trait is needed for the time units (even if there are no time
/// units) and optionally for time keywords (like `yesterday` and `tomorrow` etc.). Optionally, an
/// implementation of the [`NumbersLike`] trait can be provided, too. The `custom` and `standard`
/// features have such implementations and their parsers are more convenient to use than using this
/// parser directly. However, for example, the `custom` feature's [`fundu::CustomDurationParser`]
/// cannot be fully built in `const` context and is a slightly slower than this parser. So, using
/// this parser is more involved but if maximum performance and building a parser in `const` context
/// is wanted then this parser is the better choice.
///
/// # Examples
///
/// ```rust
/// use fundu_core::error::ParseError;
/// use fundu_core::parse::Parser;
/// use fundu_core::time::TimeUnit::*;
/// use fundu_core::time::{Duration, Multiplier, TimeUnit, TimeUnitsLike};
///
/// struct TimeUnits {}
///
/// impl TimeUnitsLike for TimeUnits {
///     #[inline]
///     fn is_empty(&self) -> bool {
///         false
///     }
///
///     #[inline]
///     fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
///         match identifier {
///             "s" | "sec" | "secs" => Some((Second, Multiplier(1, 0))),
///             "m" | "min" | "mins" => Some((Minute, Multiplier(1, 0))),
///             _ => None,
///         }
///     }
/// }
///
/// let parser = Parser::new();
/// let time_units = TimeUnits {};
///
/// assert_eq!(
///     parser.parse("1.0s", &time_units, None, None),
///     Ok(Duration::positive(1, 0))
/// );
/// assert_eq!(
///     parser.parse("1min", &time_units, None, None),
///     Ok(Duration::positive(60, 0))
/// );
/// assert_eq!(
///     parser.parse("1ms", &time_units, None, None),
///     Err(ParseError::TimeUnit(
///         1,
///         "Invalid time unit: 'ms'".to_string()
///     ))
/// );
/// ```
///
/// [`fundu::CustomDurationParser`]: https://docs.rs/fundu/latest/fundu/struct.CustomDurationParser.html
#[derive(Debug, PartialEq, Eq)]
pub struct Parser<'a> {
    /// The [`crate::config::Config`] of this [`Parser`]
    ///
    /// For convenience, there are also the const [`Parser::new`] and const [`Parser::with_config`]
    /// methods to create a new [`Parser`].
    pub config: Config<'a>,
}

impl<'a> Parser<'a> {
    /// Convenience method to create a new parser with the default [`crate::config::Config`]
    pub const fn new() -> Self {
        Self {
            config: DEFAULT_CONFIG,
        }
    }

    /// Convenience method to create a new parser with the the given [`crate::config::Config`]
    pub const fn with_config(config: Config<'a>) -> Self {
        Self { config }
    }

    #[inline]
    pub fn parse_multiple(
        &self,
        source: &str,
        time_units: &dyn TimeUnitsLike,
        keywords: Option<&dyn TimeUnitsLike>,
        numerals: Option<&dyn NumbersLike>,
    ) -> Result<Duration, ParseError> {
        let mut duration = Duration::ZERO;

        let mut parser = &mut ReprParserMultiple::new(source);
        loop {
            let (mut duration_repr, maybe_parser) =
                parser.parse(&self.config, time_units, keywords, numerals)?;
            let parsed_duration = duration_repr.parse()?;
            duration = if !self.config.allow_negative && parsed_duration.is_negative() {
                return Err(ParseError::NegativeNumber);
            } else if parsed_duration.is_zero() {
                duration
            } else if duration.is_zero() {
                parsed_duration
            } else {
                duration.saturating_add(parsed_duration)
            };
            match maybe_parser {
                Some(p) => parser = p,
                None => break Ok(duration),
            }
        }
    }

    #[inline]
    pub fn parse_single(
        &self,
        source: &str,
        time_units: &dyn TimeUnitsLike,
        keywords: Option<&dyn TimeUnitsLike>,
        numerals: Option<&dyn NumbersLike>,
    ) -> Result<Duration, ParseError> {
        ReprParserSingle::new(source)
            .parse(&self.config, time_units, keywords, numerals)
            .and_then(|mut duration_repr| {
                duration_repr.parse().and_then(|duration| {
                    if !self.config.allow_negative && duration.is_negative() {
                        Err(ParseError::NegativeNumber)
                    } else {
                        Ok(duration)
                    }
                })
            })
    }

    /// Parse the `source` string into a saturating [`crate::time::Duration`]
    ///
    /// This method needs a struct implementing the [`crate::time::TimeUnitsLike`] for time units
    /// and optionally for time keywords (like `yesterday`, `tomorrow`). But also [`NumbersLike`]
    /// implementations for words like `one`, `next`, `last` are supported. The `standard` and
    /// `custom` features of `fundu`  offer such implementations and are more convenient to use than
    /// using this method directly. They both provide facades and an own parser which calls this
    /// method.
    ///
    /// # Errors
    ///
    /// Returns a [`crate::error::ParseError`] if the given `source` string is invalid
    ///
    /// # Examples
    ///
    /// An example with a quick and dirty implementation of [`crate::time::TimeUnitsLike`]
    ///
    /// ```rust
    /// use fundu_core::error::ParseError;
    /// use fundu_core::parse::Parser;
    /// use fundu_core::time::TimeUnit::*;
    /// use fundu_core::time::{Duration, Multiplier, TimeUnit, TimeUnitsLike};
    ///
    /// struct TimeUnits {}
    ///
    /// impl TimeUnitsLike for TimeUnits {
    ///     #[inline]
    ///     fn is_empty(&self) -> bool {
    ///         false
    ///     }
    ///
    ///     #[inline]
    ///     fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
    ///         match identifier {
    ///             "s" | "sec" | "secs" => Some((Second, Multiplier(1, 0))),
    ///             "m" | "min" | "mins" => Some((Minute, Multiplier(1, 0))),
    ///             _ => None,
    ///         }
    ///     }
    /// }
    ///
    /// let parser = Parser::new();
    /// let time_units = TimeUnits {};
    ///
    /// assert_eq!(
    ///     parser.parse("1.0s", &time_units, None, None),
    ///     Ok(Duration::positive(1, 0))
    /// );
    /// assert_eq!(
    ///     parser.parse("1min", &time_units, None, None),
    ///     Ok(Duration::positive(60, 0))
    /// );
    /// assert_eq!(
    ///     parser.parse("1ms", &time_units, None, None),
    ///     Err(ParseError::TimeUnit(
    ///         1,
    ///         "Invalid time unit: 'ms'".to_string()
    ///     ))
    /// );
    /// ```
    #[inline]
    pub fn parse(
        &self,
        source: &str,
        time_units: &dyn TimeUnitsLike,
        keywords: Option<&dyn TimeUnitsLike>,
        numerals: Option<&dyn NumbersLike>,
    ) -> Result<Duration, ParseError> {
        if self.config.allow_multiple {
            self.parse_multiple(source, time_units, keywords, numerals)
        } else {
            self.parse_single(source, time_units, keywords, numerals)
        }
    }
}

impl<'a> Default for Parser<'a> {
    fn default() -> Self {
        Self::new()
    }
}

pub trait Parse8Digits {
    // This method is based on the work of Johnny Lee and his blog post
    // https://johnnylee-sde.github.io/Fast-numeric-string-to-int
    unsafe fn parse_8_digits(digits: &[u8]) -> u64 {
        // cov:excl-start
        debug_assert!(
            digits.len() >= 8,
            "Call this method only if digits has length >= 8"
        ); // cov:excl-stop

        // This cast to a more strictly aligned type is safe since we're using
        // ptr.read_unaligned
        #[allow(clippy::cast_ptr_alignment)]
        let ptr = digits.as_ptr().cast::<u64>();
        let mut num = u64::from_le(ptr.read_unaligned());
        num = ((num & 0x0F0F_0F0F_0F0F_0F0F).wrapping_mul(2561)) >> 8i32;
        num = ((num & 0x00FF_00FF_00FF_00FF).wrapping_mul(6_553_601)) >> 16i32;
        num = ((num & 0x0000_FFFF_0000_FFFF).wrapping_mul(42_949_672_960_001)) >> 32i32;
        num
    }
}

#[derive(Debug, PartialEq, Eq, Default, Copy, Clone)]
pub struct Whole(pub usize, pub usize);

impl Parse8Digits for Whole {}

impl Whole {
    pub fn parse_slice(mut seconds: u64, digits: &[u8]) -> Option<u64> {
        if digits.len() >= 8 {
            let mut iter = digits.chunks_exact(8);
            for digits in iter.by_ref() {
                match seconds
                    .checked_mul(100_000_000)
                    // SAFETY: We have chunks of exactly 8 bytes
                    .and_then(|s| s.checked_add(unsafe { Self::parse_8_digits(digits) }))
                {
                    Some(s) => seconds = s,
                    None => {
                        return None;
                    }
                }
            }
            for num in iter.remainder() {
                match seconds
                    .checked_mul(10)
                    .and_then(|s| s.checked_add(u64::from(*num - b'0')))
                {
                    Some(s) => seconds = s,
                    None => {
                        return None;
                    }
                }
            }
        } else {
            for num in digits {
                match seconds
                    .checked_mul(10)
                    .and_then(|s| s.checked_add(u64::from(*num - b'0')))
                {
                    Some(s) => seconds = s,
                    None => {
                        return None;
                    }
                }
            }
        }
        Some(seconds)
    }

    pub fn parse(digits: &[u8], append: Option<&[u8]>, zeros: Option<usize>) -> Option<u64> {
        if digits.is_empty() && append.map_or(true, <[u8]>::is_empty) {
            return Some(0);
        }

        Self::parse_slice(0, digits).and_then(|s| {
            append
                .map_or(Some(s), |append| Self::parse_slice(s, append))
                .and_then(|seconds| {
                    if seconds == 0 {
                        Some(0)
                    } else {
                        match zeros {
                            Some(num_zeros) if num_zeros > 0 => POW10
                                .get(num_zeros)
                                .and_then(|pow| seconds.checked_mul(*pow)),
                            Some(_) | None => Some(seconds),
                        }
                    }
                })
        })
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.1 - self.0
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.1 == self.0
    }
}

#[derive(Debug, PartialEq, Eq, Default, Copy, Clone)]
pub struct Fract(pub usize, pub usize);

impl Parse8Digits for Fract {}

impl Fract {
    pub fn parse_slice(mut attos: u64, max_to_parse: usize, digits: &[u8]) -> (u64, usize) {
        let num_parsable = digits.len().min(max_to_parse);
        if num_parsable >= 8 {
            let mut iter = digits[..num_parsable].chunks_exact(8);
            for digits in iter.by_ref() {
                // SAFETY: We have chunks of exactly 8 bytes
                attos = attos * 100_000_000 + unsafe { Self::parse_8_digits(digits) };
            }
            for num in iter.remainder() {
                attos = attos * 10 + u64::from(*num - b'0');
            }
        } else {
            for num in &digits[..num_parsable] {
                attos = attos * 10 + u64::from(*num - b'0');
            }
        }
        (attos, max_to_parse - num_parsable)
    }

    pub fn parse(digits: &[u8], prepend: Option<&[u8]>, zeros: Option<usize>) -> u64 {
        if digits.is_empty() && prepend.map_or(true, <[u8]>::is_empty) {
            return 0;
        }

        let max_to_parse = match zeros {
            Some(z) if z > 18 => return 0,
            Some(z) => 18 - z,
            None => 18,
        };

        match prepend {
            Some(prepend) if !prepend.is_empty() => {
                let (attos, remainder) = Self::parse_slice(0, max_to_parse, prepend);
                if remainder == 0 {
                    attos
                } else if digits.is_empty() {
                    attos * POW10[remainder]
                } else {
                    let (attos, remainder) = Self::parse_slice(attos, remainder, digits);
                    if remainder > 0 {
                        attos * POW10[remainder]
                    } else {
                        attos
                    }
                }
            }
            Some(_) | None => {
                let (attos, remainder) = Self::parse_slice(0, max_to_parse, digits);
                if remainder > 0 {
                    attos * POW10[remainder]
                } else {
                    attos
                }
            }
        }
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.1 - self.0
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.1 == self.0
    }
}

#[derive(Debug, Default)]
pub struct DurationRepr<'a> {
    pub default_unit: TimeUnit,
    pub unit: Option<TimeUnit>,
    pub is_negative: Option<bool>,
    pub is_infinite: bool,
    pub whole: Option<Whole>,
    pub fract: Option<Fract>,
    pub input: &'a [u8],
    pub exponent: i16,
    pub multiplier: Multiplier,
    pub numeral: Option<Multiplier>,
}

impl<'a> DurationRepr<'a> {
    #[allow(clippy::too_many_lines)]
    pub fn parse(&mut self) -> Result<Duration, ParseError> {
        if self.is_infinite {
            return Ok(Duration::from_std(
                self.is_negative.unwrap_or_default(),
                StdDuration::MAX,
            ));
        }

        if self.whole.is_none() && self.fract.is_none() {
            return if self.numeral.is_some() {
                let time_unit = self.unit.expect("Numeral without time unit");
                let numeral = self.numeral.unwrap();
                let Multiplier(coefficient, exponent) =
                    numeral * time_unit.multiplier() * self.multiplier;

                Ok(self.parse_duration_with_fixed_number(coefficient, exponent))
            // We're here when parsing keywords or time units without a number if the configuration
            // option `number_is_optional` is set.
            } else if self.unit.is_some() {
                let time_unit = self.unit.unwrap_or(self.default_unit);
                let Multiplier(coefficient, exponent) = time_unit.multiplier() * self.multiplier;

                Ok(self.parse_duration_with_fixed_number(coefficient, exponent))
            // We're here if we wouldn't have parsed anything usable.
            } else {
                unreachable!() // cov:excl-line
            };
        }

        let time_unit = self.unit.unwrap_or(self.default_unit);
        // Panic on overflow during the multiplication of the multipliers or adding the exponents
        let Multiplier(coefficient, exponent) = time_unit.multiplier() * self.multiplier;
        if coefficient == 0 {
            return Ok(Duration::ZERO);
        }
        let exponent = i32::from(exponent) + i32::from(self.exponent);

        // The maximum absolute value of the exponent is `2 * abs(i16::MIN)`, so it is safe to cast
        // to usize
        let exponent_abs: usize = exponent.unsigned_abs().try_into().unwrap();
        let duration_is_negative = self.is_negative.unwrap_or_default() ^ coefficient.is_negative();

        // We're operating on slices to minimize runtime costs. Applying the exponent before parsing
        // to integers is necessary, since the exponent can move digits into the to be considered
        // final integer domain.
        let digits = self.input;
        let (seconds, attos) = match (exponent.cmp(&0i32), &self.whole, &self.fract) {
            (Less, Some(whole), fract) if whole.len() > exponent_abs => {
                match Whole::parse(&digits[whole.0..whole.1 - exponent_abs], None, None) {
                    Some(seconds) => {
                        let attos = Fract::parse(
                            fract.map_or_else(|| [].as_ref(), |fract| &digits[fract.0..fract.1]),
                            Some(&digits[whole.1 - exponent_abs..whole.1]),
                            None,
                        );
                        (seconds, attos)
                    }
                    None if duration_is_negative => return Ok(Duration::MIN),
                    None => return Ok(Duration::MAX),
                }
            }
            (Less, whole, fract) => {
                let attos = match fract {
                    Some(fract) if fract.is_empty() => Fract::parse(
                        whole.map_or_else(|| [].as_ref(), |whole| &digits[whole.0..whole.1]),
                        None,
                        Some(exponent_abs - whole.map_or(0, |w| w.len())),
                    ),
                    Some(fract) => Fract::parse(
                        &digits[fract.0..fract.1],
                        whole.and_then(|whole| {
                            (!whole.is_empty()).then(|| &digits[whole.0..whole.1])
                        }),
                        Some(exponent_abs - whole.map_or(0, |w| w.len())),
                    ),
                    None => Fract::parse(
                        whole.map_or_else(|| [].as_ref(), |whole| &digits[whole.0..whole.1]),
                        None,
                        Some(exponent_abs - whole.map_or(0, |w| w.len())),
                    ),
                };
                (0, attos)
            }
            (Equal, whole, fract) => {
                match whole.map_or(Some(0), |whole| {
                    Whole::parse(&digits[whole.0..whole.1], None, None)
                }) {
                    Some(seconds) => {
                        let attos = fract.map_or(0, |fract| {
                            Fract::parse(&digits[fract.0..fract.1], None, None)
                        });
                        (seconds, attos)
                    }
                    None if duration_is_negative => return Ok(Duration::MIN),
                    None => return Ok(Duration::MAX),
                }
            }
            (Greater, whole, Some(fract)) if fract.len() > exponent_abs => {
                match Whole::parse(
                    whole.map_or_else(|| [].as_ref(), |whole| &digits[whole.0..whole.1]),
                    Some(&digits[fract.0..fract.0 + exponent_abs]),
                    None,
                ) {
                    Some(seconds) => {
                        let attos =
                            Fract::parse(&digits[fract.0 + exponent_abs..fract.1], None, None);
                        (seconds, attos)
                    }
                    None if duration_is_negative => return Ok(Duration::MIN),
                    None => return Ok(Duration::MAX),
                }
            }
            (Greater, whole, fract) => {
                match Whole::parse(
                    whole.map_or_else(|| [].as_ref(), |whole| &digits[whole.0..whole.1]),
                    fract.map(|fract| &digits[fract.0..fract.1]),
                    Some(exponent_abs - fract.map_or(0, |fract| fract.len())),
                ) {
                    Some(seconds) => (seconds, 0),
                    None if duration_is_negative => return Ok(Duration::MIN),
                    None => return Ok(Duration::MAX),
                }
            }
        };

        Ok(Self::calculate_duration(
            duration_is_negative,
            seconds,
            attos,
            coefficient,
        ))
    }

    #[inline]
    pub fn parse_duration_with_fixed_number(&self, coefficient: i64, exponent: i16) -> Duration {
        if coefficient == 0 {
            return Duration::ZERO;
        }
        let duration_is_negative = coefficient.is_negative() ^ self.is_negative.unwrap_or_default();
        let (seconds, attos) = match exponent.cmp(&0i16) {
            Less if exponent < -18 => return Duration::ZERO,
            Less => (0, POW10[usize::try_from(18 + exponent).unwrap()]),
            Equal => {
                return Duration::from_std(
                    duration_is_negative,
                    StdDuration::new(coefficient.unsigned_abs(), 0),
                );
            }
            Greater if exponent > 19 => {
                return if coefficient.is_negative() {
                    Duration::MIN
                } else {
                    Duration::MAX
                };
            }
            Greater => (POW10[usize::try_from(exponent).unwrap()], 0),
        };

        Self::calculate_duration(duration_is_negative, seconds, attos, coefficient)
    }

    #[inline]
    pub fn calculate_duration(
        is_negative: bool,
        seconds: u64,
        attos: u64,
        coefficient: i64,
    ) -> Duration {
        if (seconds == 0 && attos == 0) || coefficient == 0 {
            Duration::ZERO
        } else if attos == 0 {
            let unsigned_coefficient = coefficient.unsigned_abs();
            match seconds.checked_mul(unsigned_coefficient) {
                Some(s) => Duration::from_std(is_negative, StdDuration::new(s, 0)),
                None if is_negative => Duration::MIN,
                None => Duration::MAX,
            }
        } else if coefficient == 1 || coefficient == -1 {
            Duration::from_std(
                is_negative,
                StdDuration::new(seconds, (attos / ATTOS_PER_NANO).try_into().unwrap()),
            )
        } else {
            let unsigned_coefficient = coefficient.unsigned_abs();
            if let Some(attos) = unsigned_coefficient.checked_mul(attos) {
                match seconds
                    .checked_mul(unsigned_coefficient)
                    .and_then(|s| s.checked_add(attos / ATTOS_PER_SEC))
                {
                    Some(s) => Duration::from_std(
                        is_negative,
                        StdDuration::new(s, ((attos / ATTOS_PER_NANO) % NANOS_PER_SEC) as u32),
                    ),
                    None if is_negative => Duration::MIN,
                    None => Duration::MAX,
                }
            } else {
                let time = seconds.checked_mul(unsigned_coefficient).and_then(|s| {
                    let attos = u128::from(attos) * u128::from(unsigned_coefficient);
                    s.checked_add((attos / ATTOS_PER_SEC_U128).try_into().unwrap())
                        .map(|s| (s, attos))
                });
                match time {
                    Some((s, attos)) => Duration::from_std(
                        is_negative,
                        StdDuration::new(
                            s,
                            ((attos / ATTOS_PER_NANO_U128) % NANOS_PER_SEC_U128) as u32,
                        ),
                    ),
                    None if is_negative => Duration::MIN,
                    None => Duration::MAX,
                }
            }
        }
    }
}

pub struct BytesRange(usize, usize);

pub struct Bytes<'a> {
    pub current_pos: usize, // keep first. Has better performance.
    pub current_byte: Option<&'a u8>,
    pub buffer: Option<(usize, &'a [u8], Option<&'a u8>)>,
    pub input: &'a [u8],
}

impl<'a> Bytes<'a> {
    #[inline]
    pub const fn new(input: &'a [u8]) -> Self {
        Self {
            current_pos: 0,
            current_byte: input.first(),
            input,
            buffer: None,
        }
    }

    #[inline]
    pub fn advance(&mut self) {
        self.current_pos += 1;
        self.current_byte = self.input.get(self.current_pos);
    }

    #[inline]
    pub unsafe fn advance_by(&mut self, num: usize) {
        self.current_pos += num;
        self.current_byte = self.input.get(self.current_pos);
    }

    pub fn advance_to<F>(&mut self, delimiter: F) -> &'a [u8]
    where
        F: Fn(u8) -> bool,
    {
        let start = self.current_pos;
        while let Some(byte) = self.current_byte {
            if delimiter(*byte) {
                break;
            }
            self.advance();
        }
        &self.input[start..self.current_pos]
    }

    pub fn buffered_advance_to<F>(&mut self, delimiter: F) -> &'a [u8]
    where
        F: Fn(u8) -> bool,
    {
        match self.buffer {
            Some((pos, buffer, Some(next))) if pos == self.current_pos && delimiter(*next) => {
                // SAFETY: The buffer length does not exceed the input because we parsed it before
                // at the same position
                unsafe { self.advance_by(buffer.len()) };
                buffer
            }
            Some((pos, buffer, None)) if pos == self.current_pos => {
                // SAFETY: The buffer length does not exceed the input because we parsed it before
                // at the same position
                unsafe { self.advance_by(buffer.len()) };
                buffer
            }
            None | Some(_) => {
                let start = self.current_pos;
                while let Some(byte) = self.current_byte {
                    if delimiter(*byte) {
                        break;
                    }
                    self.advance();
                }
                let buffer = &self.input[start..self.current_pos];
                self.buffer = Some((start, buffer, self.current_byte));
                buffer
            }
        }
    }

    #[inline]
    pub fn peek(&self, num: usize) -> Option<&[u8]> {
        self.input.get(self.current_pos..self.current_pos + num)
    }

    #[inline]
    pub fn get_remainder(&self) -> &[u8] {
        &self.input[self.current_pos..]
    }

    #[inline]
    pub unsafe fn get_remainder_str_unchecked(&self) -> &str {
        std::str::from_utf8_unchecked(self.get_remainder())
    }

    #[inline]
    pub fn get_remainder_str(&self) -> Result<&str, ParseError> {
        std::str::from_utf8(self.get_remainder())
            .map_err(|err| ParseError::InvalidInput(err.to_string()))
    }

    #[inline]
    pub fn get_current_str(&self, start: usize) -> Result<&str, Utf8Error> {
        std::str::from_utf8(&self.input[start..self.current_pos])
    }

    #[inline]
    pub fn finish(&mut self) {
        self.current_pos = self.input.len();
        self.current_byte = None;
    }

    #[inline]
    pub fn reset(&mut self, position: usize) {
        self.current_pos = position;
        self.current_byte = self.input.get(position);
    }

    #[inline]
    pub fn parse_digit(&mut self) -> Option<u8> {
        self.current_byte.and_then(|byte| {
            let digit = byte.wrapping_sub(b'0');
            (digit < 10).then(|| {
                self.advance();
                *byte
            })
        })
    }

    pub fn parse_digits_strip_zeros(&mut self) -> BytesRange {
        const ASCII_EIGHT_ZEROS: u64 = 0x3030_3030_3030_3030;

        debug_assert!(self.current_byte.map_or(false, u8::is_ascii_digit)); // cov:excl-stop

        let mut start = self.current_pos;
        let mut strip_leading_zeros = true;
        while let Some(eight) = self.parse_8_digits() {
            if strip_leading_zeros {
                if eight == ASCII_EIGHT_ZEROS {
                    start += 8;
                } else {
                    strip_leading_zeros = false;

                    // eight is little endian so we need to count the trailing zeros
                    let leading_zeros = (eight - ASCII_EIGHT_ZEROS).trailing_zeros() / 8;
                    start += leading_zeros as usize;
                }
            }
        }

        while let Some(byte) = self.current_byte {
            let digit = byte.wrapping_sub(b'0');
            if digit < 10 {
                if strip_leading_zeros {
                    if digit == 0 {
                        start += 1;
                    } else {
                        strip_leading_zeros = false;
                    }
                }
                self.advance();
            } else {
                break;
            }
        }

        BytesRange(start, self.current_pos)
    }

    pub fn parse_digits(&mut self) -> BytesRange {
        debug_assert!(self.current_byte.map_or(false, u8::is_ascii_digit)); // cov:excl-stop

        let start = self.current_pos;
        while self.parse_8_digits().is_some() {}
        while self.parse_digit().is_some() {}

        BytesRange(start, self.current_pos)
    }

    /// This method is based on the work of Daniel Lemire and his blog post
    /// <https://lemire.me/blog/2018/09/30/quickly-identifying-a-sequence-of-digits-in-a-string-of-characters/>
    pub fn parse_8_digits(&mut self) -> Option<u64> {
        self.input
            .get(self.current_pos..(self.current_pos + 8))
            .and_then(|digits| {
                // This cast to a more strictly aligned type is safe since we're using
                // ptr.read_unaligned
                #[allow(clippy::cast_ptr_alignment)]
                let ptr = digits.as_ptr().cast::<u64>();
                // SAFETY: We just ensured there are 8 bytes
                let num = u64::from_le(unsafe { ptr.read_unaligned() });
                ((num & (num.wrapping_add(0x0606_0606_0606_0606)) & 0xf0f0_f0f0_f0f0_f0f0)
                    == 0x3030_3030_3030_3030)
                    .then(|| {
                        // SAFETY: We just ensured there are 8 bytes
                        unsafe { self.advance_by(8) }
                        num
                    })
            })
    }

    #[inline]
    pub fn next_is_ignore_ascii_case(&self, word: &[u8]) -> bool {
        self.peek(word.len())
            .map_or(false, |bytes| bytes.eq_ignore_ascii_case(word))
    }

    #[inline]
    pub const fn is_end_of_input(&self) -> bool {
        self.current_byte.is_none()
    }

    #[inline]
    pub fn check_end_of_input(&self) -> Result<(), ParseError> {
        self.current_byte.map_or(Ok(()), |_| {
            self.get_remainder_str().and_then(|remainder| {
                Err(ParseError::Syntax(
                    self.current_pos,
                    format!("Expected end of input but found: '{remainder}'"),
                ))
            })
        })
    }

    pub fn try_consume_delimiter(&mut self, delimiter: Delimiter) -> Result<(), ParseError> {
        debug_assert!(delimiter(*self.current_byte.unwrap())); // cov:excl-line
        if self.current_pos == 0 {
            return Err(ParseError::Syntax(
                0,
                "Input may not start with a delimiter".to_owned(),
            ));
        }

        let start = self.current_pos;
        self.advance();
        while let Some(byte) = self.current_byte {
            if delimiter(*byte) {
                self.advance();
            } else {
                break;
            }
        }

        match self.current_byte {
            Some(_) => Ok(()),
            None => Err(ParseError::Syntax(
                start,
                "Input may not end with a delimiter".to_owned(),
            )),
        }
    }
}

pub trait ReprParserTemplate<'a> {
    type Output;

    fn bytes(&mut self) -> &mut Bytes<'a>;

    fn make_output(&'a mut self, duration_repr: DurationRepr<'a>) -> Self::Output;

    fn parse_infinity_remainder(
        &'a mut self,
        duration_repr: DurationRepr<'a>,
        config: &'a Config,
    ) -> Result<Self::Output, ParseError>;

    fn parse_keyword(
        &mut self,
        keywords: Option<&dyn TimeUnitsLike>,
        config: &'a Config,
    ) -> Result<Option<(TimeUnit, Multiplier)>, ParseError>;

    fn parse_time_unit(
        &mut self,
        config: &'a Config,
        time_units: &dyn TimeUnitsLike,
    ) -> Result<Option<(TimeUnit, Multiplier)>, ParseError>;

    fn parse_number_time_unit(
        &mut self,
        duration_repr: &mut DurationRepr<'a>,
        config: &'a Config,
        time_units: &dyn TimeUnitsLike,
    ) -> Result<bool, ParseError>;

    fn finalize(
        &'a mut self,
        duration_repr: DurationRepr<'a>,
        config: &'a Config,
    ) -> Result<Self::Output, ParseError>;

    #[inline]
    fn parse_whole(&mut self) -> Whole {
        let BytesRange(start, end) = self.bytes().parse_digits_strip_zeros();
        Whole(start, end)
    }

    #[inline]
    fn parse_numeral(
        &'_ mut self,
        numerals: Option<&'a dyn NumbersLike>,
        config: &'a Config,
    ) -> Result<Option<(&'a str, Multiplier)>, ParseError> {
        if let Some(numerals) = numerals {
            let bytes = self.bytes();
            let start = bytes.current_pos;
            let buffer = bytes.buffered_advance_to(config.inner_delimiter);

            if buffer.is_empty() {
                // Haven't found a way to trigger this line, so it's excluded from coverage for now
                return Ok(None); // cov:excl-line
            }
            // SAFETY: we've only parsed valid utf-8 up to this point and the delimiter only matches
            // ascii
            let string = unsafe { std::str::from_utf8_unchecked(buffer) };
            return match numerals.get(string) {
                None => {
                    bytes.reset(start);
                    Ok(None)
                }
                some_option => match bytes.current_byte {
                    Some(byte) if (config.inner_delimiter)(*byte) => {
                        bytes.try_consume_delimiter(config.inner_delimiter)?;
                        Ok(some_option.map(|m| (string, m)))
                    }
                    None | Some(_) => {
                        bytes.reset(start);
                        Ok(None)
                    }
                },
            };
        }
        Ok(None)
    }

    fn parse(
        &'a mut self,
        config: &'a Config,
        time_units: &dyn TimeUnitsLike,
        keywords: Option<&dyn TimeUnitsLike>,
        numerals: Option<&'a dyn NumbersLike>,
    ) -> Result<Self::Output, ParseError> {
        if self.bytes().current_byte.is_none() {
            return Err(ParseError::Empty);
        }

        let mut duration_repr = DurationRepr {
            default_unit: config.default_unit,
            input: self.bytes().input,
            ..Default::default()
        };

        self.parse_number_sign(&mut duration_repr, config)?;

        // parse infinity, keywords, ... or the whole number part of the input
        match self.bytes().current_byte.copied() {
            Some(byte) if byte.is_ascii_digit() => {
                duration_repr.whole = Some(self.parse_whole());
            }
            Some(b'.') => {}
            Some(_)
                if !config.disable_infinity && self.bytes().next_is_ignore_ascii_case(b"inf") =>
            {
                // SAFETY: We just checked that there are at least 3 bytes
                unsafe { self.bytes().advance_by(3) }
                return self.parse_infinity_remainder(duration_repr, config);
            }
            Some(_) => {
                if let Some((unit, multi)) = self.parse_keyword(keywords, config)? {
                    duration_repr.unit = Some(unit);
                    duration_repr.multiplier = multi;
                    return self.finalize(duration_repr, config);
                }
                if config.number_is_optional {
                    let start = self.bytes().current_pos;
                    match self.parse_time_unit(config, time_units)? {
                        Some((time_unit, multiplier)) => {
                            duration_repr.unit = Some(time_unit);
                            duration_repr.multiplier = multiplier;
                            return self.finalize(duration_repr, config);
                        }
                        None => {
                            self.bytes().reset(start);
                        }
                    }
                }
                if let Some((id, numeral)) = self.parse_numeral(numerals, config)? {
                    match self.parse_time_unit(config, time_units)? {
                        Some((time_unit, multiplier)) => {
                            duration_repr.numeral = Some(numeral);
                            duration_repr.unit = Some(time_unit);
                            duration_repr.multiplier = multiplier;
                            return self.finalize(duration_repr, config);
                        }
                        None if time_units.is_empty() => {
                            return Err(ParseError::TimeUnit(
                                self.bytes().current_pos,
                                format!("Found numeral '{id}' without time units being defined"),
                            ));
                        }
                        None => {
                            return Err(ParseError::TimeUnit(
                                self.bytes().current_pos,
                                format!("Found numeral '{id}' without a time unit"),
                            ));
                        }
                    }
                }
                return self
                    .bytes()
                    .get_remainder_str()
                    .and_then(|remainder| Err(ParseError::InvalidInput(remainder.to_owned())));
            }
            // This is currently unreachable code since empty input and a standalone sign are
            // already handled as errors before. However, keep this code as safety net.
            // cov:excl-start
            None => {
                return Err(ParseError::Syntax(
                    self.bytes().current_pos,
                    "Unexpected end of input".to_owned(),
                ));
            } // cov:excl-stop
        }

        if !self.parse_number_fraction(&mut duration_repr, config.disable_fraction)? {
            return Ok(self.make_output(duration_repr));
        }

        if !self.parse_number_exponent(&mut duration_repr, config.disable_exponent)? {
            return Ok(self.make_output(duration_repr));
        }

        if !self.parse_number_delimiter(
            config
                .allow_time_unit_delimiter
                .then_some(config.inner_delimiter),
        )? {
            return Ok(self.make_output(duration_repr));
        }

        if !self.parse_number_time_unit(&mut duration_repr, config, time_units)? {
            // Currently unreachable but let's keep it for clarity and safety especially because
            // the parse_number_time_unit must be implemented by this traits implementations
            return Ok(self.make_output(duration_repr)); // cov:excl-line
        }

        self.finalize(duration_repr, config)
    }

    /// Parse and consume the sign if present. Return true if sign is negative.
    fn parse_sign_is_negative(&mut self) -> Result<Option<bool>, ParseError> {
        let bytes = self.bytes();
        match bytes.current_byte {
            Some(byte) if *byte == b'+' => {
                bytes.advance();
                Ok(Some(false))
            }
            Some(byte) if *byte == b'-' => {
                bytes.advance();
                Ok(Some(true))
            }
            Some(_) => Ok(None),
            None => Err(ParseError::Syntax(
                bytes.current_pos,
                "Unexpected end of input".to_owned(),
            )),
        }
    }

    fn parse_number_sign(
        &mut self,
        duration_repr: &mut DurationRepr,
        config: &Config,
    ) -> Result<(), ParseError> {
        if let Some(is_negative) = self.parse_sign_is_negative()? {
            duration_repr.is_negative = Some(is_negative);

            let bytes = self.bytes();
            match bytes.current_byte {
                Some(byte) if config.allow_sign_delimiter && (config.inner_delimiter)(*byte) => {
                    return bytes.try_consume_delimiter(config.inner_delimiter);
                }
                Some(_) => {}
                None => {
                    return Err(ParseError::Syntax(
                        bytes.current_pos,
                        "Unexpected end of input. Sign without a number".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn parse_fract(&mut self) -> Fract {
        let BytesRange(start, end) = self.bytes().parse_digits();
        Fract(start, end)
    }

    fn parse_number_fraction(
        &mut self,
        duration_repr: &mut DurationRepr<'a>,
        disable_fraction: bool,
    ) -> Result<bool, ParseError> {
        let bytes = self.bytes();
        match bytes.current_byte {
            Some(byte) if *byte == b'.' && !disable_fraction => {
                bytes.advance();
                let fract = match bytes.current_byte {
                    Some(byte) if byte.is_ascii_digit() => Some(self.parse_fract()),
                    Some(_) | None if duration_repr.whole.is_none() => {
                        // Use the decimal point as anchor for the error position. Subtraction by 1
                        // is safe since we were advancing by one before.
                        return Err(ParseError::Syntax(
                            bytes.current_pos - 1,
                            "Either the whole number part or the fraction must be present"
                                .to_owned(),
                        ));
                    }
                    Some(_) => Some(Fract(self.bytes().current_pos, self.bytes().current_pos)),
                    None => {
                        duration_repr.fract =
                            Some(Fract(self.bytes().current_pos, self.bytes().current_pos));
                        return Ok(false);
                    }
                };
                duration_repr.fract = fract;
                Ok(true)
            }
            Some(byte) if *byte == b'.' => Err(ParseError::Syntax(
                bytes.current_pos,
                "No fraction allowed".to_owned(),
            )),
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    fn parse_exponent(&mut self) -> Result<i16, ParseError> {
        let is_negative = self.parse_sign_is_negative()?.unwrap_or_default();
        let bytes = self.bytes();

        let mut exponent = 0i16;
        let start = bytes.current_pos;
        while let Some(byte) = bytes.current_byte {
            let digit = byte.wrapping_sub(b'0');
            if digit < 10 {
                exponent = if is_negative {
                    match exponent
                        .checked_mul(10)
                        .and_then(|e| e.checked_sub(i16::from(digit)))
                    {
                        Some(exponent) => exponent,
                        None => return Err(ParseError::NegativeExponentOverflow),
                    }
                } else {
                    match exponent
                        .checked_mul(10)
                        .and_then(|e| e.checked_add(i16::from(digit)))
                    {
                        Some(exponent) => exponent,
                        None => return Err(ParseError::PositiveExponentOverflow),
                    }
                };
                bytes.advance();
            } else {
                break;
            }
        }

        if bytes.current_pos - start > 0 {
            Ok(exponent)
        } else if bytes.is_end_of_input() {
            Err(ParseError::Syntax(
                bytes.current_pos,
                "Expected exponent but reached end of input".to_owned(),
            ))
        } else {
            Err(ParseError::Syntax(
                bytes.current_pos,
                "The exponent must have at least one digit".to_owned(),
            ))
        }
    }

    fn parse_number_exponent(
        &mut self,
        duration_repr: &mut DurationRepr<'a>,
        disable_exponent: bool,
    ) -> Result<bool, ParseError> {
        let bytes = self.bytes();
        match bytes.current_byte {
            Some(byte) if byte.eq_ignore_ascii_case(&b'e') && !disable_exponent => {
                bytes.advance();
                duration_repr.exponent = self.parse_exponent()?;
                Ok(true)
            }
            Some(byte) if byte.eq_ignore_ascii_case(&b'e') => Err(ParseError::Syntax(
                bytes.current_pos,
                "No exponent allowed".to_owned(),
            )),
            Some(_) => Ok(true),
            None => Ok(false),
        }
    }

    fn parse_number_delimiter(&mut self, delimiter: Option<Delimiter>) -> Result<bool, ParseError> {
        let bytes = self.bytes();

        // If allow_time_unit_delimiter is true and there are any delimiters between the number and
        // the time unit, the delimiters are consumed before trying to parse the time units
        match (bytes.current_byte, delimiter) {
            (Some(byte), Some(delimiter)) if delimiter(*byte) => {
                bytes.try_consume_delimiter(delimiter)?;
                Ok(true)
            }
            (Some(_), _) => Ok(true),
            (None, _) => Ok(false),
        }
    }
}

pub struct ReprParserSingle<'a> {
    pub bytes: Bytes<'a>,
}

impl<'a> ReprParserSingle<'a> {
    pub const fn new(input: &'a str) -> Self {
        Self {
            bytes: Bytes::new(input.as_bytes()),
        }
    }
}

impl<'a> ReprParserTemplate<'a> for ReprParserSingle<'a> {
    type Output = DurationRepr<'a>;

    #[inline]
    fn bytes(&mut self) -> &mut Bytes<'a> {
        &mut self.bytes
    }

    #[inline]
    fn make_output(&'a mut self, duration_repr: DurationRepr<'a>) -> Self::Output {
        duration_repr
    }

    #[inline]
    fn parse_infinity_remainder(
        &'a mut self,
        mut duration_repr: DurationRepr<'a>,
        _: &Config,
    ) -> Result<DurationRepr<'a>, ParseError> {
        if self.bytes.is_end_of_input() {
            duration_repr.is_infinite = true;
            return Ok(duration_repr);
        }

        let expected = b"inity";
        for byte in expected {
            match self.bytes.current_byte {
                Some(current) if current.eq_ignore_ascii_case(byte) => self.bytes.advance(),
                // wrong character
                Some(current) => {
                    return Err(ParseError::Syntax(
                        self.bytes.current_pos,
                        format!(
                            "Error parsing infinity: Invalid character '{}'",
                            *current as char
                        ),
                    ));
                }
                None => {
                    return Err(ParseError::Syntax(
                        self.bytes.current_pos,
                        "Error parsing infinity: Premature end of input".to_owned(),
                    ));
                }
            }
        }

        duration_repr.is_infinite = true;
        self.bytes.check_end_of_input().map(|()| duration_repr)
    }

    #[inline]
    fn parse_keyword(
        &mut self,
        keywords: Option<&dyn TimeUnitsLike>,
        _: &Config,
    ) -> Result<Option<(TimeUnit, Multiplier)>, ParseError> {
        if let Some(keywords) = keywords {
            // SAFETY: we've only parsed valid utf-8 up to this point
            let keyword = unsafe { self.bytes.get_remainder_str_unchecked() };
            match keywords.get(keyword) {
                None => Ok(None),
                some_time_unit => {
                    self.bytes.finish();
                    Ok(some_time_unit)
                }
            }
        } else {
            Ok(None)
        }
    }

    fn parse_time_unit(
        &mut self,
        config: &Config,
        time_units: &dyn TimeUnitsLike,
    ) -> Result<Option<(TimeUnit, Multiplier)>, ParseError> {
        // cov:excl-start
        debug_assert!(
            self.bytes.current_byte.is_some(),
            "Don't call this function without being sure there's at least 1 byte remaining"
        ); // cov:excl-stop

        if config.allow_ago {
            let start = self.bytes.current_pos;
            // SAFETY: The delimiter may not match non-ascii bytes and we've parsed only valid utf-8
            // so far
            let string = unsafe {
                std::str::from_utf8_unchecked(self.bytes.advance_to(config.inner_delimiter))
            };

            let (time_unit, mut multiplier) = if string.is_empty() {
                // Haven't found a way to trigger this line, so it's excluded from coverage for now
                return Ok(None); // cov:excl-line
            } else {
                match time_units.get(string) {
                    None => {
                        self.bytes.reset(start);
                        return Ok(None);
                    }
                    Some(unit) => unit,
                }
            };

            // At this point, either there are one or more bytes of which the first is the
            // delimiter or we've reached the end of input
            if self.bytes.current_byte.is_some() {
                self.bytes.try_consume_delimiter(config.inner_delimiter)?;
                if self.bytes.next_is_ignore_ascii_case(b"ago") {
                    // SAFETY: We have checked that there are at least 3 bytes
                    unsafe { self.bytes.advance_by(3) };
                    // We're applying the negation on the multiplier only once so we don't need
                    // the operation to be reflexive and using saturating neg is fine
                    multiplier = multiplier.saturating_neg();
                } else {
                    self.bytes.reset(start);
                    return Ok(None);
                }
            };

            Ok(Some((time_unit, multiplier)))
        } else {
            // SAFETY: The input of `parse` is &str and therefore valid utf-8 and we have read
            // only ascii characters up to this point.
            let string = unsafe { self.bytes.get_remainder_str_unchecked() };
            let result = match time_units.get(string) {
                None => return Ok(None),
                some_time_unit => Ok(some_time_unit),
            };
            self.bytes.finish();
            result
        }
    }

    #[inline]
    fn parse_number_time_unit(
        &mut self,
        duration_repr: &mut DurationRepr<'a>,
        config: &'a Config,
        time_units: &dyn TimeUnitsLike,
    ) -> Result<bool, ParseError> {
        match self.bytes.current_byte {
            Some(_) if !time_units.is_empty() => {
                if let Some((unit, multi)) = self.parse_time_unit(config, time_units)? {
                    duration_repr.unit = Some(unit);
                    duration_repr.multiplier = multi;
                    Ok(true)
                } else {
                    self.bytes.get_remainder_str().and_then(|remainder| {
                        Err(ParseError::TimeUnit(
                            self.bytes.current_pos,
                            format!("Invalid time unit: '{remainder}'"),
                        ))
                    })
                }
            }
            Some(_) => {
                Err(ParseError::TimeUnit(
                    self.bytes.current_pos,
                    // SAFETY: We've parsed only valid utf-8 so far
                    format!("No time units allowed but found: '{}'", unsafe {
                        self.bytes.get_remainder_str_unchecked()
                    }),
                ))
            }
            // This branch is excluded from coverage because parsing with parse_number_delimiter
            // already ensures that there's at least 1 byte.
            None => Ok(false), // cov:excl-line
        }
    }

    #[inline]
    fn finalize(
        &'a mut self,
        duration_repr: DurationRepr<'a>,
        _: &Config,
    ) -> Result<Self::Output, ParseError> {
        self.bytes.check_end_of_input().map(|()| duration_repr)
    }
}

pub struct ReprParserMultiple<'a> {
    pub bytes: Bytes<'a>,
}

impl<'a> ReprParserMultiple<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            bytes: Bytes::new(input.as_bytes()),
        }
    }

    #[inline]
    pub fn is_next_duration(byte: u8) -> bool {
        byte.is_ascii_digit() || byte == b'+' || byte == b'-'
    }

    pub fn try_consume_connection(
        &mut self,
        delimiter: Delimiter,
        conjunctions: &'a [&'a str],
    ) -> Result<(), ParseError> {
        debug_assert!(delimiter(*self.bytes.current_byte.unwrap()));

        self.bytes.try_consume_delimiter(delimiter)?;
        let start = self.bytes.current_pos;
        // try_consume_delimiter ensures there's at least one byte here
        for word in conjunctions {
            if self.bytes.next_is_ignore_ascii_case(word.as_bytes()) {
                // SAFETY: We're advancing by the amount of bytes of the word we just found
                unsafe { self.bytes.advance_by(word.len()) };
                match self.bytes.current_byte {
                    Some(byte) if delimiter(*byte) => {
                        self.bytes.try_consume_delimiter(delimiter)?;
                    }
                    Some(byte) if Self::is_next_duration(*byte) => {}
                    Some(byte) => {
                        return Err(ParseError::Syntax(
                            self.bytes.current_pos,
                            format!(
                                "A conjunction must be separated by a delimiter, sign or digit \
                                 but found: '{}'",
                                *byte as char
                            ),
                        ));
                    }
                    None => {
                        return Err(ParseError::Syntax(
                            start,
                            format!("Input may not end with a conjunction but found: '{word}'"),
                        ));
                    }
                }
                break;
            }
        }
        Ok(())
    }
}

impl<'a> ReprParserTemplate<'a> for ReprParserMultiple<'a> {
    type Output = (DurationRepr<'a>, Option<&'a mut ReprParserMultiple<'a>>);

    #[inline]
    fn bytes(&mut self) -> &mut Bytes<'a> {
        &mut self.bytes
    }

    #[inline]
    fn make_output(&'a mut self, duration_repr: DurationRepr<'a>) -> Self::Output {
        (duration_repr, self.bytes().current_byte.map(|_| self))
    }

    #[inline]
    fn parse_infinity_remainder(
        &'a mut self,
        mut duration_repr: DurationRepr<'a>,
        config: &'a Config,
    ) -> Result<(DurationRepr<'a>, Option<&'a mut ReprParserMultiple<'a>>), ParseError> {
        match self.bytes.current_byte {
            Some(byte) if (config.outer_delimiter)(*byte) => {
                duration_repr.is_infinite = true;
                return self
                    .try_consume_connection(
                        config.outer_delimiter,
                        config.conjunctions.unwrap_or_default(),
                    )
                    .map(|()| (duration_repr, Some(self)));
            }
            Some(_) => {}
            None => {
                duration_repr.is_infinite = true;
                return Ok((duration_repr, None));
            }
        }

        let expected = "inity";
        let start = self.bytes.current_pos;
        for byte in expected.as_bytes() {
            match self.bytes.current_byte {
                Some(current) if current.eq_ignore_ascii_case(byte) => self.bytes.advance(),
                // wrong character
                Some(current) => {
                    return Err(ParseError::Syntax(
                        self.bytes.current_pos,
                        format!(
                            "Error parsing infinity: Invalid character '{}'",
                            *current as char
                        ),
                    ));
                }
                None => {
                    return Err(ParseError::Syntax(
                        // This subtraction is safe since we're here only if there's at least `inf`
                        // present
                        start - 3,
                        format!(
                            "Error parsing infinity: 'inf{}' is an invalid identifier for infinity",
                            self.bytes.get_current_str(start).unwrap() // unwrap is safe
                        ),
                    ));
                }
            }
        }

        duration_repr.is_infinite = true;
        match self.bytes.current_byte {
            Some(byte) if (config.outer_delimiter)(*byte) => {
                self.try_consume_connection(
                    config.outer_delimiter,
                    config.conjunctions.unwrap_or_default(),
                )?;
                Ok((duration_repr, Some(self)))
            }
            Some(byte) => Err(ParseError::Syntax(
                self.bytes.current_pos,
                format!(
                    "Error parsing infinity: Expected a delimiter but found '{}'",
                    *byte as char
                ),
            )),
            None => Ok((duration_repr, None)),
        }
    }

    #[inline]
    fn parse_keyword(
        &mut self,
        keywords: Option<&dyn TimeUnitsLike>,
        config: &'a Config,
    ) -> Result<Option<(TimeUnit, Multiplier)>, ParseError> {
        if let Some(keywords) = keywords {
            let start = self.bytes.current_pos;
            let buffer = self.bytes.buffered_advance_to(|byte: u8| {
                (config.outer_delimiter)(byte) || Self::is_next_duration(byte)
            });

            if buffer.is_empty() {
                // Haven't found a way to trigger this line, so it's excluded from coverage for now
                return Ok(None); // cov:excl-line
            }

            // SAFETY: The delimiter may not match non-ascii bytes and we've parsed only valid utf-8
            // so far
            let string = unsafe { std::str::from_utf8_unchecked(buffer) };

            match keywords.get(string) {
                None => {
                    self.bytes.reset(start);
                    Ok(None)
                }
                some_time_unit => {
                    if let Some(byte) = self.bytes.current_byte {
                        if (config.outer_delimiter)(*byte) {
                            self.try_consume_connection(
                                config.outer_delimiter,
                                config.conjunctions.unwrap_or_default(),
                            )?;
                        }
                    }
                    Ok(some_time_unit)
                }
            }
        } else {
            Ok(None)
        }
    }

    #[inline]
    fn parse_time_unit(
        &mut self,
        config: &'a Config,
        time_units: &dyn TimeUnitsLike,
    ) -> Result<Option<(TimeUnit, Multiplier)>, ParseError> {
        // cov:excl-start
        debug_assert!(
            self.bytes.current_byte.is_some(),
            "Don't call this function without being sure there's at least 1 byte remaining"
        ); // cov:excl-stop

        let start = self.bytes.current_pos;
        let buffer = if config.allow_ago {
            self.bytes.buffered_advance_to(|byte: u8| {
                (config.inner_delimiter)(byte)
                    || (config.outer_delimiter)(byte)
                    || Self::is_next_duration(byte)
            })
        } else {
            self.bytes.buffered_advance_to(|byte: u8| {
                (config.outer_delimiter)(byte) || Self::is_next_duration(byte)
            })
        };
        if buffer.is_empty() {
            return Ok(None);
        }

        // SAFETY: The delimiter may not match non-ascii bytes and we've parsed only valid utf-8 so
        // far
        let string = unsafe { std::str::from_utf8_unchecked(buffer) };

        let (time_unit, mut multiplier) = match time_units.get(string) {
            None => {
                self.bytes.reset(start);
                return Ok(None);
            }
            Some(some_time_unit) => some_time_unit,
        };

        match self.bytes.current_byte {
            Some(byte) if config.allow_ago && (config.inner_delimiter)(*byte) => {
                let start = self.bytes.current_pos;
                self.bytes.try_consume_delimiter(config.inner_delimiter)?;
                if self.bytes.next_is_ignore_ascii_case(b"ago") {
                    // SAFETY: We know that next is `ago` which has 3 bytes
                    unsafe { self.bytes.advance_by(3) };
                    match self.bytes.current_byte {
                        Some(byte)
                            if (config.outer_delimiter)(*byte) || Self::is_next_duration(*byte) =>
                        {
                            multiplier = multiplier.saturating_neg();
                        }
                        Some(_) => {
                            self.bytes.reset(start);
                        }
                        None => {
                            multiplier = multiplier.saturating_neg();
                        }
                    }
                } else {
                    self.bytes.reset(start);
                }
            }
            _ => {}
        }

        match self.bytes.current_byte {
            Some(byte) if (config.outer_delimiter)(*byte) => {
                self.try_consume_connection(
                    config.outer_delimiter,
                    config.conjunctions.unwrap_or_default(),
                )?;
            }
            Some(_) | None => {}
        }

        Ok(Some((time_unit, multiplier)))
    }

    #[inline]
    fn parse_number_time_unit(
        &mut self,
        duration_repr: &mut DurationRepr<'a>,
        config: &'a Config,
        time_units: &dyn TimeUnitsLike,
    ) -> Result<bool, ParseError> {
        match self.bytes().current_byte {
            Some(_) if !time_units.is_empty() => {
                if let Some((unit, multi)) = self.parse_time_unit(config, time_units)? {
                    duration_repr.unit = Some(unit);
                    duration_repr.multiplier = multi;
                }
            }
            Some(_) => {}
            // This branch is excluded from coverage because parse_number_delimiter already ensures
            // that there's at least 1 byte.
            None => return Ok(false), // cov:excl-line
        }
        Ok(true)
    }

    #[inline]
    fn finalize(
        &'a mut self,
        duration_repr: DurationRepr<'a>,
        config: &'a Config,
    ) -> Result<Self::Output, ParseError> {
        match self.bytes().current_byte {
            Some(byte) if (config.outer_delimiter)(*byte) => self
                .try_consume_connection(
                    config.outer_delimiter,
                    config.conjunctions.unwrap_or_default(),
                )
                .map(|()| (duration_repr, Some(self))),
            Some(_) => Ok((duration_repr, Some(self))),
            None => Ok((duration_repr, None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use rstest_reuse::{apply, template};

    use super::*;

    struct TimeUnitsFixture;

    // cov:excl-start This is just a fixture
    impl TimeUnitsLike for TimeUnitsFixture {
        fn is_empty(&self) -> bool {
            true
        }

        fn get(&self, _: &str) -> Option<(TimeUnit, Multiplier)> {
            None
        }
    } // cov:excl-stop

    #[rstest]
    #[case::zeros("00000000", Some(0x3030_3030_3030_3030))]
    #[case::one("00000001", Some(0x3130_3030_3030_3030))]
    #[case::ten_millions("10000000", Some(0x3030_3030_3030_3031))]
    #[case::nines("99999999", Some(0x3939_3939_3939_3939))]
    fn test_duration_repr_parser_parse_8_digits(
        #[case] input: &str,
        #[case] expected: Option<u64>,
    ) {
        let mut parser = ReprParserSingle::new(input);
        assert_eq!(parser.bytes.parse_8_digits(), expected);
    }

    #[rstest]
    #[case::empty("", None)]
    #[case::one_non_digit_char("a0000000", None)]
    #[case::less_than_8_digits("9999999", None)]
    fn test_duration_repr_parser_parse_8_digits_when_not_8_digits(
        #[case] input: &str,
        #[case] expected: Option<u64>,
    ) {
        let mut parser = ReprParserSingle::new(input);
        assert_eq!(parser.bytes.parse_8_digits(), expected);
        assert_eq!(parser.bytes.get_remainder(), input.as_bytes());
        assert_eq!(parser.bytes.current_byte, input.as_bytes().first());
        assert_eq!(parser.bytes.current_pos, 0);
    }

    #[test]
    fn test_duration_repr_parser_parse_8_digits_when_more_than_8() {
        let mut parser = ReprParserSingle::new("00000000a");
        assert_eq!(parser.bytes.parse_8_digits(), Some(0x3030_3030_3030_3030));
        assert_eq!(parser.bytes.get_remainder(), b"a");
        assert_eq!(parser.bytes.current_byte, Some(&b'a'));
        assert_eq!(parser.bytes.current_pos, 8);
    }

    #[template]
    #[rstest]
    #[case::zero("0", Whole(1, 1))]
    #[case::one("1", Whole(0, 1))]
    #[case::nine("9", Whole(0, 1))]
    #[case::ten("10", Whole(0, 2))]
    #[case::eight_leading_zeros("00000000", Whole(8, 8))]
    #[case::fifteen_leading_zeros("000000000000000", Whole(15, 15))]
    #[case::ten_with_leading_zeros_when_eight_digits("00000010", Whole(6, 8))]
    #[case::ten_with_leading_zeros_when_nine_digits("000000010", Whole(7, 9))]
    #[case::mixed_number("12345", Whole(0, 5))]
    #[case::max_8_digits("99999999", Whole(0, 8))]
    #[case::max_8_digits_minus_one("99999998", Whole(0, 8))]
    #[case::min_nine_digits("100000000", Whole(0, 9))]
    #[case::min_nine_digits_plus_one("100000001", Whole(0, 9))]
    #[case::eight_zero_digits_start("0000000011111111", Whole(8, 16))]
    #[case::eight_zero_digits_end("1111111100000000", Whole(0, 16))]
    #[case::eight_zero_digits_middle("11111111000000001", Whole(0, 17))]
    #[case::max_16_digits("9999999999999999", Whole(0, 16))]
    fn test_duration_repr_parser_parse_whole(#[case] input: &str, #[case] expected: Whole) {}

    #[apply(test_duration_repr_parser_parse_whole)]
    fn test_duration_repr_parser_parse_whole_single(input: &str, expected: Whole) {
        let mut parser = ReprParserSingle::new(input);
        assert_eq!(parser.parse_whole(), expected);
    }

    #[apply(test_duration_repr_parser_parse_whole)]
    fn test_duration_repr_parser_parse_whole_multiple(input: &str, expected: Whole) {
        let mut parser = ReprParserMultiple::new(input); // cov:excl-line
        assert_eq!(parser.parse_whole(), expected);
    }

    // TODO: add test case for parse_multiple
    #[test]
    fn test_duration_repr_parser_parse_whole_when_more_than_max_exponent() {
        let config = Config::new();
        let input = &"1".repeat(i16::MAX as usize + 100);
        let mut parser = ReprParserSingle::new(input);
        let duration_repr = parser
            .parse(&config, &TimeUnitsFixture, None, None)
            .unwrap();
        assert_eq!(duration_repr.whole, Some(Whole(0, i16::MAX as usize + 100)));
        assert_eq!(duration_repr.fract, None);
    }

    // TODO: use own test case for parse_multiple
    #[test]
    fn test_duration_repr_parser_parse_fract_when_more_than_max_exponent() {
        let input = format!(".{}", "1".repeat(i16::MAX as usize + 100));
        let config = Config::new();

        let mut parser = ReprParserSingle::new(&input);
        let duration_repr = parser
            .parse(&config, &TimeUnitsFixture, None, None)
            .unwrap();
        assert_eq!(duration_repr.whole, None);
        assert_eq!(duration_repr.fract, Some(Fract(1, i16::MAX as usize + 101)));

        let mut config = Config::new();
        config.allow_multiple = true;
        let mut parser = ReprParserMultiple::new(&input);
        let (duration_repr, maybe_parser) = parser
            .parse(&config, &TimeUnitsFixture, None, None)
            .unwrap();
        assert!(maybe_parser.is_none());
        assert_eq!(duration_repr.whole, None);
        assert_eq!(duration_repr.fract, Some(Fract(1, i16::MAX as usize + 101)));
    }

    #[template]
    #[rstest]
    #[case::zero("0", Fract(0, 1))]
    #[case::one("1", Fract(0, 1))]
    #[case::nine("9", Fract(0, 1))]
    #[case::ten("10", Fract(0, 2))]
    #[case::leading_zero("01", Fract(0, 2))]
    #[case::leading_zeros("001", Fract(0, 3))]
    #[case::eight_leading_zeros("000000001", Fract(0, 9))]
    #[case::mixed_number("12345", Fract(0, 5))]
    #[case::max_8_digits("99999999", Fract(0, 8))]
    #[case::max_8_digits_minus_one("99999998", Fract(0, 8))]
    #[case::nine_digits("123456789", Fract(0, 9))]
    fn test_duration_repr_parser_parse_fract(#[case] input: &str, #[case] expected: Fract) {}

    #[apply(test_duration_repr_parser_parse_fract)]
    fn test_duration_repr_parser_parse_fract_single(input: &str, expected: Fract) {
        let mut parser = ReprParserSingle::new(input);
        assert_eq!(parser.parse_fract(), expected);
    }

    #[apply(test_duration_repr_parser_parse_fract)]
    fn test_duration_repr_parser_parse_fract_multiple(input: &str, expected: Fract) {
        let mut parser = ReprParserMultiple::new(input); // cov:excl-line
        assert_eq!(parser.parse_fract(), expected);
    }

    #[test]
    fn test_fract_is_empty() {
        assert!(Fract(0, 0).is_empty());
        assert!(Fract(9, 9).is_empty());
    }

    #[test]
    fn test_whole_is_empty() {
        assert!(Whole(0, 0).is_empty());
        assert!(Whole(9, 9).is_empty());
    }

    #[rstest]
    #[case::one_digit(b"1", None, None, 100_000_000_000_000_000)]
    #[case::one_digit_one_zero(b"1", None, Some(1), 10_000_000_000_000_000)]
    #[case::two_digits(b"12", None, None, 120_000_000_000_000_000)]
    #[case::two_digits_one_zero(b"12", None, Some(1), 12_000_000_000_000_000)]
    #[case::one_digit_prepend(b"2", Some(b"1".as_ref()), None, 120_000_000_000_000_000)]
    #[case::empty_digits_prepend_one(b"", Some(b"1".as_ref()), None, 100_000_000_000_000_000)]
    #[case::more_than_18_digits_when_zeros(b"234", Some(b"1".as_ref()), Some(16), 12)]
    fn test_fract(
        #[case] digits: &[u8],
        #[case] prepend: Option<&[u8]>,
        #[case] zeros: Option<usize>,
        #[case] expected: u64,
    ) {
        assert_eq!(Fract::parse(digits, prepend, zeros), expected);
    }

    #[test]
    fn test_try_consume_delimiter_when_input_starts_with_delimiter_then_error() {
        let mut bytes = Bytes::new(b" some");
        assert_eq!(
            bytes.try_consume_delimiter(|byte| byte == b' '),
            Err(ParseError::Syntax(
                0,
                "Input may not start with a delimiter".to_owned()
            ))
        );
    }
}
