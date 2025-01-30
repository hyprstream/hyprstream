// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use fundu_core::config::NumbersLike;
use fundu_core::time::Multiplier;

pub(crate) mod builder;
pub(crate) mod parser;
pub(crate) mod time_units;

/// A [`Numeral`] can occur where numbers usually occur in the source string
///
/// `Numerals` express a number as a word like `one` or `next` instead of `1` or `half` instead of
/// `0.5` but also `last` instead of `-1` etc. This symbolic number is defined as a [`Multiplier`].
/// In contrast to numbers, `Numerals` without a time unit are considered an error and therefore
/// have to be followed by a time unit.
///
/// # Examples
///
/// ```rust
/// use fundu::TimeUnit::*;
/// use fundu::{CustomDurationParserBuilder, CustomTimeUnit, Duration, Multiplier, Numeral};
///
/// let parser = CustomDurationParserBuilder::new()
///     .numerals(&[
///         Numeral::new(&["last"], Multiplier(-1, 0)),
///         Numeral::new(&["this"], Multiplier(0, 0)),
///         Numeral::new(&["one", "next"], Multiplier(1, 0)),
///         Numeral::new(&["two"], Multiplier(2, 0)),
///     ])
///     .time_unit(CustomTimeUnit::with_default(NanoSecond, &["nano", "nanos"]))
///     .allow_negative()
///     .build();
///
/// assert_eq!(parser.parse("last nano"), Ok(Duration::negative(0, 1)));
/// assert_eq!(parser.parse("this nano"), Ok(Duration::negative(0, 0)));
/// assert_eq!(parser.parse("next nano"), Ok(Duration::positive(0, 1)));
/// assert_eq!(parser.parse("two nanos"), Ok(Duration::positive(0, 2)));
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Numeral<'a> {
    identifiers: &'a [&'a str],
    multiplier: Multiplier,
}

impl<'a> Numeral<'a> {
    /// Create a new [`Numeral`]
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use fundu::Numeral;
    ///
    /// let numeral = Numeral::new(&["next", "one"], Multiplier(1, 0));
    /// ```
    pub fn new(identifiers: &'a [&'a str], multiplier: Multiplier) -> Self {
        Self {
            identifiers,
            multiplier,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Numerals<'a> {
    data: Vec<Numeral<'a>>,
}

impl<'a> Numerals<'a> {
    pub(crate) fn new() -> Self {
        Self { data: vec![] }
    }

    pub(crate) fn with_numerals(numerals: Vec<Numeral<'a>>) -> Self {
        Self { data: numerals }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a> NumbersLike for Numerals<'a> {
    fn get(&self, input: &str) -> Option<Multiplier> {
        self.data.iter().find_map(|numeral| {
            numeral
                .identifiers
                .iter()
                .find_map(|id| (*id == input).then_some(numeral.multiplier))
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    pub fn numeral_this() -> Numeral<'static> {
        Numeral::new(&["this"], Multiplier(0, 0))
    }

    #[fixture]
    pub fn numeral_last() -> Numeral<'static> {
        Numeral::new(&["last"], Multiplier(-1, 0))
    }

    #[fixture]
    pub fn numeral_next() -> Numeral<'static> {
        Numeral::new(&["next"], Multiplier(1, 0))
    }

    #[rstest]
    fn test_numeral_new(numeral_this: Numeral) {
        assert_eq!(numeral_this.identifiers, &["this"]);
        assert_eq!(numeral_this.multiplier, Multiplier(0, 0));
    }

    #[test]
    fn test_numerals_new() {
        let numerals = Numerals::new();
        assert!(numerals.is_empty());
    }

    #[rstest]
    #[case::last("last", Multiplier(-1, 0))]
    #[case::this("this", Multiplier(0, 0))]
    #[case::next("next", Multiplier(1, 0))]
    fn test_numerals_with_valid_numerals(
        #[case] input: &str,
        #[case] expected: Multiplier,
        numeral_this: Numeral,
        numeral_last: Numeral,
        numeral_next: Numeral,
    ) {
        let numerals = Numerals::with_numerals(vec![numeral_last, numeral_this, numeral_next]);
        assert!(!numerals.is_empty());
        assert_eq!(numerals.get(input), Some(expected));
    }

    #[rstest]
    #[case::empty("")]
    #[case::incomplete("las")]
    #[case::too_much("lasts")]
    #[case::capitalized("Last")]
    #[case::upper_case("LAST")]
    fn test_numerals_with_invalid_numerals(#[case] input: &str, numeral_last: Numeral) {
        let numerals = Numerals::with_numerals(vec![numeral_last]);
        assert_eq!(numerals.get(input), None);
    }
}
