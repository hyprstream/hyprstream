// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! Provide the errors used in fundu like [`ParseError`] and [`TryFromDurationError`]

use std::error::Error;
use std::fmt::Display;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Error type emitted during the parsing
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum ParseError {
    /// Returned, if the input was empty.
    Empty,
    /// A syntax error. Syntax errors report the position (column) where it was encountered and a
    /// reason.
    Syntax(usize, String),
    /// Currently only used internally for overflows of the maximum Duration.
    /// TODO: Rename to positive overflow
    /// TODO: Add `NegativeOverflow`
    Overflow,
    /// An error concerning time units. Like [`ParseError::Syntax`]  the position where the error
    /// occurred is included.
    TimeUnit(usize, String),
    /// The exponent exceeded the minimum negative exponent (`-32768`)
    NegativeExponentOverflow,
    /// The exponent exceeded the maximum positive exponent (`+32767`)
    PositiveExponentOverflow,
    /// The input number was negative. Note that numbers close to `0` (`< 1e-18`) are not negative
    /// but resolve to `0`
    NegativeNumber,
    /// A generic error if no other error type fits
    InvalidInput(String),
}

impl Error for ParseError {}

impl Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::Syntax(column, reason) => {
                format!("Syntax error: {reason} at column {column}")
            }
            Self::Overflow => "Number overflow".to_owned(),
            Self::TimeUnit(pos, reason) => {
                format!("Time unit error: {reason} at column {pos}")
            }
            Self::NegativeExponentOverflow => {
                "Negative exponent overflow: Minimum is -32768".to_owned()
            }
            Self::PositiveExponentOverflow => {
                "Positive exponent overflow: Maximum is +32767".to_owned()
            }
            Self::NegativeNumber => "Number was negative".to_owned(),
            Self::InvalidInput(reason) => format!("Invalid input: {reason}"),
            Self::Empty => "Empty input".to_owned(),
        };
        f.write_str(&msg)
    }
}

/// This error may occur when converting a [`crate::time::Duration`] to a different duration like
/// [`std::time::Duration`]
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TryFromDurationError {
    /// The duration was negative and the destination duration doesn't support negative durations
    NegativeDuration,
    /// The duration was higher than the maximum of the destination duration
    PositiveOverflow,
    /// The duration was lower than the minimum of the destination duration
    NegativeOverflow,
}

impl From<TryFromDurationError> for ParseError {
    fn from(error: TryFromDurationError) -> Self {
        match error {
            TryFromDurationError::NegativeDuration => Self::NegativeNumber,
            TryFromDurationError::PositiveOverflow | TryFromDurationError::NegativeOverflow => {
                Self::Overflow
            }
        }
    }
}

impl Error for TryFromDurationError {}

impl Display for TryFromDurationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let description = match self {
            Self::NegativeDuration => "Error converting duration: value is negative",
            Self::PositiveOverflow => {
                "Error converting duration: value overflows the positive value range"
            }
            Self::NegativeOverflow => {
                "Error converting duration: value overflows the negative value range"
            }
        };
        description.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    #[cfg(feature = "serde")]
    use serde_test::{assert_tokens, Token};

    use super::*;

    #[rstest]
    #[case::syntax_error(
        ParseError::Syntax(10, "Invalid character".to_owned()),
        "Syntax error: Invalid character at column 10"
    )]
    #[case::overflow(ParseError::Overflow, "Number overflow")]
    #[case::time_unit_error(
        ParseError::TimeUnit(10, "Found invalid 'y'".to_owned()),
        "Time unit error: Found invalid 'y' at column 10"
    )]
    #[case::negative_exponent_overflow_error(
        ParseError::NegativeExponentOverflow,
        "Negative exponent overflow: Minimum is -32768"
    )]
    #[case::positive_exponent_overflow_error(
        ParseError::PositiveExponentOverflow,
        "Positive exponent overflow: Maximum is +32767"
    )]
    #[case::negative_number_error(ParseError::NegativeNumber, "Number was negative")]
    #[case::invalid_input(
        ParseError::InvalidInput("Unexpected".to_owned()),
        "Invalid input: Unexpected"
    )]
    #[case::empty(ParseError::Empty, "Empty input")]
    fn test_error_messages_parse_error(#[case] error: ParseError, #[case] expected: &str) {
        assert_eq!(error.to_string(), expected);
    }

    #[rstest]
    #[case::negative_overflow(TryFromDurationError::NegativeOverflow, ParseError::Overflow)]
    #[case::positive_overflow(TryFromDurationError::PositiveOverflow, ParseError::Overflow)]
    #[case::negative_number(TryFromDurationError::NegativeDuration, ParseError::NegativeNumber)]
    fn test_from_for_parse_error(#[case] from: TryFromDurationError, #[case] expected: ParseError) {
        assert_eq!(ParseError::from(from), expected);
    }

    #[rstest]
    #[case::negative_number(
        TryFromDurationError::NegativeDuration,
        "Error converting duration: value is negative"
    )]
    #[case::positive_overflow(
        TryFromDurationError::PositiveOverflow,
        "Error converting duration: value overflows the positive value range"
    )]
    #[case::positive_overflow(
        TryFromDurationError::NegativeOverflow,
        "Error converting duration: value overflows the negative value range"
    )]
    fn test_error_messages_try_from_duration_error(
        #[case] error: TryFromDurationError,
        #[case] expected: &str,
    ) {
        assert_eq!(error.to_string(), expected);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_try_from_duration_error() {
        let error = TryFromDurationError::NegativeDuration;

        assert_tokens(
            &error,
            &[
                Token::Enum {
                    name: "TryFromDurationError",
                },
                Token::Str("NegativeDuration"),
                Token::Unit,
            ],
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_parse_error() {
        let error = ParseError::Empty;

        assert_tokens(
            &error,
            &[
                Token::Enum { name: "ParseError" },
                Token::Str("Empty"),
                Token::Unit,
            ],
        );
    }
}
