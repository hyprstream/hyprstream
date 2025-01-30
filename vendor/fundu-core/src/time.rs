// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

//! Contains all time related structures used in fundu like [`TimeUnit`] and [`Duration`]

use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use TimeUnit::*;

use crate::error::TryFromDurationError;

/// The default identifier of [`TimeUnit::NanoSecond`]
pub const DEFAULT_ID_NANO_SECOND: &str = "ns";
/// The default identifier of [`TimeUnit::MicroSecond`]
pub const DEFAULT_ID_MICRO_SECOND: &str = "Ms";
/// The default identifier of [`TimeUnit::MicroSecond`]
pub const DEFAULT_ID_MILLI_SECOND: &str = "ms";
/// The default identifier of [`TimeUnit::Second`]
pub const DEFAULT_ID_SECOND: &str = "s";
/// The default identifier of [`TimeUnit::Minute`]
pub const DEFAULT_ID_MINUTE: &str = "m";
/// The default identifier of [`TimeUnit::Hour`]
pub const DEFAULT_ID_HOUR: &str = "h";
/// The default identifier of [`TimeUnit::Day`]
pub const DEFAULT_ID_DAY: &str = "d";
/// The default identifier of [`TimeUnit::Week`]
pub const DEFAULT_ID_WEEK: &str = "w";
/// The default identifier of [`TimeUnit::Month`]
pub const DEFAULT_ID_MONTH: &str = "M";
/// The default identifier of [`TimeUnit::Year`]
pub const DEFAULT_ID_YEAR: &str = "y";

pub(crate) const DEFAULT_TIME_UNIT: TimeUnit = Second;

const SECS_PER_MINUTE: u64 = 60;
const SECS_PER_HOUR: u64 = 3600;
const SECS_PER_DAY: u64 = 86400;
const SECS_PER_WEEK: u64 = 86400 * 7;

const NANOS_PER_MILLI: i32 = 1_000_000;
const NANOS_PER_MICRO: i32 = 1_000;

const MILLIS_PER_SEC: i128 = 1_000;
const MICROS_PER_SEC: i128 = 1_000_000;
const NANOS_PER_SEC: i128 = 1_000_000_000;

/// The time units used to define possible time units in the input string
///
/// The parser calculates the final [`Duration`] based on the parsed number and time unit. Each
/// `TimeUnit` has an inherent [`Multiplier`] and a default id. The [`Multiplier`] influences the
/// final value of the [`Duration`] and is seconds based. See also the documentation of
/// [`Multiplier`]
///
/// # Examples
///
/// ```rust
/// use fundu_core::time::Multiplier;
/// use fundu_core::time::TimeUnit::*;
///
/// assert_eq!(NanoSecond.default_identifier(), "ns");
/// assert_eq!(Second.default_identifier(), "s");
/// assert_eq!(Hour.default_identifier(), "h");
///
/// assert_eq!(NanoSecond.multiplier(), Multiplier(1, -9));
/// assert_eq!(MilliSecond.multiplier(), Multiplier(1, -3));
/// assert_eq!(Second.multiplier(), Multiplier(1, 0));
/// assert_eq!(Hour.multiplier(), Multiplier(60 * 60, 0));
/// ```
///
/// [`Multiplier`]: crate::time::Multiplier
/// [`Duration`]: crate::time::Duration
#[derive(Debug, PartialEq, Eq, Clone, Copy, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TimeUnit {
    /// Represents the lowest possible time unit. The default id is given by
    /// [`DEFAULT_ID_NANO_SECOND`] = `ns`
    NanoSecond,
    /// The default id is given by [`DEFAULT_ID_MICRO_SECOND`] = `Ms`
    MicroSecond,
    /// The default id is given by [`DEFAULT_ID_MILLI_SECOND`] = `ms`
    MilliSecond,
    /// The default if no time unit is given. The default id is given by [`DEFAULT_ID_SECOND`] =
    /// `s`
    Second,
    /// The default id is given by [`DEFAULT_ID_MINUTE`] = `m`
    Minute,
    /// The default id is given by [`DEFAULT_ID_HOUR`] = `h`
    Hour,
    /// The default id is given by [`DEFAULT_ID_DAY`] = `d`
    Day,
    /// The default id is given by [`DEFAULT_ID_WEEK`] = `w`
    Week,
    /// The default id is given by [`DEFAULT_ID_MONTH`] = `M`
    Month,
    /// Represents the hightest possible time unit. The default id is given by [`DEFAULT_ID_YEAR`]
    /// = `y`
    Year,
}

impl Default for TimeUnit {
    fn default() -> Self {
        DEFAULT_TIME_UNIT
    }
}

impl TimeUnit {
    /// Return the default identifier
    pub const fn default_identifier(&self) -> &'static str {
        match self {
            NanoSecond => DEFAULT_ID_NANO_SECOND,
            MicroSecond => DEFAULT_ID_MICRO_SECOND,
            MilliSecond => DEFAULT_ID_MILLI_SECOND,
            Second => DEFAULT_ID_SECOND,
            Minute => DEFAULT_ID_MINUTE,
            Hour => DEFAULT_ID_HOUR,
            Day => DEFAULT_ID_DAY,
            Week => DEFAULT_ID_WEEK,
            Month => DEFAULT_ID_MONTH,
            Year => DEFAULT_ID_YEAR,
        }
    }

    /// Return the base [`Multiplier`] of this [`TimeUnit`].
    ///
    /// This multiplier is always seconds based so for example:
    ///
    /// ```ignore
    /// NanoSecond: Multiplier(1, -9)
    /// Second: Multiplier(1, 0)
    /// Year: Multiplier(31557600, 0)
    /// ```
    pub const fn multiplier(&self) -> Multiplier {
        const MULTIPLIERS: [Multiplier; 10] = [
            Multiplier(1, -9),
            Multiplier(1, -6),
            Multiplier(1, -3),
            Multiplier(1, 0),
            Multiplier(60, 0),
            Multiplier(3_600, 0),
            Multiplier(86_400, 0),
            Multiplier(604_800, 0),
            Multiplier(2_629_800, 0),  // Year / 12
            Multiplier(31_557_600, 0), // 365.25 days
        ];
        MULTIPLIERS[*self as usize]
    }
}

/// To be able to use the basic [`crate::parse::Parser`] this trait needs to be implemented
///
/// Usually, time units are a fixed set of strings and implementing `TimeUnitsLike` is a simple
/// straight forward process. For more advanced usages see for example the implementations of fundu.
/// The most important method is [`TimeUnitsLike::get`]. See there for additional information.
///
/// # Examples
///
/// Example for a simple and small fixed set of time units.
///
/// ```rust
/// use fundu_core::time::{Multiplier, TimeUnit, TimeUnitsLike};
///
/// struct TimeUnits {}
/// impl TimeUnitsLike for TimeUnits {
///     fn is_empty(&self) -> bool {
///         false
///     }
///
///     fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
///         match identifier {
///             "s" | "sec" => Some((TimeUnit::Second, Multiplier(1, 0))),
///             "min" | "minutes" => Some((TimeUnit::Minute, Multiplier(1, 0))),
///             _ => None,
///         }
///     }
/// }
/// let time_units = TimeUnits {};
///
/// assert_eq!(time_units.is_empty(), false);
/// assert_eq!(
///     time_units.get("sec"),
///     Some((TimeUnit::Second, Multiplier(1, 0)))
/// );
/// assert_eq!(
///     time_units.get("minutes"),
///     Some((TimeUnit::Minute, Multiplier(1, 0)))
/// );
/// assert_eq!(time_units.get("does_not_exist"), None);
/// ```
pub trait TimeUnitsLike {
    /// Return true if there are no time units configured to be matched against
    ///
    /// # Examples
    ///
    /// Example for an empty set of time units and `is_empty` returning `true`
    ///
    /// ```rust
    /// use fundu_core::time::{Multiplier, TimeUnit, TimeUnitsLike};
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
    /// let time_units = TimeUnits {};
    ///
    /// assert!(time_units.is_empty());
    /// ```
    ///
    /// Example for a fixed set of time units. To keep the example small only a few possibilities
    /// are given, but the essential point is, that as soon as the `string` can be matched against
    /// one time unit in the `get` method, `is_empty` must return false.
    ///
    /// ```rust
    /// use fundu_core::time::{Multiplier, TimeUnit, TimeUnitsLike};
    ///
    /// struct TimeUnits {}
    /// impl TimeUnitsLike for TimeUnits {
    ///     fn is_empty(&self) -> bool {
    ///         false
    ///     }
    ///
    ///     fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
    ///         match identifier {
    ///             "s" | "sec" => Some((TimeUnit::Second, Multiplier(1, 0))),
    ///             "min" | "minutes" => Some((TimeUnit::Minute, Multiplier(1, 0))),
    ///             _ => None,
    ///         }
    ///     }
    /// }
    /// let time_units = TimeUnits {};
    ///
    /// assert_eq!(time_units.is_empty(), false);
    /// ```
    fn is_empty(&self) -> bool;

    /// Return the result of the match of the `identifier` against a set of time units
    ///
    /// This method must return `None` if the `identifier` does not match one of the time units and
    /// return `Some` tuple with a [`TimeUnit`] and an additional [`Multiplier`]. The [`Multiplier`]
    /// is not the multiplier of the `TimeUnit` but is instead applied additionally to the inherent
    /// `Multiplier` of the `TimeUnit`. A [`TimeUnit`] ranges from [`TimeUnit::NanoSecond`] to
    /// [`TimeUnit::Year`]. The additional `Multiplier` allows to create other time units derived
    /// from the existent ones like `century` `(TimeUnit::Year, Multiplier(100, 0))` or `fortnight`
    /// `(TimeUnit::Week, Multiplier(2, 0))` ...
    ///
    /// # Problems
    ///
    /// This method is time critical and the `identifier` can theoretically range from `0` to
    /// `usize::MAX` length. So, every measure to avoid unnecessary calculations should be taken.
    ///
    /// # Examples
    ///
    /// Full example for a fixed fantasy set of time units with a derived time unit `fortnight` and
    /// micro seconds with an utf-8 multi-byte character
    ///
    /// ```rust
    /// use fundu_core::time::{Multiplier, TimeUnit, TimeUnitsLike};
    ///
    /// struct TimeUnits {}
    /// impl TimeUnitsLike for TimeUnits {
    ///     fn is_empty(&self) -> bool {
    ///         false
    ///     }
    ///
    ///     fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
    ///         match identifier {
    ///             "ns" | "nsec" => Some((TimeUnit::NanoSecond, Multiplier(1, 0))),
    ///             "µs" | "us" | "usec" => Some((TimeUnit::MicroSecond, Multiplier(1, 0))),
    ///             "ms" | "msec" => Some((TimeUnit::MilliSecond, Multiplier(1, 0))),
    ///             "s" | "sec" => Some((TimeUnit::Second, Multiplier(1, 0))),
    ///             "m" | "min" | "mins" | "minutes" => Some((TimeUnit::Minute, Multiplier(1, 0))),
    ///             "h" | "hr" | "hrs" => Some((TimeUnit::Hour, Multiplier(1, 0))),
    ///             "d" | "day" | "days" => Some((TimeUnit::Day, Multiplier(1, 0))),
    ///             "w" | "weeks" => Some((TimeUnit::Week, Multiplier(1, 0))),
    ///             "fortnight" | "fortnights" => Some((TimeUnit::Week, Multiplier(2, 0))),
    ///             "month" | "months" => Some((TimeUnit::Month, Multiplier(1, 0))),
    ///             "year" | "years" => Some((TimeUnit::Year, Multiplier(1, 0))),
    ///             _ => None,
    ///         }
    ///     }
    /// }
    /// let time_units = TimeUnits {};
    ///
    /// assert_eq!(time_units.is_empty(), false);
    /// assert_eq!(
    ///     time_units.get("nsec"),
    ///     Some((TimeUnit::NanoSecond, Multiplier(1, 0)))
    /// );
    /// assert_eq!(
    ///     time_units.get("fortnight"),
    ///     Some((TimeUnit::Week, Multiplier(2, 0)))
    /// );
    /// assert_eq!(
    ///     time_units.get("years"),
    ///     Some((TimeUnit::Year, Multiplier(1, 0)))
    /// );
    /// assert_eq!(time_units.get("does_not_match"), None);
    /// ```
    ///
    /// Whatever may seems reasonable to match against the set of time units is allowed like
    /// matching case insensitive. See the following short example which also tries to avoid any
    /// unnecessary calculation to the lowercase pendant of the original `identifier`
    ///
    /// ```rust
    /// use fundu_core::time::{Multiplier, TimeUnit, TimeUnitsLike};
    ///
    /// struct TimeUnits {}
    /// impl TimeUnitsLike for TimeUnits {
    ///     fn is_empty(&self) -> bool {
    ///         false
    ///     }
    ///
    ///     fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
    ///         // We use the fact that none of our time units have a string length greater than `4`
    ///         // and lower than `1`
    ///         if identifier.len() >= 1 && identifier.len() <= 4 {
    ///             match identifier.to_ascii_lowercase().as_str() {
    ///                 "µs" | "us" | "usec" => Some((TimeUnit::MicroSecond, Multiplier(1, 0))),
    ///                 "ms" | "msec" => Some((TimeUnit::MilliSecond, Multiplier(1, 0))),
    ///                 "s" | "sec" => Some((TimeUnit::Second, Multiplier(1, 0))),
    ///                 _ => None,
    ///             }
    ///         } else {
    ///             None
    ///         }
    ///     }
    /// }
    /// let time_units = TimeUnits {};
    ///
    /// assert_eq!(time_units.is_empty(), false);
    /// assert_eq!(
    ///     time_units.get("USEC"),
    ///     Some((TimeUnit::MicroSecond, Multiplier(1, 0)))
    /// );
    /// assert_eq!(
    ///     time_units.get("MSec"),
    ///     Some((TimeUnit::MilliSecond, Multiplier(1, 0)))
    /// );
    /// assert_eq!(
    ///     time_units.get("sEc"),
    ///     Some((TimeUnit::Second, Multiplier(1, 0)))
    /// );
    /// assert_eq!(time_units.get("does_not_match"), None);
    /// ```
    fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)>;
}

/// The multiplier of a [`TimeUnit`].
///
/// The multiplier consists of two numbers `(m, e)` which are applied to another number `x` as
/// follows:
///
/// `x * m * 10 ^ e`
///
/// Examples:
///
/// ```ignore
/// let nano_second = Multiplier(1, -9);
/// let second = Multiplier(1, 0);
/// let hour = Multiplier(3600, 0);
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Multiplier(pub i64, pub i16);

impl Default for Multiplier {
    fn default() -> Self {
        Self(1, 0)
    }
}

impl Multiplier {
    /// Return the coefficient component of the `Multiplier`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Multiplier;
    ///
    /// let multiplier = Multiplier(123, 45);
    /// assert_eq!(multiplier.coefficient(), 123);
    /// ```
    #[inline]
    pub const fn coefficient(&self) -> i64 {
        self.0
    }

    /// Return the exponent component of the `Multiplier`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Multiplier;
    ///
    /// let multiplier = Multiplier(123, 45);
    /// assert_eq!(multiplier.exponent(), 45);
    /// ```
    #[inline]
    pub const fn exponent(&self) -> i16 {
        self.1
    }

    /// Return true if the `Multiplier` is negative
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Multiplier;
    ///
    /// let multiplier = Multiplier(-123, 45);
    /// assert!(multiplier.is_negative());
    /// ```
    #[inline]
    pub const fn is_negative(&self) -> bool {
        !self.is_positive()
    }

    /// Return true if the `Multiplier` is positive
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Multiplier;
    ///
    /// let multiplier = Multiplier(123, 45);
    /// assert!(multiplier.is_positive());
    /// ```
    #[inline]
    pub const fn is_positive(&self) -> bool {
        self.0 == 0 || self.0.is_positive()
    }

    /// Checked `Multiplier` multiplication. Computes `self * other`, returning `None` if an
    /// overflow occurred.
    ///
    /// Let `a, b` be multipliers, with `m` being the coefficient and `e` the exponent. The
    /// multiplication is performed such that `(x.m, x.e) = (a.m * b.m, a.e + b.e)`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Multiplier;
    ///
    /// assert_eq!(
    ///     Multiplier(1, 2).checked_mul(Multiplier(3, 4)),
    ///     Some(Multiplier(3, 6))
    /// );
    /// assert_eq!(
    ///     Multiplier(-1, 2).checked_mul(Multiplier(3, -4)),
    ///     Some(Multiplier(-3, -2))
    /// );
    /// assert_eq!(Multiplier(2, 0).checked_mul(Multiplier(i64::MAX, 1)), None);
    /// assert_eq!(Multiplier(1, 2).checked_mul(Multiplier(1, i16::MAX)), None);
    /// ```
    #[inline]
    pub const fn checked_mul(&self, rhs: Self) -> Option<Self> {
        if let Some(coefficient) = self.0.checked_mul(rhs.0) {
            if let Some(exponent) = self.1.checked_add(rhs.1) {
                return Some(Self(coefficient, exponent));
            }
        }
        None
    }

    /// Saturating negation. Computes `-self`, returning `i64::MAX` if `self.coefficient() ==
    /// i64::MIN` instead of overflowing.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Multiplier;
    ///
    /// assert_eq!(Multiplier(1, 2).saturating_neg(), Multiplier(-1, 2));
    /// assert_eq!(Multiplier(-1, 2).saturating_neg(), Multiplier(1, 2));
    /// assert_eq!(
    ///     Multiplier(i64::MIN, 2).saturating_neg(),
    ///     Multiplier(i64::MAX, 2)
    /// );
    /// ```
    #[inline]
    pub const fn saturating_neg(&self) -> Self {
        Self(self.0.saturating_neg(), self.1)
    }
}

impl Mul for Multiplier {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.checked_mul(rhs)
            .expect("Multiplier: Overflow when multiplying")
    }
}

/// Conversion which saturates at the maximum or maximum instead of overflowing
pub trait SaturatingInto<T>: Sized {
    /// Performs the saturating conversion
    fn saturating_into(self) -> T;
}

/// The duration which is returned by the parser
///
/// The `Duration` of this library implements conversions to a [`std::time::Duration`] and if the
/// feature is activated into a [`time::Duration`] respectively [`chrono::Duration`]. This crates
/// duration is a superset of the aforementioned durations, so converting to fundu's duration with
/// `From` or `Into` is lossless. Converting from [`crate::time::Duration`] to the other durations
/// can overflow the other duration's value range, but `TryFrom` is implement for all of these
/// durations. Note that fundu's duration also implements [`SaturatingInto`] for the above durations
/// which performs the conversion saturating at the maximum or minimum of these durations.
///
/// # Examples
///
/// Basic conversions from [`Duration`] to [`std::time::Duration`].
///
/// ```rust
/// use std::time::Duration as StdDuration;
///
/// use fundu_core::error::TryFromDurationError;
/// use fundu_core::time::{Duration, SaturatingInto};
///
/// let result: Result<StdDuration, TryFromDurationError> = Duration::positive(1, 2).try_into();
/// assert_eq!(result, Ok(StdDuration::new(1, 2)));
///
/// let result: Result<StdDuration, TryFromDurationError> = Duration::negative(1, 2).try_into();
/// assert_eq!(result, Err(TryFromDurationError::NegativeDuration));
///
/// let duration: StdDuration = Duration::negative(1, 2).saturating_into();
/// assert_eq!(duration, StdDuration::ZERO);
///
/// let duration: StdDuration = Duration::MAX.saturating_into();
/// assert_eq!(duration, StdDuration::MAX);
/// ```
///
/// [`time::Duration`]: https://docs.rs/time/latest/time/struct.Duration.html
#[derive(Debug, Eq, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duration {
    is_negative: bool,
    inner: std::time::Duration,
}

impl Duration {
    /// A duration of zero time
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::ZERO;
    /// assert!(duration.is_zero());
    /// ```
    pub const ZERO: Self = Self {
        is_negative: false,
        inner: std::time::Duration::ZERO,
    };

    /// The minimum duration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::MIN;
    /// assert_eq!(Duration::negative(u64::MAX, 999_999_999), duration);
    /// ```
    pub const MIN: Self = Self {
        is_negative: true,
        inner: std::time::Duration::MAX,
    };

    /// The maximum duration
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::MAX;
    /// assert_eq!(Duration::positive(u64::MAX, 999_999_999), duration);
    /// ```
    pub const MAX: Self = Self {
        is_negative: false,
        inner: std::time::Duration::MAX,
    };

    /// Creates a new `Duration` from a [`std::time::Duration`] which can be negative or positive
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::from_std(false, std::time::Duration::new(1, 0));
    /// assert_eq!(Duration::positive(1, 0), duration);
    ///
    /// let duration = Duration::from_std(true, std::time::Duration::new(1, 0));
    /// assert_eq!(Duration::negative(1, 0), duration);
    /// ```
    pub const fn from_std(is_negative: bool, inner: std::time::Duration) -> Self {
        Self { is_negative, inner }
    }

    /// Creates a new positive `Duration`
    ///
    /// # Panics
    ///
    /// This constructor will panic if creating a [`std::time::Duration`] with the same parameters
    /// would panic
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::positive(1, 0);
    /// assert!(duration.is_positive());
    ///
    /// let duration = Duration::positive(0, 0);
    /// assert!(duration.is_positive());
    /// ```
    pub const fn positive(secs: u64, nanos: u32) -> Self {
        Self {
            is_negative: false,
            inner: std::time::Duration::new(secs, nanos),
        }
    }

    /// Creates a new negative `Duration`
    ///
    /// # Panics
    ///
    /// This constructor will panic if creating a [`std::time::Duration`] with the same parameters
    /// would panic
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::negative(1, 0);
    /// assert!(duration.is_negative());
    /// ```
    pub const fn negative(secs: u64, nanos: u32) -> Self {
        Self {
            is_negative: true,
            inner: std::time::Duration::new(secs, nanos),
        }
    }

    // factor must be greater than 1
    const fn as_whole(&self, factor: u64) -> i64 {
        debug_assert!(factor > 1);

        #[allow(clippy::cast_possible_wrap)]
        let result = (self.inner.as_secs() / factor) as i64;
        if self.is_negative { -result } else { result }
    }

    /// Return the number of *whole* weeks in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(86400 * 7, 0).as_weeks(), 1);
    /// assert_eq!(Duration::negative(86400 * 7, 0).as_weeks(), -1);
    /// assert_eq!(Duration::positive(1, 0).as_weeks(), 0);
    /// assert_eq!(Duration::positive(1_500_000, 0).as_weeks(), 2);
    /// ```
    #[inline]
    pub const fn as_weeks(&self) -> i64 {
        self.as_whole(SECS_PER_WEEK)
    }

    /// Return the number of *whole* days in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(86400, 0).as_days(), 1);
    /// assert_eq!(Duration::negative(86400, 0).as_days(), -1);
    /// assert_eq!(Duration::positive(1, 0).as_days(), 0);
    /// assert_eq!(Duration::positive(1_500_000, 0).as_days(), 17);
    /// ```
    #[inline]
    pub const fn as_days(&self) -> i64 {
        self.as_whole(SECS_PER_DAY)
    }

    /// Return the number of *whole* hours in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(3600, 0).as_hours(), 1);
    /// assert_eq!(Duration::negative(3600, 0).as_hours(), -1);
    /// assert_eq!(Duration::positive(1, 0).as_hours(), 0);
    /// assert_eq!(Duration::positive(1_500_000, 0).as_hours(), 416);
    /// ```
    #[inline]
    pub const fn as_hours(&self) -> i64 {
        self.as_whole(SECS_PER_HOUR)
    }

    /// Return the number of *whole* minutes in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(60, 0).as_minutes(), 1);
    /// assert_eq!(Duration::negative(60, 0).as_minutes(), -1);
    /// assert_eq!(Duration::positive(1, 0).as_minutes(), 0);
    /// assert_eq!(Duration::positive(1_500_000, 0).as_minutes(), 25_000);
    /// ```
    #[inline]
    pub const fn as_minutes(&self) -> i64 {
        self.as_whole(SECS_PER_MINUTE)
    }

    /// Return the number of *whole* seconds in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(1, 0).as_seconds(), 1);
    /// assert_eq!(Duration::negative(1, 0).as_seconds(), -1);
    /// assert_eq!(Duration::positive(0, 1_000_000).as_seconds(), 0);
    /// assert_eq!(Duration::positive(1_500_000, 0).as_seconds(), 1_500_000);
    /// ```
    #[inline]
    pub const fn as_seconds(&self) -> i128 {
        let seconds = self.inner.as_secs() as i128;
        if self.is_negative { -seconds } else { seconds }
    }

    /// Return the total number of *whole* milliseconds in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(1, 0).as_millis(), 1_000);
    /// assert_eq!(Duration::negative(1, 0).as_millis(), -1_000);
    /// assert_eq!(Duration::positive(0, 1_000_000).as_millis(), 1);
    /// assert_eq!(Duration::positive(12, 3_000_000).as_millis(), 12_003);
    /// ```
    #[inline]
    pub const fn as_millis(&self) -> i128 {
        self.as_seconds() * MILLIS_PER_SEC + self.subsec_millis() as i128
    }

    /// Return the total number of *whole* microseconds in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(1, 0).as_micros(), 1_000_000);
    /// assert_eq!(Duration::negative(1, 0).as_micros(), -1_000_000);
    /// assert_eq!(Duration::positive(0, 1_000).as_micros(), 1);
    /// assert_eq!(Duration::positive(12, 3_000).as_micros(), 12_000_003);
    /// ```
    #[inline]
    pub const fn as_micros(&self) -> i128 {
        self.as_seconds() * MICROS_PER_SEC + self.subsec_micros() as i128
    }

    /// Return the total number of nanoseconds in the `Duration`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(1, 0).as_nanos(), 1_000_000_000);
    /// assert_eq!(Duration::negative(1, 0).as_nanos(), -1_000_000_000);
    /// assert_eq!(Duration::positive(0, 1).as_nanos(), 1);
    /// assert_eq!(Duration::positive(12, 3).as_nanos(), 12_000_000_003);
    /// ```
    #[inline]
    pub const fn as_nanos(&self) -> i128 {
        self.as_seconds() * NANOS_PER_SEC + self.subsec_nanos() as i128
    }

    /// Return the fractional part of the `Duration` in *whole* milliseconds
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(0, 123_456_789).subsec_millis(), 123);
    /// assert_eq!(Duration::negative(0, 123_456_789).subsec_millis(), -123);
    /// assert_eq!(Duration::positive(1, 0).subsec_millis(), 0);
    /// assert_eq!(Duration::positive(0, 1).subsec_millis(), 0);
    /// ```
    #[inline]
    pub const fn subsec_millis(&self) -> i32 {
        self.subsec_nanos() / NANOS_PER_MILLI
    }

    /// Return the fractional part of the `Duration` in *whole* microseconds
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(0, 123_456_789).subsec_micros(), 123_456);
    /// assert_eq!(Duration::negative(0, 123_456_789).subsec_micros(), -123_456);
    /// assert_eq!(Duration::positive(1, 0).subsec_micros(), 0);
    /// assert_eq!(Duration::positive(0, 1).subsec_micros(), 0);
    #[inline]
    pub const fn subsec_micros(&self) -> i32 {
        self.subsec_nanos() / NANOS_PER_MICRO
    }

    /// Return the fractional part of the `Duration` in nanoseconds
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(Duration::positive(0, 123_456_789).subsec_nanos(), 123_456_789);
    /// assert_eq!(Duration::negative(0, 123_456_789).subsec_nanos(), -123_456_789);
    /// assert_eq!(Duration::positive(1, 0).subsec_nanos(), 0);
    /// assert_eq!(Duration::positive(0, 1).subsec_nanos(), 1);
    #[inline]
    pub const fn subsec_nanos(&self) -> i32 {
        #[allow(clippy::cast_possible_wrap)]
        let nanos = self.inner.subsec_nanos() as i32;
        if self.is_negative { -nanos } else { nanos }
    }

    fn extract_i64(&mut self, factor: u64) -> i64 {
        debug_assert!(factor > 1);

        let secs = self.inner.as_secs();
        let extracted = i64::try_from(secs / factor).unwrap();
        if extracted > 0 {
            self.inner = std::time::Duration::new(secs % factor, self.inner.subsec_nanos());
            if self.is_negative {
                extracted.neg()
            } else {
                extracted
            }
        } else {
            0
        }
    }

    /// Return the number of *whole* weeks in this `Duration` and reduce it by that amount
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let mut duration = Duration::positive(86400 * 7, 0);
    /// assert_eq!(duration.extract_weeks(), 1);
    /// assert_eq!(duration, Duration::ZERO);
    ///
    /// let mut duration = Duration::positive(123, 456);
    /// assert_eq!(duration.extract_weeks(), 0);
    /// assert_eq!(duration, Duration::positive(123, 456));
    /// ```
    #[inline]
    pub fn extract_weeks(&mut self) -> i64 {
        self.extract_i64(SECS_PER_WEEK)
    }

    /// Return the number of *whole* days in this `Duration` and reduce it by that amount
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let mut duration = Duration::positive(86400, 0);
    /// assert_eq!(duration.extract_days(), 1);
    /// assert_eq!(duration, Duration::ZERO);
    ///
    /// let mut duration = Duration::positive(123, 456);
    /// assert_eq!(duration.extract_days(), 0);
    /// assert_eq!(duration, Duration::positive(123, 456));
    /// ```
    #[inline]
    pub fn extract_days(&mut self) -> i64 {
        self.extract_i64(SECS_PER_DAY)
    }

    /// Return the number of *whole* hours in this `Duration` and reduce it by that amount
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let mut duration = Duration::positive(3600, 0);
    /// assert_eq!(duration.extract_hours(), 1);
    /// assert_eq!(duration, Duration::ZERO);
    ///
    /// let mut duration = Duration::positive(123, 456);
    /// assert_eq!(duration.extract_hours(), 0);
    /// assert_eq!(duration, Duration::positive(123, 456));
    /// ```
    #[inline]
    pub fn extract_hours(&mut self) -> i64 {
        self.extract_i64(SECS_PER_HOUR)
    }

    /// Return the number of *whole* minutes in this `Duration` and reduce it by that amount
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let mut duration = Duration::positive(60, 0);
    /// assert_eq!(duration.extract_minutes(), 1);
    /// assert_eq!(duration, Duration::ZERO);
    ///
    /// let mut duration = Duration::positive(12, 456);
    /// assert_eq!(duration.extract_minutes(), 0);
    /// assert_eq!(duration, Duration::positive(12, 456));
    /// ```
    #[inline]
    pub fn extract_minutes(&mut self) -> i64 {
        self.extract_i64(SECS_PER_MINUTE)
    }

    /// Return the number of seconds in this `Duration` and reduce it by that amount
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let mut duration = Duration::positive(1, 0);
    /// assert_eq!(duration.extract_seconds(), 1);
    /// assert_eq!(duration, Duration::ZERO);
    ///
    /// let mut duration = Duration::positive(12, 456);
    /// assert_eq!(duration.extract_seconds(), 12);
    /// assert_eq!(duration, Duration::positive(0, 456));
    /// ```
    #[inline]
    pub fn extract_seconds(&mut self) -> i128 {
        let extracted = self.as_seconds();
        if extracted == 0 {
            0
        } else {
            self.inner = std::time::Duration::new(0, self.inner.subsec_nanos());
            extracted
        }
    }

    /// Returns true if the `Duration` is negative
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::MIN;
    /// assert!(duration.is_negative());
    ///
    /// let duration = Duration::negative(0, 1);
    /// assert!(duration.is_negative());
    /// ```
    #[inline]
    pub const fn is_negative(&self) -> bool {
        self.is_negative
    }

    /// Returns true if the `Duration` is positive
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::ZERO;
    /// assert!(duration.is_positive());
    ///
    /// let duration = Duration::positive(0, 1);
    /// assert!(duration.is_positive());
    /// ```
    #[inline]
    pub const fn is_positive(&self) -> bool {
        !self.is_negative
    }

    /// Returns true if the `Duration` is zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::ZERO;
    /// assert!(duration.is_zero());
    ///
    /// let duration = Duration::positive(0, 0);
    /// assert!(duration.is_zero());
    ///
    /// let duration = Duration::negative(0, 0);
    /// assert!(duration.is_zero());
    /// ```
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns the absolute value of the duration
    ///
    /// This operation is lossless.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// let duration = Duration::MIN;
    /// assert_eq!(duration.abs(), Duration::MAX);
    ///
    /// let duration = Duration::negative(1, 0);
    /// assert_eq!(duration.abs(), Duration::positive(1, 0));
    ///
    /// let duration = Duration::positive(1, 0);
    /// assert_eq!(duration.abs(), Duration::positive(1, 0));
    /// ```
    #[inline]
    pub const fn abs(&self) -> Self {
        Self::from_std(false, self.inner)
    }

    /// Sums this duration with the `other` duration, returning None if an overflow occurred
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(
    ///     Duration::positive(1, 0).checked_add(Duration::positive(1, 0)),
    ///     Some(Duration::positive(2, 0))
    /// );
    /// assert_eq!(
    ///     Duration::positive(u64::MAX, 0).checked_add(Duration::positive(1, 0)),
    ///     None
    /// );
    /// assert_eq!(
    ///     Duration::negative(u64::MAX, 0).checked_add(Duration::negative(1, 0)),
    ///     None
    /// );
    /// ```
    pub fn checked_add(&self, other: Self) -> Option<Self> {
        match (
            self.is_negative,
            other.is_negative,
            self.inner.cmp(&other.inner),
        ) {
            (true, true, _) => self
                .inner
                .checked_add(other.inner)
                .map(|d| Self::from_std(true, d)),
            (true, false, Ordering::Equal) | (false, true, Ordering::Equal) => Some(Self::ZERO),
            (true, false, Ordering::Greater) => {
                Some(Self::from_std(true, self.inner.sub(other.inner)))
            }
            (true, false, Ordering::Less) => {
                Some(Self::from_std(false, other.inner.sub(self.inner)))
            }
            (false, true, Ordering::Greater) => {
                Some(Self::from_std(false, self.inner.sub(other.inner)))
            }
            (false, true, Ordering::Less) => {
                Some(Self::from_std(true, other.inner.sub(self.inner)))
            }
            (false, false, _) => self
                .inner
                .checked_add(other.inner)
                .map(|d| Self::from_std(false, d)),
        }
    }

    /// Subtracts this duration with the `other` duration, returning None if an overflow occurred
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(
    ///     Duration::positive(1, 0).checked_sub(Duration::positive(1, 0)),
    ///     Some(Duration::ZERO)
    /// );
    /// assert_eq!(
    ///     Duration::negative(u64::MAX, 0).checked_sub(Duration::positive(1, 0)),
    ///     None
    /// );
    /// ```
    #[inline]
    pub fn checked_sub(&self, other: Self) -> Option<Self> {
        self.checked_add(other.neg())
    }

    /// Saturating [`Duration`] addition. Computes `self + other`, returning [`Duration::MAX`] or
    /// [`Duration::MIN`] if an overflow occurred.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(
    ///     Duration::positive(1, 0).saturating_add(Duration::positive(0, 1)),
    ///     Duration::positive(1, 1)
    /// );
    /// assert_eq!(
    ///     Duration::positive(u64::MAX, 0).saturating_add(Duration::positive(1, 0)),
    ///     Duration::MAX
    /// );
    /// ```
    pub fn saturating_add(&self, other: Self) -> Self {
        match self.checked_add(other) {
            Some(d) => d,
            // checked_add only returns None if both durations are either negative or positive so it
            // is enough to check one of the durations for negativity
            None if self.is_negative => Self::MIN,
            None => Self::MAX,
        }
    }

    /// Saturating [`Duration`] subtraction. Computes `self - other`, returning [`Duration::MAX`] or
    /// [`Duration::MIN`] if an overflow occurred.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fundu_core::time::Duration;
    ///
    /// assert_eq!(
    ///     Duration::positive(1, 0).saturating_sub(Duration::positive(1, 0)),
    ///     Duration::ZERO
    /// );
    /// assert_eq!(
    ///     Duration::negative(u64::MAX, 0).saturating_sub(Duration::positive(1, 0)),
    ///     Duration::MIN
    /// );
    /// ```
    #[inline]
    pub fn saturating_sub(&self, other: Self) -> Self {
        self.saturating_add(other.neg())
    }
}

impl Display for Duration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const YEAR: u64 = Year.multiplier().0.unsigned_abs();
        const MONTH: u64 = Month.multiplier().0.unsigned_abs();
        const WEEK: u64 = Week.multiplier().0.unsigned_abs();
        const DAY: u64 = Day.multiplier().0.unsigned_abs();
        const HOUR: u64 = Hour.multiplier().0.unsigned_abs();
        const MINUTE: u64 = Minute.multiplier().0.unsigned_abs();
        const MILLIS_PER_NANO: u32 = 1_000_000;
        const MICROS_PER_NANO: u32 = 1_000;

        if self.is_zero() {
            return f.write_str("0ns");
        }

        let mut result = Vec::with_capacity(10);
        let mut secs = self.inner.as_secs();
        if secs > 0 {
            if secs >= YEAR {
                result.push(format!("{}y", secs / YEAR));
                secs %= YEAR;
            }
            if secs >= MONTH {
                result.push(format!("{}M", secs / MONTH));
                secs %= MONTH;
            }
            if secs >= WEEK {
                result.push(format!("{}w", secs / WEEK));
                secs %= WEEK;
            }
            if secs >= DAY {
                result.push(format!("{}d", secs / DAY));
                secs %= DAY;
            }
            if secs >= HOUR {
                result.push(format!("{}h", secs / HOUR));
                secs %= HOUR;
            }
            if secs >= MINUTE {
                result.push(format!("{}m", secs / MINUTE));
                secs %= MINUTE;
            }
            if secs >= 1 {
                result.push(format!("{secs}s"));
            }
        }

        let mut nanos = self.inner.subsec_nanos();
        if nanos > 0 {
            if nanos >= MILLIS_PER_NANO {
                result.push(format!("{}ms", nanos / MILLIS_PER_NANO));
                nanos %= MILLIS_PER_NANO;
            }
            if nanos >= MICROS_PER_NANO {
                result.push(format!("{}Ms", nanos / MICROS_PER_NANO));
                nanos %= MICROS_PER_NANO;
            }
            if nanos >= 1 {
                result.push(format!("{nanos}ns"));
            }
        }

        if self.is_negative() {
            f.write_str(&format!("-{}", &result.join(" -")))
        } else {
            f.write_str(&result.join(" "))
        }
    }
}

impl Add for Duration {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.checked_add(rhs)
            .expect("Overflow when adding duration")
    }
}

impl AddAssign for Duration {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Duration {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.checked_sub(rhs)
            .expect("Overflow when subtracting duration")
    }
}

impl SubAssign for Duration {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Duration {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            is_negative: self.is_negative ^ true,
            inner: self.inner,
        }
    }
}

impl PartialEq for Duration {
    fn eq(&self, other: &Self) -> bool {
        (self.is_zero() && other.is_zero())
            || (self.is_negative == other.is_negative && self.inner == other.inner)
    }
}

impl Hash for Duration {
    fn hash<H: Hasher>(&self, state: &mut H) {
        if self.is_zero() {
            false.hash(state);
        } else {
            self.is_negative.hash(state);
        }
        self.inner.hash(state);
    }
}

impl PartialOrd for Duration {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Duration {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.is_negative, other.is_negative) {
            (true, true) => other.inner.cmp(&self.inner),
            (true, false) | (false, true) if self.is_zero() && other.is_zero() => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.inner.cmp(&other.inner),
        }
    }
}

/// Convert a [`std::time::Duration`] into a [`Duration`]
impl From<std::time::Duration> for Duration {
    fn from(duration: std::time::Duration) -> Self {
        Self {
            is_negative: false,
            inner: duration,
        }
    }
}

impl SaturatingInto<std::time::Duration> for Duration {
    fn saturating_into(self) -> std::time::Duration {
        self.try_into().unwrap_or_else(|error| match error {
            TryFromDurationError::NegativeDuration => std::time::Duration::ZERO,
            _ => unreachable!(), // cov:excl-line
        })
    }
}

/// Convert a [`Duration`] into a [`std::time::Duration`]
impl TryFrom<Duration> for std::time::Duration {
    type Error = TryFromDurationError;

    fn try_from(duration: Duration) -> Result<Self, Self::Error> {
        if !duration.is_zero() && duration.is_negative {
            Err(TryFromDurationError::NegativeDuration)
        } else {
            Ok(duration.inner)
        }
    }
}

#[cfg(feature = "time")]
/// Convert a [`time::Duration`] into a [`Duration`]
///
/// [`time::Duration`]: <https://docs.rs/time/latest/time/struct.Duration.html>
impl From<time::Duration> for Duration {
    fn from(duration: time::Duration) -> Self {
        Self {
            is_negative: duration.is_negative(),
            inner: std::time::Duration::new(
                duration.whole_seconds().unsigned_abs(),
                duration.subsec_nanoseconds().unsigned_abs(),
            ),
        }
    }
}

#[cfg(feature = "time")]
/// Convert a [`Duration`] into a [`time::Duration`]
///
/// [`time::Duration`]: <https://docs.rs/time/latest/time/struct.Duration.html>
impl TryFrom<Duration> for time::Duration {
    type Error = TryFromDurationError;

    fn try_from(duration: Duration) -> Result<Self, Self::Error> {
        (&duration).try_into()
    }
}

#[cfg(feature = "time")]
/// Convert a [`Duration`] into a [`time::Duration`]
///
/// [`time::Duration`]: <https://docs.rs/time/latest/time/struct.Duration.html>
impl TryFrom<&Duration> for time::Duration {
    type Error = TryFromDurationError;

    fn try_from(duration: &Duration) -> Result<Self, Self::Error> {
        #[allow(clippy::cast_possible_wrap)]
        match (duration.is_negative, duration.inner.as_secs()) {
            (true, secs) if secs > i64::MIN.unsigned_abs() => {
                Err(TryFromDurationError::NegativeOverflow)
            }
            (true, secs) => Ok(Self::new(
                secs.wrapping_neg() as i64,
                -(duration.inner.subsec_nanos() as i32),
            )),
            (false, secs) if secs > i64::MAX as u64 => Err(TryFromDurationError::PositiveOverflow),
            (false, secs) => Ok(Self::new(secs as i64, duration.inner.subsec_nanos() as i32)),
        }
    }
}

#[cfg(feature = "time")]
impl SaturatingInto<time::Duration> for Duration {
    fn saturating_into(self) -> time::Duration {
        self.try_into().unwrap_or_else(|error| match error {
            TryFromDurationError::PositiveOverflow => time::Duration::MAX,
            TryFromDurationError::NegativeOverflow => time::Duration::MIN,
            TryFromDurationError::NegativeDuration => unreachable!(), // cov:excl-line
        })
    }
}

#[cfg(feature = "chrono")]
/// Convert a [`Duration`] into a [`chrono::Duration`]
impl TryFrom<Duration> for chrono::Duration {
    type Error = TryFromDurationError;

    fn try_from(duration: Duration) -> Result<Self, Self::Error> {
        (&duration).try_into()
    }
}

#[cfg(feature = "chrono")]
/// Convert a [`Duration`] into a [`chrono::Duration`]
impl TryFrom<&Duration> for chrono::Duration {
    type Error = TryFromDurationError;

    fn try_from(duration: &Duration) -> Result<Self, Self::Error> {
        const MAX: chrono::Duration = chrono::Duration::max_value();
        const MIN: chrono::Duration = chrono::Duration::min_value();

        #[allow(clippy::cast_possible_wrap)]
        match (duration.is_negative, duration.inner.as_secs()) {
            (true, secs) if secs > MIN.num_seconds().unsigned_abs() => {
                Err(TryFromDurationError::NegativeOverflow)
            }
            (true, secs) => {
                let nanos = Self::nanoseconds((i64::from(duration.inner.subsec_nanos())).neg());
                Self::seconds((secs as i64).neg())
                    .checked_add(&nanos)
                    .ok_or(TryFromDurationError::NegativeOverflow)
            }
            (false, secs) if secs > MAX.num_seconds().unsigned_abs() => {
                Err(TryFromDurationError::PositiveOverflow)
            }
            (false, secs) => {
                let nanos = Self::nanoseconds(i64::from(duration.inner.subsec_nanos()));
                Self::seconds(secs as i64)
                    .checked_add(&nanos)
                    .ok_or(TryFromDurationError::PositiveOverflow)
            }
        }
    }
}

#[cfg(feature = "chrono")]
impl SaturatingInto<chrono::Duration> for Duration {
    fn saturating_into(self) -> chrono::Duration {
        self.try_into().unwrap_or_else(|error| match error {
            TryFromDurationError::PositiveOverflow => chrono::Duration::max_value(),
            TryFromDurationError::NegativeOverflow => chrono::Duration::min_value(),
            TryFromDurationError::NegativeDuration => unreachable!(), // cov:excl-line
        })
    }
}

#[cfg(feature = "chrono")]
/// Convert a [`chrono::Duration`] into a [`Duration`]
impl From<chrono::Duration> for Duration {
    fn from(duration: chrono::Duration) -> Self {
        duration.to_std().map_or_else(
            |_| Self {
                is_negative: true,
                inner: duration.abs().to_std().unwrap(),
            },
            |inner| Self {
                is_negative: false,
                inner,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::time::Duration as StdDuration;

    #[cfg(feature = "chrono")]
    use chrono::Duration as ChronoDuration;
    use rstest::rstest;
    use rstest_reuse::{apply, template};
    #[cfg(feature = "serde")]
    use serde_test::{assert_tokens, Token};

    use super::*;

    #[cfg(feature = "chrono")]
    const CHRONO_MIN_DURATION: Duration =
        Duration::negative(i64::MIN.unsigned_abs() / 1000, 807_000_000);
    #[cfg(feature = "chrono")]
    const CHRONO_MAX_DURATION: Duration = Duration::positive(i64::MAX as u64 / 1000, 807_000_000);

    const YEAR_AS_SECS: u64 = 60 * 60 * 24 * 365 + 60 * 60 * 24 / 4; // 365 days + day/4
    const MONTH_AS_SECS: u64 = YEAR_AS_SECS / 12;

    #[cfg(feature = "serde")]
    #[test]
    fn test_time_unit_serde() {
        let time_unit = TimeUnit::Day;

        assert_tokens(
            &time_unit,
            &[
                Token::Enum { name: "TimeUnit" },
                Token::Str("Day"),
                Token::Unit,
            ],
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_multiplier() {
        let multiplier = Multiplier(1, 2);

        assert_tokens(
            &multiplier,
            &[
                Token::TupleStruct {
                    name: "Multiplier",
                    len: 2,
                },
                Token::I64(1),
                Token::I16(2),
                Token::TupleStructEnd,
            ],
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_serde_duration() {
        let duration = Duration::positive(1, 2);

        assert_tokens(
            &duration,
            &[
                Token::Struct {
                    name: "Duration",
                    len: 2,
                },
                Token::Str("is_negative"),
                Token::Bool(false),
                Token::Str("inner"),
                Token::Struct {
                    name: "Duration",
                    len: 2,
                },
                Token::Str("secs"),
                Token::U64(1),
                Token::Str("nanos"),
                Token::U32(2),
                Token::StructEnd,
                Token::StructEnd,
            ],
        );
    }

    #[rstest]
    #[case::nano_second(NanoSecond, "ns")]
    #[case::micro_second(MicroSecond, "Ms")]
    #[case::milli_second(MilliSecond, "ms")]
    #[case::second(Second, "s")]
    #[case::minute(Minute, "m")]
    #[case::hour(Hour, "h")]
    #[case::day(Day, "d")]
    #[case::week(Week, "w")]
    #[case::month(Month, "M")]
    #[case::year(Year, "y")]
    fn test_time_unit_default_identifier(#[case] time_unit: TimeUnit, #[case] expected: &str) {
        assert_eq!(time_unit.default_identifier(), expected);
    }

    #[rstest]
    #[case::nano_second(NanoSecond, Multiplier(1, -9))]
    #[case::micro_second(MicroSecond, Multiplier(1, -6))]
    #[case::milli_second(MilliSecond, Multiplier(1, -3))]
    #[case::second(Second, Multiplier(1, 0))]
    #[case::minute(Minute, Multiplier(60, 0))]
    #[case::hour(Hour, Multiplier(60 * 60, 0))]
    #[case::day(Day, Multiplier(60 * 60 * 24, 0))]
    #[case::week(Week, Multiplier(60 * 60 * 24 * 7, 0))]
    #[case::month(Month, Multiplier((60 * 60 * 24 * 365 + 60 * 60 * 24 / 4) / 12, 0))] // (365 days + day/4) / 12
    #[case::year(Year, Multiplier(60 * 60 * 24 * 365 + 60 * 60 * 24 / 4, 0))] // 365 days + day/4
    fn test_time_unit_multiplier(#[case] time_unit: TimeUnit, #[case] expected: Multiplier) {
        assert_eq!(time_unit.multiplier(), expected);
    }

    #[test]
    fn test_multiplier_get_coefficient() {
        let multi = Multiplier(1234, 0);
        assert_eq!(multi.coefficient(), 1234);
    }

    #[test]
    fn test_multiplier_get_exponent() {
        let multi = Multiplier(0, 1234);
        assert_eq!(multi.exponent(), 1234);
    }

    #[rstest]
    #[case::zero(Multiplier(0, 0), false)]
    #[case::negative(Multiplier(-1, 0), true)]
    #[case::positive(Multiplier(1, 0), false)]
    #[case::negative_exponent(Multiplier(1, -1), false)]
    #[case::positive_exponent(Multiplier(-1, 1), true)]
    fn test_multiplier_is_negative_and_is_positive(
        #[case] multi: Multiplier,
        #[case] expected: bool,
    ) {
        assert_eq!(multi.is_negative(), expected);
        assert_eq!(multi.is_positive(), !expected);
    }

    #[rstest]
    #[case::nano_second(NanoSecond, Multiplier(i64::MAX, i16::MIN + 9))]
    #[case::micro_second(MicroSecond, Multiplier(i64::MAX, i16::MIN + 6))]
    #[case::milli_second(MilliSecond, Multiplier(i64::MAX, i16::MIN + 3))]
    #[case::second(Second, Multiplier(i64::MAX, i16::MIN))]
    #[case::minute(Minute, Multiplier(i64::MAX / 60, i16::MIN))]
    #[case::hour(Hour, Multiplier(i64::MAX / (60 * 60), i16::MIN))]
    #[case::day(Day, Multiplier(i64::MAX / (60 * 60 * 24), i16::MIN))]
    #[case::week(Week, Multiplier(i64::MAX / (60 * 60 * 24 * 7), i16::MIN))]
    #[case::month(Month, Multiplier(3_507_252_276_543, i16::MIN))]
    #[case::year(Year, Multiplier(292_271_023_045, i16::MIN))]
    fn test_multiplier_multiplication_barely_no_panic(
        #[case] time_unit: TimeUnit,
        #[case] multiplier: Multiplier,
    ) {
        _ = time_unit.multiplier() * multiplier;
    }

    #[rstest]
    #[case::nano_second(NanoSecond, Multiplier(i64::MAX, i16::MIN + 8))]
    #[case::micro_second(MicroSecond, Multiplier(i64::MAX, i16::MIN + 4))]
    #[case::milli_second(MilliSecond, Multiplier(i64::MAX, i16::MIN + 2))]
    #[case::minute(Minute, Multiplier(i64::MAX / 60 + 1, i16::MIN))]
    #[case::hour(Hour, Multiplier(i64::MAX / (60 * 60) + 1, i16::MIN))]
    #[case::day(Day, Multiplier(i64::MAX / (60 * 60 * 24) + 1, i16::MIN))]
    #[case::week(Week, Multiplier(i64::MAX / (60 * 60 * 24 * 7) + 1, i16::MIN))]
    #[case::month(Month, Multiplier(3_507_252_276_543 + 1, i16::MIN))]
    #[case::year(Year, Multiplier(292_271_023_045 + 1, i16::MIN))]
    #[should_panic(expected = "Overflow")]
    fn test_multiplier_multiplication_then_panic(
        #[case] time_unit: TimeUnit,
        #[case] multiplier: Multiplier,
    ) {
        _ = time_unit.multiplier() * multiplier;
    }

    #[rstest]
    #[case::positive_zero(Duration::positive(0, 0), true)]
    #[case::negative_zero(Duration::negative(0, 0), false)] // FIXME: This should return true
    #[case::positive_one(Duration::positive(1, 0), true)]
    #[case::negative_one(Duration::negative(1, 0), false)]
    fn test_fundu_duration_is_positive_and_is_negative(
        #[case] duration: Duration,
        #[case] expected: bool,
    ) {
        assert_eq!(duration.is_positive(), expected);
        assert_eq!(duration.is_negative(), !expected);
    }

    #[rstest]
    #[case::positive_zero(Duration::positive(0, 0), Duration::positive(0, 0))]
    #[case::negative_zero(Duration::negative(0, 0), Duration::positive(0, 0))]
    #[case::positive_one(Duration::positive(1, 0), Duration::positive(1, 0))]
    #[case::positive_one_nano(Duration::positive(0, 1), Duration::positive(0, 1))]
    #[case::negative_one(Duration::negative(1, 0), Duration::positive(1, 0))]
    #[case::negative_one_nano(Duration::negative(0, 1), Duration::positive(0, 1))]
    #[case::negative_one_one(Duration::negative(1, 1), Duration::positive(1, 1))]
    fn test_fundu_duration_abs(#[case] duration: Duration, #[case] expected: Duration) {
        assert_eq!(duration.abs(), expected);
    }

    #[rstest]
    #[case::zero(Duration::ZERO, "0ns")]
    #[case::nano_second(Duration::positive(0, 1), "1ns")]
    #[case::micro_second(Duration::positive(0, 1_000), "1Ms")]
    #[case::milli_second(Duration::positive(0, 1_000_000), "1ms")]
    #[case::second(Duration::positive(1, 0), "1s")]
    #[case::minute(Duration::positive(60, 0), "1m")]
    #[case::hour(Duration::positive(60 * 60, 0), "1h")]
    #[case::day(Duration::positive(60 * 60 * 24, 0), "1d")]
    #[case::week(Duration::positive(60 * 60 * 24 * 7, 0), "1w")]
    #[case::month(Duration::positive(MONTH_AS_SECS, 0), "1M")]
    #[case::year(Duration::positive(YEAR_AS_SECS, 0), "1y")]
    #[case::all_one(
        Duration::positive(
            YEAR_AS_SECS + MONTH_AS_SECS + 60 * 60 * 24 * 8 + 60 * 60 + 60 + 1,
            1_001_001
        ),
        "1y 1M 1w 1d 1h 1m 1s 1ms 1Ms 1ns"
    )]
    #[case::max(Duration::MAX, "584542046090y 7M 2w 1d 17h 30m 15s 999ms 999Ms 999ns")]
    fn test_fundu_display(#[case] duration: Duration, #[case] expected: &str) {
        assert_eq!(duration.to_string(), expected.to_owned());
        if duration.is_zero() {
            assert_eq!(duration.neg().to_string(), expected.to_owned());
        } else {
            assert_eq!(
                duration.neg().to_string(),
                format!("-{}", expected.replace(' ', " -"))
            );
        }
    }

    #[template]
    #[rstest]
    #[case::both_standard_zero(Duration::ZERO, Duration::ZERO, Duration::ZERO)]
    #[case::both_positive_zero(
        Duration::positive(0, 0),
        Duration::positive(0, 0),
        Duration::positive(0, 0)
    )]
    #[case::both_negative_zero(Duration::negative(0, 0), Duration::negative(0, 0), Duration::ZERO)]
    #[case::one_add_zero(Duration::positive(1, 0), Duration::ZERO, Duration::positive(1, 0))]
    #[case::minus_one_add_zero(Duration::negative(1, 0), Duration::ZERO, Duration::negative(1, 0))]
    #[case::minus_one_add_plus_one(
        Duration::negative(1, 0),
        Duration::positive(1, 0),
        Duration::ZERO
    )]
    #[case::minus_one_add_plus_two_then_carry(
        Duration::negative(1, 0),
        Duration::positive(2, 0),
        Duration::positive(1, 0)
    )]
    #[case::minus_one_nano_add_one_then_carry(
        Duration::negative(0, 1),
        Duration::positive(1, 0),
        Duration::positive(0, 999_999_999)
    )]
    #[case::plus_one_nano_add_minus_one_then_carry(
        Duration::positive(0, 1),
        Duration::negative(1, 0),
        Duration::negative(0, 999_999_999)
    )]
    #[case::plus_one_add_minus_two_then_carry(
        Duration::positive(1, 0),
        Duration::negative(2, 0),
        Duration::negative(1, 0)
    )]
    #[case::one_sec_below_min_add_max(
        Duration::negative(u64::MAX - 1, 999_999_999),
        Duration::MAX,
        Duration::positive(1, 0),
    )]
    #[case::one_nano_below_min_add_max(
        Duration::negative(u64::MAX, 999_999_998),
        Duration::MAX,
        Duration::positive(0, 1)
    )]
    #[case::one_sec_below_max_add_min(
        Duration::positive(u64::MAX - 1, 999_999_999),
        Duration::MIN,
        Duration::negative(1, 0)
    )]
    #[case::one_nano_below_max_add_min(
        Duration::positive(u64::MAX, 999_999_998),
        Duration::MIN,
        Duration::negative(0, 1)
    )]
    #[case::min_and_max(Duration::MIN, Duration::MAX, Duration::ZERO)]
    fn test_fundu_duration_add_no_overflow_template(
        #[case] lhs: Duration,
        #[case] rhs: Duration,
        #[case] expected: Duration,
    ) {
    }

    #[apply(test_fundu_duration_add_no_overflow_template)]
    fn test_fundu_duration_add(lhs: Duration, rhs: Duration, expected: Duration) {
        assert_eq!(lhs + rhs, expected);
        assert_eq!(rhs + lhs, expected);
    }

    #[apply(test_fundu_duration_add_no_overflow_template)]
    fn test_fundu_duration_add_assign(lhs: Duration, rhs: Duration, expected: Duration) {
        let mut res = lhs;
        res += rhs;
        assert_eq!(res, expected);
        let mut res = rhs;
        res += lhs;
        assert_eq!(res, expected);
    }

    #[apply(test_fundu_duration_add_no_overflow_template)]
    fn test_fundu_duration_checked_add(lhs: Duration, rhs: Duration, expected: Duration) {
        assert_eq!(lhs.checked_add(rhs), Some(expected));
        assert_eq!(rhs.checked_add(lhs), Some(expected));
    }

    #[apply(test_fundu_duration_add_no_overflow_template)]
    fn test_fundu_duration_sub(lhs: Duration, rhs: Duration, expected: Duration) {
        assert_eq!(lhs - rhs.neg(), expected);
        assert_eq!(rhs - lhs.neg(), expected);
    }

    #[apply(test_fundu_duration_add_no_overflow_template)]
    fn test_fundu_duration_sub_assign(lhs: Duration, rhs: Duration, expected: Duration) {
        let mut res = lhs;
        res -= rhs.neg();
        assert_eq!(res, expected);
        let mut res = rhs;
        res -= lhs.neg();
        assert_eq!(res, expected);
    }

    #[apply(test_fundu_duration_add_no_overflow_template)]
    fn test_fundu_duration_checked_sub(lhs: Duration, rhs: Duration, expected: Duration) {
        assert_eq!(lhs.checked_sub(rhs.neg()), Some(expected));
        assert_eq!(rhs.checked_sub(lhs.neg()), Some(expected));
    }

    #[template]
    #[rstest]
    #[case::min(Duration::MIN, Duration::MIN)]
    #[case::min_minus_one(Duration::MIN, Duration::negative(1, 0))]
    #[case::min_minus_one_nano(Duration::MIN, Duration::negative(0, 1))]
    #[case::max(Duration::MAX, Duration::MAX)]
    #[case::max_plus_one(Duration::MAX, Duration::positive(1, 0))]
    #[case::max_plus_one_nano(Duration::MAX, Duration::positive(0, 1))]
    #[case::positive_middle_nano_overflow(
        Duration::positive(u64::MAX / 2 + 1, 999_999_999 / 2 + 1),
        Duration::positive(u64::MAX / 2, 999_999_999 / 2 + 1)
    )]
    #[case::positive_middle_secs_overflow(
        Duration::positive(u64::MAX / 2 + 1, 999_999_999 / 2 + 1),
        Duration::positive(u64::MAX / 2 + 1, 999_999_999 / 2)
    )]
    #[case::negative_middle_nano_overflow(
        Duration::negative(u64::MAX / 2 + 1, 999_999_999 / 2 + 1),
        Duration::negative(u64::MAX / 2, 999_999_999 / 2 + 1)
    )]
    #[case::negative_middle_secs_overflow(
        Duration::negative(u64::MAX / 2 + 1, 999_999_999 / 2 + 1),
         Duration::negative(u64::MAX / 2 + 1, 999_999_999 / 2)
    )]
    fn test_fundu_duration_add_overflow_template(#[case] lhs: Duration, #[case] rhs: Duration) {}

    #[apply(test_fundu_duration_add_overflow_template)]
    #[should_panic = "Overflow when adding duration"]
    fn test_fundu_duration_add_and_add_assign_then_overflow(mut lhs: Duration, rhs: Duration) {
        lhs += rhs;
    }

    #[apply(test_fundu_duration_add_overflow_template)]
    #[should_panic = "Overflow when subtracting duration"]
    #[allow(clippy::no_effect)]
    fn test_fundu_duration_sub_and_sub_assign_then_overflow(mut lhs: Duration, rhs: Duration) {
        lhs -= rhs.neg();
    }

    #[apply(test_fundu_duration_add_overflow_template)]
    fn test_fundu_duration_checked_add_then_overflow(lhs: Duration, rhs: Duration) {
        assert_eq!(lhs.checked_add(rhs), None);
        assert_eq!(rhs.checked_add(lhs), None);
    }

    #[apply(test_fundu_duration_add_overflow_template)]
    fn test_fundu_duration_checked_sub_then_overflow(lhs: Duration, rhs: Duration) {
        assert_eq!(lhs.checked_sub(rhs.neg()), None);
    }

    #[apply(test_fundu_duration_add_overflow_template)]
    fn test_fundu_duration_saturating_add_then_saturate(lhs: Duration, rhs: Duration) {
        let expected = if lhs.checked_add(rhs).is_none() && lhs.is_positive() && rhs.is_positive() {
            Duration::MAX
        } else {
            Duration::MIN
        };
        assert_eq!(lhs.saturating_add(rhs), expected);
        assert_eq!(rhs.saturating_add(lhs), expected);
    }

    #[apply(test_fundu_duration_add_overflow_template)]
    fn test_fundu_duration_saturating_sub_then_saturate(lhs: Duration, rhs: Duration) {
        let expected =
            if lhs.checked_sub(rhs.neg()).is_none() && lhs.is_negative() && rhs.neg().is_positive()
            {
                Duration::MIN
            } else {
                Duration::MAX
            };
        assert_eq!(lhs.saturating_sub(rhs.neg()), expected);
    }

    #[rstest]
    #[case::zero(Duration::ZERO, 0)]
    #[case::positive_one(Duration::positive(1, 0), 1)]
    #[case::positive_one_nano(Duration::positive(0, 1), 0)]
    #[case::positive_one_sec_and_nano(Duration::positive(1, 1), 1)]
    #[case::negative_one(Duration::negative(1, 0), -1)]
    #[case::negative_one_nano(Duration::negative(0, 1), 0)]
    #[case::negative_one_sec_and_nano(Duration::negative(1, 1), -1)]
    #[case::some_positive_value(Duration::positive(123_456_789, 987_654_321), 123_456_789)]
    #[case::some_negative_value(Duration::negative(123_456_789, 987_654_321), -123_456_789)]
    #[case::min(Duration::MIN, i128::from(u64::MAX).neg())]
    #[case::max(Duration::MAX, i128::from(u64::MAX))]
    fn test_fundu_duration_as_seconds(#[case] duration: Duration, #[case] expected: i128) {
        assert_eq!(duration.as_seconds(), expected);
    }

    #[rstest]
    #[case::zero(Duration::ZERO)]
    #[case::one(Duration::positive(1, 0))]
    #[case::one_nano(Duration::positive(0, 1))]
    #[case::one_sec_and_nano(Duration::positive(1, 1))]
    #[case::one_minute(Duration::positive(60, 0))]
    #[case::one_minute_plus_one_sec(Duration::positive(61, 0))]
    #[case::one_minute_minus_one_sec(Duration::positive(59, 0))]
    #[case::two_minutes(Duration::positive(120, 0))]
    #[case::some_value(Duration::positive(123_456_789, 987_654_321))]
    #[case::min(Duration::MIN)]
    #[case::max(Duration::MAX)]
    fn test_fundu_duration_as_whole(
        #[case] duration: Duration,
        #[values(2, 3, 4, 60, u64::MAX / 2, u64::MAX)] factor: u64,
    ) {
        let mut expected = i64::try_from(duration.inner.as_secs() / factor).unwrap();
        if duration.is_negative() {
            expected = expected.neg();
        }
        assert_eq!(duration.as_whole(factor), expected);
        assert_eq!(duration.neg().as_whole(factor), expected.neg());
    }

    #[rstest]
    #[case::one_week(86400 * 7, 1)]
    #[case::one_week_plus_sec(86400 * 7 + 1, 1)]
    #[case::one_week_minus_sec(86400 * 7 - 1, 0)]
    fn test_fundu_duration_as_weeks(#[case] seconds: u64, #[case] expected: i64) {
        let duration = Duration::positive(seconds, 0);

        assert_eq!(duration.as_weeks(), expected);
        assert_eq!(duration.neg().as_weeks(), expected.neg());
    }

    #[rstest]
    #[case::one_day(86400, 1)]
    #[case::one_day_plus_sec(86400 + 1, 1)]
    #[case::one_day_minus_sec(86400 - 1, 0)]
    fn test_fundu_duration_as_days(#[case] seconds: u64, #[case] expected: i64) {
        let duration = Duration::positive(seconds, 0);

        assert_eq!(duration.as_days(), expected);
        assert_eq!(duration.neg().as_days(), expected.neg());
    }

    #[rstest]
    #[case::one_hour(3600, 1)]
    #[case::one_hour_plus_sec(3600 + 1, 1)]
    #[case::one_hour_minus_sec(3600 - 1, 0)]
    fn test_fundu_duration_as_hours(#[case] seconds: u64, #[case] expected: i64) {
        let duration = Duration::positive(seconds, 0);

        assert_eq!(duration.as_hours(), expected);
        assert_eq!(duration.neg().as_hours(), expected.neg());
    }

    #[rstest]
    #[case::one_minute(60, 1)]
    #[case::one_minute_plus_sec(60 + 1, 1)]
    #[case::one_minute_minus_sec(60 - 1, 0)]
    fn test_fundu_duration_as_minutes(#[case] seconds: u64, #[case] expected: i64) {
        let duration = Duration::positive(seconds, 0);

        assert_eq!(duration.as_minutes(), expected);
        assert_eq!(duration.neg().as_minutes(), expected.neg());
    }

    #[template]
    #[rstest]
    #[case::zero(Duration::ZERO, 0)]
    #[case::one_second(Duration::positive(1, 0), 1_000_000_000)]
    #[case::one_second_and_one_nano(Duration::positive(1, 1), 1_000_000_001)]
    #[case::one_second_and_one_nano(Duration::positive(1, 1), 1_000_000_001)]
    #[case::two_second(Duration::positive(2, 0), 2_000_000_000)]
    #[case::two_second_and_one_nano(Duration::positive(2, 1), 2_000_000_001)]
    #[case::one_sec_and_one_micro(Duration::positive(2, 1_000), 2_000_001_000)]
    #[case::one_sec_and_one_milli(Duration::positive(2, 1_000_000), 2_001_000_000)]
    #[case::some_value(Duration::positive(123_456_789, 987_654_321), 123_456_789_987_654_321)]
    #[case::max(Duration::MAX, i128::from(u64::MAX) * 1_000_000_000 + 999_999_999)]
    #[case::min(Duration::MIN, i128::from(u64::MAX).neg() * 1_000_000_000 - 999_999_999)]
    fn test_fundu_duration_as_sub_sec_template(#[case] duration: Duration, #[case] expected: i128) {
    }

    #[apply(test_fundu_duration_as_sub_sec_template)]
    fn test_fundu_duration_as_nanos(duration: Duration, expected: i128) {
        assert_eq!(duration.as_nanos(), expected);
        assert_eq!(duration.neg().as_nanos(), expected.neg());
    }

    #[apply(test_fundu_duration_as_sub_sec_template)]
    fn test_fundu_duration_as_micros(duration: Duration, expected: i128) {
        assert_eq!(duration.as_micros(), expected / 1_000);
        assert_eq!(duration.neg().as_micros(), expected.neg() / 1_000);
    }

    #[apply(test_fundu_duration_as_sub_sec_template)]
    fn test_fundu_duration_as_millis(duration: Duration, expected: i128) {
        assert_eq!(duration.as_millis(), expected / 1_000_000);
        assert_eq!(duration.neg().as_millis(), expected.neg() / 1_000_000);
    }

    #[template]
    #[rstest]
    #[case::zero(Duration::ZERO, 0i32)]
    #[case::one_sec(Duration::positive(1, 0), 0i32)]
    #[case::one_sec_and_one_nano(Duration::positive(1, 1), 1i32)]
    #[case::two_nano(Duration::positive(0, 2), 2i32)]
    #[case::one_micro(Duration::positive(0, 1_000), 1_000i32)]
    #[case::one_milli(Duration::positive(0, 1_000_000), 1_000_000i32)]
    #[case::some_value(Duration::positive(0, 123_456_789), 123_456_789i32)]
    #[case::max(Duration::MAX, 999_999_999i32)]
    #[case::min(Duration::MIN, -999_999_999i32)]
    fn test_fundu_duration_subsec_template(#[case] duration: Duration, #[case] expected: i32) {}

    #[apply(test_fundu_duration_subsec_template)]
    fn test_fundu_duration_subsec_nanos(duration: Duration, expected: i32) {
        assert_eq!(duration.subsec_nanos(), expected);
        assert_eq!(duration.neg().subsec_nanos(), expected.neg());
    }

    #[apply(test_fundu_duration_subsec_template)]
    fn test_fundu_duration_subsec_micros(duration: Duration, expected: i32) {
        let expected = expected / 1000i32;
        assert_eq!(duration.subsec_micros(), expected);
        assert_eq!(duration.neg().subsec_micros(), expected.neg());
    }

    #[apply(test_fundu_duration_subsec_template)]
    fn test_fundu_duration_subsec_millis(duration: Duration, expected: i32) {
        let expected = expected / 1_000_000i32;
        assert_eq!(duration.subsec_millis(), expected);
        assert_eq!(duration.neg().subsec_millis(), expected.neg());
    }

    #[rstest]
    #[case::zero(Duration::ZERO)]
    #[case::one(Duration::positive(1, 0))]
    #[case::one_plus_one_nano(Duration::positive(1, 1))]
    #[case::two(Duration::positive(2, 0))]
    #[case::two_plus_one_nano(Duration::positive(2, 1))]
    #[case::sixty(Duration::positive(60, 0))]
    #[case::sixty_plus_one(Duration::positive(61, 0))]
    #[case::sixty_minus_one(Duration::positive(59, 0))]
    #[case::some_value(Duration::positive(123_456_789, 987_654_321))]
    #[case::max(Duration::MAX)]
    #[case::min(Duration::MIN)]
    fn test_fundu_duration_extract_i64(
        #[case] mut duration: Duration,
        #[values(2, 3, 4, 5, 60, u64::MAX / 2, u64::MAX)] factor: u64,
    ) {
        let mut expected_number = i64::try_from(duration.inner.as_secs() / factor).unwrap();
        if duration.is_negative() {
            expected_number = expected_number.neg();
        }
        let expected_duration = Duration::from_std(
            duration.is_negative(),
            StdDuration::new(
                duration.inner.as_secs() % factor,
                duration.inner.subsec_nanos(),
            ),
        );

        let actual_number = duration.extract_i64(factor);
        assert_eq!(actual_number, expected_number);
        assert_eq!(duration, expected_duration);
    }

    #[rstest]
    #[case::one(Duration::positive(86400 * 7, 123), 1, Duration::positive(0, 123))]
    #[case::one_plus_sec(
        Duration::positive(86400 * 7 + 1, 123),
        1,
        Duration::positive(1, 123)
    )]
    #[case::one_minus_sec(
        Duration::positive(86400 * 7 - 1, 123),
        0,
        Duration::positive(86400 * 7 - 1, 123)
    )]
    fn test_fundu_duration_extract_weeks(
        #[case] duration: Duration,
        #[case] expected_number: i64,
        #[case] expected_duration: Duration,
    ) {
        let mut duration = duration;
        let number = duration.extract_weeks();
        assert_eq!(number, expected_number);
        assert_eq!(duration, expected_duration);
    }

    #[rstest]
    #[case::one(Duration::positive(86400, 123), 1, Duration::positive(0, 123))]
    #[case::one_plus_sec(
        Duration::positive(86400 + 1, 123),
        1,
        Duration::positive(1, 123)
    )]
    #[case::one_minus_sec(
        Duration::positive(86400 - 1, 123),
        0,
        Duration::positive(86400 - 1, 123)
    )]
    fn test_fundu_duration_extract_days(
        #[case] duration: Duration,
        #[case] expected_number: i64,
        #[case] expected_duration: Duration,
    ) {
        let mut duration = duration;
        let number = duration.extract_days();
        assert_eq!(number, expected_number);
        assert_eq!(duration, expected_duration);
    }

    #[rstest]
    #[case::one(Duration::positive(3600, 123), 1, Duration::positive(0, 123))]
    #[case::one_plus_sec(Duration::positive(3600 + 1, 123), 1, Duration::positive(1, 123))]
    #[case::one_minus_sec(Duration::positive(3600 - 1, 123), 0, Duration::positive(3599, 123))]
    fn test_fundu_duration_extract_hours(
        #[case] duration: Duration,
        #[case] expected_number: i64,
        #[case] expected_duration: Duration,
    ) {
        let mut duration = duration;
        let number = duration.extract_hours();
        assert_eq!(number, expected_number);
        assert_eq!(duration, expected_duration);
    }

    #[rstest]
    #[case::one(Duration::positive(60, 123), 1, Duration::positive(0, 123))]
    #[case::one_plus_sec(Duration::positive(60 + 1, 123), 1, Duration::positive(1, 123))]
    #[case::one_minus_sec(Duration::positive(60 - 1, 123), 0, Duration::positive(59, 123))]
    fn test_fundu_duration_extract_minutes(
        #[case] duration: Duration,
        #[case] expected_number: i64,
        #[case] expected_duration: Duration,
    ) {
        let mut duration = duration;
        let number = duration.extract_minutes();
        assert_eq!(number, expected_number);
        assert_eq!(duration, expected_duration);
    }

    #[rstest]
    #[case::zero(Duration::ZERO, 0, Duration::ZERO)]
    #[case::one(Duration::positive(1, 123), 1, Duration::positive(0, 123))]
    #[case::two(Duration::positive(2, 123), 2, Duration::positive(0, 123))]
    #[case::zero_sec(Duration::positive(0, 123), 0, Duration::positive(0, 123))]
    #[case::max(
        Duration::MAX,
        i128::from(u64::MAX),
        Duration::positive(0, 999_999_999)
    )]
    #[case::min(Duration::MIN, i128::from(u64::MAX).neg(), Duration::negative(0, 999_999_999))]
    fn test_fundu_duration_extract_seconds(
        #[case] duration: Duration,
        #[case] expected_number: i128,
        #[case] expected_duration: Duration,
    ) {
        let mut duration_copy = duration;
        let number = duration_copy.extract_seconds();
        assert_eq!(number, expected_number);
        assert_eq!(duration_copy, expected_duration);

        let mut duration = duration.neg();
        let number = duration.extract_seconds();
        assert_eq!(number, expected_number.neg());
        assert_eq!(duration, expected_duration.neg());
    }

    #[rstest]
    #[case::positive_zero(Duration::ZERO, Duration::ZERO, Ordering::Equal)]
    #[case::negative_zero(Duration::negative(0, 0), Duration::negative(0, 0), Ordering::Equal)]
    #[case::negative_zero_and_positive_zero(
        Duration::negative(0, 0),
        Duration::ZERO,
        Ordering::Equal
    )]
    #[case::both_positive_one_sec(
        Duration::positive(1, 0),
        Duration::positive(1, 0),
        Ordering::Equal
    )]
    #[case::both_negative_one_sec(
        Duration::negative(1, 0),
        Duration::negative(1, 0),
        Ordering::Equal
    )]
    #[case::negative_and_positive_one_sec(
        Duration::negative(1, 0),
        Duration::positive(1, 0),
        Ordering::Less
    )]
    #[case::one_nano_second_difference_positive(
        Duration::ZERO,
        Duration::positive(0, 1),
        Ordering::Less
    )]
    #[case::one_nano_second_difference_negative(
        Duration::ZERO,
        Duration::negative(0, 1),
        Ordering::Greater
    )]
    #[case::max(Duration::MAX, Duration::MAX, Ordering::Equal)]
    #[case::one_nano_below_max(
        Duration::MAX,
        Duration::positive(u64::MAX, 999_999_998),
        Ordering::Greater
    )]
    #[case::min(Duration::MIN, Duration::MIN, Ordering::Equal)]
    #[case::one_nano_above_min(
        Duration::MIN,
        Duration::negative(u64::MAX, 999_999_998),
        Ordering::Less
    )]
    #[case::min_max(Duration::MIN, Duration::MAX, Ordering::Less)]
    #[case::mixed_positive(
        Duration::positive(123, 987),
        Duration::positive(987, 123),
        Ordering::Less
    )]
    #[case::mixed_negative(
        Duration::negative(123, 987),
        Duration::negative(987, 123),
        Ordering::Greater
    )]
    #[case::mixed_positive_and_negative(
        Duration::positive(123, 987),
        Duration::negative(987, 123),
        Ordering::Greater
    )]
    fn test_duration_partial_and_total_ordering(
        #[case] lhs: Duration,
        #[case] rhs: Duration,
        #[case] expected: Ordering,
    ) {
        assert_eq!(lhs.partial_cmp(&rhs), Some(expected));
        assert_eq!(rhs.partial_cmp(&lhs), Some(expected.reverse()));
    }

    #[rstest]
    #[case::positive_zero(Duration::ZERO, Duration::ZERO)]
    #[case::negative_zero(Duration::negative(0, 0), Duration::negative(0, 0))]
    #[case::negative_zero_and_positive_zero(Duration::negative(0, 0), Duration::ZERO)]
    #[case::positive_one_sec(Duration::positive(1, 0), Duration::positive(1, 0))]
    #[case::negative_one_sec(Duration::negative(1, 0), Duration::negative(1, 0))]
    #[case::max(Duration::MAX, Duration::MAX)]
    #[case::min(Duration::MIN, Duration::MIN)]
    fn test_duration_hash_and_eq_property_when_equal(
        #[case] duration: Duration,
        #[case] other: Duration,
    ) {
        assert_eq!(duration, other);
        assert_eq!(other, duration);

        let mut hasher = DefaultHasher::new();
        duration.hash(&mut hasher);

        let mut other_hasher = DefaultHasher::new();
        other.hash(&mut other_hasher);

        assert_eq!(hasher.finish(), other_hasher.finish());
    }

    #[rstest]
    #[case::zero_and_positive_one(Duration::ZERO, Duration::positive(1, 0))]
    #[case::zero_and_negative_one(Duration::ZERO, Duration::negative(1, 0))]
    #[case::positive_sec(Duration::positive(1, 0), Duration::negative(2, 0))]
    #[case::positive_and_negative_one_sec(Duration::positive(1, 0), Duration::negative(1, 0))]
    #[case::min_and_max(Duration::MIN, Duration::MAX)]
    fn test_duration_hash_and_eq_property_when_not_equal(
        #[case] duration: Duration,
        #[case] other: Duration,
    ) {
        assert_ne!(duration, other);
        assert_ne!(other, duration);

        let mut hasher = DefaultHasher::new();
        duration.hash(&mut hasher);

        let mut other_hasher = DefaultHasher::new();
        other.hash(&mut other_hasher);

        assert_ne!(hasher.finish(), other_hasher.finish());
    }

    #[rstest]
    #[case::positive_zero(Duration::ZERO, StdDuration::ZERO)]
    #[case::negative_zero(Duration::negative(0, 0), StdDuration::ZERO)]
    #[case::positive_one(Duration::positive(1, 0), StdDuration::new(1, 0))]
    #[case::positive_one_nano(Duration::positive(0, 1), StdDuration::new(0, 1))]
    #[case::negative_one(Duration::negative(1, 0), StdDuration::ZERO)]
    #[case::negative_one_nano(Duration::negative(0, 1), StdDuration::ZERO)]
    #[case::max(Duration::MAX, StdDuration::MAX)]
    fn test_fundu_duration_saturating_into_std_duration(
        #[case] duration: Duration,
        #[case] expected: StdDuration,
    ) {
        assert_eq!(
            SaturatingInto::<std::time::Duration>::saturating_into(duration),
            expected
        );
    }

    #[cfg(feature = "time")]
    #[rstest]
    #[case::positive_zero(Duration::ZERO, time::Duration::ZERO)]
    #[case::negative_zero(Duration::negative(0, 0), time::Duration::ZERO)]
    #[case::positive_one(Duration::positive(1, 0), time::Duration::new(1, 0))]
    #[case::negative_one(Duration::negative(1, 0), time::Duration::new(-1, 0))]
    #[case::negative_barely_no_overflow(
        Duration::negative(i64::MIN.unsigned_abs(), 999_999_999),
        time::Duration::MIN
    )]
    #[case::negative_barely_overflow(
        Duration::negative(i64::MIN.unsigned_abs() + 1, 0),
        time::Duration::MIN
    )]
    #[case::negative_max_overflow(Duration::negative(u64::MAX, 999_999_999), time::Duration::MIN)]
    #[case::positive_barely_no_overflow(
        Duration::positive(i64::MAX as u64, 999_999_999),
        time::Duration::MAX
    )]
    #[case::positive_barely_overflow(
        Duration::positive(i64::MAX as u64 + 1, 999_999_999),
        time::Duration::MAX
    )]
    #[case::positive_max_overflow(Duration::positive(u64::MAX, 999_999_999), time::Duration::MAX)]
    fn test_fundu_duration_saturating_into_time_duration(
        #[case] duration: Duration,
        #[case] expected: time::Duration,
    ) {
        assert_eq!(
            SaturatingInto::<time::Duration>::saturating_into(duration),
            expected
        );
    }

    #[cfg(feature = "chrono")]
    #[rstest]
    #[case::positive_zero(Duration::ZERO, chrono::Duration::zero())]
    #[case::negative_zero(Duration::negative(0, 0), chrono::Duration::zero())]
    #[case::positive_one(Duration::positive(1, 0), chrono::Duration::seconds(1))]
    #[case::negative_one(Duration::negative(1, 0), chrono::Duration::seconds(-1))]
    #[case::negative_barely_no_overflow(
        Duration::negative(i64::MIN.unsigned_abs() / 1000, 808_000_000),
        chrono::Duration::min_value()
    )]
    #[case::negative_barely_overflow(
        Duration::negative(i64::MIN.unsigned_abs() / 1000 + 1, 0),
        chrono::Duration::min_value()
    )]
    #[case::negative_max_overflow(
        Duration::negative(u64::MAX, 999_999_999),
        chrono::Duration::min_value()
    )]
    #[case::positive_barely_no_overflow(
        Duration::positive(i64::MAX as u64 / 1000, 807_000_000),
        chrono::Duration::max_value()
    )]
    #[case::positive_barely_overflow(
        Duration::positive(i64::MAX as u64 / 1000 + 1, 807_000_000),
        chrono::Duration::max_value()
    )]
    #[case::positive_max_overflow(
        Duration::positive(u64::MAX, 999_999_999),
        chrono::Duration::max_value()
    )]
    fn test_fundu_duration_saturating_into_chrono_duration(
        #[case] duration: Duration,
        #[case] expected: chrono::Duration,
    ) {
        assert_eq!(
            SaturatingInto::<chrono::Duration>::saturating_into(duration),
            expected
        );
    }

    #[rstest]
    #[case::negative_one(Duration::negative(1, 0))]
    #[case::negative_one_nano(Duration::negative(0, 1))]
    #[case::one_one(Duration::negative(1, 1))]
    #[case::min(Duration::MIN)]
    fn test_std_duration_try_from_for_fundu_duration_then_error(#[case] duration: Duration) {
        assert_eq!(
            TryInto::<std::time::Duration>::try_into(duration).unwrap_err(),
            TryFromDurationError::NegativeDuration
        );
    }

    #[cfg(feature = "chrono")]
    #[rstest]
    #[case::zero(Duration::ZERO, ChronoDuration::zero())]
    #[case::negative_zero(Duration::negative(0, 0), ChronoDuration::zero())]
    #[case::positive_one_sec(Duration::positive(1, 0), ChronoDuration::seconds(1))]
    #[case::positive_one_sec_and_nano(
        Duration::positive(1, 1),
        ChronoDuration::nanoseconds(1_000_000_001)
    )]
    #[case::negative_one_sec(Duration::negative(1, 0), ChronoDuration::seconds(-1))]
    #[case::negative_one_sec_and_nano(
        Duration::negative(1, 1),
        ChronoDuration::nanoseconds(-1_000_000_001)
    )]
    #[case::max_nanos(
        Duration::positive(0, 999_999_999),
        ChronoDuration::nanoseconds(999_999_999)
    )]
    #[case::min_nanos(
        Duration::negative(0, 999_999_999),
        ChronoDuration::nanoseconds(-999_999_999)
    )]
    #[case::max_secs(
        Duration::positive(i64::MAX as u64 / 1000, 0),
        ChronoDuration::seconds(i64::MAX / 1000))
    ]
    #[case::max_secs_and_nanos(
        Duration::positive(i64::MAX as u64 / 1000, 807_000_000),
        ChronoDuration::max_value())
    ]
    #[case::secs_and_nanos_one_below_max(
        Duration::positive(i64::MAX as u64 / 1000, 807_000_000 - 1),
        ChronoDuration::max_value().checked_sub(&ChronoDuration::nanoseconds(1)).unwrap())
    ]
    #[case::min_secs(
        Duration::negative(i64::MIN.unsigned_abs() / 1000, 0),
        ChronoDuration::seconds(i64::MIN / 1000))
    ]
    #[case::min_secs_and_nanos(
        Duration::negative(i64::MIN.unsigned_abs() / 1000, 807_000_000),
        ChronoDuration::min_value())
    ]
    #[case::secs_and_nanos_one_above_min(
        Duration::negative(i64::MIN.unsigned_abs() / 1000, 807_000_000 - 1),
        ChronoDuration::min_value().checked_add(&ChronoDuration::nanoseconds(1)).unwrap())
    ]
    fn test_chrono_duration_try_from_fundu_duration(
        #[case] duration: Duration,
        #[case] expected: ChronoDuration,
    ) {
        let chrono_duration: ChronoDuration = duration.try_into().unwrap();
        assert_eq!(chrono_duration, expected);
    }

    #[cfg(feature = "chrono")]
    #[rstest]
    #[case::positive_overflow_secs(
        Duration::positive(i64::MAX as u64 / 1000 + 1, 0),
        TryFromDurationError::PositiveOverflow)
    ]
    #[case::positive_overflow_secs_and_nanos(
        Duration::positive(i64::MAX as u64 / 1000, 807_000_000 + 1),
        TryFromDurationError::PositiveOverflow)
    ]
    #[case::positive_overflow_max_fundu_duration(
        Duration::MAX,
        TryFromDurationError::PositiveOverflow
    )]
    #[case::negative_overflow_secs(
        Duration::negative(i64::MIN.unsigned_abs() / 1000 + 1, 0),
        TryFromDurationError::NegativeOverflow)
    ]
    #[case::negative_overflow_secs_and_nanos(
        Duration::negative(i64::MIN.unsigned_abs() / 1000, 808_000_001),
        TryFromDurationError::NegativeOverflow)
    ]
    #[case::negative_overflow_min_fundu_duration(
        Duration::MIN,
        TryFromDurationError::NegativeOverflow
    )]
    fn test_chrono_duration_try_from_fundu_duration_then_error(
        #[case] duration: Duration,
        #[case] expected: TryFromDurationError,
    ) {
        let result: Result<ChronoDuration, TryFromDurationError> = duration.try_into();
        assert_eq!(result.unwrap_err(), expected);
    }

    #[cfg(feature = "time")]
    #[test]
    fn test_time_duration_try_from_fundu_duration() {
        let duration = Duration::from_std(false, std::time::Duration::new(1, 0));
        let time_duration: time::Duration = duration.try_into().unwrap();
        assert_eq!(time_duration, time::Duration::new(1, 0));
    }

    #[rstest]
    #[case::zero(
        std::time::Duration::ZERO,
        Duration::from_std(false, std::time::Duration::ZERO)
    )]
    #[case::one(
        std::time::Duration::new(1, 0),
        Duration::from_std(false, std::time::Duration::new(1, 0))
    )]
    #[case::with_nano_seconds(
        std::time::Duration::new(1, 123_456_789),
        Duration::from_std(false, std::time::Duration::new(1, 123_456_789))
    )]
    #[case::max(
        std::time::Duration::MAX,
        Duration::from_std(false, std::time::Duration::MAX)
    )]
    fn test_fundu_duration_from_std_time_duration(
        #[case] std_duration: std::time::Duration,
        #[case] expected: Duration,
    ) {
        assert_eq!(Duration::from(std_duration), expected);
    }

    #[cfg(feature = "time")]
    #[rstest]
    #[case::zero(time::Duration::ZERO, Duration::ZERO)]
    #[case::positive_one(time::Duration::new(1, 0), Duration::positive(1, 0))]
    #[case::positive_one_nano(time::Duration::new(0, 1), Duration::positive(0, 1))]
    #[case::negative_one(time::Duration::new(-1, 0), Duration::negative(1, 0))]
    #[case::negative_one_nano(time::Duration::new(0, -1), Duration::negative(0, 1))]
    #[case::positive_one_negative_one_nano(
        time::Duration::new(1, -1),
        Duration::positive(0, 999_999_999)
    )]
    #[case::negative_one_positive_one_nano(
        time::Duration::new(-1, 1),
        Duration::negative(0, 999_999_999)
    )]
    #[case::min(time::Duration::MIN, Duration::negative(i64::MIN.unsigned_abs(), 999_999_999))]
    #[case::max(time::Duration::MAX, Duration::positive(i64::MAX as u64, 999_999_999))]
    fn test_fundu_duration_from_time_duration(
        #[case] time_duration: time::Duration,
        #[case] expected: Duration,
    ) {
        assert_eq!(Duration::from(time_duration), expected);
    }

    #[cfg(feature = "chrono")]
    #[rstest]
    #[case::zero(chrono::Duration::zero(), Duration::ZERO)]
    #[case::positive_one(chrono::Duration::seconds(1), Duration::positive(1, 0))]
    #[case::positive_one_nano(chrono::Duration::nanoseconds(1), Duration::positive(0, 1))]
    #[case::negative_one(chrono::Duration::seconds(-1), Duration::negative(1, 0))]
    #[case::negative_one_nano(chrono::Duration::nanoseconds(-1), Duration::negative(0, 1))]
    #[case::min(chrono::Duration::min_value(), CHRONO_MIN_DURATION)]
    #[case::max(chrono::Duration::max_value(), CHRONO_MAX_DURATION)]
    fn test_fundu_duration_from_chrono_duration(
        #[case] chrono_duration: chrono::Duration,
        #[case] expected: Duration,
    ) {
        assert_eq!(Duration::from(chrono_duration), expected);
    }
}
