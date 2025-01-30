// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

use fundu_core::time::TimeUnitsLike;

use crate::TimeUnit::*;
use crate::{
    Multiplier, TimeUnit, DEFAULT_ID_DAY, DEFAULT_ID_HOUR, DEFAULT_ID_MICRO_SECOND,
    DEFAULT_ID_MILLI_SECOND, DEFAULT_ID_MINUTE, DEFAULT_ID_MONTH, DEFAULT_ID_NANO_SECOND,
    DEFAULT_ID_SECOND, DEFAULT_ID_WEEK, DEFAULT_ID_YEAR,
};

const DEFAULT_TIME_UNITS: [&str; 10] = [
    DEFAULT_ID_NANO_SECOND,
    DEFAULT_ID_MICRO_SECOND,
    DEFAULT_ID_MILLI_SECOND,
    DEFAULT_ID_SECOND,
    DEFAULT_ID_MINUTE,
    DEFAULT_ID_HOUR,
    DEFAULT_ID_DAY,
    DEFAULT_ID_WEEK,
    DEFAULT_ID_MONTH,
    DEFAULT_ID_YEAR,
];

/// Interface for [`TimeUnit`]s providing common methods to manipulate the available time units.
#[derive(Debug, PartialEq, Eq, Clone)]
pub(super) struct TimeUnits {
    data: [Option<TimeUnit>; 10],
}

impl Default for TimeUnits {
    fn default() -> Self {
        Self::with_default_time_units()
    }
}

impl TimeUnitsLike for TimeUnits {
    /// Return `true` if this set of time units is empty.
    #[inline]
    fn is_empty(&self) -> bool {
        self.data.iter().all(std::option::Option::is_none)
    }

    /// Return the [`TimeUnit`] associated with the provided `identifier`.
    ///
    /// Returns `None` if no [`TimeUnit`] with the provided `identifier` is present in the current
    /// set of time units.
    #[inline]
    fn get(&self, identifier: &str) -> Option<(TimeUnit, Multiplier)> {
        match identifier.len() {
            1 => self.data.iter().skip(3).filter_map(|t| *t).find_map(|t| {
                let unit = DEFAULT_TIME_UNITS[t as usize];
                (unit == identifier).then(|| (t, Multiplier::default()))
            }),
            2 => self.data.iter().take(3).filter_map(|t| *t).find_map(|t| {
                let unit = DEFAULT_TIME_UNITS[t as usize];
                (unit == identifier).then(|| (t, Multiplier::default()))
            }),
            _ => None,
        }
    }
}

impl TimeUnits {
    /// Create an empty set of [`TimeUnit`]s.
    pub(super) const fn new() -> Self {
        Self { data: [None; 10] }
    }

    /// Create [`TimeUnits`] with a custom set of [`TimeUnit`]s.
    pub(super) const fn with_time_units(units: &[TimeUnit]) -> Self {
        let mut data: [Option<TimeUnit>; 10] = [None; 10];
        let mut counter = 0;
        while counter < units.len() {
            let unit = units[counter];
            data[unit as usize] = Some(unit);
            counter += 1;
        }
        Self { data }
    }

    /// Create [`TimeUnits`] with default [`TimeUnit`]s.
    pub(super) const fn with_default_time_units() -> Self {
        Self {
            data: [
                Some(NanoSecond),
                Some(MicroSecond),
                Some(MilliSecond),
                Some(Second),
                Some(Minute),
                Some(Hour),
                Some(Day),
                Some(Week),
                None,
                None,
            ],
        }
    }

    /// Create [`TimeUnits`] with a all available [`TimeUnit`]s.
    pub(super) const fn with_all_time_units() -> Self {
        Self {
            data: [
                Some(NanoSecond),
                Some(MicroSecond),
                Some(MilliSecond),
                Some(Second),
                Some(Minute),
                Some(Hour),
                Some(Day),
                Some(Week),
                Some(Month),
                Some(Year),
            ],
        }
    }

    /// Return all [`TimeUnit`]s from the set of active time units ordered.
    pub(super) fn get_time_units(&self) -> Vec<TimeUnit> {
        self.data.iter().filter_map(|&p| p).collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn test_time_units_new() {
        let time_units = TimeUnits::new();
        assert!(time_units.data.iter().all(std::option::Option::is_none));
        assert!(time_units.is_empty());
        assert_eq!(time_units.get_time_units(), vec![]);
    }

    #[test]
    fn test_time_units_with_default_time_units() {
        let time_units = TimeUnits::with_default_time_units();
        assert!(!time_units.is_empty());
        assert_eq!(time_units, TimeUnits::default());
        assert_eq!(
            time_units.get_time_units(),
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
    fn test_time_units_with_all_time_units() {
        let time_units = TimeUnits::with_all_time_units();
        assert!(!time_units.is_empty());
        assert_eq!(
            time_units.get_time_units(),
            vec![
                NanoSecond,
                MicroSecond,
                MilliSecond,
                Second,
                Minute,
                Hour,
                Day,
                Week,
                Month,
                Year
            ]
        );
    }

    #[rstest]
    fn test_time_units_with_time_units_when_single_unit(
        #[values(
            NanoSecond,
            MicroSecond,
            MilliSecond,
            Second,
            Minute,
            Hour,
            Day,
            Week,
            Month,
            Year
        )]
        time_unit: TimeUnit,
    ) {
        let mut expected_data: [Option<TimeUnit>; 10] = [None; 10];
        expected_data[time_unit as usize] = Some(time_unit);

        let time_units = TimeUnits::with_time_units(&[time_unit]);
        assert!(!time_units.is_empty());
        assert_eq!(time_units.data, expected_data);
    }

    #[test]
    fn test_time_units_with_time_units_when_all_time_units() {
        let expected_data: [Option<TimeUnit>; 10] = [
            Some(NanoSecond),
            Some(MicroSecond),
            Some(MilliSecond),
            Some(Second),
            Some(Minute),
            Some(Hour),
            Some(Day),
            Some(Week),
            Some(Month),
            Some(Year),
        ];
        let time_units = TimeUnits::with_time_units(&[
            NanoSecond,
            MicroSecond,
            MilliSecond,
            Second,
            Minute,
            Hour,
            Day,
            Week,
            Month,
            Year,
        ]);
        assert!(!time_units.is_empty());
        assert_eq!(time_units.data, expected_data);
    }

    #[test]
    fn test_time_units_when_empty_then_return_true() {
        assert!(TimeUnits::new().is_empty());
    }

    #[rstest]
    fn test_time_units_is_empty_when_not_empty_then_return_false(
        #[values(
            NanoSecond,
            MicroSecond,
            MilliSecond,
            Second,
            Minute,
            Hour,
            Day,
            Week,
            Month,
            Year
        )]
        time_unit: TimeUnit,
    ) {
        let time_units = TimeUnits::with_time_units(&[time_unit]);
        assert!(!time_units.is_empty());
    }

    #[rstest]
    #[case::nano_second("ns", Some((NanoSecond, Multiplier(1,0))))]
    #[case::micro_second("Ms", Some((MicroSecond, Multiplier(1,0))))]
    #[case::milli_second("ms", Some((MilliSecond, Multiplier(1,0))))]
    #[case::second("s", Some((Second, Multiplier(1,0))))]
    #[case::minute("m", Some((Minute, Multiplier(1,0))))]
    #[case::hour("h", Some((Hour, Multiplier(1,0))))]
    #[case::day("d", Some((Day, Multiplier(1,0))))]
    #[case::week("w", Some((Week, Multiplier(1,0))))]
    #[case::month("M", Some((Month, Multiplier(1,0))))]
    #[case::year("y", Some((Year, Multiplier(1,0))))]
    fn test_time_units_get(#[case] id: &str, #[case] expected: Option<(TimeUnit, Multiplier)>) {
        assert_eq!(TimeUnits::with_all_time_units().get(id), expected);
        assert_eq!(TimeUnits::new().get(id), None);
    }

    #[test]
    fn test_time_units_get_time_units() {
        let time_units = TimeUnits::with_all_time_units();
        assert_eq!(
            time_units.get_time_units(),
            vec![
                NanoSecond,
                MicroSecond,
                MilliSecond,
                Second,
                Minute,
                Hour,
                Day,
                Week,
                Month,
                Year
            ]
        );
    }
}
