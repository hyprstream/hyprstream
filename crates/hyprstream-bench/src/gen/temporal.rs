//! Temporal reasoning — **held-out gate family** (firewalled from training
//! synthesis; measured by the P1.4 leg-(e) zero-shot transfer gate). Date
//! arithmetic uses Hinnant's days-from-civil algorithm implemented in-crate so
//! truth stays mechanical and dependency-free.

use hyprstream_decision::Entry;

use crate::family::{Family, Stratum};
use crate::gen::stream;
use crate::item::Item;

const FAMILY: Family = Family::Temporal;

/// Days since 1970-01-01 for a proleptic-Gregorian (y, m, d).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = (m + 9).rem_euclid(12);
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

struct Scenario {
    text: String,
    start_days: i64,
    end_days: i64,
}

fn scenario(seed: u64) -> Scenario {
    let mut rng = stream(seed);
    let (y0, m0, d0) = (rng.range(2019, 2031), rng.range(1, 12), rng.range(1, 28));
    let span = rng.range(1, 400);
    let start = days_from_civil(y0, m0, d0);
    let end = start + span;
    let (y1, m1, d1) = civil_from_days(end);
    Scenario {
        text: format!(
            "Deployment A started on {y0:04}-{m0:02}-{d0:02}. Deployment B shipped on {y1:04}-{m1:02}-{d1:02}."
        ),
        start_days: start,
        end_days: end,
    }
}

pub fn noul(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xA0);
    let s = scenario(seed);
    let gap = s.end_days - s.start_days;
    let (claim, truth) = match stratum {
        Stratum::NearMiss => {
            let slip = gap + [1, -1, 7][rng.below(3) as usize];
            if rng.below(2) == 0 {
                (slip, false)
            } else {
                (gap, true)
            }
        }
        _ => {
            let offset = rng.range(-5, 5);
            (gap + offset, offset == 0)
        }
    };
    Item::noul(
        FAMILY,
        stratum,
        seed,
        Entry::Str(s.text),
        format!("Did exactly {claim} days elapse between the two deployments?"),
        truth,
    )
}

pub fn choice(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xC0);
    let s = scenario(seed);
    let gap = s.end_days - s.start_days;
    let mut offsets: Vec<i64> = match stratum {
        Stratum::NearMiss => vec![1, -1, 7, -7],
        _ => vec![rng.range(2, 10), -rng.range(2, 10), rng.range(11, 60)],
    };
    offsets.retain(|o| gap + o > 0 && *o != 0);
    offsets.dedup();
    let mut values: Vec<i64> = offsets.iter().map(|o| gap + o).collect();
    let correct = rng.below(values.len() as u64 + 1) as usize;
    values.insert(correct, gap);
    Item::choice(
        FAMILY,
        stratum,
        seed,
        Entry::Str(s.text),
        "How many days elapsed between deployment A and deployment B?",
        values.iter().map(|v| format!("{v} days")).collect(),
        correct,
    )
}

pub fn score(seed: u64, stratum: Stratum) -> Item {
    let s = scenario(seed);
    let gap = s.end_days - s.start_days;
    Item::score(
        FAMILY,
        stratum,
        seed,
        Entry::Str(s.text),
        "How long was the gap between the deployments? Levels: 0 = under a week, 1 = 1–4 weeks, 2 = 1–6 months, 3 = over 6 months.",
        4,
        match gap {
            g if g < 7 => 0,
            g if g < 28 => 1,
            g if g < 183 => 2,
            _ => 3,
        },
    )
}
