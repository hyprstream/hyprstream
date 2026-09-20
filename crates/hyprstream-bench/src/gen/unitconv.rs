//! Unit conversion (km↔mi, kg↔lb, °C↔°F). Truth is computed with pinned
//! conversion constants and rounded to integers so outcomes are exact.

use hyprstream_decision::Entry;

use crate::family::{Family, Stratum};
use crate::gen::stream;
use crate::item::Item;

const FAMILY: Family = Family::UnitConv;

struct Conversion {
    text: String,
    answer: i64,
    unit: &'static str,
}

fn conversion(seed: u64) -> Conversion {
    let mut rng = stream(seed);
    match rng.below(3) {
        0 => {
            let km = rng.range(1, 500);
            Conversion {
                text: format!("A route is {km} km long."),
                answer: (km as f64 * 0.621_371).round() as i64,
                unit: "mi",
            }
        }
        1 => {
            let kg = rng.range(1, 300);
            Conversion {
                text: format!("A crate weighs {kg} kg."),
                answer: (kg as f64 * 2.204_62).round() as i64,
                unit: "lb",
            }
        }
        _ => {
            let celsius = rng.range(-40, 45);
            Conversion {
                text: format!("The sensor reads {celsius} °C."),
                answer: (celsius as f64 * 9.0 / 5.0 + 32.0).round() as i64,
                unit: "°F",
            }
        }
    }
}

pub fn noul(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xA0);
    let conv = conversion(seed);
    let (claim, truth) = match stratum {
        Stratum::NearMiss => {
            let slip = conv.answer + [1, -1, 2][rng.below(3) as usize];
            if rng.below(2) == 0 {
                (slip, false)
            } else {
                (conv.answer, true)
            }
        }
        _ => {
            let offset = rng.range(-4, 4);
            (conv.answer + offset, offset == 0)
        }
    };
    Item::noul(
        FAMILY,
        stratum,
        seed,
        Entry::Str(conv.text),
        format!("Is that {claim} {} (rounded to the nearest integer)?", conv.unit),
        truth,
    )
}

pub fn choice(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xC0);
    let conv = conversion(seed);
    let spread: Vec<i64> = match stratum {
        // Close distractors: ±1 and ±2 — answers within rounding distance.
        Stratum::NearMiss => vec![1, -1, 2],
        _ => vec![rng.range(3, 8), -rng.range(3, 8), rng.range(9, 20)],
    };
    let mut values: Vec<i64> = spread.iter().map(|d| conv.answer + d).collect();
    values.retain(|v| *v != conv.answer);
    values.dedup();
    let correct = rng.below(values.len() as u64 + 1) as usize;
    values.insert(correct, conv.answer);
    Item::choice(
        FAMILY,
        stratum,
        seed,
        Entry::Str(conv.text),
        format!(
            "What is the converted value in {} (rounded to the nearest integer)?",
            conv.unit
        ),
        values.iter().map(|v| format!("{v} {}", conv.unit)).collect(),
        correct,
    )
}

pub fn score(seed: u64, stratum: Stratum) -> Item {
    let conv = conversion(seed);
    Item::score(
        FAMILY,
        stratum,
        seed,
        Entry::Str(conv.text),
        format!(
            "How large is the converted value in {}? Levels: 0 = negative, 1 = 0–99, 2 = 100–499, 3 = 500 or more.",
            conv.unit
        ),
        4,
        match conv.answer {
            v if v < 0 => 0,
            v if v < 100 => 1,
            v if v < 500 => 2,
            _ => 3,
        },
    )
}
