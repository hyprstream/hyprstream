//! Arithmetic word problems. Truth is computed in `i64`; distractors and
//! near-miss claims are the classic slips (±1, off-by-ten, transposed digits,
//! add/subtract confusion).

use hyprstream_decision::Entry;

use crate::family::{Family, Stratum};
use crate::gen::stream;
use crate::item::Item;

const FAMILY: Family = Family::Arith;

struct Problem {
    text: String,
    answer: i64,
}

fn problem(seed: u64) -> Problem {
    let mut rng = stream(seed);
    let subject = ["apples", "bolts", "tickets", "bricks", "litres"][rng.below(5) as usize];
    let start = rng.range(3, 97);
    let added = rng.range(2, 88);
    let removed = rng.range(1, added + start - 1);
    let answer = start + added - removed;
    let text = format!(
        "A crate holds {start} {subject}. {added} more are delivered, then {removed} are shipped out."
    );
    Problem { text, answer }
}

fn transpose_slip(value: i64) -> i64 {
    let text = value.to_string();
    // Load-bearing edge case: a palindrome-ish value (e.g. 22) transposes to
    // itself, so this can return the answer unchanged. Both call sites rely
    // on that: noul checks `slip == answer`, choice's `retain` drops it.
    if text.len() >= 2 {
        let mut chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        chars.swap(n - 2, n - 1);
        chars.into_iter().collect::<String>().parse().unwrap_or(value)
    } else {
        value + 10
    }
}

pub fn noul(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xA0);
    let problem = problem(seed);
    let (claim, truth) = match stratum {
        Stratum::NearMiss => {
            // Near-miss arithmetic: the claim is one slip away from correct
            // half the time, so the stratum mixes true and false at the
            // hardest margin.
            let slip = [problem.answer + 1, problem.answer - 1, transpose_slip(problem.answer)]
                [rng.below(3) as usize];
            if rng.below(2) == 0 {
                (slip, slip == problem.answer)
            } else {
                (problem.answer, true)
            }
        }
        _ => {
            let offset = rng.range(-3, 3);
            (problem.answer + offset, offset == 0)
        }
    };
    Item::noul(
        FAMILY,
        stratum,
        seed,
        Entry::Str(problem.text),
        format!("Is the final count exactly {claim}?"),
        truth,
    )
}

pub fn choice(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xC0);
    let problem = problem(seed);
    let mut distractors: Vec<i64> = match stratum {
        Stratum::NearMiss => vec![
            problem.answer + 1,
            problem.answer - 1,
            transpose_slip(problem.answer),
            problem.answer + 10,
        ],
        _ => vec![
            problem.answer + rng.range(2, 5),
            problem.answer - rng.range(2, 5),
            problem.answer + rng.range(6, 12),
            problem.answer + rng.range(13, 25),
        ],
    };
    distractors.retain(|d| *d != problem.answer && *d >= 0);
    distractors.dedup();
    let count = 3 + rng.below(2) as usize; // 3–4 options
    distractors.truncate(count - 1);
    let mut values = distractors;
    let correct = rng.below(values.len() as u64 + 1) as usize;
    values.insert(correct, problem.answer);
    Item::choice(
        FAMILY,
        stratum,
        seed,
        Entry::Str(problem.text),
        "What is the final count?",
        values.iter().map(ToString::to_string).collect(),
        correct,
    )
}

/// Magnitude buckets: 0–9, 10–49, 50–99, 100+.
fn bucket(value: i64) -> usize {
    match value {
        v if v < 10 => 0,
        v if v < 50 => 1,
        v if v < 100 => 2,
        _ => 3,
    }
}

pub fn score(seed: u64, stratum: Stratum) -> Item {
    let problem = problem(seed);
    Item::score(
        FAMILY,
        stratum,
        seed,
        Entry::Str(problem.text),
        "How large is the final count? Levels: 0 = under 10, 1 = 10–49, 2 = 50–99, 3 = 100 or more.",
        4,
        bucket(problem.answer),
    )
}
