//! Independent truth-checker: re-decides every frozen item from its rendered
//! text alone — a second, structurally different implementation from the
//! generators (text parsing + independent date math + repair-by-application
//! instead of threaded provenance) — and asserts agreement with `item.truth`.
//!
//! Added after the r1 review blocker: positional text extraction in the
//! syllogism generator mislabeled ~7.4% of vob-1.0 items, and the release
//! suite (determinism / closure / mixing) could not see it. This suite fails
//! on any truth-label divergence, per family, across the full frozen release.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::print_stderr)]

use std::collections::HashMap;

use hyprstream_bench::family::Family;
use hyprstream_bench::gen::{generate_all, BenchConfig};
use hyprstream_bench::Item;
use hyprstream_decision::QuestionKind;

fn instructions(item: &Item) -> String {
    item.question
        .instructions
        .as_ref()
        .map(hyprstream_decision::Entry::canonical_text)
        .unwrap_or_default()
}

fn state(item: &Item) -> String {
    item.state.canonical_text()
}

// --- arith: "A crate holds S X. A more are delivered, then R are shipped out." ---

fn arith_answer(text: &str) -> i64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    let s: i64 = words[3].parse().unwrap();
    let a: i64 = words[5].parse().unwrap();
    let r: i64 = words[10].parse().unwrap();
    s + a - r
}

fn check_arith(item: &Item) -> usize {
    let answer = arith_answer(&state(item));
    match item.question.kind {
        QuestionKind::Noul => {
            let claim: i64 = instructions(item)
                .trim_end_matches('?')
                .rsplit(' ')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            usize::from(claim == answer)
        }
        QuestionKind::Choice => {
            let labels = item.question.labels();
            labels
                .iter()
                .position(|l| l.parse::<i64>().unwrap() == answer)
                .unwrap()
        }
        QuestionKind::Score => match answer {
            v if v < 10 => 0,
            v if v < 50 => 1,
            v if v < 100 => 2,
            _ => 3,
        },
        _ => unreachable!(),
    }
}

// --- unitconv: "A route is {km} km long." / "A crate weighs {kg} kg." / "The sensor reads {c} °C." ---

fn unitconv_answer(text: &str) -> (i64, &'static str) {
    let words: Vec<&str> = text.split_whitespace().collect();
    if text.contains(" km ") {
        let km: f64 = words[3].parse().unwrap();
        ((km * 0.621_371).round() as i64, "mi")
    } else if text.contains(" kg") {
        let kg: f64 = words[3].parse().unwrap();
        ((kg * 2.204_62).round() as i64, "lb")
    } else {
        let celsius: f64 = words[3].parse().unwrap();
        ((celsius * 9.0 / 5.0 + 32.0).round() as i64, "°F")
    }
}

fn check_unitconv(item: &Item) -> usize {
    let (answer, unit) = unitconv_answer(&state(item));
    match item.question.kind {
        QuestionKind::Noul => {
            // "Is that {claim} {unit} (rounded to the nearest integer)?"
            let text = instructions(item);
            let claim: i64 = text
                .strip_prefix("Is that ")
                .unwrap()
                .split(' ')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            assert!(text.contains(unit));
            usize::from(claim == answer)
        }
        QuestionKind::Choice => {
            let labels = item.question.labels();
            labels
                .iter()
                .position(|l| {
                    l.split(' ').next().unwrap().parse::<i64>().unwrap() == answer
                        && l.ends_with(unit)
                })
                .unwrap()
        }
        QuestionKind::Score => match answer {
            v if v < 0 => 0,
            v if v < 100 => 1,
            v if v < 500 => 2,
            _ => 3,
        },
        _ => unreachable!(),
    }
}

// --- temporal: dates re-decoded with an independent month-table walk (not
// the generator's days-from-civil formula). ---

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) => 29,
        2 => 28,
        _ => unreachable!(),
    }
}

/// Days from (y1,m1,d1) to (y2,m2,d2), by walking months. Independent of
/// Hinnant's algorithm used by the generator.
fn days_between(a: (i64, i64, i64), b: (i64, i64, i64)) -> i64 {
    assert!(a <= b, "test items always order A before B");
    let (mut y, mut m, mut d) = a;
    let mut days = 0i64;
    while (y, m, d) != b {
        d += 1;
        days += 1;
        if d > days_in_month(y, m) {
            d = 1;
            m += 1;
            if m > 12 {
                m = 1;
                y += 1;
            }
        }
    }
    days
}

fn parse_date(token: &str) -> (i64, i64, i64) {
    let parts: Vec<i64> = token
        .trim_end_matches('.')
        .split('-')
        .map(|p| p.parse().unwrap())
        .collect();
    (parts[0], parts[1], parts[2])
}

fn temporal_gap(text: &str) -> i64 {
    let words: Vec<&str> = text.split_whitespace().collect();
    // "Deployment A started on YYYY-MM-DD. Deployment B shipped on YYYY-MM-DD."
    let start = parse_date(words[4]);
    let end = parse_date(words[9]);
    days_between(start, end)
}

fn check_temporal(item: &Item) -> usize {
    let gap = temporal_gap(&state(item));
    match item.question.kind {
        QuestionKind::Noul => {
            // "Did exactly {claim} days elapse between the two deployments?"
            let claim: i64 = instructions(item)
                .strip_prefix("Did exactly ")
                .unwrap()
                .split(' ')
                .next()
                .unwrap()
                .parse()
                .unwrap();
            usize::from(claim == gap)
        }
        QuestionKind::Choice => {
            let labels = item.question.labels();
            labels
                .iter()
                .position(|l| {
                    l.strip_suffix(" days").unwrap().parse::<i64>().unwrap() == gap
                })
                .unwrap()
        }
        QuestionKind::Score => match gap {
            g if g < 7 => 0,
            g if g < 28 => 1,
            g if g < 183 => 2,
            _ => 3,
        },
        _ => unreachable!(),
    }
}

// --- jsoncheck: validity re-decided by serde_json; the correct repair is
// found by *applying* each candidate repair and re-parsing. ---

fn is_valid_json(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text).is_ok()
}

/// Apply repair option `which` (generator's repair-menu order) to the text.
fn apply_repair(text: &str, which: usize) -> String {
    match which {
        // remove the trailing comma
        0 => text.replace(",}", "}"),
        // quote the bare object key: `{ident:` → `{"ident":`
        1 => {
            let mut out = String::with_capacity(text.len() + 4);
            let chars: Vec<char> = text.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if (chars[i] == '{' || chars[i] == ',')
                    && i + 1 < chars.len()
                    && chars[i + 1].is_ascii_alphabetic()
                {
                    out.push(chars[i]);
                    out.push('"');
                    i += 1;
                    while i < chars.len() && chars[i] != ':' {
                        out.push(chars[i]);
                        i += 1;
                    }
                    out.push('"');
                } else {
                    out.push(chars[i]);
                    i += 1;
                }
            }
            out
        }
        // replace single quotes with double quotes
        2 => text.replace('\'', "\""),
        // add the missing closing brace
        3 => format!("{text}}}"),
        // already valid
        _ => text.to_owned(),
    }
}

fn check_jsoncheck(item: &Item) -> usize {
    let text = state(item);
    match item.question.kind {
        QuestionKind::Noul => usize::from(is_valid_json(&text)),
        QuestionKind::Score => usize::from(!is_valid_json(&text)),
        QuestionKind::Choice => {
            let labels = item.question.labels();
            let menu = [
                "remove the trailing comma",
                "quote the bare object key",
                "replace single quotes with double quotes",
                "add the missing closing brace",
                "the document is already valid",
            ];
            // The correct repair is the one that yields valid JSON; a valid
            // document's correct option is "already valid". Construction
            // guarantees uniqueness — assert it rather than assume it.
            let working: Vec<usize> = (0..4)
                .filter(|&which| {
                    let repaired = apply_repair(&text, which);
                    repaired != text && is_valid_json(&repaired)
                })
                .collect();
            let expected_menu = if is_valid_json(&text) {
                assert!(working.is_empty(), "{}: valid doc has a repair", item.id);
                4
            } else {
                assert_eq!(working.len(), 1, "{}: repair not unique", item.id);
                working[0]
            };
            labels
                .iter()
                .position(|l| l == menu[expected_menu])
                .unwrap()
        }
        _ => unreachable!(),
    }
}

// --- syllogism: premise parser; transitive-chain decision made from the
// parsed premises alone. ---

struct Premises {
    /// (subject, predicate) of "All X are Y."
    p1: (String, String),
    p2: (String, String),
}

fn parse_premises(text: &str) -> Premises {
    let mut sentences = text.split(". ");
    let parse = |sentence: &str| {
        let words: Vec<&str> = sentence.trim_end_matches('.').split_whitespace().collect();
        assert_eq!(words[0], "All");
        assert_eq!(words[2], "are");
        (words[1].to_owned(), words[3].to_owned())
    };
    Premises {
        p1: parse(sentences.next().unwrap()),
        p2: parse(sentences.next().unwrap()),
    }
}

/// The `All _ are _` conclusion that follows from the premises, if any:
/// transitivity requires p1.predicate == p2.subject.
fn implied_all(p: &Premises) -> Option<(String, String)> {
    if p.p1.1 == p.p2.0 {
        Some((p.p1.0.clone(), p.p2.1.clone()))
    } else {
        None
    }
}

fn check_syllogism(item: &Item) -> usize {
    let premises = parse_premises(&state(item));
    let implied = implied_all(&premises);
    match item.question.kind {
        QuestionKind::Noul => {
            // "Does it follow that all {a} are {c}?"
            let text = instructions(item);
            let inner = text
                .strip_prefix("Does it follow that all ")
                .unwrap()
                .trim_end_matches('?');
            let (a, c) = inner.split_once(" are ").unwrap();
            usize::from(implied.as_ref().map(|(x, y)| (x.as_str(), y.as_str())) == Some((a, c)))
        }
        QuestionKind::Score => usize::from(implied.is_some()),
        QuestionKind::Choice => {
            let labels = item.question.labels();
            match implied {
                Some((a, c)) => {
                    let expected = format!("All {a} are {c}");
                    labels.iter().position(|l| *l == expected).unwrap()
                }
                None => labels
                    .iter()
                    .position(|l| l == "None of these follow")
                    .unwrap(),
            }
        }
        _ => unreachable!(),
    }
}

fn check(item: &Item) -> usize {
    match item.family {
        Family::Arith => check_arith(item),
        Family::UnitConv => check_unitconv(item),
        Family::Temporal => check_temporal(item),
        Family::JsonCheck => check_jsoncheck(item),
        Family::Syllogism => check_syllogism(item),
    }
}

#[test]
fn independent_truth_checker_agrees_with_every_frozen_item() {
    let items = generate_all(&BenchConfig::default());
    assert_eq!(items.len(), 3956, "frozen release item count changed");
    let mut mismatches: Vec<String> = Vec::new();
    let mut per_family: HashMap<Family, usize> = HashMap::new();
    for item in &items {
        *per_family.entry(item.family).or_insert(0) += 1;
        let decided = check(item);
        if decided != item.truth {
            mismatches.push(format!(
                "{}: checker={decided} truth={}",
                item.id, item.truth
            ));
        }
    }
    for family in Family::ALL {
        eprintln!("{family}: {} items checked", per_family[&family]);
    }
    assert!(
        mismatches.is_empty(),
        "{} truth mismatches:\n{}",
        mismatches.len(),
        mismatches[..mismatches.len().min(20)].join("\n")
    );
}
