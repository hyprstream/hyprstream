//! Set-logic syllogisms — **held-out gate family** (firewalled from training
//! synthesis; measured by the P1.4 leg-(e) zero-shot transfer gate). Chains of
//! All/Some/No premises over synthetic category names; validity is decided by
//! the chain structure, so truth is mechanical.

use hyprstream_decision::Entry;

use crate::family::{Family, Stratum};
use crate::gen::stream;
use crate::item::Item;

const FAMILY: Family = Family::Syllogism;

const NAMES: [&str; 12] = [
    "glorps", "wibbles", "snarfs", "krendels", "plughs", "zorbs", "mippels", "tarns",
    "quells", "droves", "fims", "jastrels",
];

struct Chain {
    text: String,
    /// Does `All A are C` follow from the premises?
    all_ac: bool,
}

fn chain(seed: u64, _stratum: Stratum) -> Chain {
    let mut rng = stream(seed);
    let base = rng.below(NAMES.len() as u64) as usize;
    let a = NAMES[base];
    let b = NAMES[(base + 1) % NAMES.len()];
    let c = NAMES[(base + 2) % NAMES.len()];
    // Valid chain: All A are B. All B are C. ⇒ All A are C.
    // Near-miss invalid chain: All A are B. All C are B. (⇏ All A are C) —
    // the classic undistributed-middle fallacy, one word swapped away from
    // the valid form.
    // Half of every stratum is the valid chain; the NearMiss invalid form is
    // the undistributed-middle fallacy — one word swapped away from valid.
    let valid = rng.below(2) == 0;
    let text = if valid {
        format!("All {a} are {b}. All {b} are {c}.")
    } else {
        format!("All {a} are {b}. All {c} are {b}.")
    };
    Chain {
        text,
        all_ac: valid,
    }
}

pub fn noul(seed: u64, stratum: Stratum) -> Item {
    let chain = chain(seed, stratum);
    let c_name = extract_third(&chain.text);
    let a_name = extract_first(&chain.text);
    Item::noul(
        FAMILY,
        stratum,
        seed,
        Entry::Str(chain.text),
        format!("Does it follow that all {a_name} are {c_name}?"),
        chain.all_ac,
    )
}

fn extract_first(text: &str) -> String {
    text.split_whitespace().nth(1).unwrap_or("A").to_owned()
}

fn extract_third(text: &str) -> String {
    text.split_whitespace()
        .nth_back(0)
        .unwrap_or("C")
        .trim_end_matches('.')
        .to_owned()
}

pub fn choice(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xC0);
    let chain = chain(seed, stratum);
    let a_name = extract_first(&chain.text);
    let c_name = extract_third(&chain.text);
    let options = [
        format!("All {a_name} are {c_name}"),
        format!("No {a_name} are {c_name}"),
        format!("Some {a_name} are not {c_name}"),
        "None of these follow".to_owned(),
    ];
    let correct = if chain.all_ac { 0 } else { 3 };
    let order = rng.permutation(options.len());
    let shuffled: Vec<String> = order.iter().map(|&i| options[i].clone()).collect();
    let new_correct = order
        .iter()
        .position(|&i| i == correct)
        .unwrap_or_else(|| unreachable!("permutation contains every index"));
    Item::choice(
        FAMILY,
        stratum,
        seed,
        Entry::Str(chain.text),
        "Which conclusion follows from the premises?",
        shuffled,
        new_correct,
    )
}

pub fn score(seed: u64, stratum: Stratum) -> Item {
    let chain = chain(seed, stratum);
    Item::score(
        FAMILY,
        stratum,
        seed,
        Entry::Str(chain.text),
        "How strong is the inference from the premises to \"all of the first category are the third category\"? Levels: 0 = does not follow, 1 = follows necessarily.",
        2,
        usize::from(chain.all_ac),
    )
}
