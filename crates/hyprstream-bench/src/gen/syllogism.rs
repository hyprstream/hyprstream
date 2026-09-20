//! Set-logic syllogisms — **held-out gate family** (firewalled from training
//! synthesis; measured by the P1.4 leg-(e) zero-shot transfer gate). Chains of
//! All premises over synthetic category names; validity is decided by the
//! chain structure, so truth is mechanical.
//!
//! The category bindings are threaded through [`Chain`] — never recovered
//! from the rendered text by position (r1 review blocker: positional
//! extraction returned the middle term for invalid chains and mislabeled
//! ~7.4% of the frozen vob-1.0 items).

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
    /// First category (subject of premise 1).
    a: &'static str,
    /// Third category (predicate of the queried conclusion).
    c: &'static str,
    /// Does `All A are C` follow from the premises?
    all_ac: bool,
}

fn chain(seed: u64, stratum: Stratum) -> Chain {
    let mut rng = stream(seed);
    let base = rng.below(NAMES.len() as u64) as usize;
    let a = NAMES[base];
    let b = NAMES[(base + 1) % NAMES.len()];
    let c = NAMES[(base + 2) % NAMES.len()];
    // Valid chain: All A are B. All B are C. ⇒ All A are C.
    // Near-miss invalid chain: All A are B. All C are B. (⇏ All A are C) —
    // the classic undistributed-middle fallacy, one word swapped away from
    // the valid form. Strata differentiate difficulty: `clean` is mostly the
    // valid transitive chain, `nearmiss` is mostly the fallacy.
    let valid = match stratum {
        Stratum::NearMiss => rng.below(4) == 0,
        _ => rng.below(4) != 0,
    };
    let text = if valid {
        format!("All {a} are {b}. All {b} are {c}.")
    } else {
        format!("All {a} are {b}. All {c} are {b}.")
    };
    Chain {
        text,
        a,
        c,
        all_ac: valid,
    }
}

pub fn noul(seed: u64, stratum: Stratum) -> Item {
    let chain = chain(seed, stratum);
    Item::noul(
        FAMILY,
        stratum,
        seed,
        Entry::Str(chain.text),
        format!("Does it follow that all {} are {}?", chain.a, chain.c),
        chain.all_ac,
    )
}

pub fn choice(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xC0);
    let chain = chain(seed, stratum);
    let options = [
        format!("All {} are {}", chain.a, chain.c),
        format!("No {} are {}", chain.a, chain.c),
        format!("Some {} are not {}", chain.a, chain.c),
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
