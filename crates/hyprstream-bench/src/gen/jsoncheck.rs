//! JSON well-formedness / extraction. The `NearMiss` stratum is near-miss
//! JSON: documents broken by exactly one classic syntax error (trailing
//! comma, unquoted key, single quotes, missing closing brace). Validity is
//! decided by `serde_json`, so truth is mechanical.

use hyprstream_decision::Entry;

use crate::family::{Family, Stratum};
use crate::gen::stream;
use crate::item::Item;

const FAMILY: Family = Family::JsonCheck;

struct Doc {
    text: String,
    valid: bool,
    /// Number of injected syntax errors (0 = clean).
    errors: usize,
    /// The injected break (0=trailing comma, 1=unquoted key, 2=single quotes,
    /// 3=missing brace) or 4 = valid document. Threaded through instead of
    /// re-derived from the text — the text is the artifact, this is the truth.
    repair: usize,
}

fn document(seed: u64, stratum: Stratum) -> Doc {
    let mut rng = stream(seed);
    let key = ["name", "port", "zone", "owner", "quota"][rng.below(5) as usize];
    let value = rng.range(1, 9999);
    let host = format!("srv-{}", rng.range(1, 99));
    let clean = format!("{{\"{key}\": {value}, \"host\": \"{host}\"}}");
    if stratum != Stratum::NearMiss {
        return Doc {
            text: clean,
            valid: true,
            errors: 0,
            repair: 4,
        };
    }
    // Exactly one near-miss break, chosen by stream; a fraction of the
    // stratum stays valid so the truth labels mix.
    let variant = rng.below(5);
    let (text, errors) = match variant {
        0 => (format!("{{\"{key}\": {value}, \"host\": \"{host}\",}}"), 1), // trailing comma
        1 => (format!("{{{key}: {value}, \"host\": \"{host}\"}}"), 1),      // unquoted key
        2 => (format!("{{\"{key}\": {value}, \"host\": '{host}'}}"), 1),    // single quotes
        3 => (format!("{{\"{key}\": {value}, \"host\": \"{host}\""), 1),    // missing brace
        _ => (clean, 0),
    };
    let valid = errors == 0;
    debug_assert_eq!(valid, serde_json::from_str::<serde_json::Value>(&text).is_ok());
    Doc {
        text,
        valid,
        errors,
        repair: if valid { 4 } else { variant as usize },
    }
}

pub fn noul(seed: u64, stratum: Stratum) -> Item {
    let doc = document(seed, stratum);
    Item::noul(
        FAMILY,
        stratum,
        seed,
        Entry::Str(doc.text),
        "Is the state a syntactically valid JSON document?",
        doc.valid,
    )
}

pub fn choice(seed: u64, stratum: Stratum) -> Item {
    let mut rng = stream(seed ^ 0xC0);
    let doc = document(seed, stratum);
    let options = [
        "remove the trailing comma",
        "quote the bare object key",
        "replace single quotes with double quotes",
        "add the missing closing brace",
        "the document is already valid",
    ];
    // Map the injected break to its repair — threaded from `document`, not
    // re-derived by probing the text.
    let correct = doc.repair;
    // Shuffle options except we keep identity by value; correct rotates.
    let order = rng.permutation(options.len());
    let shuffled: Vec<String> = order.iter().map(|&i| options[i].to_owned()).collect();
    let new_correct = order
        .iter()
        .position(|&i| i == correct)
        .unwrap_or_else(|| unreachable!("permutation contains every index"));
    let _ = rng;
    Item::choice(
        FAMILY,
        stratum,
        seed,
        Entry::Str(doc.text),
        "Which single repair makes the state a valid JSON document?",
        shuffled,
        new_correct,
    )
}

pub fn score(seed: u64, stratum: Stratum) -> Item {
    let doc = document(seed, stratum);
    Item::score(
        FAMILY,
        stratum,
        seed,
        Entry::Str(doc.text),
        "How many JSON syntax errors does the state contain? Levels: 0 = none, 1 = one or more.",
        2,
        doc.errors.min(1),
    )
}
