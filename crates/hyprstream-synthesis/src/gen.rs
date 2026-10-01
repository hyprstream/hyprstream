//! Per-family item generators. Every generator is a pure function of
//! `(seed, paraphrase)` over the pinned [`BenchRng`] stream — same seed ⇒
//! bit-identical item, forever. Paraphrase index `p` selects instruction and
//! rubric variants from each family's pools (`p == 0` is the base phrasing);
//! this is the mandatory rubric-paraphrase augmentation surface (S6b1).

use hyprstream_bench::rng::BenchRng;
use hyprstream_decision::{ChoiceOption, Entry, NoulCriteria, QuestionBody};

use crate::family::SynthFamily;
use crate::item::SynthItem;

/// Number of instruction paraphrase variants every family template offers.
/// `pipeline::SynthConfig::paraphrase_variants` must be in `2..=PARAPHRASES`.
pub const PARAPHRASES: u32 = 3;

/// Generate one base or paraphrased item (rotation applied mechanically by
/// the caller via [`SynthItem::cyclic_permutation`]).
pub(crate) fn generate(family: SynthFamily, kind: u64, seed: u64, paraphrase: u32) -> SynthItem {
    match (family, kind) {
        (SynthFamily::Triage, 0) => triage::noul(seed, paraphrase),
        (SynthFamily::Triage, 1) => triage::choice(seed, paraphrase),
        (SynthFamily::Triage, _) => triage::score(seed, paraphrase),
        (SynthFamily::Compliance, 0) => compliance::noul(seed, paraphrase),
        (SynthFamily::Compliance, 1) => compliance::choice(seed, paraphrase),
        (SynthFamily::Compliance, _) => compliance::score(seed, paraphrase),
        (SynthFamily::Extraction, 0) => extraction::noul(seed, paraphrase),
        (SynthFamily::Extraction, 1) => extraction::choice(seed, paraphrase),
        (SynthFamily::Extraction, _) => extraction::score(seed, paraphrase),
        (SynthFamily::Approvals, 0) => approvals::noul(seed, paraphrase),
        (SynthFamily::Approvals, 1) => approvals::choice(seed, paraphrase),
        (SynthFamily::Approvals, _) => approvals::score(seed, paraphrase),
    }
}

/// Deterministic per-(family, kind, index) seed. Distinct tag space from the
/// benchmark's `item_seed` (different base and layout — the two corpora must
/// never share a stream).
pub(crate) fn item_seed(seed_base: u64, family: SynthFamily, kind: u64, index: u32) -> u64 {
    seed_base ^ (family.tag() << 52) ^ (kind << 44) ^ (u64::from(index) << 4)
}

/// Pick pool element `index % pool.len()` (paraphrase pools are indexed by
/// the paraphrase coordinate; slot pools by rng draws).
fn pick<'a>(pool: &[&'a str], index: usize) -> &'a str {
    pool.get(index % pool.len()).copied().unwrap_or("")
}

/// Draw a cardinality in `lo..=hi` and a shuffled option subset of that size.
fn option_subset(
    rng: &mut BenchRng,
    pool: &[(&str, [&str; 2])],
    lo: usize,
    hi: usize,
) -> Vec<usize> {
    debug_assert!(hi <= pool.len() && lo >= 2);
    let k = lo + rng.below((hi - lo + 1) as u64) as usize;
    let mut order = rng.permutation(pool.len());
    order.truncate(k);
    order
}

/// Rubric for a pool entry at paraphrase coordinate `p` (2 variants each).
fn rubric(variants: &[&str; 2], paraphrase: u32) -> Option<Entry> {
    Some(Entry::Str(
        pick(&variants[..], paraphrase as usize % 2).to_owned(),
    ))
}

mod triage {
    use super::*;

    const PRODUCTS: [&str; 6] = [
        "FerruleOS workstation image",
        "hyprstream inference gateway",
        "DeltaPool tenant adapter",
        "cas-serve content store",
        "MoQ streaming plane",
        "git2db model registry",
    ];
    // Each issue maps to the team that owns it — the correct route is
    // state-derived, so every offered subset can contain a valid label
    // (review thread 2026-09-22 20:58).
    const ISSUES: [(&str, usize); 12] = [
        ("intermittent checksum failures", 1),
        ("a sudden latency regression", 1),
        ("an authentication loop", 1),
        ("unexpected restarts", 1),
        ("garbled streamed output", 1),
        ("a billing discrepancy", 0),
        ("an unexpected subscription charge", 0),
        ("a refund that never arrived", 0),
        ("a lost shipment", 2),
        ("a delivery stuck in transit", 2),
        ("an account lockout after a tenant transfer", 3),
        ("a question about upgrading the plan", 4),
    ];
    const TEAMS: [(&str, [&str; 2]); 5] = [
        (
            "billing",
            [
                "Invoices, refunds, and plan changes.",
                "Anything money-side: charges, credits, subscriptions.",
            ],
        ),
        (
            "technical-support",
            [
                "Product defects and usage problems.",
                "Break/fix issues with the product itself.",
            ],
        ),
        (
            "shipping",
            [
                "Delivery status and logistics.",
                "Orders in transit, lost or late packages.",
            ],
        ),
        (
            "account-management",
            [
                "Account access, ownership, and tenancy.",
                "Login, provisioning, and account lifecycle.",
            ],
        ),
        (
            "sales",
            [
                "New purchases and upgrades.",
                "Pre-sales questions and expansion.",
            ],
        ),
    ];
    const NOUL_Q: [&str; 3] = [
        "Is this ticket urgent?",
        "Does this ticket require immediate attention?",
        "Should this ticket jump the normal queue?",
    ];
    const CHOICE_Q: [&str; 3] = [
        "Which team should handle this ticket?",
        "Route this ticket to the best team.",
        "Pick the team best suited to resolve this ticket.",
    ];
    const SCORE_Q: [&str; 3] = [
        "Rate the severity of this ticket.",
        "How severe is the customer impact?",
        "Score the urgency of this report.",
    ];

    fn state(rng: &mut BenchRng) -> (Entry, usize) {
        let ticket = 1000 + rng.below(90_000);
        let product = pick(&PRODUCTS, rng.below(PRODUCTS.len() as u64) as usize);
        let (issue, team) = ISSUES[rng.below(ISSUES.len() as u64) as usize];
        let hours = 1 + rng.below(72);
        (
            Entry::Str(format!(
                "Ticket #{ticket}: {issue} with the {product}. Customer first reported it {hours} hours ago."
            )),
            team,
        )
    }

    pub(super) fn noul(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let (state, _team) = state(&mut rng);
        SynthItem::new(
            SynthFamily::Triage,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&NOUL_Q, paraphrase as usize).to_owned()),
            QuestionBody::Noul {
                criteria: Some(NoulCriteria {
                    on_true: Some(Entry::Str(
                        "The customer is blocked or losing service right now.".to_owned(),
                    )),
                    on_false: Some(Entry::Str(
                        "Normal queue handling is acceptable.".to_owned(),
                    )),
                }),
            },
        )
    }

    pub(super) fn choice(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let (state, truth) = state(&mut rng);
        // The state-derived owning team must be among the offered options.
        let mut subset = option_subset(&mut rng, &TEAMS, 2, 4);
        if !subset.contains(&truth) {
            let slot = rng.below(subset.len() as u64) as usize;
            subset[slot] = truth;
        }
        let options = subset
            .into_iter()
            .map(|i| ChoiceOption {
                name: TEAMS[i].0.to_owned(),
                rubric: rubric(&TEAMS[i].1, paraphrase + i as u32),
            })
            .collect();
        SynthItem::new(
            SynthFamily::Triage,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&CHOICE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Choice { options },
        )
    }

    pub(super) fn score(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let (state, _team) = state(&mut rng);
        let levels = 2 + rng.below(4) as usize; // 2..=5
        let rubrics = [
            "No customer impact.",
            "Minor inconvenience.",
            "Work impaired.",
            "Work blocked.",
            "Outage-level impact.",
        ];
        let levels = (0..levels)
            .map(|i| {
                if paraphrase == 0 || i % 2 == 0 {
                    Some(Entry::Str(rubrics[i].to_owned()))
                } else {
                    Some(Entry::Str(format!(
                        "Level {i}: {}",
                        rubrics[i].to_lowercase()
                    )))
                }
            })
            .collect();
        SynthItem::new(
            SynthFamily::Triage,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&SCORE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Score { levels },
        )
    }
}

mod compliance {
    use super::*;

    // Policy -> policy-area index (see AREAS): the choice truth is
    // state-derived, so the rendered Policy/Action pair is area-coherent.
    const POLICY_AREA: [usize; 6] = [0, 1, 2, 3, 0, 3];
    const POLICIES: [&str; 6] = [
        "Customer data must be deleted within 30 days of account closure.",
        "Production access requires an approved change ticket.",
        "Refunds over 500 USD need a manager sign-off.",
        "Internal benchmarks may not be shared outside the company.",
        "Vendor contracts must be stored in the document system.",
        "On-call handoffs must be acknowledged in writing.",
    ];
    // Action -> policy-area index (see AREAS).
    const ACTION_AREA: [usize; 8] = [0, 1, 2, 3, 0, 3, 2, 1];
    const ACTIONS: [&str; 8] = [
        "An engineer wiped the account archive 45 days after closure.",
        "A contractor logged into production with a shared key.",
        "Support issued a 650 USD refund without escalation.",
        "A slide deck quoted internal latency numbers at a conference.",
        "The signed agreement was attached to a chat thread.",
        "The incoming on-call replied with a thumbs-up reaction.",
        "Finance reimbursed a 120 USD team dinner.",
        "A developer reproduced a staging bug in the test environment.",
    ];
    const AREAS: [(&str, [&str; 2]); 4] = [
        (
            "data-retention",
            [
                "How long records are kept and when they are destroyed.",
                "Retention windows and deletion duties.",
            ],
        ),
        (
            "access-control",
            [
                "Who may reach production and under what approval.",
                "Production entitlements and approvals.",
            ],
        ),
        (
            "spending-limits",
            [
                "Approval thresholds for money movement.",
                "Refund and expense authorization limits.",
            ],
        ),
        (
            "communications",
            [
                "What may be said or shared externally.",
                "External disclosure rules.",
            ],
        ),
    ];
    const NOUL_Q: [&str; 3] = [
        "Does the action violate the stated policy?",
        "Is this action out of policy?",
        "Does the described action breach the policy?",
    ];
    const CHOICE_Q: [&str; 3] = [
        "Which policy area does this case fall under?",
        "Classify this case by policy area.",
        "Pick the policy area this case concerns.",
    ];
    const SCORE_Q: [&str; 3] = [
        "Rate the compliance risk of this case.",
        "How risky is this case from a compliance view?",
        "Score the policy exposure of this case.",
    ];

    fn state(rng: &mut BenchRng) -> (Entry, usize) {
        // The case's policy area is state-derived; within the requested
        // area the policy and action are drawn coherently so the pair is
        // about the same rule (review thread 2026-09-22 20:58).
        let area = rng.below(AREAS.len() as u64) as usize;
        let policies: Vec<usize> = (0..POLICIES.len())
            .filter(|i| POLICY_AREA[*i] == area)
            .collect();
        let actions: Vec<usize> = (0..ACTIONS.len())
            .filter(|i| ACTION_AREA[*i] == area)
            .collect();
        let policy = POLICIES[policies[rng.below(policies.len() as u64) as usize]];
        let action = ACTIONS[actions[rng.below(actions.len() as u64) as usize]];
        (
            Entry::Str(format!("Policy: {policy}\nAction: {action}")),
            area,
        )
    }

    pub(super) fn noul(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let (state, _area) = state(&mut rng);
        SynthItem::new(
            SynthFamily::Compliance,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&NOUL_Q, paraphrase as usize).to_owned()),
            QuestionBody::Noul { criteria: None },
        )
    }

    pub(super) fn choice(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let (state, truth) = state(&mut rng);
        // The state-derived policy area must be among the offered options.
        let mut subset = option_subset(&mut rng, &AREAS, 2, 4);
        if !subset.contains(&truth) {
            let slot = rng.below(subset.len() as u64) as usize;
            subset[slot] = truth;
        }
        let options = subset
            .into_iter()
            .map(|i| ChoiceOption {
                name: AREAS[i].0.to_owned(),
                rubric: rubric(&AREAS[i].1, paraphrase + i as u32),
            })
            .collect();
        SynthItem::new(
            SynthFamily::Compliance,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&CHOICE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Choice { options },
        )
    }

    pub(super) fn score(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let (state, _area) = state(&mut rng);
        let levels = 2 + rng.below(4) as usize;
        let rubrics = [
            "No exposure.",
            "Administrative gap.",
            "Needs remediation.",
            "Reportable incident.",
            "Severe breach.",
        ];
        let levels = (0..levels)
            .map(|i| {
                if paraphrase == 0 {
                    Some(Entry::Str(rubrics[i].to_owned()))
                } else {
                    Some(Entry::Str(format!(
                        "Risk {i}: {}",
                        rubrics[i].to_lowercase()
                    )))
                }
            })
            .collect();
        SynthItem::new(
            SynthFamily::Compliance,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&SCORE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Score { levels },
        )
    }
}

mod extraction {
    use super::*;

    const VENDORS: [&str; 6] = [
        "Acme Parts",
        "Northern Freight",
        "Kessler & Sohn",
        "Bluepeak Supply",
        "Ridgeline Office",
        "Halvard Logistics",
    ];
    const DOCTYPES: [(&str, [&str; 2]); 4] = [
        (
            "invoice",
            [
                "A request for payment with line items and totals.",
                "Billing document requesting payment.",
            ],
        ),
        (
            "receipt",
            [
                "Proof that payment already happened.",
                "Post-payment confirmation record.",
            ],
        ),
        (
            "purchase-order",
            [
                "A buyer-issued order before fulfillment.",
                "Pre-fulfillment ordering document.",
            ],
        ),
        (
            "quote",
            [
                "A price offer, not a demand for payment.",
                "Non-binding price proposal.",
            ],
        ),
    ];
    const NOUL_Q: [&str; 3] = [
        "Does the record contain a valid total?",
        "Is there a well-formed total in this record?",
        "Can a valid total be extracted from this record?",
    ];
    const CHOICE_Q: [&str; 3] = [
        "Which document type is this record?",
        "Classify the document type of this record.",
        "What kind of document is this?",
    ];
    const SCORE_Q: [&str; 3] = [
        "Rate the completeness of this record.",
        "How complete is this record?",
        "Score the extraction readiness of this record.",
    ];

    fn state(rng: &mut BenchRng, doctype: Option<usize>) -> Entry {
        let vendor = pick(&VENDORS, rng.below(VENDORS.len() as u64) as usize);
        let number = 10_000 + rng.below(89_999);
        let currency = pick(&["USD", "EUR", "GBP"], rng.below(3) as usize);
        let total_present = rng.below(4) > 0;
        let mut record = vec![
            ("vendor".to_owned(), Entry::Str(vendor.to_owned())),
            ("doc_number".to_owned(), Entry::Str(format!("D-{number}"))),
            ("currency".to_owned(), Entry::Str(currency.to_owned())),
        ];
        if total_present {
            let cents = 100 + rng.below(9_900_000);
            record.push(("total".to_owned(), Entry::Number(cents as f64 / 100.0)));
        }
        // Type-discriminating evidence: the choice variant asks for the
        // document type, so the record must carry state-derived evidence
        // that determines the class (review thread 2026-09-22 18:27 —
        // without it the correct answer reflects teacher priors, not the
        // input, contaminating the extraction-choice training slice).
        match doctype {
            Some(0) => {
                // invoice: line items + a payment demand date.
                record.push((
                    "line_items".to_owned(),
                    Entry::Number((1 + rng.below(12)) as f64),
                ));
                record.push(("due_date".to_owned(), Entry::Str("2026-11-15".to_owned())));
            }
            Some(1) => {
                // receipt: payment already settled.
                record.push((
                    "payment_method".to_owned(),
                    Entry::Str(pick(&["card", "cash", "transfer"], rng.below(3) as usize).to_owned()),
                ));
                record.push(("paid".to_owned(), Entry::Bool(true)));
            }
            Some(2) => {
                // purchase-order: buyer-issued ordering reference.
                record.push(("po_number".to_owned(), Entry::Str(format!("PO-{number}"))));
                record.push(("approver".to_owned(), Entry::Str("procurement".to_owned())));
            }
            Some(3) => {
                // quote: non-binding price offer with a validity window.
                record.push((
                    "validity_days".to_owned(),
                    Entry::Number((5 + rng.below(30)) as f64),
                ));
                record.push(("binding".to_owned(), Entry::Bool(false)));
            }
            _ => {}
        }
        Entry::Map(record)
    }

    pub(super) fn noul(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let state = state(&mut rng, None);
        SynthItem::new(
            SynthFamily::Extraction,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&NOUL_Q, paraphrase as usize).to_owned()),
            QuestionBody::Noul {
                criteria: Some(NoulCriteria {
                    on_true: Some(Entry::Str(
                        "A numeric total with a currency is present.".to_owned(),
                    )),
                    on_false: None,
                }),
            },
        )
    }

    pub(super) fn choice(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        // The correct class is state-derived: pick the true document type
        // first, build the record with that type's discriminating evidence,
        // and guarantee the true type is among the offered options.
        let truth = rng.below(DOCTYPES.len() as u64) as usize;
        let state = state(&mut rng, Some(truth));
        let mut subset = option_subset(&mut rng, &DOCTYPES, 2, 4);
        if !subset.contains(&truth) {
            let slot = rng.below(subset.len() as u64) as usize;
            subset[slot] = truth;
        }
        let options = subset
            .into_iter()
            .map(|i| ChoiceOption {
                name: DOCTYPES[i].0.to_owned(),
                rubric: rubric(&DOCTYPES[i].1, paraphrase + i as u32),
            })
            .collect();
        SynthItem::new(
            SynthFamily::Extraction,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&CHOICE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Choice { options },
        )
    }

    pub(super) fn score(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let state = state(&mut rng, None);
        let levels = 2 + rng.below(3) as usize; // 2..=4
        let rubrics = [
            "Key fields missing.",
            "Partially populated.",
            "Mostly complete.",
            "Fully populated.",
        ];
        let levels = (0..levels)
            .map(|i| {
                if paraphrase == 0 {
                    Some(Entry::Str(rubrics[i].to_owned()))
                } else {
                    Some(Entry::Str(format!("Completeness {i} of {}.", levels - 1)))
                }
            })
            .collect();
        SynthItem::new(
            SynthFamily::Extraction,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&SCORE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Score { levels },
        )
    }
}

mod approvals {
    use super::*;

    const REQUESTERS: [&str; 6] = [
        "the platform team",
        "field operations",
        "the data engineering group",
        "customer success",
        "the security team",
        "regional sales",
    ];
    const PURPOSES: [&str; 6] = [
        "replacement GPUs for the reference rig",
        "a team offsite",
        "renewal of the monitoring contract",
        "conference travel",
        "emergency hardware spares",
        "a training-data storage expansion",
    ];
    const DISPOSITIONS: [(&str, [&str; 2]); 3] = [
        (
            "approve",
            [
                "The request meets policy and budget.",
                "Within limits; sign off.",
            ],
        ),
        (
            "reject",
            [
                "The request is out of policy or unjustified.",
                "Decline: policy or justification fails.",
            ],
        ),
        (
            "escalate",
            [
                "A higher approver must decide.",
                "Above this approver's authority.",
            ],
        ),
    ];
    const NOUL_Q: [&str; 3] = [
        "Does this request exceed the stated approval limit?",
        "Is this request over the approval limit?",
        "Does the amount breach the approval threshold?",
    ];
    const CHOICE_Q: [&str; 3] = [
        "What should happen to this request?",
        "Decide the disposition of this request.",
        "How should this request be handled?",
    ];
    const SCORE_Q: [&str; 3] = [
        "Rate the priority of this request.",
        "How urgent is this request?",
        "Score the handling priority.",
    ];

    fn state(rng: &mut BenchRng) -> Entry {
        let requester = pick(&REQUESTERS, rng.below(REQUESTERS.len() as u64) as usize);
        let purpose = pick(&PURPOSES, rng.below(PURPOSES.len() as u64) as usize);
        let amount = 50 + rng.below(49_950);
        let limit = pick(&["500", "1000", "5000", "10000"], rng.below(4) as usize);
        Entry::Str(format!(
            "Approval request from {requester}: {purpose}, amount {amount} USD. The requester's approval limit is {limit} USD."
        ))
    }

    pub(super) fn noul(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let state = state(&mut rng);
        SynthItem::new(
            SynthFamily::Approvals,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&NOUL_Q, paraphrase as usize).to_owned()),
            QuestionBody::Noul { criteria: None },
        )
    }

    pub(super) fn choice(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let state = state(&mut rng);
        // All three dispositions are always present; cardinality variation
        // comes from paraphrase-driven rubric variants here and from the
        // other families' option subsets.
        let options = DISPOSITIONS
            .iter()
            .enumerate()
            .map(|(i, (name, rubrics))| ChoiceOption {
                name: (*name).to_owned(),
                rubric: rubric(rubrics, paraphrase + i as u32),
            })
            .collect();
        SynthItem::new(
            SynthFamily::Approvals,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&CHOICE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Choice { options },
        )
    }

    pub(super) fn score(seed: u64, paraphrase: u32) -> SynthItem {
        let mut rng = BenchRng::new(seed);
        let state = state(&mut rng);
        let levels = 2 + rng.below(3) as usize; // 2..=4
        let rubrics = [
            "Handle whenever.",
            "Handle this week.",
            "Handle today.",
            "Handle immediately.",
        ];
        let levels = (0..levels)
            .map(|i| {
                if paraphrase == 0 {
                    Some(Entry::Str(rubrics[i].to_owned()))
                } else {
                    Some(Entry::Str(format!("Priority {i}.")))
                }
            })
            .collect();
        SynthItem::new(
            SynthFamily::Approvals,
            seed,
            0,
            paraphrase,
            state,
            Some(pick(&SCORE_Q, paraphrase as usize).to_owned()),
            QuestionBody::Score { levels },
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_decision::QuestionKind;

    #[test]
    fn generation_is_deterministic_per_seed_and_paraphrase() {
        for family in SynthFamily::ALL {
            for kind in 0..3 {
                let a = generate(family, kind, 0xABCD, 1);
                let b = generate(family, kind, 0xABCD, 1);
                assert_eq!(a.canonical_bytes(), b.canonical_bytes());
            }
        }
    }

    #[test]
    fn paraphrase_changes_text_but_not_identity_slots() {
        for family in SynthFamily::ALL {
            for kind in 0..3 {
                let base = generate(family, kind, 42, 0);
                let rephrased = generate(family, kind, 42, 1);
                assert_eq!(base.group, rephrased.group);
                assert_eq!(base.seed, rephrased.seed);
                assert_ne!(base.hash(), rephrased.hash());
                assert_ne!(base.canonical_bytes(), rephrased.canonical_bytes());
            }
        }
    }

    #[test]
    fn every_item_conforms_to_the_jev1_contract() {
        for family in SynthFamily::ALL {
            for kind in 0..3 {
                for index in 0..8u64 {
                    let item = generate(family, kind, index * 0x1111 + 7, 0);
                    let cardinality = item.question.cardinality();
                    assert!((2..=255).contains(&cardinality), "{}", item.id);
                    assert_eq!(item.question.id, item.id);
                    // The jev-1 Arrow contract grammar: [A-Za-z_][A-Za-z0-9_]*.
                    let mut chars = item.id.chars();
                    let first = chars.next().unwrap_or('-');
                    assert!(
                        first.is_ascii_alphabetic() || first == '_',
                        "{}: identifier start",
                        item.id
                    );
                    assert!(
                        chars.all(|c| c.is_ascii_alphanumeric() || c == '_'),
                        "{}: identifier-safe for Arrow field names",
                        item.id
                    );
                    match kind {
                        0 => assert_eq!(item.question.kind, QuestionKind::Noul),
                        1 => assert_eq!(item.question.kind, QuestionKind::Choice),
                        _ => assert_eq!(item.question.kind, QuestionKind::Score),
                    }
                }
            }
        }
    }

    #[test]
    fn cardinality_varies_across_seeds() {
        // Cardinality variation is part of the diversity contract: over a
        // seed sweep, choice and score items must take more than one width.
        for kind in [1u64, 2] {
            let widths: std::collections::BTreeSet<usize> = (0..64u64)
                .map(|i| {
                    generate(SynthFamily::Triage, kind, i * 977 + 3, 0)
                        .question
                        .cardinality()
                })
                .collect();
            assert!(widths.len() > 1, "kind {kind}: widths {widths:?}");
        }
    }
}
