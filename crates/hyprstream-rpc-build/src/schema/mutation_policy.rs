//! Checked per-method mutation-policy metadata for the generated v16 inventory.
//!
//! A mutating scope action must declare one of these exact annotation values.
//! The parser is intentionally closed: an absent, padded, or unknown value is
//! a build error, so code generation cannot infer retry safety from a scope.

/// The explicit semantics declared by `$mutationSemantics` on a mutating leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeclaredMutationSemantics {
    /// Retrying the same intended operation is safe without extra machinery.
    NaturallyIdempotent,
    /// The method payload carries an application idempotency key.
    IdempotencyKeyRequired,
    /// Correct retry semantics require an atomic result/mutation ledger or fencing.
    TransactionLedgerRequired,
}

/// Parse the closed `$mutationSemantics` annotation grammar.
pub fn parse_mutation_semantics(text: &str) -> Result<DeclaredMutationSemantics, String> {
    match text {
        "naturally-idempotent" => Ok(DeclaredMutationSemantics::NaturallyIdempotent),
        "idempotency-key-required" => Ok(DeclaredMutationSemantics::IdempotencyKeyRequired),
        "transaction-ledger-required" => Ok(DeclaredMutationSemantics::TransactionLedgerRequired),
        "" => Err("missing required `$mutationSemantics` declaration".into()),
        _ if text.trim() != text => Err(format!(
            "`$mutationSemantics` value {text:?} is padded; use one of naturally-idempotent, idempotency-key-required, transaction-ledger-required"
        )),
        _ => Err(format!(
            "unknown `$mutationSemantics` value {text:?}; use one of naturally-idempotent, idempotency-key-required, transaction-ledger-required"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_only_the_three_authoritative_values() {
        assert_eq!(
            parse_mutation_semantics("naturally-idempotent"),
            Ok(DeclaredMutationSemantics::NaturallyIdempotent)
        );
        assert_eq!(
            parse_mutation_semantics("idempotency-key-required"),
            Ok(DeclaredMutationSemantics::IdempotencyKeyRequired)
        );
        assert_eq!(
            parse_mutation_semantics("transaction-ledger-required"),
            Ok(DeclaredMutationSemantics::TransactionLedgerRequired)
        );
        for invalid in ["", " naturally-idempotent", "automatic", "NaturallyIdempotent"] {
            assert!(parse_mutation_semantics(invalid).is_err(), "{invalid:?}");
        }
    }
}
