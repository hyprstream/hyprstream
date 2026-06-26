//! Policy CSV migration from flat to dot-namespaced action strings.
//!
//! Hyprstream 0.x policy files used flat action names (`infer`, `manage`, etc.).
//! Version 0.y and later use dot-namespaced actions (`infer.generate`, `ttt.writeback`).
//! This module provides a helper that rewrites flat action strings to their
//! dot-namespaced equivalents in a `policy.csv` string.

/// Map a flat action string to its canonical dot-namespaced equivalent.
/// Returns `None` if the string is not a recognized flat action — this includes
/// already dot-namespaced strings as well as completely unknown strings.
pub fn migrate_action(action: &str) -> Option<&'static str> {
    match action {
        "infer"   => Some("infer.generate"),
        "train"   => Some("ttt.train"),
        "query"   => Some("query.status"),
        "write"   => Some("persist.save"),
        "serve"   => Some("serve.api"),
        "manage"  => Some("ttt.writeback"),
        "context" => Some("context.augment"),
        _ => None,
    }
}

/// Rewrite a `policy.csv` string, upgrading flat action strings to dot-namespaced form.
///
/// - Lines that already have dot-namespaced actions (containing `.`) are left unchanged.
/// - `g`-lines (role inheritance) and comments are left unchanged.
/// - Only 6-field `p`-lines (`p, sub, dom, obj, act, eft`) are rewritten.
pub fn migrate_policy_csv(csv: &str) -> String {
    csv.lines()
        .map(|line| {
            let trimmed = line.trim();
            // Only rewrite p-lines (policy rules), not g-lines or comments
            if !trimmed.starts_with("p,") {
                return line.to_owned();
            }
            let parts: Vec<&str> = trimmed.splitn(6, ',').collect();
            // p, sub, dom, obj, act, eft — 6 parts
            if parts.len() == 6 {
                let act = parts[4].trim();
                if let Some(new_act) = migrate_action(act) {
                    return format!(
                        "{}, {}, {}, {}, {}, {}",
                        parts[0].trim(),
                        parts[1].trim(),
                        parts[2].trim(),
                        parts[3].trim(),
                        new_act,
                        parts[5].trim(),
                    );
                }
            }
            line.to_owned()
        })
        .collect::<Vec<_>>()
        .join("\n")
        + if csv.ends_with('\n') { "\n" } else { "" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrate_flat_p_rule() {
        let csv = "p, alice, *, model:*, infer, allow";
        assert_eq!(
            migrate_policy_csv(csv),
            "p, alice, *, model:*, infer.generate, allow"
        );
    }

    #[test]
    fn test_migrate_all_flat_actions() {
        for (flat, expected) in [
            ("infer",   "infer.generate"),
            ("train",   "ttt.train"),
            ("query",   "query.status"),
            ("write",   "persist.save"),
            ("serve",   "serve.api"),
            ("manage",  "ttt.writeback"),
            ("context", "context.augment"),
        ] {
            let csv = format!("p, alice, *, model:*, {flat}, allow");
            let result = migrate_policy_csv(&csv);
            assert!(
                result.contains(expected),
                "Expected '{expected}' in migrated output for flat action '{flat}', got: {result}"
            );
        }
    }

    #[test]
    fn test_migrate_leaves_g_lines_unchanged() {
        let csv = "g, alice, ttt.user";
        assert_eq!(migrate_policy_csv(csv), csv);
    }

    #[test]
    fn test_migrate_leaves_already_namespaced_unchanged() {
        let csv = "p, alice, *, model:*, infer.generate, allow";
        assert_eq!(migrate_policy_csv(csv), csv);
    }

    #[test]
    fn test_migrate_leaves_comments_unchanged() {
        let csv = "# This is a comment";
        assert_eq!(migrate_policy_csv(csv), csv);
    }

    #[test]
    fn test_migrate_preserves_trailing_newline() {
        let csv_with = "p, alice, *, model:*, infer, allow\n";
        let csv_without = "p, alice, *, model:*, infer, allow";
        assert!(migrate_policy_csv(csv_with).ends_with('\n'));
        assert!(!migrate_policy_csv(csv_without).ends_with('\n'));
    }

    #[test]
    fn test_migrate_multiline() {
        let csv = "# Comment\np, alice, *, model:*, infer, allow\ng, alice, ttt.user\np, bob, *, model:*, manage, allow";
        let result = migrate_policy_csv(csv);
        assert!(result.contains("infer.generate"));
        assert!(result.contains("ttt.writeback"));
        assert!(result.contains("g, alice, ttt.user"));
        assert!(result.contains("# Comment"));
    }
}
