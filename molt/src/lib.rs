// Upstream molt code — suppress clippy lints that predate our fork.
// Our additions (instruction_limit) follow current Rust style.
#![allow(clippy::all, dead_code, unused_imports)]

//! # Molt Client Library
//!
//! This module defines the API for Molt clients.
//! The [`interp`] module defines the Molt interpreter itself, and provides the primary
//! API.  Values in the Molt language are stored internally using the [`Value`] struct.  Other
//! relevant data types, including [`MoltResult`] and [`ResultCode`], are defined in
//! the [`types`] module.
//!
//! The [`test_harness`] module defines the test runner for Molt's TCL-level testing.  It
//! can be used directly in Cargo integration tests or via a Molt shell, whether standard or
//! custom.
//!
//! See [The Molt Book] for an introduction to Molt.
//!
//! [The Molt Book]: https://wduquette.github.io/molt/
//! [`MoltResult`]: types/type.MoltResult.html
//! [`ResultCode`]: types/enum.ResultCode.html
//! [`Value`]: value/index.html
//! [`interp`]: interp/index.html
//! [`types`]: types/index.html
//! [`test_harness`]: test_harness/index.html

#![doc(html_root_url = "https://docs.rs/molt/0.3.0")]
#![doc(html_logo_url = "https://github.com/wduquette/molt/raw/master/MoltLogo.png")]

pub use crate::interp::Interp;
pub use crate::test_harness::test_harness;
pub use crate::types::*;

mod commands;
pub mod dict;
mod eval_ptr;
mod expr;
pub mod interp;
mod list;
mod tokenizer;
#[macro_use]
mod macros;
mod parser;
pub mod scope;
pub mod test_harness;
pub mod types;
mod util;
pub mod value;

/// This function is used in command functions to check whether the command's argument
/// list is of a proper size for the given command.  If it is, `check_args` returns
/// the empty result; if not, it returns a Molt error message
/// `wrong # args: should be "syntax..."`, where _syntax_ is the command's syntax.
/// It is typically called at the beginning of a command function.
///
/// The _argv_ is the argument list, including the command name.
///
/// The _namec_ is the number of tokens in the argument that constitute the command
/// name.  It is usually 1, but would be 2 for a command with subcommands.  The
/// error message will take those tokens verbatim from _argv_.
///
/// _min_ and _max_ are the minimum and maximum valid length for _argv_.  If
/// _max_ is zero, the command takes an arbitrary number of arguments (but at least _min_).
///
/// _argsig_ is the argument signature, to be appended to the command name for inclusion
/// in the error message.
///
/// ## Example
///
/// Here are a couple of examples from the Molt code base.  The relevant commands are
/// documented in the Molt Book.
///
/// First, here is the call from the definition of the `set` command, which has the signature
/// `set varName ?newValue?`.  In TCL command signatures, question marks denote optional
/// values.  The first argument is the command name, and the _argv_ array must be at least 2
/// arguments in length but no more than 3.
///
/// ```ignore
/// check_args(1, argv, 2, 3, "varName ?newValue?")?;
/// ```
///
/// Next, here the call from the definition of the `append` command, which appends strings to
/// the content of a variable.  It has signature `append varName ?value value ...?`.  The first
/// argument is the command name, and
/// the second is the variable name to which data will be appended.  The remaining arguments
/// are string values to append; the question marks indicate that they are optional, and the
/// ellipsis indicates that there can be any number of them.
///
/// ```ignore
/// check_args(1, argv, 2, 0, "varName ?value value ...?")?;
/// ```
pub fn check_args(
    namec: usize,
    argv: &[Value],
    min: usize,
    max: usize,
    argsig: &str,
) -> MoltResult {
    assert!(namec >= 1);
    assert!(min >= 1);
    assert!(!argv.is_empty());

    if argv.len() < min || (max > 0 && argv.len() > max) {
        let cmd_tokens = Value::from(&argv[0..namec]);
        molt_err!(
            "wrong # args: should be \"{} {}\"",
            cmd_tokens.to_string(),
            argsig
        )
    } else {
        molt_ok!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_args() {
        assert_ok(&check_args(1, &mklist(vec!["mycmd"].as_slice()), 1, 1, ""));
        assert_ok(&check_args(
            1,
            &mklist(vec!["mycmd"].as_slice()),
            1,
            2,
            "arg1",
        ));
        assert_ok(&check_args(
            1,
            &mklist(vec!["mycmd", "data"].as_slice()),
            1,
            2,
            "arg1",
        ));
        assert_ok(&check_args(
            1,
            &mklist(vec!["mycmd", "data", "data2"].as_slice()),
            1,
            0,
            "arg1",
        ));

        assert_err(
            &check_args(1, &mklist(vec!["mycmd"].as_slice()), 2, 2, "arg1"),
            "wrong # args: should be \"mycmd arg1\"",
        );
        assert_err(
            &check_args(
                1,
                &mklist(vec!["mycmd", "val1", "val2"].as_slice()),
                2,
                2,
                "arg1",
            ),
            "wrong # args: should be \"mycmd arg1\"",
        );
    }

    // TODO: stopgap until we have finalized the MoltList API.
    fn mklist(argv: &[&str]) -> MoltList {
        argv.iter().map(|s| Value::from(*s)).collect()
    }

    // Helpers

    fn assert_err(result: &MoltResult, msg: &str) {
        assert_eq!(molt_err!(msg), *result);
    }

    fn assert_ok(result: &MoltResult) {
        assert!(result.is_ok(), "Result is not Ok");
    }
}
