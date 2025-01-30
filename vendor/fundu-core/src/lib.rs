// Copyright (c) 2023 Joining7943 <joining@posteo.de>
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc(test(attr(warn(unused))))]
#![doc(test(attr(allow(unused_extern_crates))))]
#![warn(clippy::pedantic)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::else_if_without_else)]
#![warn(clippy::fn_to_numeric_cast_any)]
#![warn(clippy::get_unwrap)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::mixed_read_write_in_expression)]
#![warn(clippy::partial_pub_fields)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::todo)]
#![warn(clippy::try_err)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unneeded_field_pattern)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]

pub mod config;
pub mod error;
pub mod parse;
pub mod time;
pub mod util;

#[cfg(test)]
pub use rstest_reuse;

#[cfg(test)]
mod tests {
    use crate::config::Delimiter;
    use crate::error::{ParseError, TryFromDurationError};
    use crate::time::{Duration, Multiplier, TimeUnit};

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Delimiter>();

        assert_send::<TimeUnit>();
        assert_send::<Duration>();
        assert_send::<Multiplier>();

        assert_send::<ParseError>();
        assert_send::<TryFromDurationError>();
    }

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<Delimiter>();

        assert_sync::<TimeUnit>();
        assert_sync::<Duration>();
        assert_sync::<Multiplier>();

        assert_sync::<ParseError>();
        assert_sync::<TryFromDurationError>();
    }
}
