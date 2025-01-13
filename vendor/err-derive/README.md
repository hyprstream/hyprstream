# err-derive

[![Build status](https://img.shields.io/gitlab/pipeline/torkleyy/err-derive.svg?style=flat-square)](https://gitlab.com/torkleyy/err-derive)
[![Crates.io](https://img.shields.io/crates/v/err-derive.svg?style=flat-square)](https://crates.io/crates/err-derive)
[![API Docs](https://img.shields.io/badge/API-on%20docs.rs-blue.svg?style=flat-square)](https://docs.rs/err-derive)

A failure-like derive macro for the std `Error`.
The source code is mostly copied from `failure-derive`.

Minimum Rust version: 1.34.0

## Motivation

Why yet another error handling library? There already are dozen others, the most popular being:

* [failure](https://docs.rs/failure/0.1.3/failure/)
* [error-chain](https://docs.rs/error-chain/0.12.0/error_chain/)

The former provides a nice `#[derive(Fail)]` macro, but it uses its own error type (`Fail`) and its usage is rather discouraged since `std::error::Error`
is getting fixed to provide the same benefits as `Fail`.

`error-chain` does support `std::error::Error`, but it uses a declarative for generating the error implementation which makes the syntax too obscure
in my opinion.

This crate tries to combine both advantages:

* work with `std::error::Error`
* provide a custom `#[derive(Error)]` that works just like `failure-derive`

`err-derive` is compatible with `std`, including the recent change to deprecate `Error::cause` in favour of `Error::source`,
and provides an easy syntax for generating the `Display` and `Error` boilerplate (the latter being 99% copied from `failure-derive`).

## Features

`err-derive` can be applied to your error struct / enum and does the following for you:

* Derive `Display` implementation
* Derive `Error` implementation (implementing `source` to return the cause of the error)
* Derive `From<OtherError>` implementations

## Usage

Cargo.toml:

```toml
[dependencies]
err-derive = "0.1"
```

Rust code:

```rust,no_run
#[cfg(feature = "std")]
use std::error::Error;
#[cfg(not(feature = "std"))]
use std::fmt::Display;
use std::path::PathBuf;

use err_derive::Error;

#[derive(Debug, Error)]
pub enum FormatError {
    #[error(display = "invalid header (expected: {:?}, got: {:?})", expected, found)]
    InvalidHeader {
        expected: String,
        found: String,
    },
    #[error(display = "missing attribute: {:?}", _0)]
    MissingAttribute(String),
}

#[derive(Debug, Error)]
pub enum LoadingError {
    #[error(display = "could not decode file")]
    FormatError(#[error(source)] FormatError),
    #[error(display = "could not find file: {:?}", path)]
    NotFound { path: PathBuf },
}

fn main() {
    let my_error: LoadingError = FormatError::MissingAttribute("some_attr".to_owned()).into();

    print_error(&my_error);
}

#[cfg(feature = "std")]
fn print_error(e: &dyn Error) {
    eprintln!("error: {}", e);
    let mut cause = e.source();
    while let Some(e) = cause {
        eprintln!("caused by: {}", e);
        cause = e.source();
    }
}

#[cfg(not(feature = "std"))]
fn print_error(e: &dyn Display) {
    eprintln!("error: {}", e);
}
```

## no_std

You can use this library in your `#![no_std]` projects by disabling the default `std` feature.

```toml
[dependencies]
err-derive = { version = "...", default-features = false }
```

Without the default `std` feature, only the `From` and `Display` implementations are derived, as `Error`
requires `std`. 

## Credit

Credit goes to [@withoutboats](https://github.com/withoutboats) and
[other contributors of `failure`](https://github.com/rust-lang-nursery/failure/graphs/contributors).

## License

This project is dual-licensed under Apache-2.0 / MIT. You're free to choose one of both licenses.
Every contribution made to this project is assumed to be licensed according to these terms.

See LICENSE, [LICENSE-MIT](LICENSE-MIT) and [LICENSE-APACHE](LICENSE-APACHE) for more information.
