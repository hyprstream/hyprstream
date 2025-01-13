# `atomic_float`

[![Build Status](https://github.com/thomcc/atomic_float/workflows/CI/badge.svg)](https://github.com/thomcc/atomic_float/actions)
[![codecov](https://codecov.io/gh/thomcc/atomic_float/branch/main/graph/badge.svg)](https://codecov.io/gh/thomcc/atomic_float)
[![Docs](https://docs.rs/atomic_float/badge.svg)](https://docs.rs/atomic_float)
[![Latest Version](https://img.shields.io/crates/v/atomic_float.svg)](https://crates.io/crates/atomic_float)

This crate provides `AtomicF32` and `AtomicF64` types that behave almost identically to the integer atomics in the stdlib.

## Usage

```rust
use atomic_float::AtomicF32;
use core::sync::atomic::Ordering::Relaxed;

static A_STATIC: AtomicF32 = AtomicF32::new(800.0);

// Should support the full std::sync::atomic::AtomicFoo API
A_STATIC.fetch_add(30.0, Relaxed);
A_STATIC.fetch_sub(-55.0, Relaxed);
// But also supports things that can be implemented
// efficiently easily, like sign-bit operations.
A_STATIC.fetch_neg(Relaxed);

assert_eq!(A_STATIC.load(Relaxed), -885.0);
```

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](./LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](./LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Public domain, as explained by the Unlicense ([UNLICENSE](./UNLICENSE) or http://opensource.org/licenses/Unlicense)

at your option.
