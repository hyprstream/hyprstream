Performance & Memory Model

    Optimize for zero-copy where possible (minimize unnecessary data copying).
    Leverage SIMD instructions (e.g., via std::arch or other crates) for performance-critical loops.
    Validate types rigorously—avoid conversions that can silently lose data or create alignment issues.

Async + Concurrency

    Prioritize async features (e.g., tokio, async-std) over an OOP approach when concurrency is required.
    Ensure thread-safety: use Send + Sync bounds where appropriate. Avoid data races or shared mutable state without synchronization.

Interfacing with DuckDB & Hypr Types*

    Maintain clear distinctions between Hypr* types and DuckDB types:
        Provide naming patterns or modules so it’s immediately obvious which type belongs to which subsystem.
    Protect against SQL injections: use parameterized queries, statement bindings, or safe builder patterns for all DuckDB interactions.
    If bridging between Hyprstream and DuckDB, handle type conversions carefully (e.g., numeric boundaries, string encoding).

adbc_core Integration

    Remember that adbc_core uses underscores in its modules/types.
    Allow Arrow versions to differ if necessary; perform conversions with DRY (Don’t Repeat Yourself) principles—centralize the logic for Arrow array or schema conversions.
    The types adbc_core::{Connection, Database, Driver, Statement} are not dyn-compatible—avoid dynamic dispatch with these.
    Reference the adbc_core type docs: https://docs.rs/adbc_core/latest/src/adbc_core/ffi/types.rs.html

Error Handling & Safety

    Use robust error handling (e.g., Result<T, E> or custom error types). Provide clear error messages for debugging.
    Ensure type-safety with trait bounds or newtypes as needed. Avoid insecure conversions or unwrapping external data unsafely.
    SQL Injection defenses (reiterated): always sanitize/parameterize queries.

Code Organization & Style

    Follow Rust idioms (e.g., modules, naming conventions, cargo fmt, cargo clippy).
    Use DRY—avoid duplicate logic, especially for repeated data conversions or schema definitions.
    Prefer functional or trait-based designs over heavy inheritance; Rust is not an OOP language.

Documentation & Testing

    Add doc-comments (///) for public functions, types, and modules.
    Where possible, provide examples (doc tests) to illustrate usage of new APIs.
    Write unit tests and consider concurrency tests to ensure thread-safe logic under load.

Additional Reminders

    Zero-copy also applies to passing references or slices where possible (rather than cloning).
    Keep SIMD usage behind feature flags if it introduces portability issues (e.g., certain CPU instruction sets).
    Use type conversions mindfully for Arrow arrays: ensure no data corruption between versions if we must handle multiple Arrow versions.

Tracing and observability

Use the trace crate for logging

Use safe_arch on x86/x86_64

    Ensure all SIMD code is protected by cfg blocks, such as #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] to modules or functions importing safe_arch for SSE/AVX intrinsics.

Use packed_simd on Other Architectures

    For non-x86 targets (#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]), import packed_simd for SIMD operations.

Unified API

    Provide a consistent, architecture-agnostic API by re-exporting or aliasing your SIMD functions in a single module. This helps avoid scattered cfg logic throughout the code.

Implementation Example

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod simd_x86 {
    use safe_arch::*; // SSE/AVX intrinsics
    // ...
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod simd_other {
    use packed_simd::*; // For ARM, etc.
    // ...
}

We can use the arrow_convert crate to convert types but pass-thru conversions perform a single memory copy. Deserialization performs a copy from arrow to the destination. Serialization performs a copy from the source to arrow. In-place deserialization is theoretically possible but currently not supported by arrow_convert. Minimize the use of this crate where possible to ensure the use of zero-copy.

There is no use arrow_convert::TryIntoArrow, the correct scope arrow_convert::serialize::TryIntoArrow;

Develop and maintain tests with continous testing

Following code changes, run `cargo check` to verify syntax and type safety. Run `cargo test` only after a successful `cargo check`. The project is ready when the project successfully implements HYPRSTREAM_PAPER_DRAFT.md, has tests for its functionality, and passes these tests.]

GPU Acceleration

Support GPU acceleration using the burn crate. Allow configuration of burn settings such as selection of a backend (including but not limited to: libtorch, cuda, candle, ndarray, autodiff)

The zerocopy crate would provide significant benefits for improving memory safety and performance in the codebase:

In storage/zerocopy.rs:
Replace unsafe ModelWeightArray implementation with zerocopy's derive macros for safe memory layout
Use AsBytes and FromBytes traits for zero-copy buffer access
Add compile-time layout verification instead of runtime checks
In storage/duckdb.rs and storage/adbc.rs:
Use zerocopy for safer data conversion between Arrow and database formats
Implement zero-copy serialization for batch operations
Add compile-time guarantees for memory layouts
Performance Benefits:
Reduced runtime overhead by moving checks to compile-time
More efficient memory access with verified layouts
Safer zero-copy operations with Arrow data
Safety Improvements:
Elimination of unsafe blocks in ModelWeightArray
Compile-time verification of memory layouts
Type-safe buffer access
Since zerocopy is already a dependency, these improvements can be implemented incrementally without major architectural changes. The main work would be refactoring the ModelWeightArray implementation and data conversion code to leverage zerocopy's type system and traits.

