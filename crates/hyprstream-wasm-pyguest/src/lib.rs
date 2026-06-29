//! RustPython guest for the #505 capability sandbox (P1 host).
//!
//! Compiled to `wasm32-unknown-unknown` (NOT wasi). There is deliberately NO WASI
//! import in this module: the only thing the guest can reach outside its linear
//! memory is the single host import `host_random`. `import os; os.system(...)`
//! therefore has nothing to call — it is inert by construction.
//!
//! Build:
//!   cargo build --target wasm32-unknown-unknown --manifest-path crates/hyprstream-wasm-pyguest/Cargo.toml
//!
//! Exports:
//!   - `eval(ptr, len) -> i32`  : run UTF-8 python source, 0 = ok, nonzero = py error
//!   - `alloc(len) -> *mut u8`  : let the host allocate guest memory to write source into
//!   - `dealloc(ptr, len)`      : free it again
//!
//! P1 changes vs. the P0 spike:
//!   * rustpython-vm bumped 0.3 -> 0.5 (0.3.1 force-enabled getrandom/js, which
//!     out-ranked the custom backend and trapped at VM bootstrap reaching for
//!     undefined `__wbindgen_placeholder__` JS imports). 0.5 uses getrandom 0.3
//!     with no `js`, so the custom backend is actually reachable.
//!   * Dropped the getrandom-0.2 dual backend; ONLY the getrandom-0.3 custom
//!     backend remains.
//!   * Construct via `Interpreter::builder(Settings).add_frozen_modules(...)` rather
//!     than `Interpreter::without_stdlib` (which panics under freeze-stdlib in 0.5
//!     because it registers an EMPTY frozen set, and essential init then fails to
//!     `import encodings`).

use rustpython_vm as vm;

// ---------------------------------------------------------------------------
// getrandom 0.3 custom backend -> host import.
//
// rustpython-vm 0.5 pulls getrandom 0.3 (no `js` feature). On
// wasm32-unknown-unknown getrandom 0.3 has no default backend, so the build cfg
// `getrandom_backend="custom"` (see .cargo/config.toml) selects an external symbol
// `__getrandom_v03_custom`, which we provide here and which delegates to a host
// function `host_random`. The wasmtime host (crate `hyprstream-wasm`) fills it from
// its OWN entropy source. The guest never has direct OS entropy access.
// ---------------------------------------------------------------------------
extern "C" {
    /// Host import: fill `len` bytes at guest `ptr` with random data.
    /// Provided by the wasmtime `Linker` in the host crate.
    fn host_random(ptr: *mut u8, len: usize);
}

/// getrandom 0.3 custom backend: getrandom looks this symbol up by name when the
/// build cfg `getrandom_backend="custom"` is set. Signature is fixed by getrandom 0.3:
/// `fn(*mut u8, usize) -> Result<(), getrandom::Error>`.
///
/// # Safety
/// `dest`..`dest+len` must be a valid, exclusively-borrowable region; getrandom
/// guarantees this for the buffer it passes.
#[no_mangle]
unsafe fn __getrandom_v03_custom(dest: *mut u8, len: usize) -> Result<(), getrandom::Error> {
    // SAFETY: getrandom 0.3 passes a valid writable `dest`/`len`; the host writes
    // exactly `len` bytes into that range.
    let buf = std::slice::from_raw_parts_mut(dest, len);
    host_random(buf.as_mut_ptr(), buf.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory ABI for the host to ship source code into guest linear memory.
// ---------------------------------------------------------------------------

/// Allocate `len` bytes in guest memory and return the pointer.
///
/// # Safety
/// The host must later call `dealloc` with the same `ptr`/`len`.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Free memory previously returned by [`alloc`].
///
/// # Safety
/// `ptr`/`len` must come from a prior `alloc` call.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        drop(Vec::from_raw_parts(ptr, 0, len));
    }
}

/// Build a fresh interpreter WITH the frozen stdlib registered.
///
/// rustpython-vm 0.5: `Interpreter::without_stdlib` registers an empty frozen set,
/// so under the `freeze-stdlib` feature `VirtualMachine::initialize` panics
/// ("essential initialization failed") trying to `import encodings`. The supported
/// embedding path is the builder, registering `rustpython_pylib::FROZEN_STDLIB`
/// (the bytecode-frozen pure-Python stdlib baked into the binary — no filesystem
/// import path, a capability win).
fn build_interpreter() -> vm::Interpreter {
    vm::Interpreter::builder(vm::Settings::default())
        .add_frozen_modules(rustpython_pylib::FROZEN_STDLIB)
        .build()
}

/// Interpret UTF-8 Python source in a fresh interpreter.
///
/// Returns 0 on success, nonzero on any Python or decode error.
///
/// # Safety
/// `ptr`..`ptr+len` must be a valid readable region of guest memory holding
/// UTF-8 source (typically written via [`alloc`]).
#[no_mangle]
pub unsafe extern "C" fn eval(ptr: *const u8, len: usize) -> i32 {
    let src = match std::str::from_utf8(std::slice::from_raw_parts(ptr, len)) {
        Ok(s) => s,
        Err(_) => return 2, // not valid UTF-8
    };

    // Fresh interpreter per call: no shared mutable state across evaluations.
    let interp = build_interpreter();

    interp.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let code = match vm.compile(src, vm::compiler::Mode::Exec, "<guest>".to_owned()) {
            Ok(c) => c,
            Err(_) => return 3, // compile error
        };
        match vm.run_code_obj(code, scope) {
            Ok(_) => 0,
            Err(_) => 1, // runtime python exception (e.g. os.system unsupported)
        }
    })
}
