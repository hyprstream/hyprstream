//! RustPython guest for the #505 capability-sandbox build-viability spike.
//!
//! Compiled to `wasm32-unknown-unknown` (NOT wasi). There is deliberately NO WASI
//! import in this module: the only thing the guest can reach outside its linear
//! memory is the single host import `host_random`. `import os; os.system(...)`
//! therefore has nothing to call — it is inert by construction.
//!
//! Build:
//!   cargo build --target wasm32-unknown-unknown -p hyprstream-wasm-pyguest
//!
//! Exports:
//!   - `eval(ptr, len) -> i32`  : run UTF-8 python source, 0 = ok, nonzero = py error
//!   - `alloc(len) -> *mut u8`  : let the host allocate guest memory to write source into
//!   - `dealloc(ptr, len)`      : free it again

use rustpython_vm as vm;

// ---------------------------------------------------------------------------
// getrandom 0.2 custom backend -> host import.
//
// rustpython-vm pulls rand 0.8 -> rand_core -> getrandom 0.2. On
// wasm32-unknown-unknown getrandom 0.2 has no default backend, so without this
// any call to getrandom (e.g. seeding the hash randomization / `random` module)
// would be a link error. We register a custom backend that delegates to a host
// function `host_random`, which the wasmtime host (crate `hyprstream-wasm`) fills
// from its OWN entropy source. The guest never has direct OS entropy access.
// ---------------------------------------------------------------------------
extern "C" {
    /// Host import: fill `len` bytes at guest `ptr` with random data.
    /// Provided by the wasmtime `Linker` in the host crate.
    fn host_random(ptr: *mut u8, len: usize);
}

/// Shared backend: fill `buf` from the host import.
fn fill_from_host(buf: &mut [u8]) {
    // SAFETY: `buf` is a valid, exclusively-borrowed slice for `buf.len()` bytes.
    // The host writes exactly `len` bytes into that range.
    unsafe {
        host_random(buf.as_mut_ptr(), buf.len());
    }
}

// --- getrandom 0.2 custom backend (macro + `custom` feature) ---
fn custom_getrandom02(buf: &mut [u8]) -> Result<(), getrandom02::Error> {
    fill_from_host(buf);
    Ok(())
}
getrandom02::register_custom_getrandom!(custom_getrandom02);

// --- getrandom 0.3 custom backend (cfg getrandom_backend="custom" + extern symbol) ---
// getrandom 0.3 looks up `__getrandom_v03_custom` by symbol when the build cfg is set.
// Signature is `fn(*mut u8, usize) -> Result<(), getrandom03::Error>`.
#[no_mangle]
unsafe fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom03::Error> {
    let buf = std::slice::from_raw_parts_mut(dest, len);
    fill_from_host(buf);
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
    let interp = vm::Interpreter::without_stdlib(Default::default());

    interp.enter(|vm| {
        let scope = vm.new_scope_with_builtins();
        let code = match vm
            .compile(src, vm::compiler::Mode::Exec, "<guest>".to_owned())
        {
            Ok(c) => c,
            Err(_) => return 3, // compile error
        };
        match vm.run_code_obj(code, scope) {
            Ok(_) => 0,
            Err(_) => 1, // runtime python exception (e.g. os.system unsupported)
        }
    })
}
