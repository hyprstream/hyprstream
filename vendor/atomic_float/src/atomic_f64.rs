use core::cell::UnsafeCell;
use core::sync::atomic::{
    AtomicU64,
    Ordering::{self, *},
};

/// A floating point type which can be safely shared between threads.
///
/// This type has the same in-memory representation as the underlying floating
/// point type, [`f64`].
///
/// See the module documentation for [core::sync::atomic] for information about
/// the portability of various atomics (this one is mostly as portable as
/// [`AtomicU64`](core::sync::atomic::AtomicU64), with the caveat that we
/// additionally require that the platform support 64-bit floats).
///
/// # Example
///
/// The intended use case is situations like this:
///
/// ```
/// # use atomic_float::AtomicF64;
/// # use std::sync::atomic::Ordering;
/// static DELTA_TIME: AtomicF64 = AtomicF64::new(1.0);
///
/// // In some main simulation loop:
/// # fn compute_delta_time() -> f64 { 1.0 / 60.0 }
/// DELTA_TIME.store(compute_delta_time(), Ordering::Release);
///
/// // elsewhere, perhaps on other threads:
/// let dt = DELTA_TIME.load(Ordering::Acquire);
/// // Use `dt` to compute simulation...
/// ```
///
/// As well as any other cases where use of locking would be too substantial of
/// overhead.
///
/// Note that when used like this (with Acquire and Release orderings), on
/// x86_64 this compiles to the same code as you would get from a C++ global,
/// (or a Rust `static mut`), while offering full synchronization.
///
/// (While caveats exist, the cases where you'd need the total order guaranteed
/// by [SeqCst](core::sync::atomic::Ordering::SeqCst) for something like
/// [`AtomicF64`] seem very rare).
///
/// # Implementation
///
/// Note: These details are not part of the stability guarantee of this crate,
/// and are subject to change without a semver-breaking change.
///
/// Under the hood we use a transparent `UnsafeCell<f64>`, and cast the
/// `&UnsafeCell<f64>` to an [`&AtomicU64`](core::sync::atomic::AtomicU64) in
/// order to perform atomic operations.
///
/// However, operations like [`fetch_add`](AtomicF64::fetch_add) are
/// considerably slower than would be the case for integer atomics.
#[cfg_attr(target_arch = "x86", repr(C, align(8)))]
#[cfg_attr(not(target_arch = "x86"), repr(transparent))]
pub struct AtomicF64(
    // FIXME: Once we can do `f32::from_bits` in const fn, this should be an
    // `AtomicU64` (or at least `UnsafeCell<u64>`).
    UnsafeCell<f64>,
);

// SAFETY: We only ever access the underlying data by refcasting to AtomicU64,
// which guarantees no data races.
unsafe impl Send for AtomicF64 {}
unsafe impl Sync for AtomicF64 {}

// Static assertions that the layout is identical, we cite these in a safety
// comment in `AtomicF64::atom()`. This is possible on some targets (like 32-bit
// x86, which has different alignments for `AtomicU64` and `u64` for example),
// so please file a bug for your target if you hit these.
const _: [(); core::mem::size_of::<AtomicU64>()] = [(); core::mem::size_of::<AtomicF64>()];
const _: [(); 1] =
    [(); (core::mem::align_of::<AtomicF64>() >= core::mem::align_of::<AtomicU64>()) as usize];

impl AtomicF64 {
    /// Initialize a `AtomicF64` from an `f64`.
    ///
    /// # Example
    ///
    /// Use as a variable
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::Relaxed;
    /// let x = AtomicF64::new(3.0f64);
    /// assert_eq!(x.load(Relaxed), 3.0f64);
    /// ```
    ///
    /// Use as a static:
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::Relaxed;
    /// static A_STATIC: AtomicF64 = AtomicF64::new(800.0);
    /// assert_eq!(A_STATIC.load(Relaxed), 800.0);
    /// ```
    #[inline]
    pub const fn new(float: f64) -> Self {
        Self(UnsafeCell::new(float))
    }

    /// Returns a mutable reference to the underlying float.
    ///
    /// This is safe because the mutable reference guarantees that no other
    /// threads are concurrently accessing the atomic data.
    ///
    /// # Example
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering;
    /// let mut some_float = AtomicF64::new(1.0);
    /// assert_eq!(*some_float.get_mut(), 1.0);
    /// *some_float.get_mut() += 1.0;
    /// assert_eq!(some_float.load(Ordering::SeqCst), 2.0);
    /// ```
    #[inline]
    pub fn get_mut(&mut self) -> &mut f64 {
        // SAFETY: the mutable reference guarantees unique ownership.
        unsafe { &mut *self.0.get() }
    }

    /// Consumes the atomic and returns the contained value.
    ///
    /// This is safe because passing `self` by value guarantees that no other
    /// threads are concurrently accessing the atomic data.
    ///
    /// # Example
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// let v = AtomicF64::new(6.0);
    /// assert_eq!(v.into_inner(), 6.0f64);
    /// ```
    #[inline]
    pub fn into_inner(self) -> f64 {
        self.0.into_inner()
    }

    /// Loads a value from the atomic float.
    ///
    /// `load` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. Possible values are [`SeqCst`], [`Acquire`]
    /// and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is [`Release`] or [`AcqRel`].
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    /// [`Relaxed`]: core::sync::atomic::Ordering::Relaxed
    /// [`Release`]: core::sync::atomic::Ordering::Release
    /// [`Acquire`]: core::sync::atomic::Ordering::Acquire
    /// [`AcqRel`]: core::sync::atomic::Ordering::AcqRel
    /// [`SeqCst`]: core::sync::atomic::Ordering::SeqCst
    ///
    /// # Example
    ///
    /// ```
    /// use atomic_float::AtomicF64;
    /// use std::sync::atomic::Ordering;
    /// let v = AtomicF64::new(22.5);
    /// assert_eq!(v.load(Ordering::SeqCst), 22.5);
    /// ```
    #[inline]
    pub fn load(&self, ordering: Ordering) -> f64 {
        f64::from_bits(self.as_atomic_bits().load(ordering))
    }

    /// Store a value into the atomic float.
    ///
    /// `store` takes an [`Ordering`] argument which describes the memory
    /// ordering of this operation. Possible values are [`SeqCst`], [`Release`]
    /// and [`Relaxed`].
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is [`Acquire`] or [`AcqRel`].
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    /// [`Relaxed`]: core::sync::atomic::Ordering::Relaxed
    /// [`Release`]: core::sync::atomic::Ordering::Release
    /// [`Acquire`]: core::sync::atomic::Ordering::Acquire
    /// [`AcqRel`]: core::sync::atomic::Ordering::AcqRel
    /// [`SeqCst`]: core::sync::atomic::Ordering::SeqCst
    ///
    /// # Example
    ///
    /// ```
    /// use atomic_float::AtomicF64;
    /// use std::sync::atomic::Ordering;
    /// let v = AtomicF64::new(22.5);
    /// assert_eq!(v.load(Ordering::SeqCst), 22.5);
    /// v.store(30.0, Ordering::SeqCst);
    /// assert_eq!(v.load(Ordering::SeqCst), 30.0);
    /// ```
    #[inline]
    pub fn store(&self, value: f64, ordering: Ordering) {
        self.as_atomic_bits().store(value.to_bits(), ordering);
    }

    /// Stores a value into the atomic float, returning the previous value.
    ///
    /// `swap` takes an [`Ordering`] argument which describes the memory ordering
    /// of this operation. All ordering modes are possible. Note that using
    /// [`Acquire`] makes the store part of this operation [`Relaxed`], and
    /// using [`Release`] makes the load part [`Relaxed`].
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    /// [`Relaxed`]: core::sync::atomic::Ordering::Relaxed
    /// [`Release`]: core::sync::atomic::Ordering::Release
    /// [`Acquire`]: core::sync::atomic::Ordering::Acquire
    ///
    /// # Example
    ///
    /// ```
    /// use atomic_float::AtomicF64;
    /// use std::sync::atomic::Ordering;
    /// let v = AtomicF64::new(4.5);
    /// assert_eq!(v.swap(100.0, Ordering::Relaxed), 4.5);
    /// assert_eq!(v.load(Ordering::Relaxed), 100.0);
    /// ```
    #[inline]
    pub fn swap(&self, new_value: f64, ordering: Ordering) -> f64 {
        f64::from_bits(self.as_atomic_bits().swap(new_value.to_bits(), ordering))
    }

    /// Stores a value into the atomic float if the current value is *bitwise
    /// identical* to the `current` value.
    ///
    /// The return value is always the previous value. If it is equal to
    /// `current`, then the value was updated.
    ///
    /// `compare_and_swap` also takes an [`Ordering`] argument which describes
    /// the memory ordering of this operation. Notice that even when using
    /// `AcqRel`, the operation might fail and hence just perform an `Acquire`
    /// load, but not have `Release` semantics. Using `Acquire` makes the store
    /// part of this operation `Relaxed` if it happens, and using `Release`
    /// makes the load part `Relaxed`.
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    ///
    /// # Caveats
    ///
    /// As the `current` must be bitwise identical to the previous value, you
    /// should not get the `current` value using any sort of arithmetic (both
    /// because of rounding, and to avoid any situation where -0.0 and +0.0
    /// would be compared).  Additionally, on some platforms (WASM and ASM.js
    /// currently) LLVM will canonicalize NaNs during loads, which can cause
    /// unexpected behavior here — typically in the other direction (two values
    /// being unexpectedly equal).
    ///
    /// In practice, typical patterns for CaS tend to avoid these issues, but
    /// you're encouraged to avoid relying on the behavior of cas-family APIs in
    /// the face of rounding, signed zero, and NaNs.
    ///
    /// # Example
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering;
    /// let v = AtomicF64::new(5.0);
    /// assert_eq!(v.compare_and_swap(5.0, 10.0, Ordering::Relaxed), 5.0);
    /// assert_eq!(v.load(Ordering::Relaxed), 10.0);
    ///
    /// assert_eq!(v.compare_and_swap(6.0, 12.0, Ordering::Relaxed), 10.0);
    /// assert_eq!(v.load(Ordering::Relaxed), 10.0);
    /// ```
    #[inline]
    #[allow(deprecated)]
    pub fn compare_and_swap(&self, current: f64, new: f64, order: Ordering) -> f64 {
        f64::from_bits(self.as_atomic_bits().compare_and_swap(
            current.to_bits(),
            new.to_bits(),
            order,
        ))
    }

    /// Stores a value into the atomic float if the current value is the bitwise
    /// identical as the `current` value.
    ///
    /// The return value is a result indicating whether the new value was
    /// written and containing the previous value. On success this value is
    /// guaranteed to be bitwise identical to `current`.
    ///
    /// `compare_exchange` takes two [`Ordering`] arguments to describe the
    /// memory ordering of this operation. The first describes the required
    /// ordering if the operation succeeds while the second describes the
    /// required ordering when the operation fails. Using `Acquire` as success
    /// ordering makes the store part of this operation `Relaxed`, and using
    /// `Release` makes the successful load `Relaxed`. The failure ordering can
    /// only be `SeqCst`, `Acquire` or `Relaxed` and must be equivalent to or
    /// weaker than the success ordering.
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    ///
    /// # Notes
    ///
    /// Note that in many cases, when `while` loops where the condition contains
    /// a `compare_exchange` operation are better written to use a
    /// [`compare_exchange_weak`](AtomicF64::compare_exchange_weak) in the
    /// condition instead (as on weakly ordered platforms like ARM, the
    /// `compare_exchange` operation itself can require a loop to perform).
    ///
    /// ## Caveats
    ///
    /// As the `current` parameter must be bitwise identical to the previous
    /// value, you should not get the `current` value using any sort of
    /// arithmetic (both because of rounding, and to avoid any situation where
    /// -0.0 and +0.0 would be compared). Additionally, on Wasm, in some cases
    /// `NaN` values have been known to cause problems for non-typical usage of
    /// this API. See [`AtomicF64::as_atomic_bits`] if performing the
    /// `compare_exchange` on the raw bits of this atomic float would solve an
    /// issue for you.
    ///
    /// # Example
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let v = AtomicF64::new(5.0);
    /// assert_eq!(v.compare_exchange(5.0, 10.0, Acquire, Relaxed), Ok(5.0));
    /// assert_eq!(v.load(Relaxed), 10.0);
    ///
    /// assert_eq!(v.compare_exchange(6.0, 12.0, SeqCst, Relaxed), Err(10.0));
    /// assert_eq!(v.load(Relaxed), 10.0);
    /// ```
    #[inline]
    pub fn compare_exchange(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        convert_result(self.as_atomic_bits().compare_exchange(
            current.to_bits(),
            new.to_bits(),
            success,
            failure,
        ))
    }

    /// Stores a value into the atomic integer if the current value is the same
    /// as the `current` value.
    ///
    /// Unlike [`compare_exchange`](Self::compare_exchange), this function is
    /// allowed to spuriously fail even when the comparison succeeds, which can
    /// result in more efficient code on some platforms. The return value is a
    /// result indicating whether the new value was written and containing the
    /// previous value.
    ///
    /// `compare_exchange_weak` takes two [`Ordering`] arguments to describe the
    /// memory ordering of this operation. The first describes the required
    /// ordering if the operation succeeds while the second describes the
    /// required ordering when the operation fails. Using `Acquire` as success
    /// ordering makes the store part of this operation `Relaxed`, and using
    /// `Release` makes the successful load `Relaxed`. The failure ordering can
    /// only be `SeqCst`, `Acquire` or `Relaxed` and must be equivalent to or
    /// weaker than the success ordering.
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    ///
    /// ## Caveats
    ///
    /// As the `current` parameter must be bitwise identical to the previous
    /// value, you should not get the `current` value using any sort of
    /// arithmetic (both because of rounding, and to avoid any situation where
    /// -0.0 and +0.0 would be compared). Additionally, on Wasm, in some cases
    /// `NaN` values have been known to cause problems for non-typical usage of
    /// this API. See [`AtomicF64::as_atomic_bits`] if performing the
    /// `compare_exchange` on the raw bits of this atomic float would solve an
    /// issue for you.
    ///
    /// # Example
    ///
    /// Note that this sort of CaS loop should generally use [`fetch_update`]
    /// instead.
    ///
    /// [`fetch_update`]: AtomicF64::fetch_update
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let v = AtomicF64::new(5.0);
    /// let mut old = v.load(Relaxed);
    /// loop {
    ///     let new = old * 2.0;
    ///     match v.compare_exchange_weak(old, new, SeqCst, Relaxed) {
    ///         Ok(_) => break,
    ///         Err(x) => old = x,
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn compare_exchange_weak(
        &self,
        current: f64,
        new: f64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f64, f64> {
        convert_result(self.as_atomic_bits().compare_exchange_weak(
            current.to_bits(),
            new.to_bits(),
            success,
            failure,
        ))
    }

    /// Fetches the value, and applies a function to it that returns an optional
    /// new value. Returns a `Result` of `Ok(previous_value)` if the function
    /// returned `Some(_)`, else `Err(previous_value)`.
    ///
    /// Note: This may call the function multiple times if the value has been
    /// changed from other threads in the meantime, as long as the function
    /// returns `Some(_)`, but the function will have been applied only once to
    /// the stored value.
    ///
    /// `fetch_update` takes two [`Ordering`] arguments to describe the memory
    /// ordering of this operation. The first describes the required ordering
    /// for when the operation finally succeeds while the second describes the
    /// required ordering for loads. These correspond to the success and failure
    /// orderings of [`compare_exchange`][AtomicF64::compare_exchange]
    /// respectively.
    ///
    /// Using `Acquire` as success ordering makes the store part of this
    /// operation `Relaxed`, and using `Release` makes the final successful load
    /// `Relaxed`. The (failed) load ordering can only be `SeqCst`, `Acquire` or
    /// `Relaxed` and must be equivalent to or weaker than the success ordering.
    ///
    /// [`Ordering`]: core::sync::atomic::Ordering
    ///
    /// # Example
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let x = AtomicF64::new(7.0);
    ///
    /// assert_eq!(x.fetch_update(SeqCst, SeqCst, |_| None), Err(7.0));
    /// assert_eq!(x.fetch_update(SeqCst, SeqCst, |x| Some(x + 1.0)), Ok(7.0));
    /// assert_eq!(x.fetch_update(SeqCst, SeqCst, |x| Some(x + 1.0)), Ok(8.0));
    /// assert_eq!(x.load(SeqCst), 9.0);
    /// ```
    #[inline]
    pub fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        mut update: F,
    ) -> Result<f64, f64>
    where
        F: FnMut(f64) -> Option<f64>,
    {
        let res = self
            .as_atomic_bits()
            .fetch_update(set_order, fetch_order, |prev| {
                update(f64::from_bits(prev)).map(f64::to_bits)
            });
        convert_result(res)
    }
    // Not exposing this unless I can come up with a better name...
    /// A (nonstandard) convenience wrapper around [`fetch_update`](Self::fetch_update).
    ///
    /// A call like:
    ///
    /// ```
    /// # const _: &str = stringify!{
    /// let res = atom.update_with(order, |f| update f...);
    /// # };
    /// ```
    ///
    /// Is morally equivalent to:
    ///
    /// ```
    /// # const _: &str = stringify!{
    /// let res = atom.fetch_update(
    ///     order,
    ///     failure_order_for(order),
    ///     |f| Some(update f...),
    /// ).unwrap();
    /// # };
    /// ```
    ///
    /// Where `failure_order_for` returns the strongest failure order you'd be
    /// allowed to pass into `fetch_update` given the success order, that is:
    ///
    /// ```
    /// # const _: &str = stringify!{
    /// fn failure_order_for(order: Ordering) -> Ordering {
    ///     Release | Relaxed => Relaxed,
    ///     Acquire | AcqRel => Acquire,
    ///     SeqCst => SeqCst,
    /// }
    /// # };
    /// ```
    #[inline]
    fn update_with<F>(&self, order: Ordering, mut update: F) -> f64
    where
        F: FnMut(f64) -> f64,
    {
        self.fetch_update(order, super::fail_order_for(order), |f| Some(update(f)))
            .unwrap()
    }

    /// Adds to the current value, returning the previous value.
    ///
    /// Because this returns the previous value, you may want to call it like:
    /// `atom.fetch_add(x, order) + x`
    ///
    /// # Examples
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let x = AtomicF64::new(7.0);
    ///
    /// assert_eq!(x.fetch_add(2.0, Relaxed), 7.0);
    /// assert_eq!(x.fetch_add(1.0, SeqCst), 9.0);
    /// assert_eq!(x.fetch_add(-100.0, AcqRel), 10.0);
    /// ```
    #[inline]
    pub fn fetch_add(&self, val: f64, order: Ordering) -> f64 {
        self.update_with(order, |f| f + val)
    }

    /// Subtract from the current value, returning the previous value.
    ///
    /// Because this returns the previous value, you may want to call it like:
    /// `atom.fetch_sub(x, order) - x`
    ///
    /// Note: This operation uses [`fetch_update`](Self::fetch_update) under the
    /// hood, and is likely to be slower than the equivalent operation for
    /// atomic integers.
    ///
    /// # Examples
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let x = AtomicF64::new(7.0);
    /// assert_eq!(x.fetch_sub(2.0, Relaxed), 7.0);
    /// assert_eq!(x.fetch_sub(-1.0, SeqCst), 5.0);
    /// assert_eq!(x.fetch_sub(0.5, AcqRel), 6.0);
    /// ```
    #[inline]
    pub fn fetch_sub(&self, val: f64, order: Ordering) -> f64 {
        self.update_with(order, |f| f - val)
    }

    /// Produce the absolute value of the current value, returning the previous
    /// value.
    ///
    /// Because this returns the previous value, you may want to call it like:
    /// `atom.fetch_abs(x, order).abs()`
    ///
    /// # Examples
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let x = AtomicF64::new(-7.0);
    /// assert_eq!(x.fetch_abs(Relaxed), -7.0);
    /// assert_eq!(x.fetch_abs(SeqCst), 7.0);
    /// ```
    #[inline]
    pub fn fetch_abs(&self, order: Ordering) -> f64 {
        f64::from_bits(
            self.as_atomic_bits()
                .fetch_and(0x7fff_ffff_ffff_ffff, order),
        )
    }

    /// Negates the current value, returning the previous value.
    ///
    /// As a result of returning the previous value, you may want to invoke it like:
    /// `-atom.fetch_neg(Relaxed)`.
    ///
    /// # Examples
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering::*;
    /// let x = AtomicF64::new(-7.0);
    /// assert_eq!(x.fetch_neg(Relaxed), -7.0);
    /// assert_eq!(x.fetch_neg(SeqCst), 7.0);
    /// assert_eq!(x.fetch_neg(AcqRel), -7.0);
    /// ```
    #[inline]
    pub fn fetch_neg(&self, order: Ordering) -> f64 {
        f64::from_bits(
            self.as_atomic_bits()
                .fetch_xor(0x8000_0000_0000_0000, order),
        )
    }

    /// Minimum with the current value.
    ///
    /// Finds the minimum of the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value. Because of this, you may want to call it
    /// like: `atom.fetch_min(x, order).min(x)`
    ///
    /// Note: This operation uses [`fetch_update`](Self::fetch_update) under the
    /// hood, and is likely to be slower than the equivalent operation for
    /// atomic integers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering;
    ///
    /// let foo = AtomicF64::new(23.0);
    /// assert_eq!(foo.fetch_min(42.0, Ordering::Relaxed), 23.0);
    /// assert_eq!(foo.load(Ordering::Relaxed), 23.0);
    /// assert_eq!(foo.fetch_min(22.0, Ordering::Relaxed), 23.0);
    /// assert_eq!(foo.load(Ordering::Relaxed), 22.0);
    /// ```
    #[inline]
    pub fn fetch_min(&self, value: f64, order: Ordering) -> f64 {
        self.update_with(order, |f| f.min(value))
    }

    /// Maximum with the current value.
    ///
    /// Finds the maximum of the current value and the argument `val`, and sets
    /// the new value to the result.
    ///
    /// Returns the previous value. Because of this, you may want to call it
    /// like: `atom.fetch_max(x, order).max(x)`
    ///
    /// Note: This operation uses [`fetch_update`](Self::fetch_update) under the
    /// hood, and is likely to be slower than the equivalent operation for
    /// atomic integers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering;
    ///
    /// let foo = AtomicF64::new(23.0);
    /// assert_eq!(foo.fetch_max(22.0, Ordering::Relaxed), 23.0);
    /// assert_eq!(foo.load(Ordering::Relaxed), 23.0);
    /// assert_eq!(foo.fetch_max(42.0, Ordering::Relaxed), 23.0);
    /// assert_eq!(foo.load(Ordering::Relaxed), 42.0);
    /// ```
    #[inline]
    pub fn fetch_max(&self, value: f64, order: Ordering) -> f64 {
        self.update_with(order, |f| f.max(value))
    }

    /// Returns a reference to an atomic integer which can be used to access the
    /// atomic float's underlying bits in a thread safe manner.
    ///
    /// This is essentially a `transmute::<&Self, &AtomicU64>(self)`, and is
    /// zero cost.
    ///
    /// # Motivation
    ///
    /// This is exposed as an escape hatch because of the caveats around the
    /// `AtomicF64` CaS-family APIs ([`compare_and_swap`], [`compare_exchange`],
    /// [`compare_exchange_weak`], ...) and the notion of bitwise identicality
    /// which they require being somewhat problematic for NaNs, especially on
    /// targets like Wasm (see [rust-lang/rust#73328]).
    ///
    /// [`compare_and_swap`]: AtomicU64::compare_and_swap
    /// [`compare_exchange`]: AtomicU64::compare_exchange
    /// [`compare_exchange_weak`]: AtomicU64::compare_exchange_weak
    /// [rust-lang/rust#73328]: https://github.com/rust-lang/rust/issues/73328
    ///
    /// In general despite how bad this might sound, in practice we're fairly
    /// safe: LLVM almost never optimizes through atomic operations, this
    /// library is written to try to avoid potential issues from most naive
    /// usage, and I'm optimistic the situation will clean itself up in the
    /// short-to-medium-term future.
    ///
    /// However, if you need peace of mind, or find yourself in a case where you
    /// suspect you're hitting this issue, you can access the underlying atomic
    /// value using this function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use atomic_float::AtomicF64;
    /// # use std::sync::atomic::Ordering;
    /// let v = AtomicF64::new(22.5);
    /// assert_eq!(v.as_atomic_bits().load(Ordering::Relaxed), 22.5f64.to_bits());
    /// ```
    #[inline]
    pub fn as_atomic_bits(&self) -> &AtomicU64 {
        // Safety: All potentially shared reads/writes go through this, and the
        // static assertions above ensure that AtomicU64 and UnsafeCell<f64> are
        // compatible as pointers.
        unsafe { &*(&self.0 as *const _ as *const AtomicU64) }
    }
}

/// Return a zero-initialized atomic.
///
/// # Example
///
/// ```
/// # use atomic_float::AtomicF64;
/// # use core::sync::atomic::Ordering;
/// let x = AtomicF64::default();
/// assert_eq!(x.load(Ordering::SeqCst), 0.0);
/// ```
impl Default for AtomicF64 {
    #[inline]
    fn default() -> Self {
        Self::from(0.0)
    }
}

/// Equivalent to `<f64 as core::fmt::Debug>::fmt`.
///
/// # Example
///
/// ```
/// # use atomic_float::AtomicF64;
/// # use core::sync::atomic::Ordering;
/// let v = AtomicF64::new(40.0);
/// assert_eq!(format!("{:?}", v), format!("{:?}", 40.0));
/// ```
impl core::fmt::Debug for AtomicF64 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.load(SeqCst).fmt(f)
    }
}

/// Equivalent to `AtomicF64::new`.
///
/// # Example
///
/// ```
/// # use atomic_float::AtomicF64;
/// # use core::sync::atomic::Ordering;
/// let v = AtomicF64::from(10.0);
/// assert_eq!(v.load(Ordering::SeqCst), 10.0);
/// ```
impl From<f64> for AtomicF64 {
    #[inline]
    fn from(f: f64) -> Self {
        Self::new(f)
    }
}

#[cfg(feature = "serde")]
/// Serializes the AtomicF64
///
/// The value is loaded with the `Ordering::SeqCst` and then serializes
/// it as a normal `f64`. The information about the object being atomic is lost.
impl serde::Serialize for AtomicF64 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.load(Ordering::SeqCst))
    }
}

#[cfg(feature = "serde")]
/// Deserializes the AtomicF64
///
/// Attempts to deserialize f64 and, if successful, creates a new
/// AtomicF64 with this value.
impl<'de> serde::Deserialize<'de> for AtomicF64 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        f64::deserialize(deserializer).map(AtomicF64::new)
    }
}

#[inline(always)]
fn convert_result(r: Result<u64, u64>) -> Result<f64, f64> {
    r.map(f64::from_bits).map_err(f64::from_bits)
}

// XXX: This is dubious since the actual atomic types don't implement this, but
// I need it to use `serde_test`, so I might as well add it.
/// Compare two [`AtomicF64`]s.
///
/// ```
/// # use atomic_float::AtomicF64;
/// # use std::sync::atomic::Ordering;
/// let a = AtomicF64::new(1.0);
/// a.fetch_add(1.0, Ordering::Relaxed);
/// assert_ne!(a, AtomicF64::new(1.0));
/// assert_eq!(a, AtomicF64::new(2.0));
/// ```
///
/// # Caveats
/// Relaxed ordering is used for each load, so additional fencing (or avoiding
/// the use of this `PartialEq` implementation) may be desirable.
///
/// Additionally, this is implemented in terms of `f64`'s `PartialEq`, so NaNs
/// will compare as inequal. For example:
/// ```
/// # use atomic_float::AtomicF64;
/// let a = AtomicF64::new(f64::NAN);
/// assert_ne!(a, a);
/// ```
impl PartialEq for AtomicF64 {
    #[inline]
    fn eq(&self, o: &AtomicF64) -> bool {
        self.load(Relaxed) == o.load(Relaxed)
    }
}
