//! This crate provides [`AtomicF32`] and [`AtomicF64`] types. They're
//! implemented on top of `AtomicU32` and `AtomicU64` respectively.
//!
//! ```
//! # use atomic_float::AtomicF32;
//! # use std::sync::atomic::Ordering;
//! static DELTA_TIME: AtomicF32 = AtomicF32::new(1.0);
//!
//! // In some main simulation loop:
//! # fn compute_delta_time() -> f32 { 1.0 / 60.0 }
//! DELTA_TIME.store(compute_delta_time(), Ordering::Release);
//!
//! // elsewhere, perhaps on other threads:
//! let dt = DELTA_TIME.load(Ordering::Acquire);
//! // Use `dt` to compute simulation...
//! ```
//!
//! # Portability
//!
//! In general, this library is as portable as [`AtomicU32`]/[`AtomicU64`]
//! (fairly portable). See the module documentation for [core::sync::atomic] for
//! information about the portability of atomic operations as a whole.
//!
//! [`AtomicU32`]: core::sync::atomic::AtomicU32
//! [`AtomicU64`]: core::sync::atomic::AtomicU64
//!
//! Some architectures do not support 64-bit atomics, so [`AtomicF64`] is not
//! available on such architectures. Examples include 32-bit PowerPC, MIPS, and
//! Arm M-Profile.
//!
//! # Potential Use Cases
//!
//! The motivating cases for this were:
//!
//! - Tunable parameters loaded from a file that otherwise behaved as global
//!   constants (still compelling to me).
//!
//! - Global variables like time deltas (see example above) which would need to
//!   be threaded through a large amount of code. (Not as compelling).
//!
//! But really it was another 90% finished project that I had meant to get out
//! the door.
//!
//! # Performance
//!
//! On x86 and x86_64: basically 0 cost if you pick the right orderings and
//! stick to load/store.
//!
//! On everything else: acceptable if you pick the right orderings.
//!
//! In general, this depends on your architecture. If you're on x86{,_64}, you
//! can get away with a lot of dodgy atomic code. Even `SeqCst` usage won't bite
//! you too bad, so long as stores are rare. That said, I'd try to stick to
//! `Acquire`/`Release` even on x86. For load/store, this has roughly 0 cost
//! compared to write/read to a global variable directly. Also, if you just need
//! atomicity, and not any global orderings, feel free to use Relaxed.
//!
//! (I normally wouldn't give this advice, but you're probably not using
//! floating point in a situation where the exact value you get must follow
//! absolute rules).
//!
//! Beyond all of this, we provide a few convenient RMW operations. Ones that
//! have to perform actual float operations, such as `fetch_add`/`fetch_sub`
//! (but not ones that operate solely on the binary representation, like
//! `fetch_abs` or `fetch_neg`) need to perform a CAS loop. That means they're
//! much slower than `fetch_add`/`fetch_sub` are for AtomicU32, for example.
#![no_std]
#![deny(missing_docs)]

mod atomic_f32;
pub use atomic_f32::AtomicF32;

#[cfg(target_has_atomic = "64")]
mod atomic_f64;

#[cfg(target_has_atomic = "64")]
pub use atomic_f64::AtomicF64;

use core::sync::atomic::Ordering;

#[inline]
fn fail_order_for(order: Ordering) -> Ordering {
    match order {
        Ordering::Release | Ordering::Relaxed => Ordering::Relaxed,
        Ordering::Acquire | Ordering::AcqRel => Ordering::Acquire,
        Ordering::SeqCst => Ordering::SeqCst,
        o => unreachable!("Unknown ordering: {:?} (file a bug with atomic_float)", o),
    }
}
