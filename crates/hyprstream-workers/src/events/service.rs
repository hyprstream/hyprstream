//! EventService — event bus using `MoqEventBarrierService`.
//!
//! The legacy ZMQ XPUB/XSUB `ProxyService` has been removed (#138).
//! Event distribution is now handled by `MoqEventBarrierService` in the
//! service factory.
