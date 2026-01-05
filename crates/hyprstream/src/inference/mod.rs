//! Inference service abstraction for in-process and future IPC access.
//!
//! This module provides a service-oriented interface to inference operations,
//! allowing components to share inference capabilities through client handles.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │           InferenceClient (trait)               │
//! │  - generate()       - generate_stream()         │
//! │  - model_info()     - load_lora()               │
//! │  - set_session()    - health_check()            │
//! └─────────────────────────────────────────────────┘
//!                         ▲
//!                         │
//!               ┌─────────┴─────────┐
//!               │ LocalInferenceClient │
//!               │   (mpsc channels)  │
//!               └───────────────────┘
//!                         │
//!                         ▼
//! ┌─────────────────────────────────────────────────┐
//! │          LocalInferenceService                  │
//! │  - engine: TorchEngine                          │
//! │  - runs on dedicated thread                     │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use hyprstream::inference::{LocalInferenceService, InferenceClient};
//! use std::sync::Arc;
//!
//! // Start the service (returns client handle)
//! let client = LocalInferenceService::start("/path/to/model", config).await?;
//!
//! // Generate text
//! let result = client.generate(request).await?;
//!
//! // Streaming generation
//! let mut stream = client.generate_stream(request).await?;
//! while let Some(chunk) = stream.next().await {
//!     print!("{}", chunk?);
//! }
//! let stats = stream.stats().await?;
//!
//! // Client can be cloned and shared
//! let client2 = client.clone();
//! ```
//!
//! # Thread Model
//!
//! The service runs on a dedicated thread with its own single-threaded tokio
//! runtime. This isolates GPU operations and handles tch-rs types that contain
//! raw pointers (not Send). This matches the pattern used by git2db's
//! LocalService.

mod client;
mod local;
mod request;
mod response;

pub use client::{InferenceClient, InferenceError};
pub use local::{LocalInferenceClient, LocalInferenceService};
pub use request::InferenceRequest;
pub use response::{StreamHandle, StreamStats};
