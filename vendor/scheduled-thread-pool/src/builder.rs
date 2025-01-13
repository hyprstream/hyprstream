//! Staged builder types.

use crate::{OnPoolDropBehavior, ScheduledThreadPool};

/// The builder stage expecting the `num_threads` value.
pub struct NumThreadsStage(pub(crate) ());

impl NumThreadsStage {
    /// Specifies the number of threads the pool should use.
    ///
    /// # Panics
    ///
    /// Panics if `num_threads` is 0.
    pub fn num_threads<'a>(self, num_threads: usize) -> FinalStage<'a> {
        assert!(num_threads > 0, "num_threads must be positive");
        FinalStage {
            num_threads,
            thread_name_pattern: None,
            on_drop_behavior: OnPoolDropBehavior::CompletePendingScheduled,
        }
    }
}

/// The final builder stage, allowing configuration of optional paramters.
pub struct FinalStage<'a> {
    pub(crate) num_threads: usize,
    pub(crate) thread_name_pattern: Option<&'a str>,
    pub(crate) on_drop_behavior: OnPoolDropBehavior,
}

impl<'a> FinalStage<'a> {
    /// Sets the pattern to be used when naming threads created to be part of the
    /// pool.
    ///
    /// The substring `{}` in the name will be replaced with an integer
    /// identifier of the thread.
    ///
    /// Defaults to `None`.
    pub fn thread_name_pattern(mut self, thread_name_pattern: &'a str) -> Self {
        self.thread_name_pattern = Some(thread_name_pattern);
        self
    }

    /// Sets the behavior for what to do with pending scheduled executions when
    /// the pool is dropped.
    ///
    /// Defaults to [OnPoolDropBehavior::CompletePendingScheduled].
    pub fn on_drop_behavior(mut self, on_drop_behavior: OnPoolDropBehavior) -> Self {
        self.on_drop_behavior = on_drop_behavior;
        self
    }

    /// Consumes the builder, returning the constructed thread pool.
    pub fn build(self) -> ScheduledThreadPool {
        ScheduledThreadPool::new_inner(self)
    }
}
