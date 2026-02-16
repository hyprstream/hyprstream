//! Service layer constants.

/// Maximum I/O size per read/write operation (16 MiB).
pub const MAX_FS_IO_SIZE: u64 = 16 * 1024 * 1024;
/// Maximum open file descriptors per client.
pub const MAX_FDS_PER_CLIENT: u32 = 64;
/// Maximum open file descriptors globally.
pub const MAX_FDS_GLOBAL: u32 = 4096;
