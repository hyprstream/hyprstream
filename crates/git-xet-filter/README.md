# git-xet-filter

XET large file storage filter integration for libgit2.

This crate provides transparent integration with [XET](https://github.com/huggingface/xet-core) repositories via libgit2's filter mechanism, enabling automatic handling of large files stored in XET's content-addressable storage.

## Features

- **Standalone or Integrated**: Can be used independently or as part of git2db
- **Transparent**: Seamlessly handles XET pointer files during git operations
- **Async**: Built on Tokio for efficient async I/O
- **Thread-Safe**: Designed for concurrent use

## Usage

### Standalone

```rust
use git_xet_filter::{XetConfig, initialize};
use git2::Repository;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure XET
    let config = XetConfig::new("https://cas.xet.dev")
        .with_token("your-token-here");

    // Initialize the filter (registers with libgit2)
    initialize(config).await?;

    // Now all git operations will transparently handle XET files
    let repo = Repository::open("/path/to/xet/repo")?;
    repo.checkout_head(None)?; // Large files automatically fetched from CAS

    Ok(())
}
```

### With git2db

```rust
use git2db::{Git2DB, config::XetConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = XetConfig::default(); // Uses XETHUB_TOKEN env var
    git2db::xet_filter::initialize(config).await?;

    let registry = Git2DB::open("/path/to/registry").await?;
    // XET filter is now active for all git operations
    Ok(())
}
```

## Configuration

The filter is configured via `XetConfig`:

- `endpoint`: CAS endpoint URL (default: `https://cas.xet.dev`)
- `token`: Authentication token (falls back to `XETHUB_TOKEN` env var)
- `compression`: Optional compression scheme

## Features

- `xet-storage` (optional): Enables full XET integration. Without this feature, only error types are available.

## Architecture

The filter uses libgit2's filter API to intercept file reads/writes:

1. **Clean filter**: Converts files to XET pointers during `git add`
2. **Smudge filter**: Fetches files from CAS during `git checkout`

The filter is registered globally and applies to all repositories opened in the process.

## Thread Safety

All operations are thread-safe. The filter uses:
- `OnceCell` for initialization
- `DashMap` for concurrent callback tracking
- Thread-local storage for error reporting

## Error Handling

Errors are available via the `last_error()` function:

```rust
use git2::Repository;
use git_xet_filter;

let repo = Repository::open("path/to/repo")?;
if let Err(e) = repo.checkout_head(None) {
    // Check for XET-specific errors
    if let Some(xet_error) = git_xet_filter::last_error() {
        eprintln!("XET filter error: {:?}", xet_error);
    }
}
```

## License

MIT
