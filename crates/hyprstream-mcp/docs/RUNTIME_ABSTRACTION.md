# TCL Runtime Abstraction

This document describes the TCL runtime abstraction layer that allows the TCL MCP Server to support multiple TCL interpreter implementations.

## Overview

The TCL MCP Server uses a trait-based abstraction to support different TCL interpreters:

- **Molt** (default): A safe, embedded TCL interpreter written in Rust
- **TCL** (optional): The official TCL interpreter via Rust bindings

## Architecture

```rust
pub trait TclRuntime {
    fn new() -> Self where Self: Sized;
    fn eval(&mut self, script: &str) -> Result<String>;
    fn set_var(&mut self, name: &str, value: &str) -> Result<()>;
    fn get_var(&self, name: &str) -> Result<String>;
    fn has_command(&self, command: &str) -> bool;
    fn name(&self) -> &'static str;
}
```

## Building with Different Runtimes

### Default Build (Molt)
```bash
cargo build --release
```

### Official TCL Interpreter
```bash
# Requires TCL development libraries installed on system
cargo build --release --no-default-features --features tcl
```

### Both Runtimes
```bash
cargo build --release --features molt,tcl
```

## Runtime Selection

The runtime is selected at compile time based on feature flags:

```toml
[features]
default = ["molt"]
molt = ["dep:molt"]
tcl = ["dep:tcl"]
```

## Implementation Details

### Molt Runtime

The Molt runtime provides:
- Memory safety through Rust implementation
- Subset of TCL commands suitable for most use cases
- No external dependencies
- Thread safety handled by the executor layer

### TCL Runtime

The official TCL runtime provides:
- Full TCL language support
- Access to all TCL extensions and packages
- Requires TCL installed on the system
- C FFI bindings

## Thread Safety

Since interpreters may not be thread-safe (e.g., Molt uses Rc internally), the `TclExecutor` runs in a dedicated thread and communicates via channels:

```rust
pub fn spawn(privileged: bool) -> mpsc::Sender<TclCommand> {
    let (tx, mut rx) = mpsc::channel::<TclCommand>(100);
    
    thread::spawn(move || {
        let mut executor = TclExecutor::new(privileged);
        // Handle commands in dedicated thread
    });
    
    tx
}
```

## Testing

Both runtimes are tested with the same test suite to ensure compatibility:

```bash
# Test Molt runtime
cargo test --features molt

# Test TCL runtime (if TCL is installed)
cargo test --features tcl

# Test with both
cargo test --features molt,tcl
```

## Performance Considerations

- **Molt**: Faster startup, lower memory usage, suitable for embedded scenarios
- **TCL**: More features, better for complex scripts, higher memory usage

## Future Enhancements

1. Runtime selection at startup (when both are compiled in)
2. Plugin system for custom TCL commands
3. Performance benchmarking suite
4. Additional interpreter support (Jim TCL, etc.)