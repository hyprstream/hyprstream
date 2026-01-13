# Changelog

## 0.1.1

### Added
- TCL runtime abstraction layer to support multiple interpreters
- Support for official TCL interpreter via `tcl` crate (feature flag: `tcl`)
- Build-time feature selection for TCL runtime (`molt` or `tcl`)
- Comprehensive test suite for runtime abstraction
- Documentation for runtime abstraction architecture

### Changed
- Refactored `TclExecutor` to use the new `TclRuntime` trait
- Molt interpreter is now behind a feature flag (default enabled)
- Updated README with build instructions for different runtimes

### Technical Details
- Created `TclRuntime` trait with methods: `new()`, `eval()`, `set_var()`, `get_var()`, `has_command()`, `name()`
- Implemented `MoltRuntime` for Molt interpreter
- Implemented `TclInterpreter` for official TCL (requires system TCL libraries)
- Thread safety maintained through existing executor architecture
- All existing functionality preserved with runtime abstraction
