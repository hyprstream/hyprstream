# TCL MCP Server Test Suite

This directory contains comprehensive tests for the TCL MCP server, with a focus on the `bin__exec_tool` functionality.

## Test Structure

```
tests/
├── unit/                           # Rust unit tests
│   └── test_bin_exec_tool.rs      # Unit tests for bin__exec_tool
├── integration/                    # Rust integration tests
│   └── test_bin_exec_tool_integration.rs
├── examples/                       # Example usage documentation
│   └── bin_exec_tool_examples.md
├── fixtures/                       # Test data and fixtures
├── test_bin_exec_tool.tcl         # TCL-based tests
├── test_bin_exec_tool_mcp.py     # Python MCP protocol tests
├── run_bin_exec_tool_tests.sh    # Test runner script
└── README.md                      # This file
```

## Running Tests

### All Tests
Run all tests using the provided script:
```bash
./tests/run_bin_exec_tool_tests.sh
```

### Individual Test Types

#### Rust Unit Tests
```bash
cargo test --test test_bin_exec_tool
```

#### Rust Integration Tests
```bash
cargo test --test test_bin_exec_tool_integration
```

#### TCL Tests
```bash
tclsh tests/test_bin_exec_tool.tcl
```

#### Python MCP Tests
```bash
python3 -m pytest tests/test_bin_exec_tool_mcp.py -v
```

## Test Coverage

### Unit Tests (`test_bin_exec_tool.rs`)
- Basic tool execution
- Missing required parameters
- Optional parameters
- Non-existent tool handling
- Parameter type validation
- Special character handling
- Script error handling
- Complex TCL operations
- Concurrent execution
- Privilege mode restrictions

### Integration Tests (`test_bin_exec_tool_integration.rs`)
- End-to-end MCP protocol testing
- Tool addition and execution flow
- Error handling through MCP
- Privilege restrictions
- Concurrent access
- Complex parameter scenarios

### TCL Tests (`test_bin_exec_tool.tcl`)
- TCL script execution
- Parameter validation
- Namespace isolation
- State management
- Error propagation

### Python MCP Tests (`test_bin_exec_tool_mcp.py`)
- Full MCP client simulation
- Protocol compliance
- Real server interaction
- Performance testing
- Security validation

## Key Test Scenarios

### 1. Tool Lifecycle
- Creating tools with `tcl_tool_add`
- Executing tools with `bin__exec_tool`
- Listing tools with `tcl_tool_list`
- Removing tools with `tcl_tool_remove`

### 2. Parameter Handling
- Required vs optional parameters
- Type validation (string, number, boolean, array)
- Special characters and escaping
- Complex nested structures

### 3. Error Scenarios
- Missing required parameters
- Invalid parameter types
- Non-existent tools
- TCL script errors
- Permission denied (non-privileged mode)

### 4. Concurrency
- Multiple simultaneous tool executions
- Thread safety validation
- Race condition testing
- Resource contention

### 5. Security
- Privileged vs non-privileged mode
- Input sanitization
- Command injection prevention
- Resource limits

## Adding New Tests

### Rust Tests
Add test functions to the appropriate test module:
```rust
#[tokio::test]
async fn test_new_feature() {
    // Test implementation
}
```

### TCL Tests
Add test cases using the `test_case` helper:
```tcl
test_case "new_feature" {
    # Test implementation
    assert {condition} "Test description"
}
```

### Python Tests
Add test methods to the test class:
```python
@pytest.mark.asyncio
async def test_new_feature(self, client):
    # Test implementation
    assert result == expected
```

## Test Data

The `fixtures/` directory can be used for:
- Sample TCL scripts
- Test configuration files
- Mock data for testing
- Expected output files

## Continuous Integration

These tests are designed to be run in CI/CD pipelines:
- Fast execution (< 1 minute total)
- No external dependencies
- Clear pass/fail status
- Detailed error reporting

## Troubleshooting

### Common Issues

1. **Cargo not found**: Ensure Rust is installed
2. **tclsh not found**: Install TCL interpreter
3. **pytest not found**: Install with `pip install pytest`
4. **Permission denied**: Make scripts executable with `chmod +x`

### Debug Mode

Set environment variables for more verbose output:
```bash
RUST_LOG=debug ./tests/run_bin_exec_tool_tests.sh
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add appropriate documentation
4. Update this README if needed