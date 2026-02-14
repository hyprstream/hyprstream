@0xf1c2d3e4a5b6c7d8;

# Custom annotations for MCP tool generation and documentation.
#
# These annotations are extracted at build time and used to generate
# descriptive MCP tool definitions automatically.

# Type-safe scope actions — invalid action = capnp compile error.
enum ScopeAction {
  query @0;
  write @1;
  manage @2;
  infer @3;
  train @4;
  serve @5;
  context @6;
}

# MCP tool description - used for method-level documentation
annotation mcpDescription(field, union, group, struct, enum, enumerant) :Text;

# Parameter description - used for field-level documentation in MCP tools
annotation paramDescription(field) :Text;

# MCP scope annotation — type-safe enum replaces raw Text strings.
# Usage: $mcpScope(write) — invalid action is a capnp compile error.
annotation mcpScope(field) :ScopeAction;

# Mark as deprecated with reason
annotation deprecated(field, union, struct, enum) :Text;

# Example value for documentation
annotation example(field) :Text;

# Hide a method from the CLI (internal-only methods like session management, streaming auth)
annotation cliHidden(field) :Void;

# Fixed-size constraint for Data fields — generates [u8; N] instead of Vec<u8>.
# Usage: serverPubkey @2 :Data $fixedSize(32);
annotation fixedSize(field) :UInt32;
