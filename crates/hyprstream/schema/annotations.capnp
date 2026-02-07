@0xf1c2d3e4a5b6c7d8;

# Custom annotations for MCP tool generation and documentation.
#
# These annotations are extracted at build time and used to generate
# descriptive MCP tool definitions automatically.

# MCP tool description - used for method-level documentation
annotation mcpDescription(field, union, group, struct, enum, enumerant) :Text;

# Parameter description - used for field-level documentation in MCP tools
annotation paramDescription(field) :Text;

# Mark as deprecated with reason
annotation deprecated(field, union, struct, enum) :Text;

# Example value for documentation
annotation example(field) :Text;
