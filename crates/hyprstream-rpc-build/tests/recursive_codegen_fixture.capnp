@0xe79fdb14cdd725a2;
# Test-only canonical schemas. Never included in the production RPC catalog.
# GroupNode's sole cycle edge lives inside the inline group's metadata.
struct GroupNode {
  union {
    end @0 :Void;
    branch :group {
      next @1 :GroupNode;
    }
  }
}
enum RecursiveColor {
  red @0;
  blue @1;
  green @2;
}
struct EnumNode {
  union {
    color @0 :RecursiveColor;
    next @1 :EnumNode;
  }
}
struct NestedLeaf {
  union {
    text @0 :Text;
    flag @1 :Bool;
  }
}
struct LeafParent {
  union {
    end @0 :Void;
    next @1 :LeafParent;
    leaf @2 :NestedLeaf;
  }
}
# The only self edge is inside a NAMED (non-union-member) inline group.
struct NamedGroupNode {
  label @0 :Text;
  body :union {
    end @1 :Void;
    next @2 :NamedGroupNode;
  }
}
