@0xf9827f99f1d00ad6;

# Cap'n Proto schema for compositor/ChatApp IPC via Wanix named pipes.
#
# Replaces the OSC escape code multiplexing (0xFE/0xFF/0xFD) and 9-byte
# framed stdin protocol with typed capnp messages on dedicated pipes.
#
# Wire format (both directions):
#   [4B little-endian message length][capnp single-segment bytes, no segment table]

# ═══════════════════════════════════════════════════════════════════
# Compositor IPC (browser ↔ compositor WASM task via Wanix named pipe)
# ═══════════════════════════════════════════════════════════════════

# Browser → Compositor (replaces 9-byte framed stdin)
struct CompositorIpcIn {
  union {
    ansiFrame @0 :FrameMsg;       # ANSI from server pane
    keyboard @1 :Data;             # raw key bytes
    resize @2 :ResizeMsg;          # terminal resize
    appFrame @3 :FrameMsg;         # ANSI from ChatApp
    windowList @4 :Data;           # JSON window state
    paneClosed @5 :UInt32;         # pane ID
    sessionReset @6 :Void;
  }
}

struct FrameMsg {
  id @0 :UInt32;
  data @1 :Data;
}

struct ResizeMsg {
  cols @0 :UInt16;
  rows @1 :UInt16;
}

# Compositor → Browser (replaces OSC 0xFE/0xFF)
struct CompositorIpcOut {
  union {
    rpcRequest @0 :Data;           # JSON-serialized RpcRequest (serde)
    routeInput @1 :RouteInputMsg;
  }
}

struct RouteInputMsg {
  appId @0 :UInt32;
  data @1 :Data;
}

# ═══════════════════════════════════════════════════════════════════
# ChatApp IPC (browser ↔ ChatApp WASM task via Wanix named pipe)
# ═══════════════════════════════════════════════════════════════════

# Browser → ChatApp (replaces framed stdin 0x01-0x05)
struct ChatAppIpcIn {
  union {
    keyboard @0 :Data;
    resize @1 :ResizeMsg;
    inferenceToken @2 :Text;
    inferenceComplete @3 :Void;
    inferenceError @4 :Text;
    inferenceCancel @5 :Void;        # main.rs:321 — on_stream_cancelled()
  }
}

# ChatApp → Browser (replaces OSC 0xFD)
struct ChatAppIpcOut {
  union {
    inferenceRequest @0 :Data;     # JSON payload (OpenAI-compat)
    ansiFrame @1 :Data;            # ANSI render output from ChatApp
  }
}
