@0xb8b2fdca21f1727e;

# Chat Core IPC Schema
#
# Messages between chat-core (WASI process or WASM module) and its clients
# (web GUI, TUI renderer, host orchestrator).
#
# Carried over ZMTP multipart frames on DMA ring buffers.
# Same wire format as all other hyprstream Cap'n Proto messages.

struct ChatCoreIn {
  # Commands sent to chat-core
  union {
    userMessage @0 :Text;
    inferenceToken @1 :Text;
    inferenceComplete @2 :Void;
    inferenceError @3 :Text;
    toolCallResult @4 :ToolCallResultMsg;
    cancel @5 :Void;
  }
}

struct ToolCallResultMsg {
  id @0 :Text;
  result @1 :Text;
}

struct ChatCoreOut {
  # Events emitted by chat-core
  union {
    content @0 :Text;
    thinking @1 :Text;
    toolCallDetected @2 :ToolCallDetectedMsg;
    complete @3 :Void;
    error @4 :Text;
  }
}

struct ToolCallDetectedMsg {
  id @0 :Text;
  uuid @1 :Text;
  description @2 :Text;
  arguments @3 :Text;
}
