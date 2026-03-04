@0xd4e8f2a1b3c5d7e9;

# Cap'n Proto schema for TUI display server
#
# The TUI service provides a terminal multiplexer over ZMQ RPC,
# supporting session persistence, multi-pane layouts, remote access,
# and MCP-controllable display surfaces.
# Uses REQ/REP pattern. Runs on thread spawner (ratatui Buffer is !Send).

using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;

# ═══════════════════════════════════════════════════════════════════
# Frame Types (streamed via StreamPublisher, not RPC)
# ═══════════════════════════════════════════════════════════════════

# A single frame of TUI output, sent over the streaming channel.
struct TuiFrame {
  # Frame generation counter (monotonically increasing)
  generation @0 :UInt64;

  # Timestamp in milliseconds since epoch
  timestampMs @1 :UInt64;

  # Frame content
  union {
    # Incremental update (only changed cells + scroll ops)
    incremental @2 :IncrementalFrame;

    # Full frame (all cells, used after resize or when >50% dirty)
    full @3 :FullFrame;
  }

  # Cursor state after this frame
  cursor @4 :CursorInfo;
}

# Incremental frame: scroll operations + changed cells only
struct IncrementalFrame {
  # Scroll operations to apply before cell deltas
  scrolls @0 :List(ScrollOp);

  # Changed cells (sparse)
  deltas @1 :List(CellDelta);
}

# Full frame: complete cell grid
struct FullFrame {
  # Terminal dimensions
  cols @0 :UInt16;
  rows @1 :UInt16;

  # All cells packed row-major. Each cell is a CellDelta.
  cells @2 :List(CellDelta);
}

# A single cell change
struct CellDelta {
  # Position
  x @0 :UInt16;
  y @1 :UInt16;

  # Character (UTF-8 encoded, usually 1-4 bytes)
  symbol @2 :Text;

  # Style
  fg @3 :UInt32;  # RGBA packed (or indexed color encoding)
  bg @4 :UInt32;
  modifiers @5 :UInt16;  # Bitfield matching ratatui::style::Modifier
}

# Scroll operation
struct ScrollOp {
  # Region (top..bottom rows, 0-indexed)
  top @0 :UInt16;
  bottom @1 :UInt16;

  # Positive = scroll up (content moves up, new lines at bottom)
  # Negative = scroll down
  amount @2 :Int16;
}

# Cursor state
struct CursorInfo {
  x @0 :UInt16;
  y @1 :UInt16;
  visible @2 :Bool;

  # Cursor shape
  shape @3 :CursorShape;
}

enum CursorShape {
  block @0;
  underline @1;
  bar @2;
  blinkingBlock @3;
  blinkingUnderline @4;
  blinkingBar @5;
}

# ═══════════════════════════════════════════════════════════════════
# RPC Request / Response
# ═══════════════════════════════════════════════════════════════════

# Unified TUI request with union discriminator
struct TuiRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload
  union {
    # Connect a new viewer to a session (creates session if needed)
    connect @1 :ConnectRequest
      $mcpScope(write) $mcpDescription("Connect viewer to TUI session");

    # Disconnect a viewer
    disconnect @2 :UInt32  # viewer_id
      $mcpScope(write) $mcpDescription("Disconnect viewer from TUI session");

    # Create a new window in the current session
    createWindow @3 :Void
      $mcpScope(write) $mcpDescription("Create a new window");

    # Close a window
    closeWindow @4 :UInt32  # window_id
      $mcpScope(write) $mcpDescription("Close a window");

    # List all windows in a session
    listWindows @5 :UInt32  # session_id
      $mcpScope(query) $mcpDescription("List all windows in a session");

    # Focus a window
    focusWindow @6 :UInt32  # window_id
      $mcpScope(write) $mcpDescription("Focus a window");

    # Split the active pane
    splitPane @7 :SplitRequest
      $mcpScope(write) $mcpDescription("Split active pane horizontally or vertically");

    # Close a pane
    closePane @8 :UInt32  # pane_id
      $mcpScope(write) $mcpDescription("Close a pane");

    # Focus a pane
    focusPane @9 :UInt32  # pane_id
      $mcpScope(write) $mcpDescription("Focus a pane");

    # Get a snapshot of the current display state
    snapshot @10 :UInt32  # session_id (0 = current)
      $mcpScope(query) $mcpDescription("Get snapshot of current display");

    # Resize the terminal
    resize @11 :ResizeRequest
      $mcpScope(write) $mcpDescription("Resize terminal dimensions");

    # Send input to the active pane
    sendInput @12 :SendInputRequest
      $mcpScope(write) $mcpDescription("Send input to active pane");
  }
}

# Send input request
struct SendInputRequest {
  # Viewer ID that is sending the input
  viewerId @0 :UInt32;
  # Raw input bytes (keypresses, paste data, etc.)
  data @1 :Data;
}

# Connection request
struct ConnectRequest {
  # Session to connect to (0 = create new or reattach most recent)
  sessionId @0 :UInt32;

  # Desired display mode
  displayMode @1 :DisplayMode;

  # Terminal dimensions
  cols @2 :UInt16;
  rows @3 :UInt16;
}

enum DisplayMode {
  # ANSI escape sequences (for terminal viewers)
  ansi @0;
  # Cap'n Proto TuiFrame (for structured viewers)
  capnp @1;
  # Packed binary cells (for StructuredBackend viewers)
  structured @2;
}

# Split request
struct SplitRequest {
  # Split direction
  direction @0 :SplitDirection;
  # Split ratio (0.0-1.0, default 0.5)
  ratio @1 :Float32;
}

enum SplitDirection {
  horizontal @0;
  vertical @1;
}

# Resize request
struct ResizeRequest {
  cols @0 :UInt16;
  rows @1 :UInt16;
}

# Unified TUI response
struct TuiResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload
  union {
    # Error occurred
    error @1 :ErrorInfo;

    # Connection established (for connect)
    connectResult @2 :ConnectResult;

    # Disconnect confirmed (for disconnect)
    disconnectResult @3 :Void;

    # Window created (for createWindow)
    createWindowResult @4 :WindowInfo;

    # Window closed (for closeWindow)
    closeWindowResult @5 :Void;

    # Window list (for listWindows)
    listWindowsResult @6 :WindowList;

    # Window focused (for focusWindow)
    focusWindowResult @7 :Void;

    # Pane split (for splitPane)
    splitPaneResult @8 :PaneInfo;

    # Pane closed (for closePane)
    closePaneResult @9 :Void;

    # Pane focused (for focusPane)
    focusPaneResult @10 :Void;

    # Snapshot (for snapshot)
    snapshotResult @11 :TuiSnapshot;

    # Resize confirmed (for resize)
    resizeResult @12 :Void;

    # Input sent (for sendInput)
    sendInputResult @13 :Void;
  }
}

# Error information
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}

# Connection result
struct ConnectResult {
  # Assigned viewer ID
  viewerId @0 :UInt32;

  # Session this viewer is connected to
  sessionId @1 :UInt32;

  # Stream info for receiving frames
  streamInfo @2 :StreamInfo;

  # Current window list
  windows @3 :List(WindowInfo);
}

# Stream connection info
struct StreamInfo {
  # Topic to subscribe to for frame data
  topic @0 :Text;
  # SUB endpoint
  subEndpoint @1 :Text;
  # MAC key for HMAC verification (32 bytes)
  macKey @2 :Data;
}

# Window information
struct WindowInfo {
  id @0 :UInt32;
  name @1 :Text;
  panes @2 :List(PaneInfo);
  activePaneId @3 :UInt32;
}

# Window list
struct WindowList {
  windows @0 :List(WindowInfo);
}

# Pane information
struct PaneInfo {
  id @0 :UInt32;
  cols @1 :UInt16;
  rows @2 :UInt16;
  title @3 :Text;
}

# Full display snapshot
struct TuiSnapshot {
  sessionId @0 :UInt32;
  windows @1 :List(WindowInfo);
  activeWindowId @2 :UInt32;
  generation @3 :UInt64;
  # Full frame data for the active window
  frame @4 :FullFrame;
}

# ═══════════════════════════════════════════════════════════════════
# Control Messages (bidirectional, over WebTransport or ZMQ)
# ═══════════════════════════════════════════════════════════════════

# Client-to-server control message
struct TuiControlMessage {
  union {
    # Keyboard/mouse input from viewer
    input @0 :Data;

    # Acknowledge receipt of frame generation
    ack @1 :UInt64;

    # Viewer resized its terminal
    resize @2 :ResizeRequest;
  }
}
