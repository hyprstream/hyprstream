# Attach stdin lifecycle regression (Unix)

Compile through the normal BuildQ wrapper:

`cargo test --release --locked --offline -p hyprstream --lib attach_pty_child --no-run`

Then run under the same CPU libtorch/BuildQ environment, using the exact test binary printed by Cargo:

`python3 crates/hyprstream/tests/tui_attach_pty.py /absolute/path/to/hyprstream_core-HASH`

The driver owns a PTY and child process, sends Ctrl-B d to the production terminal loop, checks termios restoration and real runtime drop, and asserts successful child exit within 10 seconds. It kills/reaps its own child on a liveness failure and prints a JSON receipt with binary hash. No TUI service, credentials, network, or model is used. The Rust child seam is ignored in ordinary test runs, not a passing no-op regression. Attach assumes it is the sole stdin reader. This covers normal detach; cancellation of the whole attach future is outside this regression.
