# Git Remote Helper for GitTorrent

The `git-remote-gittorrent` binary is a Git remote helper that allows standard Git commands to work transparently with `gittorrent://` URLs.

## How It Works

Git remote helpers are invoked automatically by Git when it encounters URLs with their protocol scheme. When you run:

```bash
git clone gittorrent://COMMIT_HASH
```

Git automatically invokes `git-remote-gittorrent` and communicates with it via the Git remote helper protocol over stdin/stdout.

## Installation

1. Build the project: `cargo build --release`
2. Install the binary in your PATH: `cp target/release/git-remote-gittorrent /usr/local/bin/`
3. Make sure it's executable: `chmod +x /usr/local/bin/git-remote-gittorrent`

## Configuration

Since Git invokes the remote helper automatically, configuration is done through:

### 1. TOML Configuration File

Create `~/.config/gittorrent/git-remote-gittorrent.toml`:

```toml
# P2P port for DHT (0 = random)
p2p_port = 4001

# Bootstrap nodes (multiaddr format with peer IDs)
bootstrap_nodes = [
    "/ip4/example.com/tcp/4001/p2p/12D3KooWExamplePeerID...",
    "/ip4/127.0.0.1/tcp/9999/p2p/12D3KooWAnotherPeerID..."
]

# Storage directory for cached repositories
storage_dir = "/home/user/.cache/gittorrent"

# Service configuration
bind_address = "127.0.0.1"
bind_port = 8081

# Enable auto-discovery via mDNS (default: true)
auto_discovery = true
```

### 2. Environment Variables

You can also configure via environment variables:

```bash
# Bootstrap nodes (comma-separated)
export GITTORRENT_BOOTSTRAP_NODES="/ip4/example.com/tcp/4001/p2p/12D3KooW...,/ip4/127.0.0.1/tcp/9999/p2p/12D3KooW..."

# P2P port
export GITTORRENT_P2P_PORT=4001

# Storage directory
export GITTORRENT_STORAGE_DIR="/tmp/gittorrent"

# Run in standalone mode (disable bootstrap)
export GITTORRENT_STANDALONE=true

# Disable auto-discovery
export GITTORRENT_NO_AUTO_DISCOVERY=true
```

## Usage Examples

### Basic Usage (mDNS auto-discovery)
```bash
git clone gittorrent://abc123...def456
```

### With Bootstrap Configuration
```bash
# Set bootstrap nodes
export GITTORRENT_BOOTSTRAP_NODES="/ip4/node1.example.com/tcp/4001/p2p/12D3KooW..."

# Clone from P2P network
git clone gittorrent://abc123...def456 my-repo
cd my-repo
git log
```

### Standalone Mode
```bash
# Disable P2P networking
export GITTORRENT_STANDALONE=true

# This will create a basic repository
git clone gittorrent://abc123...def456 offline-repo
```

## Supported URL Formats

1. **Commit Hash**: `gittorrent://COMMIT_SHA256`
2. **Commit with Refs**: `gittorrent://COMMIT_SHA256?refs`
3. **Legacy Git Server**: `gittorrent://github.com/user/repo`
4. **Username**: `gittorrent://username`

## Network Discovery

The remote helper uses multiple methods to find GitTorrent peers:

1. **Bootstrap Nodes**: Explicit peers configured via config/environment
2. **mDNS Discovery**: Automatic discovery of local network peers
3. **DHT**: Distributed hash table for peer discovery

## Troubleshooting

### Enable Debug Logging
```bash
export RUST_LOG=debug
git clone gittorrent://...
```

### Check Configuration
The remote helper looks for config in this order:
1. Environment variables (highest priority)
2. `~/.config/gittorrent/git-remote-gittorrent.toml`
3. Built-in defaults (mDNS enabled, no bootstrap nodes)

### Common Issues

**No peers found**:
- Check bootstrap node configuration
- Ensure gittorrentd is running if using local network
- Verify multiaddr format includes peer IDs

**Permission denied**:
- Ensure `git-remote-gittorrent` is in PATH and executable
- Check file permissions: `chmod +x /usr/local/bin/git-remote-gittorrent`

**Repository not found**:
- The commit hash may not exist in the P2P network
- Try with `GITTORRENT_STANDALONE=true` for offline testing