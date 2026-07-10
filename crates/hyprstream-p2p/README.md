# GitTorrent-RS

### A Rust Implementation of the Decentralized Git Network

**GitTorrent-RS** is a Rust rewrite of [GitTorrent](http://gittorrent.org), a peer-to-peer network of Git repositories shared over BitTorrent. This implementation focuses on performance, safety, and modern networking capabilities.

This project diverges in protocol from GitTorrent, and utilizes [libp2p](https://libp2p.io/) for its modern P2P networking foundation, including Kademlia DHT implementation, and supports Git objects in SHA-256 format.

## How It Works

GitTorrent-RS operates as a peer-to-peer Git hosting network:

1. **Repository Publishing**: When you publish a repository, its Git objects are stored in the libp2p DHT using content-addressed storage
2. **Commit-Based Discovery**: Each commit hash serves as a direct address for locating repository data in the DHT network
3. **Peer-to-Peer Transfer**: Git objects are transferred directly between peers using libp2p protocols, eliminating central servers
4. **Git Integration**: The `git-remote-gittorrent` helper seamlessly integrates with standard Git commands
5. **Large File Support**: Integration with XET (eXtensible rEmote Transfer) handles large files efficiently

This architecture provides true decentralization while maintaining full compatibility with existing Git workflows.

## Quick Start

### Requirements

- **Git 2.29+**: SHA-256 object format support requires Git version 2.29 or later
- **Rust 1.70+**: For building from source

### Installation

Build from source:
```bash
cargo build --release
```

The binaries will be available at:
- `target/release/gittorrentd` - The daemon for serving repositories
- `target/release/git-remote-gittorrent` - Git transport helper

### Usage

Clone a repository:
```bash
# Clone by commit hash (recommended)
git clone gittorrent://7d234ffe4317d150166acb545b83712eac417b8d

# Clone with references included
git clone gittorrent://7d234ffe4317d150166acb545b83712eac417b8d?refs

# Legacy: Clone via Git server (hybrid mode)
git clone gittorrent://github.com/someuser/somerepo

# Clone via username (requires external lookup)
git clone gittorrent://username
```

Create a SHA-256 Git repository:
```bash
# Create a new SHA-256 repository
git init --object-format=sha256 myproject
cd myproject

# Add some content
echo "# My Project" > README.md
git add README.md
git commit -m "Initial commit"

# Get the commit hash for gittorrent:// URL
git rev-parse HEAD
# Example output: 7d234ffe4317d150166acb545b83712eac417b8d85f2c4b5a3f8e1d6c9b0a1e2f
```

Convert existing repository to SHA-256:
```bash
# Note: This creates a new repository with new commit hashes
git clone --object-format=sha256 /path/to/old/repo /path/to/new/sha256/repo
```

Check if your repository uses SHA-256:
```bash
# Check the object format of your repository
git rev-parse HEAD
# SHA-256 hashes are 64 characters, SHA-1 hashes are 40 characters
# Example SHA-256: 7d234ffe4317d150166acb545b83712eac417b8d85f2c4b5a3f8e1d6c9b0a1e2f
# Example SHA-1:   7d234ffe4317d150166acb545b83712eac417b8d

# Or check directly:
git config core.objectformat
# Should output: sha256
```

Serve your repositories:
```bash
# Mark a repository as exportable
touch myproject/.git/git-daemon-export-ok

# Start the daemon
./target/release/gittorrentd
```

## Architecture

GitTorrent-RS implements the same five-component design as the original GitTorrent, modernized with libp2p:

1. **Git Transport Helper** (`git-remote-gittorrent`) - Integrates with Git to handle `gittorrent://` URLs
2. **Distributed Hash Table** - libp2p Kademlia DHT for advertising available commits and repository metadata
3. **P2P Protocol Layer** - libp2p networking stack with custom protocols for Git object transfers
4. **Key/Value Store** - Structured storage of user profiles and repository metadata in the DHT
5. **Username Registration** - External username resolution system (Bitcoin blockchain or other registries)

### Key Improvements in Rust

- **Memory Safety**: Leverages Rust's ownership system to prevent common networking vulnerabilities
- **Async Performance**: Built on Tokio for high-performance concurrent networking
- **Type Safety**: Strong typing prevents protocol mismatches and data corruption
- **Modern P2P**: Uses libp2p for enhanced peer discovery, transport security, and protocol negotiation
- **Modular Design**: Clean separation between Git operations, networking, and storage layers

## Project Structure

```
src/
├── bin/
│   ├── gittorrentd.rs           # Daemon binary
│   └── git-remote-gittorrent.rs # Git transport helper
├── crypto/                      # Cryptographic primitives (ed25519)
├── dht/                         # libp2p Kademlia DHT implementation
├── git/                         # Git protocol, objects, and remote helper
├── service.rs                   # Core GitTorrent service coordination
├── types.rs                     # Core types and URL parsing
├── error.rs                     # Error handling
├── lfs_xet.rs                   # Large file support integration
└── lib.rs                       # Library interface
```

## Supported URL Formats

GitTorrent-RS supports several URL formats for different use cases:

### Commit-Based URLs (Recommended)
- `gittorrent://COMMIT_SHA256` - Clone specific commit from DHT (64-character hex SHA-256)
- `gittorrent://COMMIT_SHA256?refs` - Clone commit with all Git references included

### Legacy URL Formats
- `gittorrent://server.com/user/repo` - Hybrid mode using existing Git servers for discovery
- `gittorrent://username` - Username resolution via external registry (Bitcoin blockchain, etc.)

### Examples
```bash
# Direct commit access (most efficient)
git clone gittorrent://7d234ffe4317d150166acb545b83712eac417b8d

# Commit with branch/tag references
git clone gittorrent://7d234ffe4317d150166acb545b83712eac417b8d?refs

# Legacy server-based (hybrid P2P + centralized discovery)
git clone gittorrent://github.com/user/project

# Username-based (requires external resolution)
git clone gittorrent://myusername
```

## Configuration

Configuration files are stored in:
- Linux: `~/.config/gittorrent/`
- macOS: `~/Library/Application Support/gittorrent/`
- Windows: `%APPDATA%\gittorrent\`

## Development

### Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with debug logging
RUST_LOG=debug cargo run --bin gittorrentd
```

### Dependencies

- **Tokio**: Async runtime for high-performance networking
- **libp2p**: Modern P2P networking stack with Kademlia DHT, transport security, and protocol negotiation
- **git2**: Git repository operations and object manipulation
- **ed25519-dalek**: Cryptographic signatures for secure identity and authentication
- **serde**: Serialization for network protocols and configuration
- **tracing**: Structured logging and debugging
- **clap**: Command-line interface parsing

## Contributing

Pull requests are welcome! The project follows standard Rust conventions:

- Use `cargo fmt` for code formatting
- Ensure `cargo clippy` passes without warnings
- Add tests for new functionality
- Update documentation as needed

## License

MIT License. Copyright (c) Erica Windisch

Credit to Chris Ball for his inspirational work on the original Javascript-based Gittorrent.
