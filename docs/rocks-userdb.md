# RocksDB User Store

## Overview

The `RocksDbUserStore` replaces `LocalKeyStore` (age-encrypted TOML) with RocksDB for atomic single-key updates and concurrent access. It supports multiple Ed25519 pubkeys per user, similar to GitHub SSH keys.

## Architecture

```
UserStore: username → {
    sub (UUID),
    profile (name, email, etc.),
    pubkeys: [
        { fingerprint, label, created_at, last_used_at },
        ...
    ]
}
```

### Key Scheme

```
user:{username}       → UserInfo (Cap'n Proto packed)
pubkey:{fingerprint}  → username (reverse lookup for auth)
```

### Storage Location

```
~/.config/hyprstream/credentials/users.db/
```

## Data Model

### UserInfo (Cap'n Proto)

```capnp
struct PubkeyEntry {
  fingerprint @0 :Text;      # base64url of pubkey bytes
  label @1 :Text;            # user-provided label
  createdAt @2 :Int64;       # unix timestamp
  lastUsedAt @3 :Int64;      # unix timestamp, 0 if never
}

struct UserInfo {
  username @0 :Text;
  sub @1 :Text;              # stable UUID (OIDC subject)
  name @3 :Text;
  email @4 :Text;
  emailVerified @5 :Bool;
  active @6 :Bool;
  externalId @7 :Text;       # upstream IdP identifier
  pubkeys @8 :List(PubkeyEntry);
}
```

### Fingerprint Format

Fingerprints are base64url-no-pad encoded SHA-256 hashes of the raw 32-byte Ed25519 public key:

```rust
fn fingerprint(pubkey: &VerifyingKey) -> String {
    use sha2::{Sha256, Digest};
    let hash = Sha256::digest(pubkey.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}
```

## API

### UserStore Trait

```rust
#[async_trait]
pub trait UserStore: Send + Sync {
    // Profile CRUD
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>>;
    async fn register(&self, username: &str) -> Result<String>;  // returns subject UUID
    async fn set_profile(&self, username: &str, profile: UserProfile) -> Result<()>;
    async fn remove(&self, username: &str) -> Result<bool>;
    async fn list_users(&self) -> Vec<String>;
    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>>;
    async fn set_active(&self, username: &str, active: bool) -> Result<()>;
    
    // Pubkey management
    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>>;
    async fn add_pubkey(&self, username: &str, pubkey: VerifyingKey, label: Option<String>) -> Result<String>;
    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool>;
    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>>;
    async fn touch_pubkey(&self, username: &str, fingerprint: &str) -> Result<()>;
}
```

## Auth Flows

### JWT (OAuth/OIDC)

1. User authenticates via external IdP or local OAuth
2. JWT contains `sub` claim (UUID)
3. Profile lookup by subject

### Ed25519 Challenge-Response

1. Client presents pubkey in envelope signer
2. Server computes fingerprint
3. Reverse lookup: `pubkey:{fingerprint}` → username
4. Verify signature against challenge
5. Update `last_used_at` timestamp

## CLI Commands

```bash
# Register user (creates UUID, no pubkey)
hyprstream user register alice

# Add pubkey (generates fingerprint, stores reverse index)
hyprstream user add-key alice --label "laptop"

# List pubkeys
hyprstream user list-keys alice

# Remove pubkey
hyprstream user remove-key alice <fingerprint>

# Show user profile
hyprstream user show alice
```

## Migration

### From LocalKeyStore

On first access, if `users.toml.age` exists but `users.db/` doesn't:

1. Decrypt and parse TOML
2. For each user entry:
   - Create RocksDB record with same `sub` (preserve UUIDs)
   - If `pubkey` field exists, add to pubkeys list with label "migrated"
   - Create reverse index entry
3. Rename `users.toml.age` to `users.toml.age.migrated`

### Rollback

If migration fails or rollback needed:
1. Delete `users.db/`
2. Rename `users.toml.age.migrated` back to `users.toml.age`

## Security

- RocksDB files are stored with 0600 permissions
- No encryption at rest (relies on filesystem permissions)
- Pubkey fingerprints are non-reversible (SHA-256 hash)
- Subject UUIDs are stable and never regenerated

## Comparison

| Feature | LocalKeyStore | RocksDbUserStore |
|---------|---------------|------------------|
| Storage | age-encrypted TOML | RocksDB |
| Concurrency | Full rewrite per mutation | Atomic key updates |
| Pubkeys | Single per user | Multiple per user |
| Reverse lookup | None | fingerprint → username |
| Encryption | age (X25519) | None (fs perms) |
