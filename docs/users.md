# Users and Keys

Hyprstream identifies humans by **username + Ed25519 public key**. There are no
passwords, no API keys to copy around, and no central directory — the user store
is local (RocksDB) or pluggable (Valkey, see `auth_config.md`).

This doc covers what an operator needs to manage that: where keys live, how the
CLI authenticates, and how to rotate.

## Identity model

A user record contains:

- `username` (string, unique)
- One or more registered public keys, each with: fingerprint (`SHA256:…`),
  base64-encoded raw Ed25519 pubkey, optional label, `created_at` timestamp

The store maintains a reverse index `pubkey:{fingerprint} → username` so that
services receiving a signed token can map it back to a user without scanning.

Multiple keys per user is the rotation handle — you add the new key, cut over,
then remove the old one.

## The local signing secret

Every host has a single per-OS-user Ed25519 signing key at:

```
~/.config/hyprstream/credentials/user-signing-key
```

This file is generated on first run (wizard or first CLI invocation needing it).
It is **not** the same thing as the registered pubkey:

- The file is the **secret** — used by `hyprstream sign-challenge` and the CLI's
  internal JWT minting.
- What gets **registered** in the user store is the corresponding *public* half,
  associated with a username.

You may bring your own key (e.g., an existing SSH Ed25519) by importing its
public half via `user keys import`. The local secret file is then optional —
but commands that need to sign locally (CLI JWTs, `sign-challenge`) will use
whatever is at the file path.

## CLI: managing users

```bash
hyprstream user register <username>      # Create user (no key required up front)
hyprstream user list                     # List all registered users
hyprstream user remove <username>        # Delete user and all their keys
```

`register` creates an empty user record. Add a key in a second step.

## CLI: managing keys

```bash
hyprstream user keys list   <username>
hyprstream user keys import <username> ssh-ed25519 --file ~/.ssh/id_ed25519.pub [--label laptop]
hyprstream user keys import <username> ssh-ed25519 --data "ssh-ed25519 AAAA… user@host" [--label laptop]
hyprstream user keys import <username> raw         --data <base64-32-bytes> [--label …]
hyprstream user keys remove <username> <fingerprint>
```

The fingerprint is the SHA-256 of the raw pubkey, in OpenSSH form
(`SHA256:base64-no-padding`). `keys remove` accepts the fingerprint with or
without the `SHA256:` prefix.

## How the CLI authenticates to services

The CLI does **not** go through OAuth/PKCE. Instead, on each invocation it:

1. Loads `user-signing-key`.
2. Derives a JWT-purpose signing key via HKDF-style domain separation
   (`derive_purpose_key(sk, "hyprstream-jwt-v1")`).
3. Mints a short-lived JWT with `iss = <oauth issuer URL>`, `aud = <OAI resource
   URL>`, `sub = <username>`, `exp = now + <duration>`.
4. Sends the JWT as a bearer token.

Receiving services look up the pubkey by `sub` in the user store, apply the
same purpose-key derivation, and verify the signature.

**Lifecycle:** the JWT is valid until `exp`. There is currently no
revocation list and no introspection endpoint — once minted, a JWT works
until it expires. The CLI mints a fresh JWT per command, so the effective
exposure window is "one command invocation" in practice.

`sign-challenge` uses the same `user-signing-key` but signs the OAuth
authorize-page nonce, for the browser PKCE flow. Same key material, different
purpose tag.

## Wizard behavior

Both the non-interactive (`hyprstream wizard --non-interactive --start`, using
the local OS username) and the interactive path (operator-entered username +
role) register identity the same way, via `register_local_identity`:

1. Generate `user-signing-key` if absent.
2. Create a user record for the username (OS user, or the entered name).
3. Call `add_pubkey(<username>, <pubkey>, label="wizard")` to bind the
   `user-signing-key` pubkey — this is the credential the CLI authenticates
   with, so it must be bound or the CLI falls back to `anonymous`.
4. (Non-interactive `--start`) Issue an initial bearer token for the user.

If the user already exists, the record is left alone and the pubkey bind is a
no-op when the fingerprint already matches. If the `user-signing-key` pubkey
was previously bound to a *different* user (e.g. an `anonymous` record left by a
prior partial run), it is **re-pointed** to the current user — there is exactly
one local user-signing-key, so it maps to exactly one identity.

## Key rotation

There is no `hyprstream user keys rotate` convenience command. Run an
**add → swap → remove** cycle so old and new tokens overlap during the
transition:

```bash
# 1. Generate a new key off-line
ssh-keygen -t ed25519 -f /tmp/hypr_new -N ""

# 2. Register the new pubkey alongside the old one
hyprstream user keys import alice ssh-ed25519 --file /tmp/hypr_new.pub --label rotation-2026-05

# 3. Verify both keys are present
hyprstream user keys list alice

# 4. Swap the local secret. Anything still using the old secret keeps working
#    until its tokens expire.
mv ~/.config/hyprstream/credentials/user-signing-key{,.old}
cp /tmp/hypr_new ~/.config/hyprstream/credentials/user-signing-key
chmod 600  ~/.config/hyprstream/credentials/user-signing-key

# 5. Once you're confident the new key is in use everywhere, remove the old one.
hyprstream user keys remove alice <old-fingerprint>
rm ~/.config/hyprstream/credentials/user-signing-key.old
```

**Compromise scenario.** If the secret has leaked, skip the overlap window:
`user keys remove` the compromised fingerprint immediately, then add the new
one. Any JWTs minted with the compromised key before removal remain valid
until their `exp` — there is no token revocation. Practical guidance: keep
`mint_local_token` durations short (minutes, not days).

## Known gaps

- **No `user keys rotate`** — the procedure above is manual.
- **No JWT revocation** — adding `jti` claims + a `jti-revocation` column
  family in the user store would close this, checked at verify time.
- **No `user show`** — to see a user's record, use `user keys list <name>`.
- **CLI authenticates with self-minted JWTs.** Services trust them because
  the pubkey is in the user store. Removing a user's last key effectively
  locks them out at next token mint, but does not invalidate already-minted
  tokens (see above).

## Related

- `auth_config.md` — user store backends (RocksDB / Valkey), SCIM
- `tls.md` — TLS modes and the unrelated JWT *signing* key rotation
