# TLS Configuration

Hyprstream supports three TLS modes, configured via `[tls] mode` in the config file.

## Modes

### `self-signed` (default)

Auto-generates a self-signed certificate at startup using `rcgen`. Suitable for development
and air-gapped deployments where a CA is not available. No renewal occurs.

```toml
[tls]
mode = "self-signed"
server_name = "localhost"
```

**Limitations:**
- Browsers show a security warning — clients must accept or pin the cert.
- WebTransport has a 14-day cert validity limit; self-signed certs use a longer validity
  and therefore require `x_cert_hash` pinning via the OAuth metadata endpoint.
- Not suitable for production internet-facing deployments.

### `acme`

Obtains and renews certificates automatically via the ACME protocol (RFC 8555).
Works with Let's Encrypt (public CAs) or self-hosted ACME servers (step-ca, Pebble).

```toml
[tls]
mode = "acme"
acme_domain = "node.example.com"
acme_contact = "mailto:ops@example.com"
# acme_directory defaults to Let's Encrypt production
acme_cache_dir = "/var/lib/hyprstream/acme"
```

**Renewal:** `rustls-acme` handles TLS-ALPN-01 challenges and certificate renewal
automatically. No explicit rotation is needed — the ServerConfig updates in-place.

**No `x_cert_hash` needed:** CA-signed certificates are trusted via the OS trust store,
eliminating the need for certificate pinning.

#### Let's Encrypt

Use the default `acme_directory` (or set it explicitly):

```toml
[tls]
mode = "acme"
acme_domain = "node.example.com"
acme_contact = "mailto:ops@example.com"
acme_directory = "https://acme-v02.api.letsencrypt.org/directory"
```

#### Self-hosted ACME with step-ca

[step-ca](https://smallstep.com/docs/step-ca/) is an open-source (Apache 2.0) ACME CA
ideal for private/air-gapped networks. Install it as a single Go binary.

```bash
# Initialize a new CA
step ca init --name "Hyprstream CA" --provisioner admin --dns node.example.com

# Start step-ca
step-ca $(step path)/config/ca.json
```

```toml
[tls]
mode = "acme"
acme_domain = "node.example.com"
acme_contact = "mailto:ops@example.com"
acme_directory = "https://ca.internal:9000/acme/acme/directory"
```

Clients must trust the step-ca root certificate:

```bash
# Export and install root cert on clients
step ca root root_ca.crt
sudo cp root_ca.crt /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

#### Testing with Pebble

[Pebble](https://github.com/letsencrypt/pebble) is Let's Encrypt's lightweight test ACME
server. Use it for end-to-end ACME testing without real DNS or certificates.

```bash
podman run -p 14000:14000 -p 15000:15000 letsencrypt/pebble
```

```toml
[tls]
mode = "acme"
acme_domain = "localhost"
acme_contact = "mailto:test@example.com"
acme_directory = "https://localhost:14000/dir"
acme_cache_dir = "/tmp/hyprstream-acme"
```

Pebble uses a self-signed root; set `PEBBLE_VA_NOSLEEP=1` for faster challenges:

```bash
PEBBLE_VA_NOSLEEP=1 LIBTORCH_BYPASS_VERSION_CHECK=1 cargo run -- serve
```

### `files`

Load a certificate and key from PEM files. The operator is responsible for renewal.

```toml
[tls]
mode = "files"
cert_path = "/etc/hyprstream/tls/cert.pem"
key_path = "/etc/hyprstream/tls/key.pem"
```

Files mode is also activated when `cert_path`/`key_path` are set without an explicit `mode`.

## JWT Signing Key Rotation

Independent of TLS, hyprstream rotates the JWT signing key automatically.
Keys cycle through three slots: **lead** (pre-published) → **active** (issuance) → **drain** (verification only).

```toml
[oauth]
jwt_key_active_days = 14   # How long a key is used for issuance
jwt_key_lead_days = 7      # Pre-generation window before active expires
jwt_key_drain_days = 30    # Verification-only window after active expires
```

Keys are persisted in `$HYPRSTREAM_SECRETS_DIR` as:
- `jwt-signing-key.active` / `jwt-signing-key.active.meta`
- `jwt-signing-key.drain` / `jwt-signing-key.drain.meta`
- `jwt-signing-key.lead` / `jwt-signing-key.lead.meta`

The JWKS endpoint (`GET /oauth/jwks`) serves all three slots simultaneously.
Clients that cache JWKS will automatically pick up new keys before rotation.

The rotation task checks every 6 hours. No restart is required.
