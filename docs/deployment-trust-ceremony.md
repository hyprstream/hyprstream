# Deployment trust: the root ceremony and the autonomous path

**Who this is for:** whoever holds the hardware tokens, and whoever has to reason about what an
automated deployment is allowed to do without them.

This document covers the **operational** half of deployment trust — which keys are minted by a human
at a ceremony, which are minted by automation, and what each one can do. The cryptographic design is
in [`cryptography-architecture.md`](cryptography-architecture.md); the wire contracts are in
[`deployment-registry-trust.md`](deployment-registry-trust.md). Neither of those says who holds what,
which is the gap this fills.

The ceremony procedure was exercised against a real YubiKey 5C NFC. The command and credential
contracts in this document are normative only where they agree with
[`deployment-trust-contract.md`](deployment-trust-contract.md) and
[`deployment-registry-trust.md`](deployment-registry-trust.md); those contracts win on disagreement.

---

## The property we are buying

> **The root of trust is offline and physically gated. Everything the fleet does day to day runs on
> short-lived, narrowly-scoped credentials that the root authorized once.**

Concretely: an automated deploy can mint the credential a registry needs, all day, unattended. It
**cannot** mint a new authority, rotate the root, or widen its own scope — those require a human
holding a token. Compromising the whole automation fleet does not compromise the deployment root.

## Three layers

| Layer | Command | Who runs it | Hardware needed |
|---|---|---|---|
| **Root** — the deployment authority | `trust mint-deployment-ca`, `trust rotate-authority` | human, at a ceremony | no token to *create*; token/backup identity to *use* |
| **Delegation** — a scoped online signer | `trust delegate-registry-signer` | human, at a ceremony | **token: PIN + touch** |
| **Autonomous** — the actual credential | `trust mint-registry-jwt --via-delegated-signer` | automation | **none** |

The asymmetry is the design: **creating** the root needs no token (it only *encrypts to* public
recipients), while **using** the root costs a human a PIN and a physical touch, because using it means
decrypting the authority bundle.

```
   ceremony (rare, human, touch-gated)          autonomous (constant, unattended)
   ─────────────────────────────────────        ────────────────────────────────
   mint-deployment-ca   ── authority ──►  delegate-registry-signer
                                                  │  emits scoped signer + delegation
                                                  ▼
                                          mint-registry-jwt --via-delegated-signer
                                                  │  ≤1 hour, registry-only
                                                  ▼
                                            deployed service
```

## Rules the tooling enforces

**A sole token is refused.** The root demands at least two distinct age recipients:

```
Error: root authority requires at least two distinct age recipients
       (primary + backup/break-glass); a sole YubiKey is forbidden
```

This is not bureaucratic. A single token is a single point of *loss* — lose it and the deployment
root is unrecoverable. The intended topology is **hardware primary + break-glass backup**.

**The delegation is scoped and expiring.** `delegate-registry-signer` emits, verified:

```json
"scope": { "aud": "urn:hyprstream:service:registry",
           "sub": "service:registry",
           "profile": "hyprstream.registry-deployment.v1",
           "max_ttl_seconds": 3600 }
```

The online signer cannot mint outside `service:registry`, and nothing it mints outlives an hour —
regardless of the 30-day (max 1-year) delegation lifetime.

**The authority bundle is kept offline.** The mint output records
`"authority_key_export_allowed": false`; publisher manifests additionally assert
`"private_authority_exported": false`. Those assertions do not make the local encrypted bundle
magically non-copyable: operators must keep it out of deployed hosts, repositories, IaC state, and
publisher manifests.

## ⚠️ The break-glass recipient must be encrypted

**This is the rule most easily gotten wrong, and the tooling cannot enforce it.**

The CLI verifies you supplied *two distinct recipients*. It has no way to check that the second one
is actually protected — a recipient string looks identical whether its private half lives on a
hardware token or in a world-readable file.

`age-keygen -o backup.key` writes an **unencrypted** identity. If you hand that recipient to
`mint-deployment-ca`, the deployment root is decryptable by anyone who can read that file, and the
hardware token guarding the primary path becomes irrelevant. **A root is exactly as strong as its
weakest recipient.**

The break-glass **MUST** be one of:

1. **A second hardware token** — best; the same properties as the primary.
2. **A passphrase-encrypted identity** — `age-keygen | age -p > backup.key.age`, passphrase held
   separately from the file.
3. **Split offline media** — Shamir-split or sealed and stored apart.

An unencrypted identity file is acceptable **only** for a throwaway validation root that protects
nothing and is destroyed immediately afterward.

## Hardware: what your token actually protects

There are two distinct YubiKey roles, and conflating them overstates what you have.

| Flag | What it does | Firmware |
|---|---|---|
| `--yubikey age1yubikey1…` | Token acts as an **age recipient**: it decrypts the authority bundle. The bundle's private key is briefly **in host memory** during a ceremony. | any PIV-capable (verified on **5.4.3**) |
| `--piv-slot <SLOT>` | The Ed25519 root is generated in host memory, imported into the PIV slot, and its recovery seed is stored inside the age-encrypted authority bundle — so any bundle recipient can reconstruct the software signer. The token only **gates signing** behind PIN + touch; it does **not** confine the key. | **≥ 5.7.4** — Ed25519-in-PIV. **Not verified here** (test token was 5.4.3) |

Neither role confines the Ed25519 key to the hardware. Both the `--yubikey` age recipient and the
`--piv-slot` import path place the bundle private key (or its recoverable seed) inside host memory
and inside the age-encrypted authority bundle. The PIV path additionally gates signing behind a
hardware touch prompt, but a bundle recipient can still sign without the token by reconstructing
the software signer from the seed. Say which one you are running, and do not let "we use YubiKeys"
imply hardware confinement that neither path actually provides.

`--kms-plugin` accepts a cloud/PQ-HSM age-plugin recipient for organizations preferring an HSM.

### Token preparation

After a PIV reset, verify the PIV PIN, PUK, and management-key posture before generating an age
identity. A PIV reset returns those PIV values to factory defaults; it does not reset FIDO2,
OpenPGP, OATH, or the whole device. Do not automate this interactive step or capture it through a
pipe.

**Override the plugin's PIN-policy default.** It defaults to `once`, which caches the PIN for a
session so one entry can authorize several root operations. For a deployment root:

```bash
age-plugin-yubikey --generate --name "<deployment> root" \
  --pin-policy always --touch-policy always
```

`always` means every decryption costs a PIN **and** a touch.

## The ceremony (verified end to end)

Run in a clean directory. **All paths must be absolute** — see Gotchas.

### 1. Prepare the token

```bash
age-plugin-yubikey --generate --name "hyprstream deployment root" \
  --pin-policy always --touch-policy always
age-plugin-yubikey --identity > "$CEREMONY/yubikey-identity.txt"
```
Record the printed `age1yubikey1…` **recipient** (public — safe to store anywhere).

### 2. Prepare the break-glass — encrypted (see the rule above)

Second token preferred. If using software, passphrase-encrypt it and store the passphrase separately.

### 3. Mint the root — no touch required

```bash
hyprstream trust mint-deployment-ca \
  --yubikey    "age1yubikey1…" \
  --recipient  "<encrypted break-glass recipient>" \
  --public-ca            "$CEREMONY/deployment-ca.hybrid" \
  --authority-key        "$CEREMONY/deployment-ca.age" \
  --authority-log        "$CEREMONY/deployment-authority.log.json" \
  --authority-checkpoint "$CEREMONY/deployment-authority.head.json"
```

Produces a **1984-byte** hybrid public root (32-byte Ed25519 ‖ 1952-byte ML-DSA-65), the
age-encrypted authority bundle, a signed rotation log, and an anti-rollback checkpoint. Record
`public_ca_sha256` out of band — it is what downstream verification pins.

The public artifacts install to `/etc/hyprstream/trust/`. **`deployment-ca.age` does not** — it stays
operator-held and never reaches a deployed host.

### 4. Delegate an online signer — **PIN + touch**

```bash
hyprstream trust delegate-registry-signer \
  --public-ca "$CEREMONY/deployment-ca.hybrid" \
  --authority-key "$CEREMONY/deployment-ca.age" \
  --authority-log "$CEREMONY/deployment-authority.log.json" \
  --authority-checkpoint "$CEREMONY/deployment-authority.head.json" \
  --yubikey-identity "$CEREMONY/yubikey-identity.txt" \
  --signer-recipient "<recipient automation can decrypt>" \
  --delegated-key "$CEREMONY/registry-delegated-signer.age" \
  --delegation    "$CEREMONY/registry-signer.delegation.json"
```

The token will prompt for a PIN and require a physical touch. **This is the ceremony.** If it ever
completes without one, stop and investigate — the root is not gated.

For the ordinary age-recipient design, break-glass decrypts `deployment-ca.age` with the separately
held encrypted recovery identity passed via `--identity`; it does not require `--software-recovery`.
`--software-recovery` is only for the distinct PIV-backed-authority mode, where it authorizes use of
the age-wrapped recovery copy of that PIV Ed25519 key.

### 5. Autonomous minting — no token, no root

```bash
hyprstream trust mint-registry-jwt \
  --registry-public-key "<raw 32-byte Ed25519 service public key>" \
  --public-ca … --authority-log … --authority-checkpoint … \
  --via-delegated-signer "$DEPLOY/registry-delegated-signer.age" \
  --delegation           "$DEPLOY/registry-signer.delegation.json" \
  --identity             "$DEPLOY/online-signer.key" \
  --jwt      "$DEPLOY/registry-service.jwt" \
  --contract "$DEPLOY/deployment-trust.contract.json"
```

Verified: succeeds with **no root bundle and no hardware token**. The credential is the required
`ML-DSA-65-Ed25519` composite, `typ: wit+jwt`, bound to
`urn:hyprstream:service:registry`, and capped at one hour. This is what a deploy runs.

Verify anything with `hyprstream trust verify-deployment`, which uses the production verifier.

#### What unattended refresh costs you

The credential above expires in an hour, so a deployed host re-mints it on a timer with no
operator present. That is only possible if the host can decrypt the delegated signer by itself —
so `--identity` (`online-signer.key` above) must be resident on the host, root-owned and `0600`,
where `hyprstream trust install --refresh-identity` puts it.

Accept this deliberately: **root compromise on a deployed host yields the delegated signer**, and
with it the ability to mint registry credentials until the delegation expires. That is the price of
unattended operation, and it is why the delegation is scoped to the registry service and given a
short life rather than being a second root. It is *not* a path to the root authority — the root
bundle is sealed to a disjoint recipient set that this identity cannot open, and it never touches a
deployed host.

Two consequences worth planning for: keep the delegation TTL short enough that a compromise window
you would tolerate is the same window you actually have, and treat rotating the delegated signer —
not just the JWT — as a routine operation rather than an incident-only one.

### 6. After the ceremony

- `deployment-ca.age` → offline storage with the token. Never onto a deployed host.
- Public artifacts (`deployment-ca.hybrid`, authority log, checkpoint) → `/etc/hyprstream/trust/`.
- Delegated signer + delegation → the deployment environment.
- Online-signer identity → the deployment environment, for unattended refresh only (see above).
- **Destroy the ceremony working directory** (`shred` key material, then remove it).

## Gotchas (each cost real time)

**Relative paths fail.** Every default is a bare filename, and using them dies with
`Error: inspect output parent / No such file or directory`. Pass absolute paths for every input and
output.

**`age-plugin-yubikey --generate` cannot be piped.** Through `| tee` it fails with
`Failed to get input from user: IO error: not a terminal`. It needs a real TTY — correct for a
ceremony tool, but it means the ceremony cannot be captured by naive output redirection, and an
automated wrapper will not work.

**A build-queue slot is not durable storage.** If you build the binary through a shared/pooled
`CARGO_TARGET_DIR`, copy it out before using it — pooled slots are reclaimed and your binary will
vanish mid-ceremony.

## Running the ceremony from the wizard

`hyprstream wizard --deployment-trust` runs the same four steps after node bootstrap. It is opt-in:
without the flag the wizard sets up node-local trust only, exactly as before.

The wizard enumerates attached tokens with `ykman list` and picks the mode from the firmware it
reports, saying which one it chose and why:

| What it finds | Mode |
|---|---|
| no token | software recipients, labelled **dev-grade** on screen, in the summary, and in `deployment-trust-mode.json` |
| firmware < 5.7.4 | `--yubikey` age-recipient mode |
| firmware >= 5.7.4 | `--piv-slot` mode (slot `9c` by default) |
| several tokens | asks which one; `--deployment-trust-serial <SERIAL>` answers ahead of time |
| detection failed | **refuses to continue** — not knowing whether hardware is attached is not the same as knowing it is absent |

A software root is never selected while a token is attached unless `--deployment-trust-software` says
so explicitly. The break-glass prompt defaults to `age-keygen | age -p` and refuses a bare identity
file unless `--deployment-trust-allow-plaintext-break-glass` is passed.

`--non-interactive` never reaches a prompt: without `--deployment-trust` the phase does not run at
all, and with it, an attached token stops the run with both ways forward rather than hanging on a PIN
that no one is there to type.

The wizard leaves the public artifacts in the ceremony directory; installing them to
`/etc/hyprstream/trust/` is still the operator's step (see *After the ceremony*).

> The `--piv-slot` path remains **unverified on real 5.7.4 hardware**. The firmware rule that selects
> it is tested; what `ykman piv keys import` and `yubico-piv-tool` do on a token is not.

## Local development

`dev/local-e2e/` mints a **throwaway** deployment authority, standing in for the offline ceremony so
a rootless local stack can run without hardware. That is correct for local dev and **must never be
used for anything real** — throwaway material, software recipients, no hardware gating.

The registry *service* signing key is an **online autonomous key**, not ceremony material: it is
generated locally and then authorized by a root-signed delegation. Do not confuse it with the
deployment authority. If you find yourself about to make an automated path generate something at the
*root* layer, stop — that is the property this whole design exists to protect.
