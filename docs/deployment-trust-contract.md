# Deployment trust minting and Metal consumption contract

This is the operator and Metal #38 contract for deployment registry trust. The
scope is one Hyprstream deployment, not a PDS or host. For the demo, the same
deployment CA, authority log, independently trusted log-head checkpoint, and
current registry credential are projected to both PDS hosts. The v1 credential
has no PDS/host claim; consumers must not invent one.

## Fixed runtime artifacts

Metal's root-owned projection service installs these files atomically after
validating them through `hyprstream trust verify-deployment`:

| Artifact | Fixed path | Required representation |
| --- | --- | --- |
| Deployment public CA | `/etc/hyprstream/trust/deployment-ca.hybrid` | Exactly 1,984 raw bytes: 32-byte Ed25519 public key followed by 1,952-byte ML-DSA-65 public key |
| Authority rotation log | `/etc/hyprstream/trust/deployment-authority.log.json` | Root-anchored `hyprstream.deployment-authority-log.v1` JSON; public metadata, required for every credential |
| Authority head checkpoint | `/etc/hyprstream/trust/deployment-authority.head.json` | Independently provisioned `hyprstream.deployment-authority-checkpoint.v1` JSON containing the expected domain, log DID, sequence, and head CID |
| Registry credential | `/run/hyprstream/credentials/registry-service.jwt` | Compact hybrid JWT with no whitespace or newline, profile `hyprstream.registry-deployment.v1`, audience `urn:hyprstream:service:registry`, lifetime at most 3,600 seconds |

Every fixed-path file and parent directory is a regular root-owned OS path and
must not be group/world writable. The verifier requires the log and checkpoint for
direct-root and delegated credentials alike. A supplied log is accepted only
when its verified DID, sequence, and head CID exactly equal the independently
provisioned checkpoint; a signature-valid historical prefix therefore cannot
reactivate a retired authority. The UCAN issuer must also remain in the
checkpointed any-of-N active authority set. Replacing/removing an authority
revokes both its direct credentials and delegations without changing the
original 1,984-byte root pin. Logless genesis validation is a separately named,
one-time tooling mode and is never inferred from a missing production file.

For rootless development and `systemd --user`, the daemon also supports the
explicit loader contract below:

```text
HYPRSTREAM_DEPLOYMENT_TRUST_DIR=/absolute/trust/directory
CREDENTIALS_DIRECTORY=/absolute/systemd/credentials/directory
```

The trust directory supplies `deployment-ca.hybrid`,
`deployment-authority.log.json`, and `deployment-authority.head.json`.
`CREDENTIALS_DIRECTORY` supplies `registry-service.jwt`. Each variable controls
only its named side of the split; an absent variable falls back to the fixed
path in the table, while an invalid present value fails startup without
fallback. User-service paths require real, non-group/world-writable files and
ancestors owned by root or the daemon's effective user, and symlinks are
rejected.

This is path and ownership resolution only. It does not add a genesis mode or
relax the 1,984-byte hybrid CA, exact checkpoint/log binding, JWT profile,
audience, freshness, claim-shape, delegation, or hybrid-signature checks.

## Metal #38 input schema

Metal/OpenTofu accepts this reference-only object. These field names and
semantics are pinned:

```hcl
deployment_trust = {
  public_ca = {
    secret_arn = "exact cloud-secret ARN"
    version_id = "immutable version"
    sha256     = "lowercase SHA-256 of the raw 1,984 bytes"
    size_bytes = 1984
  }
  authority_log = {
    secret_arn = "exact cloud-secret ARN"
    version_id = "immutable version authorized for this rollout"
    sha256     = "lowercase SHA-256 of the exact JSON bytes"
    schema     = "hyprstream.deployment-authority-log.v1"
    max_bytes  = 65536
  }
  authority_checkpoint = {
    secret_arn = "exact cloud-secret ARN"
    version_id = "same rollout as authority_log"
    sha256     = "lowercase SHA-256 of the exact checkpoint JSON bytes"
    schema     = "hyprstream.deployment-authority-checkpoint.v1"
    max_bytes  = 65536
  }
  registry_credential = {
    secret_arn             = "exact rotating cloud-secret ARN"
    version_stage          = "AWSCURRENT"
    profile                = "hyprstream.registry-deployment.v1"
    audience               = "urn:hyprstream:service:registry"
    max_ttl_seconds        = 3600
    refresh_before_seconds = 300
  }
}
```

OpenTofu sees only these references and public integrity values. It must never
read, render, output, log, or store the CA/JWT values in state, plans, tfvars,
user data, or cloud-init. The exact secret ARNs and required KMS encryption
context are the entire instance IAM read scope. The projection unit retrieves
the referenced values at runtime, validates the CA length/layout, authority-log
root and exact independently trusted head, JWT
profile/audience/freshness/signatures, then uses
write-fsync-rename to replace the fixed files. A failed fetch or verification
never replaces the last valid file, and startup fails when no current valid
set exists.

The mint and verifier cap the rotating JWT, public authority log, and checkpoint
at 65,536 bytes so each remains a valid cloud-secret value. The delegation names
the stable authority-log DID instead of embedding its growing DidOp history, so
hourly JWT size does not grow with rotations. A deployment that exhausts the
log's 65,536-byte append budget must perform the documented new-CA
reprovisioning flow; v1 does not claim a history-compaction protocol.

The CA, authority-log, and authority-checkpoint versions are immutable inputs
for a rollout. The log and checkpoint are minted and durably committed together,
and Metal must project them as one rollout unit; a mixed pair fails closed. The
JWT intentionally follows `AWSCURRENT` and is refreshed before expiry. An
authority rotation is a deliberate Metal input update to the new immutable log
and checkpoint versions; it is not the hourly JWT refresh path.

## Out-of-band mint and publish flow

The root authority is never placed in AWS, Metal, Terraform/OpenTofu, an
instance, or an online refresh job. Generate it on an operator machine with an
any-one-decrypts age recipient ring:

```console
hyprstream trust mint-deployment-ca \
  --public-ca deployment-ca.hybrid \
  --authority-key deployment-ca.age \
  --authority-log deployment-authority.log.json \
  --authority-checkpoint deployment-authority.head.json \
  --yubikey age1yubikey1PRIMARY... \
  --yubikey age1yubikey1BACKUP... \
  --recipient age1BREAKGLASS...
```

At least two distinct root recipients are enforced; three (primary YubiKey,
backup YubiKey, offline break-glass) are the operational baseline. Native age
recipients and age-plugin recipients are interchangeable and any one can
decrypt. `--piv-slot` optionally keeps the Ed25519 signing leg in a YubiKey PIV
slot with always-touch policy. To prevent that device from becoming a sole
holder, the same Ed25519 seed exists only as a recovery copy inside the
multi-recipient age bundle; using it requires the explicit
`--software-recovery` break-glass flag. YubiKey does not provide native
ML-DSA-65: the PQ seed also remains inside that age-encrypted bundle. Age's
standard native at-rest recipient is classical X25519; a future PQ-HSM plugin
can use the same recipient interface.

Touch the rare authority once at deployment setup to authorize a separate
online hybrid signer:

```console
hyprstream trust delegate-registry-signer \
  --public-ca deployment-ca.hybrid \
  --authority-log deployment-authority.log.json \
  --authority-checkpoint deployment-authority.head.json \
  --authority-key deployment-ca.age \
  --yubikey-identity primary-yubikey-identity.txt \
  --signer-recipient age1ONLINE_KMS_OR_SERVICE_IDENTITY... \
  --delegated-key registry-delegated-signer.age \
  --delegation registry-signer.delegation.json
```

The signed UCAN grants only
`hyprstream://deployment/<deployment_domain>/service/registry` with ability
`mint-registry-jwt`, exact profile/audience/domain/delegated-key caveats, and a
3,600-second JWT ceiling. It cannot delegate further or mint any other token.
The online job uses it hourly without the root:

```console
hyprstream trust mint-registry-jwt \
  --public-ca deployment-ca.hybrid \
  --authority-log deployment-authority.log.json \
  --authority-checkpoint deployment-authority.head.json \
  --identity online-signer-identity.txt \
  --via-delegated-signer registry-delegated-signer.age \
  --delegation registry-signer.delegation.json \
  --registry-public-key registry-service.ed25519.pub \
  --ttl-seconds 3600 \
  --jwt registry-service.jwt \
  --contract deployment-trust.publisher-manifest.json
```

The common-path key is named by `--via-delegated-signer`; `--authority-key`
is used only by the explicit rare/bootstrap `--root` alternative.

The JSON `--contract` output is
`hyprstream.deployment-trust-publisher-manifest.v1`: it contains local paths,
base64 artifact values, and fingerprints for an out-of-band secret publisher.
It contains no authority key. Because it contains the JWT and public artifacts,
it is sensitive operational input and must never be passed to OpenTofu. The
publisher writes the public CA, authority log, and authority checkpoint to
immutable cloud-secret versions, rotates only the JWT's `AWSCURRENT` version
hourly, and returns the reference-only Metal object above through a separate
control path.

Verify the exact bytes before publishing:

```console
hyprstream trust verify-deployment \
  --public-ca deployment-ca.hybrid \
  --authority-log deployment-authority.log.json \
  --authority-checkpoint deployment-authority.head.json \
  --jwt registry-service.jwt \
  --contract deployment-trust.publisher-manifest.json
```

## Authority rotation, revocation, and loss recovery

`hyprstream trust rotate-authority --add` signs a DidOp head that retains the
old active keys and adds the new hybrid key. Publish and roll out the new
authority log and its same-transaction checkpoint, create a new delegation with
the new authority, then use `--replace` to publish a later log/checkpoint pair
that retires the old set. During the add
phase, delegations from retained authorities remain valid because they name the
same log DID; after replacement, the same delegations fail because their issuer
is no longer active.

Loss of one age recipient is recovered by decrypting with another, then
rotating the recipient ring and authority as appropriate. Full loss of every
root recipient is not cryptographically recoverable: generate a new deployment
CA, re-provision `/etc/hyprstream/trust/deployment-ca.hybrid`, the new authority
log, and its head checkpoint on every instance, issue a new delegation, and
replace the registry credential.

Threshold M-of-N signing is deliberately deferred. FROST-style Ed25519
threshold schemes do not solve the ML-DSA-65 leg, whose threshold signing and
hardware support remain research-grade. The authority-set/DidOp seam supports
operational key rotation today without claiming PQ threshold security.
