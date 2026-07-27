# Rootless local signup → inference E2E harness

Status: **staged, intentionally not activatable at the current source head**.
The files here author a native `systemd --user` fabric; no service, trust mint,
cloud resource, Tofu operation, container, or privileged port was started while
staging it.

The rootless loader implementation is no longer a design/code blocker. Open
stacked PR #1373 at `a3d38414e0dbeb0fd2d1495c3799dc119bf11c39`
implements the exact selector seam and has two independent security-review
PASS verdicts. It is stacked on still-open #1371 at `3997039a9`; current
`main` contains neither composed stack. `bringup.sh` therefore remains
fail-closed before `systemctl --user daemon-reload` or any service start until
#1371 then #1373 are merged/composed into the binary source.

## What is staged

- native release build: `cargo build --release --bin hyprstream`
- exact daemon invocation:
  `hyprstream --config PATH service start NAME --foreground --ipc`
- one Policy, one Discovery, one Registry, and one OAuth/PDS process
- two isolated CPU inference processes, replicas `0` and `1`
- PR #1371 rare-root → delegated signer → one-hour registry JWT mint
- development CA and `*.accounts.localhost` P-256 TLS certificate
- optional dnsmasq fixture on unprivileged `127.0.0.1:53535`
- a suffix-identical rootless adapter for final reviewed Metal #37 at `167a53ac`
- exact #1373 rootless trust-loader environment and pre-start source guard
- static validation and a browser/manual E2E boundary driver

The PDS/authorization-server origin is
`https://pds.accounts.localhost:6791`. Inference binds UDP `7440` and `7441`.
Policy and Discovery expose authenticated Unix sockets only.

## Metal #37-compatible projection

Metal confirmed this user-runtime adapter on 2026-07-27 against `3c6f360e`,
then rebased it without contract drift to `0003cb0a`. The final reviewed and
restacked #37 head is
`167a53ac3a39e61b8f4e9f06988f8a68dd5dd9b5`. All five projection inputs
remain byte-equivalent between `0003cb0a` and `167a53ac`:

| Projection input | Identical Git blob |
|---|---|
| `runtime.tf` | `e8e2b50fb84ff894b6e01a79d3a2e4a8ec3686be` |
| inference Quadlet template | `d8f44f10da2c9f971c852367ee53b4e5f6281e4a` |
| runtime bootstrap template | `94f525fbb6380529e677b73984470fdcc46a38d9` |
| runtime fixture | `108237f482f488c56d886562160e46641a7356e1` |
| inference contract fixture | `d41ef103e79e76aa866d4ac6d47d885659fff097` |

Only separate DNS/trust files changed in the enclosing module. Durable source
material lives under the gitignored `dev/local-e2e/state/`; the ephemeral
projection is:

```text
$XDG_RUNTIME_DIR/hyprstream/
├── policy.sock
├── discovery.sock
├── credentials/
│   ├── registry-service.jwt
│   ├── trust/
│   │   ├── deployment-ca.hybrid
│   │   ├── deployment-authority.log.json
│   │   └── deployment-authority.head.json
│   ├── tls/
│   │   ├── quic-chain.pem
│   │   └── quic-key.pem
│   ├── inference-cpu-0/{signing-key,service-jwt,...}
│   └── inference-cpu-1/{signing-key,service-jwt,...}
├── inference-cpu-0/
│   ├── policy.sock -> ../policy.sock
│   ├── discovery.sock -> ../discovery.sock
│   └── inference.sock
├── inference-cpu-1/...
├── inference-cpu-0.sock -> inference-cpu-0/inference.sock
└── inference-cpu-1.sock -> inference-cpu-1/inference.sock
```

All projected directories are real `0700` directories. Credential/TLS files
are regular, non-symlink `0600` files. The credentials tree is assembled in a
fresh same-filesystem staging directory, validated, and renamed into place.
Unexpected links or an already-populated shared runtime are rejected.

Each inference process has its own directory selected by
`HYPRSTREAM__SECRETS__PATH`, sets
`HYPRSTREAM_SECRETS_PROFILE=per-service-scoped`, and has a distinct
`HYPRSTREAM_INSTANCE=inference-cpu-N`. Current Hyprstream bootstrap mints one
`service:inference` identity; that same authorized identity is copied into two
separate, non-fallback directories, matching #37's
`identity_reference = "identity://inference"`.

This emulates #37's bind mounts with checked relative links. It does **not**
claim the cloud projector's SELinux Enforcing state,
`hyprstream_inference_t`, immutable OCI digest, container UID/GID,
read-only/capability-dropped filesystem, or landed gate evidence.

## Verified code contract

| Surface | Verified value | Source/ref inspected |
|---|---|---|
| native binary | `target/release/hyprstream` | workspace `[[bin]]`, `crates/hyprstream/src/bin/main.rs` |
| daemon CLI | global `--config`; `service start <name> --foreground --ipc` | `crates/hyprstream/src/cli/commands/mod.rs`, generated unit code |
| user runtime | `$XDG_RUNTIME_DIR/hyprstream` | `crates/hyprstream-rpc/src/paths.rs` |
| fabric sockets | `policy.sock`, `discovery.sock` | `crates/hyprstream-rpc/src/paths.rs` |
| PDS/AS | OAuth service, HTTPS TCP `6791` | `OAuthConfig`, `services/oauth/mod.rs` |
| signup | PAR + `/oauth/authorize`, POST `action=signup` | open #1370 at `2bbdf3b6c` |
| service auth | `GET /xrpc/com.atproto.server.getServiceAuth` | `feat/37-get-service-auth` at `6a2b35e13` |
| session | `POST /api/session/exchange`, `GET /api/session/whoami`, cookie `hyprstream_session` | `feat/session-exchange` at `c69937d53` / #1354 queue ref |
| inference | service `inference`, replicas 0/1, UDP `7440/7441`, aliases `inference-cpu-N.sock` | `feat/1236-1247-inference-service` at `241392795`, Metal #37 at `167a53ac` |
| trust mint | `mint-deployment-ca`, `delegate-registry-signer`, `mint-registry-jwt`, `verify-deployment` | `feat/trust-mint-v2` at `3997039a9` |
| rootless trust loader | `HYPRSTREAM_DEPLOYMENT_TRUST_DIR` for three fixed leaves; `CREDENTIALS_DIRECTORY` for `registry-service.jwt` only | open #1373 at `a3d38414e`, double-review PASS |
| deployment CA | raw 1984 bytes: Ed25519 32 + ML-DSA-65 1952 | #1371 trust implementation |
| registry JWT | profile `hyprstream.registry-deployment.v1`, audience `urn:hyprstream:service:registry`, max 3600 seconds | #1371 trust implementation |

The mint script uses #1371's actual repeatable `--recipient`,
`--identity`, and `--signer-recipient` flags. It derives the registry's raw
32-byte Ed25519 public key from the service's raw seed, and the production
`verify-deployment` command verifies all four public artifacts after minting.

## Prerequisites

- Linux with a working user systemd manager and `XDG_RUNTIME_DIR`
- Rust/Cargo toolchain required by this workspace
- normal Hyprstream native build dependencies, including its libtorch setup
- `age`, `age-keygen`, `openssl`, `xxd`, `curl`, `git`, and `systemd-analyze`
- an immutable model directory inside a Git checkout:
  exact `HEAD` OID and no tracked, untracked, **or ignored** changes beneath it
- a non-loopback local IPv4 address for signed inference advertisement

The inference code rejects loopback/unspecified advertisement. Binding remains
`0.0.0.0:7440/7441`, but signed `advertise_addr` must be a non-loopback IP with
the matching port.

## Static verification

This is safe to run during staging. It does not build, mint, contact a network,
or invoke systemctl:

```bash
dev/local-e2e/verify-staged.sh
```

## Activation after blockers land

Do not remove the trust guard. The harness now generates #1373's exact required
environment:

```text
HYPRSTREAM_DEPLOYMENT_TRUST_DIR=$XDG_RUNTIME_DIR/hyprstream/credentials/trust
CREDENTIALS_DIRECTORY=$XDG_RUNTIME_DIR/hyprstream/credentials
```

The first selector supplies only `deployment-ca.hybrid`,
`deployment-authority.log.json`, and `deployment-authority.head.json`; the
second supplies only `registry-service.jwt`. `require-user-trust-loader.sh`
checks both the generated environment and that the checkout being built
contains #1373's selector implementation. Current `main` fails that pre-start
check. Merge/compose #1371 then #1373 and the rest of the E2E source stack,
then rerun static review before activation.

Choose an immutable model checkout and a non-loopback address:

```bash
MODEL=/absolute/path/to/model-checkout
OID="$(git -C "$MODEL" rev-parse HEAD)"
IP=192.0.2.10  # replace with an address assigned to this workstation

dev/local-e2e/bringup.sh --execute \
  --model-path "$MODEL" \
  --model-oid "$OID" \
  --advertise-ip "$IP"
```

`--execute` is mandatory. The controller then builds, generates local TLS,
bootstraps service identities, mints/verifies deployment trust, atomically
projects credentials, installs user units, starts Policy then Discovery, checks
their sockets, creates the rootless #37 links, starts the remaining services,
and waits for both inference sockets and OAuth metadata.

The two throwaway rare-root identity files are under the gitignored
`dev/local-e2e/private/`. All other durable generated state is under the
gitignored `dev/local-e2e/state/`. Never publish either.

Run the automatable boundary checks and display the manual browser segment:

```bash
dev/local-e2e/run-e2e.sh --execute
```

It exits `2` after successful surface checks because the landed tree has no
non-browser owner for the complete DPoP/signup/MoQ sequence. It will not report
an E2E pass from socket existence. After browser exchange, the session check
can be added with:

```bash
LOCAL_E2E_SESSION_COOKIE='cookie-value' \
  dev/local-e2e/run-e2e.sh --execute
```

Teardown preserves durable secrets/state but stops units and removes only an
ephemeral tree carrying this harness's exact ownership marker:

```bash
dev/local-e2e/teardown.sh --execute
```

## DNS/TLS fixture boundaries

`*.localhost` resolves to loopback on the staging workstation; the optional
dnsmasq file provides deterministic queries on port `53535` without modifying
the OS resolver. The harness never installs the `/etc/hosts` example.

The wildcard certificate covers the account zone and PDS host, and the PDS
serves it on unprivileged `6791`. A host-form account DID such as
`did:web:alice.accounts.localhost` cannot encode a port: external DID
resolution therefore requires HTTPS on TCP 443. A rootless user service cannot
bind 443 on this host (`net.ipv4.ip_unprivileged_port_start=1024`), and the
inspected PDS code does not yet expose account documents through a host-routed
`/.well-known/did.json` listener. Internal PDS storage-based resolution can
exercise signup, but full network `did:web` resolution is a separate blocker.

Trust the generated CA only in the test browser/process. Do not install it
system-wide.

## TODOs and blockers

1. **Trust loader is implementation-unblocked, merge/composition blocked.**
   Open #1373 at `a3d38414e` implements the reviewed rootless seam and is
   double-review PASS with green hosted checks. It is stacked on open #1371 at
   `3997039a9`. Current `main` has not composed them, so the exact environment
   is encoded but the source guard still refuses daemon pre-start.
2. **One composed Hyprstream revision.** The required surfaces are distributed
   across #1354, #1371, #1372, the cold-signup/#1370 work, and inference
   branches. No inspected revision contains the entire flow. Rechecked
   2026-07-27 09:35 UTC: #1370 is open and mergeable at `2bbdf3b6c`. The
   original orphan-genesis P1 and later trusted-409 defects have been
   remediated; exact-head R4 independent review and some hosted checks are
   still pending (Clippy and WASM were in progress), so it is review/check
   blocked—not still REVISE at `63cd7d911`.
3. **Gate revisions.** Metal #37 still requires exact landed revisions for
   #873, #1236, #1247, and #1267. Current `main` includes #1267 but is not the
   composed runtime acceptance revision.
4. **www !56.** The browser owns PAR, signup proof, DPoP key, token, and cookie.
   Its final origin/config contract must replace the documented
   `http://localhost:3000` assumption if different.
5. **Authenticated inference readiness/client.** Metal acceptance calls
   authenticated `isReady` and `healthCheck` and expects
   `modelLoaded=true,status=ok`. The current harness can validate the exact
   sockets but has no landed authenticated query/MoQ client, so application
   output remains manual.
6. **Host-form DID HTTPS.** Rootless port 443 and account-host document serving
   need a reviewed local solution. Do not weaken `did:web` or encode a port.
7. **First-hop routing.** #37 exposes replica-pinned reach and health aliases;
   the caller-owned authenticated load balancer remains outside this harness.
8. **Credential multiplicity assumption.** Bootstrap currently authorizes one
   `service:inference` key pair and the two replicas receive copies in isolated
   directories. If the landed inference contract requires distinct keys
   rather than one shared service identity, add an actual policy-authorized
   multi-instance mint path; do not synthesize JWTs.
