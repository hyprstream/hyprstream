# Offline service identity bootstrap

Before starting separate network service containers, initialize the deployment
checkpoint store, provision the intended service keys through the existing
credential workflow, and admit their identities:

```sh
hyprstream pds init-deployment-store
hyprstream pds provision-services \
  --service registry,policy,discovery,event,model,streams,oai,oauth \
  --valid-for-seconds 86400
```

The command is for an OS-owned deployment with existing, authenticated public
CA/authority/checkpoint files and a current registry deployment JWT. It uses
the configured shared credentials directory: Policy uses its root signing key;
other services use their service subdirectories. Missing keys are errors. The
registry signing key must match the authenticated deployment credential.

Run the command before the registry opens the store. It takes the normal sole
writer lock and uses the same guarded genesis/successor admission and atomic
checkpoint writer as the registry RPC. It does not insert synthetic identity
records, clear security history, weaken signature checks, or start a network
listener. It verifies the full requested roster before reporting success.

Each identity advertises its canonical `#service` entry, signed public hybrid
request KEM (`requestKem`), and the Iroh carrier address derived from that
service's existing key. The address is not used as
application identity. Genesis commits the existing hybrid signer for the first
successor; that successor has a bounded expiry and retains future commitments.

Retries reuse checkpoint-verified identities, including a genesis accepted
before an interrupted first-successor operation. A live matching identity is
left unchanged while at least half the requested lifetime remains. Otherwise,
the command advances the existing chain through the guard and preserves other
current keys, service entries and commitments. Ambiguous matching identities,
uncommitted signers, renewal of terminal identities and invalid checkpoints fail closed.

The default lifetime is 24 hours; accepted values are 600–86400 seconds. This
command does not install an automatic renewal timer. Operators must renew
before expiry, either through the authorized live registry successor-ingest
RPC or in a maintenance window with the registry stopped.

In `network-iroh-required` mode, native process bootstrap authenticates the
OS-owned deployment artifacts and projects Discovery and Policy reach directly
from checkpoint-verified accepted state. These two fixed roles must match the
authenticated service keys, have current bounded successors, and carry signed
Iroh reach and complete hybrid request/response material. Missing or invalid
authority fails closed. Ordinary service names still resolve through Discovery
RPC. The bootstrap client binds an outbound Iroh carrier and resolves lazily;
it does not require Discovery or Policy to answer during client construction.

Discovery starts before Policy in a combined required-profile process. After
its Iroh carrier binds, Discovery publishes through an owned handle sharing its
actual state store and normal JWT/accepted-state validation. Required readiness
waits for the first successful publication, then signals internal readiness and
systemd `READY=1`. Failed publication or cancellation does not signal readiness.
Refreshes reread current accepted state and stop on service shutdown. Separate
processes retain their own credentials; Policy loads its provisioned service
JWT without an RPC to itself. Required Policy clients and the policy CLI resolve
over Iroh. The required service loop does not bind a local RPC socket.

In a split required-profile deployment, start Policy and Discovery concurrently
once offline provisioning and key generation finish. Policy must not be ordered
`After=Discovery` (nor Discovery after Policy readiness): Discovery probes the
canonical Policy revocation/session stores before its factory can bind. Policy's
Iroh handler binds before its announcement wait, so it can answer authenticated
probes while Discovery starts. No authority store is bypassed or replaced.

Only required Iroh Policy startup retries initial announcement availability
failures. The wait has a 120-second monotonic deadline and 500ms retry delay;
each connect retains the transport's existing 10-second timeout. Every retry
reprojects current signed state and the current service credential, and each
publication is additionally bounded by that credential and identity's expiry.
Only typed pre-dispatch address-discovery, connect-timeout and backoff failures
retry. Authentication/authorization rejection and invalid local authority are
terminal. A deadline, cancellation or failure never signals READY. Successful
initial publication is followed by the existing 25-second refresh cadence.
Set these bootstrap units' `TimeoutStartSec` above the application wait plus
startup allowance (180 seconds is the intended staging contract). This requires
a concurrent launch contract in deployment units, not a readiness-order cycle.

The pinned bootstrap MAC table does not yet declare the Policy revocation/session
probe leaves or Discovery announcement. The reviewed generated inventory and
its production MAC consumer must provide these declarations before a complete
isolated-process IdentityAware startup can succeed. Publication retry does not
supply policy labels, widen activation or replace the #1506 human decision.

Compatibility clients permit validated QUIC or Iroh reach; browser provisioning
retains its explicit WebTransport profile. Systemd notification sockets carry
lifecycle signals only. Event/Streams MoQL networking and deployment unit wiring
are separate integration work; this RPC bootstrap does not establish their
readiness or complete deployment validation.

Normal output includes only service names, public DIDs, epochs and completion
counts. Private key bytes and credential token contents are never printed.
