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

Each identity advertises its canonical `#service` entry and the Iroh carrier
address derived from that service's existing key. The address is not used as
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
RPC or in a maintenance window with the registry stopped. Identity provisioning
alone does not implement Discovery's network startup or the Iroh event runtime;
those remain required before declaring the networked deployment ready.

Normal output includes only service names, public DIDs, epochs and completion
counts. Private key bytes and credential token contents are never printed.
