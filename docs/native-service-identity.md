# Required-native split-service identity

A required-native foreground service selects its provisioned identity before
installing the process resolver or constructing a client. That same key becomes
the ServiceContext process identity and the named service key used to project
checkpoint admission proofs. `--ipc` is not needed to select a service identity.

For the eight-service staging roster, each container executes one of:

```
hyprstream service start event --foreground
hyprstream service start discovery --foreground
hyprstream service start policy --foreground
hyprstream service start registry --foreground
hyprstream service start streams --foreground
hyprstream service start model --foreground
hyprstream service start oai --foreground
hyprstream service start oauth --foreground
```

Each uses the existing deployment configuration:

```
HYPRSTREAM__QUIC__ENABLED=true
HYPRSTREAM__QUIC__IROH=true
HYPRSTREAM__QUIC__NATIVE_NETWORK_PROFILE=network-iroh-required
HYPRSTREAM__SECRETS__PATH=/var/lib/hyprstream/config/credentials
HYPRSTREAM_SECRETS_PROFILE=shared-directory
```

The secrets profile defaults to `shared-directory`. Policy uses the canonical
flat `credentials/signing-key`; other services use
`credentials/SERVICE/signing-key`. With an explicitly scoped credential provider,
`HYPRSTREAM_SECRETS_PROFILE=per-service-scoped` selects the flat `signing-key` in
that service's configured directory. It must actually be scoped by the provider.
Config-specified `[secrets].path` is honored by identity loading and service-start
credential consumers; the existing secrets-path environment override retains
precedence. Deployment trust, enrolled public anchors and required service JWTs
must still be mounted/provisioned through their existing contracts.

Startup reads the existing 32-byte seed. Missing or malformed service keys fail;
startup does not generate a replacement, fall back to the CLI/Policy key, or
create a CLI `.registry/keys` identity. Provisioning/keygen is a separate earlier
step. No signing-key bypass is introduced.

`service start --foreground --services model` selects `model`, not the command's
synthetic `multi` name. Required-native foreground/standalone startup with more
than one service is rejected before resolver installation: the process-global
MoQL client proof currently represents one service identity. Launch separate
containers. Compatibility-profile identity selection and its existing transport
flags retain their behavior.

This change does not activate MAC policy, issue user/service grants, solve the
cold-start authority cycle, provision CLI proofs/PEPs, or implement credential
renewal. It neither changes Policy's canonical identity nor resolves #1506's
AsOriginator/RFC8693 decision. Streams PR1583 and Event PR1581 remain prerequisites;
real staging authorization and crosshost probes still require the other tracked
source and deployment work.
