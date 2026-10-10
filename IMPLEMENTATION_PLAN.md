# Federate runtime wiring — implementation plan

## Current state
All Federate types exist and compile (federate_admission.rs 31KB, federate_policy.rs 8.8KB,
federate_host.rs). The gap is: no factory constructs FederateIssuer, no code installs it as
federate_issuer, and no serving route mounts the prepare/commit handlers.

## Minimal wiring changes

### 1. OAuth state initialization (state.rs ~line 1366)
Replace `federate_issuer: None` with:
```rust
federate_issuer: federate_issuer.map(Arc::new),
```
Where `federate_issuer` is constructed before the state and passed in.

### 2. OAuth service factory (factories.rs or the service constructor)
After the PolicyClient is available (for Policy prepare/commit RPC), construct:
```rust
let admission = Arc::new(PolicyFederateAdmission::new(policy_client));
let signer = Arc::new(FederateSignerImpl::new(signing_key)); // implement FederateSigner
let issuer = FederateIssuer::new(admission, signer)?;
state.federate_issuer = Some(Arc::new(issuer));
```

### 3. FederateSigner trait implementation
The `FederateSigner` trait (federate_host.rs:72) needs an implementation that signs
challenge responses. Use the OAuth service's existing signing key or a dedicated
Federate signer key from the enrollment manifest.

### 4. UserStore prerequisite removal
federate_admission.rs currently requires `PolicyAccountReader` to map (issuer, sub)
to username + active profile + account UUID. If Policy prepare/commit already returns
sufficient session/authority data, the local account link can be omitted. Otherwise,
document the residual (suspension/revocation/ownership/audit) and don't fake success.

### 5. Route mounting
OAuth mounts /oauth/federate/challenge already. Verify the route handler correctly
delegates to the FederateIssuer when federate_issuer is Some.

## Key construction parameters
- FederateIssuer::new(admission: Arc<dyn FederateAdmission>, signer: Arc<dyn FederateSigner>)
- PolicyFederateAdmission::new(client: PolicyClient)
- The PolicyClient must connect to the real Policy service

## Tests needed
1. Positive: FederateIssuer challenge + prepare + commit with a real Policy
2. Negative: wrong issuer, wrong DID, missing grant, replay, revocation
3. Session persistence after commit
4. Policy grant enforcement at Registry/Model use-time
