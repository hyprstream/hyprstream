# bootstrap-pubkeys format

`bootstrap-pubkeys` is a node-local credential file: the public keys of the
services a node must trust before Discovery is available to look anything up
(chicken-and-egg). It is unrelated to deployment trust (see
[deployment-registry-trust.md](deployment-registry-trust.md)) — it seeds an
unattested, non-expiring, local-trust-on-first-use entry per service, not a
CA-verified credential.

## Wire format

The file is a flat JSON object mapping service name to a base64-encoded
verifying key. The classical form — a 32-byte Ed25519 key — is shown first;
the hybrid form is described under [Hybrid entries](#hybrid-entries):

```json
{
  "policy": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
  "discovery": "q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s"
}
```

(The values above are synthetic 32-byte patterns for illustration — a real
value is any 43-character URL-safe-no-pad base64 string that decodes to the
32 raw bytes of the Ed25519 verifying key.)

- **Shape**: flat `{ "<service>": "<key>" }`. Not an envelope, not a list of
  `{service, key_type, public_key}` objects.
- **Base64 alphabet**: `URL_SAFE_NO_PAD` (RFC 4648 §5, no `+`, `/`, or `=`).
  Standard-alphabet base64 (with `+`/`/`/`=`) will fail to decode.
- **Key type**: an entry is either **classical** — exactly 32 raw Ed25519
  bytes after decoding — or **hybrid**, exactly 1984 bytes (see
  [Hybrid entries](#hybrid-entries) below). Any other decoded length is
  rejected. In particular, do **not** encode a DER/SPKI
  (`SubjectPublicKeyInfo`) blob — that decodes to 44 bytes and is rejected.
  If you have a PEM/DER key (e.g. from `openssl genpkey -algorithm ed25519`),
  extract the raw key first: the raw 32 bytes are the *last* 32 bytes of the
  DER SPKI (`openssl pkey -pubout -outform DER | tail -c 32`), then base64
  those with the URL-safe-no-pad alphabet.
- **Service names**: the loader (`load_bootstrap_pubkeys`) does **not**
  validate names — it accepts any JSON string key. `validate_service_name`
  (non-empty, at most 64 bytes, `[a-z0-9-]` only) runs producer-side only
  (wizard / bootstrap manager). Consumers look up the exact names
  `discovery` and `policy`, so anything else parses but is ignored.

Produced by `write_bootstrap_pubkeys` / `write_bootstrap_pubkeys_hybrid` and
consumed by `load_bootstrap_pubkeys` / `load_bootstrap_pubkeys_hybrid` in
`crates/hyprstream/src/auth/identity_store.rs`.

## Hybrid entries

A value may also carry a post-quantum key bound to the Ed25519 anchor. The
hybrid encoding is the **concatenation** `Ed25519 (32 B) ‖ ML-DSA-65 (1952 B)`
= 1984 bytes, base64'd with the same URL-safe-no-pad alphabet — byte-for-byte
the same layout the deployment CA root uses for its 1984-byte public pair.

The two forms are distinguished by **decoded length only**: 32 bytes is
classical, 1984 bytes is hybrid, anything else is an error. The file stays a
flat `{ "<service>": "<base64>" }` map, so a file may freely mix classical and
hybrid entries, and a file written before hybrid entries existed loads
unchanged.

```json
{
  "policy": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
  "discovery": "JIrL26-eBQGW3nBL6i1odw5RkVDRA7WH2uLZytU92TAXHtJkdXNEM19SNj1GPxnt-7mImdVPdRRENw5lnPLIVkDc3BiWsqDvmLI2x9roidQrSRArrQ-itYpEyQaLf-y6ckpa1NdkeN9ibwNu393M3E8PD2N3Vh9vx_sD2mbn532l0QXGV3i-9lQppfKcFwzinBkQN7eLLWRrInMfbE3IY_7FnC9Q0scH7bUMBvNY7k5jEDNtz1ZpEWgR-1u2M-iF221WgzT9AnQX5X88dd9gYs0SsIHLjx5o8-nnHk0XRxPq1JtegEupUKLf5a7e3IEm62VSXRJhH_V2IA9PAJMlSL8rJGYHdMwjscP1DyKCpCquSgYIH8zJDQCPqVUFBujtdq3x-eMFd0CxpVssPuLeoNb_4YZbtSwdk6iMMPupiDYIKOgknxymsnYGxR1PiA_7b1lZVaD62pCZHSCi9nJEDVoWuQ0_cSMB589zaDSTcI7r5Ks6l0MUPtVsa_yWQ4jwS8GnDMikXXmLvKegE5V6rZpG6AXx707kEvpxyESIzwnMU9UO2h4MGI1uWp36SXzHxF5RSZrY3Wr1ei1MwnJj7kgve4w1skjtZ32EE4A8r7R8mIl8L6LkXsD6n4FXFWQWocLdBerVvVBSDTDca8mQ3Lqr-3Q8wasUx-Ifqk6AAOZTmPseQ50uoiZnr0MCzxVkbAmvB5o_I5J_BqxKqQUimv0xiq69n9Vr__7FS8HTd9139LUPBkBDduZeH4qqJyIkDshEPJeXqOwBQZDIupnjc-PvDNKRM6o8Kg1OtaBjYwWWL6pVFRXYw3NeGb52N77-T9dEOCR659RHFrI-741xIaJ_gTJb6G43C7Go-EMI0aMkzakh-LSyTrv4C2s-KYksgVap-EsREaKmX1FAinmWsn0222V-tzJ-9dvPKKwCnTyTXpYRK3ltR-3LnP-onbg4ysP_tpVMg8K7jAeCzYsEEZCIjoJr5qpTkwGncfytwVvT6sHsoD988S10in7Dz9TNcNnFmb_foYqELht6jhAv_X6nf8IJEZgaLNaz9eV-_L87TdgQDF4iKrgHcuUZjPlfi-az1B6uV1Z7pqUsqJBT5XQkCqr3j4chZ2gNEw8SpconZVuXsT1tEtSWyo9GmvNntsiCvGM30faWGj-_cIsrCveMeMfLLsfNP3X7jG7Jgb7uYbj50Mwf7b-VqrdAmIJ0CMmCSygBaWV7YDx_qTEEuE7oL-R6_fXkhmXeZYyhFPF-As70CrVW-KwAJVp0mYasmPwu9YZ-Mlerg-QgHiAXnNpVjXE-wvnqP6JKbrbgaUZiLC4gXn0VTnB70y4e97Efl8ztGqE7AazPxwRMDyfher5oGpVm2lh9StBZYqaoAO7XQTUKvczld7imzqDugagVpWRW7ffE5qKp0ezq5PUwws4GB7UfTrmBm2gQ2HmT8aouV15b5zJaI1x9sOsNUjFrL1yL3V-Fi8adOXfRc5MR1JzdsaAki5YZhyjjYEHzdZfD8mwvN8jxN5U03tq495bR87ER_tHZNNd875gYZNYQ-pkD8zBlLWq54mDj5AKCwdTce3gOVTx9CtNYm3LFxsaB9VL_B93X81IR_Czaf_cHNQPsTpd8wWCWTZbc9zBhHUYtlrTrxhzXV68yBDcjnVF3LkbOcqapBCsUTrQ6XgloOp-fkUYd1LWTQN7ofcCXvR7x30BXXsM_qZnlUR4zminj2lWlJacV90Uqf7SL4HAEefqH8btyoVepUG_5JYQ7bp7NsUWUlw-Jc0hkumvNcaNX5uNFCLIIVTGgA9dG8ELuC3g-bhVjokIr4wSHVCyjFJs2qo93VdHy-abKdwpjeRwo6r66uaideZKEBxKUlZ_XHiZmT_FksJZIOHrq-8gQVPyjwo0oyI9Dq8xF-5aVN8NpHaJU7x4VcsndfHqsY5CGQUV0ZfuDpA94Srx-vvnvAJeM8ql63Ng2-0dNDra6fr8fgdEZeyJaIhdVBU8wBlshyeX32N2dbiJR7qWY5VqLbaKjePvqLa1LnnMtWdS2PqMX2UE_Q44Oa_FUdOB2sFdvKpxEbHDS74w5sd6C5AZzYPNUL1ICrV1F-c0Mxm9Z2j8PFEuYDz6fpR4Z1iDzLH9QKFUwus9fPzCU7b-ptTU3T3vnFTN4NLgxR6Ih78B-Mp3QrrCOxW3NtBNpnImuqHgRYBBPkg_dM659hgNtgCMYefiQuWJlpS22GG3kwPVuHoG8eqa-NDtxgMgFOPy30IKJcLh9ooG-QD9reZht_TBkLWbDW1JfeINAeJ86oZ5g3qtTdVAftVHaRCHLQN0ClLXPquM-v1GJm2F_tIXFJF-utsp0SgCDKo2VUymmMSgXQL3iP29lHQcp6Jzgisk6ZAEs6MqzrGV-T0cl0sQ4pKECzGHUL-8B8CKmMtshHZLCX4hvc3ALGd8ls72e6sh2JkFcU84I_8GLPgJu66dN5eZtOPR9CkmTKa2Ph1MgXqBGwmcYcvlg5GVZvzZC-UlhTrF7WAbObzDrGQbIGHBKG09dobSuSXpiSmZljJUh-88qsTXEVW12C4ZoWyDQ4OzH09JofMByTzb-k5Hn8cKIawcJBqCaAXElF5Z6MLrxxwagcL9XuMMdKG3UtEWY9HrsGY6qsA"
}
```

That `discovery` value is a real 2646-character encoding of a real key pair: it
decodes to 1984 bytes whose first 43 base64 characters
(`JIrL26-eBQGW3nBL6i1odw5RkVDRA7WH2uLZytU92TA`) are the Ed25519 anchor and whose
remaining bytes are the bound ML-DSA-65 verifying key. A test in
`identity_store.rs` decodes this exact string and asserts it loads as a hybrid
entry, so the example cannot drift from the parser. The `policy` value beside it
is a classical entry — mixing the two forms in one file is supported.

## Verification policy

Verification is **per identity**, matching the hybrid posture used elsewhere in
the system. There is no global "post-quantum required" switch:

- A service whose entry is classical (32 bytes) verifies **classically**. It is
  never asked for a post-quantum signature; that is the classical floor for
  identities that carry no PQ key.
- A service whose entry is hybrid (1984 bytes) requires **both** signatures —
  the Ed25519 signature *and* the ML-DSA-65 signature must be present and
  verify. A hybrid entry never verifies on its classical half alone.
- Offering a post-quantum signature against a classical entry is an error, not
  a silently ignored input: there is no bound key to check it with, and
  accepting it would let a caller believe a downgrade was a hybrid verification.

Upgrading a service is therefore a per-entry operation: rewrite one value in
the 1984-byte form, leave the rest alone. Nothing else in the file, and no
other node, has to change at the same time.

## Hybrid is mandatory for this node's own services

The two forms above describe what the *parser* accepts. What this node's
provisioning *writes* is narrower: **every service entry is hybrid**. There is
no classical provisioning mode and no flag to request one.

The ML-DSA-65 half of each entry is derived from that service's Ed25519 key
(HKDF, `hyprstream-mesh-mldsa-v1`) rather than generated independently. That is
required, not merely convenient: every signer in the tree derives its
post-quantum key the same way, so an independently generated key would publish
a public key that nothing ever signs with. It also means a service's secret
material stays a single Ed25519 seed — there is no second private key to
persist, protect, back up or rotate.

Consequently a classical **service** entry is not a supported configuration; it
is stale material from a pre-hybrid provisioning run. The runtime refuses it
with an actionable error when it resolves service keys, rather than trusting it
classically and then failing each RPC later with an opaque "no anchored
ML-DSA-65 signer key". The fix is to re-provision with `hyprstream wizard`.
Per-service keys and JWTs are preserved across a re-run, so the identities do
not change — only the published entries gain their post-quantum half.

The low-level loader still reads classical entries so tooling, and that error
itself, can report precisely which services are stale. This rule is scoped to
the node's own services: external classical clients and federated peers do not
appear in this file and are unaffected by it.

## Which service names matter

The wizard mints a keypair for every registered factory, but the runtime only
*resolves* two of them before Discovery is up:

- `discovery` — required. `install_process_production_resolver`
  (`crates/hyprstream/src/bin/main.rs`) resolves this on every command that
  reaches the shared production resolver; bootstrap/provisioning commands
  (`trust`, `wizard`, `pds init-deployment-store`, `pds join`, and the
  no-subcommand first-run wizard path) dispatch before it. Without it,
  deployment bootstrap fails with "trust store has no authenticated
  discovery key."
- `policy` — required for policy enforcement, but resolved by a different
  path: the standalone worker startup (when the CLI spawns its own
  `WorkerService` because the `worker` service is not already running)
  resolves `policy` to build its authorization client;
  `install_process_production_resolver` itself resolves only `discovery`.

Provisioning keys under any other service name (for example a name matching
some other artifact you have on hand) satisfies nothing — the loader will
parse the file successfully but the process will still fail to find the
`discovery` entry it actually needs.

## Not deployment trust

This file has no cryptographic tie to the deployment CA, the authority log,
or the registry-service credential. It is entirely local to the node and can
be regenerated by re-running `hyprstream wizard`. External provisioning
systems that also need to deliver deployment trust material (the four
`/etc/hyprstream/trust/*` and `/run/hyprstream/credentials/*` artifacts) must
provision those separately, in the formats documented in
[deployment-registry-trust.md](deployment-registry-trust.md) and
[deployment-trust-ceremony.md](deployment-trust-ceremony.md) — this format
does not apply to them.
