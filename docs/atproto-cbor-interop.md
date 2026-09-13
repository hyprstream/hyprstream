# Public AT encoding boundary

The PDS `DagCbor` implementation historically sorts raw UTF-8 text keys
lexicographically. Existing native capsule, operation and account artifacts use
that implementation. Its previous comments incorrectly claimed this was the
public AT encoding order.

Public DAG-CBOR compares encoded text keys, including their length header; for
text keys this is byte length first, followed by lexical bytes. See
https://ipld.io/specs/codecs/dag-cbor/spec/#strictness.

The additive `hyprstream_pds::atproto_cbor::{encode,decode}` boundary uses that
public order. It shares scalar parsing and the existing value type, preserves
strict parsing, and checks the signed-64-bit AT integer domain. The default
native encoder and decoder retain their existing behavior. Neither decoder tries
the alternate ordering after rejection.

The interoperability vector in the unit test was independently generated with
`@atproto/lex-cbor` 0.1.6:

```javascript
import {encode, cidForLex} from '@atproto/lex-cbor'
const record = {
  $type: 'app.bsky.feed.post',
  text: 'Hello from Torment Nexus',
  createdAt: '2026-09-09T00:00:00.000Z',
}
console.log(Buffer.from(encode(record)).toString('hex'))
console.log((await cidForLex(record)).toString())
```

The expected CID is
`bafyreiaapk47b4fmdslcizkj5aj6ws6bmpmvfvjq3ryhx5qwgug6tldalm`.
The upstream wire map orders `text`, `$type`, `createdAt`. The test checks exact
bytes and CID, rather than merely a round trip through our own implementation.
Negative cases cover cross-format rejection, duplicate keys, nonminimal widths,
invalid tags/UTF-8, trailing data, signed integer limits and excessive nesting.

## Integration and migration gate

This module is a serialization primitive, not a new authority, a schema validator
or a public PDS rollout. Existing `Commit`, MST, CAR and record callers do not
switch formats in this change. Public repository wiring must consistently use
the public format for every relevant block, with independent signature/MST/CAR
interop tests and a single-writer migration contract.

Do not re-encode a stored native signed artifact, replace a native DID or broaden
its canonical verifier to make public tests pass. Native security-critical data
keeps its accepted bytes and verification policy. An exported public artifact
gets its own bytes/CID and, where applicable, separately managed classical
repository signature. Existing native/public mappings must record both identities
explicitly; the encoding boundary cannot confer native authority on public input.
