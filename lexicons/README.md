# AT Protocol compatibility contracts

`ai/` contains Hyprstream extensions. `upstream/atproto/` contains unmodified
upstream account, identity, repository, sync and social schemas, selected by
`atproto-selection.json`. The manifest pins the full upstream commit and SHA-256
of every original schema and license file. The current selection is 36 roots
with a 53-schema reference closure; dependencies do not imply implemented features.

Verify without any network access:

```sh
python3 scripts/check_atproto_lexicons.py
python3 -m unittest discover -s scripts/tests -p test_check_atproto_lexicons.py -v
```

To deliberately update the contract, change the full upstream commit or roots in
the selection file, run `python3 scripts/check_atproto_lexicons.py --refresh`, and
review the schema and manifest diff together. Refresh fetches only the pinned
bluesky-social/atproto revision, resolves references and validates their definition
targets before writing. Interrupted writes are detected by offline verification.
If the closure shrinks, refresh reports leftover files instead of deleting them;
review those removals explicitly. The adjacent upstream license texts are retained
verbatim and apply to the vendored files.

This is source-integrity/reference verification, not a Lexicon record validator,
PDS implementation or permission registry. In particular:

- Schema availability does not enable createAccount, sessions, repository writes,
  blobs or subscribeRepos on the running Hyprstream service.
- Replies are app.bsky.feed.post records, not a custom reply collection. Upstream
  embed dependencies are included even when their media services are not built.
- Standard AT account and OAuth interfaces adapt the existing native account and
  authorization model. JWT scopes cannot become a second authority system.
  Native policy and assurance enforcement remain mandatory.
- Custom native proofs, account relationships and delegation contracts must reuse
  the existing native interfaces; no new authority is implied by an ai.hyprstream
  namespace. Delegated/on-behalf-of behavior remains behind the recorded #1506
  human disposition, regardless of the issue's administrative open/closed state.
- Standard OAuth/permissions specifications are separate normative documents.
  This source pin does not claim to snapshot those documents or implement their
  complete behavior.

For repository adapters, preserve the upstream createRecord/putRecord validation
contract: true requires schema validation, false skips schema validation, and an
omitted value validates known schemas. Structural constraints, size limits and
authorization apply in every mode. This bundle is not a closed allowlist of all
collections a future generic PDS may store. Unknown collection content must not
gain permissions or be projected as a recognized social record.

Do not add mandatory native fields to com.atproto or app.bsky definitions. Native
private account administration and grant material do not belong in public records.
