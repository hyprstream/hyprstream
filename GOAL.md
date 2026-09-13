# Commander: native social / AT Protocol integration

Active goal: execute the shared commander integration plan while preserving native identity, hybrid assurance, policy enforcement and deployment gates.

This worktree owns the first independent change: a pinned, offline-verifiable upstream social/account Lexicon bundle and its dependency-closure checks. It does not own PR1585, Metal !331, #1506 disposition or a competing deployment.

Done conditions for this change:
- Exact upstream commit, original schema bytes and licensing provenance recorded.
- Selected account, identity, repository, sync and social definitions include their complete referenced definition closure.
- Offline verification rejects changed bytes, missing definitions and unexpected files.
- Tests and CI execute that verification without network access.
- Account and native-authority mapping remains explicit; schemas do not enable endpoints or confer authority.

Next changes: standard record interoperability fixtures and general repository support; native-authorized direct-self adapter and runner after their reviewed caller/proof/carrier contracts. Full lane acceptance remains the shared plan's responsibility and is not satisfied by this bundle.
