# Recover the full-panel evidence

Every catalogue execution lists its exact `execution_fingerprint`, original
factory, implementation digest, panel fingerprint, scorer hash, seed contract,
240-path contract, source closure hash, numerical environment and archive member
paths. The ledger retains original tested identities; later verified factory
bridges do not create additional models or replace historical execution hashes.

The complete task loss vectors, native errors, original execution manifests,
implementations, inputs, environment and reports were preserved losslessly in
the linked private `Hutch2064/simfolio-engine` draft research releases. They were
freshly downloaded, fully disk-restored and SHA-256 verified before historical
local copies were retired. These are archival evidence releases, not deployments.
The catalogue records the historical recovery counts and exact archive/manifest
digests. A later lightweight remote check verifies current server archive digests
and fresh manifest bytes; it is not presented as a new full recovery.

For each needed archive in the execution's `archives` array:

```sh
gh release download RELEASE_TAG --repo Hutch2064/simfolio-engine \
  --pattern ARCHIVE_NAME --pattern MANIFEST_ASSET --dir recovery-download
shasum -a 256 recovery-download/ARCHIVE_NAME recovery-download/MANIFEST_ASSET
```

Compare both digests with `catalogue.json`. Authenticate to the private repository;
do not publish provider data or credentials. Draft releases are accessible through
`gh release download` by tag even when a tag lookup through the REST API returns
404. The remote verification tool discovers them through the releases listing.

Restore the lossless tar members into a new owned recovery directory, preserving
their paths relative to the original workspace. Verify every restored file against
the manifest (`entries`: path, bytes, sha256). Reject absolute paths, path traversal,
unexpected members and duplicate members. The original execution manifest and task
prefix named by each record identify the appropriate full-panel model. Required
parent implementation archives and the common source/input archive are linked in
the same record. Do not reconstruct losses from rounded chat numbers.

The unchanged scoring, scheduler and input-builder source snapshots are in
[`harness`](harness/); [SCORING.md](SCORING.md) specifies both v3 profiles. Reuse the
archived fixed input schedule and exact numerical environment for reproduction.
The input fingerprint is
`1c0b7714ec6f2581750fe214927cf03e887a8bbe52674f5488272c26965dd105`.
Reaggregation requires restored complete raw task losses or the archived exact
SQLite index. Missing intentionally archived local files are not failed forecasts
and do not justify rerunning models or changing the frozen scorer.

FAST's 17 native fitting errors are part of its complete *attempted* history.
Its 36,796 successful tasks must be compared with Public on those same tasks,
and labelled incomplete/preliminary. Do not fill failures with synthetic penalties
or compare that subset against another method's 36,813-task score.
