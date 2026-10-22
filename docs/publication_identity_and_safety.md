# Publication safety and execution identity

Publication approvals are manual review records, not automatically generated exemptions.
`resources/publication_safety_review.json` binds exact repository-relative paths and SHA256
bytes of previously published evidence. The reviewed categories permit only private archive
repository references, literal historical source imports, and local source paths retained for
recovery and reproducibility. A changed file or a new path has no approval. Reviewers must
classify the provenance and existing publication authorization before editing this manifest.
Do not regenerate approvals from scanner findings as an acceptance step.

Numeric data and archives require a separate exact-byte public-payload review. The initial
inventory retains already-published score/cell evidence, source archives, test fixtures and
the explicitly authorized canonical snapshot. It does not authorize new provider data,
private arrays or forecast path dumps. Unknown CSV, array, database and archive payloads
fail closed. Explicit private-provider/full-path JSON fields also fail closed. Content
scanning cannot infer whether arbitrary unlabeled numbers are private; publication review
and the source/data manifests remain required.

Credentials, private keys, signed URLs and credential-bearing URLs cannot be suppressed by
any provenance or payload approval. Archive members and numerical binary content are
scanned too. Unknown text continues to use the strict `scan_text` rules. The original
scientific evidence is never rewritten to make the scanner pass.

`french_panel.selected.source_identity` binds the artifact that actually executes. An actual
checkout must place the package under `src/` and have matching project metadata. Its keys
use the `source-tree-v1/` namespace and retain package, repository tool and definition hashes.
An installed wheel instead uses `installed-wheel-v1/<version>/` keys. Every packaged source
and resource plus distribution METADATA/WHEEL must have a matching SHA256 RECORD entry;
the RECORD must bind the executing module. Missing metadata, missing hashes or modified
bytes fail closed. Runtime bytecode and installer-only metadata are not scientific sources.
Metadata discovery is restricted to the executing package's installation directory, so
unrelated checkout egg-info or another installation on `sys.path` cannot shadow its RECORD.
Missing or ambiguous metadata in that actual directory fails closed.
These two identities intentionally differ: neither invents a Git revision, consults an
unrelated parent directory, nor claims that a wheel contains checkout-only research tools.
Changing identity metadata does not change a model's forecasting law.
