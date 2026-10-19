# Unified Frontier catalogue preservation preparation

`preserve_study.py` adds immutable studies to **new** matching sidecars named
`unified_frontier_studies.json` and generated `unified_frontier_studies.md` in two
explicit destination directories. It never edits the historical `refinement52`
catalogues, their 24 existing rows, or numerical campaign files. This directory contains the publication helper and synthetic fixtures. Actual
completed study packets live in the additive result sidecars and numerical archives.

Use review directories first. After parent review, the intended JSON destinations
are methodology `src/simfolio_forecasting_methodology/resources/results/` and
master `catalogs/`. Reports can be copied to the respective documentation trees.
Repository integration, discoverability links, consumer modules/tests, commits and
pushes remain a separate parent-reviewed step against current remote `main`.

```sh
python3 preserve_study.py --evidence-root /absolute/evidence/root \
  --study-packet /absolute/closed-study.json \
  --methodology-dir /absolute/review/methodology \
  --master-dir /absolute/review/master
python3 -m unittest -v test_preserve_study.py
```

Both destination directories must exist. Existing sidecars must agree. An exact
repeat is idempotent; a changed study with the same ID is rejected. Add later
results under a new study ID referencing the prior study, preserving the earlier
record. Validation completes before output is rendered/written. This is a local
preparation helper, not a cross-repository transactional publication mechanism.

## Study packet schema

The executable synthetic fixture factory in `test_preserve_study.py` shows every
required field without claiming real scores. The packet is JSON with:

| Field | Contract |
| --- | --- |
| `schema`, `study_id` | `unified-frontier-study-v1`, immutable `UF-*` identifier |
| `purpose`, `methodology`, `uniform_asset_law` | Nonempty explanation; one asset-level law must be true |
| `variants` | Label, research ID, methodology, retained/rejected/diagnostic decision and reason for **every** variant, including failures. Optional `panel_labels` maps `52`/`80` to existing saved-control label spellings. |
| `revisions`, `sources` | Repository-to-full-40-character-commit map; evidence-root-relative `{path, sha256}` refs for methodology, decision ledger and relevant implementation sources |
| `data_manifest_sha256` | Frozen canonical identity `3825258f753ba0a4ec653c56403ef7e163e80ea89eb1c06856b6415160672a76` |
| `panels` | Exactly `52` and `80`; each has its own completion status and scores |
| `growth_diagnostics` | `status`: `not_evaluated` or `recorded`; `scope`: `diagnostic_no_historical_return_floor`; recorded evidence refs |
| `runtime_evidence` | `not_measured` or `representative_matched_compute`; explicit scope; `production_browser_ctv: false` |
| `validation` | `canonical_use: exploratory_reused_panels`; separate `seed_robustness` and `independent_post_selection` objects with `not_tested`/`tested` status and evidence refs when tested; `production: not_validated` |
| `archive_receipt` | Ref to the normalized independently verified archive closure described below |

The packet may include additional explanatory metadata, prior-study IDs, source
citations, no-lookahead checks and mechanism decisions; it is preserved verbatim.
The helper validates file closure, **not the truth of a prose methodology claim**.
Canonical panels were already used in development, so their scores never establish
independent post-selection validation. No aggregate blended-panel winner is made.

An unscored panel is exactly `{status: not_evaluated, reason: ...}`. A complete
panel uses `status: complete_scored_and_audited`, a complete `scores` map, saved
`controls` refs including `M261_saved_control` and at least one fast/reference
control, `runner_sources`, and seven `evidence` refs:

- `manifest`: original frozen run manifest; its `task_identities[panel]`,
  `labels[panel]`, `simulations`, `source_hashes` and `snapshot_sources` are checked.
- `reference_manifest`: original canonical panel manifest with the unchanged
  task-identity list and frozen data hash. Dates, seeds, training/realized/future
  digests and task IDs are retained; the complete identity list must match.
- `completion`: original dual-panel receipt with `complete_origins`,
  `required_origins`, and explicit per-panel `failures`.
- `results`: original per-panel result (`complete_origins`, `required_origins`,
  `complete_portfolios`, `scores`).
- `analysis`: independent reducer output keyed by panel, with full denominators
  and `models[label].CRPS`. Published `scores` must match these canonical C-order
  means exactly. Original result scalars either match exactly or use the narrowly
  closed roundoff receipt below, with explicit `raw_results_scores` in the packet.
- `cells`: original final full-panel cell tensor file.
- `audit`: normalized publication audit closing all the preceding files.

The full52 contract is 52 assets, 2,652 origins, 455,832 cells/model,
3,904,212 origin-horizon contributions; full80 is 80 portfolios, 4,080 origins,
701,280 cells/model, 6,006,480 origin-horizon contributions. Both use 8,766 daily
horizons and 240 paths/origin. Counts remain separately hashed, not inferred from
cell counts. Controls and runner sources must be bound by the original manifest.
All manifest/snapshot source refs must exist unchanged and be archive-closed.

Matched runtime evidence requires both `cases.single_asset` and
`cases.multi_asset`, `fresh_and_cached_fit: true`, evidence refs, and
`decomposition: [fitting, dependence, simulation, accounting]`. This is a
representative research-compute record, never a browser CTV or all-work speed claim.
Timings from concurrent campaign totals do not meet this contract.

## Independent publication audit contract

The live `summarize.py` was not changed. Its current audit is necessary but lacks
explicit hashes for both final cell tensors and per-panel results. After numerical
completion, a **separate independent auditor** must reconstruct every checkpoint,
validate all task identities/seeds/input hashes against the frozen references,
rebuild cell sums and counts byte exactly, verify saved controls byte exactly, and
close the resulting final artifacts. Merely wrapping old booleans is insufficient.

The normalized audit schema is `unified-frontier-publication-audit-v1` with:

- `panel`, `independent_from_runner: true`, frozen `data_manifest_sha256`,
  canonical `complete_origins`, and `failures: 0`.
- True gates: `all_task_identities_match_reference`,
  `all_cells_and_counts_rebuilt_byte_exactly`, `all_saved_controls_byte_equal`,
  `all_bound_sources_and_controls_match`.
- `artifact_sha256`: exact role-to-hash map for all six non-audit evidence refs.
- `auditor_source`: separately hashed auditor implementation ref, distinct from
  the run sources. The source ref must also be archive-closed.
- `auditor_dependencies`: the pinned nested audit implementations. For the
  current publication auditor these are `summarize.py` and `growth_summary.py`.
- `required_publication_source_refs`: all source refs required to reproduce the
  publication audit, including `auditor_source` and every `auditor_dependencies`
  ref. Every ref must hash-verify, appear exactly in `study.sources` (same path
  and SHA256), and appear in archive `closed_artifacts` with the same hash.
  Archiving the wrapper alone is insufficient. Both arrays are mandatory.
- `task_identities_sha256`: SHA256 of the original task list serialized with
  Python `json.dumps(value, sort_keys=True, separators=(",", ":"))` in UTF-8.
- `counts_sha256`: SHA256 of the auditor's rebuilt canonical counts array bytes
  in C order; record dtype/shape explicitly as supplementary fields.
- `denominator_sha256`: canonical-JSON digest of `{origins, portfolios,
  simulations, cells_per_model, origin_horizon_contributions, horizons,
  counts_sha256}` with exact canonical values above.

The helper verifies this closure and all local hashes; it does **not** rerun the
raw-cell audit. A trusted independently closed auditor is the numerical authority.
The helper’s fixtures deliberately use text stand-ins for cells, so their success
proves catalogue logic only, never any numerical result.

### Scalar aggregation order

Original result scalars can differ from canonical C-order cell means by a few
ULPs even when every saved cell/count byte is unchanged. Such differences are
accepted only via the independently computed audit field `score_aggregation`:

```text
schema: unified-frontier-scalar-aggregation-v1
method: float64_gamma_n_mean_absolute_cells
dtype: float64
canonical_order: C
n: canonical cells_per_model
epsilon: binary64 machine epsilon (2**-52)
gamma_n: n*epsilon / (1-n*epsilon)
canonical_scores: exact label -> canonical C-order cell mean
raw_results_scores: exact label -> untouched original results.scores
per_label: label -> {
  mean_absolute_cells: independently computed mean(abs(C-order cells)),
  absolute_difference: abs(raw-canonical),
  error_bound: gamma_n * mean_absolute_cells,
  ulp_difference: abs(uint64(raw)-uint64(canonical)),
  within_bound: true
}
```

The consumer recomputes the fixed binary64 bound, scalar difference and ULP count
exactly, requires each difference within its derived bound, and requires complete
matching label sets. It accepts no financial tolerance or manually chosen error
budget. The auditor must compute `mean_absolute_cells` from the bound cell tensor,
never select it to accommodate an observed discrepancy. The enclosing audit’s six
artifact hashes must close the unchanged raw results, canonical cells and analysis;
the complete auditor implementation/dependency closure remains mandatory.
The auditor verifies every CRPS cell is nonnegative; accordingly the consumer
requires `mean_absolute_cells == canonical_scores[label]` exactly, preventing
inflation of the derived error budget through that field.

When this receipt is present, the panel packet must preserve
`raw_results_scores` exactly. `scores` is the canonical publication representation;
analysis and audit canonical scores must equal it exactly. Reports retain raw
differences and ULP counts. A scalar receipt cannot excuse any changed cell,
control, input, task identity, count or denominator. Closed numerical files are
never rewritten to force scalar equality.

## Parent-supplied archive closure

After a completed batch is independently downloaded/restored and member-verified,
the parent supplies a small normalized receipt:

```json
{
  "schema": "unified-frontier-archive-closure-v1",
  "study_id": "UF-the-immutable-study-id",
  "status": "verified",
  "independent_download_and_member_verification": true,
  "source_input_seed_environment_closure": true,
  "remote_receipt_url": "immutable remote receipt URL",
  "restore_command": "the verified restore command",
  "verification_sources": [{"path": "original-receipt.json", "sha256": "..."}],
  "closed_artifacts": {"evidence-root-relative/path": "verified-member-sha256"}
}
```

`verification_sources` must bind actual original archive/member manifests, restore
and independent download receipts, including separately archived dependency
closures. `closed_artifacts` maps every packet evidence/source/auditor file to its
verified archive-member hash, including dependencies. Preserve raw receipt
identity, release, URLs, environments, canonical-data and prior-source closures in
the original bound receipts. This helper does not download or independently
authenticate remote receipts, authorize deletion, or run cleanup.

Avoid circular identity: the study packet, normalized closure and catalogue
helper are not required members of the numerical archive they describe. Numerical
and auditor sources **are** immutable archived evidence. Version the catalogue
helper/tests/docs separately in Git. Fixing this helper therefore does not require
rearchiving the raw numerical batch. Parent review and fresh canonical/master
integration tests are required before any push. Nothing here establishes a winner,
a finished research stopping condition, or production readiness.
