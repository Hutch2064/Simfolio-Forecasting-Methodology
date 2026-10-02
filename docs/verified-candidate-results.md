# Independently Audited Candidate Results

`simfolio-oos candidate-scores --json` reports a 176-row numeric sort: the 175
retained canonical score rows plus one separately registered candidate. Lower
exact empirical CRPS is better. The report preserves each historical
`canonical_rank` and assigns a separate `display_rank`; the new candidate has
`canonical_rank: null` and `display_rank: 1`.

| Paired-run rank | Model | Exact empirical CRPS | Paired-run evidence |
| ---: | --- | ---: | --- |
| 1 | Filtered-Innovation Moment-Matched SV with Fixed Historical Mean | 0.25439860867855635 | Newly executed; independently audited |
| 2 | Filtered-Innovation Moment-Matched SV with Dynamic Mean | 0.2547972662964723 | Newly executed in the same run; not added to the 176-row report |
| 3 | M001 — Asset FastMAP + Dynamic Gaussian Factor | 0.25554347087969403 | Reused independently audited incumbent reference |

The paired run used the canonical dense task constructor: 80 portfolios, 4,080
origin tasks, 240 simulations per origin, and 701,280 scored cells per model.
The selected candidate's exact empirical CRPS was 0.4480107424% lower than the
reused Frontier reference in that paired report. The candidate and dynamic-mean
candidate were newly executed; the Frontier loss vectors were reused from a
separately audited run with matching canonical task identities, data, future
calendars, seeds, source closure, numerical environment, and scoring. The
packaged cell-loss evidence retains those origins separately.

The full published-code replay evaluated all 4,080 candidate tasks and found
every per-task loss vector byte-equal to the scored checkpoint. Its receipt is
packaged at
`simfolio_forecasting_methodology/resources/verified_candidates/filtered_innovation_fixed_mean/full_replay_receipt.json`
and hash-checked by the report command. The recorded 501.900-second, four-worker
elapsed time is a run-context diagnostic. It is not a pure CLI throughput claim.
The receipt records hashes for the numerical source closure and prior native
parity evidence: the native and public `predictive_state_moments` and
`moment_return_curves` function ASTs match, and the compared 302,400,000 native
candidate-rank source values match byte-for-byte.

The candidate was replayed by directly instantiating the public model class and
calling the canonical origin-task evaluator. The recorded `cli.py` hash
predates the candidate selector and score-report wrapper. Those wrappers have
focused plan/report coverage; the 4,080-task numerical replay did not invoke
them. Separately, all 52 assets at three actual origins each had exact
filtered-pool, multiscale-field, 25,200-day mean/standard-deviation curve, and
37,440 empirical-quantile-node parity across the recorded Python/NumPy
environments. Only a sanitized summary and receipt hash are packaged; no fit
pickle or data payload is included.

## Interpretation

The 176-row display combines different evidence origins. Its new candidate row
is a full canonical execution; the other 175 rows retain their historical
tokens, ranks, and verification statuses. The historical M001 score in the
combined display (0.2557255171048505) is not the 0.25554347087969403 reused
reference from the new paired run. Those values belong to distinct evidence
records. The original 175-row ledger and membership digest are unchanged, and
the result does not claim a fresh execution of all 176 methods or a reproduction
of the historical white-paper ranking.

The candidate was selected adaptively from a search family of 23 candidates.
Accordingly, the reported rank and score are descriptive conditional on that
search. Ordinary single-comparison HAC or stationary-bootstrap inference does
not account for this selection. Confirmatory claims require search-aware
multiplicity control or evaluation on an independent holdout; this report does
not make a post-selection significance claim.

## Reproduction commands

These commands inspect the result and plan without starting forecasts:

```bash
simfolio-oos candidate-scores --json
simfolio-oos candidate --model experimental_filtered_innovation_moment_sv_fixed_mean --plan --json
```

To execute the candidate on all canonical tasks, use the same selector without
`--plan` and retain 240 simulations per origin:

```bash
simfolio-oos candidate --model experimental_filtered_innovation_moment_sv_fixed_mean --simulations 240 --workers 4 --json
```

The candidate uses the existing Frontier dependence seed schedule so its paired
random streams match the incumbent reference. `simfolio-oos scores --json`
continues to inspect only the retained 175-row historical score evidence.
The [complete 176-row ranked JSON](results/combined-176-score-ranking.json)
contains every row with its `display_rank`, preserved `canonical_rank`, exact
score token, and evidence status.
