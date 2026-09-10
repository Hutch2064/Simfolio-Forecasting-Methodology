# Historical runtime discrepancies

These findings qualify historical score reproduction. They are not repaired by
matching model names or the total number of tasks. Retained scores remain
imported historical evidence, never newly computed results.

## Bounded source-dispatch evidence

The retained current-control and asset-level dispatchers were replayed on the
packaged four-asset fixture with 600 observations, origin `2026-05-13`,
monthly rebalancing, eight forecast days, and 16 simulations. The source
candidate was the 82-key full-INLA descriptor with digest
`472d8bac243f3b1b123071a21a52d96394e775e8c4accb8d5fa18fd676451034`.

The current-control terminal-log matrix and the exact asset-dispatch terminal
log matrix matched their packaged adapters byte-for-byte (maximum absolute
difference `0.0`). The asset replay also matched the source marginal cube,
Gaussian dependence uniforms, mapped asset paths, and rejoined daily logs.
The adapter therefore preserves the source simple-return to log-return input
round trip, seed contexts, source asset ordering, and rejoin arithmetic for
this bounded case. The replay evidence is kept in
`resources/test_fixtures/inla/source_dispatcher_parity_report.json`. The replay
redirects Numba's cache and loads the machine-local alternate-wrapper path in
memory without writing to the source tree.

This is bounded source-method and adapter evidence. It does not rerun the
4,080-task, 240-simulation study and does not verify the retained scalar score
artifact or its historical ranking.

## Daily values consumed as terminal predictions

Both recovered dense wrappers have the same base-model branch behavior:
`_generic_terminal_matrix` selects `paths[:, horizon - 1]` from
`_simulate_candidate_log_paths`, which produces daily log-return increments.
The branch does not cumulatively sum those increments. The moving-block
Bayesian branch has the same selection pattern. Other branches return
cumulative terminal log returns through their source dispatcher.

Recovered wrapper identities:

- `tmp/oos_all_daily_exact_crps.py`: SHA-256
  `88aaa532d8a0a63e24964305a26ee6fa45d0d9099dd5da800ab691afad583ea7`.
- `tmp/asset_level_full_panel_20260823/asset_level_full_exact_crps.py`: SHA-256
  `5beb318b918ea367f7048d71a4e53e1bcc9b343faccda2ef493ddcf81bf3d465`.
- `tmp/asset_level_full_panel_20260823/simfolio_oos_two_candidates.py`:
  SHA-256
  `0577e79abee09d09735ee76f85941e007f995cc6a370ee375a0743cf440a23c2`.
- `tmp/asset_level_full_panel_20260823/simfolio_oos_copula_alternatives.py`:
  SHA-256
  `413d2ca7f74cda13dd228c8974f822ce23e690cc56e098babf3fd2c121fbf95e`.
- `tmp/asset_level_full_panel_20260823/simfolio_adaptive_pgas.py`: SHA-256
  `f7988a6cfbdf674c1efeb1ca6836b34e5f34c1ee241e97ef44551ded7ba87c69`.
- Source engine: SHA-256
  `702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665`.

The standalone evaluator scores cumulative terminal log returns from coherent
daily model paths, as specified by the publication. The bounded current-control
and exact asset-dispatch replays above establish adapter agreement for their
tested source paths, while the retained direct-index branch remains a separate
historical behavior. The retained result does not establish that its scalar
scores came from the corrected adapter, source revision, and dependency matrix
now recorded by this package. Full white-paper reproduction remains
unverified; a corrected calculation must not be represented as reproduction of
the retained ranking.

## Historical descriptors versus later defaults

The retained runtime catalogue SHA-256 is
`b80e3e3c9616916f090b0449cdbe3a79f150340dd98846d1f40b0691dae537a8`.
Its raw descriptors are distinct from later expanded dictionaries. Inserting
explicit defaults changes seed-bearing fit signatures even when the effective
numerical parameter happens to be unchanged. The canonical MCMC adapters
therefore preserve raw historical descriptors and document resolved defaults
separately.

The raw descriptor for `stochastic_volatility_ar1_student_t` contains only its
ID and `type="sv"`. The recovered dispatcher defaults a missing `innovation`
to `empirical`. A later Student-t implementation differs from that historical
branch: the bounded comparison observed a maximum terminal-array difference
of `0.019680524468041617`. The public mapping now preserves the raw descriptor and source-default empirical branch. An independent bounded original-dispatcher replay matched all 35 terminal values exactly. This establishes source-dispatch agreement, while the conflict with the Student-t label remains documented.
