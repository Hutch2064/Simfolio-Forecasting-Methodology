# Historical runtime discrepancies

These findings qualify historical score reproduction. They are not repaired by
matching model names or the total number of tasks. Retained scores remain
imported historical evidence, never newly computed results.

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
- `asset_level_full_exact_crps.py`: SHA-256
  `5beb318b918ea367f7048d71a4e53e1bcc9b343faccda2ef493ddcf81bf3d465`.
- Source engine: SHA-256
  `702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665`.

The standalone evaluator scores cumulative terminal log returns from coherent
daily model paths, as specified by the publication. Its base-model numerical
path parity therefore does not establish equivalence to those recovered
wrappers' scoring behavior. The retained result does not provide a complete
executed source/environment fingerprint that resolves this conflict. Full
white-paper reproduction remains unverified; a corrected calculation must
not be represented as reproduction of the retained ranking.

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
