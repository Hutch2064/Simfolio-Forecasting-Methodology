# Batched per-fold historical scale contrast

Use the existing 166-component bank and fast batched path generation. Divide each completed inner fold contribution by that fold training portfolio standard deviation (ddof=1), using only its historical prefix. This matches the fixed canonical outer normalization; it does not change outer scoring, panel coverage, folds, component definitions, or final 1,000 paths. Positive scale is required without an epsilon or fallback.

The prototype preserves the existing fold-normalized model identity: batching and an explicit internal Monte Carlo repetition index do not define a new financial method. Replication zero must reproduce all prior fold-normalized coefficients, allocations and final paths. The non-normalized current selector remains the comparison control; no precision or accuracy promotion follows from qualification tests.

Real-panel admission waits for the current controlled precision study and its independent saved-loss audit. Use the same 12-worker pool, existing immutable cases and matched simulation counts. No additional scientific workers have been launched for this preparation.
