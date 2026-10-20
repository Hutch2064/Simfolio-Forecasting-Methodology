# Public-bootstrap comparators

| Name | Block selection | Replacement |
| --- | --- | --- |
| Public — Adaptive Blocks | Original Politis–White-style training estimate sets the mean; realized lengths are geometric | With replacement |
| Public — Random Blocks (Replacement) | Uniform lengths from 1 to min(training observations, requested forecast observations) | With replacement |
| Public — Random Blocks (No Replacement, Cycled) | Uniform lengths subject to the same bound and available contiguous unused rows | No observation repeats within a history cycle; refill only after exhausting that cycle |

All three consume the same training histories and raw historical returns, with no mean shrinkage or volatility overlay. The original reference remains the adaptive-block comparator. The two new comparators have new immutable identities and inherit no historical scores.

The cycle rule was explicitly selected by the user. It permits forecasts longer than the available history without describing across-cycle reuse as strictly replacement-free. Available blocks shorten as the current cycle is depleted. The final forecast block may be truncated.

Names are presentation metadata in `model_catalogue.json`; they do not alter execution fingerprints, seeds, return accounting, or score definitions. Every newly admitted comparator starts with zero scored origins. The preserved adaptive-block reference retains its completed 551-series result.

## Execution sequence

The user has authorized setup and development research. First run representative family and subgroup tests, retain the three comparators plus the best research candidate, and iterate from simpler to more complex evidence-supported methods. Automatically launch the full-panel evaluation only after the stated non-speculative-improvement stopping requirement is reached. The full-panel control remains stopped during this development stage.
