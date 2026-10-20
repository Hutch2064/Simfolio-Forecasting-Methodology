# Student-likelihood training fit for the retained Gaussian mean pool

The new accuracy-maximization goal permits evidence-based mechanisms beyond the
previous closed finite search. The current winner's FHS component estimates its
GARCH dynamics using a Gaussian quasi-likelihood, although its future innovations
are empirical and its training returns can remain heavy-tailed after filtering.
The training-only preflight found median excess kurtosis 3.294777 across the frozen
245 development prefixes, spanning all49 families and6 asset classes. Every
prefix obtained a qualified Student-t fit with a better training likelihood than
the Gaussian member. These are in-sample diagnostics, not forecast improvements.

Bollerslev (1987), A Conditionally Heteroskedastic Time Series Model for
Speculative Prices and Rates of Return, https://doi.org/10.2307/1925546,
provides the conditional Student-t mechanism. The standardized likelihood and
numerical df bounds are independently checked against arch.StudentsT:
https://arch.readthedocs.io/en/latest/_modules/arch/univariate/distribution.html.
This candidate uses that likelihood to fit dynamics, not parametric t shocks.
Heavy-tailed likelihood fitting can allocate the influence of extremes
differently from Gaussian fitting; it does not remove extreme observations,
promise robustness against every contamination mechanism or establish superior
forecasts. The original unconditional sample variance target still remains.

Only the FHS-half training likelihood changes. Keep the historical sample mean,
sample variance target, initial normalized variance1, empirical standardized
zero-mean/unit-variance innovation pool, native random indices, Gaussian
batch-mean uncertainty, component namespaces, 120/120 whole-path weights,
native wealth accounting, fixed722 inputs and both frozen v3 scores unchanged.
The Student-t degrees of freedom are fitted from the training prefix with the
library's numerical support bounds2.05..500. Include the exact original Gaussian
fit as a boundary member, comparing complete normalized likelihoods. Three fixed
starts at most; require optimizer success and a projected-gradient check. If no
Student fit qualifies, raise an error rather than reuse a failed fit. The
Gaussian boundary is a fitted model member, not an error fallback. No weights,
seeds, thresholds, horizon routes or family routes use forecast scores.

Four focused tests verify the independent library likelihood, analytic-gradient
finite differences, exact unchanged variance recursion, equivariance/innovation
normalization, and byte-identical full native losses when the Gaussian boundary
is imposed. The245-prefix fitter qualification used training history only.

Freeze the existing metadata-selected245 development cases, then245 reserved,
then81-series/4122-origin representative coverage. These are reused adaptive
benchmark cases, not an independent holdout. Replay current winner and Public
freshly beside every candidate forecast with240 paths and original origin seeds.
Check all490 development baseline loss hashes against the retained prior
summary, which was originally verified against full-panel checkpoints. Do not
restore large historical payloads merely to repeat those exact checks.

Advance each stage only after every case is finite, failure-free and the primary
score is lower than the current winner on identical cases. Both profile results,
class/family regressions and raw horizon losses remain visible. A representative
win is insufficient for promotion: require a fresh full722 attempt and catalogue
registration, including failures and non-winning results. Rejecting this exact
likelihood contrast does not close the wider goal or rule out other mechanisms.

One nice15 numerical worker, single-threaded native libraries, at most1GiB owned
pilot outputs within the12GiB active research budget and100GiB free reserve.
Preserve all unique results losslessly in repository-linked private archives,
fresh-download/disk-restore/SHA256 verify before retiring local outputs. Existing
shared inputs, original source/runtime, prior winner and completed evidence stay
intact. No production changes or website speed claims are part of this test.
