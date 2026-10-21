# Whole-path diagnostic trade-offs and sequential prototype

The official full-panel result and active355-case study are unchanged. The
closed72-case1000path whole-path learner is21.309474%better than fixedWR, versus
20.540747%for its endpoint control. Its0.768727percentage-point incremental gain
is not uniform. An exact216checkpoint decomposition reconciles all dimensions
to the same unchanged canonicalQ. Equities andFX improve; fixed income and mixed
portfolios regress. Descriptive horizon contributions improve through31days,
32to366days,367to1826days and over7305days, but regress over1827to7305days.
These are additive contributions on adaptively reused research cases, not
within-class percentage skills, independent evidence, new scores, forecast
routes or model promotion gates. Diagnostics have zero additional score weight.

The separate scale-free sequential prototype replaces the batch-optimal mixture
with historical-fold CRPS-gradient updates. Its learning-rate scale is derived
from observed gradient magnitudes. It keeps the166option bank, asset paths,
validation schedule and outer score. It introduces no manual blend percentage,
fixed learning-rate cap, gradient clipping or previous-winner fallback. The
minimum-norm initial simplex state is an uninformed symmetry rule; final weights
are learned. This is SOLO-FTRL adapted to a convex CRPS mixture, not quantile BOAG
or a proof of future financial accuracy. See itsDESIGN.md for primary references,
fixed-bank eligibility assumptions and limitations.

Six prototype tests pass, including independent projection/gradient checks,
chronological weight prefixes, scale/common-loss-shift invariance, edge cases,
and exact output preservation of both joint-path batch controls. No real-panel
accuracy result or new forecast pool is admitted. A future matched100path72case
screen can share historical component simulation between both batch controls
and the sequential rule, after the current and queued validations are reviewed.
No full-panel catalogue entry is added.

A separate read-only workload projection, timestamped2026-10-21T13:23:11Z, uses
59nonwarm1kprecision request CPU observations with the exact same numerical
source closure. For281queued/inflight requests, mean/median ideal12core remaining
compute projections are14025/14176seconds. They include the entire cost of
already running requests rather than deducting elapsed progress, and exclude
setup,IO,contention and failed-fit repair. They are workload projections, not
calibrated wall completion times, original broad-market probes or forecast
speedup claims. The active study's live estimator remains unset until enough
within-stage nonwarm requests complete.

The exact-reducer replay/reserved stage remains queued behind the current
independent audit, subject to development improvement. No current numerical
source, scoring weight, preserved outcome or goal stopping rule has changed.
