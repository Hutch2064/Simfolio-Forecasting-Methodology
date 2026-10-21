# Isolate internal Monte Carlo precision

Final forecasts always use1,000paths. Only internal path count and its random
seed replication change. The candidate is an isolated copy of the qualified
whole-path reducer with an explicit internal seed suffix. Replication0 preserves
all original coefficients, weights and final paths in the synthetic test.
Internal seed replications are executions of the same configuration, not new
model identities. Each request still learns solely from its historical prefix.

The initial exploratory grid is10,30,100,300internal paths per option and three
seed repetitions on the same72requests/70families, with the preserved1,000inner
and1,000final-path reference. This means864new final forecasts and72reused high
precision forecasts. The grid is an experiment, not a production path cap, a
promise that ten is enough, a new canonical score or an automatic admission rule.
The high-precision reference has one internal seed replication; repeated lower
precision seeds alone do not measure its own variability. Further repetitions,
new precision levels and reserved confirmation can be added from the evidence.

Compare each configuration on exactly the same requests and1,000final paths,
using only the frozen main score against fixedWR. Report seed variation as
Monte Carlo diagnostics, not independent market observations or an extra score.
Do not choose per-request counts, weights or future candidates from outer future
outcomes. Any selected production rule requires further validation. The default
driver prepares a plan only; --run requires recovered published source and the
existing shared12-worker scientific lock. Existing studies and sources remain
unchanged; no new process pool has been started by this qualification.

Six tests check default-control equality, repeatable/different internal seeds,
fixed1,000final paths at multiple low counts, invalid counts/seeds, the precise
source-clone delta and seed-independent model identities. Tests are synthetic,
not a claim of panel accuracy, runtime or numerical convergence at ten paths.

Small ensembles can change more than ranking noise. The usual empirical CRPS
has ensemble-size effects when interpreted as a Monte Carlo distribution
estimate. Ferro(2014), Fair scores for ensemble forecasts, discusses adjusted
scores and dependence limitations:
https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf
Our option draws include antithetic mean uncertainty, shared random numbers and
stratified components. An independent-member fair-CRPS correction is therefore
NOT automatically applicable. No correction or outer score change is made here;
this is a recorded further research hypothesis requiring a dependence-aware
justification and matched forecast tests.
