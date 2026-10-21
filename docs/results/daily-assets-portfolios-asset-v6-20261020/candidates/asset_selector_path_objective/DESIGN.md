# Requested-horizon path objective: qualified prototype, no panel result

The completed355case terminal-pool gain over the working parent is concentrated:
assets contribute+3.55878percentagepoints of extra improvement against the fixed
reference, portfolios-.68169; the short-term reversal family contributes+4.46911
while the whole gain is+2.87709. Leads through31calendar days contribute-.26815,
fixedincome-.58743, and rolling-origin cases-.22778. These are additive pieces of
the unchanged mainQ, not extra scores, selection weights or significance claims.

This motivates testing a different internal fitting objective. Keep one joint
path mixture for the entire requested horizon, but learn its weights using all
historically validated leads through that horizon, with the existing canonical
log-calendar lead masses and sqrt(nativelead)normalization. For each lead,
CRPS is A'w-(w'Bw)/2. Positive prescribed lead weights and averaging across
chronological folds preserve this convex proper marginal scoring objective.
There is no manually chosen short/long split or percentage mix. The panel score
is never changed. The forecaster remains coherent through joint component paths.

The integrated candidate and numerical reducer pass seven synthetic/math
qualification tests. A/B accumulate in chronological order across memory-bounded
chunks. No simulation, option or lead is omitted. Tests verify temporal causality,
short-history eligibility, exact finite-ensemble algebra, request/date binding,
chunk invariance and exact endpoint-control parity. Representative scientific
testing is still pending; no financial improvement is claimed. Inputs for calendar masses must come only
from known training dates in historical validation, never future outer dates.
A full-request historicalSD normalization is constant across options and can
cancel from fitting; no arbitrary zero-SD normalization fallback is needed.

The forthcoming experiment must isolate endpoint versus whole-path fitting
using the same166-option universe, history-only folds and matched simulation
counts. It must disclose that reducing a long requested horizon to the longest
supported validation horizon still gives limited long-horizon evidence. New
internal objective is a methodology candidate, not a claimed pure speed change.
Qualification must cover temporal causality, portfolio accounting, chunk
invariance, exact finite-ensemble score algebra and unchanged outer scoring.

Academic basis: proper-score predictive stacking(Yaoetal2018,
https://arxiv.org/abs/1704.02030)and CRPS convex aggregation(Dombry/Zaoui2024,
https://arxiv.org/abs/2411.13974). No iid error/regret guarantee is imported to
these financial histories. Testing remains required.
