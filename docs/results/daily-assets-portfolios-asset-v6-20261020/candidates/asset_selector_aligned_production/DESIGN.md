# Aligned single-objective request kernel, research only

This is a specialization of frozen score-alignment config18: uniform_days,
train_SD_sqrt_lead, batch, inner2, half_doubling. It starts with dated joint-asset
arrays and produces the complete requested1000-path portfolio forecast. It does not
alter production repositories, the benchmark, scoring goal, recipes, tolerance,
seed contract, simulation count, forecast horizon or within-path accounting.

The fixed final objective is mean daily cumulative-log-return CRPS divided by
training-prefix portfolio SD(ddof1) and sqrt(native lead), with equal native-day
weights. Internally the existing reference accumulates the daily scaled losses as
a SUM before represented-origin weighting. Within a request its internal fold
horizon is fixed, so dividing every fold by that common h is a positive constant
and leaves the simplex objective's minimizer unchanged. The specialization retains
the original SUM arithmetic byte-for-byte; external scoring uses the actual mean.

Each fold still fits the complete eligible167-recipe bank, generates every native
lead and all inner paths with the original seeds, and evaluates every canonical
empirical pair coefficient. It accumulates only the one requested objective and
runs one unchanged certified simplex fit. The complete all-mode reference's final
prefix fit context is also retained, including recipes not ultimately allocated
paths. No fit-context subset or asset subset is introduced. The frozen path adapter
preserves component order, antithetic allocation, seeds and joint-asset accounting.

The removed work consists of11 unused coefficient streams,11 unused simplex fits,
and unused BOA/progressive learners. Historical A/B are accumulated chronologically
without a persistent cache. Only the current request's fit context is retained in
RAM while its paths stream. No winner fallback, candidate cap, lead cap, path cap,
fold cap or tolerance change exists.

## Measured baseline and first specialization

The canonical CTV protocolv1.5 was fetched/read from engine origin/main. These are
local research kernel profiles, not browser CTV or deployment evidence. Both dirty
production checkouts were left unchanged. Measurements use nice15, all numerical
threads1, and the exclusive shared scientific lock. One entry begins with dated
arrays; selection, full refit and every1000-member forecast chunk are inside the
clock. CRPS comparison and evidence serialization happen after the output clock.

| Request | Reference warm wall | Specialized warm wall | Ratio |
|---|---:|---:|---:|
| Actual week0, CORN, H5 | 4.777s | 1.124s | 4.25x |
| Prior slow broad350, H126 | 5.045s | 2.096s | 2.41x |
| Original portfolio9,10assets,H1969 | 11.250s | 8.427s | 1.34x |

Each cold/warm profile persists stage CPU/wall, exact request and source bindings,
pstats and full output hashes. Cold means the first call in a fresh process with
existing numerical disk caches; warm still recomputes all request fits. cProfile
overhead is included. The original profiling driver bytes are retained separately
as profile_driver_reference_v1.py. source_attestation_v1.json binds the new
candidate hash and every profiled result; inherited reference closure alone is not
used as the new implementation identity.

The week reference spent4.090s in12 mixture fits, versus4.777s total. On the longer
portfolio, after specialization, the unchanged coefficient routine takes7.322s
of8.427s, with7.296s self time in its Python pair loop. This remaining hotspot is
not claimed to be optimized by this first specialization or to be unavoidable.
A separate compiled outer pair-loop variant may be investigated without changing
this now-frozen source or the canonical cross-absolute arithmetic.

## Exactness evidence and limits

Three focused tests pass, including identical coefficients, weights, certificates,
all fold traces, full1000-path outputs, all per-lead losses, two final seeds and
three output chunk widths. Direct real-request verification additionally compares
all three profiled requests, two complete final forecast seeds each, every path
float and per-lead loss, without truncation. The associated receipt contains full
SHA256 bindings. Expanded12context x3inner-seed qualification is a separate bounded
stage; its concurrent timings are diagnostic, not paired speed evidence.

The retained source and compact raw profiles/evidence remain owner-held under this
folder for root review and preservation. They may be archived after qualification
and independent review. No full-panel sweep, production release or browser test
is authorized by this module's existence. It makes no general production latency
claim, and it does not prove numerical parity for arbitrary future code changes.
