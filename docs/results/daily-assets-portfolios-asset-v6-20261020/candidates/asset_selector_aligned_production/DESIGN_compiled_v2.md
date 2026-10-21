# Exact compiled outer pair loop, separate V2

V1 candidate.py and its completed36-pair qualification remain unchanged. V2 lives
in compiled_candidate.py with coefficients.py. The candidate copies V1's aligned
single-objective control flow and changes only coefficient dispatch plus its
research implementation identity. No canonical coefficient source was edited.

The new helper retains the exact NumPy cast, sort, absolute differences and row
mean for A. It compiles the existing Python outer i/j loop that fills B, calling
the **unchanged canonical cross_absolute_sorted** for each canonical lower-triangle
pair and performing the same mirroring into every full-matrix entry. Float64,
iteration order and fastmath=False are explicit. No P2-only formula, alternate
estimator, pair reduction, rank reduction, tolerance change or algebraic rewrite
is introduced. Tests include2,4,8,32members and up to167recipes, repeated values,
zero distributions and scaled inputs. The helper is cache=False: it creates no
persistent numerical/data cache; its first compilation cost is included below.

V1 profiling identified7.322s of8.427s on portfolio9 in coefficient construction,
with7.296s self time in Python pair dispatch. The new compiled loop removes that
interpreter dispatch while retaining every call's arithmetic and all lead work.

Serial local full-request profiles, including selection/refit/full1000path output:

| Request | All-work reference warm | V2 warm | Ratio |
|---|---:|---:|---:|
| Actual week0,H5 | 4.777s | .950s | 5.03x |
| Original slow broad350,H126 | 5.045s | .560s | 9.01x |
| Portfolio9,10assets,H1969 | 11.250s | 1.210s | 9.30x |

First-call V2 wall times were1.173s,.883s,1.463s respectively, including new helper
JIT compilation in fresh processes with existing unrelated numerical disk caches.
Warm calls recompute all request fits. cProfile overhead is included. Every serial
sample matches reference A/B, weights, full path stream and full raw loss SHA256.
These are bounded local kernel measurements, not browser CTV, deployment evidence
or a population median/tail performance claim. Bulk disk/network work was deferred
until the serial timing phase completed, then permitted during qualification.

The immutable source_attestation_compiled_v2.json binds implementation, helper,
all frozen dependencies and every timing JSON. timing_comparison_v2.json provides
paired CPU/wall details. Four focused tests pass, including all-work exact fit/
trace/certificate equality and full1000paths at two seeds and three chunk widths.

Expanded V2 qualification is48 complete paired requests: the original12long smoke
contexts at internal seed replications0,1,2, plus all12actual calendar-week contexts
at seed0. Each pair independently learns the same aligned mixture on the same
prefix, requested H, schedule and inner2paths, then compares every1000-member path
and per-lead loss. Known invalid options and full dated traces are compared too.
No long-horizon learned weights are substituted for weekly requests. Raw loss
vectors and compact provenance persist as evidence; prepared fit/coefficient caches
do not. This qualification is separate from any full-panel accuracy admission.
Its concurrent timings are diagnostic only. Read qualification48_compiled's final
receipt for completion status; this design note alone makes no completion claim.

The objective remains the exact reference chronological internal lead SUM of
CRPS/(fold-prefix portfolio SDddof1*sqrt(nativelead)), with the unchanged certified
simplex solver. External aligned scoring is the mean; common1/h normalization
within a request is deliberately not introduced into the internal arithmetic.
All167recipes, historical prefix fits, all leads, full refit context, seeds and
within-path joint-asset accounting remain intact. The final output adapter is the
same frozen path implementation. There is no production repository or UI change.
