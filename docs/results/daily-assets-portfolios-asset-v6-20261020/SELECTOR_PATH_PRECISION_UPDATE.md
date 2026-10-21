# Whole-path precision results and broader development

Goal remains active. Panel, canonical main score, sole fixed worse asset-level
with-replacement reference and 1,000-path official protocol remain unchanged.
The official full-panel winner remains the circular/GARCH-fade model, 3.205447%
better than the reference on all 64,049 matched forecasts. No result below
promotes a new full-panel winner. Catalogue remains 23 unique methods/28 executions.

## Completed 1,000-path precision comparison

On the same 72 requests, with 1,000 internal and final paths for both methods:

| Method | Main Q, lower is better | Better than fixed reference |
| --- | ---: | ---: |
| Fixed with-replacement reference | 0.9660201133857566 | 0% |
| Endpoint-criterion pool | 0.767592367846421 | 20.540747% |
| Whole-path-criterion pool | 0.7601663127925109 | 21.309474% |

All 144 evaluations completed with zero native failures or canonical scoring
errors. Independent verification checks all 144 candidate and 216 reference
checkpoints, main-score reductions, historical-prefix/date causality and all 72
exact endpoint-control loss replays. These are adaptively reused research cases,
not independent validation or evidence of a full-panel improvement.

The whole-path criterion learns one joint-path mixture from historical forecast
scores across all available native leads through the requested horizon, using
positive log-calendar masses and sqrt(native-lead) normalization. It keeps the
166-option universe and historical validation schedule fixed. It introduces no
manual percentage blend or outer-outcome fitting.

## Completed exploratory block-menu comparison

At 100 internal/final paths on the same 72 requests, adding stationary and
circular corrected-PPW block-length options to the 166-option menu worsened
both whole-path and endpoint variants slightly. Whole-path original/expanded
scores were 0.7587879853116555/0.7600849889235529, respectively 21.599394% and
21.465383% better than their matched reference. Endpoint original/expanded
scores were 0.7741087467314783/0.7756244427417525, respectively 20.016400% and
19.859793% better. All 288 evaluations and 144 original-control trace/loss replays
passed independent audit, including percentage calculations, with zero errors.

The additions do not advance from this screen. The small differences at 100
paths do not establish that no precision reversal or useful block-based method
exists. Do not rank these 100-path results against the 1,000-path results above.

## Current broader check

The qualified whole-path method is running the original 355-request development
slice at 1,000 internal/final paths with 12 shared workers. All 72 verified
precision pairs are reused, leaving 283 new requests/566 new outer evaluations.
Every new endpoint control must reproduce its preserved complete loss vector.
Internal A/B coefficient matrices are saved for the 283 new requests only,
allowing later solver audits and compatible numerical reuse without repeating
historical simulation. No stopping decision or full-panel admission is implied.

Profiling 13 selected first requests at 100 paths attributes about 79% of measured
preparation CPU to dense CRPS coefficient assembly. This is a local bottleneck
measurement, not a complete-workload, 1,000-path or production speedup claim.
Any optimization must retain all options, leads, origins and simulation members.

## Percentage correction and preservation

The original one-time precision driver accidentally changed its percentage
multiplier from 100 to 1,000 while changing the simulation count. Its original
summary consequently contains incorrect 213.09%/205.41% displays. Saved losses,
main Q, coverage and ranking were unaffected. Correct values are 21.309474% and
20.540747%. Original source/report are retained; REPORT_AUTHORITY.json and
percentage_metadata_erratum.json identify immutable originals and corrected
reports. Public summary.json is the corrected compact view; raw full summaries
are in the private archive. Percentage calculation now has a separate helper
and regression tests preventing simulation count from changing its scale.

The linked private archive preserves 782 closed results, sources, profiles and
audit files, freshly downloaded, fully disk-restored and SHA256-verified. Active
355-request outputs are excluded. Scientific originals remain local while active
readers need them. The public compact files are derived views; private manifests
bind original full evidence. Historical percentage corrections remain documented
separately in PERCENTAGE_METADATA_ERRATA.json.
