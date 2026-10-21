# Shared simulation across all forecast leads

The earlier terminal-only batching implementation shares volatility recursions
and innovations between the six mean/mean-uncertainty variants of each filter
family and innovation law. This isolated implementation exposes the cumulative
portfolio value at EVERY native lead, retaining the original joint asset states,
AR mean recursion, random streams and log-domain portfolio accounting.

The copied kernel changes only the output buffer and per-step capture. The new
all_chunks generator combines grouped filtered options with the unchanged raw
and parent generators, preserving the original eligible-option order. The
isolated candidate substitutes that stream into the already qualified fast
whole-path coefficient implementation. There is no path, horizon, candidate,
score or information reduction, and no manual blend percentage.

Five tests compare every option and lead at10and1000paths, crossing64lead
boundaries, unordered subsets, zero-weight constituents, invalid menu inputs,
both fitted coefficient/weight controls and final1,000path forecasts. All are
byte exact against the corresponding original implementation on those fixtures.
This is not an all355case or full64,049request retained optimization claim.

A synthetic160row/3asset/128lead benchmark uses three interleaved warm-code pairs
per workload. At10paths per option, grouped simulation is1.97x faster and grouped
simulation plus the existing fast coefficient kernel is1.52x faster. At1000paths,
those ratios are1.66x and1.18x. All streamed path and coefficient hashes match.
The comparison excludes fitting, weight optimization, final prediction, CRPS and
IO. Hashing is excluded equally from measured CPU. Twelve existing scientific
workers were active, so wall times include concurrent-workload effects. No full
request or deployed speedup is inferred by multiplying these phase ratios.

The live355case reference and queued precision study are unchanged. Further
complete-request parity and runtime validation is needed before this backend
is retained. No new financial model result or full-panel catalogue entry exists.
