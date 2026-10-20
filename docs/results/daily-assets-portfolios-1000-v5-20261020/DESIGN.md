# 1,000-simulation execution v5

Human authorization retains three Public comparators plus the current best,
removes FAST, and resets every baseline using1,000 simulations per forecast.
All1,264 input series,64,049 origin/horizon tasks per model, and the sole v4 main
score remain fixed. Historical240-path results remain a separate execution.

Each forecast fits once to its historical portfolio return stream. The best
model keeps a50/50 whole-path mixture (500 paths each) of raw bounded blocks
and variance-targeted empirical GARCH FHS, with the same training-only smooth
mean shrinkage and stratified antithetic Gaussian mean uncertainty. The three
Public forecast laws and cycle exhaustion rule remain unchanged.

Forecasts are streamed in64-horizon columns. All1,000 terminal values enter
the original exact ECDF CRPS at every horizon (n-squared empirical denominator).
No path subsampling, quantization, simulated-observation thinning, or approximate
CRPS is used. Wealth, variance, index and cumulative log-return state continue
across column batches. No full1,000-by-H wealth or index matrix is allocated.

Raw WR/cycle indices are losslessly represented by block start/length
descriptors; their original path-major NumPy draw order is retained. FHS IID
innovations use a time-major stream. Adaptive initial indices use the native
seed; restart flags and points use distinct declared namespace seeds and
time-major streams. Therefore this is a new numerical sampling version, not
bitwise parity with the old FHS/Adaptive ensemble. Training fits, stochastic
forecast laws, clipping and wealth accounting are unchanged. Chunk size does
not change the generated ensemble. Numerical identities bind all code, original
dependencies, simulation count and stream namespace; no240-path result is reused
as a1,000-path baseline.

Block descriptors consume memory proportional to actual sampled blocks, rather
than path count times forecast horizon. The real-data smoke covers every asset
class and portfolio economic family, including its longest origin. Peak process
memory and full ensemble counts must be checked before full admission. The
shared scheduler provides12 independent single-threaded forecast slots and
supports source-bound later model admission from the first origin.

Reporting uses only the unchanged v4 main score and the worst of the three
Public comparators on exact matched successful coverage. Diagnostics have zero
score or model-selection influence. Broad new baselines are required before
further candidate promotion. No statistical independence, significance,
production latency or1,000-path winner claim follows from smoke checks.
