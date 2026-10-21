# Matrix-free sequential historical CRPS pooling

This is a faster implementation candidate for the separately qualified SOLO-FTRL
history-only learner, not an accuracy-proven replacement for the batch leader.
For a known pre-fold mixture w, the learner needs A-Bw and its observed mixture
objective, not the full B matrix. A weighted empirical-CDF sweep calculates Bw
from all option ensembles directly. Every eligible option is evaluated, including
all currently zero-weight options. No option or validation lead is discarded.

Two kernels are retained. A Numba global sweep is the numerical control. The
vectorized kernel chooses between a global sweep and a nonzero-mixture CDF plus
queries of EVERY option, using estimated comparison counts. This computational
choice changes neither the mathematical mixture nor the candidate universe.
Observation A preserves the original member summation order. Weighted CDF terms
change floating-point summation and are not claimed globally bit exact.

The standalone joint-asset candidate first determines the same eligible-bank
intersection from known historical training prefixes, then replays those fits
chronologically and observes gradients only after each historical fold. It keeps
the requested-horizon fold schedule, all known validation leads, source-bound
seeds, scale-free update, joint asset paths and within-path portfolio accounting.
A prefix eligibility pass repeats fitting and is included in phase timings; no
unbounded collection of fitted history arrays is cached. This may be costly for
very short requested horizons and is not hidden from runtime claims.

Ten tests compare gradients with dense matrices, ties, extreme rescalings,
chunk continuation, sparse mixtures including zero-weight alternatives, invalid
inputs and sequential weight chronology. A40row/3asset fixture compares final
weights within2e-11 and all1,000 forecast path values exactly with the existing
dense sequential learner. This fixture is not a complete panel or a200asset
runtime proof. Bank-intersection eligibility is known from the outer historical
prefix; no prequential regret claim is made for a postselected eligibility set.

Synthetic166option/1000member/16lead, three interleaved warm-code pairs show
approximately1.56x(uniform)to1.79x(three positive weights)CPU speedup against the
already optimized dense global reducer. Earlier global-only and vectorized-only
benchmark outputs are preserved too. They are kernel timings with12scientific
workers concurrently active, not full-request or production latency gains.
The five-to-ten-second local request target remains unachieved. No real-panel
accuracy test, full-panel promotion or new scientific worker pool is admitted.

Research basis and limits follow the existing sequential prototype:
Orabona and Pal(2018), Scale-Free Online Learning,
https://david.palenica.com/papers/scale-free-ftrl/Scale-Free-Online-Learning-TCS-final.pdf
and Berrisch and Ziel, CRPS Learning,
https://arxiv.org/html/2102.00968
This implementation is not their quantile BOAG and makes no future-accuracy claim.
