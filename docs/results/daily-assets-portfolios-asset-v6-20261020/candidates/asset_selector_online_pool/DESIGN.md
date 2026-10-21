# Sequential whole-path CRPS mixture: qualified prototype

This is a separate research contrast, not a financial result or an admitted
full-panel winner. Data, outer main score, reference and166option bank stay fixed.
The running development and queued replay/reserved study keep their frozen code.

## Mechanism

The existing learner minimizes a batch historical CRPS mixture quadratic. This
variant processes those same completed historical validation folds in date order.
At each fold it records the mixture chosen before the outcome, evaluates that
fold's whole-path CRPS gradient A-Bw, and updates a simplex mixture using SOLO-FTRL
with Euclidean regularizer R(w)=||w||^2/2. Its scale is the square root of cumulative
squared gradient norms, derived from observed gradients rather than a tuned
learning rate. The all-ones gradient direction is removed: it changes no feasible
simplex directional derivative. Final learned weights select whole joint asset
paths, preserving within-path portfolio accounting. Weights never switch by lead
or constituent within the issued path.

Before score evidence, the minimum-norm simplex point is symmetric across eligible
options. This is an initialization, not a fixed final blend, accuracy floor or
fallback to a previous winner. Equal gradients leave that uninformed state
unchanged; one valid component gives its unique possible predictive law. No
learning-rate cap, gradient clipping, fold-count cap or manual percentage is added.

Component fits remain causal at each historical origin. As in the batch control,
weights use the intersection of options that fit successfully across the known
outer-training folds. That eligibility information is known at the actual outer
forecast date. We do not claim a prequential-regret guarantee for postselected
eligibility sets. The abstract online optimizer itself chooses each recorded
pre-outcome weight only from earlier coefficient pairs. No outer test losses,
future dates or previous benchmark scores enter fitting.

## Basis and limitations

Orabona and Pal, *Scale-free online learning*, Theoretical Computer Science716
(2018), DOI10.1016/j.tcs.2017.11.021, Section3 and Algorithm1:
https://david.palenica.com/papers/scale-free-ftrl/Scale-Free-Online-Learning-TCS-final.pdf
The paper develops FTRL algorithms adapting to observed loss-vector magnitudes.
Our simplex/CRPS-gradient application is a research adaptation; the paper does
not establish a financial accuracy gain, statistical independence or improved
canonical benchmark score for it. The quadratic regularizer is a structural
modeling choice, not a data-estimated universal optimum.

Berrisch and Ziel, *CRPS learning*, DOI10.1016/j.jeconom.2021.11.008:
https://arxiv.org/html/2102.00968 . Its online-learning sections motivate examining
sequential combination. This prototype is not their quantile-specific BOAG
algorithm and does not inherit a coherent multivariate construction from
univariate quantile aggregation. It instead keeps the existing whole-path mixture.

Long-horizon folds can be few; correlated leads are not extra independent origins.
Sequential regularization may stabilize weights, or may underfit relative to the
batch learner. Only matched forecasting tests can determine its value here.

## Qualification and planned test

Six tests cover the closed-form simplex projection against an independent solver,
feasible CRPS gradient, chronological-prefix invariance, positive-scale/common-loss
shift invariance within floating tolerance, uninformative/single-option states,
and exact joint-path output preservation of both batch controls. No real-panel
forecasting result is claimed from those fixtures.

A proposed first screen uses the same72cases/70families at100inner/final paths,
with batch endpoint, batch path and sequential path variants sharing one set of
historical component forecasts per request and the same12worker pool. Batch
controls must replay saved complete loss vectors exactly. A negative result does
not reject all online learning; a positive screen still requires1000path and
broader confirmation. This document admits no extra running forecast pool.
