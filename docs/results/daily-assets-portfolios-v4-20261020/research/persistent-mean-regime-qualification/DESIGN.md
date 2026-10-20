# Training-fitted between-batch mean regimes: next v4 qualification

This is a distinct, not-yet-admitted contrast against the current shrunk-mean
bootstrap/FHS pool. Calibration diagnostics found heterogeneous long-horizon
undercoverage, especially fixed income. They motivate a hypothesis, not a score
correction, route or promotion criterion. Only the single fixed v4 CRPS selects.

The current pool already models uncertainty about a constant historical mean.
This contrast asks whether an additional future persistent mean regime has
training-supported variation. Partition the training prefix into the existing
floor(sqrt(n)) consecutive, nearly equal batches, using all data. Estimate each
batch's mean m_j and within-batch sampling variance v_j=s_j²/n_j. Fit nonnegative
between-batch variance tau² by the Paule–Mandel equation
sum((m_j-mu_w(tau²))²/(v_j+tau²))=q-1, with weighted nuisance mean mu_w.
If the zero-boundary equation is at most q-1, tau²=0. This uses no score or future
return. Within-batch sampling variance is a working martingale approximation,
not an assertion that observed returns or successive batches are independent.

For each whole forecast path, replace the parent's Gaussian mean draw SD se by
sqrt(se²+tau²), retaining the same stratified Normal draws and namespace, current
shrunk center, empirical resampling, volatility law, native return accounting,
120/120 whole-path pool and 240 paths. tau²=0 must produce the exact parent law.
The added term represents a future mean state, not another standard error of the
same global mean. Its variance contributes h²tau² to cumulative log-return
variance before native accounting; assuming a state persists through the forecast
could overdisperse long horizons. The main-score test must assess this risk.
No arbitrary variance multiplier, class/horizon switch, prior/lag grid or tuning
against diagnostic coverage is allowed in this contrast.

Qualification precedes field forecasts: independent statsmodels Paule–Mandel
numerical comparison; equal-variance analytic solution and zero boundary;
translation/scale/order invariance; native parent equality at tau=0; then
training-only representative prefix inspection. A frozen v4 development roster
must cover original and portfolio families, with individual-asset support for
each scored quarter. Any lower main-Q result must also pass distinct reserved
origins and broad coverage before full-panel promotion. These are adaptive
benchmark gates, not independent validation or significance tests.

Paule & Mandel(1982), Consensus Values and Weighting Factors,
https://doi.org/10.6028/jres.087.022 , supplies the within/between variance method;
its use for financial training batches and future persistent mean risk is our
hypothesis, not a result established by that paper. Independent implementation:
https://www.statsmodels.org/stable/generated/statsmodels.stats.meta_analysis._fit_tau_iterative.html .
Pástor & Stambaugh(2012), Are Stocks Really Less Volatile in the Long Run?,
https://doi.org/10.1111/j.1540-6261.2012.01722.x , motivates uncertainty about future
expected returns; its equity-specific predictive-system findings neither prove
this simplified Gaussian law nor establish performance across other assets.
