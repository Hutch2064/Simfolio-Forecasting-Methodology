# Stochastic correlation contrast on the learned working model

Preserve the working SV-RMS Gaussian / EWMA-fading raw-block mixture, training-CRPS blend estimate, shrunk means, correlated mean uncertainty and joint asset accounting. The new mechanism changes only filtered innovation dependence.

Fit the existing training-only corrected-DCC recursion using the same Ledoit-Wolf correlation target and conditional Gaussian copula likelihood as the prior correlation study. This is the corrected recursion with a fixed shrinkage target, not a claim to implement Aielli's consistent target estimator. Parameters are stationary, nonnegative and selected by training likelihood, with an explicit static endpoint.

For each path, before a future shock, Q is positive definite. Draw v=L(Q)e, e~N(0,I), set z_i=v_i/sqrt(Q_ii). Then update Q_next=(1-a-b)S+a vv'+b Q. Thus the simulated shock has unit marginal variance, the conditional correlation implied by Q, and its own future correlation state. No future Q expectation replaces this recursion. The expected unnormalized Q follows S+(a+b)^h(Q_initial-S), but normalizing that expectation is not the same law as integrating paths.

Field contrasts: training shrinkage static correlation, conditional-mean-Q correlation, and full stochastic corrected-DCC. The sample-correlation mode is an exact parent control, not another field hypothesis. A scalar asset has correlation one in every mode, with unchanged parent RNG and exact scalar paths. No fitted per-ticker, horizon or asset-class routing. Existing panel/scoring weights stay fixed.

The training mixture weight uses only marginal conditional CRPS; changing Gaussian correlation leaves each filtered one-dimensional Gaussian law unchanged. Reusing that weight is therefore coherent with its unchanged training objective. This is a dependence contrast, not a re-estimated multivariate CRPS optimizer.

Sources: Engle (2002), https://doi.org/10.1198/073500102288618487; Aielli (2013), https://doi.org/10.1080/07350015.2013.771027. Earlier full80 model evidence is supporting motivation only. Prior conditional-mean tests used a different working parent and lost; they did not test stochastic future correlation paths.

Require exact parent control, independent NumPy path-recursion parity, conditional Gaussian covariance and mean-Q Monte Carlo checks, chunk invariance, joint accounting and prefix-only fitting before admitting a small matched screen. Simulations and exact CRPS use the same declared path count; no path exclusions or hidden additional ensembles. This model may be slower, especially for large asset counts; record measured compute rather than claiming a speed improvement. No automatic full-panel escalation.
