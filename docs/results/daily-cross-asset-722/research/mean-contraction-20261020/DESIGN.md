# Match uncertainty contraction to the retained regularized mean

The current verified full722 winner changes the mean to `w*mu`, where
`w=mu^2/(mu^2+s^2)` and s is its existing training-batch Gaussian mean standard
error. It intentionally retained the original variance s^2 to isolate center
regularization. This new contrast keeps that improved center and changes only
its associated mean-draw variance using a specific working-prior identity.

For the normal summary likelihood `mu_hat | theta ~ N(theta,s^2)` and a fixed
normal prior `theta ~ N(0,tau^2)`, precision addition gives posterior mean
`tau^2/(tau^2+s^2)*mu_hat` and variance `s^2*tau^2/(tau^2+s^2)`.
The current center matches the mean expression when the working prior scale is
plugged in as `tau^2=mu_hat^2`. Under that same plug-in convention, the variance
is w*s^2. Therefore multiply the existing Gaussian draws by sqrt(w), keep their
exact random nodes/namespaces and their existing regularized center, and leave
all remaining fit/simulation/scoring behavior unchanged. No free scale multiplier
or prior-width grid is introduced.

Standard normal-prior precision updating is background material in Gelman's
Bayesian analysis notes: https://sites.stat.columbia.edu/gelman/book/220lecturenotes.pdf.
The *data-derived* prior scale here is our additional empirical working assumption,
not a prescription or theorem from those notes. Its scale uncertainty is not
integrated, and this is not fully Bayesian inference or a calibrated-coverage
claim. In particular, near-zero observed means can produce strong contraction
and overconfidence; at exactly zero the plug-in drift variance collapses to zero.
That limitation is tested explicitly rather than hidden by an arbitrary floor.
The choice has to earn its place through matched forecasts, not algebra alone.

Training-only qualification covered245prefixes spanning49families and6classes.
The median multiplier of mean-draw standard deviation is0.91025364, with minimum
0.00582265. These are not tuned thresholds or forecast results. Five tests cover
direct precision addition, exact boundaries, sign/scale/noise properties,
unchanged random nodes/center with the derived variance, and exact whole native
loss replay when contraction is disabled. No baseline module is mutated globally.

Keep both bootstrap/GARCH-FHS components, fitted volatility, innovation indices,
120/120whole-path weights,240paths, native wealth accounting, fixed722data and
both frozen v3scorers. Every fit sees only its origin's training prefix. No outer
OOS-score fit, family/horizon router, seed selection or outcome-based retuning.

Advance through the frozen245development,245reserved and81series/4122origin
coverage only after complete finite failure-free lower primary score than the
CURRENT shrunk-mean winner. Recompute that winner and Public on identical cases.
Development baseline hashes map to the corresponding prior mean_shrinkage
Candidate and Public records, not the older Gaussian-pool incumbent. Reused
benchmark cases are adaptive evidence, not independent confirmation. Disclose
both profiles, class/family/horizon losses, and require full722before promotion.
Every full-panel attempt joins the catalogue even if it loses; a pilot does not.

One nice15 numerical worker, single-thread libraries,1GiB pilot cap under the
12GiB active root budget and100GiB free reserve. Preserve raw results and exact
sources losslessly online, freshly download/disk-restore/SHA256verify before local
retirement. This contrast's failure would not by itself satisfy the wider goal's
stopping condition.
