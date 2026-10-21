# Follow-up mechanisms: training-score weight stability

This is a research rationale, not an admitted candidate, a result, a fixed
regularization constant or a change to the outer score.

The current learner has 166 options but can have only two complete historical
validation origins at long requested horizons. Whole-path scoring adds known
leads, but correlated leads are not equivalent to independent origins. The
simplex fit can consequently be well solved numerically without its estimated
weights being statistically stable. This is a mechanism to investigate; it
is not proof that regularization improves the benchmark.

Primary sources checked on 2026-10-21:

* Yao, Vehtari, Simpson and Gelman, *Using stacking to average Bayesian predictive
  distributions*, Bayesian Analysis (2018), DOI 10.1214/17-BA1091,
  https://arxiv.org/abs/1704.02030 . The authors extend stacking to predictive
  distributions and proper scores and discuss stabilizing alternatives.
* Berrisch and Ziel, *CRPS learning*, Journal of Econometrics (2023),
  DOI 10.1016/j.jeconom.2021.11.008,
  https://www.sciencedirect.com/science/article/pii/S0304407621002724 . Its
  abstract describes adaptive combination across quantiles and time. Publisher
  search text was accessible; full publisher page returned403. Do not claim
  the full derivation was reviewed.
* *CRPS-based online learning for nonlinear probabilistic forecast combination*,
  International Journal of Forecasting,
  https://www.sciencedirect.com/science/article/pii/S0169207023001371 . Its
  accessible publisher search text discusses mismatch between weight-estimation
  and evaluation data. Full publisher page returned403; no implementation
  theorem has been verified from that page.

Potential next contrasts, subject to qualification and representative tests:

1. Preserve per-fold A/B matrices during an already necessary forecast pass.
   Prequential or leave-block-out reweighting can then evaluate weight
   stabilization without resimulating every component. Any stabilization strength
   must be chosen entirely inside the outer history, with support disclosed.
   Do not set a manual mixing percentage or fit to outer benchmark outcomes.
2. An online proper-score learner may respond to historical changes rather than
   treating all folds as exchangeable. Its update times must wait until each
   historical validation outcome is known. No learning from the current outer
   test outcome, future dates or prior benchmark scores is allowed.
3. Quantile-dependent combination is a different predictive construction, not
   a free change to the existing mixture. A univariate quantile improvement
   alone would not establish coherent joint asset paths and within-path
   portfolio behavior. Do not admit it before that construction is explicit.

The current166-option menu and running development stage are unchanged.
These sources motivate investigation rather than guaranteed financial gains,
independence claims or a stopping decision.
