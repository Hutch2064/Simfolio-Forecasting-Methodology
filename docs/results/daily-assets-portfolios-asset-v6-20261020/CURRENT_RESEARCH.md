# Current asset-level accuracy research

The accuracy-maximization goal remains active. A verified winner is not the research stopping point.

## Official retained set: one main score

All retained methods below cover the identical64,049 successful forecasts across1,264streams at1,000simulations. Lower main Q is better. The fixed reporting reference is the worse completed asset-level TestFolio-style configuration, With Replacement.

| Method | Main Q | Better vs fixed reference | Recorded worker compute seconds |
|---|---:|---:|---:|
|Asset Blocks — With Replacement|0.6320715457043177|reference|13316.0431|
|Asset Blocks — Without Replacement|0.6310508866260718|0.1614784%|12956.4817|
|Asset GARCH Fade / Equal Mix / Gaussian Mean Uncertainty|0.6118108289933980|3.2054467%|10604.1350|

All retained methods have zero native failures and zero scoring integrity errors. The complete baseline pair and each promoted winner have independently verified source-bound checkpoint identities and canonical score reduction. Recorded worker compute includes input preparation, fitting, joint simulation and CRPS, including identity-verified reused pilot forecasts. It excludes IO/setup/queue and is not a controlled speedup comparison. These adaptively reused cases do not support independence or significance claims.

## Cohesive model and continuing research

The winner is a single fixed asset-level procedure. Its frozen implementation and full-panel catalogue specify the raw joint-row sampler, filtered variance law, mean treatment, joint shocks and portfolio accounting. Constituent paths and holdings evolve inside every scenario; no test-score-selected asset-class, family or horizon routing is used.

The circular raw-block contrast passed355-case development,355-case reserved and7,418-case broad checks at1,000paths before full admission. Official promotion occurs only after complete matched full results and independent verification. Low-simulation component and sampler screens remain exploratory and separate from official results and full-panel catalogue counts. The additive diagnostic reconciles to canonical Q and has zero additional score or model-selection weight.

The GARCH raw-fade contrast completed development, reserved, broad and the complete64,049-forecast panel with independent verification, then became the official best. Its scale returns toward historical variance; its mean and correlated mean uncertainty are preserved. Equilibrium renewal initialization and a plain AR1 empirical-innovation contrast did not improve their100-path screens. Neither result rules out a family.

Development is sample-first. The216-configuration study completed15,552 low-path forecasts. Twelve breadth-preserving contrasts completed355-case1,000-path confirmation; three stochastic-volatility variants improved development, reserved and7,418-case broad checks. The current working incumbent is the Gaussian conditional-RMS log-OU hybrid, separate from the last complete official GARCH-fade winner above. Its broader-sample Q is0.6472385110030847 versus0.6485664597681876 for that earlier winner on identical7,418cases/1,000paths. This is not a new full-panel result.

The premature conditional-RMS full run was intentionally checkpointed at40,710/64,049 successful native forecasts, with zero native or scoring errors. Every checkpoint remains preserved; this is a partial execution, not a completed full-panel catalogue entry. Automatic full evaluation after each gain is disabled. Continuous sample-level iteration and informative broad checks precede the unchanged evidence-based stopping review; the final retained full suite follows that review.

A19-configuration conditional-correlation screen completed1,368forecasts without improving the working control. The13-law component-weight ablation improved development but regressed on reserved cases, so no promotion occurred. Fixed-percentage grids are discontinued in favor of training-derived parameters and single generative mechanisms. These tests do not rule out entire families.

Fourteen new single-law candidates estimate joint block dependence from training returns and volatility clustering, with no fixed component allocation. All1,008100-path forecasts and72exact control replays completed successfully. Six advanced to355-case1,000-path development checks; none improved that development sample. The leading GARCH- and EWMA-faded raw laws also regressed on355reserved cases, so neither was promoted. Their stationary-bootstrap mean covariance is derived from the fitted dependence law and verified against explicit Markov and Monte Carlo calculations. Every screen remains exploratory and adaptively reused; no independence or significance claim. The final stopping requirement remains unchanged.

A second12-law single-model study fits latent log-variance by training Gaussian measurement quasi-likelihood, with causal prediction residuals, Gaussian/Student/empirical joint shocks, and two data-based mean-uncertainty estimators. All864screen forecasts and72exact working-control replays succeeded, but none improved the same100path working score; no expensive escalation was admitted. Four qualification tests checked analytical gradients, causal variance timing, synthetic latent-state recovery, exact path counts and chunk invariance. This screens the stated QML proxy approximation, not all stochastic-volatility methods.

The new working incumbent is the EWMA-faded raw/conditional-RMS Gaussian pool with its shared component fraction estimated from training-only marginal leave-one-out CRPS. It improved355-case development,355-case reserved and7,418-case broad checks at1,000paths. All8,128selected-model raw loss hashes, independent origin scores and independent hierarchical reductions verified, along with a fresh native generator replay. Its broad Q is0.6469412535998392 versus0.6472385110030847 for the prior working incumbent and0.6809035594773647 for the exact matched fixed WR reference,4.9878291%better than WR on that broad sample. This is a working model, not a completed full-panel result. No full sweep is admitted.

Four empirical joint-regime contrasts and six whole-empirical-law uncertainty contrasts completed their100-path screens without beating the newly qualified working control. Their fitted parameters and negative evidence are preserved; no state/prior/seed retuning or full escalation followed. The next concrete research lead is the training-estimated latent return-mean state used in earlier M261 methods, with fading determined by its estimated persistence. It has not yet been admitted here.

The working-incumbent snapshot and research_workflow_policy.json distinguish development from the official retained result. Thirty to forty percent or greater outperformance is an informal ambition only, never a stopping threshold or a cap. No small win or a few rejected variants closes the goal. Shared training fits and concurrent hypotheses use at most12forecast workers. Full52/80panel evidence motivates mechanisms but its different scores are not numerically comparable with this Q.

## Preservation and recovery

- Private release research-asset-v6-baselines-1000-20261020: 134,653files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-hybrid-1000-20261021: 72,258files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-circular-1000-20261021: 72,249files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-GJR-pilot-20261021: 747files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-adaptive-pilot-20261021: 368files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-empirical-mean-pilot-20261021: 1,822files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-fading-1000-20261021: 81,122files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-equilibrium-pilot-20261021: 369files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-AR1-pilot-20261021: 368files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-feature-development-20261021: 23,707files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-feature-broad-20261021: 32,924files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-sample-refocus-20261021: 49,387files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-training-laws-20261021: 5,659files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-learned-pool-pilots-20261021: 3,094files fresh-downloaded, physically disk-restored and SHA256verified.
- Private release research-asset-v6-learned-pool-broad-20261021: 15,400files fresh-downloaded, physically disk-restored and SHA256verified.

Complete raw losses, implementations, fitting/mean kernels, controls/registry, seeds, admission and verification evidence are preserved privately. Public compact receipts provide all archive/manifest digests and dependency releases. Restore baseline shared/input archives and sealed numerical parents, then the hybrid parent source and selected candidate archives at the research owner root; verify every manifest path, size and SHA256 before use. The prior hybrid raw checkpoints were intentionally retired only after complete fresh remote recovery and file-level SHA256 verification; the separate retirement receipt and ARCHIVED_LOSSES marker document this. The baseline inputs and checkpoints and currently needed candidate checkpoints remain local active research dependencies. Their archived prior versions must not be treated as missing data or silently recomputed.

The catalogue retains all fully attempted implementation versions, including losers; identical reruns count once. Source and compact evidence are pushed to the research repository and separately fresh-download/disk-restore verified. No production implementation or deployment is changed.
