# New research baseline: Shrunk Mean / Gaussian Uncertainty

**Complete 722 series and 36,813 forecasts; zero native failures and scoring integrity errors.**

The fixed equal whole-path bootstrap/GARCH-FHS pool now regularizes its historical mean using the smooth training-only rule `mu*t^2/(1+t^2)`. Its Gaussian mean-uncertainty dispersion, volatility fits, shock resampling, 120/120 weights, random namespaces, native accounting and both scoring profiles are unchanged. This is a working predictive regularizer, not a Bayesian-posterior optimality claim. See [DESIGN.md](DESIGN.md) and the exact [source snapshots](source).

| Model on identical full coverage | Primary Q | Sensitivity Q | Primary better than Public | Sensitivity better than Public |
|---|---:|---:|---:|---:|
| Public Adaptive Blocks | 0.5993442043272177 | 0.6140827734648610 | 0.000000000% | 0.000000000% |
| Previous Gaussian mean-pool winner | 0.5924398958508662 | 0.6069236896555463 | 1.151977182% | 1.165817398% |
| Shrunk Mean / Gaussian Uncertainty | 0.5909606942896891 | 0.6052612298671302 | 1.398780530% | 1.436539825% |

Lower Q is better. The full result improves the previous winner under both profiles; exact values are retained in [full-summary.json](full-summary.json). The three Public comparators and exact FAST R52 winner remain retained. FAST has 17 unchanged native fitting failures; its separate 36,796-successful-task paired comparison remains incomplete/preliminary and is not mixed into this full-coverage table.

All six asset classes beat Public, but not every family does. The new model wins 41 of 49 families versus Public. Its equity-class result regresses slightly relative to the previous winner while remaining better than Public; the other five classes improve; this is an overall-score promotion, not universal dominance.

| Asset class | Previous winner: primary versus Public | New winner: primary versus Public |
|---|---:|---:|
| commodities | 1.314541% better | 2.049343% better |
| cryptocurrencies | 0.964311% better | 3.243311% better |
| equities | 1.291220% better | 1.185233% better |
| fixed_income | 1.178694% better | 1.387389% better |
| foreign_exchange | 0.449217% worse | 0.706165% better |
| real_estate | 3.326724% better | 3.365201% better |

| Actual-calendar horizon band | New primary better than Public |
|---|---:|
| 1_30_calendar_days | 1.722916% |
| 31_365_calendar_days | 0.695841% |
| 366_1826_calendar_days | 1.541210% |
| 1827_7305_calendar_days | 2.897015% |
| 7306_plus_calendar_days | 5.276251% |

Band comparisons retain original positive v3 weights and represent additive score contributions, not renormalized standalone band leaderboards. [decomposition.json](decomposition.json) includes every family/band and origin-split result. All 36,813 checkpoint identities, complete file hashes and loss-vector hashes were checked; additive losses reconcile to both official scores within 3e-15. Fresh Public and previous-winner forecasts on all 4,122 representative origins exactly reproduced their original full-panel loss hashes, with zero scalar difference ([reference audit](reference-replay-audit.json)). The stable factory also replayed all 18 integration cases exactly.

The fixed development/reserved stages each covered 245 origins, followed by 4,122 origins across 81 representative series and this complete fresh full-panel run. Model selection used adaptively reused benchmark evidence; these results are not independent post-selection validation or significance claims. Forecast fitting uses only each origin’s available history. No family/horizon router or outcome-selected seed was fitted.

A separate 18-case, four-repeat, three-arm local warm compute check completed 216 forecasts and matched every full-panel loss hash. It showed comparable compute (median case ratio 0.9820528700512745 versus the previous winner). This is a representative local fit/simulation/exact-score diagnostic, not full-work speedup or deployed CTV proof. All raw timings are in [compute-results.json](compute-results.json).

Complete source, plans, all 50,649 new pilot/full loss vectors, full score cache, factory integration and audits were privately archived under `research-accuracy-mean-shrinkage-win-20261020T183043Z`. All 50726 files were freshly downloaded, disk-restored and SHA256 verified before redundant local raw outputs were retired. [archive-receipt.json](archive-receipt.json) contains digests and exact recovery commands. The full-panel catalogue now counts this as its 13th unique model version; older and unsuccessful models remain preserved.

**The accuracy-maximization goal remains active.** Finding this winner does not satisfy the stopping requirement. Continue from it and assess further credible mechanisms before deciding whether the search is exhausted.
