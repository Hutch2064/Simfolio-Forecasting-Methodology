# Growth and conventional conditional-moment audit — 2026-10-19

This is an additive saved-data/source assessment. It did not import forecasting or financial modules, initialize a model, fit, forecast, launch workers, or alter frozen studies. The earlier growth correction and stopping reports remain unchanged.

The latest corrected-node / conditional-rough / conventional-on full run is `full-9953676cd480585f`: all 6,732 origins closed with zero failures, and its saved core audit reconstructed the score cells and 396 long-growth records. Its CRPS is 0.6095093947 on 52 (+0.710279% versus M261, but 2.305872% worse than selected RMS) and 0.2264887964 on 80 (+0.103338% versus M261 and 3.445535% better than selected RMS). It is a small aggregate gain over M261 in both panels, but it does not win most 80 portfolios against M261 (29/80) or selected RMS (45/80); it has no measured speed result.

The calculator reports each actual horizon separately. Each ratio is a row's forecast mean- or median-path CAGR divided by that same row's historical training CAGR; only positive-training rows enter the ratio summaries. The 80-panel values are matched by portfolio, index, horizon and exact training CAGR to M261 and selected RMS. For 52, the saved selected-RMS audit has horizon medians and counts, but not its 156 source rows; no M261 52 growth comparison is available, so neither is reconstructed here.

| Panel / horizon | Model | Positive histories | Mean-path ratio, median / mean | Median-path ratio, median / mean |
| --- | --- | ---: | ---: | ---: |
| 52 / 2,922 | Candidate | 47/52 | 0.778 / 0.766 | 0.750 / 0.755 |
|  | Selected RMS | 47/52 | 0.764 / — | 0.746 / — |
| 52 / 5,844 | Candidate | 46/52 | 0.858 / 0.824 | 0.822 / 0.850 |
|  | Selected RMS | 46/52 | 0.833 / — | 0.857 / — |
| 52 / 8,766 | Candidate | 46/52 | 0.766 / 0.766 | 0.748 / 0.763 |
|  | Selected RMS | 46/52 | 0.754 / — | 0.750 / — |
| 80 / 2,922 | Candidate | 80/80 | 0.741 / 0.725 | 0.718 / 0.719 |
|  | M261 | 80/80 | 0.731 / 0.719 | 0.715 / 0.712 |
|  | Selected RMS | 80/80 | 0.736 / 0.721 | 0.727 / 0.734 |
| 80 / 5,844 | Candidate | 80/80 | 0.931 / 1.025 | 0.875 / 0.933 |
|  | M261 | 80/80 | 0.917 / 1.003 | 0.865 / 0.918 |
|  | Selected RMS | 80/80 | 0.920 / 1.000 | 0.907 / 0.996 |
| 80 / 8,766 | Candidate | 80/80 | 0.970 / 1.091 | 0.849 / 0.966 |
|  | M261 | 80/80 | 0.955 / 1.076 | 0.839 / 0.957 |
|  | Selected RMS | 80/80 | 0.989 / 1.115 | 0.984 / 1.108 |

The table's first ratio statistic is the median across rows; the second is the arithmetic mean across rows. At all three horizons, none of the candidate's positive-history rows has a nonpositive mean-path or median-path CAGR. The selected 52 audit also records zero such rows. The 80 mean-path ratio's 8,766-day median (0.970) and mean (1.091) diverge materially; this is a warning against treating the average ratio as the typical portfolio. These historical-relative growth patterns mix each panel's portfolios, histories, rebalancing and costs. They do not identify a causal defect or support a horizon/portfolio switch.

The source check finds no uncorrected conventional second-moment discrepancy. The conventional factor applied by the frozen `paths.cpp` is

`S_c,t = exp(0.5 * (level + next_t - mm_t) - 0.25 * mv_t)`.

Under the actual linear-Gaussian forecast state, `level + next_t | training` has mean `mm_t` and variance `mv_t`. Therefore `E[S_c,t^2 | training] = E[exp(level + next_t - mm_t - mv_t/2) | training] = 1`. The copied future-state recursion and its Gaussian forecast moments use the same transition and innovation covariance (`state.py`); the conventional return-variance curve is calculated from that same mean/variance pair (`models.py`). This is conditional on the fitted origin state after integrating over future conventional-state innovations, not conditional on each already-simulated intermediate future state. It leaves the conventional multiplier's serial dependence intact.

Its first moment is `E[S_c,t | training] = exp(-mv_t/8)`, below one. That is a separate mean effect; it is not evidence that the conventional second moment needs another correction. A separately saved full-total-mean branch applies first-moment compensation using `vc + vr`, but it draws no future SV states; its scores are not a matched conventional-on test. The current saved evidence supplies no quantified conventional `E[S_c^2]` residual, so a second normalization stacked onto this conventional factor would double-correct its modeled second moment. The candidate's existing node transform only normalizes the continuous interpolated node variance before the original clipping; the source proof here does not claim final clipped return variance or mapped joint-moment parity.

No new non-speculative growth- or conventional-moment interaction emerges from this audit. The latest candidate already includes the exact node × retained-rough × conventional-on interaction. The rowwise growth differences are descriptive, and the frozen conventional factor is second-moment normalized under the fitted forecast law. The original-node conventional-on 80 comparison remains a separately scoped saved/source closure if it is still needed; this audit does not substitute a new candidate or decide a speed claim. A first joint aggregate score gain alone still does not satisfy the research stopping rule.

## Evidence and reproduction

The pure-stdlib calculator is `growth_selection_audit.py`; it asserts the 396 candidate rows, 156/240 panel split, exact 80 control keys and equal historical denominators, then writes `2026-10-19-growth-selection-audit.json`. Its output hash is pinned below. The 52 selected-RMS mean-of-ratios is intentionally blank because the row-level selected data are not in its saved independent audit.

At this read, the mutable `RESEARCH_LEDGER.md` had hash `1f1e9d78488bd66a5652f02a938b7af08bf2118a2fbc941ffd9ba37c53c2e67c` and still ended at the full-run launch; the completed run is instead bound here to its saved manifest/audit/completion and score files. `STOPPING_LEDGER.md` had hash `b4bd22bd39b9dab0279552bb18ad686771c1e5053bbe7e9c76cc111ad9208b98`. Neither ledger was edited.

| Pin | SHA-256 |
| --- | --- |
| `growth_selection_audit.py` | `49740ea6038c260a27773cb48c27553d545040b49cf319a81553705c38788b5f` |
| `2026-10-19-growth-selection-audit.json` | `40a976ef112e909e1410c2bb7dec99f85a9103d6cbc894928ca0c374c751ec40` |
| `conditional_rough_conventional_corrected_nodes/full-9953676cd480585f/long-growth.json` | `2ac3c66db967b9e2ebb00ed1675048a1db12db5611175ba446aa561ebbf0c71c` |
| `conditional_rough_conventional_corrected_nodes/full-9953676cd480585f/growth-diagnostics.json` | `d6210320136a324fa9bf68dabbcfd60ab56aa85bfe5e495160c73d8229047337` |
| `conditional_rough_conventional_corrected_nodes/full-9953676cd480585f/manifest.json` | `55fc9eda76030aca4d0c950ff49d34b28061bfda87a721d0cc8188c260b99d7a` |
| `conditional_rough_conventional_corrected_nodes/full-9953676cd480585f/audit.json` | `c502aa2aa00a696d724f8fca8a8d46676fc51ca93117e8d5ac6625c61fa40df5` |
| `conditional_rough_conventional_corrected_nodes/full-9953676cd480585f/completion.json` | `8dc0cf6f303741f6fe88499e4717df38c33b17c14249dcc19bf10f0944458b69` |
| `conditional_rough_conventional_corrected_nodes/full-9953676cd480585f/analysis.json` | `98dfeee83b909205dc440831a4ab6badf6eeef7e88948f0306b93e9b7f1b20c6` |
| `diagnostics/m261-relative-long-controls.json` | `9d39a18ffb340fa0b32ffd441bb3bd9839bbccbd31e091ce88891b6323fbd4f0` |
| `../rms-growth-independent-audit.json` | `f4e15f3e3ae6fb39dda95bfe09ffbafeb6fac97cea3a035decf24d8dcddb6063` |
| `conditional_rough_conventional_corrected_nodes/models.py` | `c664248a31d45685b0a1e0f81bec862eded6b0b0c608ac5bcfddb13073b60fd7` |
| `conditional_rough_corrected_nodes/node_math.py` | `7c32074ac14808ee7ae9b18d9e1dcdc77b758daf3f874ab6fa6d2251d5872f3f` |
| `total_mean_candidates/mean.py` | `ef702223135c6f3a604e6fb6e40a9a667295798ebe9cc323aa9ae42bdaf62806` |
| `conditional_candidates/marginal.py` | `9acd310070c57efa0ef16df425ed93cca105a116ba3867c06fb4f5bf440f4b8b` |
| `total_mean_candidates/full-1aff29ce9e6500cb/analysis.json` | `ea1be2b643c9c8387170174746fd7b3fdba7b6346de7e410d133209199a72d49` |
| `snapshot/tools/coupled_multiscale_rough/paths.cpp` | `141e6baeb5191239f0e4ec5fd2533dde289e55b61805600eacdc4ded3073cb9c` |
| `snapshot/tools/coupled_multiscale_rough/state.py` | `6c3ffb1297b81bf7cd472b5097b6a785134a135461c59dc03418c19e480d38fa` |
| `snapshot/tools/coupled_multiscale_rough/models.py` | `27cbbc58b354836986385b402b4c73ac9b0fe7f0b72cb8db88aadd1d4bd521d4` |
| `snapshot/src/.../asset_level/sv_moment_functions.py` | `f93fee50220f1c2abbaa6280ce260427e302b58aa483dd231d8b73e288bb9997` |
