# Unified Frontier research: revised strict-speed objective

The current evidence does not identify a qualifying production replacement. The objective requires one methodology that improves both canonical panels while matching or beating the selected FAST candidate's runtime. Being faster than M261 alone is insufficient. No production changes or deployment were made.

The research followed the revised goal without changing its wording. Current evidence-supported methodological refinements and justified interactions have been tested or rejected by matched diagnostic/runtime gates. The stopping judgment is bounded by the available evidence; it is not a proof that no better model can exist. A new causal diagnostic or demonstrated implementation opportunity can justify reopening research.

| Methodology | Full 52-asset CRPS | Improvement vs M261 | Full 80-portfolio CRPS | Improvement vs M261 |
|---|---:|---:|---:|---:|
| M261 production reference | 0.6138695810898457 | reference | 0.22672308848311776 | reference |
| Production with rough multiplier removed (R0) | 0.6033109364170391 | +1.720% | 0.23051618147311073 | -1.673% |
| Selected FAST: R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION | 0.5957716608071161 | +2.948% | 0.23457102422540146 | -3.461% |
| Original-node conditional second-moment IID variant | 0.600668345431752 | +2.150% | 0.22653257917294423 | +0.084% |

Lower CRPS is better. Positive improvement means a lower score. Panels are evaluated separately; these percentages are not a blended ranking. The original-node IID variant narrowly improves both production scores, but its fresh-fit time was approximately 2.02 times FAST for the measured single-asset long-horizon case and 1.47 times FAST for the measured six-asset long-horizon case. Its strict-speed requirement therefore fails; its 52 score is also about 0.822% worse than selected FAST.

Selected FAST's earlier matched single-asset fresh-fit benchmark was approximately 2.86 times faster than M261. That is local compute evidence, not a website latency measurement or a claim about every portfolio dimension and horizon.

Three same-law implementation branches were also measured against FAST. In-place RMS allocation and in-place accounting had small or mixed runtime effects. A literal-dimension native cDCC optimization reduced fresh-fit time by approximately 7.06% and cached-fit time by 17.08% in the measured six-asset long-horizon case; single-asset fresh timing was essentially flat. Its 27-trial audit covers 162 forecast invocations with bounded parity evidence. No new full-panel score was measured for this implementation, and bounded parity does not establish all-work or production validation. An exact implementation optimization preserves the forecast law and cannot resolve FAST's 80-panel score deficit by itself. Unpaired timing medians are not added together to claim an unmeasured combined speedup.

The outstanding growth diagnostic is shared rather than eliminated. At 8,766 trading days, among 46 positive-history assets, typical mean/median path-CAGR ratios to historical growth were around 0.75 for M261, selected FAST and the original-node IID variant. This is a material historical shortfall to investigate, not a clean no-collapse pass. Historical growth is not imposed as a forecast floor, and this diagnostic is not used to exclude sweep results. Portfolio-level growth diagnostics must not be described as per-underlying-asset validation.

All reported canonical scores remain exploratory evidence from the evaluated data, seeds and origins. Saved audits, bounded seed/parity checks and archive restoration are distinct from genuinely independent post-selection validation, complete production integration, authenticated website performance and deployment. None of those latter claims is made here.

Recommendation under the revised requirements: retain M261 in production. Preserve selected FAST as the stronger 52-asset/speed tradeoff, the original-node IID variant as a joint-score but speed-rejected result, and native cDCC work as bounded implementation evidence. Do not manufacture a winner or route different methodologies by ticker, asset count, asset class or horizon.

Both catalogues must preserve completed results, rejected branches, source revisions, seeds, methodology, runtime and validation limits. Raw/private evidence is held in checksum- and readability-verified private archives; public-safe catalogue and closing-review material is preserved in Git. Final catalogue publication and local retirement receipts remain separate operational completion records.
