# Component coverage review and remaining evidence gaps

The original166-option menu reused qualified implementations. It was a useful starting point, not a literature-complete universe. A large Cartesian recipe count does not prove broad methodological coverage. Earlier52/80panel aggregate scores are supporting evidence, not isolated feature effects or current-panel exclusions.

| Piece | Actually implemented and tested in selector studies | Clear remaining candidates, not yet admitted |
|---|---|---|
| Mean | Sample, zero, smooth shrinkage; AR1conditional return dynamics in AR1-GARCH; correlated mean uncertainty on/off | Broader causal AR/MA order selection, inexpensive local-level/Kalman mean; robust mean contrasts |
| Volatility/dynamics | Constant, mean-reverting EWMA, recursive EWMA, GARCH, GJR, HARCH, SV path, SV RMS, AR1-GARCH | EGARCH, APARCH/power variants, parsimonious ARCH; FIGARCH/MIDAS only after measured cost and proper path-law qualification |
| Innovation law | Gaussian, fitted standardized Student, jointly resampled empirical residuals; standardized sinh-arcsinh skew extension | Hansen skew Student and generalized-error laws; supported EVT tail splice is distinct, needs stable training-only threshold/tail-moment evidence |
| Resampling | Joint IID, uniform bounded blocks with replacement, without-replacement history cycles; corrected PPW stationary and circular in explicit extensions | PPW on filtered joint residuals and additional justified block-law interactions; no automatic cross-product of incompatible laws |
| Dependence | Shared asset row draws for bootstrap; inherited fitted joint dependence for parametric recipes | Additional cheap dependence specifications justified by historical winners and current weaknesses; neither independent assets nor generic correlation substitutes are presumed equivalent |
| Combination | History-only rolling-origin CRPS simplex mixture; original nested trained hybrid; exact flattened legs | Confirm broader flattened/PPW/skew interactions after learner/precision confirmation |

## Exact nested-hybrid decomposition

The old parent contains a marginal-training-CRPS raw/filtered mixture. Its exact raw leg includes circular joint blocks, EWMA scale fading, shrinkage and correlated mean uncertainty; its exact filtered leg is SV-RMS Gaussian with the same mean treatment. Plain uniform raw blocks are a different law. Setting the parent fraction to1or0 exposes its exact legs; learned mixture weights then come from the outer internal rolling-origin learner. No financial50/50weight is introduced. Tests compare corresponding segment paths byte-for-byte at equal path counts/seeds.

On72reused development requests,1000final paths,30inner paths and3predeclared seeds: original mean21.709025%better versus fixedWRbaseline; delete-parent21.554401%; exact flattened21.907525%; flattened plus correctedPPW21.921790%. Thus deletion alone is not supported by these results. Flattening is modestly helpful; no broad/full promotion or independent validation follows. Separate skew study expanded the tested menu to223recipes, but was one seed; the active precision study deliberately holds the original166menu fixed to isolate precision rather than silently mixing methodological changes.

## Primary sources checked

- Politis and White, Automatic Block-Length Selection for the Dependent Bootstrap,2004; corrected Patton,Politis,White2009 implementation described at https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.optimal_block_length.html . Current candidate uses the corrected estimator. Taking a common maximum across asset-return/squared-return estimates is our explicit joint-history heuristic, not an optimality theorem for forecastCRPS.
- Official arch volatility definitions, including EGARCH,APARCH,FIGARCH,HARCH,MIDAS,ARCH: https://bashtage.github.io/arch/univariate/volatility.html . These establish distinct standard specifications, not cheap runtime or superiority in this panel.
- EGARCH recurrence: https://bashtage.github.io/arch/univariate/generated/arch.univariate.EGARCH.html . The exponential recursion requires innovation/moment qualification; arbitrary heavy-tail combinations are not automatically valid finite-CRPS recipes.
- APARCH recurrence and estimated power: https://bashtage.github.io/arch/univariate/generated/arch.univariate.APARCH.html . Qualify fitted power, admissibility and joint simulation against a reference before screening.
- Hansen1994 autoregressive conditional density estimation, author copy: https://users.ssc.wisc.edu/~behansen/papers/ier_94.pdf . SkewStudent is distinct from our current standardized sinh-arcsinh law.
- Existing skew implementation source: Jones and Pewsey2009 author copy https://oro.open.ac.uk/22510/1/sinhasinh.pdf .

## Admission and efficiency

Document each method as tested, qualified-only, pending or rejected with evidence. Do not reject families from low mixture weight, earlier aggregate ranks or one failed recipe. Fit shared family parameters once per complete historical prefix and reuse across compatible laws/means; batch path generation and exact CRPS reductions. Preserve exact original controls. Additional component menus require numerical smoke qualification before the same frozen matched development screen and broader confirmation. No outer score is ever available to a forecast's inner learner. No stopping condition is met by this review.
