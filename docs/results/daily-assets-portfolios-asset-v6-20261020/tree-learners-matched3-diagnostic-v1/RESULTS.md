# Completed matched tree-learner diagnostic

Three predeclared development requests were evaluated across five learner families and two feature groups. Each final forecast used 100 paths and inner selection used two paths. The score is the v7 per-origin Q described in the README; lower is better.

The final comparison contains 30 valid cells: 27 original successes retained unchanged and three case-0 MVD cells from the source-bound numerical repair. One geometry/CART output was an exact parity check and was not added as a comparison cell. A separate read-only verifier recomputed all 30 Q values from retained loss vectors and passed.

| Feature group | Learner | Mean Q | Better than fixed_WR | Worse than endpoint leader |
|---|---|---:|---:|---:|
| MVD | cart | 0.53464782 | 39.421% | 19.958% |
| MVD | extra_trees | 0.48366653 | 45.198% | 8.519% |
| MVD | lightgbm | 0.56870443 | 35.562% | 27.599% |
| MVD | random_forest | 0.50833410 | 42.403% | 14.054% |
| MVD | xgboost | 0.57577492 | 34.761% | 29.186% |
| geometry | cart | 0.52510994 | 40.502% | 17.818% |
| geometry | extra_trees | 0.54267403 | 38.512% | 21.759% |
| geometry | lightgbm | 0.58076319 | 34.196% | 30.305% |
| geometry | random_forest | 0.57053968 | 35.355% | 28.011% |
| geometry | xgboost | 0.62244052 | 29.474% | 39.656% |

Reference mean Q: fixed_WR **0.88256677**; broad-tested endpoint leader **0.44569599**.

MVD / ExtraTrees is the lowest-Q row at 0.48366653, still 8.519% worse than the endpoint-leader reference on this cohort. The results do not establish population performance, significance, a full-panel ranking, promotion, or a canonical-score change.

The root-controlled full-panel evaluation is a separate run and is not represented by this diagnostic.
