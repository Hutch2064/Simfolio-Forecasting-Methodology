# Demand-driven historical mixture coefficients

Measured motivation: thirteen first-request profiles of the 100-path PPW screen
spent 1261.31 of 1601.91 inclusive prepare_pair CPU seconds in the original
coefficient accumulator, approximately 78.7%. These are profiled exploratory
requests, not complete 1,000-path production timings. On the closed 72-case
1,000-path study the path selector's final allocation used 1 to 9 component
options, median 3, out of the 166-option universe. Sparse weights do not by
themselves prove any unselected component could safely be removed in advance.

The columns module calculates A, diagonal B and requested B columns in the exact
original per-pair summation order, including the higher-index-first orientation,
equal values and chronological lead accumulation. It never truncates leads,
paths, portfolios or candidate options. Tests compare complete arrays exactly
against the existing dense accumulator and across chunk widths.

The solver implements fully corrective simplex column generation. A gradient
computed from selected full columns checks ALL options before acceptance. The
Frank-Wolfe gap bounds convex objective suboptimality; no fixed number of
components or total iterations is accepted. Triangle inequality B_ij<=A_i+A_j
provides a scale without constructing unknown matrix entries. Restricted solves
use the existing certified solver. Scale and stopping tolerances are floating
point numerical criteria, not a forecasting model count or runtime cap.

Primary basis: Jaggi (2013), Revisiting Frank-Wolfe: Projection-Free Sparse
Convex Optimization, sections 2 and 3 and Algorithm 4 (fully corrective):
https://proceedings.mlr.press/v28/jaggi13.pdf
This application uses its finite convex simplex certificate; it does not assert
future forecast accuracy, independent validation or a production speedup.

This is NOT a bit-identical replacement for the existing weight solver: feasible
weights can differ within the certificate, potentially changing integer path
allocations. The coefficient reducer is byte exact; the search needs matched
forecast validation before retention. Closed coefficient replay measures how
many full columns are requested and checks both objective certificates, without
claiming that precomputed matrix access measures historical generation cost.

Open engineering issue: generating a new column may require replaying historical
component paths or retaining sufficient component ensembles. Full ensemble
storage can be large; repeated generation may erase the pairwise-work savings.
No such storage or new scientific worker pool is admitted here. A complete
request timing and memory comparison must include those costs before accepting
this approach. The six unit tests and closed coefficient tests alone do not
prove a ten-second request, full pipeline parity or a full-panel improvement.
