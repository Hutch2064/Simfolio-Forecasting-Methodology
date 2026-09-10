# Numerical environment and cross-platform parity

The strict numerical fixtures carry a source, data, and numerical-environment
identity. Their retained source replay was performed on macOS 27 arm64 with
Apple's Accelerate BLAS/LAPACK. The locked dependency legs are:

| Interpreter | NumPy | SciPy |
| --- | --- | --- |
| Python 3.11 | 2.4.6 | 1.17.1 |
| Python 3.12 | 2.5.3 | 1.18.1 |

The hosted reference job uses the explicit `macos-26` ARM64 label with CPython
3.11.15 and 3.12.13. The hosted image and the original macOS 27 fixture image
are different numerical environments, so a passing package or source replay
does not by itself establish byte identity across them. A retained fixture or
local source replay also does not establish live website, API, or deployed
server behavior.

## Current hosted evidence

[GitHub Actions run 34455737291](https://github.com/Hutch2064/Simfolio-Forecasting-Methodology/actions/runs/34455737291)
is the preceding macOS 26 ARM64 result. Each locked clean-wheel leg reported
158 passing checks and three INLA hash-only failures; the Frontier and factor
checks passed. Its separate Linux frozen-data job passed all nine portability
checks.

[Follow-up run 34456948385](https://github.com/Hutch2064/Simfolio-Forecasting-Methodology/actions/runs/34456948385)
completed successfully after the source-derived INLA array fix from `fefffec`.
Both macOS 26 ARM64 clean-wheel legs reported 161 passing checks, and the
Linux frozen-data job reported 9 passed. This is the current hosted CI result;
it ran at head `fefffecfaacdbe6177fb8c57f70c6bc86ac64061`. It validates the
locked package and fixture contract without certifying full
Linux numerical parity or live production behavior.

## Historical Linux comparison

The earlier Ubuntu clean-wheel result remains useful as historical
cross-platform evidence: [run 34450295393](https://github.com/Hutch2064/Simfolio-Forecasting-Methodology/actions/runs/34450295393)
reported 18 strict parity/data failures and 133 passes in each Python leg.
Those failures included numerical parity checks as well as the then-unresolved
canonical data fingerprint check. The later Linux frozen-data result above
supersedes the old data-failure conclusion, while full Linux numerical parity
remains unsupported.

Representative strict differences recorded by the historical run were:

| Matrix leg | Check | Maximum absolute difference |
| --- | --- | ---: |
| Python 3.11 | Base standardized residuals | 4.42046134e-05 |
| Python 3.11 | Current-production Frontier paths | 9.68257947e-06 |
| Python 3.11 | GJR reference paths | 3.86743633e-08 |
| Python 3.11 | GAS reference paths | 1.6479873e-17 |
| Python 3.12 | Base standardized residuals | 0.24567462 |
| Python 3.12 | Current-production Frontier paths | 8.84260058e-08 |

The Ubuntu log does not identify its BLAS/LAPACK provider, so it does not
support attributing those differences to a particular Linux backend. Where a
check compares numeric arrays, it uses its predeclared contract (including
`rtol=0`, `atol=2e-12` for the relevant factor-source arrays). Source and
resource integrity digests remain exact; no ad hoc tolerance widening replaces
an identity check.

## Native versus conditional production replay

Native Frontier factor fitting is sensitive to matrix-factor orientation and
eigenvector sign conventions. Those representations can differ across BLAS
implementations while the implied covariance is equivalent. Native same-seed
paths can differ because factor innovations use those coordinates. Tests
therefore check fitted-state equivalence under a consistent sign transform,
and separately compare simulation using the retained source orientation.
Native float64 array bytes are not a portable identity across backends.

The bounded current-production replay tests a narrower storage contract. It
quantizes the public marginal paths to float32, runs the recorded uniforms and
rank mapping, quantizes the mapped paths to float32, and performs the public
portfolio rejoin. For the three bounded cases in the audit (one synthetic and two canonical-data cases), the mapped arrays
and rejoined portfolio arrays were byte-identical to the aligned production
evidence. This is conditional production-storage parity, not a claim of
universal native Frontier byte parity.

The production comparison used local source replay and frozen inputs. No live
API call or website execution was performed, and deployment behavior is not
inferred from these files.

## Reference-platform policy

`macos-26` is the current explicit ARM64 CI reference label. Each strict job
should retain `platform.uname()`, the exact interpreter build, NumPy/SciPy
versions, BLAS/LAPACK identity, and relevant thread settings alongside source
and data digests. Run 34456948385 is green for both locked macOS legs and the
Linux frozen-data job; changing the reference image or numerical backend still
creates a new fixture identity.

GitHub documents `macos-26` in its
[hosted-runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and lists image labels in the
[runner-images repository](https://github.com/actions/runner-images).
The `macos-latest` label can migrate between OS images, so it is unsuitable as
an immutable numerical identity. Changing the reference image, interpreter
build, or numerical backend creates a fixture identity that must be regenerated
or independently reconciled.
