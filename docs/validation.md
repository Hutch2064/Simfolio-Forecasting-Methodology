# Validation and release gates

The release workflow builds and tests the distribution in a clean directory
outside the checkout for Python 3.11 and 3.12. It installs the locked runtime
and `all-models` dependencies, redirects Numba's cache to the job's temporary
directory, builds the wheel without source-path imports, and records the wheel
SHA-256 and byte size.

The installed-wheel checks cover:

- package-resource access to the frozen 60-file data snapshot;
- exact data verification and offline preparation identities;
- 80 portfolios, 4,080 origin tasks, and 701,280 scored cells;
- all 175 ledger identities and every registered factory in the executable
  map;
- bounded Frontier and base-family source-parity fixtures; and
- public-safety and canonical-ledger checks.

The ordinary local checks are:

```bash
python -m ruff check src tests
python -m pytest -q
```

## Current matrix evidence

[GitHub Actions run 34455737291](https://github.com/Hutch2064/Simfolio-Forecasting-Methodology/actions/runs/34455737291)
is the preceding macOS 26 ARM64 result: each locked CPython 3.11.15 and 3.12.13
leg reported 158 passing checks and three INLA hash-only failures. All Frontier
and factor checks passed, and its separate Linux frozen-data job passed nine
checks.

[Follow-up run 34456948385](https://github.com/Hutch2064/Simfolio-Forecasting-Methodology/actions/runs/34456948385)
completed successfully with the source-derived INLA array fix from `fefffec`.
Each macOS 26 ARM64 clean-wheel leg reported 161 passing checks, and the Linux
frozen-data job reported 9 passed at head
`fefffecfaacdbe6177fb8c57f70c6bc86ac64061`. This closes the hosted numerical CI
gate for that revision. It does not establish full Linux numerical parity,
reproduce the retained score ranking, or establish live API behavior. Independent local clean-wheel checks passed all 161 tests on both Python
versions. The final Python 3.11 wheel at the same numerical head included all
155 tracked package files with zero missing, extra, or byte-mismatched files;
`SIMFOLIO_WHEEL_TEST=1` enabled the installation gate with zero skipped tests.

The older Ubuntu run 34450295393 recorded 18 strict parity/data failures in
each leg. The newer nine-check Linux frozen-data result supersedes its data
failure, while the historical numerical differences remain evidence that full
Linux numerical parity is unsupported. Native Frontier factor fitting is
sensitive to matrix-factor orientation and eigenvector signs across numerical
backends. The bounded current-production replay has a narrower contract: it
quantizes public marginals to float32, applies the recorded uniform/rank map,
quantizes the mapped paths to float32, and rejoins the public portfolio. The
three bounded replays (one synthetic and two canonical-data cases) were byte-identical under that aligned storage
contract; this does not establish universal native Frontier byte identity or
live API behavior.

## Identity and coverage gates

The CI matrix intentionally keeps the interpreter-specific scientific pins.
SciPy 1.17.1 is used on Python 3.11 and SciPy 1.18.1 on Python 3.12. Some NIG
MCMC path digests differ between those environments even when the extracted
implementation matches its corresponding source fixture exactly. Source and
resource integrity digests remain exact. Numeric array checks use only their
predeclared contracts, including `rtol=0`, `atol=2e-12` for the relevant
factor-source arrays; no ad hoc tolerance widening is used to turn a failing
identity check into a pass.

`simfolio-oos coverage --json` is the authoritative status report for the
canonical 175 rows. It separates recovered identity, source references,
protocol/data/panel fingerprints, executable factories, complete statistical
specifications, bounded source parity, and historical score verification.
Those fields must be read independently: a registered factory or a passing
bounded fixture does not make a retained score a reproduced result.

The latest canonical-only verification reports 175 recovered identities, 175
verified source references, protocol/data/panel fingerprints, registered
factories, instantiations, forecast smoke checks, and bounded source-parity
checks. It reports 175 complete statistical specifications and zero historical
scores verified; complete specifications and factory coverage are checked
separately for every row.

The retained score artifact is evidence only. Its digest status is recorded as
`mismatch_observed_vs_declared`, and no historical score is marked verified.
CI passing proves packaging, public safety, protocol/data identity, and the
bounded numerical checks that it names; it does not prove reproduction of the
complete retained white-paper ranking.

The checkout currently has no `LICENSE` file. License selection remains an
administrative handoff item; this documentation does not invent license terms
for the source or data.
