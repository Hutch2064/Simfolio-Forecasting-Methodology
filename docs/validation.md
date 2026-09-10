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

The CI matrix intentionally keeps the interpreter-specific scientific pins.
SciPy 1.17.1 is used on Python 3.11 and SciPy 1.18.1 on Python 3.12. Some NIG
MCMC path digests differ between those environments even when the extracted
implementation matches its corresponding source fixture exactly. A digest
mismatch is investigated as a source or dependency identity issue; tolerances
are not widened to make the matrix green.

`simfolio-oos coverage --json` is the authoritative status report for the
canonical 175 rows. It separates recovered identity, source references,
protocol/data/panel fingerprints, executable factories, complete statistical
specifications, bounded source parity, and historical score verification.
Those fields must be read independently: a registered factory or a passing
bounded fixture does not make a retained score a reproduced result.

The latest canonical-only verification reports 175 recovered identities, 175
verified source references, protocol/data/panel fingerprints, registered
factories, instantiations, forecast smoke checks, and bounded source-parity
checks. It reports 159 complete statistical specifications and zero historical
scores verified; the remaining specification work is tracked per row rather
than inferred from factory coverage.

The retained score artifact is evidence only. Its digest status is recorded as
`mismatch_observed_vs_declared`, and no historical score is marked verified.
CI passing proves packaging, public safety, protocol/data identity, and the
bounded numerical checks that it names; it does not prove reproduction of the
complete retained white-paper ranking.

The checkout currently has no `LICENSE` file. That is an administrative release
item to resolve before public distribution; this documentation does not infer
license terms for the source or data.
