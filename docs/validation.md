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
- all 175 ledger rows and every registered factory instantiation;
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

The current release is incomplete. `simfolio-oos coverage` reports exact
verified counts and remaining fail-closed identities. The
historical score artifact is retained for inspection, but historical score
linkage and the full statistical-specification gate remain unresolved. CI
passing therefore proves packaging, public safety, bounded numerical parity,
and protocol/data identity; it does not prove reproduction of the complete
retained white-paper ranking.
