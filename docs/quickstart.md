# Clean wheel quickstart

Use a fresh Python 3.11 or 3.12 environment. The tested dependency matrix is
recorded in [`requirements-lock.txt`](../requirements-lock.txt); the optional
`all-models` extra is included in that lock.

```bash
python -m venv /tmp/simfolio-methodology-venv
source /tmp/simfolio-methodology-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'simfolio-forecasting-methodology[all-models]'
```

Verify the packaged data and inspect the value-free protocol plan:

```bash
simfolio-oos data verify --json
simfolio-oos data prepare --destination .simfolio-oos-data --json
simfolio-oos coverage --json
simfolio-oos canonical-175 --plan --json
```

The verifier checks the packaged 52 asset series, `EFFRX`, French and Q5
factor inputs, source hashes, the 11,687-date common calendar, and the
normalized return-matrix identity. Preparation is offline and writes only the
destination cache. It does not contact a data provider.

The plan is value-free. It describes 80 portfolios and 4,080 origin tasks with
701,280 scored cells per model. A bounded run can select the exact Frontier ID
and a small simulation count through the smoke command. The canonical command
preserves 240 simulations per origin and should be treated as a large research
run, not an installation check.

To build and inspect a wheel without importing the checkout, use a temporary
wheel and target directory:

```bash
wheel_dir=/tmp/simfolio-methodology-wheel
install_dir=/tmp/simfolio-methodology-installed
mkdir -p "$wheel_dir" "$install_dir"
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir "$wheel_dir"
python -m pip install --no-deps --target "$install_dir" "$wheel_dir"/*.whl
PYTHONPATH="$install_dir" python -c \
  'from simfolio_forecasting_methodology.data import verify_canonical_snapshot; print(verify_canonical_snapshot())'
```

The CI performs the same clean-wheel check for both supported Python versions,
then runs the bounded parity and public-safety gates followed by the complete
test suite.
