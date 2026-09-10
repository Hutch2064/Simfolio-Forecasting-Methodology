# Install and run

Use Python 3.11 or 3.12. Install from this repository; no PyPI publication is
required. The numerical dependency matrix is pinned in `requirements-lock.txt`.

```bash
git clone https://github.com/Hutch2064/Simfolio-Forecasting-Methodology.git
cd Simfolio-Forecasting-Methodology
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install '.[all-models]'
```

Inspect the packaged evidence without running forecasts:

```bash
simfolio-oos data verify --json
simfolio-oos validate --quick --json
simfolio-oos catalogue --json
simfolio-oos coverage --json
simfolio-oos scores --experiment canonical-whitepaper --json
simfolio-oos canonical-175 --plan --json
```

The snapshot contains 52 asset series, six supporting series (including the
canonical drift-factor proxies and `EFFRX`), and French and Q5 factor inputs.
Verification checks exact hashes. Preparation reconstructs the 11,687-date
common calendar and matrix fingerprints offline:

```bash
simfolio-oos data prepare --destination .simfolio-oos-data --json
```

A bounded Frontier check using the frozen canonical inputs:

```bash
simfolio-oos canonical-175 \
  --model asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced \
  --smoke --data .simfolio-oos-data --task-count 1 --horizon 8 \
  --simulations 16 --output results/frontier-smoke
```

Smoke outputs are noncanonical. They explicitly reduce portfolios, origins,
horizons, and simulations. Omitting `--data` in smoke mode uses a labeled
synthetic test fixture instead of the canonical dataset.

The later complete experiment uses 175 models, 80 portfolios, 4,080 tasks per
model, 240 simulations per origin, and 701,280 portfolio-horizon cells per
model. It preflights every selected factory and refuses incomplete coverage.
**These full-study commands are not installation tests:**

```bash
simfolio-oos canonical-175 --output results/canonical
simfolio-oos canonical-175 --output results/canonical --resume
```

Resumption requires identical model, implementation, data, panel, protocol,
and task identities. Failed tasks remain visible and cannot improve the score
by reducing its denominator. Read `docs/historical-runtime-discrepancies.md`
before interpreting a new result as a reproduction of retained scores.

For an isolated wheel installation, run from a clean clone:

```bash
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python -m venv /tmp/simfolio-wheel-check
/tmp/simfolio-wheel-check/bin/python -m pip install -r requirements-lock.txt
/tmp/simfolio-wheel-check/bin/python -m pip install --no-deps dist/*.whl
cd /tmp
/tmp/simfolio-wheel-check/bin/simfolio-oos data verify --json
/tmp/simfolio-wheel-check/bin/simfolio-oos validate --quick --json
```

CI checks wheel contents against tracked package files and runs the complete
bounded suite against the installed wheel outside the checkout on both
supported Python versions.
