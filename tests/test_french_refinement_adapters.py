"""Exact executable-source parity, with bounded synthetic history only."""

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology.french_panel import refinement_adapters as adapters
from simfolio_forecasting_methodology.french_panel.refinement_runtime import verify_source_closure
from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData


def test_factories_are_lazy_no_argument_and_exact_id(monkeypatch):
    monkeypatch.setattr(adapters, "_supplements", lambda: pytest.fail("unexpected install"))
    for factory, identifier in ((adapters.build_fast, adapters.FAST_MODEL_ID),
                                (adapters.build_iid, adapters.IID_MODEL_ID)):
        assert not inspect.signature(factory).parameters
        assert factory().model_id == identifier


def test_frozen_source_closure_matches_scored_manifest():
    manifest = verify_source_closure()
    assert manifest["snapshot_tools"]["tools/data_selected_timescales/models.py"] == (
        "478bdd30149a1ec545b1e20b562e64d338c13170493ba9514bfa1094cc3bc51c"
    )
    assert manifest["supplements"][
        "refinement52/unified_frontier/iid_original_nodes/models.py"
    ] == "89116661fb9938b999be90992d2e93aa458b11301a6b517928c9061d9e36b2e6"


@pytest.mark.parametrize("paths,horizon", [(239, 21), (240, 0)])
def test_requested_paths_and_horizon_are_not_silently_changed(paths, horizon):
    x = np.zeros(100)
    training = TrainingData(x, x[:, None], PortfolioPolicy(("x",), (1.,), "none"))
    with pytest.raises(ValueError, match="240 paths"):
        adapters.build_fast().simulate_daily_log_returns(
            training, ForecastContext(adapters.FAST_MODEL_ID, "x", "test", horizon, paths, 8)
        )


# Execute the original, sealed scored functions independently of the new
# adapter, supplying only its initialized native numerical owner and a graph
# of their original dependencies. AST extraction skips import-time experiment
# runners and canonical input readers; it does not rewrite forecast functions.
PARITY_SCRIPT = r'''
import ast
import contextlib
import hashlib
import inspect
import json
import sys
import time
from types import SimpleNamespace as NS
import numpy as np
import pandas as pd
from simfolio_forecasting_methodology.french_panel import refinement_adapters as a
from simfolio_forecasting_methodology.french_panel.catalogue_adapters import initialized_m261_owner
from simfolio_forecasting_methodology.french_panel.refinement_runtime import SOURCE_ROOT, load_supplement
from simfolio_forecasting_methodology.runner import TrainingData, PortfolioPolicy, ForecastContext

def functions(relative, names, **namespace):
    path = SOURCE_ROOT / 'refinement52' / relative
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
    assert len(tree.body) == len(names)
    scope = {'np': np, 'time': time, 'hashlib': hashlib, 'inspect': inspect,
             'contextmanager': contextlib.contextmanager, **namespace}
    exec(compile(tree, str(path), 'exec'), scope)
    return NS(**{name: scope[name] for name in names}), scope

kind = sys.argv[1]
rng = np.random.default_rng(714)
h = np.zeros(384)
for t in range(1, len(h)):
    h[t] = .94*h[t-1] + .16*rng.normal()
x = .0002 + .012*np.exp(.5*h)*rng.standard_t(7, size=len(h))
dates = pd.bdate_range('1950-01-02', periods=len(x))
training = TrainingData(x, x[:, None], PortfolioPolicy(('synthetic',), (1.,), 'none'), dates.to_numpy())
model = a.build_fast() if kind == 'FAST' else a.build_iid()
context = ForecastContext(model.model_id, 'synthetic', 'bounded-parity', 43, 240, 982341,
                         pd.bdate_range(dates[-1]+pd.offsets.BDay(1), periods=43).to_numpy(), str(dates[-1].date()))
task = NS(training=training, horizon_days=context.horizon_days, seed=context.seed,
          portfolio_id=context.portfolio_id, origin_label=context.origin_label,
          future_dates=context.future_dates, origin_date=context.origin_date)
owner = initialized_m261_owner()
ref = NS(MODELS=owner, REFERENCE=owner.CANDIDATES[0], ForecastContext=ForecastContext, MODEL_ID=owner.MODEL_ID)
base, _ = functions('variance/study.py', ['context', 'forecast'], REF=ref)
if kind == 'FAST':
    old, _ = functions('innovation_moments/models.py', ['continuous_moments'])
    correction, _ = functions('innovation_mean_variance/models.py', ['transform'], OLD=old)
    body = owner.parent.body
    original_nodes = body.learned.nodes
    def corrected(data, n):
        return correction.transform(original_nodes(data, n), 'variance_only_preserve_mean')
    body.learned.nodes = corrected
    try:
        expected, _, _ = base.forecast(task, 'deterministic_RMS')
    finally:
        body.learned.nodes = original_nodes
else:
    scale, moments, marginal = a._supplements()
    def history_groups(histories):
        groups = np.array([histories.index(item) for item in histories], dtype=np.int64)
        return groups, [hashlib.sha256(item).hexdigest() for item in histories]
    C = NS(_M=NS(REF=ref, BASE=base), MARGINAL=marginal, MATH=NS(history_groups=history_groups))
    C._M.transform = lambda nodes, mode: nodes
    conditional, _ = functions('unified_frontier/conditional_candidates/models.py', ['_sha', '_forecast'],
                               _M=C._M, MATH=C.MATH, MARGINAL=marginal)
    C._forecast = conditional._forecast
    C.forecast = lambda task, kind, runtime=True: C._forecast(task, kind)
    body = owner.parent.body
    refs = {'fit': body.rough.rough_fit, 'prepared': body.pathwise.prepared,
            'normalizer': owner.overlay.path_normalizers, 'fit_seconds': owner.CANDIDATES[0].fit_seconds}
    G = NS(_C=C, _REFS=refs, MOMENTS=moments)
    preparation, _ = functions('unified_frontier/total_mean_candidates/models.py', ['sha', 'prepare'])
    preparation.prepare.__globals__['MOMENTS'] = moments
    T = NS(G=G, prepare=preparation.prepare)
    iid, _ = functions('unified_frontier/iid_conditional_scale_candidates/models.py',
                       ['sha', 'iid_law', 'forecast'], _T=T, SCALE=scale,
                       KIND='iid_conditional_unit_second_moment_scale')
    iid._T = T
    original, _ = functions('unified_frontier/iid_original_nodes/models.py',
                            ['original_nodes', 'forecast'], IID=iid,
                            KIND='iid_conditional_unit_second_moment_scale_original_nodes')
    expected, _, _ = original.forecast(task, runtime=False)
owner.clear_path_cache()
actual = model.simulate_daily_log_returns(training, context)
assert actual.shape == expected.shape == (240, 43)
assert actual.tobytes() == expected.tobytes(), (np.max(np.abs(actual-expected)), kind)
assert np.isfinite(actual).all()
repeat = model.simulate_daily_log_returns(training, context)
assert repeat.tobytes() == actual.tobytes()
print(json.dumps({'kind': kind, 'complete_daily_path_byte_parity': True,
                  'deterministic_replay_byte_parity': True, 'shape': list(actual.shape),
                  'sha256': hashlib.sha256(actual.tobytes()).hexdigest()}))
'''


@pytest.mark.parametrize("kind", ["FAST", "IID"])
def test_exact_scored_source_forecast_parity_in_fresh_process(kind, tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment["OPENBLAS_NUM_THREADS"] = "1"
    # The calling readiness job supplies the budgeted persistent kernel root.
    environment.setdefault("TMPDIR", str(tmp_path))
    environment.setdefault("NUMBA_CACHE_DIR", str(tmp_path / "numba"))
    result = subprocess.run([sys.executable, "-c", PARITY_SCRIPT, kind], check=False, cwd=root,
                            env=environment, text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"complete_daily_path_byte_parity": true' in result.stdout
