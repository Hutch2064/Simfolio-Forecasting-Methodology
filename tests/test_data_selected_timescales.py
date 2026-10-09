"""Numerical and scope contracts for the timescale-only M256 ablation."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('numba')
pytest.importorskip('pyvinecopulib')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/mcmc_runtime'))
spec = importlib.util.spec_from_file_location('tested_timescale_models',
    ROOT/'tools/data_selected_timescales/models.py')
models = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = models
spec.loader.exec_module(models)


def history():
    rng = np.random.default_rng(82)
    h = np.empty(504)
    h[0] = 0.
    for i in range(1, len(h)):
        h[i] = .8*h[i-1] + rng.normal()*.1
    return h


def test_supplied_rates_preserve_entire_original_component_construction():
    h = history()
    old = models.ORIGINAL_COMPONENTS(h, 4)
    new = models.components(h, old['phis'], half_lives=old['half_lives'])
    assert old.keys() == new.keys()
    for key in old:
        assert np.asarray(old[key]).tobytes() == np.asarray(new[key]).tobytes(), key


def test_only_component_fields_change_in_original_asset_fit(monkeypatch):
    x = np.random.default_rng(24).normal(.0003, .01, 504)
    monkeypatch.setattr(models.bd, '_bdes_multiscale_components', models.ORIGINAL_COMPONENTS)
    before = models.RAW_FIT(x, filtered_innovations=True, fixed_mean=True)
    monkeypatch.setattr(models.bd, '_bdes_multiscale_components', models.data_components)
    after = models.RAW_FIT(x, filtered_innovations=True, fixed_mean=True)
    assert before.keys() == after.keys()
    def exact(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                exact(a[key], b[key])
        else:
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    for key in before:
        if key != 'bdes_multiscale_vol':
            exact(before[key], after[key])


def test_rates_use_history_and_ignore_legacy_calendar_arguments():
    a = models.data_components(history(), 4, 'fixed_four_scale', fitted_phi=.8)
    b = models.data_components(history(), 99, 'other_calendar', fitted_phi=.8)
    assert a is b
    assert np.all((a['phis'] > 0.) & (a['phis'] < 1.))
    assert a['scale_count'] == a['timescale_selection']['component_count']
    assert models.CANDIDATES[0].model_id != models.parent.CANDIDATES[0].model_id


def test_retained_runtime_constructor_matches_supplied_rate_reference():
    # Numerical installation mutates shared reference kernels. Keep that
    # installation isolated from unrelated tests and their own reference setup.
    code = '''
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root/'src'), str(root/'tools/mcmc_runtime'),
               str(root/'tools/data_selected_timescales')]
import models
import numpy as np
models.initialize(None, 1, 1)
h = np.random.default_rng(82).normal(size=504)
for phis in (np.array([.8]), np.array([.2, .9, .99])):
    expected = models.components(h, phis)
    actual = models.OPTIMIZED_COMPONENTS(h, len(phis), supplied_phis=phis)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert np.asarray(actual[key]).tobytes() == np.asarray(expected[key]).tobytes(), key
'''
    subprocess.run([sys.executable, '-c', code, str(ROOT)], check=True)
