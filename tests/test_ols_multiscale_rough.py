"""Independent OLS normal equations and preservation of the 100-year mean."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/ols_multiscale_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_ols_multiscale_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_multiscale_coefficients_satisfy_independent_ols_normal_equations(model):
    h=np.random.default_rng(42).normal(1.,.2,1000)
    q=model.components(h,4);centered=h-h.mean()
    states=np.column_stack([model.bd._bdes_ewma(centered,1-p) for p in q['phis']]);states-=states.mean(axis=0)
    np.testing.assert_allclose(states.T@(centered-states@q['b']),0.,atol=1e-12)
    np.testing.assert_allclose(q['b'],np.linalg.lstsq(states,centered,rcond=None)[0],rtol=5e-12,atol=1e-12)


def test_original_fit_and_century_mean_curve_unchanged(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    data=np.random.default_rng(33).normal(.0005,.01,600).tobytes()
    original=model.streamed.predecessor_fit(data);modified=model.fit_with_ols(data)
    assert all(modified[k] is v for k,v in original.items() if k!='bdes_multiscale_vol')
    expected,_=moment_return_curves(original,25200)
    mean,paths=model.predecessor_asset_paths(data,np.full((2,25200),.5))
    assert mean.tobytes()==expected.tobytes()
    assert np.isfinite(paths).all()
