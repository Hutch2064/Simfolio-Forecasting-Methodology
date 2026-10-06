"""Independent integration of interpolated quantile moments and fixed mean."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.integrate import quad

@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/continuous_quantile_norm_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_continuous_quantile',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

@pytest.mark.parametrize('raw',[[-4.,-.2,.1,.3,2.],[-2.,3.],[-5.,-3.,1.,10.]])
def test_continuous_moments_match_independent_piecewise_integration(model,raw):
    nodes=model.normalize(raw);grid=np.linspace(0,1,len(nodes))
    for order,expected in [(1,0.),(2,1.)]:
        value=sum(quad(lambda p:np.interp(p,grid,nodes)**order,a,b,epsabs=1e-12)[0] for a,b in zip(grid[:-1],grid[1:]))
        assert abs(value-expected)<1e-12


def test_century_mean_curve_unchanged(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    d=np.random.default_rng(10).normal(.0005,.01,600).tobytes()
    expected,_=moment_return_curves(model.parent.original_fit(d),25200)
    mean,paths=model.asset_paths(d,np.full((2,25200),.5))
    assert mean.tobytes()==expected.tobytes();assert np.isfinite(paths).all()
