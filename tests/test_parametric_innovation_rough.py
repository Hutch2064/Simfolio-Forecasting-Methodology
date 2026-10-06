"""Distribution references and preservation of the baseline century mean curve."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.integrate import quad
from arch.univariate import SkewStudent

@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/parametric_innovation_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_parametric_innovation_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m

@pytest.mark.parametrize('df,skew',[(5.,-.4),(8.,0.),(50.,.6)])
def test_hansen_pdf_and_quantiles_match_independent_arch_implementation(model,df,skew):
    z=np.linspace(-4,4,200);p=np.linspace(.001,.999,200);reference=SkewStudent()
    expected=reference.loglikelihood([df,skew],z,np.ones(len(z)),individual=True)
    np.testing.assert_allclose(model.density.logpdf(z,1/df,skew),expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(model.density.ppf(p,1/df,skew),reference.ppf(p,[df,skew]),rtol=1e-11,atol=1e-11)

@pytest.mark.parametrize('u,skew',[(.2,-.4),(0.,.5)])
def test_density_has_zero_mean_and_unit_variance_including_normal_limit(model,u,skew):
    _,a,b=model.density.constants(u,skew);split=-a/b
    for order,expected in [(0,1.),(1,0.),(2,1.)]:
        f=lambda z:z**order*np.exp(model.density.logpdf(z,u,skew))
        value=quad(f,-np.inf,split,epsabs=1e-9)[0]+quad(f,split,np.inf,epsabs=1e-9)[0]
        assert abs(value-expected)<1e-8


def test_both_innovation_arms_preserve_mean_and_ensemble_moments(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    data=np.random.default_rng(10).normal(.0005,.01,600).tobytes()
    expected,_=moment_return_curves(model.streamed.predecessor_fit(data),25200)
    for kind in ['student','skew_student','gaussian']:
        nodes=model.nodes(data,kind,240)
        assert abs(nodes.mean())<1e-14;assert abs(np.mean(nodes*nodes)-1)<1e-14
        mean,paths=model.asset_paths(data,np.full((2,25200),.5),kind=kind)
        assert mean.tobytes()==expected.tobytes();assert np.isfinite(paths).all()


def test_candidate_registry_retains_production_reference_identity(model):
    assert model.MODEL_IDS[0]==model.parent.MODEL_IDS[0]
    assert model.MODEL_IDS[0] not in {c.model_id for c in model.CANDIDATES}
    assert set(model.MODEL_IDS[1:])=={c.model_id for c in model.CANDIDATES}
