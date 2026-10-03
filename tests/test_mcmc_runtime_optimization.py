"""Numerical contracts for the standalone MCMC runtime research tools."""
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('numba')

TOOLS=Path(__file__).resolve().parents[1]/'tools/mcmc_runtime'
sys.path.insert(0,str(TOOLS))
from cached_dependence import make as make_dependence
from compiled_chain import chain
from exact_path_kernels import pairwise_sum
from finite_multiscale import make as make_multiscale
from parameter_mcmc_tuned import chain as reference_chain
from predictive_sv_candidates import fast_rebalanced, fast_uniforms
from steady_kalman_target import likelihood
from streamed_paths import map_asset_inplace, rejoin, uniforms

from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg


@pytest.mark.parametrize('phi',[0.,.94,.998])
def test_exact_variance_fixed_point_likelihood(phi):
    y=np.random.default_rng(21).normal(-8,2,1000)
    assert likelihood(y,-8.,phi,.35)==bd._sv_kalman_filter(y,-8.,phi,.35)[0]

@pytest.mark.parametrize('burn,kept',[(80,128),(96,128),(256,512)])
def test_compiled_chain_preserves_draws_and_acceptance(monkeypatch,burn,kept):
    from research_jit_pilot import make_accelerators
    _,_,_,fast_filter=make_accelerators()
    monkeypatch.setattr(bd,'_sv_kalman_filter',fast_filter)
    y=np.random.default_rng(54).normal(-8,2,120);theta=(-8.,2.5,math.log(.35))
    actual=chain(y,theta,burn,kept,81);expected=reference_chain(y,theta,burn,kept,81)
    assert actual[0].tobytes()==expected[0].tobytes()
    assert actual[1]==expected[1]

@pytest.mark.parametrize('assets',[6,25,50])
@pytest.mark.parametrize('horizon',[126,1050])
def test_streaming_preserves_all_asset_and_portfolio_values(assets,horizon):
    rng=np.random.default_rng(31);history=rng.normal(size=(100,1))+.3*rng.normal(size=(100,assets));model=dg.fit_dynamic_gaussian_factor_model(history)
    simulations=16;original_rng=np.random.default_rng(82);actual_rng=np.random.default_rng(82)
    original=fast_uniforms(model,simulations,horizon,original_rng)
    actual=uniforms(model,simulations,horizon,actual_rng)
    assert actual.tobytes()==original.tobytes()
    assert actual_rng.bit_generator.state==original_rng.bit_generator.state
    mean=rng.normal(0,.01,horizon);sd=rng.uniform(.001,.03,horizon);nodes=np.sort(rng.normal(size=simulations))
    source=np.broadcast_to(np.clip(mean[None,:,None]+sd[None,:,None]*nodes[:,None,None],-1,1),(simulations,horizon,assets)).copy()
    expected=dg.map_uniforms_to_marginal_paths(source,original)
    for a in range(assets):map_asset_inplace(actual[:,:,a],mean,sd,nodes)
    assert actual.tobytes()==expected.tobytes()
    weights=np.repeat(1/assets,assets);mask=np.arange(horizon)%21==0
    assert rejoin(actual,weights,mask).tobytes()==fast_rebalanced(expected,weights,mask).tobytes()

@pytest.mark.parametrize('size',[6,8,25,50,129])
def test_rejoin_reduction_matches_numpy_order(size):
    values=np.random.default_rng(34).uniform(size=size)
    assert pairwise_sum(values)==values.sum()

def test_exact_covariance_cache():
    rng=np.random.default_rng(72);history=rng.normal(size=(1000,1))+.3*rng.normal(size=(1000,6));model=dg.fit_dynamic_gaussian_factor_model(history)
    original=dg.kalman_terminal_posterior(model);actual=make_dependence()(model)
    assert all(a.tobytes()==b.tobytes() for a,b in zip(actual,original))

@pytest.mark.parametrize('size',[507,2921,10024])
def test_finite_reductions_preserve_complete_fit(size):
    path=np.random.default_rng(size).normal(-8,2,size)
    for values in (path,np.where(np.arange(size)%100==0,np.nan,path)):
        original=bd._bdes_multiscale_components(values,4)
        actual=make_multiscale()(values,4)
        assert pickle.dumps(actual,protocol=5)==pickle.dumps(original,protocol=5)
