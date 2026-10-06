"""Long-horizon growth anchoring and unchanged volatility references."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/historical_log_mean_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_historical_mean_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.initialize();return m


@pytest.mark.parametrize('mu',[-.0002,.0001,.0005,.001])
def test_50_and_100_year_median_cagr_preserves_historical_log_growth(model,monkeypatch,mu):
    # Symmetric standardized innovation pool allows exact paired verification.
    x=mu+np.linspace(-.02,.02,500);data=x.tobytes()
    fit={'bdes_multiscale_vol':dict(phis=np.array([.7,.96]),b=np.array([.4,.8]),q_var=np.array([.02,.003]),
        residual_phi=-.1,residual_innovation_sd=.1,residual_common_loading=-.2,
        ell=1.,q_last=np.array([.2,-.1]),residual_last=.15),
        'base_fit':dict(sigma=.02,dlm_state_noise_var=0.,dlm_state_transition_phi=.8,
            dlm_long_run_anchor_mean=mu,dlm_state_posterior_deviation_mean=0.,dlm_state_posterior_deviation_var=0.),
        'mean_meta':dict(hac_long_run_variance=.0004),'n_obs':500,'innovation_pool':np.linspace(-3,3,101)}
    monkeypatch.setattr(model.streamed,'predecessor_fit',lambda d:fit)
    for horizon in [12600,25200]:
        mean,paths=model.predecessor_asset_paths(data,np.tile(np.array([.0625,.9375])[:,None],(1,horizon)))
        assert np.array_equal(mean,np.full(horizon,x.mean()))
        median_log_growth=np.median(paths.sum(axis=1))/horizon*252
        assert np.expm1(median_log_growth)==pytest.approx(np.expm1(x.mean()*252),abs=1e-14)
    assert model.streamed.rough_fit is model.parent.rough_fit
    assert model.streamed.prepared is model.streamed.base.prepared
