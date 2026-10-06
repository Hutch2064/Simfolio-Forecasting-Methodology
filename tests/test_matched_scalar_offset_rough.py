"""Independent dense Gaussian conditioning and no-look-ahead checks."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz

spec=importlib.util.spec_from_file_location('matched_causal',Path(__file__).parents[1]/'tools/matched_scalar_offset_rough/matched_causal.py')
causal=importlib.util.module_from_spec(spec);sys.modules[spec.name]=causal;spec.loader.exec_module(causal)


@pytest.mark.parametrize('rho',[0.,.3,.95,.999])
def test_one_step_predictor_matches_dense_gaussian(rho):
    y=np.random.default_rng(43).normal(size=25);level=.4;sd=.1;noise=4.9
    latent=sd**2/(1-rho*rho)*toeplitz(rho**np.arange(len(y)))
    observation=latent+noise*np.eye(len(y));expected=np.full(len(y),level)
    for t in range(1,len(y)):expected[t]+=latent[t,:t]@np.linalg.solve(observation[:t,:t],y[:t]-level)
    np.testing.assert_allclose(causal.predict(y,level,rho,sd,noise),expected,atol=1e-13)


def test_current_and_future_observations_cannot_affect_offset():
    y=np.random.default_rng(49).normal(size=25);changed=y.copy();changed[12:]+=50.
    np.testing.assert_array_equal(causal.predict(y,.4,.9,.1,4.9)[:13],causal.predict(changed,.4,.9,.1,4.9)[:13])
    np.testing.assert_array_equal(causal.predict(y,.4,.9,0.,4.9),np.full(len(y),.4))


def test_nearly_unit_persistence_matches_extended_precision_reference():
    rho=np.nextafter(1.,0.);y=np.array([0.,1.,0.]);r=np.longdouble(rho)
    noise=np.longdouble('4.9');variance=np.longdouble(1.)/(1-r*r);mean=np.longdouble(0.);expected=[]
    for value in y:
        expected.append(float(mean));gain=variance/(variance+noise)
        mean=r*(mean+gain*(value-mean));variance=r*r*(variance*noise/(variance+noise))+1.
    np.testing.assert_allclose(causal.predict(y,0.,rho,1.,4.9),expected,atol=1e-15)
