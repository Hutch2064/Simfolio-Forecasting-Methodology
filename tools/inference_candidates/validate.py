"""Numerical checks before any forecasting screen."""
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mcmc_runtime'))
import numpy as np
from inference import support
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from steady_kalman_target import target
from target import theta_target_gradient, unconstrained_target_gradient, variational_objective

rng=np.random.default_rng(20261003)
y=rng.normal(-7.,2.5,507)
lower,upper=support(y)
errors=[];gradient_errors=[];transformed_errors=[]
for i in range(30):
 theta=np.array([rng.uniform(-10.,-4.),rng.uniform(-3.,6.5),rng.uniform(np.log(.03),np.log(2.))])
 value,grad=theta_target_gradient(y,theta)
 reference=target(y,*theta,y.mean())
 assert abs(value-reference)<1e-8,(value,reference)
 numeric=approx_derivative(lambda t: target(y,*t,y.mean()),theta,method='3-point').ravel()
 assert np.allclose(grad,numeric,rtol=2e-4,atol=2e-5),(grad,numeric)
 errors.append(abs(value-reference));gradient_errors.append(float(np.max(abs(grad-numeric))))
 u=rng.normal(size=3)
 value,grad=unconstrained_target_gradient(y,u,lower,upper)
 theta=lower+(upper-lower)*expit(u)
 jac=(upper-lower)*expit(u)*(1-expit(u))
 assert abs(value-target(y,*theta,y.mean())-np.log(jac).sum())<1e-8
 numeric=approx_derivative(lambda v: unconstrained_target_gradient(y,v,lower,upper)[0],u).ravel()
 assert np.allclose(grad,numeric,rtol=2e-4,atol=2e-5),(grad,numeric)
 transformed_errors.append(float(np.max(abs(grad-numeric))))
p=np.r_[np.zeros(3),np.log(.4),.1,np.log(.3),-.1,.2,np.log(.2)]
z=rng.normal(size=(64,3))
value,grad=variational_objective(y,p,z,lower,upper)
numeric=approx_derivative(lambda v: variational_objective(y,v,z,lower,upper)[0],p).ravel()
assert np.allclose(grad,numeric,rtol=2e-4,atol=2e-5),(grad,numeric)
report={'target_value_max_abs_error':max(errors),'theta_gradient_max_abs_error':max(gradient_errors),
 'bounded_transform_gradient_max_abs_error':max(transformed_errors),
 'full_rank_vi_gradient_max_abs_error':float(np.max(abs(grad-numeric))),'checks':30,'passed':True}
print(json.dumps(report,indent=2))
