"""M212 with conventional SV observation variance learned jointly with parameters.

Gaussian log-square quasi likelihood; original priors retained for SV parameters,
flat log-observation-variance prior. Rough uses the same learned variance. Mean
curve is the original predecessor's, including its century-long growth behavior.
"""
from functools import lru_cache
import hashlib,importlib.util,math
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('joint_noise_private_m212',ROOT/'tools/consistent_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
sys.path.insert(0,str(ROOT/'tools/empirical_conventional_sv_rough'))
import gaussian_sv
streamed=parent.parent.streamed;original_fit=streamed.predecessor_fit;original_noise=parent.parent.base.noise.measurement_scale
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))+(ROOT/'tools/empirical_conventional_sv_rough/gaussian_sv.py').read_bytes()).hexdigest()

@lru_cache(maxsize=64)
def refit(data):
    def compute():
        original=original_fit(data);x=np.frombuffer(data,np.float64);y,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()),winsorize=True)
        _,initial_R=original_noise(data);center=float(y.mean());lo,hi=np.quantile(y,[.01,.99])
        bounds=[(lo-4,hi+4),(-7.,7.),(math.log(.02),math.log(2.5)),(math.log(np.finfo(float).eps*max(float(y.var()),np.finfo(float).tiny)),math.log(np.finfo(float).max)/2)]
        def target(theta):
            level,p,e,r=theta;phi=expit(p);eta=math.exp(e);noise=math.exp(r)
            if phi>=.999:return 1e100
            ll=gaussian_sv.filter_states(y,level,phi,eta,noise)[0]
            prior=-.5*((level-center)/4)**2-.5*((phi-.94)/.20)**2-.5*(e-math.log(.35))**2+math.log(phi*(1-phi))+e
            return -float(ll+prior)
        level,phi,eta=original['posterior_center'][:3];start=[level,logit(phi),math.log(eta),math.log(initial_R)]
        result=minimize(target,start,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-10,'gtol':1e-5,'maxiter':200,'maxls':100})
        if not result.success:result=minimize(target,result.x,method='Powell',bounds=bounds,options={'ftol':1e-10,'xtol':1e-6,'maxiter':200})
        if not result.success or not np.isfinite(result.fun):raise ArithmeticError('joint observation-noise SV MAP failed')
        level=float(result.x[0]);phi=float(expit(result.x[1]));eta=math.exp(result.x[2]);R=math.exp(result.x[3]);ll,h,v=gaussian_sv.smooth_states(y,level,phi,eta,R)
        pool=shell.bd._standardized_empirical_innovation_pool(eps*np.exp(-.5*h),clip=None,method='mean_std')
        if pool is None:raise ArithmeticError('invalid joint-noise innovation pool')
        innov=(h[1:]-level-phi*(h[:-1]-level))/eta;rho=shell.bd._finite_correlation(pool[:len(innov)],innov)
        fitted=original.copy();fitted.update(posterior_center=(level,phi,eta,float(h[-1]),float(rho)),innovation_pool=pool,bdes_multiscale_vol=shell.bd._bdes_multiscale_components(h,4,shell.bd.BDES_MULTISCALE_GRID_FIXED),state_loglikelihood=float(ll),state_path_variance_last=float(v[-1]),joint_noise_fit={'success':bool(result.success),'evaluations':int(result.nfev),'objective':float(result.fun),'noise_variance':R,'initial_empirical_noise':initial_R})
        return fitted
    return shell.controls.cache('joint_noise_conventional_MAP',hashlib.sha256(data+SOURCE.encode()).hexdigest(),compute)

@lru_cache(maxsize=64)
def measurement_scale(data):
    R=refit(data)['joint_noise_fit']['noise_variance'];return math.sqrt(4.934802200544679/R),R

@lru_cache(maxsize=64)
def observed(data):
    x=np.frombuffer(data,np.float64);y,eps=shell.bd._sv_observed_log_variance(x,float(x.mean()));level,phi,eta,_,_=refit(data)['posterior_center'];_,R=measurement_scale(data)
    return y-parent.causal_predictor(y,level,phi,eta,R),eps

streamed.predecessor_fit=refit
parent.parent.base.base.observed=observed;parent.parent.base.noise.observed=observed
parent.parent.base.noise.measurement_scale=measurement_scale
parent.parent.SOURCE=hashlib.sha256((parent.parent.SOURCE+SOURCE).encode()).hexdigest()

def asset_paths(data,uniforms):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    from streamed_paths import map_asset_inplace
    mean,_=moment_return_curves(original_fit(data),uniforms.shape[1]);_,sd=moment_return_curves(refit(data),uniforms.shape[1]);n=len(uniforms)
    nodes=np.quantile(refit(data)['innovation_pool'],np.linspace(.5/n,1-.5/n,n));nodes-=nodes.mean();nodes/=np.sqrt(np.mean(nodes*nodes))
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes);return mean,paths
streamed.predecessor_asset_paths=asset_paths

class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context);data=np.asarray(training.asset_log_returns,float)
        for a in range(data.shape[1]):
            d=data[:,a].tobytes();r=FIT_DIAGNOSTICS[self.model_id,hashlib.sha256(d).hexdigest()];r['joint_noise_fit']=refit(d)['joint_noise_fit'];r['conventional_offset_noise']='same jointly learned observation variance as both volatility fits';r['conventional_parameter_fit']='joint MAP over level/persistence/amplitude/log observation variance';r['return_mean_curve']='unchanged_original_predecessor'
        return result

initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_joint_observation_noise_multiscale_dynamic_rough_map'),)
MODEL_IDS=(parent.MODEL_IDS[0],CANDIDATES[0].model_id)
clear_path_cache=parent.clear_path_cache;TIMINGS=parent.TIMINGS;FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths
