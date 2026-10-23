"""Joint Gaussian EGARCH path component, separate from all frozen banks."""
import hashlib,json
import numpy as np
from numba import njit
from candidates.asset_prior_best import mean
from candidates.asset_feature_batch.fits import correlation_root
from .fit import estimate,C


def signature(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float64).tobytes()).hexdigest()
def seed(namespace,value):return int.from_bytes(hashlib.sha256(f'{namespace}:{value}'.encode()).digest()[:8],'little')


def prepare(history,base=None):
    x=np.asarray(history,dtype=np.float64)
    if x.ndim!=2 or not x.shape[1]or not np.isfinite(x).all():raise ValueError('Finite joint historical prefix required')
    fitted=[estimate(x[:,a])for a in range(x.shape[1])]
    batch=mean.estimate(x)if base is None else base['mean']
    if not np.array_equal(batch['sample_mean'],x.mean(axis=0)):raise ValueError('Shared mean fit differs from prefix')
    pool=np.column_stack([f['standardized_residuals']for f in fitted])
    return {'history_sha256':signature(x),'training_rows':len(x),'fits':fitted,'parameters':np.column_stack([f['parameters']for f in fitted]),'scale':np.array([f['scale']for f in fitted]),'mean':batch,'root':correlation_root(pool),'pool':pool,'next_log_variance':np.array([f['next_log_variance']for f in fitted]),'gaussian_only':True}


@njit(cache=True,fastmath=False)
def advance(shocks,delta,mu,scale,parameters,level,target,holdings,asset_paths,portfolio_paths):
    paths,width,assets=shocks.shape;result=np.empty((paths,width))
    for p in range(paths):
        for t in range(width):
            maximum=-np.inf
            for a in range(assets):
                z=shocks[p,t,a];value=mu[a]+delta[p,a]+scale[a]*np.exp(.5*level[p,a])*z
                asset_paths[p,a]+=value;holdings[p,a]=target[a]+value;maximum=max(maximum,holdings[p,a])
                level[p,a]=parameters[0,a]+parameters[3,a]*level[p,a]+parameters[1,a]*(abs(z)-C)+parameters[2,a]*z
            total=0.
            for a in range(assets):total+=np.exp(holdings[p,a]-maximum)
            step=maximum+np.log(total);portfolio_paths[p]+=step
            for a in range(assets):holdings[p,a]-=step
            result[p,t]=portfolio_paths[p]
    return result


class GaussianEGARCH:
    def __init__(self,spec,bundle=None):
        if set(spec)!={'mean_mode','mean_uncertainty'}or spec['mean_mode']not in('sample','shrink','zero')or not isinstance(spec['mean_uncertainty'],bool):raise ValueError('Known existing mean treatments required')
        self.spec=dict(spec);self.bundle=bundle
        self.model_id='research_gaussian_egarch_component_v1_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        x=np.asarray(history,dtype=np.float64);w=np.asarray(weights,dtype=np.float64)
        if (x.ndim!=2 or w.shape!=(x.shape[1],)or not np.isfinite(x).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or min(horizon,width,paths)<1 or paths<2 or paths%2):raise ValueError('Finite dated-prefix-compatible joint request and even paths required')
        b=prepare(x)if self.bundle is None else self.bundle
        if b['history_sha256']!=signature(x):raise ValueError('EGARCH fit differs from historical prefix')
        if (b['parameters'].shape!=(4,len(w))or not np.isfinite(b['parameters']).all()or np.any(np.abs(b['parameters'][3])>=1)or not np.isfinite(b['scale']).all()or np.any(b['scale']<=0)):
            raise ValueError('Finite Gaussian EGARCH parameters and abs(beta)<1 required')
        batch={**b['mean']};mu=batch['sample_mean']
        if self.spec['mean_mode']=='sample':batch['center']=mu.copy()
        elif self.spec['mean_mode']=='zero':batch['center']=np.zeros_like(mu)
        delta=mean.draws(batch,paths,seed('FHS_joint_mean_uncertainty_v1',context_seed))if self.spec['mean_uncertainty']else np.tile(batch['center']-mu,(paths,1))
        rng=np.random.default_rng(seed('joint_feature_Gaussian_v1',context_seed));assets=len(w)
        target=np.full(assets,-np.inf);target[w>0]=np.log(w[w>0]);holdings=np.tile(target,(paths,1));asset_paths=np.zeros((paths,assets));portfolio_paths=np.zeros(paths);level=np.tile(b['next_log_variance'],(paths,1))
        for start in range(0,horizon,width):
            count=min(width,horizon-start);shocks=np.ascontiguousarray((rng.normal(size=(count,paths,assets))@b['root'].T).transpose(1,0,2))
            values=advance(shocks,delta,mu,b['scale'],b['parameters'],level,target,holdings,asset_paths,portfolio_paths)
            if not np.isfinite(values).all()or not np.isfinite(level).all()or not np.isfinite(asset_paths).all():raise ArithmeticError('Nonfinite EGARCH path/state; no clipping or path exclusions')
            self.last_log_variance=level.copy();self.last_asset_log_paths=asset_paths.copy();self.last_asset_weights=np.exp(holdings)
            yield start,values
