"""A single joint empirical-return model with a training-fitted regime chain."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from numba import njit
from candidates.asset_prior_best import mean
from . import fit

CONFIGS=[{'state_selection':states,'mean':'common'if common else 'conditional'}for states,common in itertools.product(('two','training_BIC'),(True,False))]
def identity(config):return 'model_testing_asset_empirical_regime_rows_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def seed(name,value):return int.from_bytes(hashlib.sha256(f'{name}:{value}'.encode()).digest()[:8],'little')


def prepare(history):
    x=np.asarray(history,dtype=float);f=fit.prepare(x);f['mean']=mean.estimate(x);f['history_sha256']=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();return f


@njit(cache=True,fastmath=False)
def advance(pool,cdf,transition,states,state_uniforms,row_uniforms,delta,target,current,assets,portfolio):
    width,paths=state_uniforms.shape;p=pool.shape[2];result=np.empty((paths,width))
    for t in range(width):
        for i in range(paths):
            states[i]=0 if state_uniforms[t,i]<transition[states[i],0]else 1
            state=states[i];row=np.searchsorted(cdf[state],row_uniforms[t,i],side='right');maximum=-np.inf
            for a in range(p):
                value=pool[state,row,a]+delta[i,a];assets[i,a]+=value;current[i,a]=target[a]+value;maximum=max(maximum,current[i,a])
            total=0.
            for a in range(p):total+=np.exp(current[i,a]-maximum)
            step=maximum+np.log(total);portfolio[i]+=step
            for a in range(p):current[i,a]-=step
            result[i,t]=portfolio[i]
    return result


class RegimeRows:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen regime configuration')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'regime_weights':'Training likelihood and posterior probabilities; no fixed mixing percentages','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
        if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all()or x.shape[1]!=len(w)or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or paths<2 or paths%2 or horizon<1:raise ValueError('Valid joint matrix and even ensemble required')
        f=self.bundle if self.bundle is not None else prepare(x);assert f['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        one=f['one_state']or(self.config['state_selection']=='training_BIC'and f['one_state_BIC']<=f['two_state_BIC'])
        if one:gamma=np.ones((len(x),1));transition=np.ones((1,1));filtered=np.ones(1)
        else:gamma=f['gamma'];transition=f['transition'];filtered=f['filtered']
        probabilities=gamma.T/gamma.sum(0)[:,None];cdf=np.cumsum(probabilities,axis=1);cdf[:,-1]=1.;pool=np.tile(x[None,:,:],(len(filtered),1,1))
        if self.config['mean']=='common':
            conditional=probabilities@x;pool+=x.mean(0)[None,None,:]-conditional[:,None,:]
        # One-state indexing uses two identical rows solely for the common fast kernel.
        if one:pool=np.repeat(pool,2,axis=0);cdf=np.repeat(cdf,2,axis=0);transition=np.array([[1.,0.],[1.,0.]]);filtered=np.array([1.,0.])
        states=(np.random.default_rng(seed('regime_initial_v1',context_seed)).random(paths)>=filtered[0]).astype(np.int64)
        row_rng=np.random.default_rng(seed('regime_rows_v1',context_seed));state_rng=np.random.default_rng(seed('regime_state_v1',context_seed));delta=mean.draws(f['mean'],paths,seed('regime_mean_v1',context_seed))
        target=np.full(x.shape[1],-np.inf);target[w>0]=np.log(w[w>0]);current=np.tile(target,(paths,1));assets=np.zeros((paths,x.shape[1]));portfolio=np.zeros(paths)
        for first in range(0,horizon,width):
            count=min(width,horizon-first);result=advance(pool,cdf,transition,states,state_rng.random((count,paths)),row_rng.random((count,paths)),delta,target,current,assets,portfolio)
            if not np.isfinite(result).all():raise ArithmeticError('All declared paths must remain finite')
            self.last_asset_log_paths=assets;self.last_asset_weights=np.exp(current);self.last_state_count=1 if one else 2;yield first,result
