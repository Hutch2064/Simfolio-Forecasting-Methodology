"""Corrected PPW dependence options; shared asset rows and genuine holdings."""
from pathlib import Path
import hashlib,inspect
import arch
from arch.bootstrap import optimal_block_length
import numpy as np
from asset_paths import advance
from candidates.asset_adaptive_blocks.stationary import StationaryStream

def signature(x):return hashlib.sha256(np.ascontiguousarray(x,dtype=np.float64).tobytes()).hexdigest()

def fit_lengths(history,weights):
    x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
    if(x.ndim!=2 or len(x)<2 or x.shape[1]!=len(w)or not np.isfinite(x).all()or not np.isfinite(w).all()or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)):
        raise ValueError('Finite joint history and long-only unit portfolio weights required')
    active=x[:,w>0];centered=active-active.mean(axis=0);features=np.column_stack((centered,centered*centered));varying=np.any(features!=features[0],axis=0)
    if not np.isfinite(features).all():raise ArithmeticError('Nonfinite dependence features')
    if not varying.any():
        lengths={'stationary':1.,'circular':1.};reason='Every active asset has a constant historical return; all block lengths induce the identical point law'
    else:
        # This is the actual pinned estimator's lag-window feasibility condition,
        # not a fixed training-size threshold or a replacement model.
        window=int(np.ceil(np.sqrt(len(x))))+max(5,int(np.log10(len(x))))
        if window>len(x):raise ValueError(f'PPW estimator needs its lag window{window}within the{len(x)}observed rows')
        selected=optimal_block_length(features[:,varying])
        if not np.isfinite(selected.to_numpy()).all():raise ArithmeticError('Nonfinite corrected PPW estimate')
        lengths={kind:max(1.,min(float(len(x)),float(selected[kind].max())))for kind in ('stationary','circular')}
        reason='Largest corrected estimate across active-asset centered returns and centered squares; explicit dependence-preserving multivariate heuristic'
    return {'history_sha256':signature(x),'weights_sha256':signature(w),'training_rows':len(x),'lengths':lengths,'rule':reason,'active_features':int(varying.sum()),'estimator':{'library':'arch','version':arch.__version__,'source_sha256':hashlib.sha256(Path(inspect.getsourcefile(optimal_block_length)).read_bytes()).hexdigest(),'correction':'Patton, Politis and White2009','forecast_optimality_claim':False}}

class CircularStream:
    def __init__(self,n,horizon,paths,context_seed,length):
        if min(n,horizon,paths)<1 or not isinstance(length,(int,np.integer))or not 1<=length<=n:raise ValueError('Valid dimensions and feasible integer circular block length required')
        self.n,self.horizon,self.paths,self.length=n,horizon,paths,int(length);self.position=0
        self.rng=np.random.default_rng(np.random.SeedSequence(int(context_seed)).spawn(1)[0]);self.current=None
    def take(self,width):
        if width<1 or self.position+width>self.horizon:raise ValueError('Columns outside forecast')
        out=np.empty((self.paths,width),dtype=np.int64)
        for t in range(width):
            if(self.position+t)%self.length==0:self.current=self.rng.integers(0,self.n,self.paths)
            else:self.current=(self.current+1)%self.n
            out[:,t]=self.current
        self.position+=width;return out

class PPWJointBootstrap:
    def __init__(self,kind,prepared=None):
        if kind not in ('stationary','circular'):raise ValueError('Unknown corrected PPW bootstrap')
        self.kind,self.prepared=kind,prepared;self.model_id='selector_joint_corrected_PPW_'+kind+'_v1'
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','replacement':True,'rule':'Corrected PPW length, active-asset returns and volatility features; common row for all constituents','length_rounding':'Circular uses nearest feasible integer; stationary uses the real-valued mean of geometric block lengths','portfolio_accounting':'Existing joint asset/holdings kernel, daily target-weight restoration in the fixed panel','automatic_model_fallback':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64,rebalance_every=1):
        x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);f=self.prepared if self.prepared is not None else fit_lengths(x,w)
        if f['history_sha256']!=signature(x)or f['weights_sha256']!=signature(w):raise ValueError('Dependence fit differs from this historical request')
        if min(horizon,paths,width)<1 or rebalance_every<0:raise ValueError('Positive forecast dimensions and valid holdings policy required')
        length=f['lengths'][self.kind]
        stream=StationaryStream(len(x),horizon,paths,context_seed,length)if self.kind=='stationary'else CircularStream(len(x),horizon,paths,context_seed,max(1,min(len(x),int(np.floor(length+.5)))))
        target=np.full(len(w),-np.inf);target[w>0]=np.log(w[w>0]);holdings=np.tile(target,(paths,1));assets=np.zeros((paths,len(w)));portfolio=np.zeros(paths)
        self.last_dependence_fit=f
        for start in range(0,horizon,width):
            indices=stream.take(min(width,horizon-start));values=advance(x,indices,target,holdings,assets,portfolio,start,rebalance_every)
            if not np.isfinite(values).all():raise ArithmeticError('Nonfinite asset-level forecast; no path exclusions')
            self.last_asset_log_paths=assets.copy();self.last_asset_weights=np.exp(holdings)
            yield start,values
