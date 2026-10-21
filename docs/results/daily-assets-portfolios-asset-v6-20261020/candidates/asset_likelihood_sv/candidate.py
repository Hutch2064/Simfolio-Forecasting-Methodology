"""Pure asset-level latent-volatility laws with training-estimated parameters."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from candidates.asset_feature_batch import fits,kernels
from candidates.asset_prior_best import mean
from candidates.asset_training_blocks.dependence import select_length,mean_fit
from . import fit

CONFIGS=[{'law':law,'innovations':innovation,'mean_covariance':uncertainty}for law,innovation,uncertainty in itertools.product(('SV_RMS','SV_path'),('empirical','Gaussian','Student'),('batch','stationary'))]
def identity(config):return 'model_testing_asset_likelihood_SV_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def seed(name,value):return int.from_bytes(hashlib.sha256(f'{name}:{value}'.encode()).digest()[:8],'little')


def prepare(history,base):
    x=np.asarray(history,dtype=float);n,p=x.shape
    bundle=fits.prepare(x,'constant',base);raw=bundle['raw'];f=bundle['filter'];proxies=[];pools=[];diagnostics=[]
    for a in range(p):
        y=(x[:,a]-raw['mean'][a])/np.sqrt(raw['target_variance'][a]);estimated=fit.estimate(y)
        for key in('level','sv_phi','V','q','m','P'):f[key][a]=estimated[key]
        pools.append(estimated['pool']);proxies.append(estimated['proxy']);diagnostics.append({k:estimated[k]for k in('objective','null_objective','attempts')})
    f['pool']=fits.normalized(np.stack(pools,axis=1));f['root']=fits.correlation_root(f['pool']);f['inverse_df']=fits.student_inverse_df(f['pool'])
    changes=np.diff(np.stack(proxies,axis=1),axis=0);active=np.flatnonzero(np.var(changes,axis=0)>0);f['vol_root']=np.eye(p)
    if len(active):f['vol_root'][np.ix_(active,active)]=fits.correlation_root(changes[:,active])
    length=select_length(x)['length'];bundle.update(stationary_mean=mean_fit(x,length),selected_mean_block_length=length,likelihood_diagnostics=diagnostics)
    return bundle


class LikelihoodSV:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen configuration')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'fixed_component_mixture':False,'fitting':'Training log-square QMLE; no held-out fit or case routing','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
        if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all()or x.shape[1]!=len(w)or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or paths<2 or paths%2 or horizon<1:raise ValueError('Valid joint history and even ensemble required')
        bundle=self.bundle if self.bundle is not None else prepare(x,fits.base_fit(x));assert bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        raw,f=bundle['raw'],bundle['filter'];p=x.shape[1];batch=bundle['mean']if self.config['mean_covariance']=='batch'else bundle['stationary_mean']
        delta=mean.draws(batch,paths,seed('likelihood_SV_mean_v1',context_seed));target=np.full(p,-np.inf);target[w>0]=np.log(w[w>0]);state=(np.tile(target,(paths,1)),np.zeros((paths,p)),np.zeros(paths))
        parameters=np.stack([f[k]for k in('alpha','beta','positive','negative','omega','initial','decay','level','sv_phi','q')]);variance=np.tile(f['initial'],(paths,1));ring=np.tile(f['ring'],(paths,1,1));previous=np.tile(x[-1],(paths,1))
        next_m=f['level']+f['sv_phi']*(f['m']-f['level']);next_P=f['sv_phi']**2*f['P']+f['q'];m=np.tile(next_m,(paths,1));P=np.tile(next_P,(paths,1))
        h=m+(np.random.default_rng(seed('likelihood_SV_vol_initial_v1',context_seed)).normal(size=(paths,p))@f['vol_root'].T)*np.sqrt(next_P)
        row_rng=np.random.default_rng(seed('likelihood_SV_rows_v1',context_seed));normal_rng=np.random.default_rng(seed('likelihood_SV_shock_v1',context_seed));chi_rng=np.random.default_rng(seed('likelihood_SV_radial_v1',context_seed));vol_rng=np.random.default_rng(seed('likelihood_SV_vol_v1',context_seed))
        kind=fits.FAMILIES.index(self.config['law']);innovation=('empirical','Gaussian','Student').index(self.config['innovations'])
        for start in range(0,horizon,width):
            count=min(width,horizon-start);indices=row_rng.integers(0,len(x),size=(count,paths)).T.copy();normal=np.zeros((1,1,1));vol=np.zeros((1,1,1))
            if innovation:
                normal=normal_rng.normal(size=(count,paths,p))@f['root'].T
                if innovation==2 and f['inverse_df']>0:
                    nu=1/f['inverse_df'];normal*=np.sqrt((nu-2)/chi_rng.chisquare(nu,size=(count,paths)))[...,None]
                normal=np.ascontiguousarray(normal.transpose(1,0,2))
            if kind==5:vol=np.ascontiguousarray((vol_rng.normal(size=(count,paths,p))@f['vol_root'].T).transpose(1,0,2))
            result=kernels.advance(kind,innovation,f['pool'],indices,normal,vol,delta,raw['mean'],f['target'],parameters,f['weights'],variance,ring,f['phi'],previous,h,m,P,target,*state,start)
            if not np.isfinite(result).all():raise ArithmeticError('All declared paths must remain finite')
            self.last_asset_log_paths=state[1];self.last_asset_weights=np.exp(state[0]);yield start,result
