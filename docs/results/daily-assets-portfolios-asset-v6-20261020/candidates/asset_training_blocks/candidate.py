"""Single-law asset simulations with training-selected joint row dependence."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from candidates.asset_prior_best import mean
from candidates.asset_fading_vol import fade
from candidates.asset_fading_vol.kernels import advance_raw
from candidates.asset_feature_batch import fits,kernels
from .dependence import select_length,mean_fit,StationaryRows

LAWS=('raw','raw_GARCH_fade','raw_EWMA_fade','GARCH','EWMA_reverting','SV_RMS','SV_path')
CONFIGS=[{'law':law,'mean_covariance':uncertainty}for law,uncertainty in itertools.product(LAWS,('batch','stationary'))]
def identity(config):return 'model_testing_asset_training_blocks_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def seed(name,value):return int.from_bytes(hashlib.sha256(f'{name}:{value}'.encode()).digest()[:8],'little')
def family(law):return 'constant'if law.startswith('raw')else law


def prepare(history,law,bundle):
    x=np.asarray(history,dtype=float)
    assert bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    assert bundle['family']==family(law)
    pool=x if law.startswith('raw')else bundle['filter']['pool']
    dependence=select_length(pool)
    result={'bundle':bundle,'dependence':dependence,'stationary_mean':mean_fit(x,dependence['length'])}
    if law=='raw_EWMA_fade':
        fitted=[fade.fit_ewma(x[:,a])for a in range(x.shape[1])]
        result['ewma_initial']=np.array([f['initial_variance_ratio']for f in fitted]);result['ewma_decay']=np.array([f['decay']for f in fitted])
    return result


class TrainingBlocks:
    def __init__(self,config,prepared=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen configuration')
        self.config=dict(config);self.prepared=prepared;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'fixed_component_mixture':False,
                'fitting':'Training prefix only; no case routing or held-out fitted parameters',
                'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);law=self.config['law']
        if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all()or x.shape[1]!=len(w)or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or paths<2 or paths%2 or horizon<1:
            raise ValueError('Valid joint history, weights, horizon and even ensemble required')
        prepared=self.prepared if self.prepared is not None else prepare(x,law,fits.prepare(x,family(law)))
        bundle=prepared['bundle'];assert bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();assert bundle['family']==family(law)
        raw,f=bundle['raw'],bundle['filter'];batch=bundle['mean']if self.config['mean_covariance']=='batch'else prepared['stationary_mean']
        delta=mean.draws(batch,paths,seed('training_blocks_mean_v1',context_seed));assets=x.shape[1]
        stream=StationaryRows(len(x),paths,seed('training_blocks_rows_v1',context_seed),prepared['dependence']['length'])
        target=np.full(assets,-np.inf);target[w>0]=np.log(w[w>0]);state=(np.tile(target,(paths,1)),np.zeros((paths,assets)),np.zeros(paths))
        if not law.startswith('raw'):
            parameters=np.stack([f[k]for k in('alpha','beta','positive','negative','omega','initial','decay','level','sv_phi','q')])
            variance=np.tile(f['initial'],(paths,1));ring=np.tile(f['ring'],(paths,1,1));previous=np.tile(x[-1],(paths,1))
            next_m=f['level']+f['sv_phi']*(f['m']-f['level']);next_P=f['sv_phi']**2*f['P']+f['q']
            m=np.tile(next_m,(paths,1));P=np.tile(next_P,(paths,1));rng=np.random.default_rng(seed('training_blocks_vol_initial_v1',context_seed))
            h=m+(rng.normal(size=(paths,assets))@f['vol_root'].T)*np.sqrt(next_P)
            vol_rng=np.random.default_rng(seed('training_blocks_vol_v1',context_seed));kind=fits.FAMILIES.index(law)
        for start in range(0,horizon,width):
            count=min(width,horizon-start);indices=stream.take(count)
            if law.startswith('raw'):
                if law=='raw_GARCH_fade':initial,decay=raw['next_variance_normalized'],raw['alpha']+raw['beta']
                elif law=='raw_EWMA_fade':initial,decay=prepared['ewma_initial'],prepared['ewma_decay']
                else:initial,decay=np.ones(assets),np.ones(assets)
                result=advance_raw(x,indices,delta,raw['mean'],fade.scale_table(initial,decay,start,count),target,*state)
            else:
                vol=np.zeros((1,1,1))
                if kind==5:vol=np.ascontiguousarray((vol_rng.normal(size=(count,paths,assets))@f['vol_root'].T).transpose(1,0,2))
                result=kernels.advance(kind,0,f['pool'],indices,np.zeros((1,1,1)),vol,delta,raw['mean'],f['target'],parameters,f['weights'],variance,ring,f['phi'],previous,h,m,P,target,*state,start)
                if not np.isfinite(variance).all():raise ArithmeticError('Finite filter variance required')
            if not np.isfinite(result).all():raise ArithmeticError('All paths must remain finite; no exclusions')
            self.last_asset_log_paths=state[1];self.last_asset_weights=np.exp(state[0]);self.last_selected_length=prepared['dependence']['length']
            yield start,result
