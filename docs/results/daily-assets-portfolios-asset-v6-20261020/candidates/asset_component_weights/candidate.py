"""Frozen factorial hypotheses around the verified joint-asset hybrid."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from blocks import BlockLimits
from candidates.asset_circular_blocks.circular import CircularBlockStream
from candidates.asset_prior_best import mean
from candidates.asset_fading_vol import fade
from candidates.asset_fading_vol.kernels import advance_raw
from candidates.asset_feature_batch import fits,kernels

CONFIGS=[{'family':family,'innovations':'Gaussian','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True,'raw_fraction':fraction}
         for family,fraction in itertools.product(('SV_RMS','SV_path','constant'),(0.,.25,.5,.75))]
CONFIGS.append({'family':'SV_RMS','innovations':'Gaussian','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True,'raw_fraction':1.})
def identity(config):return 'model_testing_asset_component_weight_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def seed(namespace,value):return int.from_bytes(hashlib.sha256(f'{namespace}:{value}'.encode()).digest()[:8],'little')

class ComponentPool:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen configuration')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'component_weights':[self.config['raw_fraction'],1-self.config['raw_fraction']],
                'point_parameters':'Training prefix only; no held-out calibration, ticker/horizon routing or fitted mixture weights',
                'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},'filtered_parent_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in (Path(__file__).parent.parent/'asset_feature_batch').glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
        if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all() or x.shape[1]!=len(w) or np.any(w<0) or not np.isclose(w.sum(),1,rtol=0,atol=1e-14) or paths<4 or paths%4:raise ValueError('Valid joint history and even component ensembles required')
        config=self.config;bundle=self.bundle if self.bundle is not None else fits.prepare(x,config['family'])
        assert bundle['family']==config['family']
        assert bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        raw_paths=int(paths*config['raw_fraction']);half=paths-raw_paths;assets=x.shape[1]
        if raw_paths%2 or half%2:raise ValueError('Both component counts require even antithetic mean ensembles')
        raw,f,batch=bundle['raw'],bundle['filter'],bundle['mean']
        bias=batch['center']-batch['sample_mean'];delta=[]
        for name,number in zip(('raw_joint_mean_uncertainty_v1','FHS_joint_mean_uncertainty_v1'),(raw_paths,half)):
            if number==0:values=np.empty((0,assets))
            elif config['mean_uncertainty']:
                values=mean.draws(batch,number,seed(name,context_seed))
                if not config['mean_shrinkage']:values=values-bias
            else:values=np.tile(bias if config['mean_shrinkage'] else np.zeros(assets),(number,1))
            delta.append(values)
        raw_stream=CircularBlockStream(len(x),horizon,raw_paths,context_seed,replacement=True,limits=BlockLimits())if raw_paths else None
        row_rng=np.random.default_rng(seed('joint_FHS_asset_rows_v1',context_seed))
        normal_rng=np.random.default_rng(seed('joint_feature_Gaussian_v1',context_seed));chi_rng=np.random.default_rng(seed('joint_feature_radial_v1',context_seed));vol_rng=np.random.default_rng(seed('joint_feature_vol_v1',context_seed))
        target=np.full(assets,-np.inf);target[w>0]=np.log(w[w>0]);states=[(np.tile(target,(half,1)),np.zeros((half,assets)),np.zeros(half)) for half in (raw_paths,half)]
        parameters=np.stack([f[k] for k in ('alpha','beta','positive','negative','omega','initial','decay','level','sv_phi','q')])
        if config['family']=='GJR' and config['innovations']!='empirical':
            persistence=parameters[1]+.5*(parameters[2]+parameters[3])
            factor=np.minimum(1.,(1.-1/len(x))/np.maximum(persistence,np.finfo(float).tiny))
            parameters[1]*=factor;parameters[2]*=factor;parameters[3]*=factor
            parameters[4]=1.-parameters[1]-.5*(parameters[2]+parameters[3])
        variance=np.tile(f['initial'],(half,1));ring=np.tile(f['ring'],(half,1,1));previous=np.tile(x[-1],(half,1))
        next_m=f['level']+f['sv_phi']*(f['m']-f['level']);next_P=f['sv_phi']**2*f['P']+f['q']
        m=np.tile(next_m,(half,1));P=np.tile(next_P,(half,1))
        initial_rng=np.random.default_rng(seed('joint_feature_vol_initial_v1',context_seed));h=m+(initial_rng.normal(size=(half,assets))@f['vol_root'].T)*np.sqrt(next_P)
        kind=fits.FAMILIES.index(config['family']);innovation=('empirical','Gaussian','Student').index(config['innovations'])
        for start in range(0,horizon,width):
            count=min(width,horizon-start);indices=row_rng.integers(0,len(x),size=(count,half)).T.copy()
            normal=np.zeros((1,1,1));vol=np.zeros((1,1,1))
            if innovation:
                normal=normal_rng.normal(size=(count,half,assets))@f['root'].T
                if innovation==2 and f['inverse_df']>0:
                    nu=1/f['inverse_df'];normal*=np.sqrt((nu-2)/chi_rng.chisquare(nu,size=(count,half)))[...,None]
                normal=np.ascontiguousarray(normal.transpose(1,0,2))
            if kind==5:vol=np.ascontiguousarray((vol_rng.normal(size=(count,half,assets))@f['vol_root'].T).transpose(1,0,2))
            scales=fade.scale_table(raw['next_variance_normalized'],raw['alpha']+raw['beta'],start,count) if config['raw_fade'] else np.ones((assets,count))
            first=advance_raw(x,raw_stream.take(count)if raw_stream is not None else np.empty((0,count),dtype=np.int64),delta[0],raw['mean'],scales,target,*states[0])
            second=kernels.advance(kind,innovation,f['pool'],indices,normal,vol,delta[1],raw['mean'],f['target'],parameters,f['weights'],variance,ring,f['phi'],previous,h,m,P,target,*states[1],start)
            result=np.concatenate((first,second))
            if not np.isfinite(result).all() or not np.isfinite(variance).all():raise ArithmeticError('All paths must remain finite; no path exclusions')
            self.last_component_counts=(raw_paths,half)
            self.last_variance=variance.copy();self.last_filter_parameters=parameters.copy()
            self.last_asset_log_paths=np.concatenate([state[1] for state in states]);self.last_asset_weights=np.exp(np.concatenate([state[0] for state in states]))
            yield start,result

for index,config in enumerate(CONFIGS):
    def factory(config=config):return ComponentPool(config)
    globals()[f'build_{index:03d}']=factory
