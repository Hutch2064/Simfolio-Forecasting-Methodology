"""Single-law joint return simulation from a random empirical probability measure."""
from pathlib import Path
import hashlib,itertools,json
import numpy as np
from candidates.asset_prior_best import fit
from candidates.asset_fading_vol import fade
from candidates.asset_fading_vol.kernels import advance_raw
from . import measure

CONFIGS=[{'raw_fade':fade_kind,'concentration':concentration}for fade_kind,concentration in itertools.product(('none','GARCH','EWMA'),('Rubin','training_matched'))]
def identity(config):return 'model_testing_asset_random_joint_measure_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def seed(name,value):return int.from_bytes(hashlib.sha256(f'{name}:{value}'.encode()).digest()[:8],'little')


def prepare(history,base_fits=None):
    x=np.asarray(history,dtype=float);f=measure.prepare(x);p=x.shape[1]
    fitted=[fit.fit_variance_target(x[:,a])for a in range(p)]if base_fits is None else base_fits
    f['raw']={k:np.array([row[k]for row in fitted])for k in('mean','target_variance','alpha','beta','next_variance_normalized')}
    f['ewma']=[fade.fit_ewma(x[:,a])for a in range(p)];f['history_sha256']=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest();return f


class RandomMeasure:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen random-measure configuration')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'forecast_level':'joint_assets','config':self.config,'fixed_component_mixture':False,'mean_uncertainty':'Induced by one joint random empirical law per scenario; no extra Gaussian drift offset','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}}
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);n,p=x.shape
        if n<24 or not np.isfinite(x).all()or p!=len(w)or np.any(w<0)or not np.isclose(w.sum(),1,rtol=0,atol=1e-14)or paths<2 or horizon<1:raise ValueError('Valid joint matrix and positive ensemble required')
        f=self.bundle if self.bundle is not None else prepare(x);assert f['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
        concentration=float(n)if self.config['concentration']=='Rubin'else f['matched_concentration'];cov=f['block_mean_covariance']/(concentration+1);mu=x.mean(0);v=np.diag(cov);denominator=mu*mu+v
        shrink=np.divide(mu*mu,denominator,out=np.ones_like(mu),where=denominator>0);delta=np.tile(mu*shrink-mu,(paths,1))
        stream=measure.RandomRows(n,paths,seed('random_measure_joint_rows_v1',context_seed),f['selected_length'],concentration)
        target=np.full(p,-np.inf);target[w>0]=np.log(w[w>0]);state=(np.tile(target,(paths,1)),np.zeros((paths,p)),np.zeros(paths));raw=f['raw']
        for first in range(0,horizon,width):
            count=min(width,horizon-first)
            if self.config['raw_fade']=='GARCH':initial,decay=raw['next_variance_normalized'],raw['alpha']+raw['beta']
            elif self.config['raw_fade']=='EWMA':initial=np.array([v['initial_variance_ratio']for v in f['ewma']]);decay=np.array([v['decay']for v in f['ewma']])
            else:initial,decay=np.ones(p),np.ones(p)
            result=advance_raw(x,stream.take(count),delta,raw['mean'],fade.scale_table(initial,decay,first,count),target,*state)
            if not np.isfinite(result).all():raise ArithmeticError('All random-law paths must remain finite')
            self.last_asset_log_paths=state[1];self.last_asset_weights=np.exp(state[0]);self.last_concentration=concentration;yield first,result
