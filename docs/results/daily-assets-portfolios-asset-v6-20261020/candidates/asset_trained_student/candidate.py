"""Learned Student innovation pool; experimental until qualification and fields."""
import hashlib,json
from pathlib import Path
from candidates.asset_feature_batch import fits
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare
from . import weight
BASE_CONFIG={'family':'SV_RMS','innovations':'Gaussian','raw_fade':'EWMA','mean_shrinkage':True,'mean_uncertainty':True}
CONFIGS=[{'innovations':'Student'}]
def identity(config):return 'model_testing_asset_trained_Student_CRPS_pool_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
def prepare(history,base=None,bundle=None):
    base=fits.base_fit(history)if base is None else base
    result=dict(parent_prepare(history,BASE_CONFIG['family'],BASE_CONFIG['raw_fade'],base,bundle));result['parent_pool_fit']=result['pool_fit']
    result['pool_fit']=weight.estimate(history,result,base,BASE_CONFIG['raw_fade'])
    return result
class TrainedStudentPool(TrainedPool):
    def __init__(self,config,bundle=None):
        if config not in CONFIGS:raise ValueError('Unknown frozen configuration')
        self.config={**BASE_CONFIG,**config};self.bundle=bundle;self.model_id=identity(config)
    @property
    def dependency_identity(self):
        result=super().dependency_identity
        result['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in Path(__file__).parent.glob('*.py')}
        result['component_weights']='Training marginal CRPS with a unit-variance Student filtered law and exact empirical leave-one-out raw law; no fixed fractions'
        return result
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        if self.bundle is None:self.bundle=prepare(history)
        yield from super().terminal_chunks(history,weights,horizon,paths,context_seed,width=width)
