"""Broader joint-path menu built from already qualified numerical kernels."""
import hashlib,itertools,json
import numpy as np
from asset_paths import JointAssetBootstrap
from blocks import BlockLimits
from candidates.asset_feature_batch import fits
from candidates.asset_fast_selector.bank import PurePool,prepare as pure_prepare
from candidates.asset_fast_selector.candidate import BASE_CONFIG
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare

FILTERED=[{'kind':'filtered','family':family,'innovations':law,'mean_mode':mean,'mean_uncertainty':uncertainty}
          for family,law,mean,uncertainty in itertools.product(fits.FAMILIES,('Gaussian','empirical','Student'),('sample','shrink','zero'),(False,True))]
BANK=[{'kind':'parent'},{'kind':'blocks','sampling':'uniform','replacement':True},
      {'kind':'blocks','sampling':'uniform','replacement':False},{'kind':'blocks','sampling':'IID','replacement':True},*FILTERED]

def identity(spec):return 'selector_universe_component_'+hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]

class FilteredOption(PurePool):
    def __init__(self,spec,bundle):
        if spec not in FILTERED:raise ValueError('Unknown frozen component')
        self.spec=dict(spec);self.model_id=identity(spec)
        self.config={'family':spec['family'],'innovations':spec['innovations'],'raw_fade':'none','mean_shrinkage':True,'mean_uncertainty':spec['mean_uncertainty']}
        self.bundle={**bundle,'mean':{**bundle['mean']}}
        batch=self.bundle['mean']
        if spec['mean_mode']=='sample':batch['center']=batch['sample_mean'].copy()
        elif spec['mean_mode']=='zero':batch['center']=np.zeros_like(batch['sample_mean'])
    @property
    def dependency_identity(self):
        return {'method':self.model_id,'specification':self.spec,'forecast_level':'joint_assets','raw_fraction':0.,'mean_uncertainty':'Existing correlated batch Gaussian mean draws when enabled','source':'Inherited qualified PurePool path/portfolio kernel, broader validated configuration space'}

def model(index,prepared):
    spec=BANK[index]
    if spec['kind']=='parent':return TrainedPool(BASE_CONFIG,prepared['parent'])
    if spec['kind']=='blocks':return JointAssetBootstrap(replacement=spec['replacement'],limits=BlockLimits(maximum=1)if spec['sampling']=='IID'else BlockLimits())
    return FilteredOption(spec,prepared['families'][spec['family']])

def prefix_fits(history,eligible):
    prepared={'families':{}};invalid={}
    if all(BANK[i]['kind']=='blocks'for i in eligible):return prepared,invalid
    try:base=fits.base_fit(history)
    except Exception as e:return prepared,{i:{'type':type(e).__name__,'reason':str(e)}for i in eligible if BANK[i]['kind']!='blocks'}
    needed={BANK[i]['family']for i in eligible if BANK[i]['kind']=='filtered'}|({'SV_RMS'}if 0 in eligible else set())
    for family in sorted(needed):
        try:prepared['families'][family]=pure_prepare(history,family,base)
        except Exception as e:
            for i in eligible:
                if(BANK[i]['kind']=='filtered'and BANK[i]['family']==family)or(i==0 and family=='SV_RMS'):invalid[i]={'type':type(e).__name__,'reason':str(e)}
    if 0 in eligible and 0 not in invalid:
        try:prepared['parent']=parent_prepare(history,'SV_RMS','EWMA',base,prepared['families']['SV_RMS'])
        except Exception as e:invalid[0]={'type':type(e).__name__,'reason':str(e)}
    return prepared,invalid
