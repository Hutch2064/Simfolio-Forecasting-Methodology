"""An empirical-Bayes mean contrast around the existing joint-asset law."""
import hashlib,json
import numpy as np
from candidates.asset_feature_batch import fits
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare
from . import mean

BASE_CONFIG={'family':'SV_RMS','innovations':'Gaussian','raw_fade':'EWMA','mean_shrinkage':True,'mean_uncertainty':True}
CONFIGS=[{'mean':'empirical_Bayes_center'},{'mean':'empirical_Bayes_posterior'}]


def identity(config):
    return 'model_testing_asset_joint_EB_mean_v1_'+hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]


def prepare(history,base=None):
    base=fits.base_fit(history)if base is None else base
    parent=parent_prepare(history,BASE_CONFIG['family'],BASE_CONFIG['raw_fade'],base)
    bundles={'parent':parent}
    for config in CONFIGS:
        changed=dict(parent)
        changed['mean']=mean.estimate(parent['mean'],config['mean']=='empirical_Bayes_posterior')
        # The changed point mean enters training CRPS; refit that weight honestly.
        bundles[config['mean']]=parent_prepare(history,BASE_CONFIG['family'],BASE_CONFIG['raw_fade'],base,changed)
    return bundles


class EmpiricalBayesPool:
    def __init__(self,config,bundle=None):
        if config not in CONFIGS+[{'mean':'parent'}]:raise ValueError('Unknown frozen mean law')
        self.config=dict(config);self.bundle=bundle;self.model_id=identity(config)

    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        bundle=self.bundle if self.bundle is not None else prepare(history)
        chosen=bundle[self.config['mean']]
        assert chosen['history_sha256']==hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest()
        model=TrainedPool(BASE_CONFIG,chosen)
        for start,result in model.terminal_chunks(history,weights,horizon,paths,context_seed,width=width):
            self.last_component_counts=model.last_component_counts
            self.last_asset_log_paths=model.last_asset_log_paths
            self.last_asset_weights=model.last_asset_weights
            yield start,result
