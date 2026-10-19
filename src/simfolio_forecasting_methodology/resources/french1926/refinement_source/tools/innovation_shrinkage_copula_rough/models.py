"""M248 with its return copula fitted to its existing standardized innovations."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import ndtri

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from innovation_alignment import aligned_pools

spec=importlib.util.spec_from_file_location('innovation_copula_private_M248',ROOT/'tools/static_shrinkage_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
original_configuration=parent.configuration
ENABLED=True


@lru_cache(maxsize=64)
def aligned_configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    pools=aligned_pools(assets,parent.body.refit)
    uniforms=parent.body.shell.controls.dg.pseudo_observations(pools)
    correlation,root,shrinkage=parent.shrunk_correlation(ndtri(uniforms))
    return correlation,root,shrinkage,hashlib.sha256(pools.tobytes()).hexdigest()


def configuration(shape,data):
    return aligned_configuration(shape,data)[:3]


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:parent.gaussian_uniforms.cache_clear()
    ENABLED=enabled
    parent.configuration=configuration if enabled else original_configuration
    parent.select_copula(True)


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        digest=aligned_configuration(assets.shape,assets.tobytes())[3] if ENABLED else None
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['innovation_aligned_copula_enabled']=ENABLED
            record['copula_training_input']='unchanged_chronological_Student_standardized_innovation_pool' if ENABLED else 'raw_asset_returns'
            record['copula_innovation_matrix_sha256']=digest
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_shrinkage_copula_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():parent.clear_path_cache()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
