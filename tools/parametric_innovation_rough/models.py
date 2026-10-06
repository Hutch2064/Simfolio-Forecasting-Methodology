"""M212 with learned standardized Student or Hansen skew-t return shocks.

Mean curves, both volatility fits, rough kernel and dependence remain unchanged.
"""
from dataclasses import dataclass
from functools import lru_cache,partial
import hashlib,importlib.util,time
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import density
spec=importlib.util.spec_from_file_location('parametric_innovation_private_m212',ROOT/'tools/consistent_noise_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
streamed=parent.parent.streamed
SOURCE=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(HERE.glob('*.py')))).hexdigest()

@lru_cache(maxsize=64)
def shock_fit(data,kind):
    key=hashlib.sha256(data+(SOURCE+kind).encode()).hexdigest()
    started=time.perf_counter()
    result=shell.controls.cache('parametric_filtered_innovation_fit',key,lambda:density.fit(streamed.predecessor_fit(data)['innovation_pool'],kind))
    shell.controls.timed('innovation_fit_worker',started)
    return result

@lru_cache(maxsize=128)
def nodes(data,kind,simulations):
    fit=shock_fit(data,kind)
    p=np.linspace(.5/simulations,1-.5/simulations,simulations)
    result=density.ppf(p,fit['inverse_df'],fit['skew'])
    result-=result.mean();result/=np.sqrt(np.mean(result*result))
    return result

def asset_paths(data,uniforms,*,kind):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves
    from streamed_paths import map_asset_inplace
    mean,sd=moment_return_curves(streamed.predecessor_fit(data),uniforms.shape[1])
    paths=uniforms.copy();map_asset_inplace(paths,mean,sd,nodes(data,kind,len(paths)))
    return mean,paths

@dataclass(frozen=True)
class Candidate(parent.Candidate):
    kind:str='student'
    def simulate_daily_log_returns(self,training,context):
        streamed.predecessor_asset_paths=partial(asset_paths,kind=self.kind)
        result=super().simulate_daily_log_returns(training,context)
        for key,record in FIT_DIAGNOSTICS.items():
            if key[0]==self.model_id:
                data=np.asarray(training.asset_log_returns,float)
                index=next(a for a in range(data.shape[1]) if hashlib.sha256(data[:,a].tobytes()).hexdigest()==key[1])
                record['innovation_distribution']=self.kind
                record['innovation_fit']=shock_fit(data[:,index].tobytes(),self.kind)
        return result

initialize=parent.initialize
CANDIDATES=(Candidate(model_id='asset_map_multiscale_consistent_noise_student_innovations_rough_map'),Candidate(model_id='asset_map_multiscale_consistent_noise_hansen_skew_t_rough_map',kind='skew_student'))
MODEL_IDS=(parent.MODEL_IDS[0],)+tuple(c.model_id for c in CANDIDATES)
clear_path_cache=parent.clear_path_cache
TIMINGS=parent.TIMINGS
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS
shell=parent.shell
overlay=parent.overlay
load_paths=parent.load_paths
