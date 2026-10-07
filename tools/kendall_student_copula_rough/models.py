"""M251 asset forecasts with Kendall method-of-moments Student copula scatter."""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import stdtr, stdtrit

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from scatter import kendall_correlation

spec=importlib.util.spec_from_file_location('kendall_private_M251',ROOT/'tools/innovation_student_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
original_configuration=parent.aligned_configuration
ENABLED=True


def informative_columns(uniforms):
    n=uniforms.shape[0];selected=[];seen=set();constant=[];duplicate=[]
    # Average ranks have exact half-integer values. Integer rank keys avoid
    # roundoff when recognizing reflected copies of the same history.
    ranks=np.rint(2*(n+1)*uniforms).astype(np.int64)
    for a,column in enumerate(ranks.T):
        if np.ptp(column)==0:constant.append(a);continue
        key=column.tobytes()
        if key in seen:duplicate.append(a);continue
        selected.append(a);seen.add(key);seen.add((2*(n+1)-column).tobytes())
    return selected,constant,duplicate


def fit_informative_degrees(uniforms,correlation):
    active,constant,duplicate=informative_columns(uniforms)
    if active:
        u=uniforms[:,active];c=correlation[np.ix_(active,active)]
        nu,diagnostics=parent.parent.fit_degrees_of_freedom(u,c)
        error=0. if np.isinf(nu) else float(np.max(np.abs(stdtr(nu,stdtrit(nu,u))-u)))
        if not np.isfinite(error) or error>np.sqrt(np.finfo(float).eps):
            raise ArithmeticError('unresolved Student copula quantile accuracy; no df floor substituted')
    else:
        nu=np.inf;error=0.
        diagnostics={'gaussian_log_likelihood':0.,'log_likelihood':0.,'evaluations':1}
    diagnostics.update(informative_dimensions=len(active),excluded_constant_columns=constant,
        excluded_duplicate_or_reflected_columns=duplicate,maximum_quantile_roundtrip_error=error)
    return nu,diagnostics


@lru_cache(maxsize=64)
def configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    pools=parent.aligned_pools(assets,parent.parent.parent.body.refit)
    uniforms=parent.parent.parent.body.shell.controls.dg.pseudo_observations(pools)
    c,root,repair=kendall_correlation(pools)
    nu,diagnostics=fit_informative_degrees(uniforms,c)
    return c,root,0.,nu,diagnostics,hashlib.sha256(pools.tobytes()).hexdigest(),repair


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:
        parent.parent.student_uniforms.cache_clear();parent.parent.parent.gaussian_uniforms.cache_clear()
    ENABLED=enabled
    parent.aligned_configuration=configuration if enabled else original_configuration
    parent.select_copula(True)


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        repair=configuration(assets.shape,assets.tobytes())[6] if ENABLED else None
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['kendall_scatter_enabled']=ENABLED
            record['return_copula_scatter_estimator']='Kendall_sign_U_statistic' if ENABLED else 'M251_Gaussian_rank_Ledoit_Wolf'
            record['return_copula_scatter_repair']=repair
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_kendall_student_copula_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():parent.clear_path_cache()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
