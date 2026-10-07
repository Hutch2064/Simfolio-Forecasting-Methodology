"""Unchanged M251 asset law with a training-selected static regular vine.

Sequential pairwise MLE and BIC family selection with maximum-tau spanning
trees, as in Dissmann et al. (2013). This is a simplified vine (conditional
copula parameters do not vary with conditioning values), not an HMM or a
globally optimized joint likelihood. Library parameter bounds are recorded.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvinecopulib as pv
from scipy.special import ndtr

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('static_vine_private_M251',ROOT/'tools/innovation_student_copula_rough/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
body=parent.parent.parent.body
PARAMETRIC_FAMILIES=tuple(f for f in pv.FitControlsVinecop().family_set if f!=pv.BicopFamily.tll)
ENABLED=True


@lru_cache(maxsize=64)
def configuration(shape,data):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    pools=parent.aligned_pools(assets,body.refit)
    uniforms=body.shell.controls.dg.pseudo_observations(pools)
    controls=pv.FitControlsVinecop(family_set=PARAMETRIC_FAMILIES,parametric_method='mle',
        selection_criterion='bic',preselect_families=False,allow_rotations=True,
        select_trunc_lvl=False,select_threshold=False,tree_criterion='tau',num_threads=1)
    vine=pv.Vinecop.from_data(np.asfortranarray(uniforms),controls=controls)
    diagnostics={'method':'sequential_pair_MLE_BIC_static_regular_vine','library_version':pv.__version__,
        'family_set':[f.name for f in PARAMETRIC_FAMILIES],'preselect_families':False,
        'simplifying_assumption':True,'tree_criterion':'tau','truncation':'full_dimension_minus_one',
        'threshold':0.,'degrees_of_freedom_bound_source':'native_library_student_2_to_50',
        'selected_families':[[f.name for f in tree] for tree in vine.families],
        'parameters':[[p.tolist() for p in tree] for tree in vine.parameters],
        'rotations':vine.rotations,'structure':vine.matrix.tolist(),
        'fitted_parameter_count':float(vine.npars),'log_likelihood':float(vine.loglik(uniforms)),
        'bic':float(vine.bic(uniforms)),
        'library_parameter_bounds':{f.name:{'lower':pv.Bicop(f).parameters_lower_bounds.tolist(),
            'upper':pv.Bicop(f).parameters_upper_bounds.tolist()} for f in PARAMETRIC_FAMILIES}}
    return vine,hashlib.sha256(pools.tobytes()).hexdigest(),diagnostics


@lru_cache(maxsize=2)
def vine_uniforms(shape,data,sims,horizon,seed):
    vine,_,_=configuration(shape,data)
    rng=np.random.default_rng(seed);result=np.empty((sims,horizon,shape[1]))
    for start in range(0,horizon,1024):
        stop=min(start+1024,horizon)
        u=ndtr(rng.normal(size=(stop-start,sims,shape[1])))
        mapped=vine.inverse_rosenblatt(np.asfortranarray(u.reshape(-1,shape[1])),num_threads=1)
        if not np.isfinite(mapped).all() or np.any((mapped<0)|(mapped>1)):
            raise ArithmeticError('invalid inverse Rosenblatt copula output')
        np.clip(mapped,1e-8,1-1e-8,out=mapped)
        result[:,start:stop]=mapped.reshape(stop-start,sims,shape[1]).transpose(1,0,2)
    return result


def select_copula(enabled):
    global ENABLED
    if enabled!=ENABLED:clear_path_cache()
    ENABLED=enabled
    parent.select_copula(True)
    shell.controls.gaussian_uniforms=vine_uniforms if enabled else parent.parent.student_uniforms


class Candidate(parent.parent.parent.parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        result=super().simulate_daily_log_returns(training,context)
        assets=np.asarray(training.asset_log_returns,float)
        _,digest,diagnostics=configuration(assets.shape,assets.tobytes()) if ENABLED else (None,None,None)
        for x in assets.T:
            identity=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
            record=FIT_DIAGNOSTICS[self.model_id,identity]
            record['static_innovation_vine_enabled']=ENABLED
            record['copula_training_input']='unchanged_chronological_Student_standardized_innovation_pool'
            record['copula_innovation_matrix_sha256']=digest
            record['return_copula_vine_fit']=diagnostics
        return result


CANDIDATES=(Candidate(model_id='asset_map_stationary_ar1_innovation_static_vine_dynamic_rough'),)
MODEL_IDS=(parent.CANDIDATES[0].model_id,CANDIDATES[0].model_id)
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();vine_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
