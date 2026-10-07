"""M250/M251 fits and path laws, with stratified return-copula primitives."""
import importlib.util
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from sampling import copula_uniforms

spec=importlib.util.spec_from_file_location('stratified_private_suite',ROOT/'tools/innovation_copula_suite/models.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
branches=parent.branches
ENABLED=True
FIT_DIAGNOSTICS=parent.FIT_DIAGNOSTICS;TIMINGS=parent.TIMINGS


@lru_cache(maxsize=2)
def gaussian_uniforms(shape,data,sims,horizon,seed):
    _,root,_=branches['gaussian'].parent.configuration(shape,data)
    return copula_uniforms(root,float('inf'),sims,horizon,seed)


@lru_cache(maxsize=2)
def student_uniforms(shape,data,sims,horizon,seed):
    _,root,_,nu,_=branches['student'].parent.configuration(shape,data)
    return copula_uniforms(root,nu,sims,horizon,seed)


def select_copula(enabled):
    global ENABLED
    ENABLED=enabled
    parent.select_copula(True)
    parent.clear_path_cache();gaussian_uniforms.cache_clear();student_uniforms.cache_clear()
    if enabled:
        branches['gaussian'].shell.controls.gaussian_uniforms=gaussian_uniforms
        branches['student'].shell.controls.gaussian_uniforms=student_uniforms


@dataclass(frozen=True)
class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self,training,context):
        branch=branches[self.family]
        branch.TIMINGS.clear();branch.FIT_DIAGNOSTICS.clear()
        child=replace(branch.CANDIDATES[0],model_id=self.model_id,fit_seconds=self.fit_seconds)
        result=child.simulate_daily_log_returns(training,context)
        TIMINGS.update(branch.TIMINGS);FIT_DIAGNOSTICS.update(branch.FIT_DIAGNOSTICS)
        for record in FIT_DIAGNOSTICS.values():
            record['return_shock_sampling']='randomized_Latin_hypercube' if ENABLED else 'IID'
            record['strata_per_primitive']=context.simulations if ENABLED else None
            record['cross_path_independence']=not ENABLED
        return result


CANDIDATES=tuple(Candidate(model_id=branch.CANDIDATES[0].model_id.replace('_copula_','_lhs_copula_'),family=family)
    for family,branch in branches.items())
MODEL_IDS=(parent.MODEL_IDS[0],*(c.model_id for c in CANDIDATES))
shell=parent.shell;overlay=parent.overlay;load_paths=parent.load_paths


def clear_path_cache():
    parent.clear_path_cache();gaussian_uniforms.cache_clear();student_uniforms.cache_clear()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    parent.initialize(cache_root,vine_threads,state_workers)
    select_copula(ENABLED)
