"""Execute aligned Gaussian/t arms with a shared, identical asset-fit disk cache."""
import importlib.util
import sys
from dataclasses import dataclass, replace
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
branches={}
for family,directory in [('gaussian','innovation_shrinkage_copula_rough'),('student','innovation_student_copula_rough')]:
    spec=importlib.util.spec_from_file_location('aligned_suite_'+family,ROOT/'tools'/directory/'models.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    branches[family]=module
assert branches['gaussian'].parent.body.SOURCE==branches['student'].parent.parent.body.SOURCE
FIT_DIAGNOSTICS={};TIMINGS={}


@dataclass(frozen=True)
class Candidate:
    model_id:str
    family:str
    fit_seconds:float=10.

    def simulate_daily_log_returns(self,training,context):
        branch=branches[self.family]
        branch.TIMINGS.clear();branch.FIT_DIAGNOSTICS.clear()
        child=replace(branch.CANDIDATES[0],fit_seconds=self.fit_seconds)
        result=child.simulate_daily_log_returns(training,context)
        TIMINGS.update(branch.TIMINGS);FIT_DIAGNOSTICS.update(branch.FIT_DIAGNOSTICS)
        return result


CANDIDATES=tuple(Candidate(branch.CANDIDATES[0].model_id,family) for family,branch in branches.items())
MODEL_IDS=(branches['gaussian'].MODEL_IDS[0],*(c.model_id for c in CANDIDATES))
shell=branches['gaussian'].shell;overlay=branches['gaussian'].overlay;load_paths=branches['gaussian'].load_paths


def select_copula(enabled):
    for branch in branches.values():branch.select_copula(enabled)


def clear_path_cache():
    for branch in branches.values():branch.clear_path_cache()


def initialize(cache_root=None,vine_threads=1,state_workers=1):
    for branch in branches.values():branch.initialize(cache_root,vine_threads,state_workers)
