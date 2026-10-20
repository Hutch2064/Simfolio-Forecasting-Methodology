"""Verified smooth-mean research factory; no numerical overrides."""
from pathlib import Path
import hashlib, importlib.util, sys
BASE = Path(__file__).resolve().parents[3]
SOURCE = BASE / 'daily-cross-asset-panel-20261020/research/accuracy-maximization-20261020/mean_shrinkage/candidate.py'
EXPECTED_SOURCE = '913d27d1070e93f72e8bcd67f2b997b96a2398f08e882a50506a0bf82ce19b94'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SOURCE:
    raise RuntimeError('Smooth-mean research source drift')
NAME = 'verified_smooth_mean_Gaussian_pool_v1'
spec = importlib.util.spec_from_file_location(NAME, SOURCE)
module = importlib.util.module_from_spec(spec)
sys.modules[NAME] = module
spec.loader.exec_module(module)
def build_smooth_mean_pool():
    return module.build_candidate()
