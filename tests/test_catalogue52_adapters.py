"""Catalogue scope and the mathematically exact singleton dependence boundary."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/catalogue52"))
from adapters import model_specs


def test_only_four_near_duplicate_timescale_variants_are_excluded():
    specs = model_specs()
    assert len(specs) == 259
    assert {row["canonical_rank"] for row in specs} == set(range(1, 264)) - {257, 258, 259, 260}
    assert len({row["model_id"] for row in specs}) == len(specs)
    assert all(row["adapter_file_available"] for row in specs)
    assert {193, 244, 256, 261, 262, 263}.issubset({row["canonical_rank"] for row in specs})


def test_singleton_bridge_preserves_original_candidates_and_multivariate_delegation(tmp_path):
    # Original experiment modules patch globals, so isolate the real initializer
    # from every other test and verify delegation before any forecasting fit.
    code = r'''
import importlib.util,json,os,sys
from pathlib import Path
import numpy as np
root=Path.cwd()
sys.path[:0]=[str(root/'src'),str(root/'tools/mcmc_runtime')]
os.environ['SIMFOLIO_CATALOGUE_ADAPTER_SOURCE']='tools/rough_jump_vine/models.py'
spec=importlib.util.spec_from_file_location('identity_bridge',root/'tools/catalogue52/single_asset_bridge.py')
bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
controls=bridge.original
sentinel=np.array([[[.314159]]])
calls=[]
controls.gaussian_uniforms=lambda *args: (calls.append(('gaussian',args)),sentinel)[1]
controls.dependence_fit=lambda *args: (calls.append(('fit',args)),{'dimension':2})[1]
controls.vine_uniforms=lambda *args: (calls.append(('vine',args)),[sentinel])[1]
bridge.initialize(None,1,1)
assert bridge.CANDIDATES is controls.CANDIDATES
draw=controls.gaussian_uniforms((80,1),b'',240,126,123)
assert np.array_equal(draw,np.random.default_rng(123).random((240,126,1)))
assert controls.gaussian_uniforms((80,2),b'x',1,1,1) is sentinel
fit=controls.dependence_fit((80,1),b'')
assert fit=={'dimension':1,'single_asset_identity_copula':True}
streams=controls.vine_uniforms(fit,240,126,123,2)
rng=np.random.default_rng(123);rng.random((240,126))
for stream in streams:
 assert np.array_equal(stream,rng.random((240*126,1)).reshape(240,126,1))
assert controls.dependence_fit((80,2),b'x')=={'dimension':2}
assert controls.vine_uniforms({'dimension':2},1,1,1)[0] is sentinel
assert [call[0] for call in calls]==['gaussian','fit','vine']
print(json.dumps({'candidate_identity_unchanged':True,'singleton_exact_uniform':True,
 'multivariate_calls_unchanged':True,'independent_streams_preserved':True}))
'''
    environment = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba"), NUMBA_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                            check=True, capture_output=True, text=True)
    assert all(json.loads(result.stdout.splitlines()[-1]).values())
