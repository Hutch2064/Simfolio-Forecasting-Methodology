"""Admission and native checkpoint conversion contracts without new field forecasts."""
from pathlib import Path
import os,sys,json,hashlib,tempfile,unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6' if (O/'asset_level_v6').exists() else O;sys.path.insert(0,str(V))
import common,sweep_v6 as api
from candidates.asset_feature_batch.candidate import FeaturePool,CONFIGS,identity

class AdmissionTests(unittest.TestCase):
    def test_frozen_factory_registry_accepts_complete_transitive_fit_closure(self):
        C=V/'candidates/asset_feature_batch';plan=json.loads((C/'broad1000/plan.json').read_text());extra=json.loads((C/'transitive_source_receipt.json').read_text())
        files={k[len('asset_level_v6/'):]:v for k,v in plan['source_hashes'].items()if k.startswith('asset_level_v6/')};files.update(extra['source_files_relative_to_v6'])
        self.assertIn('candidates/asset_AR1_mean/fit_ar.py',files);self.assertIn('candidates/asset_gjr/fit.py',files)
        for name,value in files.items():self.assertEqual(hashlib.sha256((V/name).read_bytes()).hexdigest(),value)
        config=next(c for c in CONFIGS if c['family']=='SV_RMS'and c['innovations']=='Gaussian'and all(c[k]for k in ('mean_shrinkage','mean_uncertainty','raw_fade')))
        mid=identity(config);factory=f'candidates.asset_feature_batch.candidate:build_{CONFIGS.index(config):03d}'
        declaration={'model_id':mid,'factory':factory,'source_files':files,'numerical_contract':api.original.dependency_identity(FeaturePool(config))};digest=api.original.stable_digest(declaration);spec={'factory':factory,'implementation_digest':digest,'enabled':True}
        with tempfile.TemporaryDirectory()as folder:
            registry=Path(folder)/'registry.json';registry.write_text(json.dumps({'schema':'french-external-model-registry-v1','source_root':str(V),'models':{mid:{'identity':declaration,'factory':factory,'implementation_digest':digest}}}))
            actual=api.admitted_contract(mid,spec,registry);self.assertEqual(actual,declaration['numerical_contract'])

    def test_native_import_preserves_loss_bytes_and_explicit_pilot_timing_scope(self):
        loss=np.random.default_rng(34).uniform(size=257);identity={'model_id':'test_conversion','execution_fingerprint':'frozen','seed':918,'origin':{'max_horizon':257}}
        diagnostics={'simulations':1000,'imported_verified_pilot':True,'forecast_and_score_seconds':.123,'timing_scope':'Pilot simulation/CRPS plus attributable fit; excludes shared input preparation; not controlled speed comparison','joint_asset_paths_simulated':True}
        with tempfile.TemporaryDirectory()as folder:
            p=Path(folder)/'checkpoint.npz';api.original._write_checkpoint(p,identity,loss,diagnostics);actual,metadata=api.original._read_checkpoint(p,identity);np.testing.assert_array_equal(actual,loss);self.assertEqual(metadata,diagnostics)
if __name__=='__main__':unittest.main()
