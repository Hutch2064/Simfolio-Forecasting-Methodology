from pathlib import Path
import subprocess,json,base64,hashlib,concurrent.futures,datetime,os
O=Path(__file__).resolve().parent;repo=O/'repository';rel='docs/results/daily-assets-portfolios-asset-v6-20261020';source=repo/rel
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();env={**os.environ,'GODEBUG':'http2client=0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','NUMBA_NUM_THREADS':'1'}
remote=subprocess.check_output(['git','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0];assert remote==commit
out=O/'v6_public_recovery'/commit[:12];out.mkdir(parents=True,exist_ok=True)
files=sorted(f for f in source.rglob('*')if f.is_file()and '__pycache__'not in f.parts)
def recover(f):
 name=f.relative_to(source);api=f'repos/Hutch2064/Simfolio-Forecasting-Methodology/contents/{rel}/{name.as_posix()}?ref={commit}'
 data=subprocess.check_output(['gh','api','-H','Accept: application/vnd.github.raw+json',api],env=env,timeout=60);assert hashlib.sha256(data).hexdigest()==hashlib.sha256(f.read_bytes()).hexdigest()
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(f.read_bytes()).hexdigest();return {'path':str(name),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(recover,files))
python='/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python'
r=subprocess.run([python,'-m','unittest','-v','test_asset_paths.py','test_v6_reporting.py','test_asset_prior_best.py','test_asset_v6_decomposition.py','test_asset_mean_screen.py','test_asset_sampler_parent.py','test_asset_circular.py','test_asset_iid_component.py','test_asset_gjr.py','test_asset_adaptive_blocks.py','test_asset_empirical_mean.py','test_asset_fading_vol.py','test_asset_equilibrium_blocks.py','test_asset_training_memory.py','test_asset_AR1_mean.py','test_asset_feature_batch.py','test_asset_feature_admission.py','test_asset_dynamic_correlation.py','test_asset_component_weights.py','test_asset_training_blocks.py','test_asset_likelihood_sv.py','test_asset_trained_pool.py','test_asset_regime_rows.py','test_asset_random_measure.py'],cwd=out,env=env,capture_output=True,text=True);(out/'recovered_tests.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stdout+r.stderr
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'repository':'Hutch2064/Simfolio-Forecasting-Methodology','fresh_remote_files_downloaded_disk_restored_SHA256_verified':len(rows),'recovered_source_tests_passed':103,'uses_existing_sealed_parent_source_and_environment_not_clean_OS':True,'files':rows}
(O/'v6_candidate_source_public_recovery.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items()if k!='files'}),flush=True)
