"""Read-only projection from completed identical-kernel1kprecision workload."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_path_objective';D=C/'development1000';W=C/'screen1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=read(D/'plan.json');oldplan=read(W/'plan.json');summary=read(W/'summary.corrected.json');audit=read(W/'verification_receipt.json');status=read(D/'progress.json');active=read(D/'active.json')
assert audit['verified']and audit['summary_sha256']==sha(W/'summary.json')and summary['paths']==summary['inner_paths']==plan['paths']==1000
assert all(plan['source_hashes'][p]==h and sha(O/p)==h for p,h in oldplan['source_hashes'].items())
samples=[r for r in summary['case_timings']if not r.get('warmup_first_request_for_pid',True)and r.get('work_units',0)>0 and r.get('cpu_seconds',0)>0];assert samples
rates=np.asarray([r['cpu_seconds']/r['work_units']for r in samples]);rows={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};completed={read(p)['case']for p in(D/'case_records').glob('*.json')};remaining={}
for i,case in enumerate(plan['cases']):
 if i in completed:continue
 row=rows[case['series_id']];assets=len(row['rational_weights'])if row['panel_branch']=='portfolios'else 1;remaining[str(i)]=1000*(case['origin']['position']+1)*166*(166+assets)
ps=subprocess.run(['ps','-axo','pid,ppid,state,command'],capture_output=True,text=True,check=True).stdout.splitlines();workers=[int(l.split()[0])for l in ps if len(l.split())>3 and l.split()[1]==str(active['pid'])and'spawn_main'in l and not l.split()[2].startswith('Z')];controller=[l for l in ps if l.split()and l.split()[0]==str(active['pid'])and'run_selector_path_development355.py'in l];assert controller and len(workers)==12 and active['status']=='running'
work=sum(remaining.values());out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'timestamped_measured_workload_lower_bound_not_calibrated_wall_ETA','stage':'selector_path_objective_development355','actual_controller_pid':active['pid'],'actual_worker_pids':workers,'expected_cases':355,'completed_cases':len(completed),'remaining_cases':len(remaining),'mean_seconds_lower_bound':work*float(np.mean(rates))/12,'median_seconds_lower_bound':work*float(np.median(rates))/12,'measured_nonwarm_samples':len(samples),'basis':'Closed matched72-case precision study,59nonwarm request process-CPU/work-unit observations; exact same1000inner/final paths,166option bank and numerical-source closure. Work=members*training_rows*options*(options+assets), ideal12core division. Entire queued/inflight request costs included; no elapsed inflight fraction deduction. Excludes setup,IO,contention,failed-fit repair. Nonstationary fits/support and predictor residuals make wall completion uncertain. Not original broad-market probe timing or live-throughput calibration.','mean_process_CPU_seconds_per_work_unit':float(np.mean(rates)),'median_process_CPU_seconds_per_work_unit':float(np.median(rates)),'reference_summary_sha256':sha(W/'summary.corrected.json'),'current_plan_sha256':sha(D/'plan.json'),'source_sha256':sha(Path(__file__)),'native_failures_snapshot':status['native_failures'],'scoring_integrity_errors_snapshot':status['scoring_integrity_errors']}
api.original._atomic_json(O/'selector_path_development_workload_projection.json',out);print(json.dumps(out,indent=2))
