from pathlib import Path
import sys,json,sqlite3,hashlib,statistics,datetime
O=Path(__file__).resolve().parent;V=O/'simulation_1000_v5';sys.path.insert(0,str(V));import sweep_v5 as api,reporter,native_index
best='model_testing_equal_bootstrap_FHS_smooth_mean_Gaussian_stable_wealth_v1';ids=[best,*reporter.PUBLIC_IDS]
c=sqlite3.connect(f'file:{V}/results/native-loss-index.sqlite3?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('BEGIN');maps={}
for mid in ids:
 rows=c.execute('SELECT task_key,checkpoint_path,execution_fingerprint FROM task_scores_main WHERE model_id=? AND scorer_hash=? AND status=?',(mid,native_index.source_digest(),'ok')).fetchall();assert len({r['execution_fingerprint'] for r in rows})==1;maps[mid]={r['task_key']:dict(r) for r in rows}
c.close();common=set.intersection(*(set(maps[m]) for m in ids));keys=sorted(common,key=lambda k:hashlib.sha256(k.encode()).hexdigest())[:256];times={m:[] for m in ids};records=[]
for key in keys:
 sid,origin=json.loads(key);case={'series_id':sid,'origin':origin,'seconds':{}}
 for mid in ids:
  r=maps[mid][key];_,d=api.original._read_checkpoint(Path(r['checkpoint_path']),api.task_identity(r['execution_fingerprint'],mid,sid,origin));assert d['simulations']==1000;t=d['forecast_and_score_seconds'];times[mid].append(t);case['seconds'][mid]=t
 records.append(case)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Deterministic SHA256-first256 cases from exact four-way completed intersection, current full-run timings. Training-fit + simulation + exact CRPS; no checkpoint IO. Warmup/OS contention not controlled; neither production latency nor full-work speedup claim.','common_successful_tasks_available':len(common),'matched_sample_tasks':len(keys),'paths':1000,'seconds_by_model':{m:{'sum':sum(t),'mean':statistics.fmean(t),'median':statistics.median(t)} for m,t in times.items()},'records':records}
result['best_to_baseline_total_time_ratio']={m:sum(times[best])/sum(times[m]) for m in reporter.PUBLIC_IDS}
(O/'v5_matched_runtime_snapshot.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
