"""Read-only matched partial report; no filtering, admission or promotion."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_path_objective/development1000';sys.path.insert(0,str(V));sys.path.insert(0,str(O))
import score_main
from score_percentage import percent_better
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=O/'selector_path_partial_score.json');args=parser.parse_args()
plan=read(D/'plan.json');assert all(sha(O/p)==h for p,h in plan['source_hashes'].items());references=read(D/'references.json')['cases'];metadata={r['series_id']:r for r in read(V/'inputs/panel.json')['series']};arms={k:[]for k in('fixed_reference','terminal','path')};snapshots=[];verified=[]
files=sorted((D/'case_records').glob('*.json'));seen=set()
for path in files:
 digest=sha(path);case=read(path);i=case['case'];assert i not in seen and not case['errors']and len(case['records'])==2;seen.add(i);definition=plan['cases'][i];origin=definition['origin'];sid=definition['series_id'];row=metadata[sid];stop=origin['position']+1;H=origin['max_horizon'];dates=np.load(V/'inputs'/row['dates_file'],mmap_mode='r',allow_pickle=False);target=np.load(V/'inputs'/row['returns_file'],mmap_mode='r',allow_pickle=False)
 def verify(label,record,checkpoint):
  assert record['series_id']==sid and record['origin_label']==origin['origin_label']and record['origin_date']==origin['origin_date']and record['paths']==1000 and sha(checkpoint)==record['checkpoint_sha256']
  with np.load(checkpoint,allow_pickle=False)as saved:loss=saved['losses'];assert loss.shape==(H,)and np.isfinite(loss).all()and np.all(loss>=0)and hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
  q=score_main.origin_score(loss,target[:stop],origin['origin_date'],dates[stop:stop+H]);assert q==record['score'];arms[label].append({**record,'score':q});verified.append({'case':i,'label':label,'checkpoint_sha256':record['checkpoint_sha256'],'loss_sha256':record['loss_sha256']})
 verify('fixed_reference',references[str(i)]['fixed_reference'],Path(references[str(i)]['fixed_reference']['checkpoint']))
 assert {r['config_index']for r in case['records']}=={0,1}
 for record in case['records']:
  assert record['case']==i and record['inner_paths']==1000;verify('terminal'if record['config_index']==0 else'path',record,D/'losses'/f"{record['config_index']:03d}-{i:03d}.npz")
 assert sha(path)==digest;snapshots.append({'file':str(path.relative_to(O)),'sha256':digest,'case':i})
keys=lambda rows:{(r['series_id'],r['origin_date'],r['origin_label'])for r in rows}
assert len(keys(arms['fixed_reference']))==len(files)and all(keys(r)==keys(arms['fixed_reference'])for r in arms.values())
reports={k:score_main.aggregate_records(r,metadata)for k,r in arms.items()};baseline=reports['fixed_reference']['score'];assert baseline is not None
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preliminary':True,'official':False,'phase':'development_partial','matched_requests':len(files),'expected_requests':len(plan['cases']),'distinct_series':len({r['series_id']for r in arms['fixed_reference']}),'completed_requests_by_branch':{branch:sum(metadata[r['series_id']]['panel_branch']==branch for r in arms['fixed_reference'])for branch in('assets','portfolios')},'all_completed_snapshot_requests_included':True,'score_protocol':score_main.SCORE_PROTOCOL_ID,'paths':1000,'inner_paths':1000,'scoring_integrity_errors':0,'saved_candidate_forecasts_verified':2*len(files),'saved_baseline_forecasts_verified':len(files),'models':{label:{'main_Q':r['score'],'percent_better_vs_fixed_reference':percent_better(r['score'],baseline),'quarter_count':r['quarter_count']}for label,r in reports.items()},'whole_path_percent_better_vs_endpoint_same_requests':percent_better(reports['path']['score'],reports['terminal']['score']),'coverage_warning':'Changing completion coverage can change cumulative scores. This is not the64,049-request official panel or an independent validation. No comparison with differently covered historical scores is warranted.','promotion_or_admission':False,'canonical_aggregation_unchanged':True,'case_record_snapshot':snapshots,'checkpoint_verification':verified,'references_sha256':sha(D/'references.json'),'plan_sha256':sha(D/'plan.json'),'verifier_sha256':sha(Path(__file__))}
args.output.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items()if k not in('case_record_snapshot','checkpoint_verification')},indent=2))
