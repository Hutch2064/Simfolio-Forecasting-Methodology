"""Read-only historical precision rescore; never imports financial candidates."""
import pathlib,json,hashlib,sqlite3,sys,collections,datetime
import numpy as np
D=pathlib.Path(__file__).resolve().parent;O=D.parent;V=O/'asset_level_v6/candidates'
sys.path.insert(0,str(O/'requested_horizon_crps_v11'))
import score_requested_horizon_crps_v11 as scoring
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
scoring.assert_frozen_sources()
assert sha(scoring.__file__)=='6fe6345747292142bb0599c1c3ec0a98b78cc39a54d7cbeb809ef7741218b52a'
db=sqlite3.connect('file:'+str(O/'requested_horizon_crps_v11/scale_index.sqlite3')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
baseline=sqlite3.connect('file:'+str(O/'asset_level_v6/full_pair_rescore_v7_scores.sqlite3')+'?mode=ro',uri=True);baseline.row_factory=sqlite3.Row
WR='model_testing_asset_joint_blocks_replacement_v6';cohorts=[]
for name in ['asset_selector_endpoint_flat','asset_selector_endpoint_precision']:
 stage='broader1000'if name.endswith('_flat')else'study1000';d=V/name/stage;p=json.loads((d/'plan.json').read_text());refs=json.loads((d/'references.json').read_text())['cases'];verification=json.loads((d/'verification_receipt.json').read_text());assert verification['verified']is True and verification['plan_sha256']==sha(d/'plan.json') and verification['summary_sha256']==sha(d/'summary.json')
 assert p['outer_paths']==1000 and len(p['cases'])==355
 scales=[];wr_verified=[]
 for i,q in enumerate(p['cases']):
  a=q['origin'];rs=db.execute('select r.*,s.origin_position,s.series_index from request_scales r join prefix_scales s using(case_index) where request_kind="original" and r.series_id=? and r.origin_date=? and native_h=? and origin_position=?',(q['series_id'],a['origin_date'],a['max_horizon'],a['position'])).fetchall();assert len(rs)==1;s=dict(rs[0]);scales.append(s)
  f=refs[str(i)]['fixed_reference'];br=baseline.execute('select * from task_scores_v7 where model_id=? and case_index=?',(WR,f"{s['series_index']:06d}/{s['origin_index']:03d}")).fetchone();assert br and f['paths']==1000 and f['series_id']==s['series_id'] and f['origin_date']==s['origin_date'] and f['checkpoint_sha256']==br['source_checkpoint_sha256'] and f['loss_sha256']==br['loss_sha256'];wr_verified.append(f['checkpoint_sha256'])
 cfgs={};losscommit=[];rows=[]
 for k,cfg in enumerate(p['configs']):
  assert cfg['seed_replication']in [0,1,2] and cfg['schedule']=='half_doubling';per=[]
  for i,s in enumerate(scales):
   rp=d/'records'/f'{k:03d}-{i:03d}.json';cp=d/'losses'/f'{k:03d}-{i:03d}.npz';r=json.loads(rp.read_text());assert r['case']==i and r['config_index']==k and r['configuration']==cfg and r['source_closure_digest']==digest(p['source_hashes']) and r['outer_paths']==1000 and r['series_id']==s['series_id'] and r['origin_date']==s['origin_date'] and sha(cp)==r['checkpoint_sha256']
   with np.load(cp,allow_pickle=False)as z:loss=z['losses'];assert loss.dtype==np.float64 and loss.shape==(s['native_h'],) and np.isfinite(loss).all() and np.all(loss>=0) and hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256']
   # The final RNG seed must match across all inner counts and seeds.
   if k==0: s['final_seed']=r['final_seed_shared_across_inner_counts']
   assert s['final_seed']==r['final_seed_shared_across_inner_counts']
   value=scoring.standardized_request_crps(float(loss.mean()),s['training_sd'],s['native_h']);per.append(value);losscommit.append(r['checkpoint_sha256']);rows.append({'cohort':name,'config_index':k,'request_id':s['request_id'],'raw_mean_crps':float(loss.mean()),'score_v11':value,'checkpoint_sha256':r['checkpoint_sha256'],'record_sha256':sha(rp)})
  buckets={}
  for b in ['short','medium','long']:
   ids=[i for i,s in enumerate(scales)if s['horizon_bucket']==b];mass=sum(scales[i]['within_bucket_weight']for i in ids);buckets[b]={'requests':len(ids),'full_bucket_weight_support':mass,'conditional_Q':sum(scales[i]['within_bucket_weight']*per[i]for i in ids)/mass if mass else None}
  assert all(v['requests']for v in buckets.values())
  cfgs[f"N{cfg['inner_paths']}_seed{cfg['seed_replication']}"]={'config':cfg,'buckets':buckets,'equal_three_bucket_conditional_Q':sum(b['conditional_Q']for b in buckets.values())/3,'pooled_fixed_support_subtotal_diagnostic':sum(s['global_weight']*v for s,v in zip(scales,per))}
 comparisons=[]
 for seed in [0,1,2]:
  control=cfgs[f'N2_seed{seed}']['equal_three_bucket_conditional_Q']
  for n in sorted({c['inner_paths']for c in p['configs']} - {2}):
   target=cfgs[f'N{n}_seed{seed}']['equal_three_bucket_conditional_Q'];comparisons.append({'inner_seed':seed,'N':n,'N2_Q':control,'target_Q':target,'percent_lower_Q_vs_matched_N2':100*(1-target/control)})
 cohorts.append({'candidate':name,'historical_menu':'flat167'if name.endswith('_flat')else'legacy nested stack, separate cohort','plan_sha256':sha(d/'plan.json'),'references_sha256':sha(d/'references.json'),'verification_sha256':sha(d/'verification_receipt.json'),'source_closure_digest':digest(p['source_hashes']),'requests':355,'unique_prefixes':len({(s['series_id'],s['origin_position'])for s in scales}),'bucket_counts':dict(collections.Counter(s['horizon_bucket']for s in scales)),'final_paths':1000,'verified_rawloss_cells':len(p['configs'])*355,'WR1000_identity_hashes_validated':355,'WR_Q':None,'WR_Q_unavailable_reason':'Historical1000rawmeans absent for330requests; no240substitution or archived-vector restore. N-to-N comparisons have exact matched complete355coverage.','scale_mapping_digest':digest(scales),'checkpoint_commitment':digest(losscommit),'configurations':cfgs,'matched_inner_seed_comparisons':comparisons,'seed_averages_descriptive_not_independent_panel':{str(n):float(np.mean([cfgs[f'N{n}_seed{s}']['equal_three_bucket_conditional_Q']for s in [0,1,2]]))for n in sorted({c['inner_paths']for c in p['configs']})}})
result={'schema':'historical-precision-V11-rescore-v1','reproducer_sha256':sha(__file__),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'score_protocol_id':scoring.PROTOCOL_ID,'scorer_sha256':sha(scoring.__file__),'protocol_sha256':sha(scoring.PROTOCOL_PATH),'scale_index_sha256':sha(O/'requested_horizon_crps_v11/scale_index.sqlite3'),'cohorts':cohorts,'scope':'Predeclared historical355original requests, fixedV9within-bucket fullschedule weights restricted to identical support then equal3bucket conditional mean. Not actual full320017headline. Flat167and legacy nestedstack separately, no cross-menu ranking.','financial_fits_or_forecasts_run':False,'baseline_forecasts':0,'archive_restores':0,'promotion':False,'modern_finite41_mapped_precision_optimality_not_established':True,'seed_scope':'Inner seeds0/1/2 separately; same final seed per request across N. Adaptively reused development sample, not independent validation.'}
(D/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps([{'candidate':c['candidate'],'comparisons':c['matched_inner_seed_comparisons'],'seedmeans':c['seed_averages_descriptive_not_independent_panel']}for c in cohorts],indent=2));print('resultSHA',sha(D/'result.json'))
