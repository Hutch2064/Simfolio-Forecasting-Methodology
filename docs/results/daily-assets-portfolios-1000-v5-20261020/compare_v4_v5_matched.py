from pathlib import Path
import sys,json,sqlite3,datetime
O=Path(__file__).resolve().parent;V=O/'simulation_1000_v5';sys.path.insert(0,str(V));import reporter,native_index
best='model_testing_equal_bootstrap_FHS_smooth_mean_Gaussian_stable_wealth_v1';old='model_testing_equal_bootstrap_FHS_smooth_mean_shrinkage_Gaussian_uncertainty_v1';ids=[best,*reporter.PUBLIC_IDS]
sets={};maps={}
for label,root in [('1000',V),('240',O/'portfolio_expansion_v4')]:
 p=root/'results/native-loss-index.sqlite3';assert p.exists(),p
 c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('BEGIN');maps[label]={}
 for mid in ids:
  saved=old if label=='240' and mid==best else mid
  rows=c.execute('SELECT task_key,series_id,origin_date,main_score,execution_fingerprint,scorer_hash FROM task_scores_main WHERE model_id=? AND status=?',(saved,'ok')).fetchall();assert len({r['execution_fingerprint'] for r in rows})==len({r['scorer_hash'] for r in rows})==1
  maps[label][mid]={r['task_key']:{'series_id':r['series_id'],'origin_date':r['origin_date'],'score':r['main_score']} for r in rows}
 c.close()
common=set.intersection(*(set(d) for dd in maps.values() for d in dd.values()));assert common
panel=json.loads((V/'inputs/panel.json').read_text());meta={r['series_id']:r for r in panel['series']}
other=json.loads((O/'portfolio_expansion_v4/inputs/panel.json').read_text());oldmeta={r['series_id']:r for r in other['series']}
assert set(meta)==set(oldmeta)
for sid in meta:
 for k in ['panel_branch','asset_class','family']:assert meta[sid][k]==oldmeta[sid][k]
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'identical_tasks_all_eight_executions':len(common),'preliminary':True,'profiles':{},'panel_data_and_main_formula_unchanged':True,'sampling_version_is_not_a_nested_identical_RNG_pathcount_ablation':True}
for label,mm in maps.items():
 scores={m:reporter.scalar([d[k] for k in sorted(common)],meta) for m,d in mm.items()};worst=max(reporter.PUBLIC_IDS,key=lambda m:scores[m]);result['profiles'][label]={'main_scores':scores,'worst_Public_reference':worst,'percent_better_vs_worst_Public':{m:100*(1-v/scores[worst]) for m,v in scores.items()}}
(O/'v4_v5_exact_matched_count_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
