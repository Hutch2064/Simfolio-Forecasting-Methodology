"""History-only scale comparison; no fits, forecasts, scores or admission."""
from pathlib import Path
import datetime,hashlib,json,statistics,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';sys.path.insert(0,str(V))
from inputs import AssetInputs
from candidates.asset_selector_universe_precision.candidate import schedule
read=lambda p:json.loads(p.read_text())
plan_path=V/'candidates/asset_selector_inner_precision/study1000/plan.json';plan=read(plan_path);data=AssetInputs(V/'inputs');records=[]
for case in plan['cases']:
 row=data.by_id[case['series_id']];dates,returns=data.source(row);stop=case['origin']['position']+1;history=np.asarray(returns[:stop]);_,folds=schedule(stop,case['origin']['max_horizon'])
 scales=[float(np.std(history[:end],ddof=1))for end,H in folds];assert np.isfinite(scales).all()and min(scales)>0
 records.append({'series_id':case['series_id'],'origin_label':case['origin']['origin_label'],'known_rows':stop,'folds':[{'training_rows':end,'validation_horizon':H,'validation_stop':end+H,'training_last_date':str(dates[end-1]),'known_validation_last_date':str(dates[end+H-1]),'training_sd_relative_to_request':scale/float(np.std(history,ddof=1))}for(end,H),scale in zip(folds,scales)],'largest_to_smallest_fold_sd_ratio':max(scales)/min(scales)})
ratios=[r['largest_to_smallest_fold_sd_ratio']for r in records]
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reference_plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'cases':len(records),'fold_scale_ratio_minimum':min(ratios),'fold_scale_ratio_median':statistics.median(ratios),'fold_scale_ratio_maximum':max(ratios),'all_scales_positive_finite':True,'only_request_history_accessed':True,'no_forecasts_fits_outer_scores_or_new_workers':True,'interpretation':'Current full-request SD would cancel uniformly; normalizing each internal fold by its own prefix SD instead changes relative influence between folds. This is a candidate objective contrast, not evidence of an accuracy gain or invalidity of previous scores.','records':records}
(O/'selector_fold_scaling_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='records'},indent=2))
