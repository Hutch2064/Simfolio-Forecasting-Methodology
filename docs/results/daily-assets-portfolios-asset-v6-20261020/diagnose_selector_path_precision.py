"""Explain matched closed72-case precision differences without changing scores."""
from pathlib import Path
from collections import defaultdict
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_path_objective/screen1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from diagnose_asset_v6_pair import origin_weights
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(D/'summary.corrected.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];audit=read(D/'verification_receipt.json');err=read(D/'percentage_metadata_erratum.json')
assert audit['verified']and audit['summary_sha256']==sha(D/'summary.json')and err['corrected_summary_sha256']==sha(D/'summary.corrected.json')and s['paths']==s['inner_paths']==1000
meta={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};records={(r['case'],r['config_index']):r for r in s['records']};roster=[(i,meta[c['series_id']],c['origin']['origin_date'])for i,c in enumerate(plan['cases'])];weights=origin_weights(roster);labels=('whole_path','endpoint','fixed_WR');pieces={label:{d:defaultdict(float)for d in('horizon','branch','class','family','split')}for label in labels};verified=0
for i,case in enumerate(plan['cases']):
 row=meta[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon'];dates,values,_=api.load_series(V/'inputs',row,verify_files=False);days=(dates[stop:stop+H]-np.datetime64(origin['origin_date'][:10],'D')).astype('timedelta64[D]').astype(int);edges=np.log1p(days.astype(float));mass=np.diff(np.r_[0.,edges])/edges[-1];scale=np.std(values[:stop],ddof=1)*np.sqrt(np.arange(1,H+1))
 for label,index in(('whole_path',1),('endpoint',0),('fixed_WR',None)):
  r=refs[str(i)]['fixed_reference']if index is None else records[i,index];cp=Path(r['checkpoint'])if index is None else D/'losses'/f'{index:03d}-{i:03d}.npz';assert sha(cp)==r['checkpoint_sha256']
  with np.load(cp,allow_pickle=False)as z:loss=z['losses'].copy()
  assert hashlib.sha256(loss.tobytes()).hexdigest()==r['loss_sha256'];cells=weights[i]*loss*mass/scale;total=float(cells.sum());verified+=1;bands=np.bincount(np.searchsorted([31,366,1826,7305],days,side='left'),weights=cells,minlength=5)
  for key,value in zip(('through31days','32to366days','367to1826days','1827to7305days','over7305days'),bands):pieces[label]['horizon'][key]+=float(value)
  for dim,key in(('branch',row['panel_branch']),('class',row['panel_branch']+'/'+row['asset_class']),('family',row['panel_branch']+'/'+row['asset_class']+'/'+row['family']),('split',origin['evaluation_split'])):pieces[label][dim][key]+=total
expected={'whole_path':next(m['main_Q']for m in s['models'].values()if m['config']['criterion']=='path'),'endpoint':next(m['main_Q']for m in s['models'].values()if m['config']['criterion']=='terminal'),'fixed_WR':s['reference_main_Q']['fixed_reference']};assert verified==216
for label,dims in pieces.items():
 for dimension,groups in dims.items():assert abs(sum(groups.values())-expected[label])<2e-12,(label,dimension)
contribution={dimension:{key:100*(pieces['endpoint'][dimension][key]-value)/expected['fixed_WR']for key,value in groups.items()}for dimension,groups in pieces['whole_path'].items()}
for dim,groups in contribution.items():assert abs(sum(groups.values())-100*(expected['endpoint']-expected['whole_path'])/expected['fixed_WR'])<2e-10
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':72,'paths':1000,'inner_paths':1000,'checkpoint_hashes_verified':216,'main_Q':expected,'additive_main_Q_pieces':{label:{dim:dict(groups)for dim,groups in dims.items()}for label,dims in pieces.items()},'incremental_percentage_point_contributions_vs_endpoint_using_fixed_WR_denominator':contribution,'all_dimensions_reconcile_to_unchanged_main_Q':True,'diagnostics_have_zero_score_and_selection_weight':True,'interpretation':'Dimensions are separate additive decompositions of the same main score, not extra scores or promotion gates. Descriptive bands never route forecasts. These adaptively reused72cases are not independent validation or the full panel.','summary_sha256':sha(D/'summary.corrected.json'),'source_sha256':sha(Path(__file__))};api.original._atomic_json(O/'selector_path_precision_diagnostics.json',out);print(json.dumps({k:v for k,v in out.items()if k!='additive_main_Q_pieces'},indent=2))
