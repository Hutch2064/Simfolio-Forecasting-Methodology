"""Freeze v4 coverage and qualify training-only regime estimates before forecasting."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys
import candidate
import random_effects
import numpy as np
import sweep as api
OWN=Path(__file__).resolve().parent
V4=OWN.parent/'portfolio_expansion_v4'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
write=lambda p,d:api.original._atomic_json(p,d)
quarter=lambda o:str(o['origin_date'])[:4]+'-Q'+str((int(str(o['origin_date'])[5:7])-1)//3+1)
assert not (OWN/'plan_receipt.json').exists()
panel=api.load_panel(V4/'inputs')
by_id={r['series_id']:r for r in panel['series']}
groups={}
for row in panel['series']:
    key=(row['panel_branch'],row['asset_class'],row['family'])
    prior=groups.get(key)
    if prior is None or (-row['observations'],row['series_id']) < (-prior['observations'],prior['series_id']):
        groups[key]=row
origins={r['series_id']:api.schedule(V4/'inputs',r) for r in panel['series']}
selected={phase:{} for phase in ('development','reserved')}
for row in sorted(groups.values(),key=lambda r:r['series_id']):
    sid=row['series_id'];schedule=origins[sid]
    rolling=sorted([o for o in schedule if o['train_fraction'] is None],key=lambda o:o['position'])
    temporal={round(o['train_fraction'],2):o for o in schedule if o['train_fraction'] is not None}
    n=len(rolling)
    indices={'development':[0,n//2,n-1], 'reserved':[1,n//4,3*n//4,n-2]}
    for phase in selected:
        chosen=[rolling[i] for i in indices[phase]]+[temporal[t] for t in ([.25,.5] if phase=='development' else [.75])]
        assert len({o['origin_label'] for o in chosen})==5
        selected[phase].update({(sid,o['origin_label']):o for o in chosen})
# Add a distinct asset support origin for each portfolio evaluation quarter.
# This prevents unsupported quarters from bypassing the frozen branch cap.
representatives={r['series_id'] for r in groups.values()}
asset_rows=sorted([r for r in panel['series'] if r['panel_branch']=='assets'],
    key=lambda r:(r['series_id'] not in representatives,-r['observations'],r['series_id']))
for phase,entries in selected.items():
    portfolio_quarters={quarter(o) for (sid,_),o in entries.items() if by_id[sid]['panel_branch']=='portfolios'}
    asset_quarters={quarter(o) for (sid,_),o in entries.items() if by_id[sid]['panel_branch']=='assets'}
    for q in sorted(portfolio_quarters-asset_quarters):
        forbidden=set(selected['development']) if phase=='reserved' else set(selected['reserved'])
        found=next(((r['series_id'],o) for r in asset_rows for o in origins[r['series_id']]
            if quarter(o)==q and (r['series_id'],o['origin_label']) not in forbidden),None)
        assert found is not None, ('No distinct individual support',phase,q)
        sid,o=found;entries[sid,o['origin_label']]=o
assert not (set(selected['development'])&set(selected['reserved']))
phases={phase:[{'series_id':sid,'origin':o} for (sid,_),o in sorted(entries.items())]
        for phase,entries in selected.items()}
preflight=[]
for phase,cases in phases.items():
    cache={}
    for case in cases:
        sid=case['series_id'];o=case['origin'];row=by_id[sid]
        if sid not in cache:
            cache[sid]=api.load_series(V4/'inputs',row)
        _,values,_=cache[sid]
        # Match the unchanged native simple/log accounting before training fitting.
        x=np.log1p(np.clip(np.expm1(values[:o['position']+1]),-.999999,np.finfo(float).max/10))
        fit=random_effects.fit_training_batches(x)
        preflight.append({'phase':phase,'series_id':sid,'origin_label':o['origin_label'],
            'batch_count':fit['batches'],'tau_squared':fit['tau_squared'],
            'training_n':len(x),'history_sha256':hashlib.sha256(x.tobytes()).hexdigest()})
    cache.clear()
plan={'schema':'v4_mean_regime_pilot_v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'panel_fingerprint':panel['panel_fingerprint'],'input_root':str(V4/'inputs'),
    'score_protocol_id':'daily_portfolio_panel_crps_v4_train_sd_sqrt_h_logcalendar_two_branch',
    'original_and_portfolio_families':len(groups),'representatives':sorted(representatives),'phases':phases,
    'representative_rule':'Longest observed history per frozen branch/class/family; series_id tie break, no outcome or score selection',
    'support_rule':'Add deterministic individual-asset origins in any selected portfolio-only quarter, using distinct native tasks across development/reserved',
    'preflight_training_prefixes':len(preflight),'preflight_positive_tau_prefixes':sum(r['tau_squared']>0 for r in preflight),
    'field_scores_read':False,'field_forecasts_started':False,'paths':240,
    'candidate_model_id':candidate.MODEL_ID,'initial_parent_model_id':candidate.parent.MODEL_ID,
    'parent_dependency_identity':candidate.build_smooth_mean_pool().dependency_identity,
    'source_hashes':{p.name:sha(p) for p in OWN.glob('*.py')},'design_sha256':sha(OWN/'DESIGN.md'),
    'native_source_closure_digest':api.original.stable_digest(api.original._source_closure()),
    'gate':'Candidate strictly lower single mainQ than current v4 incumbent on complete same development, then reserved; broad/full required before final promotion. Never select/promote from diagnostics.',
    'adaptive_benchmark_reuse':True,'independent_validation_or_significance_claim':False,
    'no_horizon_class_route_or_outcome_fit':True,'output_budget_bytes':1024**3}
write(OWN/'preflight.json',{'complete':True,'field_scores_read':False,'records':preflight})
write(OWN/'plan.json',plan)
write(OWN/'plan_receipt.json',{'utc':plan['utc'],'plan_sha256':sha(OWN/'plan.json'),'before_field_forecasts':True,
    'tests_passed':6,'preflight_sha256':sha(OWN/'preflight.json')})
print(json.dumps({k:plan[k] for k in ['original_and_portfolio_families','preflight_training_prefixes','preflight_positive_tau_prefixes','field_scores_read','field_forecasts_started']}),flush=True)
print(json.dumps({phase:len(cases) for phase,cases in phases.items()}),flush=True)
