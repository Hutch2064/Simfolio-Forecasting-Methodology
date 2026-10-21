"""Additive decomposition of matched existing Q, with zero selection weight."""
from pathlib import Path
from collections import defaultdict
import datetime,hashlib,json,sys
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_selector_universe_certified';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from diagnose_asset_v6_pair import origin_weights
from candidates.asset_selector_universe_certified.candidate import identity
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
CONFIG={'pool':'stack'}


def main():
    metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}
    result={}
    for phase in ('development1000',):
        D=C/phase
        if not(D/'summary.json').exists():continue
        summary=read(D/'summary.json');plan=read(D/'plan.json');refs=read(D/'references.json')['cases'];mid=identity(CONFIG)
        assert summary['complete_execution']and not summary['errors']and summary['paths']==1000
        rows=[r for r in summary['records']if r['model_id']==mid]
        assert len(rows)==len(plan['cases'])and len({r['case']for r in rows})==len(rows)
        weights=origin_weights([(r['case'],metadata[r['series_id']],r['origin_date'])for r in rows])
        pieces={label:{dimension:defaultdict(float)for dimension in('branch','class','family','split','horizon')}for label in('candidate','parent','fixed_WR')}
        totals={label:0. for label in pieces}
        for r in rows:
            number=r['case'];case=plan['cases'][number];row=metadata[case['series_id']];origin=case['origin'];stop=origin['position']+1;H=origin['max_horizon']
            dates,values,_=api.load_series(V/'inputs',row,verify_files=False)
            days=(dates[stop:stop+H].astype('datetime64[D]')-np.datetime64(origin['origin_date'][:10],'D')).astype(int)
            edges=np.log1p(days.astype(float));mass=np.diff(np.r_[0.,edges])/edges[-1]
            scale=np.std(values[:stop],ddof=1)*np.sqrt(np.arange(1,H+1))
            records={'candidate':{**r,'checkpoint':str(D/'losses'/f"{r['config_index']:03d}-{number:03d}.npz")},'parent':refs[str(number)]['current_best'],'fixed_WR':refs[str(number)]['fixed_reference']}
            for label,record in records.items():
                path=Path(record['checkpoint']);assert sha(path)==record['checkpoint_sha256']
                with np.load(path,allow_pickle=False)as z:loss=z['losses'if'losses'in z.files else'loss'].copy()
                assert hashlib.sha256(loss.tobytes()).hexdigest()==record['loss_sha256']
                cells=weights[number]*loss*mass/scale;total=float(cells.sum());totals[label]+=total
                by_band=np.bincount(np.searchsorted([31,366,1826,7305],days,side='left'),weights=cells,minlength=5)
                for band,value in zip(('through31days','32to366days','367to1826days','1827to7305days','over7305days'),by_band):pieces[label]['horizon'][band]+=float(value)
                for dimension,key in [('branch',row['panel_branch']),('class',row['panel_branch']+'/'+row['asset_class']),('family',row['panel_branch']+'/'+row['asset_class']+'/'+row['family']),('split',origin['evaluation_split'])]:pieces[label][dimension][key]+=total
        expected={'candidate':summary['models'][mid]['main_Q'],'parent':summary['reference_main_Q']['current_best'],'fixed_WR':summary['reference_main_Q']['fixed_reference']}
        for label in pieces:
            assert abs(totals[label]-expected[label])<2e-12
            for dimension in pieces[label]:assert abs(sum(pieces[label][dimension].values())-expected[label])<2e-12
        plain={label:{dim:dict(groups)for dim,groups in dims.items()}for label,dims in pieces.items()}
        contributions={label:{dim:{group:100*(pieces['fixed_WR'][dim][group]-value)/expected['fixed_WR']for group,value in groups.items()}for dim,groups in dims.items()}for label,dims in pieces.items()if label!='fixed_WR'}
        incremental={dim:{group:100*(pieces['parent'][dim][group]-value)/expected['fixed_WR']for group,value in groups.items()}for dim,groups in pieces['candidate'].items()}
        for dim,groups in incremental.items():assert abs(sum(groups.values())-100*(expected['parent']-expected['candidate'])/expected['fixed_WR'])<2e-10
        result[phase]={'incremental_percentage_point_contributions_over_parent_vs_fixed_WR':incremental,'matched_cases':len(rows),'paths':1000,'main_Q':expected,'additive_main_Q_pieces':plain,'percent_better_contributions_vs_fixed_WR':contributions,'canonical_Q_reconciled_all_dimensions':True}
    output={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phases':result,'diagnostic_only_zero_score_and_selection_weight':True,'not_independent_validation_or_significance':True,'horizon_bands':'Descriptive existing calendar bands only; never forecast routes, model-specific weights or new selection objectives.'}
    api.original._atomic_json(C/'diagnostics.json',output)
    print(json.dumps({'phases':list(result),'all_dimensions_reconcile':True,'diagnostics_zero_selection_weight':True}),flush=True)

if __name__=='__main__':main()
