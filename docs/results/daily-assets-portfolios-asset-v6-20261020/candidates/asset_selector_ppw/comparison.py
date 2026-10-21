"""Fit the nested-menu contrast once; reuse the exact original coefficient block."""
import time
import numpy as np
from candidates.asset_selector_universe import bank as base
from candidates.asset_selector_universe_certified import energy
from . import candidate
from .bank import schedule,BASE_COMPONENT_COUNT

def prepare_pair(history,weights,horizon,training_dates,*,inner_paths=100,width=64):
    started=time.perf_counter();expanded=candidate.prepare(history,weights,horizon,training_dates,inner_paths=inner_paths,width=width)
    eligible,_=schedule(len(history),horizon);valid=[i for i in eligible if i not in expanded['invalid_internal_options']];keep=[j for j,i in enumerate(valid)if i<BASE_COMPONENT_COUNT];original=[valid[j]for j in keep]
    learned={};fits={};coefficients={}
    for mode,values in expanded['coefficients'].items():
        A=values['A'][keep];B=values['B'][np.ix_(keep,keep)];fit=energy.fit(A,B);w=np.zeros(BASE_COMPONENT_COUNT);w[original]=fit['weights'];learned[mode]=w;fits[mode]=fit;coefficients[mode]={'A':A,'B':B}
    needed=sorted({i for w in learned.values()for i in np.flatnonzero(w>0)});full,failures=base.prefix_fits(history,needed)
    if failures:raise ArithmeticError(f'Original-menu full-history refit failed: {failures}')
    trace=[]
    for source in expanded['trace']:
        row={k:v for k,v in source.items()if k not in('PPW_dependence_fit','base_component_terminal_ensemble_sha256')};row['eligible_this_fold']=[i for i in row['eligible_this_fold']if i<BASE_COMPONENT_COUNT];row['terminal_ensemble_sha256']=source['base_component_terminal_ensemble_sha256'];trace.append(row)
    control={**expanded,**full,'weights_by_mode':learned,'fit':fits,'coefficients':coefficients,'trace':trace,'invalid_internal_options':{i:v for i,v in expanded['invalid_internal_options'].items()if i<BASE_COMPONENT_COUNT},'diagnostics':{**expanded['diagnostics'],'selection_and_refit_seconds':time.perf_counter()-started,'shared_menu_fit_scope':'All168options generated once; original166coefficient block reused exactly, not an individual runtime benchmark'}}
    return control,expanded
