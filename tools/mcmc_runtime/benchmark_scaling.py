"""Matched local asset-count scaling; native production/CTV not measured."""
import json
import time
from pathlib import Path

import numpy as np
import optimized_mcmc as opt
import pandas as pd
import parameter_mcmc_tuned as pm
from runtime_setup import DATA, initialize

from simfolio_forecasting_methodology.data import load_canonical_engine_inputs
from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
 FilteredInnovationFixedMeanMomentSV,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg
from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData


def main():
 initialize(True);original_chain=pm.chain;original_multi=bd._bdes_multiscale_components;original_terminal=dg.kalman_terminal_posterior;original_uniforms=dg.simulate_future_gaussian_uniforms;original_map=bd._fit_sv_map_state_space_params;original_fit=pm.parameter_fit
 opt.install(False);fast_chain=pm.chain;fast_multi=bd._bdes_multiscale_components;fast_terminal=dg.kalman_terminal_posterior;fast_uniforms=dg.simulate_future_gaussian_uniforms;fast_map=bd._fit_sv_map_state_space_params;fast_fit=pm.parameter_fit
 frame,_=load_canonical_engine_inputs(DATA);frame=frame.iloc[:10024];future=pd.bdate_range(frame.index[-1]+pd.offsets.BDay(),periods=2520).values;records=[]
 def clear():
  pm.parameter_fit.cache_clear();opt.nodes.cache_clear();opt.power.cache_clear()
  for cell in bd.fit_bdes_fastmap.__closure__ or ():
   if hasattr(cell.cell_contents,'cache_clear'):cell.cell_contents.cache_clear()
 for assets in (6,25,50):
  values=frame.iloc[:,:assets].to_numpy();weights=tuple(np.repeat(1/assets,assets));training=TrainingData(values.mean(axis=1),values,PortfolioPolicy(tuple(frame.columns[:assets]),weights,'annually'),frame.index.values)
  ctx=ForecastContext(pm.CANDIDATES[0].model_id,f'scaling_{assets}','diagnostic',2520,240,183,future,str(frame.index[-1].date()))
  if assets==6:
   warm=ForecastContext(ctx.model_id,ctx.portfolio_id,ctx.origin_label,2,240,183,future[:2],ctx.origin_date);opt.OptimizedMCMC().simulate_daily_log_returns(training,warm)
  for arm in ('reference','optimized1','optimized2','optimized4','optimized6','frontier','frontier','optimized6','optimized4','optimized2','optimized1','reference'):
   fast=arm.startswith('optimized');pm.chain=fast_chain if fast else original_chain;bd._bdes_multiscale_components=fast_multi if fast else original_multi;dg.kalman_terminal_posterior=fast_terminal if fast else original_terminal;dg.simulate_future_gaussian_uniforms=fast_uniforms if fast else original_uniforms;bd._fit_sv_map_state_space_params=fast_map if fast else original_map;pm.parameter_fit=fast_fit if fast else original_fit
   clear();model=opt.OptimizedMCMC(fit_workers=int(arm[-1])) if fast else (FilteredInnovationFixedMeanMomentSV() if arm=='frontier' else pm.CANDIDATES[0])
   start=time.perf_counter();cpu=time.process_time();paths=model.simulate_daily_log_returns(training,ctx);seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
   if arm=='reference':reference=paths
   elif fast:assert paths.tobytes()==reference.tobytes(),(assets,arm)
   record={'assets': assets,'observations': len(frame),'horizon': 2520,'arm': arm,'seconds': seconds,'cpu_seconds': cpu_seconds,'full_mcmc_paths_exact': (arm!='frontier')};records.append(record);print(json.dumps(record),flush=True)
 Path(__file__).with_name('scaling-benchmark-final.json').write_text(json.dumps({'scope': 'local_warm_compilers_cold_model_fits_equal_weights_not_OOS_scores_or_CTV','records': records},indent=2))
if __name__=='__main__':main()
