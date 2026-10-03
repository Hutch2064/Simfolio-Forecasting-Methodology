"""Interleaved cold-fit model timings; all daily paths checked, no CTV claim."""
import json
import time
from dataclasses import asdict
from pathlib import Path

import optimized_mcmc as opt
import parameter_mcmc_tuned as pm
from runtime_setup import DATA, initialize

from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
 FilteredInnovationFixedMeanMomentSV,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg
from simfolio_forecasting_methodology.runner import ForecastContext


def main():
 plan=build_experiment_plan(DATA);initialize(True)
 original=(pm.chain,bd._bdes_multiscale_components,dg.kalman_terminal_posterior,dg.simulate_future_gaussian_uniforms,bd._fit_sv_map_state_space_params,pm.parameter_fit)
 opt.install(False);fast=(pm.chain,bd._bdes_multiscale_components,dg.kalman_terminal_posterior,dg.simulate_future_gaussian_uniforms,bd._fit_sv_map_state_space_params,pm.parameter_fit)
 def clear():
  pm.parameter_fit.cache_clear();opt.nodes.cache_clear();opt.power.cache_clear()
  for cell in bd.fit_bdes_fastmap.__closure__ or ():
   if hasattr(cell.cell_contents,'cache_clear'):cell.cell_contents.cache_clear()
 t=plan.tasks[0];warm=ForecastContext(pm.CANDIDATES[0].model_id,t.portfolio_id,t.origin_label,2,240,t.seed,t.future_dates[:2],t.origin_date);opt.OptimizedMCMC().simulate_daily_log_returns(t.training,warm)
 records=[]
 for index in (0,40,48):
  t=plan.tasks[index];context=ForecastContext(pm.CANDIDATES[0].model_id,t.portfolio_id,t.origin_label,t.horizon_days,240,t.seed,t.future_dates,t.origin_date)
  for arm in ('reference','optimized','frontier','frontier','optimized','reference')*2:
   pm.chain,bd._bdes_multiscale_components,dg.kalman_terminal_posterior,dg.simulate_future_gaussian_uniforms,bd._fit_sv_map_state_space_params,pm.parameter_fit=fast if arm=='optimized' else original
   clear();model=opt.OptimizedMCMC() if arm=='optimized' else (FilteredInnovationFixedMeanMomentSV() if arm=='frontier' else pm.CANDIDATES[0])
   start=time.perf_counter();cpu=time.process_time();paths=model.simulate_daily_log_returns(t.training,context);seconds=time.perf_counter()-start;cpu_seconds=time.process_time()-cpu
   if arm=='reference':reference=paths
   elif arm=='optimized':assert paths.tobytes()==reference.tobytes(),index
   record={'task': index,'portfolio': t.portfolio_id,'origin': t.origin_label,'assets': t.training.asset_log_returns.shape[1],'observations': t.training.asset_log_returns.shape[0],'horizon': t.horizon_days,'arm': arm,'seconds': seconds,'cpu_seconds': cpu_seconds,'paths_byte_exact_to_mcmc_reference': arm!='frontier'};records.append(record);print(json.dumps(record),flush=True)
 Path(__file__).with_name('matched-model-cases.json').write_text(json.dumps({'scope': 'local_model_wall_and_process_CPU_time_warm_compilers_cold_model_fits_not_CTV','model': asdict(opt.OptimizedMCMC()),'records': records},indent=2))
if __name__=='__main__':main()
