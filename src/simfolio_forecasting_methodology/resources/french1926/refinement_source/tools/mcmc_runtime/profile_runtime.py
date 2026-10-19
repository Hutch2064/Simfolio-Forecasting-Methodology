"""Cold-fit, single-worker CPU decomposition of the final implementation."""
import cProfile
import io
import json
import pstats
from pathlib import Path

import optimized_mcmc as opt
import parameter_mcmc_tuned as pm
from runtime_setup import DATA, initialize

from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.runner import ForecastContext


def main():
 initialize(True);sequential_fit=pm.parameter_fit;opt.install(False);pm.parameter_fit=sequential_fit;plan=build_experiment_plan(DATA)
 t=plan.tasks[0];ctx=ForecastContext(opt.OptimizedMCMC().model_id,t.portfolio_id,t.origin_label,2,240,t.seed,t.future_dates[:2],t.origin_date);opt.OptimizedMCMC().simulate_daily_log_returns(t.training,ctx)
 for i in (0,40,48):
  t=plan.tasks[i];ctx=ForecastContext(opt.OptimizedMCMC().model_id,t.portfolio_id,t.origin_label,t.horizon_days,240,t.seed,t.future_dates,t.origin_date)
  pm.parameter_fit.cache_clear();opt.nodes.cache_clear();opt.power.cache_clear()
  for cell in bd.fit_bdes_fastmap.__closure__ or ():
   if hasattr(cell.cell_contents,'cache_clear'):cell.cell_contents.cache_clear()
  p=cProfile.Profile();p.enable();opt.OptimizedMCMC(fit_workers=1).simulate_daily_log_returns(t.training,ctx);p.disable();out=io.StringIO();pstats.Stats(p,stream=out).sort_stats('cumulative').print_stats(25)
  file=Path(__file__).with_name(f'optimized-profile-{i}.txt');file.write_text(out.getvalue());print(json.dumps({'task': i,'profile': str(file)}),flush=True)
if __name__=='__main__':main()
