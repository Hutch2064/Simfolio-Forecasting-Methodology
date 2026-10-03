"""Targeted posterior coverage checks selected by failed inference diagnostics."""
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'mcmc_runtime'))
import hashlib

import inference
import optimized_mcmc as opt
from runtime_setup import DATA, initialize

from simfolio_forecasting_methodology.experiment import build_experiment_plan

initialize(True);opt.install(False)
plan=build_experiment_plan(DATA)
checks=[]
# These are training-byte hashes of the three poorest importance diagnostics,
# selected without looking at their realized outcomes or scores.
wanted={'2893e73c582bb0d82db1e28450958ac7f7cea405bd5531976a689f6182d97ecd',
        '518279287509b803d1ec3a19b6b374ad56829369e0c100dc59756a0c9b853abf',
        'c412e5c3e9a1ae14da205506286274d6fdeef35e88f2e762128b53cb1ea9f213'}
for t in plan.tasks:
 for a in range(t.training.asset_log_returns.shape[1]):
  data=t.training.asset_log_returns[:,a].tobytes();h=hashlib.sha256(data).hexdigest()
  if h not in wanted:continue
  wanted.remove(h)
  for method in ('laplace_mixture_is','hmc_long'):
   inference.parameter_fit(data,method)
   checks.append(inference.RECORDS[-1]);print(json.dumps(checks[-1]),flush=True)
 if not wanted:break
out=Path(sys.argv[1]);out.write_text(json.dumps(checks,indent=2)+'\n')
