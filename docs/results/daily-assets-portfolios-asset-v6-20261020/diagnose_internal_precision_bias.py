"""Analytic toy mechanism, not a result for any panel model or score change."""
from pathlib import Path
import datetime,json,math
import numpy as np
from scipy.optimize import brentq
O=Path(__file__).resolve().parent

records=[]
for members in (10,30,100,300,1000):
    # Independent Gaussian X~N(0,sigma^2), Y~N(0,1). Ordinary empirical
    # CRPS includes self-pairs, whose expected distance is zero.
    c=1-1/members
    score=lambda sigma:math.sqrt(2/math.pi)*math.sqrt(1+sigma*sigma)-c*sigma/math.sqrt(math.pi)
    derivative=lambda sigma:math.sqrt(2/math.pi)*sigma/math.sqrt(1+sigma*sigma)-c/math.sqrt(math.pi)
    upper=1.
    while derivative(upper)<=0:upper*=2
    numeric=brentq(derivative,0.,upper);analytic=c/math.sqrt(2-c*c)
    assert abs(numeric-analytic)<1e-11 and score(analytic)<score(1.)
    records.append({'iid_members':members,'expected_empirical_CRPS_optimal_forecast_sd':analytic,'true_sd':1.,'sd_percent_below_true':100*(1-analytic),'expected_empirical_CRPS_at_its_optimum':score(analytic),'expected_empirical_CRPS_at_true_sd':score(1.),'root_crosscheck_absolute_error':abs(numeric-analytic)})

# A naive independent-member diagonal correction is not a convexity-preserving
# operation for mixtures with identical/common-random-number component samples.
x=np.array([-2.,-1.,0.,1.,2.,3.,4.,5.,6.,7.]);distance=float(np.mean(abs(x[:,None]-x[None,:])))
B=np.full((3,3),distance);adjusted=B.copy();np.fill_diagonal(adjusted,distance*len(x)/(len(x)-1));v=np.array([1.,-1.,0.])/math.sqrt(2)
assert abs(v@B@v)<1e-12 and v@adjusted@v>0
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Independent Gaussian finite-ensemble analytic illustration plus dependent-component convexity counterexample; no panel forecast or selection result','primary_reference':'https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf','expected_empirical_CRPS_formula':'sqrt(2/pi)*sqrt(1+sigma^2) - (1-1/M)*sigma/sqrt(pi)','assumptions':['Independent Gaussian ensemble members and verifying standard normal outcome for the analytic example','Known mean zero; only forecast standard deviation varies','These assumptions do not describe all antithetic/stratified/shared draws in the actual option bank'],'records':records,'naive_diagonal_correction_counterexample':{'identical_common_samples':True,'original_tangent_quadratic':float(v@B@v),'adjusted_tangent_quadratic':float(v@adjusted@v),'positive_value_invalidates_conditional_negative_semidefiniteness':True},'interpretation':'Small counts can change expected rankings as well as add noise. The correct actual-bank adjustment needs a dependence-aware argument; do not apply a naive fair-score diagonal multiplier to the certified convex mixture.','single_main_score_unchanged':True,'diagnostic_weight_in_score_or_selection':0,'no_internal_correction_admitted':True,'no_new_forecast_pool':True}
(O/'internal_precision_bias_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
