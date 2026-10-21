"""Add causal calendar binding; delegate all preparation/simulation/CRPS unchanged."""
from time import perf_counter
import hashlib
import numpy as np
from asset_forecast import forecast as original_forecast

def forecast(model,panel_root,row,values,dates,origin,seed):
    bind=getattr(model,'with_training_dates',None)
    if bind is None:
        return original_forecast(model,panel_root,row,values,dates,origin,seed)
    started=perf_counter();stop=origin.position+1;past=np.asarray(dates[:stop],dtype='datetime64[D]')
    if len(past)!=stop:raise ValueError('Incomplete historical calendar prefix')
    request=bind(past)
    if request.model_id!=model.model_id:raise ValueError('Request binding changed model identity')
    losses,diagnostics=original_forecast(request,panel_root,row,values,dates,origin,seed)
    return losses,{**diagnostics,'forecast_and_score_seconds':perf_counter()-started,'historical_calendar_prefix_bound':True,'training_date_rows':len(past),'training_dates_sha256':hashlib.sha256(past.tobytes()).hexdigest(),'training_first_date':str(past[0]),'training_last_date':str(past[-1]),'request_model_reused_from_another_portfolio':False}
