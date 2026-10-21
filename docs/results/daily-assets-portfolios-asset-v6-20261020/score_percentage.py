"""Display-only conversion from the single canonical mainQ; independent of sims."""
import copy,math

def percent_better(model_Q,reference_Q):
    if not math.isfinite(model_Q)or not math.isfinite(reference_Q)or reference_Q<=0:raise ValueError('Finite mainQ and a positive reference required')
    return 100.*(1.-model_Q/reference_Q)

def normalize_model_percentages(report):
    result=copy.deepcopy(report);reference=report['reference_main_Q']['fixed_reference'];changes=[]
    for mid,model in result['models'].items():
        q=model.get('main_Q')
        if q is None:continue
        expected=percent_better(q,reference)
        for key,value in list(model.items()):
            if key.startswith('percent_better_')and key.endswith('fixed_reference')and value is not None:
                if not math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-10):changes.append({'model_id':mid,'field':key,'original':value,'corrected':expected})
                model[key]=expected
    for name,rows in result.items():
        if isinstance(rows,list)and name.endswith('ranking'):
            for row in rows:
                for change in changes:
                    if row.get('model_id')==change['model_id']:row[change['field']]=change['corrected']
    return result,changes
