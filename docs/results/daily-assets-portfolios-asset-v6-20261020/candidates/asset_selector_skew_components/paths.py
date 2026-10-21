"""Shared variance-standardized SAS skew innovations, all mean/uncertainty choices."""
import hashlib
import numpy as np
from candidates.asset_feature_batch import fits
from candidates.asset_fast_selector.bank import seed
from candidates.asset_prior_best import mean
from candidates.asset_SAS_skew.skew import transform
from candidates.asset_selector_batched_paths.kernels import advance

def gjr_parameters(parameters,shape_fits,rows):
    parameters=parameters.copy()
    positive=np.array([f['positive_second_moment']for f in shape_fits]);negative=np.array([f['negative_second_moment']for f in shape_fits])
    def weighted():
        return .5*(parameters[2]+parameters[3])+parameters[2]*(positive-.5)+parameters[3]*(negative-.5)
    persistence=parameters[1]+weighted();factor=np.minimum(1.,(1.-1/rows)/np.maximum(persistence,np.finfo(float).tiny))
    parameters[1]*=factor;parameters[2]*=factor;parameters[3]*=factor;parameters[4]=1.-parameters[1]-weighted()
    return parameters

def group_chunks(history,weights,horizon,paths,context_seed,specs,bundle,*,width=64):
    x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float)
    family=specs[0]['family'];law=specs[0]['innovations'];assets=x.shape[1];count_options=len(specs)
    assert all(s['kind']=='SAS_filtered'and s['family']==family and s['innovations']=='SAS_BIC' for s in specs)
    assert bundle['family']==family and bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    raw,f=bundle['raw'],bundle['filter'];deltas=[]
    for spec in specs:
        batch={**bundle['mean']}
        if spec['mean_mode']=='sample':batch['center']=batch['sample_mean'].copy()
        elif spec['mean_mode']=='zero':batch['center']=np.zeros_like(batch['sample_mean'])
        bias=batch['center']-batch['sample_mean']
        deltas.append(mean.draws(batch,paths,seed('FHS_joint_mean_uncertainty_v1',context_seed))if spec['mean_uncertainty']else np.tile(bias,(paths,1)))
    delta=np.stack(deltas);row_rng=np.random.default_rng(seed('joint_FHS_asset_rows_v1',context_seed));normal_rng=np.random.default_rng(seed('joint_feature_Gaussian_v1',context_seed));chi_rng=np.random.default_rng(seed('joint_feature_radial_v1',context_seed));vol_rng=np.random.default_rng(seed('joint_feature_vol_v1',context_seed))
    target=np.full(assets,-np.inf);target[w>0]=np.log(w[w>0]);current=np.tile(target,(count_options,paths,1));asset=np.zeros((count_options,paths,assets));portfolio=np.zeros((count_options,paths));previous=np.tile(x[-1],(count_options,paths,1))
    parameters=np.stack([f[k]for k in('alpha','beta','positive','negative','omega','initial','decay','level','sv_phi','q')])
    if family=='GJR'and law!='empirical':
        parameters=gjr_parameters(parameters,bundle['SAS_fit'],len(x))
    variance=np.tile(f['initial'],(paths,1));ring=np.tile(f['ring'],(paths,1,1));next_m=f['level']+f['sv_phi']*(f['m']-f['level']);next_P=f['sv_phi']**2*f['P']+f['q'];m=np.tile(next_m,(paths,1));P=np.tile(next_P,(paths,1))
    initial_rng=np.random.default_rng(seed('joint_feature_vol_initial_v1',context_seed));h=m+(initial_rng.normal(size=(paths,assets))@f['vol_root'].T)*np.sqrt(next_P)
    kind=fits.FAMILIES.index(family);innovation=1
    for start in range(0,horizon,width):
        count=min(width,horizon-start);rows=row_rng.integers(0,len(x),size=(count,paths)).T.copy();normal=np.zeros((1,1,1));vol=np.zeros((1,1,1))
        if innovation:
            normal=normal_rng.normal(size=(count,paths,assets))@f['root'].T
            normal=transform(normal,np.array([entry['BIC_t']for entry in bundle['SAS_fit']]))
            normal=np.ascontiguousarray(normal.transpose(1,0,2))
        if kind==5:vol=np.ascontiguousarray((vol_rng.normal(size=(count,paths,assets))@f['vol_root'].T).transpose(1,0,2))
        result=advance(kind,innovation,f['pool'],rows,normal,vol,delta,raw['mean'],f['target'],parameters,f['weights'],variance,ring,f['phi'],previous,h,m,P,target,current,asset,portfolio,start)
        if not np.isfinite(result).all()or not np.isfinite(variance).all():raise ArithmeticError('Nonfinite grouped forecast; no path exclusions')
        yield start,result

