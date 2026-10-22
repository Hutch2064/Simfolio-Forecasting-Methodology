"""Original grouped path operations; exact fast normal-quantile draws only."""
import hashlib
import numpy as np
from candidates.asset_feature_batch import fits
from candidates.asset_fast_selector.bank import seed
from candidates.asset_selector_universe.bank import BANK
from candidates.asset_selector_menu_components.bank import model
from candidates.asset_selector_batched_paths.kernels import advance
from candidates.asset_mean_draws_v1.mean_draws import draws

def group_chunks(history,weights,horizon,paths,context_seed,indices,bundle,*,width=64,draw_cache=None):
    x=np.asarray(history,dtype=float);w=np.asarray(weights,dtype=float);specs=[BANK[i]for i in indices]
    family=specs[0]['family'];law=specs[0]['innovations'];assets=x.shape[1];count_options=len(specs)
    assert all(s['kind']=='filtered'and s['family']==family and s['innovations']==law for s in specs)
    assert bundle['family']==family and bundle['history_sha256']==hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    raw,f=bundle['raw'],bundle['filter'];deltas=[]
    for spec in specs:
        batch={**bundle['mean']}
        if spec['mean_mode']=='sample':batch['center']=batch['sample_mean'].copy()
        elif spec['mean_mode']=='zero':batch['center']=np.zeros_like(batch['sample_mean'])
        bias=batch['center']-batch['sample_mean']
        deltas.append(cached_draws(batch,paths,seed('FHS_joint_mean_uncertainty_v1',context_seed),draw_cache)if spec['mean_uncertainty']else np.tile(bias,(paths,1)))
    delta=np.stack(deltas);row_rng=np.random.default_rng(seed('joint_FHS_asset_rows_v1',context_seed));normal_rng=np.random.default_rng(seed('joint_feature_Gaussian_v1',context_seed));chi_rng=np.random.default_rng(seed('joint_feature_radial_v1',context_seed));vol_rng=np.random.default_rng(seed('joint_feature_vol_v1',context_seed))
    target=np.full(assets,-np.inf);target[w>0]=np.log(w[w>0]);current=np.tile(target,(count_options,paths,1));asset=np.zeros((count_options,paths,assets));portfolio=np.zeros((count_options,paths));previous=np.tile(x[-1],(count_options,paths,1))
    parameters=np.stack([f[k]for k in('alpha','beta','positive','negative','omega','initial','decay','level','sv_phi','q')])
    if family=='GJR'and law!='empirical':
        persistence=parameters[1]+.5*(parameters[2]+parameters[3]);factor=np.minimum(1.,(1.-1/len(x))/np.maximum(persistence,np.finfo(float).tiny));parameters[1]*=factor;parameters[2]*=factor;parameters[3]*=factor;parameters[4]=1.-parameters[1]-.5*(parameters[2]+parameters[3])
    variance=np.tile(f['initial'],(paths,1));ring=np.tile(f['ring'],(paths,1,1));next_m=f['level']+f['sv_phi']*(f['m']-f['level']);next_P=f['sv_phi']**2*f['P']+f['q'];m=np.tile(next_m,(paths,1));P=np.tile(next_P,(paths,1))
    initial_rng=np.random.default_rng(seed('joint_feature_vol_initial_v1',context_seed));h=m+(initial_rng.normal(size=(paths,assets))@f['vol_root'].T)*np.sqrt(next_P)
    kind=fits.FAMILIES.index(family);innovation=('empirical','Gaussian','Student').index(law)
    for start in range(0,horizon,width):
        count=min(width,horizon-start);rows=row_rng.integers(0,len(x),size=(count,paths)).T.copy();normal=np.zeros((1,1,1));vol=np.zeros((1,1,1))
        if innovation:
            normal=normal_rng.normal(size=(count,paths,assets))@f['root'].T
            if innovation==2 and f['inverse_df']>0:
                nu=1/f['inverse_df'];normal*=np.sqrt((nu-2)/chi_rng.chisquare(nu,size=(count,paths)))[...,None]
            normal=np.ascontiguousarray(normal.transpose(1,0,2))
        if kind==5:vol=np.ascontiguousarray((vol_rng.normal(size=(count,paths,assets))@f['vol_root'].T).transpose(1,0,2))
        result=advance(kind,innovation,f['pool'],rows,normal,vol,delta,raw['mean'],f['target'],parameters,f['weights'],variance,ring,f['phi'],previous,h,m,P,target,current,asset,portfolio,start)
        if not np.isfinite(result).all()or not np.isfinite(variance).all():raise ArithmeticError('Nonfinite grouped forecast; no path exclusions')
        yield start,result

def filtered_all_chunks(history, weights, horizon, paths, context_seed, current, prepared, *, width=64):
    draw_cache={'arrays':{},'bytes':0,'limit':8*1024**2}
    current = list(current)
    if not current or len(set(current)) != len(current) or any(i < 0 or i >= len(BANK) for i in current) or width < 1:
        raise ValueError('Unique valid options and positive chunk width required')
    positions = {index: offset for offset, index in enumerate(current)}; groups = {}; generators = []
    for index in current:
        spec = BANK[index]
        if spec['kind'] == 'filtered':
            groups.setdefault((spec['family'], spec['innovations']), []).append(index)
        else:
            generators.append(([index], model(index, prepared).terminal_chunks(history, weights, horizon, paths, context_seed, width=width), False))
    for (family, law), ids in groups.items():
        generators.append((ids, group_chunks(history, weights, horizon, paths, context_seed, ids, prepared['families'][family], width=width,draw_cache=draw_cache), True))
    for start in range(0, horizon, width):
        count = min(width, horizon-start); cube = np.empty((len(current), paths, count))
        for ids, generator, grouped in generators:
            position, values = next(generator)
            if not grouped:
                values = values[None, :, :]
            if position != start or values.shape != (len(ids), paths, count) or not np.isfinite(values).all():
                raise ArithmeticError('Incomplete grouped predictive path')
            for offset, index in enumerate(ids):
                cube[positions[index]] = values[offset]
        yield start, cube

def all_chunks(history, weights, horizon, paths, seed, valid, prepared, *, width=64):
    original = [i for i in valid if i < 166]
    positions = {i: j for j, i in enumerate(valid)}
    generators = [(i, model(i, prepared).terminal_chunks(history, weights, horizon, paths, seed, width=width))
                  for i in valid if i >= 166]
    original_chunks = filtered_all_chunks(history, weights, horizon, paths, seed, original, prepared, width=width) if original else None
    for start in range(0, horizon, width):
        count = min(width, horizon-start); cube = np.empty((len(valid), paths, count))
        if original_chunks is not None:
            position, values = next(original_chunks)
            if position != start:
                raise ArithmeticError('Original menu path cursor differs')
            for j, i in enumerate(original):
                cube[positions[i]] = values[j]
        for i, generator in generators:
            position, values = next(generator)
            if position != start or values.shape != (paths, count) or not np.isfinite(values).all():
                raise ArithmeticError('New component path is incomplete')
            cube[positions[i]] = values
        yield start, cube


def cached_draws(batch,paths,seed,cache):
    """Exact completed draw memo within one immutable-prefix cube generator."""
    if cache is None:return draws(batch,paths,seed)
    center=np.asarray(batch['center'])
    key=(id(batch['root']),id(batch['sample_mean']),center.dtype.str,center.shape,center.tobytes(),paths,seed)
    if key in cache['arrays']:return cache['arrays'][key]
    value=draws(batch,paths,seed)
    if cache['bytes']+value.nbytes<=cache['limit']:
        cache['arrays'][key]=value;cache['bytes']+=value.nbytes
    return value
