"""One pure IID conditional scale law; no financial stack or old seed helper."""
import hashlib
import json
import numpy as np

DOMAIN='simfolio.research.iid_conditional_scale.v1'
def sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def group_seed(task_seed,origin_date,history):
    if not isinstance(history,bytes):raise TypeError('exact contiguous float64 history bytes required')
    parts=[DOMAIN,int(task_seed),str(origin_date),hashlib.sha256(history).hexdigest()]
    payload=json.dumps(parts,separators=(',',':')).encode()
    seed=int.from_bytes(hashlib.sha256(payload).digest(),'little')
    return seed,parts

def draw_group(variance,simulations,task_seed,origin_date,history):
    v=np.asarray(variance,dtype=np.float64)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all() or np.any(v<0) or simulations<1:
        raise ValueError('finite nonnegative conditional variance and positive simulations required')
    seed,parts=group_seed(task_seed,origin_date,history)
    # Day-major draw order: same seed/history/N gives exact future-H prefixes.
    z=np.random.Generator(np.random.PCG64(seed)).standard_normal((len(v),simulations)).T
    noise_sha=sha(z)
    with np.errstate(over='ignore',under='ignore'):
        log_scale=.5*np.sqrt(v)[None,:]*z-v[None,:]/4.
        scale=np.exp(log_scale)
    if np.isnan(log_scale).any():raise ArithmeticError('undefined IID log scale')
    record={'domain':DOMAIN,'seed':seed,'seed_parts':parts,'bit_generator':'PCG64',
        'draw_order':'day-major, then path; normal-stream H prefix at fixed N/seed/history; scale prefix also requires identical variance prefix',
        'normal_count':int(z.size),'normal_sha256':noise_sha,'log_scale_sha256':sha(log_scale),'scale_sha256':sha(scale),
        'variance_sha256':sha(v),'scale_underflow_count':int((scale==0).sum()),'scale_overflow_count':int(np.isinf(scale).sum())}
    return scale,log_scale,record

def apply(mapped,point_mean,scale,log_scale):
    """Clip(m+(g-m)*s), preserving ordinary direct float64 arithmetic.

    Exceptional exp under/overflow is evaluated using log|g-m|+log(s), not
    a financial exponent cap. Existing final bounds absorb infinite products.
    Exactly unit scales return original mapped bits, including the tau=0 law.
    """
    g=np.asarray(mapped,dtype=np.float64);m=np.asarray(point_mean,dtype=np.float64)
    s=np.asarray(scale,dtype=np.float64);logs=np.asarray(log_scale,dtype=np.float64)
    if g.ndim!=2 or m.shape!=(g.shape[1],) or s.shape!=g.shape or logs.shape!=g.shape:
        raise ValueError('mapped paths and scale shapes must agree')
    if not np.isfinite(g).all() or not np.isfinite(m).all() or np.isnan(s).any() or np.isnan(logs).any() or np.any(s<0):
        raise ValueError('finite mapped paths/point curve and defined nonnegative scales required')
    deviation=g-m[None,:]
    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
        product=deviation*s
        exceptional=(s==0)|np.isinf(s)
        if exceptional.any():
            x=deviation[exceptional];recovered=np.zeros_like(x);nonzero=x!=0
            recovered[nonzero]=np.copysign(np.exp(np.log(np.abs(x[nonzero]))+logs[exceptional][nonzero]),x[nonzero])
            product[exceptional]=recovered
        result=m[None,:]+product
    # This is an algebraic identity, not a fitted near-one tolerance.
    result[s==1]=g[s==1]
    if np.isnan(result).any():raise ArithmeticError('undefined final IID return')
    diagnostics={'lower_final_clip_count':int((result<-1.).sum()),'upper_final_clip_count':int((result>1.).sum()),
        'lower_final_boundary_count':int((result<=-1.).sum()),'upper_final_boundary_count':int((result>=1.).sum()),
        'mapped_lower_boundary_count':int((g<=-1.).sum()),'mapped_upper_boundary_count':int((g>=1.).sum()),
        'path_day_values':int(result.size),'exceptional_scale_recovery_count':int(exceptional.sum()),
        'mapped_mean_horizon_average':float(g.mean()),'mapped_variance_horizon_average':float(g.var(axis=0).mean())}
    np.clip(result,-1.,1.,out=result)
    diagnostics.update(scaled_mean_horizon_average=float(result.mean()),scaled_variance_horizon_average=float(result.var(axis=0).mean()))
    return result,diagnostics
