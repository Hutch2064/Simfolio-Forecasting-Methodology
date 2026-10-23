"""Push forward the existing weighted representatives; exact integer feasibility."""
from collections import OrderedDict
from numbers import Integral

def map_folds(rows,horizon,minimum,folds,represented_count):
    integers=(rows,horizon,minimum,represented_count)
    if any(not isinstance(v,Integral)or isinstance(v,bool)for v in integers)or min(integers)<1:raise ValueError('Positive integer geometry required')
    N,H,m=map(int,(rows,horizon,minimum))
    if N<=m:raise ArithmeticError('Unsupported geometry: no positive validation interval beyond minimum support')
    if not folds or sum(t[2]for t in folds)!=represented_count:raise ValueError('Exact existing represented mass required')
    if any(len(t)!=3 or any(not isinstance(v,Integral)or isinstance(v,bool)for v in t)or t[0]<m or t[1]<1 or t[2]<1 or t[0]+t[1]>N for t in folds):raise ValueError('Completed positive integer existing folds required')
    if any(a[0]>=b[0]for a,b in zip(folds,folds[1:])):raise ValueError('Strictly chronological existing prefixes required')
    attainable=H*m<=N*(N-m)
    numerator,denominator=(H,N)if attainable else(N-m,m)
    upper=N*denominator//(denominator+numerator)
    mapped=OrderedDict()
    for stop,h,mass in folds:
        n=min(int(stop),upper);lead=(n*numerator+denominator-1)//denominator
        if n<m or lead<1 or n+lead>N:raise ArithmeticError('Uncompleted projected geometry; no fallback')
        mapped[(n,lead)]=mapped.get((n,lead),0)+int(mass)
    result=[(n,h,mass)for(n,h),mass in mapped.items()]
    if sum(t[2]for t in result)!=represented_count or len(result)>len(folds):raise ArithmeticError('Geometry changed represented mass or increased task count')
    return result,{'requested_ratio_numerator':H,'requested_ratio_denominator':N,'used_ratio_numerator':numerator,'used_ratio_denominator':denominator,'requested_ratio_preserved':attainable,'ratio_flag_scope':'Target rational retained; realized ceil(h)/n may exceed it by less than 1/n','requested_ratio_unattainable':not attainable,'support_projection_applied':not attainable,'minimum_support':m,'maximum_feasible_prefix':upper,'old_selected_origins':len(folds),'mapped_selected_origins':len(result),'collapsed_representatives':len(folds)-len(result),'represented_origins':int(represented_count),'one_episode_despite_mass':len(result)==1,'origin_measure':'pushforward of existing weighted representatives, not all feasible daily origins'}
