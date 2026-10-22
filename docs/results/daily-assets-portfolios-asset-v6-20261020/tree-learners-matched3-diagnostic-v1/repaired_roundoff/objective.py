"""Exact grouped mixture CRPS and first-order direction verification."""
import numpy as np
from scipy.special import softmax

class NumericalFailure(ArithmeticError):pass

def risk_gradient(logits,records,matrices,mass,scale):
    value=0.;gradients=[];weights=[]
    for ell,r,b,m in zip(logits,records,matrices,mass):
        v=softmax(ell);bv=b@v;residual=r['A']-bv
        value+=m*float(v@r['A']-.5*v@bv)/scale
        gradients.append(m*v*(residual-float(v@residual))/scale);weights.append(v)
    gradient=np.concatenate(gradients)
    if not np.isfinite(value) or not np.isfinite(gradient).all():raise NumericalFailure('Nonfinite grouped objective/gradient')
    return float(value),gradient,weights

def direction_step(logits,directions,records,matrices,mass,scale,delta):
    value,gradient,_=risk_gradient(logits,records,matrices,mass,scale)
    raw=[np.asarray(d)-np.mean(d) for d in directions]
    raw_direction=np.concatenate(raw)
    if not np.isfinite(raw_direction).all():raise NumericalFailure('Nonfinite tree direction')
    normalization=float(np.max(np.abs(raw_direction)))
    if normalization==0.:return None,{'endpoint':'zero_tree_direction','raw_direction_linf':0.}
    direction=raw_direction/normalization;slope=float(gradient@direction)
    error=64*np.finfo(float).eps*len(direction)*max(1.,float(np.sum(np.abs(gradient*direction))))
    if not np.isfinite(direction).all() or slope>=-error:return None,{'endpoint':'no_resolved_descent_direction','slope':slope,'slope_error':error}
    step=1.;evaluations=0
    while True:
        coefficient=step/normalization
        if not np.isfinite(coefficient):raise NumericalFailure('Unrepresentable effective tree coefficient')
        proposed=[ell+coefficient*d for ell,d in zip(logits,raw)]
        if all(np.array_equal(x,y) for x,y in zip(proposed,logits)):
            return None,{'endpoint':'unrepresentable_step','slope':slope,'step':step,'evaluations':evaluations}
        trial,_,_=risk_gradient(proposed,records,matrices,mass,scale);evaluations+=1
        if trial<=value+1e-4*step*slope and trial<value-delta:
            return proposed,{'step':step,'effective_tree_coefficient':coefficient,'raw_direction_linf':normalization,'slope':slope,'objective_before':value,'objective_after':trial,'evaluations':evaluations}
        # Once the greatest possible first-order decrease is unresolved, further
        # halving cannot qualify this stage. Existing stages remain finite models.
        if -step*slope<=delta:
            return None,{'endpoint':'arithmetic_resolution','slope':slope,'step':step,'evaluations':evaluations}
        step*=.5
