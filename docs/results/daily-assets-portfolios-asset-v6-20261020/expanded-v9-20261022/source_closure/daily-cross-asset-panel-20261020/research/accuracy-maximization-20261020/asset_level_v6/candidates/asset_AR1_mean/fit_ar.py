"""Training-only uncertainty-shrunk, stationary AR1 and innovation fit."""
import numpy as np
from candidates.asset_prior_best import fit

def coefficient(history):
    x=np.asarray(history,dtype=float);y=x-x.mean();lag=y[:-1];target=y[1:];den=float(lag@lag)
    if den<=0:raise ValueError('Positive training variance required')
    phi=float(lag@target/den);score=lag*(target-phi*lag);score-=score.mean()
    bandwidth=int(len(score)**(1/3));variance=float(score@score)
    for k in range(1,bandwidth+1):variance+=2*(1-k/(bandwidth+1))*float(score[k:]@score[:-k])
    if variance< -1e-12*float(score@score):raise ArithmeticError('Invalid HAC variance')
    se=np.sqrt(max(0.,variance))/den
    if phi==0:return 0.
    weight=1/(1+(se/phi)**2) if abs(phi)>=se else (phi/se)**2/(1+(phi/se)**2)
    return float(np.clip(phi*weight,-np.nextafter(1.,0.),np.nextafter(1.,0.)))

def innovation_fit(history,parent):
    x=np.asarray(history,dtype=float);phi=coefficient(x)
    if phi==0:return phi,parent
    y=x-x.mean();eps=y.copy();eps[1:]-=phi*y[:-1]
    return phi,fit.fit_variance_target(eps)
