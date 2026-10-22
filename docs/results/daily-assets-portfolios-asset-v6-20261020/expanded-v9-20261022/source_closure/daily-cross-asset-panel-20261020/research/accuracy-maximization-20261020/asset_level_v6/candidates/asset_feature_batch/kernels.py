"""Joint-asset filter variants; all paths use identical log-domain accounting."""
import numpy as np
from numba import njit

@njit(cache=True,fastmath=False)
def advance(kind,innovation,pool,indices,shocks,vol_shocks,delta,mu,target,parameters,har_weights,
            variance,ring,phi,previous,h,m,P,target_weights,current_weights,asset_paths,portfolio_paths,start):
    paths,width=indices.shape;assets=len(mu);result=np.empty((paths,width))
    for p in range(paths):
        for t in range(width):
            maximum=-np.inf
            for a in range(assets):
                z=pool[indices[p,t],a] if innovation==0 else shocks[p,t,a]
                if kind==1:
                    weight=parameters[6,a]**(start+t)
                    v=(1-weight)+weight*parameters[5,a]
                    normalized=np.sqrt(v)*z
                elif kind==5:normalized=np.exp(.5*h[p,a])*z
                elif kind==6:normalized=np.exp(.5*m[p,a]+.25*P[p,a])*z
                else:normalized=np.sqrt(variance[p,a])*z
                center=mu[a]+delta[p,a]
                if kind==7 and phi[a]!=0.:
                    value=center+phi[a]*(previous[p,a]-center)+np.sqrt(target[a])*normalized
                else:value=center+np.sqrt(target[a])*normalized
                previous[p,a]=value
                if kind in (2,7):variance[p,a]=1.-parameters[0,a]-parameters[1,a]+parameters[0,a]*normalized*normalized+parameters[1,a]*variance[p,a]
                elif kind==3:
                    coefficient=parameters[3,a] if normalized<0 else parameters[2,a]
                    variance[p,a]=parameters[4,a]+coefficient*normalized*normalized+parameters[1,a]*variance[p,a]
                elif kind==4:
                    pointer=(start+t)%22;ring[p,a,pointer]=normalized*normalized
                    recent=0.;total=0.
                    for j in range(22):total+=ring[p,a,j]
                    for j in range(5):recent+=ring[p,a,(pointer-j)%22]
                    variance[p,a]=1.-har_weights[a].sum()+har_weights[a,0]*ring[p,a,pointer]+har_weights[a,1]*recent/5+har_weights[a,2]*total/22
                elif kind==5:h[p,a]=parameters[7,a]+parameters[8,a]*(h[p,a]-parameters[7,a])+np.sqrt(parameters[9,a])*vol_shocks[p,t,a]
                elif kind==6:
                    m[p,a]=parameters[7,a]+parameters[8,a]*(m[p,a]-parameters[7,a]);P[p,a]=parameters[8,a]**2*P[p,a]+parameters[9,a]
                elif kind==8:variance[p,a]=parameters[6,a]*variance[p,a]+(1-parameters[6,a])*normalized*normalized
                asset_paths[p,a]+=value;current_weights[p,a]=target_weights[a]+value;maximum=max(maximum,current_weights[p,a])
            total=0.
            for a in range(assets):total+=np.exp(current_weights[p,a]-maximum)
            step=maximum+np.log(total);portfolio_paths[p]+=step
            for a in range(assets):current_weights[p,a]-=step
            result[p,t]=portfolio_paths[p]
    return result
