"""Allocation-free small-matrix cDCC likelihood; the statistical law is unchanged."""
import numpy as np
from numba import njit


@njit(cache=True)
def evaluate(z,s,a,b,q0,nu,constant,derivatives=True):
    p=len(s);q=q0.copy();da=np.zeros_like(q);db=np.zeros_like(q)
    root=np.zeros_like(q);inverse_root=np.zeros_like(q);inv=np.zeros_like(q)
    sd=np.empty(p);w=np.empty(p);v=np.empty(p);wa=np.empty(p);wb=np.empty(p)
    loss=-len(z)*constant;gradient=np.zeros(2)
    for row in z:
        logdet=0.
        for i in range(p):
            sd[i]=np.sqrt(q[i,i]);w[i]=sd[i]*row[i]
            for j in range(i+1):
                value=q[i,j]
                for k in range(j):value-=root[i,k]*root[j,k]
                if i==j:
                    if value<=0:raise np.linalg.LinAlgError('nonpositive cDCC covariance')
                    root[i,j]=np.sqrt(value)
                else:root[i,j]=value/root[j,j]
            logdet+=2*np.log(root[i,i])-np.log(q[i,i])
        maha=0.;marginal=0.
        if derivatives:
            # Only the gradient needs the full inverse. Finite-difference
            # likelihood calls solve one triangular system for the quadratic.
            for i in range(p):
                for j in range(i+1):
                    value=1. if i==j else 0.
                    for k in range(j,i):value-=root[i,k]*inverse_root[k,j]
                    inverse_root[i,j]=value/root[i,i]
            for i in range(p):
                for j in range(i+1):
                    value=0.
                    for k in range(i,p):value+=inverse_root[k,i]*inverse_root[k,j]
                    inv[i,j]=value;inv[j,i]=value
            for i in range(p):
                value=0.
                for j in range(p):value+=inv[i,j]*w[j]
                v[i]=value;maha+=w[i]*value
        else:
            for i in range(p):
                value=w[i]
                for j in range(i):value-=root[i,j]*v[j]
                v[i]=value/root[i,i];maha+=v[i]*v[i]
        for i in range(p):
            if nu>0:marginal+=np.log1p(row[i]*row[i]/(nu-2))
            else:marginal+=row[i]*row[i]
        weight=(nu+p)/(nu-2+maha) if nu>0 else 1.
        if nu>0:loss+=.5*logdet+.5*(nu+p)*np.log1p(maha/(nu-2))-.5*(nu+1)*marginal
        else:loss+=.5*(logdet+maha-marginal)
        if derivatives:
            for i in range(p):
                wa[i]=.5*row[i]/sd[i]*da[i,i];wb[i]=.5*row[i]/sd[i]*db[i,i]
            for i in range(p):
                for j in range(i+1):
                    g=.5*inv[i,j]-.5*weight*v[i]*v[j]
                    if i==j:g+=-.5/q[i,i]+.5*weight*v[i]*row[i]/sd[i]
                    scale=1. if i==j else 2.
                    gradient[0]+=scale*g*da[i,j];gradient[1]+=scale*g*db[i,j]
        for i in range(p):
            for j in range(i+1):
                outer=w[i]*w[j];old=q[i,j]
                if derivatives:
                    da[i,j]=outer-s[i,j]+a*(wa[i]*w[j]+w[i]*wa[j])+b*da[i,j]
                    db[i,j]=old-s[i,j]+a*(wb[i]*w[j]+w[i]*wb[j])+b*db[i,j]
                    da[j,i]=da[i,j];db[j,i]=db[i,j]
                q[i,j]=(1-a-b)*s[i,j]+a*outer+b*old
                q[j,i]=q[i,j]
    return loss,gradient,q


def gaussian_likelihood(z,s,a,b,q0):
    return evaluate(z,s,a,b,q0,0.,0.)
