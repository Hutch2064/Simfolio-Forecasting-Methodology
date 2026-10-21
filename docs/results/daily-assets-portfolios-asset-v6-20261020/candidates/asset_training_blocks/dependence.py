"""Stationary joint row bootstrap and its exact conditional mean covariance."""
import numpy as np
from arch.bootstrap import optimal_block_length


def select_length(history):
    x=np.asarray(history,dtype=float)
    if x.ndim!=2 or len(x)<24 or not np.isfinite(x).all():
        raise ValueError('Finite training matrix required')
    centered=x-x.mean(0);sd=np.sqrt(np.mean(centered*centered,axis=0))
    if np.any(sd<=0):raise ValueError('Positive marginal variation required')
    z=centered/sd;features=np.concatenate((z,z*z),axis=1)
    # A genuinely constant square feature has no estimable serial dependence.
    active=np.ptp(features,axis=0)>0
    marginal=np.ones(features.shape[1])
    estimates=optimal_block_length(features[:,active])['stationary'].to_numpy()
    if not np.isfinite(estimates).all() or np.any(estimates<0):
        raise ArithmeticError('Invalid training-selected block length')
    marginal[active]=np.maximum(1.,estimates)
    return {'length':float(marginal.max()),'marginal_lengths':marginal}


class StationaryRows:
    def __init__(self,n,paths,seed,length):
        if n<1 or paths<1 or not np.isfinite(length) or length<1:
            raise ValueError('Valid row count, ensemble and expected block length required')
        self.n=n;self.paths=paths;self.length=length;self.position=0
        # Separate row and restart streams make chunk widths irrelevant.
        children=np.random.SeedSequence(int(seed)).spawn(2)
        self.rows=np.random.default_rng(children[0]);self.restart=np.random.default_rng(children[1])
        self.current=self.rows.integers(0,n,size=paths)

    def take(self,width):
        if width<1:raise ValueError('Positive chunk width required')
        out=np.empty((self.paths,width),dtype=np.int64)
        for t in range(width):
            if self.position:
                candidate=self.rows.integers(0,self.n,size=self.paths)
                restart=self.restart.random(self.paths)<1/self.length
                self.current=np.where(restart,candidate,(self.current+1)%self.n)
            out[:,t]=self.current;self.position+=1
        return out


def mean_covariance(history,length):
    x=np.asarray(history,dtype=float);n=len(x)
    if x.ndim!=2 or n<2 or not np.isfinite(x).all() or not np.isfinite(length) or length<1:
        raise ValueError('Valid training matrix and block length required')
    z=x-x.mean(0);r=1.-1./length;k=np.arange(1,n)
    weights=np.zeros(n);weights[0]=n
    values=(n-k)*r**k
    weights[k]+=values;weights[(-k)%n]+=values
    spectral=np.fft.fft(weights).real;f=np.fft.fft(z,axis=0)
    covariance=(f.real.T@(spectral[:,None]*f.real)+f.imag.T@(spectral[:,None]*f.imag))/float(n)**4
    return (covariance+covariance.T)/2


def mean_fit(history,length):
    x=np.asarray(history,dtype=float);mu=x.mean(0);covariance=mean_covariance(x,length)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    if eigenvalues.min() < -1e-10*max(float(eigenvalues.max()),np.finfo(float).tiny):
        raise ArithmeticError('Mean covariance is not positive semidefinite')
    # Roundoff projection only, not a financial variance floor or multiplier.
    root=eigenvectors*np.sqrt(np.maximum(eigenvalues,0.))
    se2=np.maximum(np.diag(covariance),0.);denominator=mu*mu+se2
    weight=np.divide(mu*mu,denominator,out=np.ones_like(mu),where=denominator>0)
    return {'sample_mean':mu,'center':mu*weight,'covariance':covariance,'root':root}
