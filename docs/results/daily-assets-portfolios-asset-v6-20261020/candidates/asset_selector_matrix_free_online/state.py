"""Same scale-free update with vector observations instead of dense matrices."""
import numpy as np
from candidates.asset_selector_online_pool.online import project_simplex

class State:
    def __init__(self,options):
        if options<1:raise ValueError('Nonempty fixed eligible option universe required')
        self.weights=np.ones(options)/options;self.cumulative=np.zeros(options);self.norm=0.;self.trace=[]
    def observe(self,A,cross):
        a=np.asarray(A,dtype=float);c=np.asarray(cross,dtype=float)
        if a.shape!=self.weights.shape or c.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(c).all():raise ValueError('Finite gradient observation vectors required')
        before=self.weights.copy();gradient=a-c;gradient-=gradient.mean();self.cumulative+=gradient
        self.norm=float(np.hypot(self.norm,np.linalg.norm(gradient)))
        if not np.isfinite(self.cumulative).all() or not np.isfinite(self.norm):raise ArithmeticError('Nonfinite online optimization state')
        self.weights=project_simplex(-self.cumulative/self.norm) if self.norm>0 else np.ones(len(before))/len(before)
        self.trace.append({'fold':len(self.trace),'weights_before_outcome':before.tolist(),'weights_after_outcome':self.weights.tolist(),'observed_mixture_objective':float(a@before-.5*before@c),'gradient_norm_accumulator':self.norm})
        return self.weights.copy()
