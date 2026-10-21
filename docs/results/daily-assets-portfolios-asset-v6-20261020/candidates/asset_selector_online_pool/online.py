"""SOLO-FTRL on simplex-tangent CRPS gradients; no tuned learning rate."""
import numpy as np

def project_simplex(values):
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('Finite nonempty projection vector required')
    # A common translation does not change Euclidean simplex projection.
    x=x-x.max();ordered=np.sort(x)[::-1];shift=(np.cumsum(ordered)-1)/np.arange(1,len(x)+1)
    active=np.flatnonzero(ordered>shift);theta=shift[active[-1]];w=np.maximum(x-theta,0.)
    return w/w.sum()

def fit_sequence(terms):
    if not terms:raise ValueError('At least one completed historical validation fold required')
    first=np.asarray(terms[0][0],dtype=float);k=len(first)
    if first.ndim!=1 or k<1:raise ValueError('Nonempty fixed option bank required')
    cumulative=np.zeros(k);norm=0.;w=np.ones(k)/k;trace=[]
    for number,(a,b)in enumerate(terms):
        A=np.asarray(a,dtype=float);B=np.asarray(b,dtype=float)
        if A.shape!=(k,)or B.shape!=(k,k)or not np.isfinite(A).all()or not np.isfinite(B).all()or not np.array_equal(B,B.T):raise ValueError('Finite aligned symmetric fold coefficients required')
        before=w.copy();gradient=A-B@before
        # The all-ones direction adds the same scalar on every simplex choice.
        # Projecting it out leaves every feasible directional derivative intact.
        gradient=gradient-gradient.mean();cumulative+=gradient;norm=float(np.hypot(norm,np.linalg.norm(gradient)))
        if not np.isfinite(cumulative).all()or not np.isfinite(norm):raise ArithmeticError('Nonfinite online optimization state')
        w=project_simplex(-cumulative/norm)if norm>0 else np.ones(k)/k
        trace.append({'fold':number,'weights_before_outcome':before.tolist(),'weights_after_outcome':w.tolist(),'observed_mixture_objective':float(A@before-.5*before@B@before),'gradient_norm_accumulator':norm})
    return {'weights':w,'folds':len(terms),'trace':trace,'rule':'SOLO-FTRL; Euclidean simplex; centered CRPS gradients; R(w)=||w||2_squared/2','objective_scope':'Sequential convex-loss update, not a claim to minimize the final batch quadratic','fixed_learning_rate':None,'gradient_clipping':False,'max_folds':None,'automatic_previous_winner_fallback':False,'initial_state':'Minimum-norm simplex point before any score evidence; not a persistent equal-weight blend'}
