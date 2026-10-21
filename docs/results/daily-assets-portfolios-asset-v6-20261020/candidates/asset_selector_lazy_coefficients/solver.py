"""Fully corrective simplex search with a certificate over the entire menu."""
import numpy as np
from candidates.asset_selector_universe_certified.energy import fit as dense_fit

def fit(A,diagonal,column):
    """column(j) returns B[:,j]; no unrequested matrix entry is needed.

    A_i=E|X_i-y| and B_ij=E|X_i-X_j|, with positive accumulated lead masses.
    The triangle inequality bounds every B_ij by A_i+A_j, so 2*max(A)
    supplies a scale without first constructing the full matrix. All options
    participate in every global-gradient certificate. This does not cap the
    number of admitted components or iterations.
    """
    A=np.asarray(A,dtype=float);diagonal=np.asarray(diagonal,dtype=float);n=len(A)
    if A.ndim!=1 or n<1 or diagonal.shape!=A.shape or not np.isfinite(A).all() or not np.isfinite(diagonal).all() or np.any(A<0):
        raise ValueError('Finite absolute-distance observation and diagonal coefficients required')
    scale=max(2*float(A.max()),float(np.max(np.abs(diagonal))),np.finfo(float).tiny)
    tolerance=np.sqrt(np.finfo(float).eps)*n
    active=[int(np.argmin(A-.5*diagonal))];cache={};history=[]
    while True:
        for j in active:
            if j not in cache:
                values=np.asarray(column(j),dtype=float).copy()
                if values.shape!=(n,) or not np.isfinite(values).all():raise ArithmeticError('Invalid coefficient column')
                if not np.isclose(values[j],diagonal[j],rtol=64*np.finfo(float).eps,atol=np.finfo(float).tiny):raise ArithmeticError('Column differs from frozen diagonal')
                cache[j]=values
        C=np.column_stack([cache[j] for j in active]);restricted=C[active,:]
        if not np.allclose(restricted,restricted.T,rtol=64*np.finfo(float).eps,atol=np.finfo(float).tiny):raise ArithmeticError('Inconsistent coefficient columns')
        solution=dense_fit(A[active],restricted);local=solution['weights'];gradient=A-C@local
        gap=float(local@gradient[active]-gradient.min());j=int(np.argmin(gradient))
        value=float(local@A[active]-.5*local@restricted@local)
        history.append({'active_options':list(active),'global_scaled_gap':gap/scale,'objective':value})
        if gap/scale<=tolerance:
            weights=np.zeros(n);weights[active]=local
            return {'weights':weights,'objective':value,'KKT_gap':gap,'scaled_KKT_gap':gap/scale,'scale':scale,'tolerance':tolerance,'requested_columns':list(cache),'search_trace':history,'all_options_checked_in_global_certificate':True,'full_pair_matrix_constructed':False,'total_iteration_or_component_cap':None}
        if j in active:raise ArithmeticError('Restricted optimizer did not establish a global certificate')
        active.append(j)
