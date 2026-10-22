"""Exact quadratic active-face solver; unchanged full-menu CRPS certificate.

Only the numerical optimizer changes. Degenerate/failed fast solves delegate to
the unchanged certified reference optimizer, never to a different forecast law.
Equal objectives do not prove identical weights, allocations or forecast paths.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from candidates.asset_selector_universe_certified import energy as reference
coefficients=reference.coefficients
objective=reference.objective
even_allocation=reference.even_allocation


def certify(w,a,b,precision,tolerance,best_vertex):
    if not np.isfinite(w).all() or np.min(w)<0 or abs(float(w.sum())-1)>precision:return None
    gradient=a-b@w;gap=float(w@gradient-gradient.min());value=objective(w,a,b)
    if not np.isfinite(gradient).all() or not np.isfinite(value) or not np.isfinite(gap) or gap < -precision:return None
    if gap<=tolerance and value<=best_vertex+precision:return {'gap':gap,'value':value}
    return None


def fit(A,B):
    A=np.asarray(A,dtype=np.float64);B=np.asarray(B,dtype=np.float64);k=len(A)
    if A.ndim!=1 or k<1 or B.shape!=(k,k) or not np.isfinite(A).all() or not np.isfinite(B).all():raise ValueError('Finite mixture coefficients required')
    if not np.array_equal(B,B.T):raise ValueError('Exactly symmetric CRPS coefficients required')
    scale=max(float(np.max(np.abs(A))),float(np.max(np.abs(B))),np.finfo(float).tiny)
    a=A/scale;b=B/scale;vertices=a-.5*np.diag(b);best=int(np.argmin(vertices));w=np.zeros(k);w[best]=1.
    eps=np.finfo(float).eps;precision=64*eps*k;reference_tolerance=np.sqrt(eps)*k
    # Tighter fast-phase target; never weaken the reference global acceptance.
    target=precision;active=[best];iterations=0;reason=None
    if k==1:return {**reference.fit(A,B),'solver':'active_face_quadratic_v1','reference_fallback':False}
    centered=b-b.mean(axis=0)[None,:]-b.mean(axis=1)[:,None]+b.mean()
    eigenvalues=np.linalg.eigvalsh(-.5*(centered+centered.T))
    spectral_tolerance=64*eps*k*max(float(np.max(np.abs(eigenvalues))),np.finfo(float).tiny)
    if float(eigenvalues.min()) < -spectral_tolerance:raise ArithmeticError('CRPS Hessian is not PSD on the simplex tangent')
    rank=int(np.sum(eigenvalues>spectral_tolerance))
    defect=max(0.,-float(eigenvalues.min()))
    def legacy(reason,iterations):
        fitted=reference.fit(A,B);accepted=certify(fitted['weights'],a,b,precision,reference_tolerance,float(vertices[best]))
        if accepted is None:raise ArithmeticError('Reference fallback failed unchanged global certificate')
        return {**fitted,'solver':'active_face_quadratic_v1','reference_fallback':True,'fallback_reason':reason,
            'fast_phase_iterations':iterations,'all_options_checked_in_global_certificate':True,
            'tangent_numerical_rank':rank,'nonunique_geometry_possible':rank<k-1,
            'tangent_CND_roundoff_defect':defect,'reference_tolerance':reference_tolerance,'accepted_tolerance':reference_tolerance}
    if rank<k-1:return legacy('Numerically nonunique tangent geometry; preserve legacy allocation',0)
    # This is a fast-phase watchdog, not accepted iteration truncation: any
    # numerical failure/degeneracy delegates to the unchanged unbounded solver.
    for iterations in range(1,max(100,4*k*k)+1):
        certificate=certify(w,a,b,precision,target,float(vertices[best]))
        if certificate is not None:
            return {'weights':w,'objective':objective(w,A,B),'best_vertex_objective':float(vertices[best]*scale),
                'KKT_gap':certificate['gap']*scale,'scaled_KKT_gap':certificate['gap'],'iterations':iterations-1,
                'solver':'active_face_quadratic_v1','reference_fallback':False,'all_options_checked_in_global_certificate':True,
                'scale':scale,'reference_tolerance':reference_tolerance,'accepted_tolerance':target,
                'tangent_numerical_rank':rank,'nonunique_geometry_possible':False,'tangent_CND_roundoff_defect':defect,
                'precision_scope':'Original CRPS coefficients/objective; full-menu certificate tighter than reference. Weight/path parity requires separate qualification.'}
        gradient=a-b@w;enter=int(np.argmin(gradient))
        if enter not in active:active.append(enter)
        elif len(active)==1:reason='Single-vertex numerical certificate failure';break
        # Fully correct the current working face using its exact fixed Hessian.
        while True:
            ids=np.array(active);m=len(ids)
            if m==1:w[:]=0.;w[ids[0]]=1.;break
            h=-b[np.ix_(ids,ids)];reduced=h[:-1,:-1]-h[:-1,-1,None]-h[-1,None,:-1]+h[-1,-1]
            g=(a-b@w)[ids];rhs=-(g[:-1]-g[-1])
            try:coordinates=cho_solve(cho_factor(reduced,lower=True,check_finite=False),rhs,check_finite=False)
            except np.linalg.LinAlgError:reason='Singular or numerically indefinite working face';break
            direction=np.r_[coordinates,-float(coordinates.sum())]
            candidate=w[ids]+direction
            if not np.isfinite(candidate).all():reason='Nonfinite working-face solution';break
            if np.min(candidate)>=0:
                old=w.copy();w[:]=0.;w[ids]=candidate;w/=float(w.sum())
                if np.array_equal(w,old):reason='Working-face numerical stagnation'
                break
            falling=np.flatnonzero(direction<0)
            if not len(falling):reason='Infeasible face without a boundary direction';break
            ratios=-w[ids[falling]]/direction[falling];hit_pos=int(np.argmin(ratios));hit=int(falling[hit_pos]);step=float(ratios[hit_pos])
            if not np.isfinite(step) or step<0 or step>1:reason='Invalid boundary step';break
            w[ids]+=step*direction;w[ids[hit]]=0.
            if np.min(w)<0 or not np.isfinite(w).all():reason='Boundary arithmetic left simplex';break
            w/=float(w.sum());active.remove(int(ids[hit]))
        if reason is not None:break
    else:reason='Fast-phase watchdog; no uncertified iterate accepted'
    return legacy(reason,iterations)
