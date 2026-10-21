# Certified numerical solution of the same convex CRPS objective

The initial precision run rejected a fitted simplex solution at case49 solely because SLSQP reported its1000iterationlimit, despite a scaled convex optimality gap of8.4e-12. Keep that failed attempt visible. Do not change its running sources or drop its outer cases.

This revision uses the same empirical CRPS objective, initial vertex, SLSQP iterations, constraints, tolerances, componentbank and data. Acceptance depends on the global simplex KKT certificate and objective comparison to the best vertex, rather than requiring the solver success flag in addition. An uncertified iteration-limit result continues from its current state with a doubled solver chunk; there is no total iteration cap or fixed model fallback. Other numerical failures remain explicit. Existing successful solutions must replay bitwise; certified recovery of failed cases needs fresh actual-source validation. A certificate refers to inner empirical optimization, never outer forecast superiority.

This is a separate source-bound numerical repair. Neither its tests nor any repaired panel score is claimed until actually executed. The running initial precision attempt must close and be preserved; checkpoint reuse requires source-compatibility, identical request/seed/reference and per-file hash verification, with the old failed logs retained.
