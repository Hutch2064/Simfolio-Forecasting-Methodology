# Next accuracy/efficiency experiment: per-lead matrix-free CRPS learning

Status: researched proposal, not implemented/admitted or a new winner. The active goal remains unchanged and uncompleted.

## Evidence and purpose

The fast training-size-spaced learner completed355matched requests with0native/scoring failures. Its best mainQ0.7072403547768923is11.63938848031153%better than the fixed with-replacement baseline. The preserved slower endpoint selector achieved0.6901704242531206,13.772057366485136%better on those same355requests. Do not promote the faster model as an accuracy winner. The measured prefix-fitting problem is addressed by the new spacing, but accuracy and some long-request cost remain open. Existing three-case profiles also show substantial full pairwise CRPS-matrix cost. The existing matrix-free prototype updates only once per historical fold and still repeats unbatched component simulations; it has not established field superiority.

## Candidate mechanism

Retain the same financial recipes, history-only component fits, fast grouped/IID backend, nonoverlapping information-spaced internal origins, coverage-count weights and final1000paths. Replace the dense quadratic learner with scale-free simplex updates after each known historical lead. For its positive prescribed mass a_t, component forecasts X_i and current mixture w, the empirical CRPS gradient is

g_i=a_t[mean|X_i-y|-sum_j w_j mean|X_i-X_j|].

Evaluate the cross term with the qualified weighted-CDF reducer, querying every valid recipe, including zero-weight ones, without constructing the full166by166matrix. Centering the gradient leaves simplex optimization unchanged. Accumulate G=sum(g)and S=sqrt(sum||g||^2), then project -G/Sonto the simplex. Zero-gradient initialization is the regularizer's uniform solution, not a fixed financial blend or automatic winner fallback. This is a proposed concrete use of SOLO-FTRL; theorem assumptions and parameterization must be checked, not copied as a financial guarantee.

Compare final iterate versus the normalized positive-mass average of pre-outcome iterates. The average is learned through the data/optimization path, not a fixed percentage split among financial models. Keep diagnostics out of selection. Per-lead updates differ from the earlier per-fold prototype; thousands of available historical gradients may behave differently from two or ten updates. This changes the learner and requires matched score evidence; it is not an exact offline-solver parity claim.

Use nonoverlapping origins first so each previous fold finishes before the next fit date. At each update the component prediction was generated before its target observation. The final user forecast freezes the learned mixture across its complete asset paths. Overlapping-fold online updates require a separately qualified chronological event order and are not silently admitted.

## Reuse and memory

Fit all selected historical prefixes once, establish the valid-option intersection, and reuse the immutable fitted bundles across internal counts/modes. Under geometric training-size spacing, the sum of retained prefix lengths is bounded by a constant times the input history; this supports a per-request linear-size cache rather than retaining hundreds of full histories. Release it after the request; no persistent cache or arbitrary capacity cap. Check actual unique NumPy storage and that simulations never mutate the fitted bundles. Share the full-history fit across variants when appropriate, disclosing fit failures. No recorded outer losses enter fitting or weights.

Do not store a166weight vector before/after every lead. Keep aggregate state, final weights and compact fold checkpoints; otherwise trace objects alone can dominate runtime and memory. Qualification must compare the compact update to the existing state equations and the matrix-free gradient to dense empirical coefficients.

## Qualification and progression

Check simplex/scale invariance, zero-gradient behavior, dense-gradient agreement, causal observation order, cache immutability/reuse, request/seed binding, exact path counts and independently scored checkpoints. Then compare last/averaged versions at4and30internal paths on the current frozen development/confirmation workflow, with one shared12worker pool. Confirm broadly before promotion. No arbitrary production precision, origin cap, hard pruning, financial weight or stopping threshold is introduced.

Other supported open work remains: internal endpoint-versus-full-path objectives under fast validation; flattened/PPW/skew and overlap-origin interactions; low-precision stability; credible missing component families. The fair-CRPS note warns that standard IID finite-ensemble corrections cannot simply be applied to the current stratified antithetic mean draws.

## Primary research checked

Orabona and Pal2018, Scale-Free Online Learning: https://arxiv.org/abs/1601.01974 and author final paper https://david.palenica.com/papers/scale-free-ftrl/Scale-Free-Online-Learning-TCS-final.pdf . The scale-free linear-optimization result motivates the update; it does not establish a forecast-accuracy improvement.

Online-to-batch averaging has statistical guarantees under stated sampling assumptions; financial validation losses are dependent, so no IID generalization or significance claim follows here. The comparison is empirical on the fixed canonical score. See the authors' stochastic-convex-optimization treatment https://www.cs.cornell.edu/~sridharan/convex.pdf .
