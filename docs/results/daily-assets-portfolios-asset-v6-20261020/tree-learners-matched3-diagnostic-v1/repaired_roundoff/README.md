# Pruning-alpha roundoff repair only

Isolated sibling of frozen asset_tree_learners_v1. objective.py and original8-test algorithm suite are exactcopies; model implementation differs only in pruning-alpha canonicalization/diagnostics. No speed feature, financial clipping, optimizer tolerance or tree architecture change.

sklearn returned ccp_alpha=-2.0543252740130515e-33 for earliest MVDcase0 training. CART/RF/ExtraTrees reject any negative library parameter, so all3cells failed explicitly. Repair only these3cells; separately run geometrycase0CART exact numerical/path parity. Retain all27successful oldcells and old3failures.

For a path containing negative entries, normalize positive training row masses p; compute mean=sum(p*y), M2=sum(p*y²), scale=M2+mean² and bound=64*float64eps*nrows*scale. Penalty alpha has response² impurity units. This is a conservative declared floating arithmetic envelope for cancellation; it is not a formal bound on every sklearn accumulation. Canonicalize only finite negative alpha with abs(alpha)<=bound to exact +0. Reject material negatives/nonfinite path/scale/mass. If scale/bound underflows to0, any negative fails. No unitfloor. Preserve raw/canonicalpath, bound, scale, repairedcount.

All finite nonnegative path values return bitexact before evaluating the repair envelope, including signedzero/subnormal values. Thus retained successful control candidatepaths are unchanged. positive_path_compatibility_receipt.json audits their saved candidate_labels without refitting; actual geometrycase0CART weights/counts/kernel/loss parity still requires root admission. No mixedgeneration scoreclaim without that qualification; new repaired cells explicitly identify this numerical generation.

New tests cover tinynegative→positivezero, materialnegative/nonfinite rejection, zerobound failure and nonnegative path bitidentity. Existing algorithm/controller tests remain required. Root owns all financialtest launches; no new inner/probe simulations or othermethod reruns. Dependency/runtime, input, sourceclosure and final100 v7 screen contracts are inherited from the frozen treepilot.
