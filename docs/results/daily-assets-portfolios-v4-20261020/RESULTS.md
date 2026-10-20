# Completed historical240-path expanded panel, v4

This initial run attempted all320,245 required tasks across1,264 series (722individual and542portfolio streams). It saved320,215 successful forecasts and30 unchanged native FAST fitting failures; scoring integrity errors were zero. The new human-authorized1,000-path benchmark removes FAST and resets the three Public comparators plus current best. These240-path scores are historical, not new baselines.

The sole main score is training-SD/sqrt-h normalized CRPS with calendar-log horizon weights and the portfolio branch capped at50%. Four models completed identical64,049-task coverage. Relative to the worst Public comparator (Adaptive):

| Model | Main Q | Better than worst Public |
|---|---:|---:|
| Equal Mix / Shrunk Mean / Gaussian Uncertainty |0.6153976193202242|1.3002180169%|
| Public Random Blocks (Replacement) |0.6225565691107722|0.1520387563%|
| Public Random Blocks (History Cycles) |0.6234750216759954|0.0047338836%|
| Public Adaptive Blocks |0.6235045376550531|0%|

FAST is separate:64,019 matched successful tasks,30 native failures. Its paired preliminary score is0.623936560373977 versus Public0.6239890451641782, or0.0084111717% better. This is not a full-coverage rank or repaired-fit result. FAST is removed from future testing by human instruction.

All184,048 common original722 task-loss hashes agree exactly with prior executions. Ten protocol/index tests and the270-forecast initial smoke passed. The complete preserved campaign was freshly downloaded and physically restored per file, SHA256 verified:326,640files, including all checkpoint losses, manifests, errors, source, inputs and closed index. No independence, significance or controlled production speedup is claimed.

See archive-receipt.json for private repository release parts, archive/manifest SHA256 values and verified recovery scope. Download each part with `gh release download research-portfolio-v4-initial-20261020 --repo Hutch2064/simfolio-engine --pattern '<part>'`, verify both archive and manifest digests, extract at the research owner root, and verify every manifest member. Restore the original source dependency archives referenced by each execution manifest. Scientific raw losses can be retired locally only after this recovery evidence and catalogue are remotely preserved.
