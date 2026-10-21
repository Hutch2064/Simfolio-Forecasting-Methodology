# Internal precision and validation design: verified development evidence

All2736candidate forecasts and144references independently audited;72established30path controls replayed byte-exactly. Final forecasts1000paths;72adaptively reused development requests. No official/full-panel promotion or independence claim.

| Schedule | Inner paths | Pool | Seeds | Mean mainQ | Mean better vs baseline |
|---|---:|---|---:|---:|---:|
|established|2|select|3|0.812910606523|15.849515%|
|established|2|stack|3|0.764874955832|20.822047%|
|established|4|select|3|0.799628522488|17.224444%|
|established|4|stack|3|0.759668407862|21.361015%|
|established|8|select|3|0.804363031698|16.734339%|
|established|8|stack|3|0.760460618382|21.279008%|
|established|16|select|3|0.801300477603|17.051367%|
|established|16|stack|3|0.761932486366|21.126644%|
|established|30|select|1|0.796133942926|17.586194%|
|established|30|stack|1|0.755672500385|21.774662%|
|overlap_midpoints|8|select|1|0.794406445365|17.765020%|
|overlap_midpoints|8|stack|1|0.762417104188|21.076477%|
|overlap_midpoints|30|select|1|0.809218777261|16.231684%|
|overlap_midpoints|30|stack|1|0.745068743146|22.872336%|
|support_balanced|8|select|1|0.848167204183|12.199840%|
|support_balanced|8|stack|1|0.768563447825|20.440223%|
|support_balanced|30|select|1|0.856213992128|11.366857%|
|support_balanced|30|stack|1|0.773470765971|19.932230%|
|target_first|8|select|1|0.927133782028|4.025416%|
|target_first|8|stack|1|0.909115163104|5.890659%|
|target_first|30|select|1|0.925765682419|4.167039%|
|target_first|30|stack|1|0.908531220195|5.951107%|

Stacking outperformed selecting one in this grid. Lower precision4paths retains much of the observed benefit at lower measured process CPU, but the three-seed mean trails the prior30path three-seed mean. Overlap-midpoint30paths improves its one-seed control and warrants repetition/broader confirmation. Support-balanced and target-first versions lose their matched controls; a short initial fit can be statistically inadequate even though every observation is past the request date. Do not infer universal failure of different horizon-adaptation mechanisms.

CPU accounting is attributed shared preparation plus a standalone final forecast, scoring/compression/hashing, including warmup. It excludes blocked IO wait and is not production latency or an all-request10second certificate. Largest-case tail imbalance in the pilot motivates runtime-only largest-work-first scheduling in the next stage; no model selection or scoring uses runtime.

No fixed production simulation count, winner, recipe-pruning rule or stopping conclusion is selected by this study. Preserve the original panel and outer score. Continue broader tests and supported component interactions.
