# Faster endpoint learner evidence

355reused development requests;1420forecasts verified with1000final paths,0native failures and0scoring errors. Canonical outer score unchanged; only internal learning criterion and eliminated unnecessary pair reductions differ. No full-panel promotion.

| Origin spacing | Internal paths | MainQ | Better vs fixed WR | Median CPU seconds |
|---|---:|---:|---:|---:|
|doubling|4|0.689813598121|13.816638%|0.759093|
|doubling|30|0.690819868105|13.690918%|0.902012|
|half_doubling|4|0.687238574495|14.138354%|0.841320|
|half_doubling|30|0.690081539815|13.783162%|1.022505|

Old slow certified selector on exactly the same355requests scored13.772057%better with1000inner paths. The endpoint4path half-doubling score is14.138354%better. These are different learners; this is not a forecast-parity or statistically significant-superiority claim. Seed confirmation, component interactions and separate cases remain.

Process CPU of shared preparation charged to either standalone output; includes both pooling fits, union refit, final1000path forecast, score, compression and hashing. Excludes blocked IO waiting and final record writing; no production latency or isolated whole-request speedup claim.
