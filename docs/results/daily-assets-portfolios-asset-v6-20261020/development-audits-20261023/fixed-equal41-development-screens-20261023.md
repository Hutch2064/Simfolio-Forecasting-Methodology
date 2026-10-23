# Fixed-recipe and equal-weight screens: 12 to 90 requests

These closed screens compare the current G2 uniform-lead 2P candidate with 41 fixed recipes and a static equal-weight blend of those recipes. They are development diagnostics, not a full-panel selection or promotion result.

## The result changed with coverage

The initial 12-request screen completed all 504 candidate arms (41 one-hot recipes plus equal-41, 240 paths each). Equal-41 scored 0.609964 versus G2 at 0.652321; the best fixed recipe, 166, scored 0.586039. The sample had three long, three medium, and six short requests.

The 90-request extension reused those 12 cases and added 78, yielding 90 requests with 30 in each horizon bucket and 3,780 completed candidate arms. On this broader screen, G2 scored 0.476027. Equal-41 scored 0.581698, 22.20% higher CRPS than G2. The best fixed recipe was 123 at 0.574441, 20.67% higher than G2. All 41 fixed recipes and equal-41 scored worse than G2. Thus the promising 12-request result did not persist; it is not a basis for replacing G2.

The score is the V11 request loss: raw mean daily CRPS divided by training-portfolio sample standard deviation and the square root of the actual requested horizon. Request weights are renormalized within short, medium, and long buckets, then the three bucket scores are averaged equally. On the complete 90-request support, WR scored 0.917256 and WOR 0.985189. The worse fixed reference is the larger of the two global aggregate Q values on the same support, so WOR is the worse reference here; the rule is not a per-request maximum. See the [V11 fixed-reference reporting definition](../aligned_panel_v7_fixed_reference_reporting_addendum.json).

The original 90-screen audit recorded 89 WOR references. A separate postcheck verified the single missing saved WOR loss vector and completed the 90-request baseline comparison. It generated no candidate forecasts and did not modify the frozen 90-screen plan or audit. The 90-screen score addendum and its one-row postcheck are bound by the hashes in the companion JSON.

## Interpretation and limits

An earlier matched 90-request native-origin screen also favored G2: native-origin uniform 2P scored 0.510342 against 0.476027 for G2, or 7.21% higher CRPS. That branch was rejected and its further seed follow-up was canceled; it is separate from the fixed-recipe screen.

The source review found that recipes 115 and 167 share a generative forecast law conditional on the same qualified fitted state and successful preparation. This is not a universal equivalence: recipe 167 has additional finite-value checks and unused EWMA fitting steps that can fail, and identical seeded path bytes were not established. Both labels therefore remain in the static 41-recipe specification; silently deduplicating them would change that specification and its finite path allocation.

The 12- and 90-request cohorts reuse historical origins and overlapping horizons. They establish neither a fresh holdout nor statistical significance. G2 remains the development candidate because it beat the static alternatives on the broader screen and the native-origin branch also lost; this is not a new full-panel result. The official V11 score is unchanged, and these screens add zero full-panel catalogue entries.

## Evidence and recovery

The companion JSON records the frozen plan fingerprints, source and independent-audit hashes, the complete-90 baseline addendum, and the one-row WOR postcheck. The 12-request source and outputs are in a private release that was freshly downloaded, disk-restored, and verified by SHA-256, size, and mode. The 90-request screen is now preserved in a separate [private release](https://github.com/Hutch2064/simfolio-engine/releases/tag/research-fixed-equal41-screen90-wor-20261023T160617Z); its receipt SHA-256 is `e5fcf8de5d3a27cd88bda5c2fd8980c0762ee506df7f63de4bc463fcd8e70c01` and inventory SHA-256 is `84b190026b3ab1b6ac036996a36d47a7eb00a56cba0469d0b5c31eaafca81419`. All 19 archive parts were freshly downloaded and disk-restored with member hashes, sizes, and modes verified (212 files; 1,636 source bindings). The segmented cache was reassembled to SHA-256 `c094cf10c384c865a0070db417110b5c0d570923705f64806616fa0825f0b1bd`. Implementations and compact local evidence remain. After fresh recovery verification and zero-open-handle/reuse-index checks, 90 closed forecast-loss NPZ files (118,024,483 bytes) were retired locally; sources, sidecars, plans, audits, and remote recovery copies remain. The retirement ledger SHA-256 is `49d1fb3210d2330e067c1265f45c5d944025c6f664b200a299ab9a949bde7970`; the ledger itself is also preserved as `fixed_equal41_screen12_screen90_raw_npz_retirement_v1.json` in the access-controlled archive, with a fresh-download verification receipt SHA-256 of `a7f66696fefaebdad565d64891c498b8d63ebd01316acb7a69f8b05739fe650a`. No provider histories, private portfolio rows, or numerical loss arrays are included in this public summary.
