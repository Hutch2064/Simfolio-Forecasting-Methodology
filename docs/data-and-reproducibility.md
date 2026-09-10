# Canonical data and reproducibility

The retained dense white-paper panel uses the 52-series snapshot identified by
`resources/data/canonical_snapshot_manifest.json`. The frozen common window is
`1979-12-31` through `2026-05-13`, with 11,687 dates. The normalized return
matrix identity is:

`52c5bdd96b39762183ef0c204fa8165c2dfd5a4864e7198662615daddb8d6a49`

The source revision is `773bc1c325559e6bf57a567f1d8bf473a3427fbc`. The package
publishes the manifest, source-relative paths, schemas, raw file hashes,
common calendar, and all 55 whitelisted compressed source files. Aidan
Hutchison confirmed redistribution approval for this exact research snapshot.
The package records that authorization and preserves the source attribution
in `resources/protocols/canonical_snapshot_attribution.json`; it does not
invent an upstream license.

The frozen calendar is `resources/data/canonical_calendar.csv`, covering the
11687 observed dates in the canonical window. Its SHA-256 is
`8204fbf08e1664a8f254d935052b34c2b4ec9f088ee91f414e7e72c95fb32e04`. The
required risk-free identity is `EFFRX`, source-relative path
`app/simulated_data/series/EFFRX.csv.gz`, SHA-256
`4aa13c0966edab3fa0d75d5b8bf0307de45cd214ad543a4fc3e888d41bb89398`, with
schema `date,daily_return,price`. The French daily factor input uses `RF` as
its risk-free column and has SHA-256
`9ec302fa1f2ac1c630e0019cfb3fcabb97a91255a1543e737aa60beea72ff192`; the Q5
daily input uses `R_F` and has SHA-256
`915434fba2c8c425a6c3b5930f3a6d05bec0f7717adab5eb35c1348ddacbc45a`. These
identities are validated as supplied. The protocol does not silently refresh,
proxy, annualize, or convert a risk-free series while preparing the canonical
snapshot; a model that transforms RF must declare that transformation in its
own model contract.

Prepare the bundled snapshot entirely offline:

```bash
simfolio-oos data verify
simfolio-oos data prepare --destination .simfolio-oos-data
```

Preparation verifies every required series, the supporting `EFFRX` series, both
factor inputs, row counts, date bounds, schemas, and the normalized matrix hash
before writing the caller's local cache. It also retains the authorized source
copy and supporting/factor inputs in that cache and constructs each scored
portfolio from its full source price history before trimming to the common
window. This preserves the source `simulate_portfolio` closure: initial capital
10,000, equal six-way weights, no cash flows, and the source 15 bps turnover
cost. The generated `canonical_portfolios/` logs and source-price-derived asset
logs are local caller-owned inputs; they are not packaged. Verification
recomputes the source-price simulation and rejects a changed portfolio even if
its local manifest hash is rewritten.

`load_canonical_returns` reads the normalized raw return identity.
`load_canonical_engine_inputs` reads the verified source-price-derived asset
logs and portfolio logs used by the canonical task constructor. A prepared
manifest records the packaged snapshot authorization and requires all exact
source and derived-value checks to pass. `redistribution_status` is kept separate and is never inferred
from a hash. The loaders never download, refresh, proxy, or import private
arrays. No public-provider proxy is accepted as the canonical dataset.

`resources/data/canonical_calendar.csv` is calendar metadata rather than a
financial dataset. Its SHA-256 is
`8204fbf08e1664a8f254d935052b34c2b4ec9f088ee91f414e7e72c95fb32e04`.
Synthetic fixtures used by tests are labeled separately and cannot satisfy the
canonical data identity.
