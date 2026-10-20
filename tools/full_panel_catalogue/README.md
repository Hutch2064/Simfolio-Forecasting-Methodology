# Full-panel catalogue maintenance

The expanded 722-series panel inventory is in
[`docs/results/daily-cross-asset-722`](../../docs/results/daily-cross-asset-722/README.md).
It is separate from the older canonical80 and French551 catalogues: their panels,
scoring protocols and counts must not be combined.

After every full-panel attempt, append its execution record, whether it improves
the incumbent or not:

```sh
python tools/full_panel_catalogue/catalogue.py \
  docs/results/daily-cross-asset-722/catalogue.json \
  --append completed-execution-record.json \
  --render docs/results/daily-cross-asset-722/README.md
```

The unique version key binds the original model ID and implementation digest.
Repeated seeds, reruns, display-name changes, removal from active testing, and
rescoring do not create a new model. A changed numerical implementation is a new
version. A verified no-op factory bridge remains an alias of its original tested
version, with its continuity evidence retained. Never replace the original ID
with an alias to inflate the count. Multiple executions are retained under the
same version; changes to an existing execution are rejected rather than silently
overwriting its evidence. Historical corrections must retain their audit trail.

`active_model_ids` is the current retained set. An execution's
`retained_in_active_set` is its historical recording-time snapshot. Replacing
the incumbent changes the active list, never removes the preceding model's
execution, definition or results.

A full-panel attempt means every frozen scheduled task was attempted. Native
fitting failures remain explicit, and do not remove the method from the count.
Partially attempted panels and pilots cannot enter this count. Full-panel scores
and matched comparisons remain separate when successful coverage differs.

Preserve the implementation closure, definitions, exact source versions, fixed
inputs, seeds, environment, raw task losses, both scoring profiles and all result
summaries in the research repository or its linked lossless archives. Large and
provider/private payloads use private repository release assets, not Git history
or the public research tree. Record archive and manifest SHA-256 values and exact
member paths. Removing a model from active control never removes its catalogue
entry, source or results.

Before removing any completed local outputs, download the remote archive afresh,
verify its SHA-256, restore every manifest file to disk, and verify every file's
SHA-256. Preserve that receipt and restore commands in the repository. A successful
upload, matching server digest or manifest download alone is insufficient.
`retirement_allowed` checks a supplied full-recovery receipt but performs no
deletion. The remote audit below deliberately grants no new retirement permission:

```sh
python tools/full_panel_catalogue/verify_remote.py \
  docs/results/daily-cross-asset-722/catalogue.json remote-verification.json
```

Publish the updated catalogue, scoring documentation and recovery references to
the research repository after each new full-panel method. Keep unsuccessful
methods too. Do not rely on an unpushed local ledger as completed preservation.
