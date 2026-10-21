# Verified private research recovery

Private release: https://github.com/Hutch2064/simfolio-engine/releases/tag/research-asset-v6-selector-iteration-studies-20261021T1810Z

The public receipt records11archive/manifest pairs. Each archive and manifest was freshly downloaded;18,961members were disk-restored and checked for byte length, SHA256and file mode. Source publication30b9c2ec8f476a19f854195b25bbc32ce431435bhas its own175file fresh-recovery receipt and7passing recovered tests.

1. Authenticate gh with access to the private repository. Download the exact named release into a new empty staging directory:

```sh
gh release download research-asset-v6-selector-iteration-studies-20261021T1810Z --repo Hutch2064/simfolio-engine --dir /chosen/new-empty-directory
```

2. Verify every archive/manifest SHA256against receipt.json before extraction. The per-part manifest supplies every member's relative path, byte length, mode and SHA256.
3. Restore into a separate empty research root, preserving relative paths and modes; reject absolute paths, traversal, symlinks, duplicates or unknown members. Check every restored file against its manifest. The preservation script in this source snapshot demonstrates the exact streamed verification procedure.
4. Recover referenced baseline inputs, fixed-reference checkpoints and sealed parent dependencies from the receipt's dependency releases before reproducing forecasts. Recorded absolute paths reflect the original research owner; preserve their relative layout or explicitly map them to the recovered root.
5. Copy into an existing research root only after matching existing files or preserving differences. Never overwrite active or unique work. Do not automatically restart a run or recompute scores merely because archived duplicate files are absent.

The8request all-origin control is intentionally stopped. Other included finite studies are complete. The information-originV2auditor corrects the auxiliary unsupported-quarter subset report while preserving the frozen original verifier. Its355request main scores are fully audited. Neither stage completion nor archive recovery completes the active accuracy-maximization goal.

Individual loss checkpoints, records, summaries and reference files remain locally needed for active comparisons. Completed per-case aggregate duplicates and regenerable source-recovery compiler caches were retired after verification; local_duplicate_retirement.jsonrecords exact paths and evidence. This changes no numerical source or score. Archives preserve rejected models and all generated evidence, not only winners.
