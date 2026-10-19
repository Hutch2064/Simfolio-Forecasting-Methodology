"""Validated short-horizon refinement evidence, separate from model rankings."""

from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from pathlib import Path

INDEX_RESOURCE = "resources/results/refinement52.json"
M261_ID = "asset_m256_data_timescales_existing_sv_rate"


def _valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def validate_refinement52_index(payload):
    """Validate the extension's separate score panels and evidence bindings."""
    if (payload.get("schema_version") != 1
            or payload.get("study_id") != "short-horizon-mechanisms-20261018"):
        raise ValueError("invalid refinement52 extension schema")
    identity = payload.get("catalogue_identity", {})
    if (identity.get("research_ids_are_immutable") is not True
            or identity.get("research_ids_are_master_model_ids") is not False
            or identity.get("m_ids_assigned") is not False
            or identity.get("catalogue_definitions_or_ranks_changed") is not False
            or identity.get("main_sweep_result_index_changed") is not False
            or identity.get("baseline_model_id") != M261_ID):
        raise ValueError("refinement evidence must remain separate from catalogue identities")
    panels = payload.get("panels", {})
    if (panels.get("canonical52", {}).get("assets") != 52
            or panels.get("canonical52", {}).get("origins") != 2652
            or panels.get("canonical52", {}).get("cells") != 455832
            or panels.get("historical80", {}).get("portfolios") != 80
            or panels.get("historical80", {}).get("origins") != 4080
            or panels.get("historical80", {}).get("cells") != 701280):
        raise ValueError("canonical52 and historical80 panel denominators must remain separate")
    baseline = payload.get("baseline_result", {})
    if (baseline.get("catalogue_model_id") != M261_ID
            or not _valid_score(baseline.get("canonical52", {}).get("exact_empirical_crps"))
            or not _valid_score(baseline.get("historical80", {}).get("exact_empirical_crps"))):
        raise ValueError("separate M261 reference scores are missing")
    hashes = payload.get("source_hashes", {})
    if (not isinstance(hashes, dict) or not hashes
            or any(not _valid_hash(digest) for digest in hashes.values())):
        raise ValueError("refinement extension source hashes are missing or malformed")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("refinement extension entries are missing")
    ids = [entry.get("research_id") for entry in entries]
    if (len(ids) != len(set(ids))
            or any(not isinstance(item, str) or not re.fullmatch(r"R52-[A-Z0-9-]+", item)
                   for item in ids)):
        raise ValueError("refinement entries need unique immutable R52 research IDs")
    rough_off = None
    for entry in entries:
        if entry.get("status") == "stopped_identity_control":
            canonical = entry.get("panel_results", {}).get("canonical52", {})
            historical = entry.get("panel_results", {}).get("historical80", {})
            if (canonical.get("status") != "stopped_identity_control"
                    or canonical.get("score") is not None
                    or "exact_empirical_crps" in canonical
                    or historical.get("status") != "not_run"
                    or "exact_empirical_crps" in historical):
                raise ValueError("stopped identity controls must remain unscored")
        elif entry.get("status") == "complete":
            panels = entry.get("panel_results", {})
            canonical = panels.get("canonical52", {})
            if (canonical.get("status") != "complete_validated_full_panel"
                    or not _valid_score(canonical.get("exact_empirical_crps"))):
                raise ValueError("only complete canonical52 results may be loaded")
            historical = panels.get("historical80", {})
            if entry.get("research_id") == "R52-ROUGH-OFF":
                if (historical.get("status") != "complete_full_panel"
                        or not _valid_score(historical.get("exact_empirical_crps"))):
                    raise ValueError("rough-off historical80 result must be full-panel evidence")
                rough_off = historical
            elif (historical.get("status") != "not_run"
                  or "exact_empirical_crps" in historical):
                raise ValueError("unrun historical80 results cannot carry a score")
        else:
            raise ValueError("unknown refinement entry status")
        if any(path not in hashes for path in entry.get("source_paths", [])):
            raise ValueError(f"source hashes are missing for {entry['research_id']}")
    if (rough_off is None or rough_off.get("status") != "complete_full_panel"
            or not _valid_score(rough_off.get("exact_empirical_crps"))):
        raise ValueError("rough-off historical80 result is missing")
    if any(path not in hashes for path in baseline.get("source_paths", [])):
        raise ValueError("M261 baseline source hashes are missing")
    runtime = payload.get("runtime_evidence", {})
    if (runtime.get("website_ctv_or_deployed_runtime_claim") is not False
            or any(path not in hashes for path in runtime.get("source_paths", []))):
        raise ValueError("runtime evidence must remain source-bound and local")
    rms_runtime = runtime.get("rms_runtime_matched_benchmark")
    if rms_runtime is not None:
        if (rms_runtime.get("status") != "complete_local_matched_compute_trials"
                or rms_runtime.get("full_panel_or_website_speed_claim") is not False
                or any(path not in hashes for path in rms_runtime.get("source_paths", []))):
            raise ValueError("RMS runtime benchmark scope or source bindings are invalid")
    winner_gate = runtime.get("best_scoring_candidate_runtime_gate", {})
    winner_benchmark = runtime.get("selected_winner_matched_benchmark", {})
    if (winner_gate.get("research_id") != "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION"
            or winner_gate.get("status")
            != "complete_full_2652_runtime_parity_and_six_context_benchmark"
            or winner_gate.get("matched_speed_benchmark_status")
            != "complete_local_matched_compute_trials"
            or winner_gate.get("goal_winner_claimed") is not False
            or winner_benchmark.get("status") != "complete_local_matched_compute_trials"
            or winner_benchmark.get("trials") != 72
            or winner_benchmark.get("samples") != 288
            or winner_benchmark.get("all_six_contexts_faster_than_production_and_rough_off") is not True
            or winner_benchmark.get("full_panel_or_website_speed_claim") is not False
            or any(winner_benchmark.get("median_speedups_by_cache_state", {}).get(state, {}).get(key, 0) <= 1
                   for state in ("fresh_fit", "cached_fit")
                   for key in ("median_speedup_vs_production", "median_speedup_vs_rough_off"))
            or any(path not in hashes for path in winner_benchmark.get("source_paths", []))):
        raise ValueError("selected-winner runtime evidence must remain bounded and hash-bound")
    diagnostics = payload.get("additional_diagnostics", {})
    node_parity = diagnostics.get("variance_only_node_runtime_parity", {})
    selected_parity = node_parity.get("selected_context", {})
    full_parity = node_parity.get("full_panel", {})
    if (selected_parity.get("status") != "complete_selected_context_exact_output_parity"
            or selected_parity.get("contexts") != 27
            or selected_parity.get("runtime_measurement_or_speed_claim") is not False
            or full_parity.get("status") != "complete_full_panel_runtime_only_exact_output_parity"
            or full_parity.get("origins") != 2652
            or full_parity.get("daily_path_values_compared") != 937010880
            or full_parity.get("all_daily_arrays_byte_equal") is not True
            or full_parity.get("all_horizon_CRPS_byte_equal") is not True
            or full_parity.get("runtime_measurement_or_speed_claim") is not False
            or any(path not in hashes for part in (selected_parity, full_parity)
                   for path in part.get("source_paths", []))):
        raise ValueError("node variance parity must remain all-work and hash-bound")
    growth = diagnostics.get("long_growth", {})
    if (growth.get("status") != "complete_non_exclusionary_diagnostic"
            or growth.get("no_asset_exclusion_rule") is not True
            or growth.get("audit_path") not in hashes
            or growth.get("audit_sha256") != hashes.get(growth.get("audit_path"))
            or any(path not in hashes for path in growth.get("source_paths", []))):
        raise ValueError("long-growth diagnostic must remain hash-bound and non-exclusionary")
    objective = payload.get("research_objective", {})
    if (objective.get("source_path") not in hashes
            or objective.get("sha256") != hashes.get(objective.get("source_path"))
            or len(objective.get("verbatim_copy_paths", [])) != 2):
        raise ValueError("verbatim research objective is not hash-bound")
    archive = payload.get("archive_evidence", {})
    if archive != diagnostics.get("private_archive"):
        raise ValueError("private archive evidence copies differ")
    if archive.get("status") == "pending":
        if (archive.get("format") != "refinement52-archive-receipt-v1"
                or archive.get("verified_online_archive") is not False
                or archive.get("scientific_restore_smoke") != "not_run"
                or "scientific_restore_smoke_receipt" in archive
                or archive.get("deletion_authorized") is not False):
            raise ValueError("pending private archive evidence is invalid")
    elif archive.get("status") == "verified_private_online_archive":
        manifest = archive.get("manifest", {})
        assets = archive.get("assets", {})
        if (archive.get("format") != "refinement52-archive-receipt-v1"
                or archive.get("verified_online_archive") is not True
                or not isinstance(archive.get("repo"), str) or not archive["repo"]
                or not isinstance(archive.get("release"), str) or not archive["release"]
                or not _valid_hash(manifest.get("sha256"))
                or not isinstance(manifest.get("bytes"), int)
                or not isinstance(manifest.get("url"), str) or not manifest["url"]
                or not isinstance(assets, dict) or not assets
                or any(not _valid_hash(item.get("sha256"))
                       or not isinstance(item.get("bytes"), int)
                       or not isinstance(item.get("url"), str) or not item["url"]
                       for item in assets.values())
                or not isinstance(archive.get("member_count"), int)
                or not isinstance(archive.get("member_bytes"), int)
                or archive.get("independent_download_and_member_verification") is not True
                or archive.get("scientific_restore_smoke") not in ("pending", "passed")
                or archive.get("deletion_authorized") is not False
                or archive.get("manifest_source_path") not in hashes
                or archive.get("manifest_sha256") != hashes.get(archive.get("manifest_source_path"))
                or archive.get("receipt_source_path") not in hashes
                or archive.get("receipt_sha256") != hashes.get(archive.get("receipt_source_path"))
                or not isinstance(archive.get("restore_command"), str)
                or not archive["restore_command"]
                or not isinstance(archive.get("external_dependencies"), list)
                or any(path not in hashes for path in archive.get("source_paths", []))):
            raise ValueError("verified private archive receipt is not checksum-bound")
        smoke = archive.get("scientific_restore_smoke_receipt")
        if archive.get("scientific_restore_smoke") == "passed":
            expected_cases = {"SPYSIM_8766", "TLTSIM_8766", "SPXUSIM_8766", "XLESIM_126"}
            if (not isinstance(smoke, dict)
                    or smoke.get("status") != "verified_restored_scientific_smoke"
                    or smoke.get("variant") != "node_spread"
                    or smoke.get("complete_path_parity") is not True
                    or smoke.get("case_count") != 4
                    or {row.get("case") for row in smoke.get("cases", [])} != expected_cases
                    or any(not _valid_hash(row.get(key)) for row in smoke["cases"]
                           for key in ("full_path_sha256", "oracle_sha256"))
                    or smoke.get("receipt_source_path") not in hashes
                    or smoke.get("receipt_sha256") != hashes.get(smoke.get("receipt_source_path"))
                    or any(path not in hashes for path in smoke.get("source_paths", []))):
                raise ValueError("restored scientific smoke receipt is not checksum-bound")
        elif smoke is not None:
            raise ValueError("pending scientific smoke cannot carry a receipt")
    else:
        raise ValueError("private archive status must be explicitly pending or verified")


def load_refinement52_results(path: Path | None = None):
    """Load the validated research sidecar without merging it into catalogue ranks."""
    source = files(__package__).joinpath(INDEX_RESOURCE) if path is None else Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_refinement52_index(payload)
    return payload
