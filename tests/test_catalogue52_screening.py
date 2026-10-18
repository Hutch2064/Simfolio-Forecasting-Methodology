"""User-authorized screening changes the work list, never model definitions."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools/catalogue52"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location("catalogue52_report", TOOLS / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.fixture
def screening(tmp_path):
    folder = tmp_path / "docs/results/catalogue52"
    folder.mkdir(parents=True)
    inventory = {"models": [
        {"public_model_id": "priority", "canonical_rank": 262,
         "campaign": {"fresh_evaluation": True}},
        {"public_model_id": "expensive", "canonical_rank": 188,
         "campaign": {"fresh_evaluation": True}},
        {"public_model_id": "control", "canonical_rank": 261,
         "campaign": {"fresh_evaluation": False}},
    ]}
    inventory_path = folder / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    policy = {"initial_inventory_sha256": report.file_hash(inventory_path),
              "priority_models": ["priority"],
              "exclusions": [{"model_id": "expensive", "canonical_rank": 188,
                              "reason": "User-authorized disproportionate runtime screen"}]}
    return tmp_path, inventory, inventory_path, policy


def test_missing_amendment_preserves_original_population(screening):
    root, inventory, path, _ = screening
    assert report.screening_policy(root, inventory, path) == (None, set())


def test_amendment_preserves_inventory_and_priority_models(screening):
    root, inventory, path, policy = screening
    original = copy.deepcopy(inventory)
    (path.parent / "screening.json").write_text(json.dumps(policy))
    assert report.screening_policy(root, inventory, path) == (policy, {"expensive"})
    assert inventory == original and json.loads(path.read_text()) == original


def test_required_breadth_cannot_be_pruned_away(screening):
    root, inventory, path, policy = screening
    policy["minimum_retained_canonical_models"] = 3
    (path.parent / "screening.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="breadth"):
        report.screening_policy(root, inventory, path)
    policy["minimum_retained_canonical_models"] = 2
    (path.parent / "screening.json").write_text(json.dumps(policy))
    assert report.screening_policy(root, inventory, path)[1] == {"expensive"}


@pytest.mark.parametrize("change", ["inventory", "duplicate", "unknown", "rank", "control", "priority", "reason"])
def test_invalid_or_protected_exclusions_fail_closed(screening, change):
    root, inventory, path, policy = screening
    item = policy["exclusions"][0]
    if change == "inventory":
        policy["initial_inventory_sha256"] = "0" * 64
    elif change == "duplicate":
        policy["exclusions"].append(dict(item))
    elif change == "unknown":
        item["model_id"] = "unknown"
    elif change == "rank":
        item["canonical_rank"] = 187
    elif change == "control":
        item.update(model_id="control", canonical_rank=261)
    elif change == "priority":
        item.update(model_id="priority", canonical_rank=262)
    else:
        item["reason"] = ""
    (path.parent / "screening.json").write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        report.screening_policy(root, inventory, path)
