
import pytest

from experiments import campaign_evaluation, campaign_protocol
from experiments import ema_stability as stability
from possession.storage import sha256_file, write_json


def _score(objective, shift=0.):
    means = {
        name: {"spell_f1_mean": base + shift,
               "transition_f1_at_12": .4 + shift}
        for name, base in (("clean", .64), ("short", .49),
                           ("long", .48), ("replay", .47))}
    by_match = {
        f"m{i}": {name: {"spell_f1_mean": means[name]["spell_f1_mean"] + i * .001,
                         "transition_f1_at_12": means[name]["transition_f1_at_12"]}
                  for name in means}
        for i in range(6)
    }
    return {"objective": objective, "means": means, "by_match": by_match}


def _selection():
    decoder = {
        "version": 1, "reference_mode": "default", "reference": {},
        "blend_mode": "static", "neural_weight": .5,
        "visible_neural_weight": .5, "missing_neural_weight": .5,
        "visible_live_weight": .5, "missing_live_weight": .5,
        "visible_owner_weight": .5, "missing_owner_weight": .5,
        "transition_seconds": 2., "live_temperature": 1., "owner_temperature": 1.,
        "boundary_temperature": 1., "switch_cost_type": "scalar", "switch_cost": 2.,
        "boundary_scale": .5,
    }
    members = [{"id": "mean_007", "steps": 7200, "spec": {"schedule": {"total_steps": 7200}}},
               {"id": "ball_attention_002", "steps": 7200,
                "spec": {"schedule": {"total_steps": 7200}}}]
    return {"kind": "convex_pair", "blend_weight": .62,
            "decoder": decoder, "members": members}


def test_inner_gate_enforces_each_declared_threshold():
    raw, good = _score(.58), _score(.583, shift=.002)
    result = stability._gate(raw, good)
    assert result["passed"]
    assert all(result["checks"].values())

    exact_margin = _score(.581999999, shift=.002)
    assert not stability._gate(raw, exact_margin)["checks"]["objective_gain_gt_0_002"]
    worse_match = _score(.583, shift=.002)
    for match in ("m0", "m1", "m2"):
        for scenario in ("short", "long", "replay"):
            worse_match["by_match"][match][scenario]["spell_f1_mean"] = .3
    assert not stability._gate(raw, worse_match)["checks"][
        "four_of_six_missing_matches_improved"]
    scenario_loss = _score(.583, shift=.002)
    scenario_loss["means"]["clean"]["spell_f1_mean"] -= .013
    assert not stability._gate(raw, scenario_loss)["checks"][
        "scenario_spell_f1_loss_at_most_0_01"]


def test_outer_guard_is_per_batch_against_raw_v4():
    assert stability._outer_gate(_score(.58), _score(.57))["passed"]
    assert not stability._outer_gate(_score(.58), _score(.569))["passed"]


def test_fold_overlap_and_selection_drift_are_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        stability._disjoint_folds([{"train": ["m0"], "validation": ["m0"]}])
    baseline = _selection()
    candidate = _selection()
    candidate["members"][0]["spec"]["schedule"]["total_steps"] = 7199
    with pytest.raises(ValueError, match="may not change frozen"):
        stability._validate_selection(candidate, baseline)


def test_reused_input_manifest_and_arrays_are_hash_verified(tmp_path):
    input_root = tmp_path / "batch" / "inner"
    output_dir = input_root / "inputs"
    rows = []
    for period in (1, 2):
        for scenario in campaign_protocol.WEIGHTS:
            key = ("m0", period, scenario)
            path = campaign_evaluation._input_path(output_dir, *key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"cached:{period}:{scenario}".encode())
            rows.append({"match": key[0], "period": period, "scenario": key[2],
                         "input_identity": "", "input_sha256": sha256_file(path)})
    signature = {"schema": "reused_test", "inputs": [row["input_sha256"] for row in rows]}
    signature_sha = stability._digest(signature)
    for row in rows:
        row["input_identity"] = signature_sha
    manifest = {"signature": signature, "signature_sha256": signature_sha,
                "records": rows, "status": "complete"}
    manifest_path = input_root / "inputs" / "input-manifest.json"
    write_json(manifest_path, manifest)
    write_json(manifest_path.with_suffix(".sha256.json"), {"sha256": sha256_file(manifest_path)})
    records, loaded = stability._verify_input_cache(input_root, signature_sha, {"m0"})
    assert loaded["signature_sha256"] == signature_sha
    assert len(records) == 8

    (input_root / "inputs" / "inputs" / "m0" / "half1" / "clean.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="array hash differs"):
        stability._verify_input_cache(input_root, signature_sha, {"m0"})


def test_cached_inputs_must_match_frozen_signature_and_full_grid(tmp_path):
    input_root = tmp_path / "batch" / "inner"
    manifest_path = input_root / "inputs" / "input-manifest.json"
    manifest_path.parent.mkdir(parents=True)
    signature = {"scenario": "fresh"}
    payload = {"signature": signature, "signature_sha256": stability._digest(signature),
               "records": [], "status": "complete"}
    write_json(manifest_path, payload)
    write_json(manifest_path.with_suffix(".sha256.json"), {"sha256": sha256_file(manifest_path)})
    with pytest.raises(ValueError, match="signature differs"):
        stability._verify_input_cache(input_root, "other", {"m0"})


def test_budget_cannot_be_extended_on_resume(tmp_path):
    stability._Budget(tmp_path, 100)
    with pytest.raises(ValueError, match="cannot extend"):
        stability._Budget(tmp_path, 101)
