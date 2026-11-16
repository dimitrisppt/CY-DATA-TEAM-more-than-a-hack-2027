import json
from pathlib import Path

import pytest

from experiments import campaign_protocol
from experiments import ensemble_polish_confirm as confirm
from experiments.tune import _seed_for_fold
from possession.storage import sha256_file


def _spec():
    return {"model_kwargs": {"width": 8, "pooling": "mean"},
            "training_args": {}, "outage_config": {}, "augmentation_config": {},
            "schedule": {"total_steps": 7200}}


def _member():
    return {"id": "member_a", "spec": _spec(), "steps": 7200}


def _write_fit(tmp_path, *, train=None):
    campaign = tmp_path / "campaign"
    protocol = {"fold": {"inner": [], "outer_train": ["A", "B", "C", "D", "E", "F"]}}
    campaign_protocol.publish(campaign / "protocol.json", protocol)
    protocol_sha = confirm._digest(protocol)
    fold = {"train": ["A", "B", "C", "D"], "validation": ["E", "F"]}
    actual_seed = _seed_for_fold(8101, "DFL-MAT-J03WMX", 0)
    path = (campaign / "fits" / "inner" / "member_a" / "seed_8101" /
            "fold_0" / "rung_7200")
    path.mkdir(parents=True)
    checkpoint = path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    identity = {"protocol_sha256": protocol_sha, "spec_sha256": confirm._digest(_spec()),
                "train_matches": train if train is not None else fold["train"],
                "seed": actual_seed, "name": "member_a"}
    campaign_protocol.publish(path / "fit.json", {"identity": identity, "steps": 7200,
                                                   "checkpoint_sha256": sha256_file(checkpoint)})
    return campaign, protocol_sha, fold, path, actual_seed


def test_fresh_scenarios_change_only_synthetic_seeds_and_reuse_replay():
    assert confirm.FRESH_SCENARIOS["short"] == {
        "kind": "synthetic", "seed": 73103, "missing_fraction": .41,
        "min_seconds": 1., "max_seconds": 10.}
    assert confirm.FRESH_SCENARIOS["long"] == {
        "kind": "synthetic", "seed": 74103, "missing_fraction": .41,
        "min_seconds": 10., "max_seconds": 30.}
    assert confirm.FRESH_SCENARIOS["replay"] == {"kind": "target_replay", "seed": 5102}


def test_fit_model_checks_existing_fit_and_checkpoint_recipe(monkeypatch, tmp_path):
    campaign, protocol_sha, fold, path, actual_seed = _write_fit(tmp_path)
    model = object()
    model_kwargs = {"player_dim": 8, "global_dim": 9, "width": 64,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
    model_kwargs.update(_spec()["model_kwargs"])
    payload = {"model_kwargs": model_kwargs,
               "training_config": {"campaign_spec": _spec(), "seed": actual_seed,
                                   "steps": 7200, "train_matches": fold["train"],
                                   "validation_matches": []}}
    monkeypatch.setattr(confirm, "load_model_checkpoint", lambda p: (model, payload))
    loaded, identity = confirm._fit_model(campaign, protocol_sha, _member(), 8101, 0,
                                          fold, "inner")
    assert loaded is model
    assert identity["actual_train_seed"] == actual_seed
    assert identity["checkpoint_sha256"] == sha256_file(path / "model.pt")


def test_fit_model_rejects_mismatched_training_fold(monkeypatch, tmp_path):
    campaign, protocol_sha, fold, _path, _actual_seed = _write_fit(tmp_path, train=["A", "B", "C"])
    monkeypatch.setattr(confirm, "load_model_checkpoint", lambda p: (
        object(), {"model_kwargs": _spec()["model_kwargs"], "training_config": {}}))
    with pytest.raises(ValueError, match="fit identity"):
        confirm._fit_model(campaign, protocol_sha, _member(), 8101, 0, fold, "inner")


def test_fit_model_rejects_checkpoint_recipe_or_actual_seed_mismatch(monkeypatch, tmp_path):
    campaign, protocol_sha, fold, _path, actual_seed = _write_fit(tmp_path)
    model_kwargs = {"player_dim": 8, "global_dim": 9, "width": 64,
                    "dilations": [1, 2, 4, 8, 16, 32, 64, 128]}
    model_kwargs.update(_spec()["model_kwargs"])
    payload = {"model_kwargs": model_kwargs,
               "training_config": {"campaign_spec": _spec(), "seed": actual_seed + 1,
                                   "steps": 7200, "train_matches": fold["train"],
                                   "validation_matches": []}}
    monkeypatch.setattr(confirm, "load_model_checkpoint", lambda p: (object(), payload))
    with pytest.raises(ValueError, match="recipe, seed, or fold"):
        confirm._fit_model(campaign, protocol_sha, _member(), 8101, 0, fold, "inner")


def test_fit_model_rejects_corrupt_checkpoint_checksum(tmp_path):
    campaign, protocol_sha, fold, path, _ = _write_fit(tmp_path)
    (path / "model.pt").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        confirm._fit_model(campaign, protocol_sha, _member(), 8101, 0, fold, "inner")


def test_no_training_api_is_imported_or_called():
    text = Path(confirm.__file__).read_text(encoding="utf-8")
    assert "build_session" not in text
    assert "TrainingSession" not in text
    assert "target_labels" not in text


def test_predict_selection_writes_predictions_under_new_output_root(monkeypatch, tmp_path):
    output_root = tmp_path / "new-confirmation"
    member = _member()
    fold = {"train": ["A", "B"], "validation": ["C"]}
    source = [{"match": "C", "period": period, "scenario": "clean",
               "input_path": str(tmp_path / f"input{period}.npz"), "input_sha256": str(period)}
              for period in (1, 2)]
    destinations = []
    monkeypatch.setattr(confirm, "_fit_model", lambda *args, **kwargs:
                        (object(), {"protocol_sha256": "p", "fresh_inputs_sha256": None}))
    def predict(model, selected, destination, identity, **kwargs):
        destinations.append(destination)
        return selected
    monkeypatch.setattr(confirm.campaign_evaluation, "predict_records", predict)
    monkeypatch.setattr(confirm.campaign_evaluation, "load_records",
                        lambda records: [{**row, "arrays": {}} for row in records])
    _, bundles = confirm._predict_selection(
        tmp_path / "old-campaign", "protocol", {"members": [member]}, 8101,
        [fold], source, "fresh", output_root, "inner")
    assert bundles[0] == source
    assert destinations == [output_root / "inner/member_a/seed_8101/fold_0"]
    assert not (tmp_path / "old-campaign/fits/inner/member_a/seed_8101/fold_0/predictions").exists()


def test_confirm_pair_mocked_run_is_hashed_resumable_and_keeps_candidate_frozen(
        monkeypatch, tmp_path):
    matches = list("ABCDEF")
    folds = []
    for index in range(3):
        validation = matches[index * 2:index * 2 + 2]
        folds.append({"train": [match for match in matches if match not in validation],
                      "validation": validation})
    protocol = {"fold": {"inner": folds, "outer_train": matches}}
    campaign = tmp_path / "campaign"
    campaign_protocol.publish(campaign / "protocol.json", protocol)
    baseline = {"members": [_member()], "decoder": {"value": 1}, "blend_weight": None}
    candidate = {"members": [_member()], "decoder": {"value": 2}, "blend_weight": None}
    output = tmp_path / "confirmation"
    input_records = [{"match": match, "period": period, "scenario": scenario,
                      "input_path": "mock", "input_sha256": "mock"}
                     for match in matches + ["DFL-MAT-J03WMX"]
                     for period in (1, 2) for scenario in confirm.FRESH_SCENARIOS]
    def prepare(data, target, directory, selected_matches, scenarios, cache, progress_callback=None):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "input-manifest.json").write_text(json.dumps({"signature_sha256": "input"}))
        return [row for row in input_records if row["match"] in selected_matches]
    monkeypatch.setattr(confirm.campaign_evaluation, "prepare_inputs", prepare)
    monkeypatch.setattr(confirm.campaign_evaluation, "load_records", lambda records: records)
    fake_record = {"input_path": "mock", "input_sha256": "mock", "prediction_path": "mock",
                   "prediction_sha256": "mock", "model_identity": {"fresh_inputs_sha256": "input"},
                   "center_size": 2048, "match": "A", "period": 1, "scenario": "clean"}
    scored = []
    def score_selection(campaign_dir, protocol_sha, selection, seed, fold_rows, inputs,
                        fresh_sha, prediction_dir, scope, check=None):
        assert prediction_dir == output / "predictions"
        assert fresh_sha == "input"
        delta = .02 if selection["decoder"]["value"] == 2 else 0.
        score = _score(delta, matches if scope == "inner" else ["DFL-MAT-J03WMX"])
        scored.append((scope, seed, selection["decoder"]["value"]))
        return score, [[fake_record]]
    monkeypatch.setattr(confirm, "_score_selection", score_selection)
    monkeypatch.setattr(confirm, "_bagging", lambda *args, **kwargs: {"available": False})
    result = confirm.confirm_pair(campaign, tmp_path / "data", tmp_path / "target", output,
                                  baseline, candidate)
    assert result["eligible"]
    assert result["gate"]["passed"]
    assert result["outer"]["gate_passed"]
    assert len(list(output.glob("seed_*.json"))) == 2
    assert len(list(output.glob("outer_seed_*.json"))) == 2
    assert len(scored) == 8
    assert confirm.confirm_pair(campaign, tmp_path / "data", tmp_path / "target", output,
                                baseline, candidate) == result
    assert len(scored) == 8


def _score(delta, matches):
    values = {"clean": .5, "short": .35, "long": .3, "replay": .32}
    by_match = {match: {name: {"spell_f1_mean": value + delta,
                              "transition_f1_at_12": .3 + delta}
                       for name, value in values.items()} for match in matches}
    means = {name: {"spell_f1_mean": value + delta, "transition_f1_at_12": .3 + delta}
             for name, value in values.items()}
    objective = sum(confirm.campaign_protocol.WEIGHTS[name] * means[name]["spell_f1_mean"]
                    for name in means)
    return {"by_match": by_match, "means": means, "objective": objective}
