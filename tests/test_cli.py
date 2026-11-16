import json

import numpy as np
import pytest

from evaluation import possession_metrics as pm
from possession.cli import infer_submission, main, write_submission


def test_full_submission_contract_and_determinism(tmp_path):
    codes = np.full(199319, 3, dtype=np.int8)
    codes[11185:90652] = 0
    codes[110808:190198] = 1
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_submission(first, codes, n_expected=199319)
    write_submission(second, codes, n_expected=199319)
    assert first.read_bytes() == second.read_bytes()
    assert pm.validate_submission(pm.read_label_csv(first), 199319) is None


@pytest.mark.parametrize("codes, count", [([0, 4], 2), ([0, 1], 3), ([0, 1.5], 2)])
def test_writer_rejects_invalid_predictions(tmp_path, codes, count):
    with pytest.raises(ValueError):
        write_submission(tmp_path / "bad.csv", np.array(codes), n_expected=count)
    assert not (tmp_path / "bad.csv").exists()


def test_inference_resets_each_half_and_emits_manifest(tmp_path, monkeypatch):
    raw, meta, out = tmp_path / "raw.txt", tmp_path / "meta.xml", tmp_path / "pred.csv"
    meta.write_text("<Root><Team id='h'/><Team id='a'/><Player teamId='h'/>"
                    "<Player teamId='a'/></Root>")
    raw.write_text("".join(f"{i+1}:0.2,0.5;0.8,0.5:0.3,0.5\n" for i in range(10)))
    calls = []

    def scores(half, config=None):
        calls.append(half.frame_id.tolist())
        n = len(half.frame_id)
        return {"live_probability": np.full(n, .99),
                "home_probability": np.full(n, .99 if len(calls) == 1 else .01),
                "boundary": np.zeros((n, 3, 3))}

    monkeypatch.setattr("possession.cli.reference_scores", scores)
    report = infer_submission(raw, meta, out, n_frames=10, spans=((1, 4), (6, 9)))
    assert calls == [[1, 2, 3], [6, 7, 8]]
    assert pm.label_codes(pm.read_label_csv(out)).tolist() == [3, 0, 0, 0, 3, 3, 1, 1, 1, 3]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["output"]["rows"] == 10
    assert manifest["inputs"]["raw"]["sha256"]
    assert report["ground_truth_used"] is False
    assert report["inference_overrides"] == {}


def test_missing_target_data_is_actionable(tmp_path, capsys):
    assert main(["infer", "--data-dir", str(tmp_path), "--output", str(tmp_path / "a.csv")]) == 2
    assert "MTAH_DATA_DIR" in capsys.readouterr().err


def test_invalid_augmentation_config_is_actionable(tmp_path, capsys):
    policy = tmp_path / "policy.json"
    policy.write_text('["time_warp"]')
    assert main(["train", "--output", str(tmp_path / "train"), "--train-matches", "a",
                 "--augmentation-config", str(policy)]) == 2
    assert "augmentation config must be" in capsys.readouterr().err
    assert not (tmp_path / "train").exists()


def test_inference_uses_saved_calibration(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    raw, meta, out = tmp_path / "raw.txt", tmp_path / "meta.xml", tmp_path / "pred.csv"
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"fixture")
    meta.write_text("<Root><Team id='h'/><Team id='a'/><Player teamId='h'/>"
                    "<Player teamId='a'/></Root>")
    raw.write_text("".join(f"{i+1}:0.2,0.5;0.8,0.5:0.3,0.5\n" for i in range(3)))
    monkeypatch.setattr("possession.predict.load_model_checkpoint", lambda *a, **kw: (
        object(), {"training_config": {"calibration": {"live_temperature": 10.0}}}))
    monkeypatch.setattr("possession.predict.predict_model", lambda *a, **kw: {
        "live_probability": np.full(3, .6), "home_probability": np.full(3, .9),
        "boundary": np.zeros((3, 3, 3)),
    })
    report = infer_submission(raw, meta, out, n_frames=3, spans=((0, 3),), checkpoint=checkpoint)
    assert report["config"]["calibration"]["live_temperature"] == 10.0
    assert pm.label_codes(pm.read_label_csv(out)).tolist() == [2, 2, 2]


def test_changed_selected_weights_rejected_before_input_read_or_output(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from possession.model import PossessionModel
    from possession.storage import sha256_file, write_json

    raw, meta = tmp_path / "raw.txt", tmp_path / "meta.xml"
    raw.write_text("".join(f"{i+1}:0.2,0.5;0.8,0.5:0.3,0.5\n" for i in range(3)))
    meta.write_text("<Root><Team id='h'/><Team id='a'/><Player teamId='h'/>"
                    "<Player teamId='a'/></Root>")
    model_kwargs = {"player_dim": 8, "global_dim": 9, "width": 8, "dilations": [1]}
    frozen = {
        "selection_fingerprint": "frozen-test", "outer_train": ["a", "b"],
        "outer_test": "c", "refit_seed": 0, "resources": {"full_budget_steps": 4},
        "decoder": {"switch_cost": 2.0, "boundary_scale": 0.0},
        "calibration": {}, "ensemble": {"neural_weight": 1.0},
        "selected_model_recipe": {"model_kwargs": model_kwargs,
                                  "training_args": {"center_size": 4}},
    }
    recipe, checkpoint = tmp_path / "frozen_recipe.json", tmp_path / "selected.pt"
    write_json(recipe, frozen)
    model = PossessionModel(**model_kwargs)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.live_head.bias.fill_(10)
    payload = {"model_kwargs": model_kwargs, "state_dict": model.state_dict(),
               "training_config": {
                   "train_matches": ["a", "b"], "validation_matches": [],
                   "selection_matches": ["a", "b"], "outer_test": "c", "steps": 4,
                   "center_size": 4,
                   "seed": 0, "selection_fingerprint": "frozen-test",
                   **{key: frozen[key] for key in ("decoder", "calibration", "ensemble")},
               }}
    torch.save(payload, checkpoint)
    write_json(tmp_path / "refit_progress.json", {
        "predictor": "selected_neural", "steps": 4, "train_matches": ["a", "b"],
        "checkpoint_sha256": sha256_file(checkpoint),
    })
    report = infer_submission(raw, meta, tmp_path / "valid.csv", recipe=recipe,
                              n_frames=3, spans=((0, 3),), center_size=4, halo=1)
    assert report["selection_integrity"]["checkpoint_sha256"] == sha256_file(checkpoint)
    assert report["selection_integrity"]["selection_verified"] is True
    assert report["inference_overrides"] == {}
    payload["state_dict"]["live_head.bias"].fill_(-10)
    torch.save(payload, checkpoint)
    monkeypatch.setattr("possession.cli.read_atd", lambda *a, **kw: pytest.fail(
        "target inputs must not be opened after a checkpoint integrity failure"))
    output = tmp_path / "changed.csv"
    with pytest.raises(ValueError, match="checkpoint hash"):
        infer_submission(raw, meta, output, recipe=recipe,
                         n_frames=3, spans=((0, 3),), center_size=4, halo=1)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
