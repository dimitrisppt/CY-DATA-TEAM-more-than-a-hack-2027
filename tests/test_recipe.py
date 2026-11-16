import json

import pytest

from possession.recipe import load_frozen_recipe, resolve_config


def test_saved_recipe_used_unless_explicitly_overridden():
    training = {"decoder": {"switch_cost": 4.7, "boundary_scale": .3},
                "ensemble": {"neural_weight": .65},
                "calibration": {"owner_temperature": 1.7}}
    cfg = resolve_config(training)
    assert cfg.switch_cost == 4.7 and cfg.boundary_scale == .3
    assert cfg.neural_weight == .65 and cfg.calibration.owner_temperature == 1.7
    changed = resolve_config(training, switch_cost=0, boundary_scale=0, neural_weight=1)
    assert changed.switch_cost == 0 and changed.boundary_scale == 0
    assert changed.neural_weight == 1
    assert changed.calibration == cfg.calibration
    assert resolve_config(has_model=False).neural_weight == 0
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_config(has_model=False, neural_weight=.5)


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan")])
def test_invalid_ensemble_weight_rejected(value):
    with pytest.raises(ValueError, match="neural_weight"):
        resolve_config({"ensemble": {"neural_weight": value}})


def test_frozen_recipe_locates_model_only_if_required(tmp_path):
    path = tmp_path / "frozen_recipe.json"
    value = {"decoder": {}, "calibration": {}, "ensemble": {"neural_weight": 0},
             "selection_fingerprint": "test", "selected_model_recipe": None}
    path.write_text(json.dumps(value))
    assert load_frozen_recipe(path)[1] is None
    value["ensemble"]["neural_weight"] = .6
    value["selected_model_recipe"] = {"model_kwargs": {"width": 64}}
    path.write_text(json.dumps(value))
    assert load_frozen_recipe(path)[1] == tmp_path / "selected.pt"
