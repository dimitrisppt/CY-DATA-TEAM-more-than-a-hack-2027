"""Scientific parity, exact resumed GPU training and search-boundary checks."""
import copy

import numpy as np
import optuna
import pytest
import torch
from test_gpu_runtime import _data_spec

from evaluation import possession_metrics as official
from experiments import campaign_training, gpu_runtime
from experiments.coarse_context import blend_coarse_context
from experiments.ema_training import _tree_equal
from experiments.gpu_tuning import ranking, study_from_rows
from experiments.gpu_tuning_evaluation import coarse_blend, comparison, spell_result
from experiments.gpu_tuning_space import sample_pipeline, sample_recipe
from experiments.gpu_tuning_training import fit, make_session
from experiments.refinement_training import load_refinement_state, save_refinement_state


@pytest.mark.parametrize("schedule", ["constant", "cosine"])
@pytest.mark.parametrize("weight,sampling", [(0., 0.), (.25, .3), (.1, 0.)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_exact_cuda_resume_with_sampled_schedule_and_losses(tmp_path, schedule, weight, sampling):
    gpu_runtime.configure_device()
    data, spec = _data_spec(tmp_path)
    spec["schedule"].update(kind=schedule, total_steps=12,
                            warmup_steps=2 if schedule == "cosine" else 0,
                            min_lr_ratio=.2 if schedule == "cosine" else 1.)
    recipe = {"spec": spec, "spell_weight": weight,
              "short_sampling_probability": sampling, "export": "ema"}
    with gpu_runtime.accelerated_preprocessing():
        whole = make_session(data, recipe, ["train"], 83)
        whole.advance_to(12)
        staged = make_session(data, recipe, ["train"], 83)
        staged.advance_to(6)
        save_refinement_state(tmp_path / "state.pt", staged, {"test": 1})
        resumed = make_session(data, recipe, ["train"], 83)
        load_refinement_state(tmp_path / "state.pt", resumed, {"test": 1})
        resumed.advance_to(12)
    assert _tree_equal(campaign_training._session_state(whole), campaign_training._session_state(resumed))
    assert _tree_equal(whole._tail_ema, resumed._tail_ema)
    assert resumed._tail_ema_updates == 12
    assert next(resumed.model.parameters()).is_cuda
    if schedule == "cosine":
        assert resumed.optimizer.param_groups[0]["lr"] < spec["training_args"]["learning_rate"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rung_promotion_and_completed_fit_reuse(tmp_path):
    from possession.storage import load_half, save_half

    gpu_runtime.configure_device()
    data, spec = _data_spec(tmp_path)
    frames, targets = load_half(data / "train_half1.npz")
    frames.match_id = "excluded"
    save_half(data / "excluded_half1.npz", frames, targets)
    recipe = {"spec": spec, "spell_weight": .25, "short_sampling_probability": .25, "export": "ema"}
    directory = tmp_path / "fit"
    args = (data, recipe, ["train"], ["excluded"], 17)
    with gpu_runtime.accelerated_preprocessing():
        lower = fit(*args, 4, directory, {"test": 1}, None, lambda **kw: None)
        upper = fit(*args, 8, directory, {"test": 1}, None, lambda **kw: None)
        reused = fit(*args, 4, directory, {"test": 1}, None, lambda **kw: None)
    assert lower == reused and upper["steps"] == 8 and upper["export"] == "ema"
    assert len(list(directory.glob("state_*.pt"))) == 1
    assert lower["model_sha256"] != upper["model_sha256"]
    with pytest.raises(ValueError, match="overlap"):
        fit(data, recipe, ["train"], ["train"], 17, 4, directory, {}, None, lambda **kw: None)


def test_default_coarse_gate_exactly_reproduces_original():
    rng = np.random.default_rng(4)
    arrays = {"neural_live": rng.random(100), "neural_home": rng.random(100),
              "neural_boundary": rng.normal(size=(100, 3, 3)),
              "ball_valid": rng.random(100) > .5,
              "nearest_ball_seconds": (rng.random(100) * 30).astype(np.float32),
              "coarse_live": rng.random(100), "coarse_home": rng.random(100)}
    expected = blend_coarse_context(arrays, arrays["coarse_live"], arrays["coarse_home"])
    actual = coarse_blend(arrays, .25, 10.)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize("seed", range(5))
def test_search_metric_matches_official_with_gaps_and_abstentions(seed):
    rng = np.random.default_rng(seed)
    gt = np.repeat(rng.integers(0, 4, 30), 7)
    pred = np.repeat(rng.integers(0, 4, 30), 7)
    expected = official.evaluate(gt, pred, bootstrap_n=0)
    actual = spell_result(gt, pred)
    assert np.mean([actual["per_iou"][t]["f1"] for t in official.SPELL_IOUS]) == expected["spell_f1_mean"]


def test_historical_match_gate_and_single_guard_uncertainty():
    means = {s: .6 for s in ("clean", "short", "long", "replay")}
    baseline = {"objective": .6, "means": means, "by_match": {str(i): means for i in range(6)}}
    candidate = copy.deepcopy(baseline)
    candidate["objective"] += .01
    candidate["means"] = {s: v + .01 for s, v in means.items()}
    candidate["by_match"] = {m: candidate["means"] for m in baseline["by_match"]}
    result = comparison(candidate, baseline)
    assert result["passed"] and "descriptive_95_interval" in result
    baseline["by_match"] = {"WMX": means}
    candidate["by_match"] = {"WMX": candidate["means"]}
    assert "descriptive_95_interval" not in comparison(candidate, baseline, guard=True)


def test_seeded_proposals_are_reproducible():
    from experiments.campaign_decode import baseline_decoder
    decoder = baseline_decoder({"ensemble": {"neural_weight": .5},
                               "calibration": {}, "decoder": {"switch_cost": 2.}})
    for function, arguments in ((sample_recipe, ()), (sample_pipeline, (decoder,))):
        a = study_from_rows([], 13).ask()
        b = study_from_rows([], 13).ask()
        assert function(a, *arguments) == function(b, *arguments)


def test_ranking_is_stable_at_ties():
    rows = [{"id": "b", "scores": {"objective": .6}}, {"id": "a", "scores": {"objective": .6}}]
    assert [r["id"] for r in ranking(rows)] == ["a", "b"]


def test_space_constructs_all_supported_models():
    from possession.model import PossessionModel
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=21))
    for _ in range(15):
        recipe = sample_recipe(study.ask())
        model = PossessionModel(**recipe["spec"]["model_kwargs"])
        assert model.radius > 0


def test_extra_seed_budget_cannot_change_selection(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments import gpu_tuning
    from experiments.campaign_protocol import BudgetExceeded, publish, read_record

    campaign = object.__new__(gpu_tuning.Campaign)
    campaign.output = tmp_path
    campaign.plan = {"cutoffs": {"stability": .98}}
    campaign.tick = lambda **kw: None
    campaign.get_bank = lambda *args: SimpleNamespace(predict=lambda *args: {})
    means = {s: .6 for s in ("clean", "short", "long", "replay")}
    scores = {"objective": .6, "means": means, "by_match": {str(i): means for i in range(6)}}
    selection = {"proposal": {"id": "test"}, "steps": 10, "pipeline": {"candidate_weight": .5}}
    publish(tmp_path / "selection.json", selection)
    publish(tmp_path / "confirmation.json", {"incumbent": scores})
    monkeypatch.setattr(gpu_tuning, "evaluate", lambda *args: scores)

    def mocked_fit(proposal, steps, seed):
        if seed > 9601:
            raise BudgetExceeded("time reached")
        return []

    campaign.fits = mocked_fit
    campaign.stability()
    result = read_record(tmp_path / "stability.json")
    assert [r["seed"] for r in result["completed"]] == [9601]
    assert result["changes_selection_or_gate"] is False
    assert read_record(tmp_path / "selection.json") == selection


def test_extra_seeds_inapplicable_for_decoder_only(tmp_path):
    from experiments.campaign_protocol import publish, read_record
    from experiments.gpu_tuning import Campaign

    campaign = object.__new__(Campaign)
    campaign.output = tmp_path
    campaign.plan = {"cutoffs": {"stability": .98}}
    publish(tmp_path / "selection.json", {"proposal": None, "pipeline": {"candidate_weight": 0.}})
    campaign.stability()
    assert read_record(tmp_path / "stability.json")["completed"] == []
