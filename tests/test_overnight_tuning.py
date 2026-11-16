"""Small orchestration tests for the restartable overnight campaign."""

from argparse import Namespace

import numpy as np
import optuna
import pytest
import torch

from experiments import overnight_tuning as overnight
from experiments.campaign_protocol import BudgetExceeded, read_record
from experiments.campaign_space import DILATIONS, OPS
from possession.augmentation import AugmentationConfig
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half


def _half(match, n=48):
    t = np.arange(n)
    players = np.zeros((n, 2, 2, 2), np.float32)
    players[:, 0, :, 0] = 20 + t[:, None] * .05
    players[:, 1, :, 0] = 70 - t[:, None] * .05
    players[..., 1] = 30
    state = np.resize(np.repeat([0, 2, 1], [9, 3, 8]), n).astype(np.int8)
    live = (state != 2).astype(np.int8)
    frames = FrameTable(t, players, np.ones((n, 2, 2), bool),
                        np.column_stack((25 + t * .1, np.full(n, 31))),
                        np.ones(n, bool), np.ones(n, bool), match_id=match)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _prepared(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in ("a", "b", "c"):
        save_half(data / f"{match}_half1.npz", *_half(match))
    return data


def _spec(total=5):
    return {
        "model_kwargs": {"player_dim": 8, "global_dim": 9, "width": 8,
                         "pooling": "mean", "dilations": DILATIONS[7]},
        "training_args": {"learning_rate": .003, "weight_decay": .0001,
                          "boundary_weight": 0., "boundary_pos_weight": 1.,
                          "center_size": 16, "augment": True},
        "outage_config": {"mode": "legacy"},
        "augmentation_config": AugmentationConfig(
            mode="spatial", operations=("coordinate_noise",), clean_probability=.5,
            strength=.5, warmup_steps=1).to_dict(),
        "schedule": {"kind": "cosine", "warmup_steps": 1,
                     "min_lr_ratio": .1, "total_steps": total},
    }


def _campaign(tmp_path):
    campaign = overnight.Campaign.__new__(overnight.Campaign)
    campaign.output = tmp_path
    campaign.data_dir = _prepared(tmp_path)
    campaign.protocol_sha = "protocol-test"
    campaign.cache = None
    campaign.tick = lambda *args, **kwargs: None
    campaign.budget = type("Budget", (), {"stage": "test", "status": lambda *a, **k: None})()
    return campaign


def _row(pooling, index, value):
    return {"id": f"{pooling}_{index:03d}", "spec": {"model_kwargs": {"pooling": pooling}},
            "steps": 20, "decoder": {"kind": "test"},
            "scores": {"objective": float(value)}}


def test_screening_keeps_only_complete_architecture_rounds_and_resumes(tmp_path, monkeypatch):
    camp = _campaign(tmp_path)
    camp.args = Namespace(per_family=2, decoder_trials=(3, 3, 3))
    camp.baseline = {"training_args": {}}
    camp.protocol = {"resources": {"rungs": [10, 20, 30]}}
    camp.studies = {name: object() for name in overnight.POOLINGS}
    monkeypatch.setattr(overnight, "_proposal", lambda study, pooling, index, *args: {
        "id": f"{pooling}_{index:03d}", "index": index,
        "spec": {"model_kwargs": {"pooling": pooling}}})
    completed = []
    calls = 0

    def evaluate(proposal, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt("simulated process interruption")
        result = _row(proposal["spec"]["model_kwargs"]["pooling"], proposal["index"], calls)
        completed.append(result)
        return result

    monkeypatch.setattr(camp, "evaluate_trial", evaluate)
    monkeypatch.setattr(overnight, "_append", lambda *args: None)
    with pytest.raises(KeyboardInterrupt):
        camp.screening([], {})
    assert not (tmp_path / "screening.json").exists()
    # A process interruption keeps per-trial results and resumes the round.
    monkeypatch.setattr(camp, "evaluate_trial", lambda proposal, *args, **kwargs: _row(
        proposal["spec"]["model_kwargs"]["pooling"], proposal["index"], 10 + proposal["index"]))
    resumed = camp.screening([], {})
    assert resumed["complete_rounds"] == 2
    assert len(resumed["rows"]) == 6
    assert {row["spec"]["model_kwargs"]["pooling"] for row in resumed["rows"]} == set(overnight.POOLINGS)


def test_phase_cutoff_retains_only_complete_three_architecture_rounds(tmp_path, monkeypatch):
    camp = _campaign(tmp_path)
    camp.args = Namespace(per_family=2, decoder_trials=(3, 3, 3))
    camp.baseline = {"training_args": {}}
    camp.protocol = {"resources": {"rungs": [10, 20, 30]}}
    camp.studies = {name: object() for name in overnight.POOLINGS}
    monkeypatch.setattr(overnight, "_proposal", lambda study, pooling, index, *args: {
        "id": f"{pooling}_{index:03d}", "index": index,
        "spec": {"model_kwargs": {"pooling": pooling}}})
    calls = 0

    def evaluate(proposal, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise BudgetExceeded("screen cutoff")
        return _row(proposal["spec"]["model_kwargs"]["pooling"], proposal["index"], calls)

    monkeypatch.setattr(camp, "evaluate_trial", evaluate)
    monkeypatch.setattr(overnight, "_append", lambda *args: None)
    partial = camp.screening([], {})
    assert partial["complete_rounds"] == 1
    assert len(partial["rows"]) == 3
    assert partial["incomplete_round_excluded"] is True
    assert partial["closed_reason"] == "screen_wall_clock_cutoff"


def test_promotions_rank_three_per_pool_then_one_each_and_selection_is_frozen(tmp_path, monkeypatch):
    camp = _campaign(tmp_path)
    camp.args = Namespace(decoder_trials=(3, 3, 3))
    camp.protocol = {"resources": {"rungs": [10, 20, 30]},
                     "selection": {"pair_blend_weights": [.25, .5, .75]}}
    rows = [_row(pooling, i, 10 - i) for pooling in overnight.POOLINGS for i in range(4)]
    monkeypatch.setattr(overnight, "read_record", lambda path: (
        None if str(path).endswith(("middle.json", "final.json", "selection.json")) else
        {"id": str(path).split("/")[-1].removesuffix(".json")}))
    calls = []

    def evaluate(proposal, steps, *args, **kwargs):
        calls.append((proposal["id"], steps))
        return next(row for row in rows if row["id"] == proposal["id"])

    monkeypatch.setattr(camp, "evaluate_trial", evaluate)
    middle = camp.promote(rows, [], {}, stage="middle", top_count=3, rung_index=1, fraction=.65)
    assert [name for name, _ in calls] == [f"{pool}_{i:03d}" for i in range(3) for pool in overnight.POOLINGS]
    assert all(steps == 20 for _, steps in calls)
    calls.clear()
    final = camp.promote(middle, [], {}, stage="final", top_count=1, rung_index=2, fraction=.75)
    assert [name for name, _ in calls] == [f"{pool}_000" for pool in overnight.POOLINGS]
    assert all(steps == 30 for _, steps in calls)

    camp.budget.seconds_left = lambda fraction: 0.
    camp.base_decoder = {"base": True}
    reference = {"decoder": {"reference": True}, "scores": {"objective": 0.1}}
    monkeypatch.setattr(overnight, "read_record", read_record)
    selected = camp.select(final, reference)
    assert selected["frozen_before_confirmation"] is True
    assert selected["members"][0]["id"] == "ball_attention_000"
    assert read_record(tmp_path / "selection.json") == selected
    changed = [_row("mean", 0, 100.)]
    assert camp.select(changed, {"decoder": {}, "scores": {"objective": 200.}}) == selected


def test_augmentation_anchors_cover_all_eleven_single_operations():
    singles = {policy for pooling in overnight.POOLINGS
               for policy in overnight.ANCHORS[pooling] if policy in OPS}
    assert singles == set(OPS)


def test_middle_protects_anchors_and_discards_unbalanced_partial_round(tmp_path, monkeypatch):
    camp = _campaign(tmp_path)
    camp.args = Namespace(decoder_trials=(3, 3, 3))
    camp.protocol = {"resources": {"rungs": [10, 20, 30]}}
    rows = [_row(pooling, i, i) for pooling in overnight.POOLINGS for i in range(4)]
    original = overnight.read_record
    monkeypatch.setattr(overnight, "read_record", lambda path: (
        {"id": path.stem} if path.parent.name == "proposals" else original(path)))
    calls = []

    def evaluate(proposal, *args, **kwargs):
        calls.append(proposal["id"])
        if len(calls) == 5:
            raise BudgetExceeded("middle phase cutoff")
        return next(row for row in rows if row["id"] == proposal["id"])

    monkeypatch.setattr(camp, "evaluate_trial", evaluate)
    result = camp.promote(rows, [], {}, stage="middle", top_count=3, rung_index=1, fraction=.65)
    assert [row["id"] for row in result] == [f"{pooling}_000" for pooling in overnight.POOLINGS]
    stage = original(tmp_path / "middle.json")
    assert stage["complete_rounds"] == 1
    assert not stage["planned_stage_completed"]
    assert stage["closed_reason"] == "stage_wall_clock_cutoff"


def test_optuna_proposal_regenerates_from_complete_history(tmp_path):
    history = optuna.create_study(direction="maximize")
    for index in range(6):
        trial = history.ask()
        overnight.sample_spec(trial, "mean")
        history.tell(trial, float(index) / 10)
    first = overnight._proposal(
        history, "mean", 8, {"training_args": {}}, tmp_path / "a.json", 30)
    second = overnight._proposal(
        history, "mean", 8, {"training_args": {}}, tmp_path / "b.json", 30)
    assert first["params"] == second["params"]
    assert first["spec"] == second["spec"]
    assert first["distributions"] == second["distributions"]


def test_predictions_reject_overlap_and_incomplete_period_grid(tmp_path, monkeypatch):
    camp = _campaign(tmp_path)
    camp.protocol = "test-protocol"
    camp.fold = {"inner": [{"train": ["a", "b"], "validation": ["b", "c"]}]}
    member = {"id": "candidate", "spec": _spec(), "steps": 5}
    complete = [{"match": match, "period": period, "scenario": scenario}
                for match in ("b", "c") for period in (1, 2)
                for scenario in overnight.WEIGHTS]
    with pytest.raises(ValueError, match="overlap"):
        camp.predictions(member, 1, complete, fraction=1.)

    camp.fold = {"inner": [{"train": ["a"], "validation": ["b", "c"]}]}
    # Same match set but one missing half: this must fail before fitting/predicting.
    inputs = [{"match": "b", "period": 1, "scenario": "clean"},
              {"match": "c", "period": 1, "scenario": "clean"},
              {"match": "c", "period": 2, "scenario": "clean"}]
    camp.fit = lambda *args, **kwargs: (object(), {"checkpoint_sha256": "x"})
    monkeypatch.setattr(overnight, "predict_records", lambda model, rows, *a, **k: rows)
    with pytest.raises(ValueError, match="grid"):
        camp.predictions(member, 1, inputs, fraction=1.)


def test_fit_resumes_across_rungs_and_reuses_completed_lower_checkpoint(tmp_path):
    camp = _campaign(tmp_path)
    spec = _spec(total=5)
    identity = {"protocol_sha256": camp.protocol_sha, "spec_sha256": overnight._digest(spec),
                "train_matches": ["a", "b"], "seed": 17, "name": "candidate"}
    directory = tmp_path / "fits" / "candidate"
    model2, fit2 = camp.fit("candidate", spec, ["a", "b"], 17, 2, directory, 1.)
    assert fit2["steps"] == 2
    model4, fit4 = camp.fit("candidate", spec, ["a", "b"], 17, 4, directory, 1.)
    assert fit4["steps"] == 4
    # A repeated completed rung returns the checkpoint without re-entering fit.
    reused, again = camp.fit("candidate", spec, ["a", "b"], 17, 2, directory, 1.)
    assert again == fit2
    for key, value in model2.state_dict().items():
        torch.testing.assert_close(value, reused.state_dict()[key], rtol=0, atol=0)
    expected = overnight.build_session(camp.data_dir, spec, ["a", "b"], 17, camp.cache)
    expected.advance_to(4)
    for key, value in model4.state_dict().items():
        torch.testing.assert_close(value, expected.model.state_dict()[key], rtol=0, atol=0)
    assert fit4["identity"] == identity
