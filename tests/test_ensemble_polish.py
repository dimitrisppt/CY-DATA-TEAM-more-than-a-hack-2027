import copy

import pytest

from experiments import ensemble_polish as polish
from experiments.campaign_protocol import BudgetExceeded, read_record


def _setup(monkeypatch):
    baseline = {"kind": "convex_pair", "members": [{"id": "mean"}, {"id": "attention"}],
                "blend_weight": .5, "decoder": {"setting": 1}, "scores": {"objective": .5}}
    monkeypatch.setattr(polish, "preflight", lambda *args: ({}, baseline, [[], []]))
    monkeypatch.setattr(polish, "sha256_file", lambda path: "fixture")
    monkeypatch.setattr(polish, "_source_hashes", lambda: {"fixture": "hash"})
    monkeypatch.setattr(polish, "_execution_environment", lambda: {"fixture": True})
    return baseline


def test_search_is_pinned_and_selection_frozen_before_confirmation(tmp_path, monkeypatch):
    baseline = _setup(monkeypatch)
    before = copy.deepcopy(baseline)
    output = tmp_path / "output"

    def tune(*args, **kwargs):
        assert read_record(output / "protocol.json")["candidate_count_entering_confirmation"] == 1
        return {"decoder": {"setting": 2}, "blend_weight": .55,
                "scores": {"objective": .6}, "best_trial": 23, "trials_complete": 64}

    def confirm(*args, **kwargs):
        selected = read_record(output / "selection.json")
        assert selected["selection"] == args[5]
        assert args[4] == before
        return {"eligible": False, "gate": {"passed": False},
                "outer": {"gate_passed": True}, "seed_bagging": {}}

    monkeypatch.setattr(polish, "tune_joint", tune)
    monkeypatch.setattr(polish, "confirm_pair", confirm)
    result = polish.run_polish(tmp_path / "campaign", tmp_path / "data",
                               tmp_path / "target", output)
    assert result["status"] == "complete"
    assert result["baseline_replaced"] is False
    assert baseline == before
    monkeypatch.setattr(polish, "tune_joint", lambda *args, **kwargs: pytest.fail("retuned"))
    monkeypatch.setattr(polish, "confirm_pair", lambda *args, **kwargs: pytest.fail("reconfirmed"))
    assert polish.run_polish(tmp_path / "campaign", tmp_path / "data",
                             tmp_path / "target", output) == result


def test_budget_stop_cannot_create_a_completed_summary(tmp_path, monkeypatch):
    _setup(monkeypatch)

    def stop(*args, **kwargs):
        raise BudgetExceeded("test boundary")

    monkeypatch.setattr(polish, "tune_joint", stop)
    monkeypatch.setattr(polish, "confirm_pair", lambda *args, **kwargs: pytest.fail("confirmed"))
    output = tmp_path / "output"
    result = polish.run_polish(tmp_path / "campaign", tmp_path / "data",
                               tmp_path / "target", output)
    assert result["status"] == "budget_reached"
    assert not (output / "summary.json").exists()


def test_search_cannot_write_into_the_original_campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(polish, "preflight", lambda *args: pytest.fail("opened inputs"))
    with pytest.raises(ValueError, match="separate"):
        polish.run_polish(tmp_path / "campaign", tmp_path / "data",
                          tmp_path / "target", tmp_path / "campaign/new")
