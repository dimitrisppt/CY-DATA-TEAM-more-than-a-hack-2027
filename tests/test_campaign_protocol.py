import pytest

from experiments.campaign_protocol import WEIGHTS, Budget, paired_gate, publish, read_record
from experiments.cross_validation import nested_splits


def _scores(delta=0.0, *, clean_delta=0.0):
    matches = [f"M{i}" for i in range(6)]
    by_match = {
        match: {
            "clean": {"spell_f1_mean": .5 + clean_delta, "transition_f1_at_12": .4},
            "short": {"spell_f1_mean": .35 + delta, "transition_f1_at_12": .3 + delta},
            "long": {"spell_f1_mean": .3 + delta, "transition_f1_at_12": .25 + delta},
            "replay": {"spell_f1_mean": .32 + delta, "transition_f1_at_12": .27 + delta},
        }
        for match in matches
    }
    means = {
        name: {"spell_f1_mean": value, "transition_f1_at_12": transition}
        for name, value, transition in (
            ("clean", .5 + clean_delta, .4), ("short", .35 + delta, .3 + delta),
            ("long", .3 + delta, .25 + delta), ("replay", .32 + delta, .27 + delta))
    }
    objective = sum(WEIGHTS[name] * means[name]["spell_f1_mean"] for name in WEIGHTS)
    return {"by_match": by_match, "means": means, "objective": objective}


def test_budget_reuses_original_deadline_and_rejects_extension(tmp_path):
    first = Budget(tmp_path, 12, now=1_000)
    restarted = Budget(tmp_path, 12, now=99_999)
    assert restarted.clock == first.clock
    assert restarted.clock["deadline_unix"] == 1_000 + 12 * 3600
    with pytest.raises(ValueError, match="at most 12 hours"):
        Budget(tmp_path, 12.5, now=99_999)


@pytest.mark.parametrize("fraction", [0, -1, 1.01, float("nan"), float("inf")])
def test_budget_stage_cannot_extend_deadline(tmp_path, fraction):
    budget = Budget(tmp_path, 12, now=1_000)
    with pytest.raises(ValueError, match="fraction"):
        budget.check(fraction)


def test_wmx_is_absent_from_every_training_and_validation_split():
    matches = ["DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "A", "B", "C", "D", "E"]
    fold = next(row for row in nested_splits(matches, inner_folds=3, seed=0)
                if row["outer_test"] == ["DFL-MAT-J03WMX"])
    assert fold["outer_test"] == ["DFL-MAT-J03WMX"]
    for inner in fold["inner"]:
        assert "DFL-MAT-J03WMX" not in inner["train"]
        assert "DFL-MAT-J03WMX" not in inner["validation"]


def test_publish_is_immutable_and_read_validates_hash(tmp_path):
    path = tmp_path / "record.json"
    assert publish(path, {"trial": 2, "score": .4}) == {"trial": 2, "score": .4}
    assert read_record(path) == {"trial": 2, "score": .4}
    with pytest.raises(ValueError, match="completed campaign record differs"):
        publish(path, {"trial": 2, "score": .5})
    path.write_text('{"sha256":"bad","payload":{}}')
    with pytest.raises(ValueError, match="hash differs"):
        read_record(path)


def test_paired_gate_fails_when_clean_regresses_beyond_limit():
    baselines = [_scores(), _scores()]
    candidates = [_scores(.02, clean_delta=-.02), _scores(.02, clean_delta=-.02)]
    result = paired_gate(baselines, candidates)
    assert not result["passed"]
    assert not result["checks"]["clean_preserved"]


def test_paired_gate_rejects_unmatched_match_sets():
    baseline = _scores()
    candidate = _scores(.02)
    del candidate["by_match"]["M5"]
    with pytest.raises(ValueError, match="same six development matches"):
        paired_gate([baseline, baseline], [candidate, _scores(.02)])


def test_paired_gate_requires_both_confirmation_seeds_to_improve():
    baselines = [_scores(), _scores()]
    candidates = [_scores(.02), _scores(-.005)]
    result = paired_gate(baselines, candidates)
    assert not result["passed"]
    assert result["checks"]["objective_improved"]
    assert not result["checks"]["both_seeds_improved"]
