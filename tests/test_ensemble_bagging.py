import json

import numpy as np
import pytest

from experiments import campaign_protocol
from experiments import ensemble_bagging as bagging
from experiments.ensemble_polish_search import blend_loaded
from possession.storage import sha256_file


def _score(scenario_offsets=None, match_offsets=None):
    scenario_offsets = scenario_offsets or {}
    match_offsets = match_offsets or {}
    matches = [f"M{i}" for i in range(6)]
    base = {"clean": .60, "short": .43, "long": .40, "replay": .42}
    by_match = {}
    for match in matches:
        by_match[match] = {}
        for name, value in base.items():
            offset = scenario_offsets.get(name, 0.) + match_offsets.get((match, name), 0.)
            by_match[match][name] = {
                "spell_f1_mean": value + offset,
                "transition_f1_at_12": .30 + offset,
            }
    means = {
        name: {"spell_f1_mean": base[name] + scenario_offsets.get(name, 0.),
               "transition_f1_at_12": .30 + scenario_offsets.get(name, 0.)}
        for name in base}
    objective = sum(campaign_protocol.WEIGHTS[name] * means[name]["spell_f1_mean"]
                    for name in base)
    return {"by_match": by_match, "means": means, "objective": objective}


def test_fresh_mask_batches_use_new_short_long_seeds_and_reused_clean_replay():
    first, second = bagging.MASK_BATCHES
    a, b = bagging._scenarios(first), bagging._scenarios(second)
    assert (a["short"]["seed"], a["long"]["seed"]) == (73104, 74104)
    assert (b["short"]["seed"], b["long"]["seed"]) == (73105, 74105)
    assert a["short"]["missing_fraction"] == b["short"]["missing_fraction"] == .41
    assert a["long"]["missing_fraction"] == b["long"]["missing_fraction"] == .41
    assert a["short"]["min_seconds"] == b["short"]["min_seconds"] == 1.
    assert a["short"]["max_seconds"] == b["short"]["max_seconds"] == 10.
    assert a["long"]["min_seconds"] == b["long"]["min_seconds"] == 10.
    assert a["long"]["max_seconds"] == b["long"]["max_seconds"] == 30.
    assert a["clean"] == b["clean"]
    assert a["replay"] == b["replay"] == {"kind": "target_replay", "seed": 5102}


def test_seed_averaging_then_architecture_blend_uses_correct_weights():
    def record(live, owner, boundary):
        return {"match": "M", "period": 1, "scenario": "clean", "input_sha256": "same",
                "model_identity": {"id": str(live)},
                "arrays": {"frame_id": np.array([0]), "gt": np.array([1]),
                           "ball_valid": np.array([True]),
                           "nearest_ball_seconds": np.array([0.]),
                           "ref_x": np.array([1.]), "neural_live": np.array([live]),
                           "neural_home": np.array([owner]),
                           "neural_boundary": np.full((1, 3, 3), boundary)}}
    seeds = [[record(.8, .7, 2.), record(.4, .3, 4.)],
             [record(.6, .5, 6.), record(.2, .1, 8.)]]
    avg_left = blend_loaded([seeds[0][0]], [seeds[1][0]], .5)
    avg_right = blend_loaded([seeds[0][1]], [seeds[1][1]], .5)
    pair = blend_loaded(avg_left, avg_right, .75)[0]["arrays"]
    logit = lambda value: np.log(value) - np.log1p(-value)
    expected = 1 / (1 + np.exp(-(.75 * (logit(.8) + logit(.6)) / 2
                                    + .25 * (logit(.4) + logit(.2)) / 2)))
    assert pair["neural_live"][0] == pytest.approx(expected)
    assert pair["neural_boundary"][0, 0, 0] == pytest.approx(.75 * 4. + .25 * 6.)


def test_gate_passes_only_when_all_scenario_match_and_transition_checks_pass():
    baseline = [_score(), _score()]
    candidate = _score({"clean": .001, "short": .01, "long": .01, "replay": .01})
    result = bagging._checks(baseline, candidate)
    assert result["passed"]
    assert result["checks"]["beats_both_seed_baselines_by_margin"]
    assert result["checks"]["four_matches_missing_improved"]

    one_bad_match = _score({"clean": .001, "short": .01, "long": .01, "replay": .01},
                           {("M0", "short"): -.2, ("M1", "long"): -.2,
                            ("M2", "replay"): -.2})
    failed = bagging._checks(baseline, one_bad_match)
    assert not failed["checks"]["four_matches_missing_improved"]


def test_gate_rejects_scenario_loss_and_missing_transition_regression():
    baseline = [_score(), _score()]
    loss = _score({"clean": .004, "short": -.02, "long": .01, "replay": .01})
    failed = bagging._checks(baseline, loss)
    assert not failed["checks"]["short_preserved"]

    transition = _score({"clean": .005, "short": .005, "long": .005, "replay": .005})
    for score in (transition,):
        for match in score["by_match"].values():
            for scenario in ("short", "long", "replay"):
                match[scenario]["transition_f1_at_12"] -= .02
        for scenario in ("short", "long", "replay"):
            score["means"][scenario]["transition_f1_at_12"] -= .02
    failed = bagging._checks(baseline, transition)
    assert not failed["checks"]["missing_transitions_preserved"]


def test_outer_guard_compares_against_the_worse_of_two_seed_scores():
    scores = [{"objective": .55}, {"objective": .58}]
    assert bagging._outer_checks(scores, {"objective": .57})["passed"]
    failed = bagging._outer_checks(scores, {"objective": .569})
    assert not failed["passed"]
    assert "beats_best_outer_seed_within_margin" in failed["checks"]


def test_pair_selection_rejects_member_drift():
    member = {"id": "a", "spec": {"model_kwargs": {}}, "steps": 7200}
    with pytest.raises(ValueError, match="two-member pair"):
        bagging._selection_pair({"members": [member], "decoder": {}, "blend_weight": None},
                               "candidate")


def _auth_fixture(tmp_path):
    campaign_dir, polish_dir = tmp_path / "campaign", tmp_path / "polish"
    data_dir, target_dir = tmp_path / "data", tmp_path / "target"
    data_dir.mkdir()
    target_dir.mkdir()
    (data_dir / "prepared.npz").write_bytes(b"prepared")
    target_manifest = target_dir / "input-manifest.json"
    target_manifest.write_text("input-only")
    member = {"id": "member_a", "spec": {"model_kwargs": {"pooling": "mean"}},
              "steps": 7200}
    baseline = {"members": [member, {**member, "id": "member_b"}],
                "decoder": {"temperature": 1.}, "blend_weight": .5}
    candidate = {**baseline, "decoder": {"temperature": 1.2}}
    env = {"versions": {"python": "test"}}
    prepared = {"status": "verified_manifest", "caches": [{"cache": "prepared.npz"}]}
    campaign_protocol = {"environment": env, "source_sha256": {"x.py": "source"},
                         "prepared": prepared,
                         "target_input_manifest_sha256": sha256_file(target_manifest)}
    campaign_protocol_sha = bagging._digest(campaign_protocol)
    campaign_summary = {"status": "complete", "candidate_eligible_for_review": True,
                        "protocol_sha256": campaign_protocol_sha, "selection": baseline}
    campaign_protocol_record = {"payload": campaign_protocol,
                                "sha256": campaign_protocol_sha}
    polish_protocol = {"campaign_protocol_sha256": campaign_protocol_sha,
                       "baseline_selection": baseline, "source_sha256": {"x.py": "source"},
                       "runtime": env}
    polish_protocol_sha = bagging._digest(polish_protocol)
    polish_selection = {"selection": candidate,
                        "protocol_sha256": polish_protocol_sha,
                        "baseline_selection_sha256": bagging._digest(baseline)}
    diagnostic = {"available": True, "not_independent_confirmation_seeds": True,
                  "score": {"objective": .6}, "objective_delta_vs_mean_singles": .01}
    polish_summary = {"status": "complete", "protocol_sha256": polish_protocol_sha,
                      "selection": candidate,
                      "confirmation": {"candidate_selection_sha256": bagging._digest(candidate),
                                       "baseline_selection_sha256": bagging._digest(baseline),
                                       "seed_bagging": diagnostic}}
    for directory, values in ((campaign_dir, {
            "protocol.json": campaign_protocol_record,
            "selection.json": {"payload": baseline, "sha256": bagging._digest(baseline)},
            "summary.json": {"payload": campaign_summary,
                             "sha256": bagging._digest(campaign_summary)}}),
            (polish_dir, {
                "protocol.json": {"payload": polish_protocol,
                                  "sha256": polish_protocol_sha},
                "selection.json": {"payload": polish_selection,
                                   "sha256": bagging._digest(polish_selection)},
                "summary.json": {"payload": polish_summary,
                                 "sha256": bagging._digest(polish_summary)}})):
        directory.mkdir(parents=True)
        for name, value in values.items():
            (directory / name).write_text(json.dumps(value))
    return campaign_dir, polish_dir, data_dir, target_dir, prepared, env


@pytest.mark.parametrize("failure", ["source", "runtime", "prepared"])
def test_authentication_rejects_changed_source_runtime_or_prepared_data(
        monkeypatch, tmp_path, failure):
    campaign, polish, data, target, prepared, _env = _auth_fixture(tmp_path)
    monkeypatch.setattr(bagging, "_source_hashes",
                        lambda: {"x.py": "wrong" if failure == "source" else "source"})
    monkeypatch.setattr(bagging, "_execution_environment",
                        lambda: {"versions": {"python": "wrong" if failure == "runtime" else "test"}})
    monkeypatch.setattr(bagging, "verify_prepared",
                        lambda *_: {**prepared, "status": "different"}
                        if failure == "prepared" else prepared)
    message = {"source": "source snapshot", "runtime": "runtime", "prepared": "prepared public data"}[failure]
    with pytest.raises(ValueError, match=message):
        bagging._authenticated_inputs(campaign, polish, data, target)


def test_resume_batch_rejects_different_input_manifest_identity(tmp_path):
    batch = {"short_seed": 73104, "long_seed": 74104}
    row = {"protocol_sha256": "protocol", "baseline_selection_sha256": "base",
           "candidate_selection_sha256": "candidate", "inner_inputs_sha256": "changed",
           "outer_inputs_sha256": "outer", "candidate_selection": {"members": []},
           "mask_seeds": {"short": 73104, "long": 74104, "replay": 5102}}
    with pytest.raises(ValueError, match="saved batch result identity"):
        bagging._verify_batch_result(row, "protocol", "base", "candidate", "fresh", "outer",
                                     tmp_path, {"fold": {"outer_train": []}}, [], batch)




def test_resume_rejects_done_summary_that_drifted_from_batch_cache():
    baseline = {"members": [], "decoder": {"x": 1}}
    candidate = {"members": [], "decoder": {"x": 2}}
    row = {"mask_seeds": {"short": 73104, "long": 74104, "replay": 5102},
           "baseline": [_score(), _score()], "candidate": _score(),
           "checks": {"passed": True},
           "outer": {"baseline": [{"objective": .5}, {"objective": .5}],
                     "candidate": {"objective": .5}, "checks": {"passed": True}}}
    done = bagging._final_payload([row, row], baseline, candidate, "protocol")
    assert bagging._validate_completed_result(done, [row, row], baseline, candidate,
                                              "protocol") == done
    changed = {**done, "batches": [dict(done["batches"][0]), done["batches"][1]]}
    changed["batches"][0]["candidate"] = _score({"clean": .02})
    with pytest.raises(ValueError, match="completed result differs"):
        bagging._validate_completed_result(changed, [row, row], baseline, candidate,
                                           "protocol")
