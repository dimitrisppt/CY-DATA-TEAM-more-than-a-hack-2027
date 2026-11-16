"""Synthetic-only checks for strict nested planning, resume and budget guards."""

import json

import numpy as np
import pytest

from experiments import timing_nested as nested
from experiments.campaign_protocol import publish
from possession.storage import sha256_file


def _cache():
    matches = [f"DFL-MAT-M{i}" for i in range(6)]
    folds = [
        {
            "train": [m for m in matches if m not in matches[i : i + 2]],
            "validation": matches[i : i + 2],
        }
        for i in (0, 2, 4)
    ]
    return {
        "schema": "public_dfl_timing_cache_v1",
        "exact_v6_parity": True,
        "inner_folds": folds,
        "records": [{"match": m} for m in matches + ["DFL-MAT-GUARD"]],
    }


def test_twelve_recipes_exclude_sample_and_outer_pair():
    cache = _cache()
    plan = nested.make_nested_plan(cache)
    assert len(plan["recipes"]) == 12
    assert len({tuple(r["train_matches"]) for r in plan["recipes"]}) == 12
    for recipe in plan["recipes"]:
        assert len(recipe["train_matches"]) == 3
        assert len(recipe["forbidden_matches"]) == 3
        assert not set(recipe["train_matches"]) & set(recipe["forbidden_matches"])
        fold = cache["inner_folds"][recipe["fold"]]
        assert set(recipe["train_matches"]) | {recipe["calibration_match"]} == set(fold["train"])
        assert set(recipe["assessment_matches"]) == set(fold["validation"])
    assert plan["search"] is False
    assert len(plan["plan"]["historical_selection_matches"]) == 7


def test_plan_rejects_changed_assessment_training_and_missing_guard():
    cache = _cache()
    cache["inner_folds"][0]["train"].pop()
    with pytest.raises(ValueError, match="training set"):
        nested.make_nested_plan(cache)
    cache = _cache()
    cache["records"].pop()
    with pytest.raises(ValueError, match="seven historical"):
        nested.make_nested_plan(cache)


def test_projection_scales_measured_first_fit_and_rejects_invalid_values():
    baseline = nested.projected_training_seconds(nested.HISTORICAL_FIT_SECONDS["mean_007"])
    assert 9 * 3600 < baseline < 10 * 3600
    assert (
        nested.projected_training_seconds(2 * nested.HISTORICAL_FIT_SECONDS["mean_007"])
        == 2 * baseline
    )
    for value in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            nested.projected_training_seconds(value)


def test_actual_seed_and_identity_are_distinct_fresh_nested_fits():
    recipes = nested.make_nested_plan(_cache())["recipes"]
    identities = [
        nested._identity(r, "mean_007", {"recipe": "fixed"}, 8101, "a" * 64) for r in recipes
    ]
    assert len({x["seed"] for x in identities}) == 12
    assert all(
        x["train_matches"] == r["train_matches"] for x, r in zip(identities, recipes, strict=True)
    )
    assert all(x["protocol_sha256"] == "a" * 64 for x in identities)


class _FakeBudget:
    def __init__(self, allowed=None):
        self.allowed, self.calls = allowed, 0

    def check(self):
        self.calls += 1
        if self.allowed is not None and self.calls > self.allowed:
            raise nested.NestedBudgetExceeded("synthetic budget boundary")

    def status(self, *_args, **_kwargs):
        pass


def test_raw_training_resumes_exact_step_schedule_and_keeps_ema_endpoints(tmp_path, monkeypatch):
    import torch

    recipe = nested.make_nested_plan(_cache())["recipes"][0]
    spec = {"recipe": "synthetic fixed"}
    identity = nested._identity(recipe, "mean_007", spec, 8101, "a" * 64)
    advances = []

    class Session:
        steps = 0

        def advance_to(self, steps):
            advances.append(steps)
            self.steps = steps

        def save_checkpoint(self, path, validation_matches, extra_training_config):
            torch.save(
                {
                    "training_config": {
                        "train_matches": recipe["train_matches"],
                        "validation_matches": validation_matches,
                        "seed": identity["seed"],
                        "steps": self.steps,
                        "campaign_spec": spec,
                        **extra_training_config,
                    }
                },
                path,
            )

        def _progress(self):
            return {"trained_halves": [f"{m}_half1.npz" for m in recipe["train_matches"]]}

    def save_state(path, session, saved_identity):
        assert saved_identity == identity
        path.write_text(str(session.steps))
        path.with_suffix(".pt.sha256.json").write_text(json.dumps({"sha256": sha256_file(path)}))

    def load_state(path, session, saved_identity):
        assert saved_identity == identity
        session.steps = int(path.read_text())

    monkeypatch.setattr(nested.campaign_training, "build_session", lambda *_args: Session())
    monkeypatch.setattr(nested.campaign_training, "save_training_state", save_state)
    monkeypatch.setattr(nested.campaign_training, "load_training_state", load_state)
    directory = tmp_path / "raw"
    with pytest.raises(nested.NestedBudgetExceeded):
        nested._train_raw(
            directory, tmp_path, spec, identity, recipe, None, _FakeBudget(2), fine=True
        )
    assert advances == [200, 400]
    fit = nested._train_raw(
        directory, tmp_path, spec, identity, recipe, None, _FakeBudget(), fine=True
    )
    assert advances == list(range(200, 7201, 200))
    assert fit["steps"] == 7200
    assert {p.name for p in nested._committed_states(directory)} == {
        "state_007000.pt",
        "state_007200.pt",
    }
    assert nested._verify_fit(directory, identity, spec, recipe) == fit
    before = advances.copy()
    assert (
        nested._train_raw(
            directory, tmp_path, spec, identity, recipe, None, _FakeBudget(), fine=True
        )
        == fit
    )
    assert advances == before
    payload = torch.load(directory / "model.pt", weights_only=True)
    payload["training_config"]["train_matches"].append(recipe["calibration_match"])
    torch.save(payload, directory / "model.pt")
    # Hash binding rejects changed checkpoint before trusting its ancestry declaration.
    with pytest.raises(ValueError, match="checksum"):
        nested._verify_fit(directory, identity, spec, recipe)
    # Even a matching updated checksum cannot hide held-out training ancestry.
    from experiments.timing_validation import canonical_sha256

    updated = {**fit, "checkpoint_sha256": sha256_file(directory / "model.pt")}
    (directory / "fit.json").write_text(
        json.dumps({"payload": updated, "sha256": canonical_sha256(updated)})
    )
    with pytest.raises(ValueError, match="ancestry"):
        nested._verify_fit(directory, identity, spec, recipe)


def test_pruning_never_uses_uncommitted_state(tmp_path):
    for step in (200, 400, 7000, 7200, 9999):
        path = tmp_path / f"state_{step:06d}.pt"
        path.write_text("synthetic")
        if step != 9999:
            path.with_suffix(".pt.sha256.json").write_text("{}")
    nested._prune_states(tmp_path, fine=True)
    assert [int(p.stem.rsplit("_", 1)[1]) for p in nested._committed_states(tmp_path)] == [
        7000,
        7200,
    ]
    assert (tmp_path / "state_009999.pt").exists()


def test_completed_fit_rejects_wrong_recipe_identity_without_training(tmp_path):
    recipe = nested.make_nested_plan(_cache())["recipes"][0]
    identity = nested._identity(recipe, "mean_007", {"fixed": True}, 8101, "a" * 64)
    publish(
        tmp_path / "fit.json",
        {"identity": {**identity, "train_matches": [recipe["calibration_match"]]}, "steps": 7200},
    )
    with pytest.raises(ValueError, match="identity"):
        nested._verify_fit(tmp_path, identity, {"fixed": True}, recipe)


def test_nested_summary_requires_six_unique_whole_match_assessments():
    results = [{"split": {"kind": "strict_nested", "assessment": ["DFL-MAT-A", "DFL-MAT-B"]}}] * 3
    with pytest.raises(ValueError, match="six unique"):
        nested.summarize_nested(results)


def test_hard_deadline_reports_resumable_stop_and_rejects_larger_budget(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(nested.time, "monotonic", lambda: now[0])
    budget = nested._Budget(tmp_path, 10)
    now[0] = 109.0
    budget.check()
    now[0] = 110.0
    with pytest.raises(nested.NestedBudgetExceeded, match="states preserved"):
        budget.check()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["phase"] == "budget_exhausted_resumable"
    assert not (tmp_path / "result.json").exists()
    for value in (43201, 0, -1, np.inf, np.nan):
        with pytest.raises(ValueError, match="12 hour"):
            nested._Budget(tmp_path, value)


def test_assessment_cache_checks_grid_bindings_inventory_and_outer_exclusion(monkeypatch):
    from dataclasses import asdict

    from experiments.timing_validation import (
        BasePredictionProvenance,
        ComponentProvenance,
        ConfigLineage,
    )

    cache = _cache()
    fold = cache["inner_folds"][0]
    universe = sorted(r["match"] for r in cache["records"])
    components = [
        ComponentProvenance(
            f"{name}_seed_{seed}",
            "coarse_model" if name == "coarse_player_only" else "fine_model",
            fold["train"],
            [],
            "a" * 64,
            "b" * 64,
        )
        for seed in nested.SEEDS
        for name in ("mean_007", "ball_attention_002", "coarse_player_only")
    ]
    components.append(
        ComponentProvenance(
            "fixed_features_reference_blend", "deterministic", [], [], "c" * 64, "c" * 64
        )
    )
    lineage = [ConfigLineage("v6_recipe", "d" * 64, [], universe, True)]
    rows = []
    for batch in ("batch_1", "batch_2"):
        for match in fold["validation"]:
            for half in (1, 2):
                for scenario in nested.SCENARIOS:
                    prov = BasePredictionProvenance(
                        match, half, scenario, "inner_fold_0", "e" * 64, components, lineage
                    )
                    rows.append(
                        {
                            "batch": batch,
                            "match": match,
                            "half": half,
                            "scenario": scenario,
                            "scope": "inner",
                            "fold": 0,
                            "path": "/synthetic/public-cache.npz",
                            "sha256": "e" * 64,
                            "provenance": asdict(prov),
                        }
                    )
    monkeypatch.setattr(nested, "_check_hash", lambda *_args: None)
    assert nested._verify_assessment_rows(rows, fold, 0, universe) == rows
    with pytest.raises(ValueError, match="grid"):
        nested._verify_assessment_rows(rows[:-1], fold, 0, universe)
    rows[0]["provenance"]["feature_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="detached"):
        nested._verify_assessment_rows(rows, fold, 0, universe)
    rows[0]["provenance"]["feature_sha256"] = "e" * 64
    rows[0]["provenance"]["components"][0]["train_matches"] = [fold["validation"][1]]
    with pytest.raises(ValueError, match="leakage"):
        nested._verify_assessment_rows(rows, fold, 0, universe)
