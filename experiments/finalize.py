"""Select on all training-match OOF predictions, refit, and export a bundle.

The reserved ``__final__`` outer ID is orchestration only. It has no prepared
halves and must never be interpreted as an independent performance evaluation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

from experiments import select, tune
from experiments.cross_validation import _match_id, nested_splits
from possession.storage import write_json
from possession.training_session import HalfCache

FINAL_ID = "__final__"
POOLINGS = ("mean", "ball_attention")


@contextmanager
def _locked(study_root: Path):
    study_root.mkdir(parents=True, exist_ok=True)
    with (study_root / "finalization.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("finalization already has a writer") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _check_existing_output(output_dir: Path, *, fingerprint_sha256: str, matches: list[str]):
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError("existing bundle path is not a directory")
    manifest_path = output_dir / "deployment_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("existing bundle lacks deployment manifest")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("predictor") not in {
            "reference", "selected_neural"}:
        raise ValueError("existing bundle manifest is invalid")
    expected = {"frozen_recipe.json", "refit_progress.json", "deployment_manifest.json"}
    if manifest["predictor"] == "selected_neural":
        expected.add("selected.pt")
    if {path.name for path in output_dir.iterdir()} != expected:
        raise ValueError("existing bundle artifact set differs")
    frozen = json.loads((output_dir / "frozen_recipe.json").read_text())
    if (not isinstance(frozen, dict)
            or frozen.get("hpo_fingerprint_sha256") != fingerprint_sha256
            or frozen.get("outer_train") != sorted(matches)
            or frozen.get("outer_test") != FINAL_ID
            or manifest.get("selection_fingerprint") != frozen.get("selection_fingerprint")):
        raise ValueError("existing bundle belongs to a different finalization")


def _final_fold(matches, seed):
    folds = nested_splits([*matches, FINAL_ID], seed=seed)
    fold, = (item for item in folds if item["outer_test"] == [FINAL_ID])
    if fold["outer_train"] != sorted(matches):
        raise AssertionError("final fold does not train on every real match")
    if sorted(match for inner in fold["inner"] for match in inner["validation"]) != fold[
        "outer_train"
    ]:
        raise AssertionError("final OOF validation does not cover every real match")
    return fold


def run_finalization(
    data_dir, output_dir, *, matches=None, n_trials=8, rungs=(600, 1200),
    decoder_trials=32, seed=0, device="cpu", center_size=512,
    cache_bytes=2_000_000_000, outage_config=None, augmentation_config=None,
    pruning=False,
):
    """Run all-match OOF selection and export a deployment bundle.

    ``n_trials`` is the exact neural HPO count, with both architecture
    baselines included. ``decoder_trials`` is the exact count per pooling.
    Resume requires the same configuration and unchanged input/source hashes.
    """
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    study_root = output_dir.parent / f"{output_dir.name}.study"
    study_dir = study_root / FINAL_ID
    if (data_dir == output_dir or data_dir == study_root
            or data_dir in output_dir.parents or data_dir in study_root.parents
            or output_dir in data_dir.parents or study_root in data_dir.parents):
        raise ValueError("data, bundle, and study directories must be separate")
    ids = [_match_id(value) for value in (matches if matches is not None
                                         else tune._discover_matches(data_dir))]
    if len(ids) < 3 or len(ids) != len(set(ids)) or FINAL_ID in ids:
        raise ValueError("at least three distinct real match IDs are required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < len(POOLINGS):
        raise ValueError("n_trials must cover both pooling baselines at full budget")
    if (not rungs or any(not isinstance(rung, int) or isinstance(rung, bool) or rung < 1
                         for rung in rungs)
            or tuple(rungs) != tuple(sorted(set(rungs)))):
        raise ValueError("rungs must be strictly increasing positive integers")
    if not isinstance(decoder_trials, int) or isinstance(decoder_trials, bool) \
            or decoder_trials < 3:
        raise ValueError("decoder_trials must be at least three")
    if not isinstance(center_size, int) or isinstance(center_size, bool) or center_size < 1:
        raise ValueError("center_size must be positive")
    if not isinstance(cache_bytes, int) or isinstance(cache_bytes, bool) or cache_bytes < 1:
        raise ValueError("cache_bytes must be positive")
    if not isinstance(pruning, bool):
        raise TypeError("pruning must be a boolean")
    rungs = tuple(rungs)
    outage_config = tune._outage_dict(outage_config)
    augmentation_config = tune._augmentation_dict(augmentation_config)
    fold = _final_fold(ids, seed)
    split = {
        "matches": sorted(ids), "seed": seed, "inner_folds": 3,
        "outer_folds": [fold], "scope": "all_training_matches",
        "evaluation_performed": False,
        "reserved_outer_id": FINAL_ID,
        "reserved_outer_id_role": "orchestration only; no prepared data or performance estimate",
    }
    fingerprint = tune._fingerprint(
        data_dir, fold, split, seed=seed, rungs=rungs, device=device,
        poolings=POOLINGS, pruning=pruning, center_size=center_size,
        cache_bytes=cache_bytes, outage_config=outage_config,
        augmentation_config=augmentation_config,
    )
    plan = {
        "protocol": "all-training-match OOF final selection v1",
        "scope": "all_training_matches", "evaluation_performed": False,
        "selection_score_role": "OOF selection criterion, not an independent performance estimate",
        "matches": sorted(ids), "reserved_outer_id": FINAL_ID,
        "n_trials": n_trials, "rungs": list(rungs),
        "decoder_trials_per_pooling": decoder_trials, "seed": seed,
        "device": str(device), "center_size": center_size,
        "cache_bytes": cache_bytes, "pruning": pruning,
        "outage_config": outage_config, "augmentation_config": augmentation_config,
        "study_fingerprint_sha256": fingerprint["sha256"],
    }
    with _locked(study_root):
        plan_path = study_dir / "finalization_plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("finalization plan or input fingerprint changed; refusing resume")
        else:
            if study_dir.exists() and any(study_dir.iterdir()):
                raise ValueError("existing final study lacks finalization plan; refusing resume")
        _check_existing_output(output_dir, fingerprint_sha256=fingerprint["sha256"],
                               matches=ids)
        if not plan_path.exists():
            write_json(plan_path, plan)
        cache = HalfCache(data_dir, max_bytes=cache_bytes)
        summary = tune._run_outer(
            data_dir, study_root, fold, n_trials=n_trials, rungs=rungs, seed=seed,
            device=device, poolings=POOLINGS, pruning=pruning,
            center_size=center_size, cache_bytes=cache_bytes, cache=cache, split=split,
            outage_config=outage_config, augmentation_config=augmentation_config,
        )
        del cache
        if summary["outer_train"] != sorted(ids) or set(summary["best_by_pooling"]) != set(POOLINGS):
            raise ValueError("final HPO did not cover all matches and both pooling families")
        selected = select.select_and_evaluate(
            data_dir, study_dir, n_trials=decoder_trials, seed=seed, device=device,
            cache_bytes=cache_bytes, evaluate_outer=False,
        )
        frozen = selected["frozen_recipe"]
        if (selected["outer_report"] is not None or frozen["outer_train"] != sorted(ids)
                or frozen["outer_test"] != FINAL_ID
                or (study_dir / "outer_report.json").exists()):
            raise ValueError("final selection unexpectedly evaluated or excluded a training match")
        from possession.deployment import export_bundle

        bundle = export_bundle(study_dir, output_dir)
        return {
            "protocol": plan["protocol"], "scope": plan["scope"],
            "evaluation_performed": False,
            "selection_score_role": plan["selection_score_role"],
            "matches": sorted(ids), "study_dir": str(study_dir),
            "bundle_dir": str(bundle["bundle_dir"]),
            "manifest_path": str(bundle["manifest_path"]),
            "bundle_reused": bundle["reused"],
            "selected_pooling": frozen["selected_pooling"],
            "selection_score": frozen["selection_score"],
            "refit_progress": selected["refit_progress"],
            "study_fingerprint_sha256": fingerprint["sha256"],
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--matches", nargs="+")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--rungs", nargs="+", type=int, default=[600, 1200])
    parser.add_argument("--decoder-trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--center-size", type=int, default=512)
    parser.add_argument("--cache-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--outage-policy", choices=("legacy", "correlated"), default="legacy")
    parser.add_argument("--augmentation-policy", choices=("none", "spatial", "temporal", "combined"),
                        default="none")
    parser.add_argument("--augmentation-config", type=Path,
                        help="JSON object with fixed augmentation settings")
    args = parser.parse_args(argv)
    augmentation_config = (json.loads(args.augmentation_config.read_text())
                           if args.augmentation_config else {"mode": args.augmentation_policy})
    report = run_finalization(
        args.data_dir, args.output_dir, matches=args.matches, n_trials=args.n_trials,
        rungs=tuple(args.rungs), decoder_trials=args.decoder_trials, seed=args.seed,
        device=args.device, center_size=args.center_size, cache_bytes=args.cache_bytes,
        pruning=args.pruning, outage_config={"mode": args.outage_policy},
        augmentation_config=augmentation_config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
