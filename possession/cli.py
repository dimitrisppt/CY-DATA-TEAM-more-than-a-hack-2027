"""Command line entry points for the reproducible possession pipeline."""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from games import GAME, GAMES, require
from possession.calibrate import apply_temperature_calibration
from possession.decode import decode_half_details, emissions_from_probabilities
from possession.io_atd import read_atd
from possession.provenance import runtime_provenance
from possession.recipe import load_frozen_recipe, resolve_config
from possession.reference import ReferenceConfig, reference_scores
from possession.storage import sha256_file, write_json

TARGET_SPANS = ((11185, 90652), (110808, 190198))


def write_submission(path, codes, *, n_expected):
    """Atomically write exactly one validated H/A/D/X label per video frame."""
    codes = np.asarray(codes)
    if (codes.shape != (n_expected,) or not np.issubdtype(codes.dtype, np.integer)
            or not np.isin(codes, [0, 1, 2, 3]).all()):
        raise ValueError("predictions must be integer H/A/D/X codes, one per expected frame")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".csv.tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write("frame,label\n")
            handle.writelines(f"{i},{'HADX'[code]}\n" for i, code in enumerate(codes))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def infer_submission(raw_path, metadata_path, output_path, *, n_frames=199319,
                     spans=TARGET_SPANS, checkpoint=None, device="cpu", switch_cost=None,
                     boundary_scale=None, calibration=None, neural_weight=None,
                     center_size=512, halo=256, recipe=None):
    """Infer each half independently. This function never opens target labels."""
    started = time.perf_counter()
    overrides = {key: value for key, value in {
        "switch_cost": switch_cost, "boundary_scale": boundary_scale,
        "neural_weight": neural_weight,
        "calibration": calibration.to_dict() if calibration is not None else None,
    }.items() if value is not None}
    output_path = Path(output_path)
    frozen = None
    if recipe is not None:
        frozen, checkpoint = load_frozen_recipe(recipe, checkpoint)
    inputs = {"raw": Path(raw_path), "metadata": Path(metadata_path)}
    if recipe is not None:
        inputs["recipe"] = Path(recipe)
    if checkpoint:
        inputs["checkpoint"] = Path(checkpoint)
    destinations = {output_path.resolve(), output_path.with_suffix(".manifest.json").resolve()}
    if any(path.resolve() in destinations for path in inputs.values()):
        raise ValueError("output must not overwrite an input")
    last_stop = 0
    for start, stop in spans:
        if not (last_stop <= start < stop <= n_frames):
            raise ValueError("half spans must be ordered, disjoint and inside the timeline")
        last_stop = stop
    if not spans:
        raise ValueError("at least one half span is required")
    if frozen is not None:
        from possession.deployment import verify_selected_checkpoint
        integrity = verify_selected_checkpoint(recipe, frozen, checkpoint)
    else:
        integrity = {"status": "standalone_checkpoint" if checkpoint else "no_selected_recipe",
                     "selection_verified": False}
    model = None
    saved_training = {}
    if checkpoint:
        from possession.predict import load_model_checkpoint
        model, payload = load_model_checkpoint(checkpoint, device=device)
        saved_training = payload["training_config"]
    if frozen is not None:
        saved_training = {**saved_training, **{key: frozen[key] for key in
                                             ("decoder", "calibration", "ensemble")}}
    selected = resolve_config(saved_training, switch_cost=switch_cost,
                              boundary_scale=boundary_scale, calibration=calibration,
                              neural_weight=neural_weight, has_model=model is not None)
    switch_cost, boundary_scale = selected.switch_cost, selected.boundary_scale
    cfg = selected.calibration
    frames = read_atd(raw_path, metadata_path, n_frames=n_frames,
                      pitch=GAMES[GAME]["pitch"], fps=GAMES[GAME]["fps"], graded_spans=spans)
    reference_config = ReferenceConfig()
    codes = np.full(n_frames, 3, dtype=np.int8)
    half_reports = []
    for period, (start, stop) in enumerate(spans, 1):
        half = frames.subset(start, stop)
        half.period = period
        if model is None or selected.neural_weight == 0:
            scores = reference_scores(half, reference_config)
        else:
            from possession.predict import predict_model
            scores = predict_model(half, model, device=device, center_size=center_size, halo=halo)
            if selected.neural_weight < 1:
                from possession.ensemble import combine_scores
                scores = combine_scores(scores, reference_scores(half, reference_config),
                                        selected.neural_weight)
        live, home, boundaries = apply_temperature_calibration(
            scores["live_probability"], scores["home_probability"], scores["boundary"], cfg)
        result = decode_half_details(emissions_from_probabilities(live, home),
                                     boundaries * boundary_scale, switch_cost)
        codes[start:stop] = result.path
        half_reports.append({"period": period, "span": [start, stop],
                             "switches": result.switches, "exact_ties": result.tie_count,
                             "ball_observed_fraction": float(half.ball_valid.mean())})
    write_submission(output_path, codes, n_expected=n_frames)
    manifest = {
        "schema_version": 1, "ground_truth_used": False,
        "selection_integrity": integrity,
        "inference_overrides": overrides,
        "backend": "neural" if model is not None else "untrained_reference",
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                   for name, path in inputs.items()},
        "output": {"path": str(output_path.resolve()), "sha256": sha256_file(output_path),
                   "rows": n_frames, "label_counts": dict(zip("HADX", np.bincount(codes, minlength=4).tolist()))},
        "config": {"switch_cost": switch_cost, "boundary_scale": boundary_scale,
                   "ensemble": {"neural_weight": selected.neural_weight},
                   "calibration": cfg.to_dict(), "reference": reference_config.to_dict(),
                   "device": device, "center_size": center_size, "halo": halo},
        "halves": half_reports, "elapsed_seconds": time.perf_counter() - started,
        "runtime": runtime_provenance(),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("infer", help="write a challenge submission from tactical ATD")
    infer.add_argument("--data-dir", type=Path)
    infer.add_argument("--raw", type=Path)
    infer.add_argument("--metadata", type=Path)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--recipe", type=Path, help="frozen_recipe.json; loads its selected.pt when needed")
    prepare = commands.add_parser("prepare", help="convert public DFL XMLs into local caches")
    prepare.add_argument("--raw-dir", type=Path, default=Path("data/external/dfl"))
    prepare.add_argument("--output", type=Path, default=Path("data/processed/dfl"))
    prepare.add_argument("--matches", nargs="+")
    train = commands.add_parser("train", help="fit on explicitly selected external matches")
    train.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--train-matches", nargs="+", required=True)
    train.add_argument("--validation-matches", nargs="+", default=[])
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--width", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--boundary-weight", type=float, default=.25)
    train.add_argument("--boundary-pos-weight", type=float, default=1.0)
    train.add_argument("--pooling", choices=("mean", "ball_attention"), default="mean")
    train.add_argument("--no-augment", action="store_true")
    train.add_argument("--outage-policy", choices=("legacy", "correlated"), default="legacy")
    evaluate = commands.add_parser("evaluate", help="score on held-out external matches")
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    evaluate.add_argument("--matches", nargs="+", required=True)
    evaluate.add_argument("--report", type=Path, required=True)
    tune = commands.add_parser("tune", help="Optuna model search with nested whole-match folds")
    tune.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    tune.add_argument("--output", type=Path, required=True)
    tune.add_argument("--outer-test", help="one match ID; omit to run all seven outer studies")
    tune.add_argument("--matches", nargs="+")
    tune.add_argument("--trials", type=int, default=8, help="total trials, including resumed trials")
    tune.add_argument("--rungs", type=int, nargs="+", default=[600, 1200])
    tune.add_argument("--poolings", nargs="+", choices=("mean", "ball_attention"),
                      default=["mean", "ball_attention"])
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--prune", action="store_true",
                      help="enable successive halving after assessing rung rank stability")
    tune.add_argument("--cache-gb", type=float, default=2.0)
    tune.add_argument("--outage-policy", choices=("legacy", "correlated"), default="legacy",
                      help="serialized training corruption policy")
    select = commands.add_parser("select", help="tune decoder/blend, freeze, refit and test one outer fold")
    select.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    select.add_argument("--study-dir", type=Path, required=True, help="directory for one outer match")
    select.add_argument("--decoder-trials", type=int, default=32)
    select.add_argument("--seed", type=int, default=0)
    select.add_argument("--cache-gb", type=float, default=2.0)
    select.add_argument("--no-outer-evaluation", action="store_true")
    finalize = commands.add_parser("finalize", help="select by OOF on all training matches, refit and export")
    finalize.add_argument("--data-dir", type=Path, default=Path("data/processed/dfl"))
    finalize.add_argument("--output", type=Path, required=True, help="final deployment bundle directory")
    finalize.add_argument("--matches", nargs="+")
    finalize.add_argument("--trials", type=int, default=8)
    finalize.add_argument("--rungs", type=int, nargs="+", default=[600, 1200])
    finalize.add_argument("--decoder-trials", type=int, default=32)
    finalize.add_argument("--seed", type=int, default=0)
    finalize.add_argument("--cache-gb", type=float, default=2.0)
    finalize.add_argument("--outage-policy", choices=("legacy", "correlated"), default="legacy")
    finalize.add_argument("--prune", action="store_true")
    export = commands.add_parser("export", help="verify and package an already selected model recipe")
    export.add_argument("--study-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for command in (train, tune, finalize):
        augmentation = command.add_mutually_exclusive_group()
        augmentation.add_argument("--augmentation-policy", default="none",
                                  choices=("none", "spatial", "temporal", "combined"),
                                  help="optional tracking augmentation preset")
        augmentation.add_argument("--augmentation-config", type=Path,
                                  help="JSON policy with operators, magnitudes and warmup")
    for command in (infer, train, evaluate, tune, select, finalize):
        command.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    for command in (infer, train, evaluate, tune, finalize):
        command.add_argument("--center-size", type=int, default=512)
    for command in (infer, train, evaluate):
        command.add_argument("--halo", type=int, default=256)
    for command in (infer, evaluate):
        command.add_argument("--checkpoint", type=Path)
        command.add_argument("--switch-cost", type=float, help="override saved decoder cost (default: 2)")
        command.add_argument("--boundary-scale", type=float, help="override saved boundary scale (default: 0)")
        command.add_argument("--neural-weight", type=float, help="override saved neural/reference blend")
    args = parser.parse_args(argv)
    try:
        if args.command in ("train", "tune", "finalize"):
            from possession.augmentation import normalize_augmentation
            augmentation_config = normalize_augmentation(
                json.loads(args.augmentation_config.read_text()) if args.augmentation_config
                else {"mode": args.augmentation_policy}).to_dict()
        if args.command == "infer":
            if bool(args.raw) != bool(args.metadata):
                raise ValueError("provide both --raw and --metadata, or use --data-dir/MTAH_DATA_DIR")
            raw, metadata = ((args.raw, args.metadata) if args.raw else
                             require(["atd_raw", "atd_meta"], data_dir=args.data_dir))
            report = infer_submission(raw, metadata, args.output, checkpoint=args.checkpoint,
                                      device=args.device, switch_cost=args.switch_cost,
                                      boundary_scale=args.boundary_scale,
                                      neural_weight=args.neural_weight,
                                      recipe=args.recipe,
                                      center_size=args.center_size, halo=args.halo)
            print(json.dumps(report["output"], indent=2))
        elif args.command == "prepare":
            from experiments.prepare import prepare_dfl
            prepare_dfl(args.raw_dir, args.output, matches=args.matches)
            print(f"Prepared data: {args.output}")
        elif args.command == "train":
            from possession.train import train_model
            train_model(args.data_dir, args.output, train_matches=args.train_matches,
                        validation_matches=args.validation_matches, steps=args.steps,
                        seed=args.seed, device=args.device, width=args.width,
                        center_size=args.center_size, halo=args.halo,
                        learning_rate=args.learning_rate, boundary_weight=args.boundary_weight,
                        weight_decay=args.weight_decay, boundary_pos_weight=args.boundary_pos_weight,
                        pooling=args.pooling,
                        augment=not args.no_augment, outage_config={"mode": args.outage_policy},
                        augmentation_config=augmentation_config)
            print(f"Training report and checkpoint: {args.output}")
        elif args.command == "evaluate":
            from experiments.evaluate import evaluate_prepared
            report = evaluate_prepared(args.data_dir, matches=args.matches,
                                       checkpoint=args.checkpoint, device=args.device,
                                       switch_cost=args.switch_cost, boundary_scale=args.boundary_scale,
                                       neural_weight=args.neural_weight,
                                       center_size=args.center_size, halo=args.halo)
            write_json(args.report, report)
            print(f"Evaluation report: {args.report}")
        elif args.command == "tune":
            from experiments.tune import run_study
            run_study(args.data_dir, args.output, outer_test=args.outer_test,
                      matches=args.matches, n_trials=args.trials, rungs=tuple(args.rungs),
                      seed=args.seed, device=args.device, poolings=tuple(args.poolings),
                      pruning=args.prune, center_size=args.center_size,
                      cache_bytes=int(args.cache_gb * 1_000_000_000),
                      outage_config={"mode": args.outage_policy},
                      augmentation_config=augmentation_config)
            print(f"Optuna studies and OOF predictions: {args.output}")
        elif args.command == "finalize":
            from experiments.finalize import run_finalization
            run_finalization(args.data_dir, args.output, matches=args.matches,
                             n_trials=args.trials, rungs=tuple(args.rungs),
                             decoder_trials=args.decoder_trials, seed=args.seed,
                             device=args.device, center_size=args.center_size,
                             cache_bytes=int(args.cache_gb * 1_000_000_000),
                             outage_config={"mode": args.outage_policy}, pruning=args.prune,
                             augmentation_config=augmentation_config)
            print(f"Final deployment bundle: {args.output}")
        elif args.command == "export":
            from possession.deployment import export_bundle
            export_bundle(args.study_dir, args.output)
            print(f"Verified deployment bundle: {args.output}")
        elif args.command == "select":
            from experiments.select import select_and_evaluate
            select_and_evaluate(args.data_dir, args.study_dir, n_trials=args.decoder_trials,
                                seed=args.seed, device=args.device,
                                cache_bytes=int(args.cache_gb * 1_000_000_000),
                                evaluate_outer=not args.no_outer_evaluation)
            print(f"Frozen selection and refit: {args.study_dir}")
    except ModuleNotFoundError as error:
        if error.name not in ("torch", "optuna", "sklearn"):
            raise
        print("Error: install model/tuning dependencies with uv sync --extra learn --extra tune "
              "(include --extra dev for testing).", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
