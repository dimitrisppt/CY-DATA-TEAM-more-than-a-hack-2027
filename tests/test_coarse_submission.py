import copy

import numpy as np
import pytest
import torch

from experiments import campaign_protocol, coarse_context, coarse_stability, coarse_submission
from experiments.robustness import _execution_environment, _source_hashes
from experiments.tune import _seed_for_fold
from possession.model import PossessionModel
from possession.storage import sha256_file


def _selection():
    spec = {"model_kwargs": {"pooling": "mean"},
            "outage_config": {"mode": "legacy"}}
    return {"kind": "convex_pair", "members": [
        {"id": "mean_007", "spec": spec, "steps": 7200},
        {"id": "ball_attention_002", "spec": spec, "steps": 7200}],
        "blend_weight": .619862, "decoder": {"version": 1}}


def _score(offset=0.):
    names = ("clean", "short", "long", "replay")
    by_match = {f"match_{index}": {name: {"spell_f1_mean": .4 + offset,
                                           "transition_f1_at_12": .3 + offset}
                                   for name in names} for index in range(6)}
    return {"by_match": by_match,
            "means": {name: {"spell_f1_mean": .4 + offset,
                             "transition_f1_at_12": .3 + offset} for name in names},
            "objective": .5 + offset}


def _write_study(root, selection, *, passed=True):
    spec = coarse_stability._spec(selection)
    protocol = {"schema": "coarse_stability_v1", "baseline_kind": "ema",
                "candidate_selection": selection, "baseline_selection": selection,
                "model_spec": spec, "coarse_derivation_sha256": "derivation",
                "source_prepared_sha256": "prepared", "ema_protocol_sha256": "ema-protocol",
                "runtime": _execution_environment(), "source_sha256": _source_hashes()}
    protocol_sha = coarse_stability._digest(protocol)
    campaign_protocol.publish(root / "protocol.json", protocol)
    batches = []
    for index in (1, 2):
        raw, candidate = _score(), _score(.01 if passed else 0.)
        inner = coarse_stability._gate(raw, candidate)
        outer = coarse_stability._outer_gate(raw, candidate)
        if not passed:
            inner["passed"] = False
        artifact = root / "artifacts" / f"batch_{index}.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f"artifact-{index}")
        row = {"schema": "coarse_stability_v1", "protocol_sha256": protocol_sha,
               "batch": f"batch_{index}", "baseline_kind": "ema",
               "baseline_scores": {"inner": raw, "outer": raw},
               "candidate_scores": {"inner": candidate, "outer": candidate},
               "inner_checks": inner, "outer_checks": outer,
               "artifact_sha256": {artifact.relative_to(root).as_posix(): sha256_file(artifact)}}
        campaign_protocol.publish(root / "batches" / row["batch"] / "result.json", row)
        batches.append(row)
    result = {"schema": "coarse_stability_v1", "protocol_sha256": protocol_sha,
              "baseline_kind": "ema", "baseline_selection": selection,
              "candidate_selection": selection, "batches": batches, "eligible": passed}
    campaign_protocol.publish(root / "result.json", result)
    return protocol, result


def test_study_verifier_recomputes_both_gates_and_artifact_hashes(tmp_path):
    selection = _selection()
    study = tmp_path / "study"
    _write_study(study, selection)
    ema = {"candidate_selection": selection, "ema_protocol_sha256": "ema-protocol"}
    protocol, result, references = coarse_submission._verify_study(study, ema)
    assert result["eligible"] is True
    assert len(references["artifacts"]) == 2
    assert protocol["baseline_kind"] == "ema"


def test_study_verifier_rejects_failed_gate_and_artifact_drift(tmp_path):
    selection = _selection()
    study = tmp_path / "study"
    _write_study(study, selection, passed=False)
    ema = {"candidate_selection": selection, "ema_protocol_sha256": "ema-protocol"}
    with pytest.raises(ValueError):
        coarse_submission._verify_study(study, ema)
    study = tmp_path / "tampered"
    _write_study(study, selection)
    artifact = study / "artifacts" / "batch_1.txt"
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="artifact hash"):
        coarse_submission._verify_study(study, ema)


def test_derived_cache_requires_all_seven_halves_and_hashes(tmp_path, monkeypatch):
    data = tmp_path / "derived"
    data.mkdir()
    names = {f"{match}_half{period}.npz" for match in coarse_submission._FINAL_FIT_MATCHES
             for period in (1, 2)}
    hashes = {}
    for name in sorted(names):
        path = data / name
        path.write_bytes(name.encode())
        hashes[name] = sha256_file(path)
    sources = {"match": {"source_sha256": "source"}}
    source_prepared = {"sources": sources}
    provenance = {"status": "verified_manifest", "sources": sources,
                  "caches": [{"cache": name} for name in names]}
    manifest = data / "manifest.json"
    manifest.write_text("manifest")
    derivation = {"schema": "coarse_context_derivation_v1",
                  "processor": coarse_context.PROCESSOR,
                  "source_prepared": source_prepared,
                  "derived_cache_sha256": hashes,
                  "derived_manifest_sha256": sha256_file(manifest)}
    coarse_context._publish_derivation(data / "coarse-derivation.json", derivation)
    monkeypatch.setattr(coarse_submission, "verify_prepared", lambda *_: provenance)
    result = coarse_submission._verify_derived_data(
        data, coarse_context._digest(derivation), coarse_context._digest(source_prepared))
    assert result[2] == hashes
    (data / min(names)).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache hashes"):
        coarse_submission._verify_derived_data(
            data, coarse_context._digest(derivation), coarse_context._digest(source_prepared))


def test_bundle_file_list_supports_exact_portable_copy(tmp_path):
    source_root, bundle_root = tmp_path / "source", tmp_path / "bundle"
    source_files = {
        "ema_bundle/manifest.json": b"nested EMA manifest",
        "proof/coarse-study/artifacts/fits/final/model.pt": b"study checkpoint proof",
        "proof/final-fit-protocol.json": b"final fit protocol",
    }
    files = {}
    for relative, contents in source_files.items():
        source = source_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(contents)
        target = bundle_root / relative
        digest = sha256_file(source)
        coarse_submission._copy_verified(source, target, digest)
        files[relative] = digest
    assert set(files) == set(source_files)
    coarse_submission._verify_file_map(bundle_root, files)
    protocol = bundle_root / "proof/final-fit-protocol.json"
    protocol.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file hash"):
        coarse_submission._verify_file_map(bundle_root, files)


def test_standalone_verifier_rejects_changed_pinned_source_or_runtime(monkeypatch):
    manifest = {"source_sha256": {"experiments/coarse_stability.py": "before"},
                "runtime": {"python": "pinned"}}
    monkeypatch.setattr(coarse_submission, "_source_hashes",
                        lambda: {"experiments/coarse_stability.py": "after",
                                 "experiments/new_file.py": "allowed addition"})
    monkeypatch.setattr(coarse_submission, "_execution_environment",
                        lambda: {"python": "pinned"})
    with pytest.raises(ValueError, match="source code differs"):
        coarse_submission._verify_source_runtime(manifest)
    manifest["source_sha256"]["experiments/coarse_stability.py"] = "after"
    monkeypatch.setattr(coarse_submission, "_execution_environment",
                        lambda: {"python": "changed"})
    with pytest.raises(ValueError, match="environment differs"):
        coarse_submission._verify_source_runtime(manifest)


def _write_fit(tmp_path, *, wrong_seed=False, wrong_split=False, wrong_spec=False):
    base_seed = 20261116
    actual_seed = _seed_for_fold(base_seed, "DFL-MAT-J03WMX", 0)
    spec = dict(coarse_stability._spec(_selection()))
    stored_spec = copy.deepcopy(spec)
    if wrong_spec:
        stored_spec["training_args"]["learning_rate"] *= 2
    fit_seed = actual_seed + 1 if wrong_seed else actual_seed
    model = PossessionModel(**spec["model_kwargs"])
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    prepared_hashes = {f"{match}_half{period}.npz": f"hash-{match}-{period}"
                       for match in coarse_submission._FINAL_FIT_MATCHES for period in (1, 2)}
    provenance = {"status": "verified_manifest", "caches": []}
    config = {"campaign_spec": stored_spec, "steps": 7200,
              "train_matches": list(coarse_submission._FINAL_FIT_MATCHES),
              "validation_matches": ["DFL-MAT-J03WMX"] if wrong_split else [],
              "seed": fit_seed, "coarse_derivation_sha256": "derivation",
              "coarse_protocol_sha256": "protocol", "coarse_scope": "final",
              "coarse_fold": 0, "coarse_seed": base_seed}
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": state, "model_kwargs": spec["model_kwargs"],
                "training_config": config, "prepared_sha256": prepared_hashes,
                "dataset_provenance": provenance}, checkpoint)
    identity = {"protocol_sha256": "protocol", "spec_sha256": coarse_stability._digest(spec),
                "train_matches": list(coarse_submission._FINAL_FIT_MATCHES),
                "seed": fit_seed, "name": "coarse_player_only"}
    fit = {"schema": "coarse_member_fit_v1", "identity": identity, "steps": 7200,
           "derivation_sha256": "derivation", "checkpoint_sha256": sha256_file(checkpoint)}
    fit_path = tmp_path / "fit.json"
    campaign_protocol.publish(fit_path, fit)
    return checkpoint, fit_path, spec, base_seed, prepared_hashes, provenance


@pytest.mark.parametrize("options", [{"wrong_seed": True}, {"wrong_split": True},
                                      {"wrong_spec": True}])
def test_final_fit_verifier_rejects_seed_spec_or_validation_leak(tmp_path, options):
    checkpoint, fit, spec, base_seed, hashes, provenance = _write_fit(tmp_path, **options)
    with pytest.raises(ValueError):
        coarse_submission._verify_final_fit(
            checkpoint, fit, spec=spec, protocol_sha="protocol",
            derivation_sha="derivation", base_seed=base_seed,
            prepared_hashes=hashes,
            prepared_provenance_sha256=coarse_stability._digest(provenance))


def test_final_fit_verifier_accepts_mapped_seed_and_empty_validation(tmp_path):
    checkpoint, fit, spec, base_seed, hashes, provenance = _write_fit(tmp_path)
    row = coarse_submission._verify_final_fit(
        checkpoint, fit, spec=spec, protocol_sha="protocol", derivation_sha="derivation",
        base_seed=base_seed, prepared_hashes=hashes,
        prepared_provenance_sha256=coarse_stability._digest(provenance))
    assert row["actual_seed"] == _seed_for_fold(base_seed, "DFL-MAT-J03WMX", 0)


def test_inference_keeps_outside_frames_x_and_resets_each_half(tmp_path, monkeypatch):
    bundle_dir, target_dir = tmp_path / "bundle", tmp_path / "target"
    bundle_dir.mkdir()
    target_dir.mkdir()
    (bundle_dir / "manifest.json").write_text("manifest")
    selection = _selection()
    ema_members = {str(seed): {member: {"checkpoint": f"members/{seed}/{member}.pt"}
                               for member in ("mean_007", "ball_attention_002")}
                   for seed in coarse_submission.SEEDS}
    manifest = {"candidate_selection": selection,
                "final_fits": {}, "coarse_spec": coarse_stability._spec(selection)}
    verified = {"manifest": manifest,
                "ema_bundle": {"manifest": {"members": ema_members}},
                "coarse_model_paths": {seed: f"coarse-{seed}.pt"
                                       for seed in coarse_submission.SEEDS}}
    monkeypatch.setattr(coarse_submission, "verify_coarse_bundle", lambda _: verified)
    monkeypatch.setattr(coarse_submission, "load_model_checkpoint", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(coarse_submission.campaign_submission, "_verified_inputs",
                        lambda _: {"raw": {"path": "raw.atd"},
                                   "metadata": {"path": "meta.json"}})

    class Frames:
        def __init__(self, n):
            self.frame_id = np.arange(n)
            self.ball_valid = np.zeros(n, dtype=bool)
            self.fps = 25.

        def subset(self, start, stop):
            return Frames(stop - start)

    monkeypatch.setattr(coarse_submission, "N_FRAMES", 10)
    monkeypatch.setattr(coarse_submission, "TARGET_SPANS", ((1, 4), (6, 9)))
    monkeypatch.setattr(coarse_submission, "read_atd", lambda *args, **kwargs: Frames(10))
    monkeypatch.setattr(coarse_submission.campaign_submission, "N_FRAMES", 10)
    monkeypatch.setattr(coarse_submission.campaign_submission, "TARGET_SPANS", ((1, 4), (6, 9)))
    monkeypatch.setattr(coarse_submission, "_predict_half",
                        lambda frames, fine, coarse, selected:
                        np.zeros(len(frames.frame_id), dtype=np.int8))
    output = tmp_path / "submission.csv"
    report = coarse_submission.infer_coarse_submission(bundle_dir, target_dir, output)
    assert len(output.read_text().splitlines()) == 11
    assert [line.split(",")[1] for line in output.read_text().splitlines()[1:]] == [
        "X", "H", "H", "H", "X", "X", "H", "H", "H", "X"]
    assert report["format_validated"] is True
