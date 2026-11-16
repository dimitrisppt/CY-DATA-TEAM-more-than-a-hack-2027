import json
import random

import numpy as np
import pytest
import torch

from experiments import campaign_protocol, campaign_training
from experiments.ema_training import _digest, _tensor_state_digest, _update_ema, replay_tail_ema
from possession.predict import load_model_checkpoint
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file


def _half(match, n=32):
    t = np.arange(n, dtype=np.int64)
    players = np.zeros((n, 2, 2, 2), np.float32)
    players[:, 0, :, 0] = 20 + t[:, None] * .05
    players[:, 1, :, 0] = 70 - t[:, None] * .05
    players[..., 1] = 30
    state = np.resize(np.array([0, 0, 1, 1, 2, 2], np.int8), n)
    live = (state != 2).astype(np.int8)
    frames = FrameTable(t, players, np.ones((n, 2, 2), bool),
                        np.column_stack((25 + t * .1, np.full(n, 31))),
                        np.ones(n, bool), np.ones(n, bool), match_id=match)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _fixture(tmp_path, *, total_steps=5, endpoint_delta=0.):
    data = tmp_path / "prepared"
    data.mkdir()
    for match in ("a", "b", "c"):
        save_half(data / f"{match}_half1.npz", *_half(match))
    spec = {
        "model_kwargs": {"player_dim": 8, "global_dim": 9, "width": 8,
                         "pooling": "mean", "dilations": [1]},
        "training_args": {"learning_rate": .003, "weight_decay": .0001,
                          "boundary_weight": .25, "boundary_pos_weight": 2.,
                          "center_size": 8, "augment": False},
        "outage_config": {"mode": "legacy"}, "augmentation_config": {"mode": "none"},
        "schedule": {"kind": "constant", "warmup_steps": 0,
                     "min_lr_ratio": 1., "total_steps": total_steps},
    }
    identity = {"protocol_sha256": "test-protocol", "spec_sha256": _digest(spec),
                "train_matches": ["a", "b"], "seed": 17, "name": "test_member"}
    state_dir = tmp_path / "source_fit" / "rung_5"
    state_dir.mkdir(parents=True)
    raw = campaign_training.build_session(data, spec, ["a", "b"], 17)
    raw.advance_to(total_steps - 2)
    start = campaign_training.save_training_state(
        state_dir / f"state_{total_steps - 2:06d}.pt", raw, identity)
    raw.advance_to(total_steps)
    endpoint = campaign_training.save_training_state(
        state_dir / f"state_{total_steps:06d}.pt", raw, identity)
    if endpoint_delta:
        payload = torch.load(endpoint, map_location="cpu", weights_only=True)
        key = next(iter(payload["state"]["model"]))
        payload["state"]["model"][key].view(-1)[0] += endpoint_delta
        torch.save(payload, endpoint)
        endpoint.with_suffix(endpoint.suffix + ".sha256.json").write_text(
            json.dumps({"sha256": sha256_file(endpoint)}))
    checkpoint = raw.save_checkpoint(tmp_path / "baseline.pt", validation_matches=["c"])
    del raw
    return data, spec, identity, start, checkpoint, state_dir


def _run(tmp_path, **kwargs):
    data, spec, identity, start, checkpoint, state_dir = _fixture(tmp_path, **kwargs)
    output = tmp_path / "ema_fit"
    fit = replay_tail_ema(
        data_dir=data, spec=spec, identity=identity, state_path=start,
        baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
        output_dir=output, validation_matches=["c"], decay=.9,
        start_steps=3, end_steps=5)
    return fit, output, data, spec, identity, start, checkpoint, state_dir


def test_tail_ema_replay_matches_raw_end_weights_and_progress_before_export(tmp_path):
    fit, output, _, _, _, _, checkpoint, _ = _run(tmp_path)
    assert fit["steps"] == 5
    assert fit["tail_ema"]["updates"] == 2
    assert fit["raw_parity"]["raw_weights_match_baseline_checkpoint"]
    assert fit["raw_parity"]["raw_training_progress_matches_endpoint"]
    assert fit["checkpoint_sha256"] == sha256_file(output / "model.pt")
    _, raw = load_model_checkpoint(checkpoint)
    _, averaged = load_model_checkpoint(output / "model.pt")
    assert any(not torch.equal(raw["state_dict"][name], averaged["state_dict"][name])
               for name in raw["state_dict"])


def test_ema_update_averages_float_tensors_and_copies_nonfloating_buffers():
    ema = {"weight": torch.tensor([2., 4.]), "running": torch.tensor([8.]),
           "count": torch.tensor(2, dtype=torch.int64),
           "enabled": torch.tensor([False])}
    raw = {"weight": torch.tensor([6., 8.]), "running": torch.tensor([4.]),
           "count": torch.tensor(5, dtype=torch.int64),
           "enabled": torch.tensor([True])}
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    torch_state = torch.get_rng_state().clone()
    _update_ema(ema, raw, .5)
    torch.testing.assert_close(ema["weight"], torch.tensor([4., 6.]), rtol=0, atol=0)
    torch.testing.assert_close(ema["running"], torch.tensor([6.]), rtol=0, atol=0)
    assert ema["count"].item() == 5
    assert ema["enabled"].item()
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert python_state == random.getstate()
    assert torch.equal(torch_state, torch.get_rng_state())


def test_successful_output_is_idempotent_and_hash_verified(tmp_path):
    fit, output, data, spec, identity, start, checkpoint, _ = _run(tmp_path)
    repeated = replay_tail_ema(
        data_dir=data, spec=spec, identity=identity, state_path=start,
        baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
        output_dir=output, validation_matches=["c"], decay=.9,
        start_steps=3, end_steps=5)
    assert repeated == fit
    (output / "model.pt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checkpoint hash or path"):
        replay_tail_ema(
            data_dir=data, spec=spec, identity=identity, state_path=start,
            baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
            output_dir=output, validation_matches=["c"], decay=.9,
            start_steps=3, end_steps=5)


def test_rejects_overlapping_or_mismatched_fit_ancestry(tmp_path):
    data, spec, identity, start, checkpoint, _ = _fixture(tmp_path)
    args = {"data_dir": data, "spec": spec, "identity": identity,
            "state_path": start, "baseline_checkpoint": checkpoint,
            "baseline_sha256": sha256_file(checkpoint),
            "output_dir": tmp_path / "ema_fit", "validation_matches": ["c"],
            "decay": .9, "start_steps": 3, "end_steps": 5}
    with pytest.raises(ValueError, match="overlap"):
        replay_tail_ema(**{**args, "validation_matches": ["a"]})
    wrong_identity = {**identity, "spec_sha256": "wrong"}
    with pytest.raises(ValueError, match="spec hash"):
        replay_tail_ema(**{**args, "identity": wrong_identity})
    with pytest.raises(ValueError, match="baseline checkpoint SHA256"):
        replay_tail_ema(**{**args, "baseline_sha256": "wrong"})


@pytest.mark.parametrize("changes", [
    {"identity": "train"}, {"validation_matches": ["different_validation"]},
])
def test_rejects_train_or_validation_split_that_differs_from_checkpoint(tmp_path, changes):
    data, spec, identity, start, checkpoint, _ = _fixture(tmp_path)
    args = {"data_dir": data, "spec": spec, "identity": identity,
            "state_path": start, "baseline_checkpoint": checkpoint,
            "baseline_sha256": sha256_file(checkpoint),
            "output_dir": tmp_path / "ema_fit", "validation_matches": ["c"],
            "decay": .9, "start_steps": 3, "end_steps": 5}
    if changes.get("identity") == "train":
        args["identity"] = {**identity, "train_matches": ["a"]}
    if "validation_matches" in changes:
        args["validation_matches"] = changes["validation_matches"]
    with pytest.raises(ValueError, match="checkpoint config, spec, or split"):
        replay_tail_ema(**args)
    assert not (tmp_path / "ema_fit" / "fit.json").exists()


def test_corrupt_endpoint_raw_parity_fails_without_publishing_fit(tmp_path):
    data, spec, identity, start, checkpoint, _ = _fixture(tmp_path, endpoint_delta=.1)
    output = tmp_path / "ema_fit"
    with pytest.raises(ValueError, match="exactly match authenticated end state"):
        replay_tail_ema(
            data_dir=data, spec=spec, identity=identity, state_path=start,
            baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
            output_dir=output, validation_matches=["c"], decay=.9,
            start_steps=3, end_steps=5)
    assert not (output / "fit.json").exists()


def test_corrupt_start_state_checksum_fails_without_publishing_fit(tmp_path):
    data, spec, identity, start, checkpoint, _ = _fixture(tmp_path)
    start.with_suffix(start.suffix + ".sha256.json").write_text('{"sha256":"bad"}')
    output = tmp_path / "ema_fit"
    with pytest.raises(ValueError, match="state SHA256"):
        replay_tail_ema(data_dir=data, spec=spec, identity=identity, state_path=start,
                        baseline_checkpoint=checkpoint,
                        baseline_sha256=sha256_file(checkpoint), output_dir=output,
                        validation_matches=["c"], decay=.9, start_steps=3, end_steps=5)
    assert not (output / "fit.json").exists()


def test_output_path_cannot_overlap_source_state_or_checkpoint(tmp_path):
    data, spec, identity, start, checkpoint, state_dir = _fixture(tmp_path)
    with pytest.raises(ValueError, match="cannot contain or replace a source state"):
        replay_tail_ema(
            data_dir=data, spec=spec, identity=identity, state_path=start,
            baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
            output_dir=state_dir, validation_matches=["c"], decay=.9,
            start_steps=3, end_steps=5)
    assert not (state_dir / "fit.json").exists()


def test_completed_fit_resume_rechecks_raw_parity_evidence(tmp_path):
    _fit, output, data, spec, identity, start, checkpoint, _ = _run(tmp_path)
    fit_path = output / "fit.json"
    payload = campaign_protocol.read_record(fit_path)
    payload["raw_parity"]["raw_training_progress_matches_endpoint"] = False
    fit_path.write_text(json.dumps({"sha256": _digest(payload), "payload": payload}))
    with pytest.raises(ValueError, match="raw parity evidence"):
        replay_tail_ema(
            data_dir=data, spec=spec, identity=identity, state_path=start,
            baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
            output_dir=output, validation_matches=["c"], decay=.9,
            start_steps=3, end_steps=5)


def test_completed_fit_resume_rechecks_checkpoint_training_provenance(tmp_path):
    _fit, output, data, spec, identity, start, checkpoint, _ = _run(tmp_path)
    model_path = output / "model.pt"
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    payload["training_config"]["seed"] = 99
    torch.save(payload, model_path)
    fit_path = output / "fit.json"
    fit_payload = campaign_protocol.read_record(fit_path)
    fit_payload["checkpoint_sha256"] = sha256_file(model_path)
    fit_path.write_text(json.dumps({"sha256": _digest(fit_payload), "payload": fit_payload}))
    with pytest.raises(ValueError, match="training provenance"):
        replay_tail_ema(
            data_dir=data, spec=spec, identity=identity, state_path=start,
            baseline_checkpoint=checkpoint, baseline_sha256=sha256_file(checkpoint),
            output_dir=output, validation_matches=["c"], decay=.9,
            start_steps=3, end_steps=5)


def test_tensor_digest_accepts_zero_dimensional_integer_buffer():
    digest = _tensor_state_digest({"counter": torch.tensor(3, dtype=torch.int64)})
    assert len(digest) == 64
