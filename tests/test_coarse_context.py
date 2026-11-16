import json

import numpy as np
import pytest
import torch

from experiments.coarse_context import (
    COARSE_MODEL_KWARGS,
    _digest,
    blend_coarse_context,
    coarsen_half,
    predict_coarse_native,
    prepare_coarse_data,
)
from possession.features import build_features
from possession.model import PossessionModel
from possession.prepared import verify_prepared
from possession.schema import FrameTable, TrainingTargets
from possession.storage import load_half, save_half, sha256_file


def _half(match="DFL-MAT-J00001", period=1, n=76, *, start=1000):
    frame_id = np.arange(n, dtype=np.int64) + start
    players = np.zeros((n, 2, 3, 2), np.float32)
    players[:, 0, :, 0] = 20 + np.arange(n)[:, None] * .08
    players[:, 1, :, 0] = 80 - np.arange(n)[:, None] * .04
    players[..., 1] = np.array([20., 30., 40.])[None, None, :]
    ball = np.column_stack((30 + np.arange(n) * .02, np.full(n, 31))).astype(np.float32)
    ball_valid = np.ones(n, bool)
    ball_valid[7:20] = False
    state = np.resize(np.array([0, 0, 1, 1, 2, 2], np.int8), n)
    live = (state != 2).astype(np.int8)
    frames = FrameTable(frame_id, players, np.ones((n, 2, 3), bool), ball,
                        ball_valid, np.ones(n, bool), fps=25., match_id=match,
                        period=period)
    targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                              live.astype(bool), state, np.ones(n, bool))
    return frames, targets


def _derived_source(tmp_path):
    data = tmp_path / "source"
    data.mkdir()
    halves = []
    for period in (1, 2):
        path = data / f"DFL-MAT-J00001_half{period}.npz"
        save_half(path, *_half(period=period, n=76, start=(period - 1) * 1000))
        halves.append({"period": period, "cache": path.name,
                       "cache_sha256": sha256_file(path)})
    fingerprint = {"path": "source.csv", "sha256": "a" * 64}
    manifest = {"version": 1, "matches": {"DFL-MAT-J00001": {
        "match_id": "DFL-MAT-J00001",
        "sources": {"positions": fingerprint, "match_info": fingerprint},
        "halves": halves}}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data


def _small_model():
    torch.manual_seed(8)
    return PossessionModel(width=8, dilations=(1, 2, 3), pooling="mean").eval()


def test_coarsen_uses_half_local_exact_anchors_and_samples_targets():
    frames, targets = _half(n=51, start=12345)
    coarse, coarse_targets, anchors = coarsen_half(frames, targets)
    np.testing.assert_array_equal(anchors, [0, 25, 50])
    np.testing.assert_array_equal(coarse.frame_id, [0, 1, 2])
    assert coarse.fps == 1.
    assert coarse.match_id == frames.match_id and coarse.period == frames.period
    np.testing.assert_array_equal(coarse.players_xy, frames.players_xy[anchors])
    np.testing.assert_array_equal(coarse.player_valid, frames.player_valid[anchors])
    assert not coarse.ball_valid.any()
    assert not coarse.ball_xy.any()
    for name in ("live", "live_valid", "owner", "owner_valid", "state", "state_valid"):
        np.testing.assert_array_equal(getattr(coarse_targets, name), getattr(targets, name)[anchors])


def test_coarsen_rejects_wrong_native_rate_and_target_length():
    frames, targets = _half(n=51)
    frames.fps = 24.
    with pytest.raises(ValueError, match="25 Hz"):
        coarsen_half(frames)
    frames.fps = 25.
    short = TrainingTargets(*(getattr(targets, name)[:-1] for name in
                              ("live", "live_valid", "owner", "owner_valid", "state", "state_valid")))
    with pytest.raises(ValueError, match="lengths differ"):
        coarsen_half(frames, short)


@pytest.mark.parametrize("mutation", ["gap", "noninteger"])
def test_coarsen_rejects_mutated_noncontiguous_or_noninteger_native_ids(mutation):
    frames, _ = _half(n=51)
    if mutation == "gap":
        frames.frame_id[1] += 2
    else:
        frames.frame_id = frames.frame_id.astype(np.float64)
    with pytest.raises(ValueError, match="contiguous integers"):
        coarsen_half(frames)


def test_prepare_coarse_data_writes_verified_derivation_and_is_idempotent(tmp_path):
    source = _derived_source(tmp_path)
    output = tmp_path / "coarse"
    record = prepare_coarse_data(source, output)
    assert record["schema"] == "coarse_context_derivation_v1"
    assert record["processor"]["stride"] == 25
    assert len(record["mapping"]) == 2
    paths = sorted(output.glob("DFL-MAT-*_half*.npz"))
    assert len(paths) == 2
    assert verify_prepared(output, paths)["status"] == "verified_manifest"
    for path in paths:
        frames, _ = load_half(path)
        assert frames.fps == 1.
        assert not frames.ball_valid.any()
    assert prepare_coarse_data(source, output) == record
    assert (source / "manifest.json").exists()


def test_prepare_coarse_data_rejects_derived_cache_hash_drift(tmp_path):
    source = _derived_source(tmp_path)
    output = tmp_path / "coarse"
    prepare_coarse_data(source, output)
    path = output / "DFL-MAT-J00001_half1.npz"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="cache hash changed"):
        prepare_coarse_data(source, output)


def test_prepare_coarse_data_rejects_derivation_manifest_tampering(tmp_path):
    source = _derived_source(tmp_path)
    output = tmp_path / "coarse"
    prepare_coarse_data(source, output)
    path = output / "coarse-derivation.json"
    record = json.loads(path.read_text())
    record["payload"]["processor"]["stride"] = 24
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest hash differs"):
        prepare_coarse_data(source, output)


def test_prepare_coarse_data_rejects_manifest_source_fingerprint_drift(tmp_path):
    source = _derived_source(tmp_path)
    output = tmp_path / "coarse"
    prepare_coarse_data(source, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["matches"]["DFL-MAT-J00001"]["sources"]["positions"]["sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    derivation_path = output / "coarse-derivation.json"
    derivation = json.loads(derivation_path.read_text())["payload"]
    derivation["derived_manifest_sha256"] = sha256_file(manifest_path)
    derivation_path.write_text(json.dumps({"sha256": _digest(derivation),
                                           "payload": derivation}))
    with pytest.raises(ValueError, match="source fingerprints differ"):
        prepare_coarse_data(source, output)


def test_coarse_prediction_ignores_all_ball_input_changes():
    frames, _ = _half(n=251)
    changed = frames.subset(0, len(frames.frame_id))
    changed.ball_valid[:] = ~changed.ball_valid
    changed.ball_xy[:] = np.array([90., 50.], dtype=np.float32)
    model = _small_model()
    first = predict_coarse_native(frames, model, center_size=4)
    second = predict_coarse_native(changed, model, center_size=4)
    np.testing.assert_array_equal(first["live_probability"], second["live_probability"])
    np.testing.assert_array_equal(first["home_probability"], second["home_probability"])


def test_chunked_coarse_predictions_match_full_coarse_timeline():
    frames, _ = _half(n=251)
    coarse, _, anchors = coarsen_half(frames)
    features = build_features(coarse)
    model = _small_model()
    inputs = (torch.from_numpy(features.player_tokens).unsqueeze(0),
              torch.from_numpy(features.player_mask).unsqueeze(0),
              torch.from_numpy(features.global_tokens).unsqueeze(0),
              torch.from_numpy(features.time_valid).unsqueeze(0))
    with torch.inference_mode():
        full = model(*inputs)
    prediction = predict_coarse_native(frames, model, center_size=3)
    full_live = full["live_logits"][0].numpy()
    full_home = full["owner_logits"][0].numpy()
    native = np.arange(len(frames.frame_id))
    expected_live = 1. / (1. + np.exp(-np.interp(native, anchors, full_live)))
    expected_home = 1. / (1. + np.exp(-np.interp(native, anchors, full_home)))
    np.testing.assert_allclose(prediction["live_probability"], expected_live, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(prediction["home_probability"], expected_home, atol=1e-6, rtol=1e-6)


def test_coarse_model_preserves_player_permutation_and_team_exchange_symmetry():
    frames, _ = _half(n=126)
    coarse, _, _ = coarsen_half(frames)
    features = build_features(coarse)
    tokens = torch.from_numpy(features.player_tokens).unsqueeze(0)
    mask = torch.from_numpy(features.player_mask).unsqueeze(0)
    context = torch.from_numpy(features.global_tokens).unsqueeze(0)
    valid = torch.from_numpy(features.time_valid).unsqueeze(0)
    model = _small_model()
    expected = model(tokens, mask, context, valid)
    permutation = torch.tensor([2, 0, 1])
    permuted = model(tokens[..., permutation, :], mask[..., permutation], context, valid)
    for name in expected:
        torch.testing.assert_close(permuted[name], expected[name], atol=1e-6, rtol=1e-6)
    swapped = model(tokens.flip(2), mask.flip(2), context, valid)
    torch.testing.assert_close(swapped["live_logits"], expected["live_logits"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(swapped["owner_logits"], -expected["owner_logits"], atol=1e-6, rtol=1e-6)
    order = torch.tensor([1, 0, 2])
    expected_boundary = expected["boundary"].index_select(-2, order).index_select(-1, order)
    torch.testing.assert_close(swapped["boundary"], expected_boundary, atol=1e-6, rtol=1e-6)


def test_blend_is_exact_fine_identity_on_ball_and_preserves_boundary_references():
    fine = {"neural_live": np.array([0., 1., .25]),
            "neural_home": np.array([1., 0., .75]),
            "neural_boundary": np.arange(27.).reshape(3, 3, 3),
            "ball_valid": np.array([True, True, True]),
            "nearest_ball_seconds": np.array([0., 0., 0.]),
            "ref_nearest_m": np.array([2., 3., 4.])}
    result = blend_coarse_context(fine, np.array([1., 0., .5]), np.array([0., 1., .5]))
    np.testing.assert_array_equal(result["neural_live"], fine["neural_live"])
    np.testing.assert_array_equal(result["neural_home"], fine["neural_home"])
    np.testing.assert_array_equal(result["neural_boundary"], fine["neural_boundary"])
    np.testing.assert_array_equal(result["ref_nearest_m"], fine["ref_nearest_m"])
    np.testing.assert_array_equal(fine["neural_live"], [0., 1., .25])


def test_blend_uses_capped_missing_ball_age_as_convex_logit_weight():
    fine = {"neural_live": np.array([.2, .2]), "neural_home": np.array([.8, .8]),
            "neural_boundary": np.zeros((2, 3, 3)),
            "ball_valid": np.array([False, True]),
            "nearest_ball_seconds": np.array([20., 5.])}
    result = blend_coarse_context(fine, np.array([.8, .8]), np.array([.3, .3]))
    alpha = .25
    logit = lambda p: np.log(p) - np.log1p(-p)
    expected = 1. / (1. + np.exp(-((1 - alpha) * logit(.2) + alpha * logit(.8))))
    assert result["neural_live"][0] == pytest.approx(expected)
    assert result["neural_live"][1] == pytest.approx(.2)
    assert result["neural_home"][1] == pytest.approx(.8)


def test_model_defaults_encode_the_declared_255_second_context():
    model = PossessionModel(**COARSE_MODEL_KWARGS)
    assert model.radius == 127
    assert model.receptive_field == 255
