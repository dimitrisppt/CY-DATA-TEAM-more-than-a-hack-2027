"""Synthetic checks for continuation isolation, alignment and exact restart."""

import copy
import json

import numpy as np
import pytest
import torch

from experiments import campaign_training as campaign
from experiments import refinement_training as refinement
from experiments.possession_objectives import complete_spells
from possession.augmentation import AugmentationConfig
from possession.schema import FrameTable, TrainingTargets
from possession.storage import save_half, sha256_file


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    n = 80
    t = np.arange(n)
    players = np.zeros((n, 2, 2, 2), np.float32)
    players[:, 0, :, 0] = 20 + t[:, None] * .05
    players[:, 1, :, 0] = 70 - t[:, None] * .05
    players[..., 1] = 30
    state = np.resize(np.repeat([0, 2, 1], [9, 3, 8]), n).astype(np.int8)
    live = (state != 2).astype(np.int8)
    for match in ("train_a", "train_b", "held_out"):
        frames = FrameTable(t, players, np.ones((n, 2, 2), bool),
                            np.column_stack((25 + t * .1, np.full(n, 31))),
                            np.ones(n, bool), np.ones(n, bool), match_id=match)
        targets = TrainingTargets(live, np.ones(n, bool), (state == 0).astype(np.int8),
                                  live.astype(bool), state, np.ones(n, bool))
        save_half(data / f"{match}_half1.npz", frames, targets)
    return data


def _spec():
    return {
        "model_kwargs": {"player_dim": 8, "global_dim": 9, "width": 8,
                         "pooling": "mean", "dilations": [1]},
        "training_args": {"learning_rate": .003, "weight_decay": .0001,
                          "boundary_weight": .25, "boundary_pos_weight": 2.,
                          "center_size": 24, "augment": True},
        "outage_config": {"mode": "legacy"},
        "augmentation_config": AugmentationConfig(
            mode="combined", operations=("time_shift", "coordinate_noise"),
            clean_probability=.2, strength=.5, warmup_steps=0).to_dict(),
        "schedule": {"kind": "constant", "warmup_steps": 0,
                     "min_lr_ratio": 1., "total_steps": 1000},
    }


def _build(data):
    return campaign.build_session(data, _spec(), ["train_a", "train_b"], 16)


def _raw(data, tmp_path):
    session = _build(data)
    session.advance_to(2)
    path = campaign.save_training_state(tmp_path / "raw.pt", session, {"parent": "same"})
    return path, copy.deepcopy(session.model)


def _configured(data, path, variant="control", teacher=None, extra=202):
    session = _build(data)
    campaign.load_training_state(path, session, {"parent": "same"})
    return refinement.configure_refinement(session, variant=variant, teacher=teacher,
                                           start_steps=2, total_extra_steps=extra)


def _assert_tree(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree(left, right)
    else:
        assert actual == expected


def test_control_exactly_preserves_original_loop_inputs_rng_base_loss_and_weights(prepared, tmp_path):
    path, _ = _raw(prepared, tmp_path)
    actual = _configured(prepared, path)
    expected = _build(prepared)
    campaign.load_training_state(path, expected, {"parent": "same"})
    expected._campaign_schedule_hook.remove()
    for group in expected.optimizer.param_groups:
        group["lr"] = .003 * .2
    expected_progress = expected.advance_to(7)
    actual_progress = actual.advance_to(7)
    assert {k: v for k, v in actual_progress.items() if k != "refinement"} == expected_progress
    _assert_tree(campaign._session_state(actual), campaign._session_state(expected))
    assert actual.optimizer.param_groups[0]["lr"] == .003 * .2
    assert actual_progress["refinement"]["tail_ema_updates"] == 3


def test_masked_teacher_preserves_student_inputs_and_generator_schedule(prepared, tmp_path):
    path, teacher = _raw(prepared, tmp_path)
    control = _configured(prepared, path)
    kd = _configured(prepared, path, "masked_teacher", teacher)
    traces = [[], []]
    handles = []
    for index, session in enumerate((control, kd)):
        handles.append(session.model.register_forward_pre_hook(
            lambda model, inputs, i=index: traces[i].append(tuple(x.detach().clone() for x in inputs))))
    try:
        control.advance_to(9)
        kd.advance_to(9)
    finally:
        for handle in handles:
            handle.remove()
    _assert_tree(traces[0], traces[1])
    assert control._training_window_hash.hexdigest() == kd._training_window_hash.hexdigest()
    for name in ("_rng", "_augmentation_rng", "_tracking_rng"):
        _assert_tree(getattr(control, name).bit_generator.state,
                     getattr(kd, name).bit_generator.state)
    assert kd._refinement_counters["aux_steps"] == 7
    assert not kd._refinement_teacher.training
    assert all(not p.requires_grad and p.grad is None for p in kd._refinement_teacher.parameters())
    _assert_tree(teacher.state_dict(), kd._refinement_teacher.state_dict())


@pytest.mark.parametrize("variant", ["control", "spell_balanced", "masked_teacher"])
@pytest.mark.parametrize("split_step", [3, 4, 5])
def test_exact_resume_before_at_and_after_ema_start(prepared, tmp_path, variant, split_step):
    path, teacher = _raw(prepared, tmp_path)
    teacher_arg = teacher if variant == "masked_teacher" else None
    whole = _configured(prepared, path, variant, teacher_arg)
    expected = whole.advance_to(7)
    staged = _configured(prepared, path, variant, teacher_arg)
    staged.advance_to(split_step)
    checkpoint = refinement.save_refinement_state(tmp_path / "resume.pt", staged,
                                                 {"parent_sha256": sha256_file(path)})
    resumed = refinement.configure_refinement(_build(prepared), variant=variant,
                                              teacher=teacher_arg,
                                              start_steps=2, total_extra_steps=202)
    refinement.load_refinement_state(checkpoint, resumed, {"parent_sha256": sha256_file(path)})
    assert resumed.advance_to(7) == expected
    _assert_tree(campaign._session_state(resumed), campaign._session_state(whole))
    _assert_tree(resumed._tail_ema, whole._tail_ema)
    assert resumed._tail_ema_updates == whole._tail_ema_updates == 3


def test_tail_ema_uses_post_update_8200_equivalent_state_and_fixed_decay(prepared, tmp_path):
    path, _ = _raw(prepared, tmp_path)
    session = _configured(prepared, path)
    session.advance_to(3)
    assert session._tail_ema is None
    session.advance_to(4)
    _assert_tree(session._tail_ema, session.model.state_dict())
    initial = copy.deepcopy(session._tail_ema)
    session.advance_to(5)
    for name, value in session.model.state_dict().items():
        torch.testing.assert_close(session._tail_ema[name], initial[name] * .99 + value * .01,
                                   rtol=1e-6, atol=1e-8)
    assert session._tail_ema_updates == 1


def test_teacher_view_corrects_swap_without_advancing_session_generators(prepared, tmp_path):
    path, teacher = _raw(prepared, tmp_path)
    session = _configured(prepared, path, "masked_teacher", teacher)
    frames, targets, features, _ = session.cache.get(session.train_paths[0])
    # Find an actual differing swap caused by the legacy outage RNG consumption.
    for seed in range(100):
        session._augmentation_rng = np.random.default_rng(seed)
        before = copy.deepcopy(session._augmentation_rng.bit_generator.state)
        tracking = copy.deepcopy(session._tracking_rng.bit_generator.state)
        inputs, loss_targets, choices = refinement._window(
            frames, targets, features, 20, 44, session.halo, session._augmentation_rng,
            session.augment, session.device, session.outage_config,
            session.augmentation_config, session.steps + 1, session._tracking_rng)
        raw_teacher, _, teacher_choices = refinement._window(
            frames, targets, features, 20, 44, session.halo,
            refinement._clone_generator(session._augmentation_rng, before),
            session.augment, session.device, refinement._CLEAN_OUTAGES,
            session.augmentation_config, session.steps + 1,
            refinement._clone_generator(session._tracking_rng, tracking))
        if choices["team_swap"] != teacher_choices["team_swap"]:
            break
    else:
        pytest.fail("synthetic seeds did not exercise team swap reconciliation")
    current_augmentation = copy.deepcopy(session._augmentation_rng.bit_generator.state)
    current_tracking = copy.deepcopy(session._tracking_rng.bit_generator.state)
    clean = refinement.aligned_teacher_view(session, frames, targets, features, 20, 44,
                                            before, tracking, loss_targets, choices)
    torch.testing.assert_close(clean[0], raw_teacher[0].flip(2), rtol=0, atol=0)
    torch.testing.assert_close(clean[1], inputs[1], rtol=0, atol=0)
    assert not any(value.requires_grad for value in clean)
    _assert_tree(session._augmentation_rng.bit_generator.state, current_augmentation)
    _assert_tree(session._tracking_rng.bit_generator.state, current_tracking)


def test_short_spell_sampler_contains_both_neighboring_frames(prepared, tmp_path):
    path, _ = _raw(prepared, tmp_path)
    session = _build(prepared)
    campaign.load_training_state(path, session, {"parent": "same"})
    refinement.configure_refinement(session, variant="spell_balanced", start_steps=2,
                                    total_extra_steps=4, short_sampling_probability=1.)
    frames, targets, features, transitions = session.cache.get(session.train_paths[0])
    spells = complete_spells(targets.state, targets.state_valid, features.time_valid, frames.frame_id)
    for _ in range(20):
        start = session._sample_start(frames, targets, features, transitions, session.center_size)
        assert any(s.state in (0, 1) and s.frames < 50 and start <= s.start - 1
                   and s.stop < start + session.center_size for s in spells)
    assert session._refinement_counters["short_spell_samples"] == 20


def test_refinement_never_loads_held_out_half(prepared, tmp_path, monkeypatch):
    path, teacher = _raw(prepared, tmp_path)
    session = _configured(prepared, path, "masked_teacher", teacher)
    loaded = []
    original = session.cache.get

    def checked(source):
        assert "held_out" not in str(source)
        loaded.append(str(source))
        return original(source)

    monkeypatch.setattr(session.cache, "get", checked)
    session.advance_to(7)
    assert loaded
    assert session.train_matches == ["train_a", "train_b"]


def test_resume_rejects_identity_teacher_and_hash_changes_without_mutation(prepared, tmp_path):
    path, teacher = _raw(prepared, tmp_path)
    session = _configured(prepared, path, "masked_teacher", teacher)
    session.advance_to(5)
    checkpoint = refinement.save_refinement_state(tmp_path / "resume.pt", session, {"fold": 1})
    receiver = _configured(prepared, path, "masked_teacher", teacher)
    before = campaign._session_state(receiver)
    with pytest.raises(ValueError, match="identity/config/data"):
        refinement.load_refinement_state(checkpoint, receiver, {"fold": 2})
    _assert_tree(campaign._session_state(receiver), before)
    with torch.no_grad():
        next(receiver._refinement_teacher.parameters()).add_(.01)
    with pytest.raises(ValueError, match="teacher weights"):
        refinement.load_refinement_state(checkpoint, receiver, {"fold": 1})
    _assert_tree(campaign._session_state(receiver), before)
    checkpoint.write_bytes(checkpoint.read_bytes()[:-5] + b"abcde")
    with pytest.raises(ValueError, match="SHA256"):
        refinement.load_refinement_state(checkpoint, receiver, {"fold": 1})


def test_export_is_prediction_compatible_binds_identity_and_restores_raw_weights(prepared, tmp_path):
    path, _ = _raw(prepared, tmp_path)
    session = _configured(prepared, path, extra=4)
    session.advance_to(6)
    raw = copy.deepcopy(session.model.state_dict())
    identity = {"parent_sha256": sha256_file(path), "fold": 1}
    output = refinement.export_refinement_checkpoint(tmp_path / "ema.pt", session, identity)
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert {"state_dict", "model_kwargs", "training_config", "prepared_sha256"} <= set(checkpoint)
    _assert_tree(checkpoint["state_dict"], session._tail_ema)
    _assert_tree(session.model.state_dict(), raw)
    assert checkpoint["training_config"]["refinement_identity"] == identity
    assert checkpoint["training_config"]["refinement"]["variant"] == "control"
    assert checkpoint["training_config"]["refinement_export"] == "tail_ema"
    assert all("held_out" not in name for name in checkpoint["prepared_sha256"])
    sidecar = output.with_suffix(output.suffix + ".sha256.json")
    assert json.loads(sidecar.read_text())["sha256"] == sha256_file(output)


def test_fresh_configured_session_cannot_train_without_restore(prepared):
    session = refinement.configure_refinement(_build(prepared), variant="control",
                                              start_steps=2, total_extra_steps=4)
    with pytest.raises(ValueError, match="restore the authenticated"):
        session.advance_to(3)


def _poison_state(state, location):
    if location == "model":
        next(iter(state["model"].values())).view(-1)[0] = float("nan")
    elif location == "optimizer":
        next(iter(state["optimizer"]["state"].values()))["exp_avg"].view(-1)[0] = float("inf")
    elif location == "base_counter":
        state["counters"]["_train_sums"]["live"] = float("inf")
    elif location == "rng":
        state["rng"]["numpy_legacy"]["cached_gaussian"] = float("nan")


@pytest.mark.parametrize("location", ["model", "optimizer", "base_counter", "rng"])
def test_restore_rejects_nonfinite_nested_state_before_mutating_receiver(prepared, tmp_path, location):
    path, _ = _raw(prepared, tmp_path)
    source = _configured(prepared, path)
    source.advance_to(5)
    checkpoint = refinement.save_refinement_state(tmp_path / "resume.pt", source, {"fold": 1})
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    _poison_state(payload["campaign"]["state"], location)
    torch.save(payload, checkpoint)
    checkpoint.with_suffix(checkpoint.suffix + ".sha256.json").write_text(
        json.dumps({"sha256": sha256_file(checkpoint)}))
    receiver = _configured(prepared, path)
    before = campaign._session_state(receiver)
    with pytest.raises(ValueError, match="nonfinite numeric state"):
        refinement.load_refinement_state(checkpoint, receiver, {"fold": 1})
    _assert_tree(campaign._session_state(receiver), before)


@pytest.mark.parametrize("location", ["model", "optimizer", "base_counter", "rng", "ema"])
def test_save_and_export_reject_nonfinite_snapshot_before_writing(prepared, tmp_path, location):
    path, _ = _raw(prepared, tmp_path)
    session = _configured(prepared, path, extra=4)
    session.advance_to(6)
    if location == "model":
        with torch.no_grad():
            next(session.model.parameters()).view(-1)[0] = float("nan")
    elif location == "optimizer":
        next(iter(session.optimizer.state.values()))["exp_avg"].view(-1)[0] = float("inf")
    elif location == "base_counter":
        session._train_sums["live"] = float("inf")
    elif location == "rng":
        session._torch_cpu_state = torch.tensor([float("nan")])
    else:
        next(iter(session._tail_ema.values())).view(-1)[0] = float("nan")
    for filename, writer in (("resume.pt", refinement.save_refinement_state),
                             ("prediction.pt", refinement.export_refinement_checkpoint)):
        destination = tmp_path / "outputs" / filename
        with pytest.raises(ValueError, match="nonfinite numeric state"):
            writer(destination, session, {"fold": 1})
        assert not destination.exists()
        assert not destination.with_suffix(destination.suffix + ".sha256.json").exists()
    assert not (tmp_path / "outputs").exists()


def test_finite_tree_accepts_integer_rng_tensors_bytes_and_large_integer_states():
    refinement._require_finite_tree({
        "rng": torch.arange(16, dtype=torch.uint8), "state": 2 ** 128,
        "transcript": b"synthetic:0:24\n", "numpy": np.array([1, 2], dtype=np.uint64),
        "nested": [1., (0j, None, True)],
    })
    with pytest.raises(ValueError, match="nonfinite numeric state"):
        refinement._require_finite_tree({"nested": [np.array([1., np.nan])]})
