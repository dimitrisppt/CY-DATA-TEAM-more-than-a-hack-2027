"""Factorial isolation and CUDA restart must preserve their paired trajectories."""
import pytest
import torch
from test_gpu_runtime import _data_spec

from experiments import campaign_training as campaign
from experiments import gpu_runtime as gpu
from experiments.ema_training import _tree_equal
from experiments.iou_followup_training import configure_factorial
from experiments.refinement_training import (
    configure_refinement,
    load_refinement_state,
    save_refinement_state,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode,paired", [("loss_only", "control"), ("sampling_only", "spell_balanced")])
def test_factorial_matches_paired_inputs_rng_and_resumes(tmp_path, mode, paired):
    device = gpu.configure_device()
    data, spec = _data_spec(tmp_path)
    with gpu.accelerated_preprocessing():
        parent = gpu.build_session(data, spec, ["train"], 13, device=device)
        parent.advance_to(2)
        campaign.save_training_state(tmp_path / "parent.pt", parent, {"parent": 1})

        def receiver(kind, weight=.25):
            session = gpu.build_session(data, spec, ["train"], 13, device=device)
            campaign.load_training_state(tmp_path / "parent.pt", session, {"parent": 1})
            settings = {"start_steps": 2, "total_extra_steps": 12, "spell_weight": weight}
            if kind in ("loss_only", "sampling_only"):
                return configure_factorial(session, kind, **settings)
            return configure_refinement(session, variant=kind, **settings)

        whole, baseline, staged = receiver(mode), receiver(paired), receiver(mode)
        traces = [[], []]
        hooks = [session.model.register_forward_pre_hook(
            lambda model, inputs, i=i: traces[i].append(tuple(x.detach().clone() for x in inputs)))
            for i, session in enumerate((whole, baseline))]
        whole.advance_to(14)
        baseline.advance_to(14)
        for hook in hooks:
            hook.remove()
        assert _tree_equal(traces[0], traces[1])
        assert whole._training_window_hash.transcript == baseline._training_window_hash.transcript
        assert whole._rng.bit_generator.state == baseline._rng.bit_generator.state
        staged.advance_to(8)
        save_refinement_state(tmp_path / "child.pt", staged, {"mode": mode})
        resumed = receiver(mode)
        load_refinement_state(tmp_path / "child.pt", resumed, {"mode": mode})
        resumed.advance_to(14)
        assert _tree_equal(campaign._session_state(whole), campaign._session_state(resumed))
        assert _tree_equal(whole._tail_ema, resumed._tail_ema)
        assert whole._progress() == resumed._progress()
        assert whole._refinement_counters["updates"] == 12
        if mode == "loss_only":
            zero = receiver(mode, weight=0.)
            zero.advance_to(14)
            assert _tree_equal(zero.model.state_dict(), baseline.model.state_dict())
            assert _tree_equal(zero.optimizer.state_dict(), baseline.optimizer.state_dict())
            assert _tree_equal(zero._tail_ema, baseline._tail_ema)


def test_factorial_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown factorial"):
        configure_factorial(None, "typo")
