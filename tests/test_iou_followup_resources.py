"""Changing cache capacity must not change a CUDA training trajectory."""
import pytest
import torch
from test_gpu_runtime import _data_spec

from experiments import campaign_training as campaign
from experiments import gpu_runtime as gpu
from experiments.ema_training import _tree_equal
from possession.training_session import HalfCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cache_capacity_preserves_exact_cuda_trajectory_and_rng(tmp_path):
    device = gpu.configure_device()
    data, spec = _data_spec(tmp_path)
    with gpu.accelerated_preprocessing():
        cold_cache, hot_cache = HalfCache(data, 1), HalfCache(data, 10_000_000)
        cold = gpu.build_session(data, spec, ["train"], 91, cold_cache, device=device)
        hot = gpu.build_session(data, spec, ["train"], 91, hot_cache, device=device)
        cold.advance_to(12)
        hot.advance_to(12)
        assert cold_cache.misses > hot_cache.misses
        assert hot_cache.hits > 0
        assert _tree_equal(campaign._session_state(cold), campaign._session_state(hot))
