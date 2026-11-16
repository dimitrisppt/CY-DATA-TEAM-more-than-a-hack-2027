"""Memory-bounded native-rate model prediction for one tracking half."""

from pathlib import Path

import numpy as np
import torch

from possession.features import build_features
from possession.model import PossessionModel
from possession.schema import FeatureBatch, FrameTable


def _slice_tensor(value, left, right, device):
    return torch.from_numpy(value[left:right]).unsqueeze(0).to(device)


def _instantiate_checkpoint(payload, device="cpu"):
    if not isinstance(payload, dict) or not all(
        key in payload for key in ("state_dict", "model_kwargs", "training_config")
    ):
        raise ValueError("checkpoint requires state_dict, model_kwargs, and training_config")
    if not isinstance(payload["model_kwargs"], dict):
        raise TypeError("model_kwargs must be a mapping")
    model = PossessionModel(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def load_model_checkpoint(path, device="cpu"):
    """Load a locally produced state-dict checkpoint for target inference."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    return _instantiate_checkpoint(payload, device)


def predict_model(
    frames: FrameTable,
    model: PossessionModel,
    device: str = "cpu",
    center_size: int = 512,
    halo: int = 256,
) -> dict[str, np.ndarray]:
    """Predict one half while moving only the active halo chunk to the device.

    Features remain on CPU. Every kept center has the same real-frame context
    as a full-half forward pass; only actual half edges use convolution padding.
    """
    features = build_features(frames)
    return predict_features(features, model, device=device, center_size=center_size, halo=halo)


def predict_features(
    features: FeatureBatch,
    model: PossessionModel,
    device: str = "cpu",
    center_size: int = 2048,
    halo: int | None = None,
) -> dict[str, np.ndarray]:
    """Predict cached, label-free features using the same bounded chunk path.

    HPO can reuse features across trials without re-reading tracking or labels.
    Chunk length affects memory and throughput, never the model context radius.
    """
    halo = model.radius if halo is None else halo
    if center_size < 1 or halo < model.radius:
        raise ValueError(f"center_size must be positive and halo >= {model.radius}")
    n = len(features.time_valid)
    if not n:
        raise ValueError("cannot predict an empty half")
    model = model.to(device)
    model.eval()
    output = {"live_logits": [], "owner_logits": [], "boundary": []}
    with torch.inference_mode():
        for start in range(0, n, center_size):
            stop = min(start + center_size, n)
            left = max(0, start - halo)
            right = min(n, stop + halo)

            scores = model(
                _slice_tensor(features.player_tokens, left, right, device),
                _slice_tensor(features.player_mask, left, right, device),
                _slice_tensor(features.global_tokens, left, right, device),
                _slice_tensor(features.time_valid, left, right, device),
            )
            for name, bucket in output.items():
                kept = scores[name][0, start - left:stop - left]
                bucket.append(kept.cpu().numpy())
    live_logits = np.concatenate(output["live_logits"])
    owner_logits = np.concatenate(output["owner_logits"])
    boundary = np.concatenate(output["boundary"])

    def sigmoid(x):
        x = np.asarray(x, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))

    return {
        "live_probability": sigmoid(live_logits),
        "home_probability": sigmoid(owner_logits),
        "boundary": np.asarray(boundary, dtype=np.float64),
    }
