import numpy as np
import pytest

torch = pytest.importorskip("torch")

from possession.model import PossessionModel
from possession.predict import predict_features
from possession.schema import FeatureBatch


@pytest.mark.parametrize("pooling", ["mean", "ball_attention"])
def test_cached_feature_prediction_independent_of_chunk_size(pooling):
    rng = np.random.default_rng(5)
    features = FeatureBatch(
        player_tokens=rng.normal(size=(43, 2, 4, 8)).astype(np.float32),
        player_mask=np.ones((43, 2, 4), dtype=bool),
        global_tokens=rng.normal(size=(43, 9)).astype(np.float32),
        time_valid=np.ones(43, dtype=bool),
    )
    before = features.player_tokens.copy()
    model = PossessionModel(width=8, dilations=(1, 2), pooling=pooling)
    small = predict_features(features, model, center_size=7)
    large = predict_features(features, model, center_size=100)
    for name in small:
        np.testing.assert_allclose(small[name], large[name], atol=1e-6)
    np.testing.assert_array_equal(features.player_tokens, before)
    with pytest.raises(ValueError, match="halo"):
        predict_features(features, model, halo=2)
