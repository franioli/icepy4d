import pytest
import numpy as np

from icepy.classes.features import Feature, Features


def test_feature():
    rng = np.random.default_rng()
    x = rng.random(dtype=np.float32)
    y = rng.random(dtype=np.float32)
    descr = rng.random(256, dtype=np.float32)
    score = rng.random(dtype=np.float32)
    f = Feature(x, y, descr=descr, score=score)
    assert f.x == np.float32(x), "Unable to create correct Feature object"
    assert f.y == np.float32(y), "Unable to create correct Feature object"
    assert np.any(
        f.descr == descr.reshape(256, 1).astype(np.float32)
    ), "Unable to create correct Feature object"
    assert f.score == score, "Unable to create correct Feature object"

    descr = rng.random((256, 1), dtype=np.float32)
    f = Feature(x, y, descr=descr, score=score)
    assert np.any(
        f.descr == descr.reshape(256, 1).astype(np.float32)
    ), "Unable to create correct Feature object"

    descr = rng.random((1, 256), dtype=np.float32)
    f = Feature(x, y, descr=descr, score=score)
    assert np.any(
        f.descr == descr.reshape(256, 1).astype(np.float32)
    ), "Unable to create correct Feature object"

    # Test different input types


def test_features():
    rng = np.random.default_rng()
    n_feat = 100
    width, height = 6000, 4000
    x = rng.integers(0, width, (n_feat, 1))
    y = rng.integers(0, height, (n_feat, 1))
    descr = rng.random((256, n_feat), dtype=np.float32)
    scores = rng.random((n_feat, 1), dtype=np.float32)
    features = Features()
    features.append_features_from_numpy(x, y, descr, scores)
    assert any(
        [features[i].track_id == i for i in range(len(features))]
    ), "Unable to create correct track id when appending new features from numpy"
    out = features.to_numpy(get_descr=True)
    assert features[0].x == x[0], "Unable to create correct Features object"
    assert np.any(
        out["kpts"]
        == np.concatenate((x.reshape(n_feat, 1), y.reshape(n_feat, 1)), axis=1)
    ), "Unable to create correct Features object"
    assert np.any(
        out["descr"] == descr.reshape(256, n_feat).astype(np.float32)
    ), "Unable to create correct Features object"


if __name__ == "__main__":
    test_feature()
