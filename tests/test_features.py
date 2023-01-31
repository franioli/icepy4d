import pytest
import numpy as np

from src.icepy.base_classes.features import Feature, Features


def test_feature():
    x = np.random.rand()
    y = np.random.rand()
    descr = np.random.rand(256)
    score = np.random.rand()
    f = Feature(x, y, descr=descr, score=score)
    assert f.x == x, "Unable to create correct Feature object"
    assert f.y == y, "Unable to create correct Feature object"
    assert np.any(
        f.descr == descr.reshape(256, 1)
    ), "Unable to create correct Feature object"
    assert f.score == score, "Unable to create correct Feature object"

    descr = np.random.rand(256, 1)
    f = Feature(x, y, descr=descr, score=score)
    assert np.any(
        f.descr == descr.reshape(256, 1)
    ), "Unable to create correct Feature object"

    descr = np.random.rand(1, 256)
    f = Feature(x, y, descr=descr, score=score)
    assert np.any(
        f.descr == descr.reshape(256, 1)
    ), "Unable to create correct Feature object"


def test_features():
    n_feat = 100
    width, height = 6000, 4000
    x = np.random.randint(0, width, (n_feat, 1))
    y = np.random.randint(0, height, (n_feat, 1))
    descr = np.random.rand(256, n_feat)
    scores = np.random.rand(n_feat, 1)
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
        out["descr"] == descr.reshape(256, n_feat)
    ), "Unable to create correct Features object"
