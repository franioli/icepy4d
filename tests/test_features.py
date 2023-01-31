import pytest
import os
import numpy as np

from pathlib import Path
from easydict import EasyDict as edict

from src.icepy.base_classes.features import Feature, Features_new


def test_feature():
    x = 2000.1
    y = 1500.2
    descr = np.arange(0, 256, dtype=float)
    score = 0.5
    f = Feature(x, y, descr=descr, score=score)
    assert f.x == x, "Unable to create correct Feature object"
    assert f.y == y, "Unable to create correct Feature object"
    assert np.any(
        f.descr == descr.reshape(256, 1)
    ), "Unable to create correct Feature object"
    assert f.score == score, "Unable to create correct Feature object"


def test_features():
    n_feat = 11
    x = np.linspace(0, 10, n_feat)  #
    y = np.linspace(0, 10, n_feat)  # .reshape(n_feat, 1)
    descr = np.random.rand(256, n_feat)
    scores = np.random.rand(n_feat, 1)
    features = Features_new()
    features.append_features_from_numpy(x, y, descr, scores)
    k, d = features.to_numpy(get_descr=True)
    assert features[0].x == x[0], "Unable to create correct Features object"
    assert np.any(
        k == np.concatenate((x.reshape(n_feat, 1), y.reshape(n_feat, 1)), axis=1)
    ), "Unable to create correct Features object"
    assert np.any(
        d == descr.reshape(256, n_feat)
    ), "Unable to create correct Features object"
