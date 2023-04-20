import pytest
import numpy as np

from icepy4d.classes.features import Feature, Features


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


def test_feature_creation():
    x = np.float32(1)
    y = np.float32(2)
    track_id = np.int32(3)
    descr = np.ones((128, 1), dtype=np.float32)
    score = np.float32(0.5)
    epoch = np.int32(1)

    feature = Feature(x, y, track_id=track_id, descr=descr, score=score, epoch=epoch)

    assert feature.x == x
    assert feature.y == y
    assert np.array_equal(feature.xy, np.array([[x, y]], dtype=np.float32))
    assert feature.track_id == track_id
    assert np.array_equal(feature.descr, descr)
    assert feature.score == score
    assert feature.epoch == epoch


def test_feature_creation_no_optional_args():
    x = np.float32(1)
    y = np.float32(2)

    feature = Feature(x, y)

    assert feature.x == x
    assert feature.y == y
    assert np.array_equal(feature.xy, np.array([[x, y]], dtype=np.float32))
    assert feature.track_id is None
    assert feature.descr is None
    assert feature.score is None
    assert feature.epoch is None


def test_feature_creation_invalid_descr_shape():
    x = np.float32(1)
    y = np.float32(2)
    descr = np.ones((10, 10), dtype=np.float32)

    with pytest.raises(AssertionError):
        Feature(x, y, descr=descr)


def test_features():
    rng = np.random.default_rng()
    n_feat = 100
    width, height = 6000, 4000
    x = rng.integers(0, width, (n_feat, 1))
    y = rng.integers(0, height, (n_feat, 1))
    descr = rng.random((256, n_feat), dtype=np.float32)

    # Test scores shape
    scores_1d = rng.random((n_feat), dtype=np.float32)
    features = Features()
    features.append_features_from_numpy(x, y, descr, scores_1d)
    assert (
        len(features) == n_feat
    ), "Unable to append features with scores of shape (n_feat,)"
    scores_2d = rng.random((n_feat, 1), dtype=np.float32)
    features = Features()
    features.append_features_from_numpy(x, y, descr, scores_2d)
    assert (
        len(features) == n_feat
    ), "Unable to append features with scores of shape (n_feat,1)"
    assert any(
        [features[i].track_id == i for i in range(len(features))]
    ), "Unable to create correct track id when appending new features from numpy"
    out = features.to_numpy(get_descr=True, get_score=True)
    assert features[0].x == x[0].squeeze().astype(
        np.float32
    ), "Unable to create correct Features object"
    assert np.any(
        out["kpts"][:, 0] == x.astype(np.float32).squeeze()
    ), "Unable to create correct Features object"
    assert np.any(
        out["kpts"]
        == np.concatenate((x.reshape(n_feat, 1), y.reshape(n_feat, 1)), axis=1)
    ), "Unable to create correct Features object"
    assert np.any(
        out["descr"] == descr.reshape(256, n_feat).astype(np.float32)
    ), "Unable to create correct Features object"


def test_filter_feature_by_mask():
    n_feat = 5
    rng = np.random.default_rng()
    width, height = 6000, 4000
    x = rng.integers(0, width, (n_feat, 1))
    y = rng.integers(0, height, (n_feat, 1))
    descr = rng.random((256, n_feat), dtype=np.float32)
    scores = rng.random((n_feat, 1), dtype=np.float32)
    features = Features()
    features.append_features_from_numpy(x, y, descr, scores)
    inlier_mask = [
        True,
        False,
        True,
        False,
        True,
    ]  # Only keep 1st, 3rd, and 5th features
    features.filter_feature_by_mask(inlier_mask)
    assert (
        len(features) == 3
    ), "Unable to filter a Features object by mask"  # Check that we only have 3 features left

    features = Features()
    features.append_features_from_numpy(x, y, descr, scores)
    indexes = [0, 2, 4]
    features.filter_feature_by_index(indexes)
    assert len(features) == 3, "Unable to filter a Features object by index"


def test_append_features_from_numpy():
    features = Features()
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    descr = np.ones((128, 5))
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    track_ids = [0, 1, 2, 3, 4]
    epoch = 1
    features.append_features_from_numpy(x, y, descr, scores, track_ids, epoch)
    assert len(features) == 5
    assert features._descriptor_size == 128
    assert np.allclose(features[0].x, 1.0)
    assert np.allclose(features[0].y, 1.0)
    assert features[0].descr.shape == (128, 1)
    assert np.allclose(features[0].score, 0.1)
    assert features[0].epoch == 1

    x_1d = np.array([1, 2, 3, 4, 5])
    y_1d = np.array([1, 2, 3, 4, 5])
    descr_1d = np.ones((128, 5))
    scores_1d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    track_ids_1d = [5, 6, 7, 8, 9]
    epoch_1d = 2
    features.append_features_from_numpy(
        x_1d, y_1d, descr_1d, scores_1d, track_ids_1d, epoch_1d
    )
    assert len(features) == 10
    assert features._descriptor_size == 128
    assert np.allclose(features[5].x, 1.0)
    assert np.allclose(features[5].y, 1.0)
    assert features[5].descr.shape == (128, 1)
    assert np.allclose(features[5].score, 0.1)
    assert features[5].epoch == 2


def test_to_numpy():

    # Create Feature object
    features = Features()
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    descr = np.ones((128, 5))
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    track_ids = [0, 1, 2, 3, 4]
    epoch = 1
    features.append_features_from_numpy(x, y, descr, scores, track_ids, epoch)

    # Test case 1: get only keypoints
    res = features.to_numpy()
    assert len(res.keys()) == 1  # Check if only one key is present
    assert np.allclose(
        res["kpts"],
        np.array(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],
            dtype=np.float32,
        ),
    )  # Check if keypoints are correct

    # Test output when get_descr and get_score are both True
    output = features.to_numpy(get_descr=True, get_score=True)
    expected_output = {
        "kpts": np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32),
        "descr": np.ones((128, 5), dtype=np.float32),
        "scores": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    }
    assert np.allclose(output["kpts"], expected_output["kpts"])
    assert np.allclose(output["descr"], expected_output["descr"])
    assert np.allclose(output["scores"], expected_output["scores"])

    # Test output when only get_descr is True
    output = features.to_numpy(get_descr=True)
    expected_output = {
        "kpts": np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32),
        "descr": np.ones((128, 5), dtype=np.float32),
    }
    assert np.allclose(output["kpts"], expected_output["kpts"])
    assert np.allclose(output["descr"], expected_output["descr"])

    # Test output when neither get_descr nor get_score are True
    output = features.to_numpy()
    expected_output = {
        "kpts": np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=np.float32)
    }
    assert np.allclose(output["kpts"], expected_output["kpts"])


if __name__ == "__main__":
    test_feature_creation()
    test_features()
    test_filter_feature_by_mask()
    test_append_features_from_numpy()
    test_to_numpy()
