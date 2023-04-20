# import numpy as np
# import pytest
# from icepy4d.matching.match_by_preselection import find_centroids_kmeans

# def test_find_centroids_kmeans():
#     data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
#     n_cluster = 2

#     centroids, classes = find_centroids_kmeans(data, n_cluster, viz_clusters=False)

#     assert centroids.shape == (n_cluster, 2)
#     assert classes.shape == (data.shape[0],)
#     assert np.all(classes == np.array([0, 0, 1, 1, 1]))

#     centroids, classes = find_centroids_kmeans(data, n_cluster, viz_clusters=True)
#     # You could add more specific visual tests if you wanted
#     assert centroids.shape == (n_cluster, 2)
#     assert classes.shape == (data.shape[0],)
#     assert np.all(classes == np.array([0, 0, 1, 1, 1]))
