from hellooo.py import
import numpy as np
from cluster import (
    choose_min_samples,
    kth_nearest_distance,
    estimate_eps_from_elbow,
    run_dbscan
)

if __name__ == "__main__":
    # Example input (replace with real data)
    points_lonlat = np.array([
        [139.7005, 35.6595],
        [139.7007, 35.6600],
        [139.7012, 35.6590],
        [139.7967, 35.7148],
        [139.7969, 35.7145],
        [139.7971, 35.7149],
        [139.8100, 35.7300]
    ])

    print("\nPoints (lon, lat):")
    print(points_lonlat)

    n = len(points_lonlat)
    min_samples = choose_min_samples(n)

    k_distances = kth_nearest_distance(points_lonlat, min_samples)
    k_distances.sort()

    eps = estimate_eps_from_elbow(k_distances)

    centroid, members, labels = run_dbscan(points_lonlat, eps, min_samples)

    print("\nmin_samples:", min_samples)
    print("k-distances:", k_distances)
    print("eps (radians):", eps)
    print("labels:", labels)
    print("members:", members)
    print("centroid (lat, lon):", centroid)
