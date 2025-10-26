import math
import numpy as np
from sklearn.cluster import DBSCAN

EARTH_R = 6_371_000  # meters


def choose_min_samples(n_points: int) -> int:
    if n_points <= 8:
        return 3
    elif n_points <= 20:
        return 4
    elif n_points <= 50:
        return 5
    else:
        return 6


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two points (degrees)."""
    lat1 = math.radians(lat1); lon1 = math.radians(lon1)
    lat2 = math.radians(lat2); lon2 = math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_R * c


def kth_nearest_distance(points_lonlat, k):
    """
    points_lonlat: (n,2) array-like with columns [lon, lat]
    Returns list of k-th NN distances (meters) per point.
    """
    pts = np.asarray(points_lonlat, dtype=float)
    n_points = pts.shape[0]
    if n_points == 0 or k < 1:
        return []
    if n_points < 2:
        return [0.0] * n_points
    if k >= n_points:
        k = n_points - 1

    # swap to [lat, lon] for distance math
    latlon = pts[:, [1, 0]]

    distances = []
    for i in range(n_points):
        lat1, lon1 = latlon[i]
        dlist = []
        for j in range(n_points):
            if i == j:
                continue
            lat2, lon2 = latlon[j]
            dlist.append(haversine(lat1, lon1, lat2, lon2))
        dlist.sort()
        distances.append(dlist[k - 1])
    return distances


def estimate_eps_from_elbow(k_distances_sorted):
    """
    Input: sorted-ascending list of k-NN distances (meters).
    Output: eps in radians (meters / EARTH_R).
    """
    n = len(k_distances_sorted)
    if n == 0:
        return None
    if n < 3:
        return k_distances_sorted[-1] / EARTH_R

    x0, y0 = 0, k_distances_sorted[0]
    x1, y1 = n - 1, k_distances_sorted[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = (dx*dx + dy*dy) ** 0.5
    if denom == 0:
        return k_distances_sorted[-1] / EARTH_R

    max_dist, max_idx = -1.0, 0
    for i, y in enumerate(k_distances_sorted):
        num = abs(dy*i - dx*y + x1*y0 - y1*x0)
        d = num / denom
        if d > max_dist:
            max_dist, max_idx = d, i
    return k_distances_sorted[max_idx] / EARTH_R  # radians


def run_dbscan(points_lonlat, eps, min_samples):
    """
    points_lonlat: (n,2) array-like with [lon, lat]
    eps: radians (float) or None
    returns: (centroid_deg [lat, lon] or None, member_indices, labels_list)
    """
    if eps is None:
        return None, [], []

    pts = np.asarray(points_lonlat, dtype=float)
    if pts.shape[0] < min_samples:
        return None, [], []

    # swap to [lat, lon] then radians for haversine metric
    pts_latlon = pts[:, [1, 0]]
    pts_rad = np.radians(pts_latlon)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(pts_rad)

    valid = labels != -1
    if valid.sum() == 0:
        return None, [], labels.tolist()

    vals, counts = np.unique(labels[valid], return_counts=True)
    best_label = vals[np.argmax(counts)]
    members = np.where(labels == best_label)[0]

    centroid_rad = pts_rad[members].mean(axis=0)          # [lat, lon] in rad
    centroid_deg = np.degrees(centroid_rad).tolist()      # [lat, lon] in deg
    return centroid_deg, members.tolist(), labels.tolist(), 2 * eps * EARTH_R
