import math
import numpy as np
from sklearn.cluster import DBSCAN
EARTH_R = 6_371_000 #meters
n_points= coordinates.shape[0]


#determine min samples
def choose_min_samples(n_points: int) -> int:
    if n_points <= 8:
        return 3
    elif n_points <= 20:
        return 4
    elif n_points <=50:
        return 5
    else: 
        return 6
    
#compute distance to each points k-th nearest neighbor (k = min_samples)
def haversine(lat1, lon1, lat2, lon2):
    R = EARTH_R
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*(math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def kth_nearest_distance(points, k):
    distances = []

    for i in range(n_points):
        lat1, lon1 = points[i]
        current_distances = []

        for j in range(n_points):
            if i == j:
                continue
            lat2, lon2 = points[j]
            d = haversine(lat1, lon1, lat2, lon2)
            current_distances.append(d)

        current_distances.sort()
        distances.append(current_distances[k-1])

    return distances


#Sort all k-distance values
k_distances = kth_nearest_distances(points,k)
k_distances.sort()

#Identify elbow point in sorted curve → estimate eps_m
def estimate_eps_m_from_elbow(k_distances_sorted):
    n = len(k_distances_sorted)
    x0, y0 = 0, k_distances_sorted[0]
    x1, y1 = n-1, k_distances_sorted[n-1]
    dx, dy = x1 - x0, y1 - y0
    denom = (dx**2 + dy**2) ** 0.5
   
    max_dist = -1
    max_idx = 0
    for i, y in enumerate(k_distances_sorted):
        num = abs(dy * i - dx * y + x1*y0 - y1*x0)
        d = num / denom
        if d > max_dist:
            max_dist = d
            max_idx = i

    return k_distances_sorted[max_idx] / 6371000
    
#Run DBSCAN
def run_dbscan(points_deg, eps, min_samples):
    """
    points_deg: list of (lat, lon) in degrees
    eps: radians
    min_samples: int
    returns: (centroid_deg [lat, lon] or None, member_indices, labels_list)
    """
    pts = np.array(points_deg, dtype=float)
    if len(pts) < min_samples:
        return None, [], []

    pts_rad = np.radians(pts)  # (lat, lon) → radians

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(pts_rad)

    valid = labels != -1
    if valid.sum() == 0:
        return None, [], labels.tolist()

    vals, counts = np.unique(labels[valid], return_counts=True)
    best_label = vals[np.argmax(counts)]
    members = np.where(labels == best_label)[0]

    centroid_rad = pts_rad[members].mean(axis=0)
    centroid_deg = np.degrees(centroid_rad).tolist()

    return centroid_deg, members.tolist(), labels.tolist()
