from backend import *
import numpy as np
import google.generativeai as genai

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

    centroid, members, labels, radius = run_dbscan(points_lonlat, eps, min_samples)

    print("\nmin_samples:", min_samples)
    print("k-distances:", k_distances)
    print("eps (radians):", eps)
    print("labels:", labels)
    print("members:", members)
    print("centroid (lat, lon):", centroid)
    print("radius:", radius)

genai.configure(api_key="APIKEY")  # or from env var

model = genai.GenerativeModel("gemini-2.5-flash")  # any Gemini model you use

# Your clustering outputs
centroid = [35.6595, 139.7008]  # [lat, lon]
members = [0,1,2]
labels  = [0,0,0,1,1,1,-1]
radius_m = 420.3

prompt = f"""
You are a Tokyo trip assistant.
Cluster center (lat, lon): {centroid}
Cluster radius (m): {radius_m:.1f}
Cluster members (indices): {members}
All labels: {labels}

Task: Suggest 8 activities within ~{int(radius_m)} m of the center, grouped by food / sights / cafes.
Return concise bullet points with names and why they fit.
"""

resp = model.generate_content(prompt)
print(resp.text)
 
