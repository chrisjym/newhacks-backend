from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os, json, numpy as np
from google import genai

from backend import (
    choose_min_samples, kth_nearest_distance,
    estimate_eps_from_elbow, run_dbscan
)

app = FastAPI()


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# config
client = genai.Client(api_key="AIzaSyD812vh66sjZF2TjI-OQeZKLNmu98oLHbc")



class PointsReq(BaseModel):
    points: List[List[float]]  # [[lon, lat], ...]

@app.post("/api/coordinates")
async def cluster_points(req: PointsReq):
    print("Received points:", req.points)
    pts = np.array(req.points, dtype=float)
    n = len(pts)
    if n < 2:
        raise HTTPException(400, "Need at least 2 points")

    # clustering
    min_samples = choose_min_samples(n)
    kdist = kth_nearest_distance(pts, min_samples); kdist.sort()
    eps = estimate_eps_from_elbow(kdist)
    centroid, members, labels, radius_m = run_dbscan(pts, eps, min_samples)
    if centroid is None:
        return {"status": "error", "message": "No cluster found"}

    # build prompt
    prompt = f"""
Respond with raw JSON only. No code fences, no labels, no explanation.

Cluster center (lat, lon): {centroid}
Cluster radius (m): {radius_m:.1f}
Cluster members (indices): {members}
All labels: {labels}

Task: Suggest at least 8 touristic activities within ~{int(radius_m)} m of the center
(If fewer exist, return as many as possible; if none, set status to "error").
For each activity include: name, longitude, latitude, type, description.
Also include an np.array-like example:
poi.lonlat = np.array([[lon1,lat1],[lon2,lat2],...])

Return a single JSON object.

Example format (treat as literal text, not data to reuse):
{{
  "status": "success",
  "centroid": {{"longitude": 139.7008, "latitude": 35.6595}},
  "radius_km": {radius_m/1000:.3f},
  "recommended_places": [
    {{"name": "Example", "longitude": 139.70, "latitude": 35.66, "type": "poi", "description": "..." }}
  ]
}}
"""
    
    # user_profile = {
    # "status": "success",
    # "centroid": {{"longitude": 139.7008, "latitude": 35.6595}},
    # "radius_km": "{radius_m/1000:.3f}",
    # "recommended_places": [
    #     {{"name": "Example", "longitude": 139.70, "latitude": 35.66, "type": "poi", "description": "..." }}
    # ]
    # }

    # call Gemini
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents = prompt,
        config={
            'response_mime_type': 'application/json',
            'response_json_schema': {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "centroid": {
                        "type": "object",
                        "properties": {
                            "longitude": {"type": "number"},
                            "latitude": {"type": "number"}
                        },
                        "required": ["longitude", "latitude"]
                    },
                    "radius_km": {"type": "number"},
                    "recommended_places": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "longitude": {"type": "number"},
                                "latitude": {"type": "number"},
                                "type": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "longitude", "latitude", "type", "description"]
                        }
                    }
                },
                "required": ["status", "centroid", "radius_km", "recommended_places"]
            }
        }

    )
    # parse JSON
    # try:
    #     data = json.loads(text)
    # except json.JSONDecodeError:
    #     raise HTTPException(502, "Gemini returned non-JSON")

    # optional: normalize centroid ordering if needed
    return response.parsed
