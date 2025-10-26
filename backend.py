from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class MarkerRequest(BaseModel):
    coordinates: list

@app.post('/api/activities/nearby')
async def get_nearby_activities(request: MarkerRequest):

    

    return {
        'activities': [

        ]
    }


