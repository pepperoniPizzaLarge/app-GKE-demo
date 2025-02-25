# input: params/data payload 
# output: score, class, latency

from pydantic import BaseModel
from pydantic import HttpUrl


class NLPDataInput(BaseModel):
    text: list[str]
    user_id: str



class ImageDataInput(BaseModel):
    url: list[HttpUrl]
    user_id: str
    
    
class NLPDataOutput(BaseModel):
    model_name: str
    text: list[str]
    labels: list[str]
    scores: list[float]
    prediction_time: float


class ImageDataOutput(BaseModel):
    model_name: str
    url: list[HttpUrl]
    labels: list[str]
    scores: list[float]
    prediction_time: float