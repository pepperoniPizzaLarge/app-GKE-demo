import os
import time
import socket
import torch
from transformers import pipeline
from scripts.data_model import NLPDataInput, ImageDataInput, NLPDataOutput, ImageDataOutput
from scripts import gcb_utils
from fastapi import FastAPI

GGACCESSKEYID = os.getenv("GGACCESSKEYID")
HMACKEY = os.getenv("HMACKEY")

# Download model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = 'vit-human-pose/'
local_path = 'ml-models/' + model_name

force_download = True  # True to force download w/o regards to existing dir


if not os.path.isdir(local_path) or force_download:
    gcb_utils.download_dir(local_path=local_path, 
                           model_name=model_name, 
                           google_access_key_id=GGACCESSKEYID,
                           google_access_key_secret=HMACKEY)

pose_classifier = pipeline('image-classification', model=local_path, device=device)


# Serving FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return f"API is up at {socket.gethostname()}."


@app.post("/api/v1/sentiment_analysis")
def sentiment_analysis(data: NLPDataInput):
    return data


@app.post("/api/v1/disaster_tweet")
def disaster_tweet(data: NLPDataInput):
    return data


@app.post("/api/v1/human_pose")
def human_pose(data: ImageDataInput):
    start_time = time.time()
    urls = [str(x) for x in data.url]
    output = pose_classifier(urls)
    end_time = time.time()
    prediction_time = end_time - start_time
    
    labels = [x[0]['label'] for x in output]
    scores = [x[0]['score'] for x in output]
    
    output = ImageDataOutput(model_name='human-pose-class',
                             url=data.url,
                             labels=labels,
                             scores=scores,
                             prediction_time=prediction_time
    )
    return output