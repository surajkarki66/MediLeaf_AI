from fastapi import FastAPI, File
from fastapi.middleware.cors import CORSMiddleware

import rembg
import tensorflow as tf

from MediLeaf_AI.utils.common import image_to_array, map_predictions_to_species_with_proba, add_white_background, read_imagefile

model_dir = "./artifacts/training/models/mobilenet.keras"
model = tf.keras.models.load_model(model_dir)
session = rembg.new_session("u2netp")

app = FastAPI()

origins = ["*"]
methods = ["GET", "POST", "OPTIONS"]
headers = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

@app.get("/")
async def root():
    return {"message": "Welcome to MediLeaf AI!"}

@app.post("/api/v1/classify/")
async def classify(input_image: bytes = File(...)):
    img = read_imagefile(input_image)
    img = add_white_background(session, img, size=(1600, 1200))
    img_arr = image_to_array(img)
    result = model.predict(img_arr)
    prediction_response = map_predictions_to_species_with_proba(result, "./classes.json")
    
    return prediction_response
    