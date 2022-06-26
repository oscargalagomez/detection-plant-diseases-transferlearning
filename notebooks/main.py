import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_file 
from uvicorn import run
import os
from io import BytesIO

from keras import preprocessing
from PIL import Image

app = FastAPI(
	title="Plant Disease API",
    	description="A simple API that uses a pre-trained MobileNetV2 model to predict the diseases of the plants",
    	version="0.1")

model_dir = "/Users/oscargalagomez/Jupyter Notebook/tfm-disease-plants/mobilenetV2/v2-05-MobileNetV2-model"
train_dir = "/Users/oscargalagomez/tmp/images"
model = load_model(model_dir)

im_width = 224
im_height = 224
channels=3

#trained_classes_labels = os.listdir(train_dir)

trained_classes_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                         'Apple___healthy', 'Blueberry___healthy', 'Cherry___healthy',
                         'Cherry___Powdery_mildew', 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
                         'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
                         'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
                         'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy',
                          'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
                          'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
                          'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
                          'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                          'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
                         ]

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

@app.get("/")
async def root():
    return {"message": "Welcome to the Plant Disease API!"}

@app.post("/prediction/image")
async def prediction(file: UploadFile = File(...)):
    
    loaded_image = read_imagefile(await file.read())

    loaded_image = np.asarray(loaded_image.resize((im_width, im_height)))[..., : channels]
    img_array = preprocessing.image.img_to_array(loaded_image)/255.
    img_array = np.expand_dims(img_array, axis = 0)

    predictions = model.predict(img_array)
    
    classidx = np.argmax(predictions[0])
    label = trained_classes_labels[classidx]
    
    predictions_pct = ["{:.2f}%".format(prob * 100) for prob in predictions[0] ]

    return {
        "model-prediction": label,
        "model-prediction-confidence-score": predictions_pct[classidx]
    }

#if __name__ == "__main__":
    
#    port = int(os.environ.get('PORT', 5000))
#   run(app, host="0.0.0.0", port=port)