import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from keras.models import load_model
import cv2
import remote_pdb
from flask import Flask, request,jsonify,render_template,redirect
import os
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


app = Flask(__name__)

def get_model():
    global model
    model = load_model('model_interarea_intercubic.h5')
    print(" * Model loaded!")

def preprocess_image(image):
    if image.shape != (40, 40):
        if image.size >= 1600:
            image=image.astype('float32')
            image = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_AREA)
        elif image.size < 1600:
            image=image.astype('float32')
            image = cv2.resize(image, dsize=(40, 40), interpolation=cv2.INTER_CUBIC)
    image = image.reshape((1, *image.shape, 1))
    return image

print(" * Loading Keras model...")
get_model()

@app.route('/')
def index():
    return render_template('hello.html')


@app.route("/predict", methods=["GET","POST"])
def predict():    
    if request.method == "POST":        
        if request.files:
            raw_data = request.files['image'].read()            
            finalNumpyArray=np.array(json.loads(raw_data))
            processed_image = preprocess_image(finalNumpyArray)    
            prediction = model.predict(processed_image).tolist()
            response = {
                'prediction': {
                    'Center': prediction[0][0],
                    'Donut': prediction[0][1],
                    'Edge-Loc': prediction[0][2],
                    'Edge-Ring': prediction[0][3],
                    'Loc': prediction[0][4],
                    'Random': prediction[0][5],
                    'Scratch': prediction[0][6],
                    'None': prediction[0][7],
                }
            }
            return jsonify(response)
    return render_template("predict.html")

if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)