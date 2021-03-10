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
from cv2 import resize, INTER_AREA, INTER_CUBIC
from flask import Flask, request,jsonify,render_template,redirect,url_for, session
import os
import json
from json import JSONEncoder
import io


app = Flask(__name__)
app.secret_key="1234"
def get_model():
    global model
    model = load_model('model_interarea_intercubic.h5')
    print(" * Model loaded!")

def preprocess_image(image):
    if image.shape != (40, 40):
        if image.size >= 1600:
            image = resize(image, dsize=(40, 40), interpolation=INTER_AREA)
        elif image.size < 1600:
            image = resize(image, dsize=(40, 40), interpolation=INTER_CUBIC)
    image = image.reshape((1, *image.shape, 1))
    return image

def preprocess_image1(image):
    if image.shape != (40, 40):
        if image.size >= 1600:
            image1 = resize(image, dsize=(40, 40), interpolation=INTER_AREA)
        elif image.size < 1600:
            image1 = resize(image, dsize=(40, 40), interpolation=INTER_CUBIC)
    return image1

print(" * Loading Keras model...")
get_model()


@app.route('/',methods=['GET','POST'])
def index():
    if request.method == "POST":        
        if request.files:
            raw_data = request.files['image'].read()         
            finalNumpyArray=np.array(json.loads(raw_data), dtype = 'uint8')
            processed_image = preprocess_image(finalNumpyArray)
            prediction = model.predict(processed_image).tolist()
            # breakpoint()
            response ={
                    'Center': str(round(100*prediction[0][0], 3)) + ' %',
                    'Donut': str(round(100*prediction[0][1], 3)) + ' %',
                    'Edge-Loc': str(round(100*prediction[0][2], 3)) + ' %',
                    'Edge-Ring': str(round(100*prediction[0][3], 3)) + ' %',
                    'Loc': str(round(100*prediction[0][4], 3)) + ' %',
                    'Random': str(round(100*prediction[0][5], 3)) + ' %',
                    'Scratch': str(round(100*prediction[0][6], 3)) + ' %',
                    'None': str(round(100*prediction[0][7], 3)) + ' %',
                }    
            # breakpoint()
            return render_template('predict.html',prediction=response)
    return render_template('semiconui.html')

@app.route('/predict/',methods=['GET','POST'])
def predict():
    if "response" in session:
        prediction=session['response']
        return render_template('predict.html', prediction=prediction)
    else:
        return redirect(url_for('index'))

    

if __name__ == '__main__':
    port=int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0', port=port)
    
    