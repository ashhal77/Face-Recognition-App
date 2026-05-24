from flask import render_template, request
import os
import cv2
import time
from app.facerecognition import pipeline_model

UPLOAD_FOLDER = 'static/upload'
PREDICT_FOLDER = 'static/predict'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)


def index():
    return render_template('index.html')


def app():                    # ← Must keep this name to match main.py
    return render_template('app.html')


def genderapp():
    if request.method == 'POST':
        f = request.files['image-name']
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(path)
       
        pred_image, predictions = pipeline_model(path)
        
        timestamp = int(time.time())
        pred_filename = "Prediction_Image.jpg"
        cv2.imwrite(os.path.join(PREDICT_FOLDER, pred_filename), pred_image)
        
        return render_template('gender.html', 
                             fileupload=True, 
                             pred_image=pred_filename,
                             timestamp=timestamp,
                             predictions=predictions)   # ← This was missing!
    
    return render_template('gender.html', fileupload=False)