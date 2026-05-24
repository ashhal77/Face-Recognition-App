import numpy as np
import cv2
import pickle
import os
from PIL import Image

import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ================== Paths ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'model'))

haar_path = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
mean_path = os.path.join(MODELS_DIR, 'mean_preprocess.pickle')
svm_path = os.path.join(MODELS_DIR, 'model.svm.pickle')
pca_path = os.path.join(MODELS_DIR, 'pca_50.pickle')

# ================== Load Models ==================
haar = cv2.CascadeClassifier(haar_path)
mean = pickle.load(open(mean_path, 'rb'))
model_svm = pickle.load(open(svm_path, 'rb'))
model_pca = pickle.load(open(pca_path, 'rb'))

print('✅ Models loaded successfully')
print(f"Mean shape: {mean.shape} | PCA features: {model_pca.n_features_in_}")

gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


def pipeline_model(path):
    img = cv2.imread(path)
    if img is None:
        from PIL import Image
        pil_img = Image.open(path).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
    
    predictions = []
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        roi = gray[y:y + h, x:x + w]
        roi = roi / 255.0
        roi_resize = cv2.resize(roi, (150, 150), cv2.INTER_AREA)
        roi_reshape = roi_resize.reshape(1, -1)
        
        roi_mean = roi_reshape - roi_reshape.mean()
        
        eigen_image = model_pca.transform(roi_mean)
        results = model_svm.predict_proba(eigen_image)[0]
        
        predict = results.argmax()
        score = results[predict]
        gender_name = gender_pre[predict]
        
        # Smaller text on image
        text = f"{gender_name} ({int(score * 100)}%)"
        cv2.putText(img, text, (x, y - 10), font, 0.8, (0, 255, 255), 2)
        
        predictions.append({
            'gender': gender_name,
            'score': round(score * 100, 2)
        })
    
    if len(faces) == 0:
        cv2.putText(img, "No Face Detected", (30, 50), font, 1.0, (0, 0, 255), 2)
    
    return img, predictions