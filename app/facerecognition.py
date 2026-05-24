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

gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


def pipeline_model(path):
    try:
        # Load image
        img = cv2.imread(path)
        if img is None:
            pil_img = Image.open(path).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
       
        predictions = []
        eigen_image_paths = []

        print(f"🔍 Found {len(faces)} face(s)")

        for idx, (x, y, w, h) in enumerate(faces, start=1):
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

            text = f"{gender_name} ({int(score * 100)}%)"
            cv2.putText(img, text, (x, y - 10), font, 0.8, (0, 255, 255), 2)
           
            predictions.append({
                'gender': gender_name,
                'score': round(score * 100, 2)
            })

            # ================== Generate Eigen Face ==================
            try:
                eigen_visual = model_pca.inverse_transform(eigen_image).reshape(150, 150)
                vmin = eigen_visual.min()
                vmax = eigen_visual.max()
                eigen_visual = (eigen_visual - vmin) / (vmax - vmin + 1e-8) * 255
                eigen_visual = eigen_visual.astype('uint8')
                
                eigen_image_path = f"eigen_face_{idx}.jpg"
                save_path = os.path.join('static/predict', eigen_image_path)
                
                success = cv2.imwrite(save_path, eigen_visual)
                print(f"✅ Eigen face saved → {save_path} | Success: {success}")
                eigen_image_paths.append(eigen_image_path)
                
            except Exception as e:
                print(f"❌ Eigen face error: {e}")

        if len(faces) == 0:
            print("⚠️ No face detected")
            cv2.putText(img, "No Face Detected", (30, 50), font, 1.0, (0, 0, 255), 2)

        return img, predictions, eigen_image_paths
    except Exception as e:
        print(f"❌ Critical Error in pipeline_model: {e}")
        # Return safe default values to avoid crash
        return None, [], None