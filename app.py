import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input as xcep_preprocess
import numpy as np
import cv2

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "static/uploaded")
MODEL_PATH = os.path.join(APP_ROOT, "xception_dr_model.h5")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!")

IMG_SIZE = (299, 299)

def preprocess_xception_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    img = xcep_preprocess(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_dr(img_path):
    img = preprocess_xception_image(img_path)
    prob = model.predict(img)[0][0]
    label = "DR" if prob >= 0.5 else "No DR"
    return label, round(float(prob), 4)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def upload_predict():
    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    label, prob = predict_dr(filepath)
    return render_template("result.html",
                           image_path="static/uploaded/" + file.filename,
                           label=label,
                           probability=prob)

if __name__ == "__main__":
    app.run(debug=True)
