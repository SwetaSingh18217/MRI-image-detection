from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import joblib
from PIL import Image, ImageEnhance
import random

app = Flask(__name__)

# Load models safely
vgg16_model = load_model("models/model.h5")

dt_model_path = "models/decision_tree.pkl"
rf_model_path = "models/random_forest.pkl"

dt_model = joblib.load(dt_model_path) if os.path.exists(dt_model_path) else None
rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None

# Load the VGG16 model for feature extraction
feature_extractor = VGG16(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
feature_extractor = Model(inputs=feature_extractor.input, outputs=feature_extractor.output)

train_dir = 'archive/Training'
class_labels = sorted(os.listdir(train_dir))

UPLOAD_FOLDER = "./uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def augment_image(image):
    image = img_to_array(image)
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    image = np.array(image) / 255.0
    return image

def extract_features(img_array):
    """Extracts VGG16 features and flattens the output."""
    features = feature_extractor.predict(img_array)
    return features.reshape(1, -1)  # Ensure it has the same shape as the training features

def predict_tumor(image_path, model_type):
    img = load_img(image_path, target_size=(128, 128))
    img = augment_image(img)
    img_array = np.expand_dims(img, axis=0)

    if model_type == "vgg16":
        predictions = vgg16_model.predict(img_array, verbose=0)
    else:
        img_features = extract_features(img_array)  # Extract VGG16 features first
        if model_type == "decision_tree" and dt_model:
            predictions = dt_model.predict_proba(img_features)
        elif model_type == "random_forest" and rf_model:
            predictions = rf_model.predict_proba(img_features)
        else:
            return "Model not available", 0.0  # Handle missing models gracefully

    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index]
    result = "No Tumor" if class_labels[predicted_class_index] == 'notumor' else f"Tumor: {class_labels[predicted_class_index]}"
    return result, confidence_score

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        model_type = request.form['model']
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)
            result, confidence = predict_tumor(file_location, model_type)
            return render_template('index.html', result=result, confidence=f'{confidence*100:.2f}%', file_path=f'/uploads/{file.filename}')
    return render_template('index.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)