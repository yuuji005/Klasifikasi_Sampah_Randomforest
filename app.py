from flask import Flask, render_template, request
import numpy as np
import cv2
import pickle
import os
import random
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder upload
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_cnn_model():
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(categories))
    model.load_state_dict(torch.load("model_cnn.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model_cnn = None

def get_cnn_model():
    global model_cnn
    if model_cnn is None:
        model_cnn = load_cnn_model()
    return model_cnn

# Transforms for MobileNetV2
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load kategori
categories = pickle.load(open('labels.pkl', 'rb'))

# Load dataset stats
dataset_stats = {}
try:
    dataset_path = os.path.join(app.root_path, "dataset", "Garbage classification", "Garbage classification")
    if os.path.exists(dataset_path):
        for folder in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder)
            if os.path.isdir(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                dataset_stats[folder] = len(files)
    else:
        # Fallback statistik jika berjalan di Render tanpa membawa folder dataset yang besar
        dataset_stats = {
            'cardboard': 403,
            'glass': 501,
            'metal': 410,
            'paper': 594,
            'plastic': 482,
            'trash': 137
        }
except Exception as e:
    print("Error loading dataset stats:", e)

# Ukuran gambar
IMG_SIZE = 64

# History prediksi
prediction_history = []

@app.route('/', methods=['GET', 'POST'])
def index():

    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':

        file = request.files.get('image')
        selected_category = request.form.get('category')

        filepath = None

        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = os.path.join('static', 'uploads', filename).replace('\\', '/')

        elif selected_category:
            # Memuat sampel dari static/samples (agar deploy ke Render lebih ringan tanpa membawa seluruh dataset)
            cat_folder = os.path.join(app.root_path, "static", "samples", selected_category)

            if os.path.exists(cat_folder):
                files = [f for f in os.listdir(cat_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    sample_file = random.choice(files)
                    sample_path = os.path.join(cat_folder, sample_file)

                    filename = f"sample_{selected_category}_{sample_file}"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    shutil.copy(sample_path, filepath)
                    image_path = os.path.join('static', 'uploads', filename).replace('\\', '/')

        if filepath and os.path.exists(filepath):

            # =========================
            # PREPROCESSING & PREDICTION
            # =========================

            # Read image using numpy to support unicode/non-ASCII paths on Windows
            file_bytes = np.fromfile(filepath, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is not None:
                # Convert BGR to RGB for PyTorch model
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_img = Image.fromarray(img_rgb)
                
                # Preprocess image
                img_tensor = inference_transform(pil_img).unsqueeze(0).to(device)

                # Get CNN model
                cnn = get_cnn_model()

                # Predict
                with torch.no_grad():
                    outputs = cnn(img_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                    
                    pred = pred_idx.item()
                    prediction = categories[pred]
                    confidence = round(conf.item() * 100, 2)

                # Simpan history
                prediction_history.append(prediction)
            else:
                prediction = "Gagal memuat gambar"
                confidence = 0.0

    # =========================
    # Statistik
    # =========================

    stats = {cat: 0 for cat in categories}
    for item in prediction_history:
        if item in stats:
            stats[item] += 1

    ordered_dataset_stats = {cat: dataset_stats.get(cat, 0) for cat in categories}

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        history=prediction_history,
        stats=stats,
        dataset_stats=ordered_dataset_stats
    )

if __name__ == '__main__':
    app.run(debug=True)