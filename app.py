from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import tensorflow as tf
from flask_cors import CORS
import gdown
from werkzeug.utils import secure_filename
import os


app = Flask(__name__)
CORS(app)


def download_model(model_id, model_path):
    model_url = f'https://drive.google.com/uc?id={model_id}'
    if not os.path.exists(model_path):
        print(f"Downloading model...")
        gdown.download(model_url, model_path, quiet=False)
    else:
        print("Model already exists. Skipping download...")


model_ids = {
    'inception': '1JQQkD-8jRQXI9Twp7BjJu_kxlmOqiljy',
    # 'densenet': '1OYIlqOvuQgD4OLKA0HAa5HCb_GovcWvI',
    # 'vgg16': '1bHoCvQIs7_jI12W50CyREDZHKnZwavnt'
}

models = {}
for name, model_id in model_ids.items():
    model_path = 'path_to_your_model.h5'
    download_model(model_id, model_path)
    models[name] = joblib.load(model_path)


class_labels = [
    "Tomato Bacterial spot", 
    "Tomato Early blight", 
    "Tomato Healthy",
    "Tomato Late blight", 
    "Tomato Leaf Mold", 
    "Tomato Mosaic virus", 
    "Tomato Septoria leaf spot",
    "Tomato Spider mites", 
    "Tomato Target Spot", 
    "Tomato Yellow Leaf Curl Virus"
]


def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def get_model_prediction(model, image):
    prediction = model.predict(image)
    return np.argmax(prediction, axis=1)[0]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        # Check if the file is selected
        if image_file.filename == '':
            return jsonify({'error': 'No selected image file'}), 400
        
        # Check if the file has an allowed extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if image_file.filename.split('.')[-1].lower() in allowed_extensions:
            image = Image.open(image_file)
            
            # Preprocess image and make prediction
            preprocessed_image = preprocess_image(image, target_size=(224, 224))
            
            model_choice = request.args.get('model', 'all')
            
            if model_choice == 'all':
                predictions = {name: class_labels[get_model_prediction(model, preprocessed_image)] for name, model in models.items()}
                return jsonify(predictions)
            
            selected_model = models.get(model_choice)
            if selected_model:
                prediction = class_labels[get_model_prediction(selected_model, preprocessed_image)]
                return jsonify({f'{model_choice}_prediction': prediction})
            
            return jsonify({'error': f'Invalid model choice. Available options are: {", ".join(models.keys())}, or "all".'}), 400
        
        return jsonify({'error': 'Invalid image file format'}), 400
    
    return jsonify({'error': 'Invalid request method'}), 400


if __name__ == '__main__':
    app.run(debug=True)