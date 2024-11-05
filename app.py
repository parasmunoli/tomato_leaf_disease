from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

model_paths = {
    'inception': 'inception_model.tflite',
    'densenet': 'densenet_model.tflite',
    'vgg16': 'VGG16_model.tflite'
}

models = {}
for name, model_path in model_paths.items():
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        models[name] = interpreter
    except Exception as e:
        print(f"Error loading model {name}: {e}")

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
    return image.astype(np.float32)

def get_model_prediction(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(prediction, axis=1)[0]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected image file'}), 400

        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if image_file.filename.split('.')[-1].lower() in allowed_extensions:
            image = Image.open(image_file)
            preprocessed_image = preprocess_image(image, target_size=(224, 224))

            model_choice = request.form.get('model', 'all')
            predictions = {}

            if model_choice == 'all':
                for name, interpreter in models.items():
                    try:
                        predictions[name] = class_labels[get_model_prediction(interpreter, preprocessed_image)]
                    except Exception as e:
                        predictions[name] = f"Error: {str(e)}"
                return jsonify(predictions)

            try:
                selected_model = models[model_choice]
                prediction = class_labels[get_model_prediction(selected_model, preprocessed_image)]
                return jsonify({f'{model_choice}_prediction': prediction})
            except KeyError:
                return jsonify({'error': f'Invalid model choice. Available options are: {", ".join(models.keys())}, or "all".'}), 400
            except Exception as e:
                return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

        return jsonify({'error': 'Invalid image file format'}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'Service is running'}), 200

if __name__ == '__main__':
    app.run(debug=True)
