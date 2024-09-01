from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('cancer_detection_model.h5')


def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to the input size of the model
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image = Image.open(io.BytesIO(file.read()))
    image = preprocess_image(image)
    prediction = model.predict(image)

    return jsonify({'prediction': 'cancer' if prediction[0][0] > 0.5 else 'non_cancer'})


if __name__ == '__main__':
    app.run(debug=True)