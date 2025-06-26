from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load CNN model
cnn_model = tf.keras.models.load_model("cnn_model.h5")

# Label list (must match model training)
cnn_labels = ['drink', 'food', 'inside', 'menu', 'outside']

def preprocess_image(image):
    image = image.resize((224, 224))  # or whatever your model input is
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    image = Image.open(request.files['image']).convert('RGB')
    processed = preprocess_image(image)

    prediction = cnn_model.predict(processed)[0]
    predicted_class = cnn_labels[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)

    return render_template(
        "index.html",
        predicted_class=predicted_class,
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 8001,debug=True)
