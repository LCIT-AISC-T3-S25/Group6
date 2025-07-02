from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Load VGG model
vgg_model = tf.keras.models.load_model("vgg_model.h5")

# Load label map
with open("label_map.json", "r") as f:
    vgg_label_map = json.load(f)

index_to_label = {v: k for k, v in vgg_label_map.items()}

def preprocess_image(image):
    image = image.resize((224, 224))  # match your VGG input
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    image = Image.open(request.files['image']).convert('RGB')
    processed = preprocess_image(image)

    prediction = vgg_model.predict(processed)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = index_to_label.get(predicted_index, "Unknown")
    confidence = round(float(np.max(prediction)) * 100, 2)

    return render_template(
        "index.html",
        predicted_class=predicted_class,
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 8002,debug=True)
