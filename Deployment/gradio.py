# -*- coding: utf-8 -*-
import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import json

# Load model and label map
model = load_model("finetuned_vgg.h5")

with open("label_map.json") as f:
    label_map = json.load(f)

index_to_label = {v: k for k, v in label_map.items()}

# Preprocess image
def preprocess_image(image):
    image = image.resize((224, 224)).convert("RGB")
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict
def predict_fn(image):
    img_input = preprocess_image(image)
    preds = model.predict(img_input)[0]
    predicted_class = index_to_label[np.argmax(preds)]
    return f"{predicted_class} ({np.max(preds)*100:.2f}%)"

gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Text(label="Predicted Label"),
    title="VGG Image Classifier",
    description="Upload an image and get the predicted class",
).launch(share=True)