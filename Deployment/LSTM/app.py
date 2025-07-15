from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)

# Load tokenizer
with open("tokenizer_lstm.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load LSTM model
lstm_model = tf.keras.models.load_model("sentiment_model.h5")

MAX_SEQUENCE_LENGTH = 100
sentiment_labels = ['Negative', 'Positive']

def preprocess(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    input_text = request.form['input_text']
    input_processed = preprocess(input_text)

    prediction = lstm_model.predict(input_processed)[0][0]  # Sigmoid output
    predicted_class = 1 if prediction >= 0.5 else 0
    confidence = round(float(prediction if predicted_class == 1 else 1 - prediction) * 100, 2)

    return render_template(
        "index.html",
        input_text=input_text,
        predicted_class=sentiment_labels[predicted_class],
        confidence=confidence
    )

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port = 5030,debug=True)
