from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

model = load_model('my_lstm_model.h5')

with open('tokenizerlstm.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100  # Change this if needed

@app.route('/')
def home():
    return "API is working. POST to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        text = data['text']
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        pred = model.predict(padded)
        return jsonify({'prediction': pred.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
