from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pickle
import os

# Use lightweight TFLite runtime
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="bbc_news_lstm_model.tflite")
interpreter.allocate_tensors()

# Get input/output details once (optimization)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

max_len = 300

app = Flask(__name__)

# Serve frontend
@app.route('/')
def serve_index():
    return send_from_directory(os.path.join(app.root_path, 'frontend'), 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'frontend'), filename)

# Manual padding (removes TensorFlow dependency)
def pad_sequence(seq, max_len):
    padded = seq[:max_len] + [0] * (max_len - len(seq))
    return padded

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequence(seq, max_len)

    input_data = np.array([padded], dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_class = np.argmax(output_data, axis=1)[0]
    label = le.inverse_transform([pred_class])[0]

    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)