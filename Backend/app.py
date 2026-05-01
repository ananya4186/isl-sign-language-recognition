from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "ISL Backend is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Check if image was sent
    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # MOCK response for now (real model comes in Week 4)
    mock_labels = ['A', 'B', 'C', 'Hello', 'Yes', 'No', 'Help', 'Water']
    label = random.choice(mock_labels)
    confidence = round(random.uniform(0.75, 0.99), 2)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "hand_detected": True
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)