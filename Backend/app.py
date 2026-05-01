from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import decode_image, extract_landmarks_from_image
import random

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "ISL Backend is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    try:
        # Decode the image
        img = decode_image(data['image'])

        # Extract hand landmarks
        landmarks = extract_landmarks_from_image(img)

        if landmarks is None:
            return jsonify({
                "label": None,
                "confidence": 0.0,
                "hand_detected": False
            })

        # MOCK prediction for now (real model comes from Riya in Week 4)
        mock_labels = ['A', 'B', 'C', 'Hello', 'Yes', 'No', 'Help', 'Water']
        label = random.choice(mock_labels)
        confidence = round(random.uniform(0.75, 0.99), 2)

        return jsonify({
            "label": label,
            "confidence": confidence,
            "hand_detected": True
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)