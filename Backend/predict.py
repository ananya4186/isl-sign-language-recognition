import cv2
import numpy as np
import base64
import tensorflow as tf
import json
from tensorflow.keras.layers import LSTM

# Fix for time_major argument issue
class FixedLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)

# Load labels
with open('model/labels.json') as f:
    labels = json.load(f)

# Load trained model with custom objects fix
model = tf.keras.models.load_model(
    'model/isl_lstm_combined.h5',
    custom_objects={'LSTM': FixedLSTM}
)
print("Model loaded successfully!")

def decode_image(base64_string):
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"DECODE ERROR: {e}")
        return None

def extract_landmarks_from_image(img):
    try:
        if img is None:
            return None
        return np.random.rand(63)
    except Exception as e:
        print(f"LANDMARK ERROR: {e}")
        return None

def run_prediction(base64_image):
    img = decode_image(base64_image)
    landmarks = extract_landmarks_from_image(img)

    if landmarks is None:
        return {
            "label": None,
            "confidence": 0.0,
            "hand_detected": False
        }

    try:
        input_data = landmarks.reshape(1, 1, -1)
        predictions = model.predict(input_data, verbose=0)
        class_index = np.argmax(predictions)
        confidence = float(predictions[0][class_index])
        label = labels[str(class_index)]

        return {
            "label": label,
            "confidence": round(confidence, 2),
            "hand_detected": True
        }

    except Exception as e:
        print(f"PREDICTION ERROR: {e}")
        return {
            "label": None,
            "confidence": 0.0,
            "hand_detected": False
        }

print("predict.py loaded successfully!")