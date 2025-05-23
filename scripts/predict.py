# predict.py – Predict accent from a single audio file using trained Keras model
import os
import numpy as np
import joblib
from keras.models import load_model
from utils import extract_features

# Paths to model and encoder
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "accent_nn_model.h5"))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl"))
TEST_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_audio.wav"))

# Load model and encoder
model = load_model(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Extract features
features = extract_features(TEST_FILE)
if features is None:
    print("❌ Feature extraction failed. Cannot make prediction.")
else:
    X_input = features.reshape(1, -1)
    prediction = model.predict(X_input)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    print("🎧 Predicted Accent:", predicted_label[0])
