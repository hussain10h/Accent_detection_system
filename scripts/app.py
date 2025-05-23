import os
import streamlit as st
import numpy as np
import joblib
from keras.models import load_model
from utils import extract_features
from pathlib import Path
def create_path():
    app_dir = Path(__file__)

# Paths to model and encoder
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "accent_nn_model.h5"))
ENCODER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "label_encoder.pkl"))

# Load model and encoder
model = load_model(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# Streamlit UI
st.title("🎙️ Accent Detection App")
st.markdown("Upload a .wav or .mp3 file to identify the English accent.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save to temporary file
    temp_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_features(temp_path)

    if features is not None:
        X_input = features.reshape(1, -1)
        prediction = model.predict(X_input)
        predicted_label = le.inverse_transform([np.argmax(prediction)])
        st.success(f"✅ Predicted Accent: {predicted_label[0]}")
    else:
        st.error("❌ Feature extraction failed. Please try another audio file.")
