# train.py – Using Keras Neural Network (compatible with TensorFlow 2.x)
import os
import numpy as np
import joblib
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from utils import extract_features

from pathlib import Path
def create_path():
    train_dir = Path(__file__)

# Paths
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
REPORT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "training_report.txt"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)


X = []
y = []

# Load audio data
for accent_folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, accent_folder)
    if not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        if filename.endswith(('.wav', '.mp3')):
            file_path = os.path.join(folder_path, filename)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(accent_folder.replace("_convert", ""))

if len(X) == 0 or len(y) == 0:
    print("❌ No training data was collected. Check your audio files.")
    exit()

X = np.array(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(X.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_test, y_test))

# Save the model and label encoder
model.save(os.path.join(MODEL_DIR, "accent_nn_model.h5"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Model accuracy: {acc:.2f}")

# Save accuracy report
with open(REPORT_PATH, "w") as f:
    f.write("Model Training Summary\n")
    f.write("======================\n")
    f.write(f"Accuracy: {acc:.2f}\n")
