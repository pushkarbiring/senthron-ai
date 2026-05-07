import cv2
import numpy as np
import tensorflow as tf
import os
import json
from collections import deque
 
print("--- Senthron Live ---")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {tf.keras.__version__}")
 
# Compatible registration for all Keras versions
try:
    register = tf.keras.saving.register_keras_serializable()
except AttributeError:
    try:
        register = tf.keras.utils.register_keras_serializable()
    except AttributeError:
        register = lambda cls: cls
 
@register
class MobileNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return tf.keras.applications.efficientnet.preprocess_input(x)
 
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "senthron_best.keras")
class_names_path = os.path.join(base_dir, "class_names.json")
dataset_path = os.path.join(base_dir, "dataset")
 
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")
 
# Load model
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"MobileNetPreprocess": MobileNetPreprocess},
    compile=False
)
print("Model loaded.")
 
# Load class names
if os.path.exists(class_names_path):
    with open(class_names_path, "r") as f:
        class_names = json.load(f)
    print("Class names loaded:", class_names)
else:
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    print("Warning: using folder names:", class_names)
 
# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
 
buffer = deque(maxlen=5)
cap = cv2.VideoCapture(0)
 
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")
 
print("Press 'q' to quit.")
 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
 
    for (x, y, w, h) in faces:
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
 
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
 
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_float = face_resized.astype(np.float32)
        input_tensor = np.expand_dims(face_float, axis=0)
 
        preds = model.predict(input_tensor, verbose=0)[0]
        buffer.append(preds)
        avg_pred = np.mean(buffer, axis=0)
 
        best = np.argmax(avg_pred)
        confidence = float(np.max(avg_pred)) * 100
 
        sorted_preds = np.sort(avg_pred)
        gap = float(sorted_preds[-1] - sorted_preds[-2])
 
        if confidence < 35 or gap < 0.1:
            emotion = "Uncertain"
            color = (128, 128, 128)
        else:
            emotion = class_names[best]
            color = (0, 255, 0)
 
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
 
    cv2.imshow("Senthron Live", frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
print("For More Visit : https://pushkarbiring.github.io/senthron")
print("Thanks for using SENTHRON, See you next time!")
