import cv2
import numpy as np
import tensorflow as tf
import os

print("--- Senthron Live Network Initializing ---")

# 1. Load your custom brain and the class names
model = tf.keras.models.load_model("senthron_cognitive_core.keras")

# Automatically retrieve class names based on your folder structure
dataset_path = "dataset"
class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
print(f"Loaded States: {class_names}")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Pre-process the frame to match the brain's requirements
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(frame_rgb, (224, 224))
    input_tensor = np.expand_dims(resized_frame, axis=0) # Create a batch of 1

    # 3. Execute Neural Prediction
    predictions = model.predict(input_tensor, verbose=0)
    best_match_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    current_emotion = class_names[best_match_index].capitalize()

    # 4. Render HUD
    text = f"State: {current_emotion} ({confidence:.1f}%)"
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Senthron Custom Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
