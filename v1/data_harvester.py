import cv2
import os
import time

print("--- Senthron Data Harvester Initialized ---")

# 1. Define target parameters dynamically
emotion = input("Enter the target cognitive state (e.g., happy, angry, surprise): ").lower().strip()
target_frames = int(input(f"How many data points (images) for '{emotion}'? (e.g., 500): "))

save_path = f"dataset/{emotion}"
os.makedirs(save_path, exist_ok=True)

# 2. Initialize visual sensor
cap = cv2.VideoCapture(0)
print(f"Prepare to express: {emotion.upper()}")
print("Initiating capture sequence in 3 seconds...")
time.sleep(3)

count = 0
print(f"Harvesting {target_frames} frames. Please hold the expression...")

while count < target_frames:
    ret, frame = cap.read()
    if not ret:
        print("Camera feed failed.")
        break
    
    # Save the frame directly into the labeled folder
    file_name = f"{save_path}/{emotion}_{count}.jpg"
    cv2.imwrite(file_name, frame)
    
    # Visual feedback HUD
    cv2.putText(frame, f"Extracting {emotion.upper()}: {count}/{target_frames}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Senthron Data Harvester", frame)
    
    count += 1
    cv2.waitKey(10) # 10ms delay between captures

cap.release()
cv2.destroyAllWindows()
print(f"Data Acquisition Complete: Saved to {save_path}")
