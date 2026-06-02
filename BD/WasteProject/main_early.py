



import cv2
from ultralytics import YOLO
import os
import time

# ==========================================
# 1. CONFIGURATION & SOURCE SELECTION
# ==========================================
# Options: "FILE", "IP_CAM", "USB_CAM"
SOURCE_MODE = "FILE" 

# Set your paths/URLs here

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "video2.mp4")

IP_URL = "http://192.168.1.15:8080/video"
USB_INDEX = 0  # 0 is usually default, 1 or 2 for DroidCam

# Path to your trained model
# Use the absolute path you found to avoid any "file not found" errors
MODEL_PATH = r"C:\Users\davis\Desktop\bakalaurs\practical\runs\waste_v26_early_trash\weights\best.pt"

# ==========================================
# 2. INITIALIZE MODEL & SOURCE
# ==========================================
# Load your model (or fallback to base if best.pt isn't found)
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"Loaded custom model: {MODEL_PATH}")
else:
    model = YOLO('yolo26n.pt')
    print("Warning: best.pt not found. Using default YOLOv26n.")

print(model.names)
# Select the input source based on SOURCE_MODE
if SOURCE_MODE == "FILE":
    input_source = VIDEO_PATH
elif SOURCE_MODE == "IP_CAM":
    input_source = IP_URL
else:
    input_source = USB_INDEX

cap = cv2.VideoCapture(input_source)

# Timing logic for consistent video playback
wait_time = 1
if SOURCE_MODE == "FILE":
    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 33

# ==========================================
# 3. MAIN PROCESSING LOOP
# ==========================================
print(f"Starting stream from {SOURCE_MODE}...")

while True:
    ret, frame = cap.read()
    
    if not ret:
        if SOURCE_MODE == "FILE":
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue
        break

    # Run the model on the frame
    # conf=0.3 means ignore anything the AI is less than 30% sure about
    # Change conf from 0.25 to 0.5 or 0.6
    # results = model(frame, conf=0.6, iou=0.45) # Use 'cuda' for GPU speed!
    results = model(frame, imgsz=640, conf=0.5)

    # Draw boxes and labels on the frame
    annotated_frame = results[0].plot()

    # Add a text overlay for clarity
    cv2.putText(annotated_frame, f"Mode: {SOURCE_MODE}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Waste Sorting System', annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



print("--- Starting Video Processing ---")
results = model.predict(
    source=VIDEO_PATH, 
    save=True, 
    conf=0.5, 
    imgsz=416, 
    project="C:/Users/davis/Desktop/bakalaurs/practical/WasteProject/processed_results",
    name="waste_video_test1"
)

print(f"--- Finished! Check the folder: processed_results/waste_video_test ---")












