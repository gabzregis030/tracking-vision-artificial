import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.detector import ObjectDetector

def debug_detection_simulation(image_path):
    print(f"DEBUG SIMULATION: Analyzing image: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found")
        return
    frame = cv2.imread(image_path)
    
    # Initialize Detector (Tiny, default)
    detector = ObjectDetector(method="yolo", model_type="tiny", confidence_threshold=0.20, nms_threshold=0.5)
    
    # Step 1: Normal Detection (Print EVERYTHING)
    print("\n--- Step 1: Normal Detection (Threshold 0.05, Look at everything) ---")
    detections = detector.detect(frame, target_classes=None, confidence_threshold=0.05)
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f" - {det}")
        
    # Check for 'cat' specifically in the list
    cat_found = any(d[4] == 'cat' for d in detections)
    print(f"Cat found in raw list? {cat_found}")

    # Skip step 2 for this debug run

    # Check if we can switch to YOLOv3 standard if tiny fails
    # This checks if the file exists and we can manually force it if needed in a future step
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../src/models/yolo')
    print(f"\nChecking available models in {base_path}:")
    for f in os.listdir(base_path):
        print(f" - {f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Update to the new uploaded image
        img_path = "/Users/gregis/.gemini/antigravity/brain/cb30d7a8-63de-40c2-beb0-fd251c3d1f77/uploaded_image_1767898444956.png"
    
    debug_detection_simulation(img_path)
