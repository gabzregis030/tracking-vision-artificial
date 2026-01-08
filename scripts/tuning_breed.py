import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.detector import ObjectDetector

def tune_breed_threshold(image_path):
    print(f"Analyzing texture for Tuning: {image_path}")
    frame = cv2.imread(image_path)
    
    # Use Standard Model to ensure we find them
    detector = ObjectDetector(method="yolo", model_type="standard", confidence_threshold=0.1, nms_threshold=0.3)
    detections = detector.detect(frame, target_classes=["cat"])
    
    print(f"Found {len(detections)} cats.")

    for i, det in enumerate(detections):
        x, y, w, h, cls, conf = det
        
        # Crop
        y_start = max(0, y)
        y_end = min(frame.shape[0], y+h)
        x_start = max(0, x)
        x_end = min(frame.shape[1], x+w)
        
        roi = frame[y_start:y_end, x_start:x_end]
        
        if roi.size == 0:
            continue
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Metric 1: Laplacian Variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Metric 2: Sobel (Alternative)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.mean(np.sqrt(sobelx**2 + sobely**2))
        
        print(f"\nCat #{i+1} Analysis:")
        print(f" - Laplacian Var (Texture Score): {laplacian_var:.2f}")
        print(f" - Sobel Magnitude (Edge Strength): {sobel_mag:.2f}")
        
        if laplacian_var < 2000:
            print(" -> Current Logic: SPHYNX")
        else:
            print(" -> Current Logic: GATO (Normal)")

if __name__ == "__main__":
    img_path = "/Users/gregis/.gemini/antigravity/brain/cb30d7a8-63de-40c2-beb0-fd251c3d1f77/uploaded_image_1767902453922.png"
    tune_breed_threshold(img_path)
