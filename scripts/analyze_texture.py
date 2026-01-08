import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.detector import ObjectDetector

def analyze_texture(image_path):
    print(f"Analyzing texture for: {image_path}")
    frame = cv2.imread(image_path)
    
    # Use Tiny Model since we know it sees the cat (albeit 4 times)
    detector = ObjectDetector(method="yolo", model_type="tiny", confidence_threshold=0.1, nms_threshold=0.3)
    
    # Debug: Print all
    all_dets = detector.detect(frame, target_classes=None)
    print(f"DEBUG: Found {len(all_dets)} total objects.")
    for d in all_dets:
        print(f" - {d[4]} ({d[5]:.2f})")

    detections = detector.detect(frame, target_classes=["cat"])
    
    if not detections:
        print("No cat found to analyze.")
        return

    for i, det in enumerate(detections):
        x, y, w, h, cls, conf = det
        
        # Crop the cat
        # Ensure bounds
        y_start = max(0, y)
        y_end = min(frame.shape[0], y+h)
        x_start = max(0, x)
        x_end = min(frame.shape[1], x+w)
        
        cat_roi = frame[y_start:y_end, x_start:x_end]
        
        if cat_roi.size == 0:
            continue
            
        # Convert to gray
        gray = cv2.cvtColor(cat_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Variance of Laplacian (Sharpness/Texture)
        # Fur should have high variance (many edges). Skin/Smooth should have low.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Edge density (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / (w * h)
        
        print(f"\nCat #{i+1} Texture Analysis:")
        print(f" - Texture Score (Laplacian Var): {laplacian_var:.2f}")
        print(f" - Edge Density: {edge_density:.4f}")
        
        if laplacian_var > 1000: # Heuristic guess
            print(" -> Result: Likely Normal Cat (High Texture)")
        else:
            print(" -> Result: Possible Hairless/Smooth Cat (Low Texture)")

if __name__ == "__main__":
    img_path = "/Users/gregis/.gemini/antigravity/brain/cb30d7a8-63de-40c2-beb0-fd251c3d1f77/uploaded_image_1767898756474.png"
    analyze_texture(img_path)
