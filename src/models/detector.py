"""
Object detector module using YOLO and other detection methods.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import os


class ObjectDetector:
    """
    Object detector supporting multiple detection methods.
    Primary support for YOLO, with fallback to classical methods.
    """
    
    def __init__(self, method: str = "yolo", confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize object detector.
        
        Args:
            method: Detection method ('yolo', 'hog', 'cascade')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
        """
        self.method = method.lower()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.net = None
        self.classes = []
        self.output_layers = []
        
        if self.method == "yolo":
            self._load_yolo()
        elif self.method == "hog":
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def _load_yolo(self):
        """Load YOLO model weights and configuration."""
        # Note: Users need to download YOLO weights separately
        # This is a placeholder that checks for local weights
        weights_path = "models/yolo/yolov3.weights"
        config_path = "models/yolo/yolov3.cfg"
        names_path = "models/yolo/coco.names"
        
        if os.path.exists(weights_path) and os.path.exists(config_path):
            self.net = cv2.dnn.readNet(weights_path, config_path)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Load class names
            if os.path.exists(names_path):
                with open(names_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
        else:
            print(f"Warning: YOLO weights not found at {weights_path}")
            print("YOLO detection will not be available.")
            print("Download weights from: https://pjreddie.com/darknet/yolo/")
    
    def detect_yolo(self, frame: np.ndarray, 
                    target_classes: Optional[List[str]] = None) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Detect objects using YOLO.
        
        Args:
            frame: Input frame
            target_classes: List of class names to detect (None = detect all)
            
        Returns:
            List of detections as (x, y, w, h, class_name, confidence)
        """
        if self.net is None:
            return []
        
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass
        outputs = self.net.forward(self.output_layers)
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Filter by target classes if specified
                    class_name = self.classes[class_id] if class_id < len(self.classes) else "unknown"
                    if target_classes is not None and class_name not in target_classes:
                        continue
                    
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                class_name = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                confidence = confidences[i]
                detections.append((x, y, w, h, class_name, confidence))
        
        return detections
    
    def detect_hog(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Detect people using HOG descriptor.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detections as (x, y, w, h, class_name, confidence)
        """
        # Detect people
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            if weight > self.confidence_threshold:
                detections.append((int(x), int(y), int(w), int(h), "person", float(weight)))
        
        return detections
    
    def detect(self, frame: np.ndarray, 
               target_classes: Optional[List[str]] = None) -> List[Tuple[int, int, int, int, str, float]]:
        """
        Detect objects using the configured method.
        
        Args:
            frame: Input frame
            target_classes: List of class names to detect (only for YOLO)
            
        Returns:
            List of detections as (x, y, w, h, class_name, confidence)
        """
        if self.method == "yolo":
            return self.detect_yolo(frame, target_classes)
        elif self.method == "hog":
            return self.detect_hog(frame)
        else:
            return []
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Tuple[int, int, int, int, str, float]]) -> np.ndarray:
        """
        Draw detection boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            x, y, w, h, class_name, confidence = detection
            
            # Draw rectangle
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """
        Get list of available detection methods.
        
        Returns:
            List of method names
        """
        return ["yolo", "hog"]
