"""
Video processor module for handling video stream and frame operations.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class VideoProcessor:
    """Handles video capture, frame processing, and display operations."""
    
    def __init__(self, source: str | int = 0, fps: Optional[int] = None):
        """
        Initialize video processor.
        
        Args:
            source: Video file path or camera index (0 for default webcam)
            fps: Target FPS for processing (None = use source FPS)
        """
        self.source = source
        self.cap = None
        self.fps = fps
        self.frame_width = 0
        self.frame_height = 0
        self.is_camera = isinstance(source, int)
        
    def open(self) -> bool:
        """
        Open video source.
        
        Returns:
            True if successfully opened, False otherwise
        """
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
            
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if self.fps is None:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if self.fps == 0:  # Some cameras return 0
                self.fps = 30
                
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video source.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
            
        success, frame = self.cap.read()
        return success, frame
    
    def release(self):
        """Release video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_roi(self, frame: np.ndarray, window_name: str = "Select ROI") -> Optional[Tuple[int, int, int, int]]:
        """
        Let user select a Region of Interest (ROI) in the frame.
        
        Args:
            frame: Frame to select ROI from
            window_name: Name of the selection window
            
        Returns:
            ROI as (x, y, width, height) or None if cancelled
        """
        roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        
        # Check if selection was made
        if roi[2] == 0 or roi[3] == 0:
            return None
            
        return roi
    
    def get_multiple_rois(self, frame: np.ndarray, num_objects: int = 1) -> list:
        """
        Let user select multiple ROIs in the frame.
        
        Args:
            frame: Frame to select ROIs from
            num_objects: Number of objects to select
            
        Returns:
            List of ROIs, each as (x, y, width, height)
        """
        rois = []
        temp_frame = frame.copy()
        
        for i in range(num_objects):
            print(f"Select object {i+1}/{num_objects}. Press ENTER when done, ESC to skip.")
            roi = cv2.selectROI(f"Select Object {i+1}", temp_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(f"Select Object {i+1}")
            
            # Check if selection was cancelled
            if roi[2] == 0 or roi[3] == 0:
                print(f"Skipped object {i+1}")
                continue
                
            rois.append(roi)
            
            # Draw rectangle on frame for reference
            x, y, w, h = roi
            cv2.rectangle(temp_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return rois
    
    def draw_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                 label: str = "", color: Tuple[int, int, int] = (0, 255, 0), 
                 thickness: int = 2) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box as (x, y, width, height)
            label: Optional label text
            color: Box color as (B, G, R)
            thickness: Line thickness
            
        Returns:
            Frame with drawn box
        """
        x, y, w, h = [int(v) for v in bbox]
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label if provided
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                label,
                (x, y - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
        
        return frame
    
    def draw_info(self, frame: np.ndarray, info_text: str, 
                  position: Tuple[int, int] = (10, 30),
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw information text on frame.
        
        Args:
            frame: Frame to draw on
            info_text: Text to display
            position: Text position (x, y)
            color: Text color as (B, G, R)
            
        Returns:
            Frame with drawn text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, info_text, position, font, 0.7, color, 2)
        return frame
    
    def show(self, window_name: str, frame: np.ndarray):
        """
        Display frame in window.
        
        Args:
            window_name: Name of the window
            frame: Frame to display
        """
        cv2.imshow(window_name, frame)
    
    def wait_key(self, delay: int = 1) -> int:
        """
        Wait for key press.
        
        Args:
            delay: Delay in milliseconds
            
        Returns:
            ASCII value of pressed key or -1
        """
        return cv2.waitKey(delay) & 0xFF
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        cv2.destroyAllWindows()
