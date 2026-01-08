"""
Application controller orchestrating tracking components.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
import time

from ..models import VideoProcessor, ObjectTracker, KalmanFilter, SpeedCalculator, ObjectDetector
from ..utils import MetricsCalculator


class AppController:
    """
    Main application controller managing the tracking pipeline.
    Coordinates video processing, object detection, tracking, and visualization.
    """
    
    def __init__(self, video_source: str | int, tracker_type: str = "CSRT",
                 use_kalman: bool = False, use_detector: bool = False,
                 detector_method: str = "yolo"):
        """
        Initialize application controller.
        
        Args:
            video_source: Video file path or camera index
            tracker_type: Type of tracker (CSRT, KCF, MOSSE, MEDIANFLOW)
            use_kalman: Whether to use Kalman filtering
            use_detector: Whether to use automatic detection
            detector_method: Detection method (yolo, hog)
        """
        self.video_source = video_source
        self.tracker_type = tracker_type
        self.use_kalman = use_kalman
        self.use_detector = use_detector
        
        # Initialize components
        self.video_processor = VideoProcessor(video_source)
        self.tracker = ObjectTracker(tracker_type)
        self.speed_calculator = None
        self.kalman_filters: List[KalmanFilter] = []
        self.detector = ObjectDetector(detector_method) if use_detector else None
        self.metrics = MetricsCalculator()
        
        # State
        self.is_running = False
        self.frame_count = 0
        self.tracking_initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize video processor and open video source.
        
        Returns:
            True if initialization successful
        """
        if not self.video_processor.open():
            print(f"Error: Could not open video source: {self.video_source}")
            return False
        
        # Initialize speed calculator with FPS
        self.speed_calculator = SpeedCalculator(fps=self.video_processor.fps)
        
        print(f"Video opened successfully")
        print(f"Resolution: {self.video_processor.frame_width}x{self.video_processor.frame_height}")
        print(f"FPS: {self.video_processor.fps}")
        
        return True
    
    def select_objects(self, num_objects: int = 1) -> bool:
        """
        Let user select objects to track in first frame.
        
        Args:
            num_objects: Number of objects to select
            
        Returns:
            True if objects selected successfully
        """
        # Read first frame
        success, frame = self.video_processor.read()
        if not success:
            print("Error: Could not read first frame")
            return False
        
        # Get ROIs from user
        print(f"\nSelect {num_objects} object(s) to track")
        print("Instructions:")
        print("  - Click and drag to select region")
        print("  - Press ENTER to confirm selection")
        print("  - Press ESC to skip")
        
        rois = self.video_processor.get_multiple_rois(frame, num_objects)
        
        if len(rois) == 0:
            print("No objects selected")
            return False
        
        # Initialize trackers
        success_list = self.tracker.init_multi(frame, rois)
        
        # Initialize Kalman filters if enabled
        if self.use_kalman:
            for roi in rois:
                kf = KalmanFilter()
                x, y, w, h = roi
                kf.init(x + w/2, y + h/2)  # Initialize at center
                self.kalman_filters.append(kf)
        
        # Initialize speed calculators for each object
        for _ in rois:
            self.speed_calculator.add_object()
        
        self.tracking_initialized = True
        print(f"\nInitialized tracking for {len(rois)} object(s)")
        
        return True
    
    def detect_and_track(self, target_classes: Optional[List[str]] = None) -> bool:
        """
        Automatically detect objects and initialize tracking.
        
        Args:
            target_classes: List of class names to detect
            
        Returns:
            True if objects detected and tracking initialized
        """
        if self.detector is None:
            print("Error: Detector not initialized")
            return False
        
        print("Detecting objects...")
        detections = []
        frame = None
        
        # Step 1: Try YOLO detection on multiple frames
        max_yolo_attempts = 10
        for attempt in range(max_yolo_attempts):
            success, frame = self.video_processor.read()
            if not success:
                print("Error reading video frame")
                return False
                
            detections = self.detector.detect(frame, target_classes)
            if len(detections) > 0:
                print(f"YOLO detected {len(detections)} object(s) on frame {attempt + 1}")
                break
        
        # Step 2: If YOLO failed, try motion detection as fallback
        if len(detections) == 0:
            print("YOLO detection failed. Trying motion detection fallback...")
            
            # Create background subtractor
            back_sub = cv2.createBackgroundSubtractorMOG2(
                history=100, 
                varThreshold=25, 
                detectShadows=False
            )
            
            # Read several frames to build background model
            frames_for_bg = []
            for _ in range(30):
                ret, tmp_frame = self.video_processor.read()
                if not ret:
                    break
                back_sub.apply(tmp_frame)
                frames_for_bg.append(tmp_frame)
            
            if len(frames_for_bg) > 0:
                # Use the last frame for detection
                frame = frames_for_bg[-1]
                fg_mask = back_sub.apply(frame)
                
                # Clean up the mask
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                
                # Find contours
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by size
                min_area = 500  # Minimum area for a vehicle
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > min_area:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Filter by aspect ratio (vehicles are usually wider than tall)
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.5 < aspect_ratio < 4.0:  # Reasonable aspect ratio for vehicles
                            detections.append((x, y, w, h, "vehicle", 0.8))
                
                # Sort by area and take top 5
                detections = sorted(detections, key=lambda d: d[2]*d[3], reverse=True)[:5]
                
                if len(detections) > 0:
                    print(f"Motion detection found {len(detections)} object(s)")

        if len(detections) == 0:
            print("No objects detected")
            return False
            
        print(f"Initializing tracking for {len(detections)} object(s)")
        
        # Extract bounding boxes
        rois = [(x, y, w, h) for x, y, w, h, _, _ in detections]
        
        # Initialize trackers
        self.tracker.init_multi(frame, rois)
        
        # Initialize Kalman filters if enabled
        if self.use_kalman:
            for roi in rois:
                kf = KalmanFilter()
                x, y, w, h = roi
                kf.init(x + w/2, y + h/2)
                self.kalman_filters.append(kf)
        
        # Initialize speed calculators
        for _ in rois:
            self.speed_calculator.add_object()
        
        self.tracking_initialized = True
        return True
    
    def run(self):
        """Run main tracking loop."""
        if not self.tracking_initialized:
            print("Error: Tracking not initialized. Call select_objects() or detect_and_track() first.")
            return
        
        self.is_running = True
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0
        
        print("\nStarting tracking...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while self.is_running:
            # Read frame
            success, frame = self.video_processor.read()
            if not success:
                print("\nEnd of video or error reading frame")
                break
            
            self.frame_count += 1
            fps_frame_count += 1
            
            # Update trackers
            success_list, bboxes = self.tracker.update(frame)
            
            # Get tracker info
            tracker_info = self.tracker.get_tracker_info()
            
            # Update Kalman filters and speed calculators
            for i, (bbox, active) in enumerate(zip(bboxes, success_list)):
                if not active:
                    continue
                
                x, y, w, h = bbox
                center_x = x + w/2
                center_y = y + h/2
                
                # Apply Kalman filtering
                if self.use_kalman and i < len(self.kalman_filters):
                    center_x, center_y = self.kalman_filters[i].update(center_x, center_y)
                    
                    # Update bbox with filtered position
                    bboxes[i] = (int(center_x - w/2), int(center_y - h/2), w, h)
                
                # Update speed
                speed = self.speed_calculator.update(i, (center_x, center_y))
                smoothed_speed = self.speed_calculator.get_smoothed_speed(i)
            
            # Visualize
            self._draw_tracking_info(frame, bboxes, success_list, current_fps)
            
            # Display frame
            self.video_processor.show("Vision Tracking", frame)
            
            # Calculate FPS
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Handle keyboard input
            key = self.video_processor.wait_key(1)
            if key == ord('q'):
                print("\nStopping tracking...")
                break
        self.cleanup()

    def process_frame(self) -> Tuple[Optional[np.ndarray], dict]:
        """
        Process a single frame for the video feed.
        
        Returns:
            Tuple containing:
            - Processed frame (RGB) or None if end/error
            - Dictionary with metrics (fps, object_count, speeds)
        """
        if not self.tracking_initialized:
            return None, {}
            
        success, frame = self.video_processor.read()
        if not success:
            return None, {}
            
        self.frame_count += 1
        
        # Update trackers
        success_list, bboxes = self.tracker.update(frame)
        
        # Update Kalman & speed
        object_speeds = {}
        for i, (bbox, active) in enumerate(zip(bboxes, success_list)):
            if not active:
                continue
            
            x, y, w, h = bbox
            center_x, center_y = x + w/2, y + h/2
            
            # Application of Kalman
            if self.use_kalman and i < len(self.kalman_filters):
                center_x, center_y = self.kalman_filters[i].update(center_x, center_y)
                bboxes[i] = (int(center_x - w/2), int(center_y - h/2), w, h)
                
            # Speed
            speed = self.speed_calculator.update(i, (center_x, center_y))
            object_speeds[f"Object {i}"] = self.speed_calculator.get_speed_kmh(i)

        # Draw info (calculate pseudo-FPS or pass 0)
        self._draw_tracking_info(frame, bboxes, success_list, fps=0.0)
        
        # Convert to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        metrics = {
            "object_count": self.tracker.count_active(),
            "speeds": object_speeds
        }
        
        return frame_rgb, metrics
    
    def _draw_tracking_info(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]],
                           success_list: List[bool], fps: float):
        """
        Draw tracking information on frame with speed in km/h.
        
        Args:
            frame: Frame to draw on
            bboxes: List of bounding boxes
            success_list: List of tracking success status
            fps: Current FPS
        """
        # Use bright green color for all tracked objects
        bright_green = (0, 255, 0)
        
        for i, (bbox, success) in enumerate(zip(bboxes, success_list)):
            if not success:
                continue
            
            x, y, w, h = bbox
            
            # Draw bounding box with bright green
            cv2.rectangle(frame, (x, y), (x + w, y + h), bright_green, 2)
            
            # Get speed in km/h
            speed_kmh = self.speed_calculator.get_speed_kmh(i)
            
            # Create speed label
            speed_label = f"{speed_kmh:.1f} km/h"
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(speed_label, font, font_scale, thickness)
            
            # Position label at top-left of bounding box
            label_x = x
            label_y = y - 10 if y > 30 else y + h + 20
            
            # Draw semi-transparent background for text
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (label_x - 2, label_y - text_h - 5),
                         (label_x + text_w + 4, label_y + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw speed text
            cv2.putText(frame, speed_label, (label_x, label_y),
                       font, font_scale, bright_green, thickness)
        
        # Draw general info at top
        info_text = f"Tracker: {self.tracker_type} | FPS: {fps:.1f} | Objects: {self.tracker.count_active()}"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def cleanup(self):
        """Clean up resources."""
        self.video_processor.release()
        cv2.destroyAllWindows()
        self.is_running = False
        
        # Print summary
        print("\n=== Tracking Summary ===")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Tracker type: {self.tracker_type}")
        print(f"Objects tracked: {len(self.tracker.trackers)}")
        
        # Print speed statistics for each object
        for i in range(len(self.tracker.trackers)):
            avg_speed = self.speed_calculator.get_average_speed(i)
            print(f"  Object {i}: Avg speed = {avg_speed:.2f} px/s")
