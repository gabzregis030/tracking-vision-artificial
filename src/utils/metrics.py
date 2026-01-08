"""
Metrics calculator for tracking performance evaluation.
"""

import numpy as np
from typing import List, Tuple, Dict
import time


class MetricsCalculator:
    """Calculate and track performance metrics for object tracking."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.tracking_history: List[Dict] = []
        self.frame_times: List[float] = []
        self.success_count = 0
        self.failure_count = 0
        self.start_time = None
        
    def start_tracking(self):
        """Start timing for tracking session."""
        self.start_time = time.time()
    
    def record_frame(self, processing_time: float, num_tracked: int, num_lost: int):
        """
        Record metrics for a frame.
        
        Args:
            processing_time: Time taken to process frame (seconds)
            num_tracked: Number of successfully tracked objects
            num_lost: Number of lost tracks
        """
        self.frame_times.append(processing_time)
        self.success_count += num_tracked
        self.failure_count += num_lost
        
        self.tracking_history.append({
            'time': time.time() - self.start_time if self.start_time else 0,
            'tracked': num_tracked,
            'lost': num_lost,
            'processing_time': processing_time
        })
    
    def calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box as (x, y, width, height)
            bbox2: Second bounding box as (x, y, width, height)
            
        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def calculate_center_error(self, bbox1: Tuple[int, int, int, int],
                              bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate center distance between two bounding boxes.
        
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Euclidean distance between centers
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1/2, y1 + h1/2)
        center2 = (x2 + w2/2, y2 + h2/2)
        
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        return distance
    
    def get_average_fps(self) -> float:
        """
        Calculate average FPS from recorded frame times.
        
        Returns:
            Average FPS
        """
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_success_rate(self) -> float:
        """
        Calculate tracking success rate.
        
        Returns:
            Success rate as percentage
        """
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        
        return (self.success_count / total) * 100
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive tracking statistics.
        
        Returns:
            Dictionary of statistics
        """
        if len(self.frame_times) == 0:
            return {
                'avg_fps': 0.0,
                'min_fps': 0.0,
                'max_fps': 0.0,
                'success_rate': 0.0,
                'total_frames': 0
            }
        
        frame_times = np.array(self.frame_times)
        fps_values = 1.0 / frame_times[frame_times > 0]
        
        return {
            'avg_fps': np.mean(fps_values) if len(fps_values) > 0 else 0.0,
            'min_fps': np.min(fps_values) if len(fps_values) > 0 else 0.0,
            'max_fps': np.max(fps_values) if len(fps_values) > 0 else 0.0,
            'std_fps': np.std(fps_values) if len(fps_values) > 0 else 0.0,
            'success_rate': self.get_success_rate(),
            'total_frames': len(self.frame_times),
            'avg_processing_time': np.mean(frame_times),
            'total_tracked': self.success_count,
            'total_lost': self.failure_count
        }
    
    def print_summary(self):
        """Print tracking performance summary."""
        stats = self.get_statistics()
        
        print("\n=== Tracking Performance Metrics ===")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"Average FPS: {stats['avg_fps']:.2f}")
        print(f"Min FPS: {stats['min_fps']:.2f}")
        print(f"Max FPS: {stats['max_fps']:.2f}")
        print(f"FPS Std Dev: {stats['std_fps']:.2f}")
        print(f"Avg Processing Time: {stats['avg_processing_time']*1000:.2f} ms")
        print(f"Success Rate: {stats['success_rate']:.2f}%")
        print(f"Total Tracked: {stats['total_tracked']}")
        print(f"Total Lost: {stats['total_lost']}")
    
    def reset(self):
        """Reset all metrics."""
        self.tracking_history.clear()
        self.frame_times.clear()
        self.success_count = 0
        self.failure_count = 0
        self.start_time = None
