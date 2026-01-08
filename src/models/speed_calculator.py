"""
Speed calculator for tracked objects.
"""

import numpy as np
from typing import Tuple, Optional, List
from collections import deque


class SpeedCalculator:
    """
    Calculate speed and velocity of tracked objects.
    Supports both pixel speed and real-world speed (if calibrated).
    """
    
    def __init__(self, fps: int = 30, pixels_per_meter: Optional[float] = None,
                 smoothing_window: int = 5):
        """
        Initialize speed calculator.
        
        Args:
            fps: Frames per second of video
            pixels_per_meter: Calibration factor for real-world measurements (pixels/meter)
            smoothing_window: Number of frames to use for smoothing speed calculations
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.smoothing_window = smoothing_window
        
        # History of positions for each tracked object
        self.position_history: List[deque] = []
        self.speed_history: List[deque] = []
    
    def add_object(self):
        """Add a new object to track."""
        self.position_history.append(deque(maxlen=self.smoothing_window))
        self.speed_history.append(deque(maxlen=self.smoothing_window))
    
    def update(self, object_id: int, position: Tuple[float, float]) -> float:
        """
        Update position and calculate speed for an object.
        
        Args:
            object_id: ID of the object
            position: Current (x, y) position in pixels
            
        Returns:
            Speed in pixels/second (or m/s if calibrated)
        """
        # Ensure we have tracking for this object
        while len(self.position_history) <= object_id:
            self.add_object()
        
        self.position_history[object_id].append(position)
        
        # Need at least 2 positions to calculate speed
        if len(self.position_history[object_id]) < 2:
            return 0.0
        
        # Calculate displacement from previous position
        prev_pos = self.position_history[object_id][-2]
        curr_pos = self.position_history[object_id][-1]
        
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        
        # Calculate distance (pixels)
        distance = np.sqrt(dx**2 + dy**2)
        
        # Calculate speed (pixels per second)
        speed_px = distance * self.fps
        
        # Convert to real-world units if calibrated
        if self.pixels_per_meter is not None:
            speed = speed_px / self.pixels_per_meter  # m/s
        else:
            speed = speed_px
        
        self.speed_history[object_id].append(speed)
        
        return speed
    
    def get_smoothed_speed(self, object_id: int) -> float:
        """
        Get smoothed speed for an object using moving average.
        
        Args:
            object_id: ID of the object
            
        Returns:
            Smoothed speed
        """
        if object_id >= len(self.speed_history) or len(self.speed_history[object_id]) == 0:
            return 0.0
        
        return np.mean(list(self.speed_history[object_id]))
    
    def get_velocity(self, object_id: int) -> Tuple[float, float]:
        """
        Get velocity vector for an object.
        
        Args:
            object_id: ID of the object
            
        Returns:
            Velocity as (vx, vy) in pixels/second (or m/s if calibrated)
        """
        if object_id >= len(self.position_history) or len(self.position_history[object_id]) < 2:
            return (0.0, 0.0)
        
        prev_pos = self.position_history[object_id][-2]
        curr_pos = self.position_history[object_id][-1]
        
        vx = (curr_pos[0] - prev_pos[0]) * self.fps
        vy = (curr_pos[1] - prev_pos[1]) * self.fps
        
        # Convert to real-world units if calibrated
        if self.pixels_per_meter is not None:
            vx /= self.pixels_per_meter
            vy /= self.pixels_per_meter
        
        return (vx, vy)
    
    def get_average_speed(self, object_id: int) -> float:
        """
        Get average speed over entire tracking history.
        
        Args:
            object_id: ID of the object
            
        Returns:
            Average speed
        """
        if object_id >= len(self.position_history) or len(self.position_history[object_id]) < 2:
            return 0.0
        
        positions = list(self.position_history[object_id])
        total_distance = 0.0
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Total time in seconds
        total_time = (len(positions) - 1) / self.fps
        
        if total_time == 0:
            return 0.0
        
        avg_speed_px = total_distance / total_time
        
        # Convert to real-world units if calibrated
        if self.pixels_per_meter is not None:
            return avg_speed_px / self.pixels_per_meter
        
        return avg_speed_px
    
    def calibrate(self, pixel_distance: float, real_distance: float):
        """
        Calibrate the speed calculator using a known distance.
        
        Args:
            pixel_distance: Distance in pixels
            real_distance: Corresponding real-world distance in meters
        """
        self.pixels_per_meter = pixel_distance / real_distance
    
    def convert_to_kmh(self, speed_ms: float) -> float:
        """
        Convert speed from m/s to km/h.
        
        Args:
            speed_ms: Speed in m/s
            
        Returns:
            Speed in km/h
        """
        return speed_ms * 3.6
    
    def convert_to_mph(self, speed_ms: float) -> float:
        """
        Convert speed from m/s to mph.
        
        Args:
            speed_ms: Speed in m/s
            
        Returns:
            Speed in mph
        """
        return speed_ms * 2.23694
    
    def reset(self):
        """Reset all tracking history."""
        self.position_history.clear()
        self.speed_history.clear()
    
    def reset_object(self, object_id: int):
        """
        Reset tracking history for a specific object.
        
        Args:
            object_id: ID of the object
        """
        if object_id < len(self.position_history):
            self.position_history[object_id].clear()
            self.speed_history[object_id].clear()
