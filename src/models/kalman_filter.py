"""
Kalman filter implementation for smooth object tracking.
"""

import numpy as np
from typing import Tuple, Optional


class KalmanFilter:
    """
    Kalman filter for tracking object position and velocity.
    Helps smooth out noisy measurements and predict future positions.
    """
    
    def __init__(self, process_noise: float = 1e-5, measurement_noise: float = 1e-4):
        """
        Initialize Kalman filter.
        
        Args:
            process_noise: Process noise covariance (how much we trust the model)
            measurement_noise: Measurement noise covariance (how much we trust measurements)
        """
        self.dt = 1.0  # Time step
        
        # State: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        # State covariance matrix
        self.P = np.eye(4) * 1000
        
        self.initialized = False
    
    def init(self, x: float, y: float):
        """
        Initialize filter with first measurement.
        
        Args:
            x: Initial x position
            y: Initial y position
        """
        self.state = np.array([x, y, 0, 0])
        self.P = np.eye(4) * 1000
        self.initialized = True
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next state.
        
        Returns:
            Predicted (x, y) position
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[0], self.state[1]
    
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update filter with new measurement.
        
        Args:
            x: Measured x position
            y: Measured y position
            
        Returns:
            Corrected (x, y) position
        """
        if not self.initialized:
            self.init(x, y)
            return x, y
        
        # Measurement
        z = np.array([x, y])
        
        # Innovation (measurement residual)
        y_innov = z - (self.H @ self.state)
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y_innov
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[0], self.state[1]
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current position estimate.
        
        Returns:
            Current (x, y) position
        """
        return self.state[0], self.state[1]
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.
        
        Returns:
            Current (vx, vy) velocity
        """
        return self.state[2], self.state[3]
    
    def reset(self):
        """Reset filter to uninitialized state."""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        self.initialized = False


class MultiKalmanFilter:
    """
    Manages multiple Kalman filters for multi-object tracking.
    """
    
    def __init__(self, num_objects: int, process_noise: float = 1e-5, 
                 measurement_noise: float = 1e-4):
        """
        Initialize multiple Kalman filters.
        
        Args:
            num_objects: Number of objects to track
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.filters = [
            KalmanFilter(process_noise, measurement_noise) 
            for _ in range(num_objects)
        ]
    
    def predict_all(self) -> list:
        """
        Predict positions for all objects.
        
        Returns:
            List of predicted (x, y) positions
        """
        return [f.predict() for f in self.filters]
    
    def update_all(self, measurements: list) -> list:
        """
        Update all filters with measurements.
        
        Args:
            measurements: List of (x, y) measurements
            
        Returns:
            List of corrected (x, y) positions
        """
        results = []
        for i, (x, y) in enumerate(measurements):
            if i < len(self.filters):
                corrected = self.filters[i].update(x, y)
                results.append(corrected)
        return results
    
    def get_positions(self) -> list:
        """
        Get current positions for all objects.
        
        Returns:
            List of (x, y) positions
        """
        return [f.get_position() for f in self.filters]
    
    def get_velocities(self) -> list:
        """
        Get current velocities for all objects.
        
        Returns:
            List of (vx, vy) velocities
        """
        return [f.get_velocity() for f in self.filters]
    
    def reset_all(self):
        """Reset all filters."""
        for f in self.filters:
            f.reset()
