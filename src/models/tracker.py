"""
Object tracker module implementing multiple tracking algorithms.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from enum import Enum


class TrackerType(Enum):
    """Available tracker types."""
    CSRT = "CSRT"
    KCF = "KCF"
    MOSSE = "MOSSE"
    MEDIANFLOW = "MEDIANFLOW"
    MIL = "MIL"
    BOOSTING = "BOOSTING"


class ObjectTracker:
    """
    Object tracker supporting multiple tracking algorithms.
    Can track single or multiple objects simultaneously.
    """
    
    def __init__(self, tracker_type: str = "CSRT"):
        """
        Initialize object tracker.
        
        Args:
            tracker_type: Type of tracker to use (CSRT, KCF, MOSSE, MEDIANFLOW, MIL, BOOSTING)
        """
        self.tracker_type = TrackerType[tracker_type.upper()]
        self.trackers: List[cv2.Tracker] = []
        self.bboxes: List[Tuple[int, int, int, int]] = []
        self.active: List[bool] = []
        self.colors: List[Tuple[int, int, int]] = []
        
    def _create_tracker(self) -> cv2.Tracker:
        """
        Create a new tracker instance based on tracker type.
        
        Returns:
            OpenCV tracker object
        """
        if self.tracker_type == TrackerType.CSRT:
            return cv2.legacy.TrackerCSRT_create()
        elif self.tracker_type == TrackerType.KCF:
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == TrackerType.MOSSE:
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == TrackerType.MEDIANFLOW:
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracker_type == TrackerType.MIL:
            return cv2.legacy.TrackerMIL_create()
        elif self.tracker_type == TrackerType.BOOSTING:
            return cv2.legacy.TrackerBoosting_create()
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")
    
    def _generate_color(self) -> Tuple[int, int, int]:
        """
        Generate a random color for tracking visualization.
        
        Returns:
            RGB color tuple
        """
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize tracker with first frame and bounding box.
        
        Args:
            frame: First frame
            bbox: Initial bounding box as (x, y, width, height)
            
        Returns:
            True if initialization successful
        """
        tracker = self._create_tracker()
        success = tracker.init(frame, bbox)
        
        if success:
            self.trackers.append(tracker)
            self.bboxes.append(bbox)
            self.active.append(True)
            self.colors.append(self._generate_color())
        
        return success
    
    def init_multi(self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]) -> List[bool]:
        """
        Initialize multiple trackers with first frame and bounding boxes.
        
        Args:
            frame: First frame
            bboxes: List of initial bounding boxes
            
        Returns:
            List of success status for each tracker
        """
        results = []
        for bbox in bboxes:
            success = self.init(frame, bbox)
            results.append(success)
        return results
    
    def update(self, frame: np.ndarray) -> Tuple[List[bool], List[Tuple[int, int, int, int]]]:
        """
        Update all active trackers with new frame.
        
        Args:
            frame: New frame to track in
            
        Returns:
            Tuple of (success_list, bbox_list)
        """
        success_list = []
        bbox_list = []
        
        for i, tracker in enumerate(self.trackers):
            if not self.active[i]:
                success_list.append(False)
                bbox_list.append((0, 0, 0, 0))
                continue
            
            success, bbox = tracker.update(frame)
            
            if success:
                self.bboxes[i] = tuple(int(v) for v in bbox)
                success_list.append(True)
                bbox_list.append(self.bboxes[i])
            else:
                self.active[i] = False
                success_list.append(False)
                bbox_list.append((0, 0, 0, 0))
        
        return success_list, bbox_list
    
    def get_active_bboxes(self) -> List[Tuple[int, int, int, int]]:
        """
        Get bounding boxes of all active trackers.
        
        Returns:
            List of active bounding boxes
        """
        return [bbox for bbox, active in zip(self.bboxes, self.active) if active]
    
    def get_active_colors(self) -> List[Tuple[int, int, int]]:
        """
        Get colors of all active trackers.
        
        Returns:
            List of active colors
        """
        return [color for color, active in zip(self.colors, self.active) if active]
    
    def get_tracker_info(self) -> List[Dict]:
        """
        Get information about all trackers.
        
        Returns:
            List of tracker info dictionaries
        """
        info = []
        for i, (bbox, active, color) in enumerate(zip(self.bboxes, self.active, self.colors)):
            info.append({
                'id': i,
                'bbox': bbox,
                'active': active,
                'color': color,
                'center': self._get_bbox_center(bbox)
            })
        return info
    
    def _get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Calculate center point of bounding box.
        
        Args:
            bbox: Bounding box as (x, y, width, height)
            
        Returns:
            Center point as (cx, cy)
        """
        x, y, w, h = bbox
        return (int(x + w/2), int(y + h/2))
    
    def reset(self):
        """Reset all trackers."""
        self.trackers.clear()
        self.bboxes.clear()
        self.active.clear()
        self.colors.clear()
    
    def count_active(self) -> int:
        """
        Count number of active trackers.
        
        Returns:
            Number of active trackers
        """
        return sum(self.active)
    
    @staticmethod
    def get_available_trackers() -> List[str]:
        """
        Get list of available tracker types.
        
        Returns:
            List of tracker type names
        """
        return [t.value for t in TrackerType]
