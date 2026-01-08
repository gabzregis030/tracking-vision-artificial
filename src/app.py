"""
Sistema de Tracking de Vehículos - Interfaz Streamlit

Esta versión tiene detección robusta y visualización mejorada.
"""

import streamlit as st
import cv2
import tempfile
import numpy as np
import time
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

st.set_page_config(
    page_title="Sistema de Tracking de Gatos",
    page_icon="🐈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 10px;
    }
    .main-header {
        text-align: center;
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)


class SimpleVehicleTracker:
    """Simple vehicle tracker using background subtraction and contour detection."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.back_sub = None
        self.tracked_objects = {}
        self.next_id = 0
        self.fps = 30
        self.total_frames = 0
        self.frame_count = 0
        
        # Speed calculation
        self.pixels_per_meter = 100.0  # Calibration factor for cats
        self.speed_history = {}
        
    def initialize(self):
        """Open video and initialize background subtractor."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Background subtractor with tuned parameters
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=200,
            varThreshold=40,
            detectShadows=False
        )
        
        # Warm up background model
        for _ in range(min(30, self.total_frames // 4)):
            ret, frame = self.cap.read()
            if ret:
                self.back_sub.apply(frame)
        
        return True
    
    def detect_vehicles(self, frame):
        """Detect moving vehicles in frame."""
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to connect nearby regions
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        min_area = 500  # Minimum area for a cat
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Filter by aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                if 0.3 < aspect_ratio < 5.0:
                    center_x = x + w // 2
                    center_y = y + h // 2
                    detections.append({
                        'bbox': (x, y, w, h),
                        'center': (center_x, center_y),
                        'area': area
                    })
        
        # Sort by area, keep top 10
        detections = sorted(detections, key=lambda d: d['area'], reverse=True)[:10]
        return detections
    
    def update_tracking(self, detections):
        """Simple tracking by matching closest objects."""
        new_tracked = {}
        used_detections = set()
        
        # Match existing tracked objects to detections
        for obj_id, obj_data in self.tracked_objects.items():
            best_match = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in used_detections:
                    continue
                    
                dist = np.sqrt(
                    (obj_data['center'][0] - det['center'][0])**2 +
                    (obj_data['center'][1] - det['center'][1])**2
                )
                
                if dist < 100 and dist < best_distance:  # Max distance threshold
                    best_distance = dist
                    best_match = i
            
            if best_match is not None:
                used_detections.add(best_match)
                det = detections[best_match]
                
                # Calculate speed
                dx = det['center'][0] - obj_data['center'][0]
                dy = det['center'][1] - obj_data['center'][1]
                pixel_speed = np.sqrt(dx**2 + dy**2) * self.fps
                speed_ms = pixel_speed / self.pixels_per_meter
                speed_kmh = speed_ms * 3.6
                
                # Smooth speed
                if obj_id not in self.speed_history:
                    self.speed_history[obj_id] = []
                self.speed_history[obj_id].append(speed_kmh)
                if len(self.speed_history[obj_id]) > 10:
                    self.speed_history[obj_id].pop(0)
                
                avg_speed = np.mean(self.speed_history[obj_id])
                
                new_tracked[obj_id] = {
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'speed': avg_speed,
                    'frames_tracked': obj_data['frames_tracked'] + 1
                }
        
        # Add new detections as new objects
        for i, det in enumerate(detections):
            if i not in used_detections:
                new_tracked[self.next_id] = {
                    'bbox': det['bbox'],
                    'center': det['center'],
                    'speed': 0.0,
                    'frames_tracked': 1
                }
                self.next_id += 1
        
        self.tracked_objects = new_tracked
    
    def draw_tracking(self, frame):
        """Draw tracking visualization on frame."""
        for obj_id, obj_data in self.tracked_objects.items():
            # Only show objects tracked for more than a few frames
            if obj_data['frames_tracked'] < 3:
                continue
                
            x, y, w, h = obj_data['bbox']
            speed = obj_data['speed']
            
            # Draw bounding box (bright green)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Speed label
            speed_text = f"{speed:.1f} km/h"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
            
            # Position above bbox
            text_x = x
            text_y = y - 8 if y > 25 else y + h + 20
            
            # Draw background rectangle
            cv2.rectangle(frame, 
                         (text_x - 2, text_y - text_h - 4),
                         (text_x + text_w + 4, text_y + 4),
                         (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, speed_text, (text_x, text_y),
                       font, font_scale, (0, 255, 0), thickness)
        
        return frame
    
    def process_frame(self):
        """Process one frame and return annotated result."""
        if self.cap is None:
            return None, {}
            
        ret, frame = self.cap.read()
        if not ret:
            return None, {}
        
        self.frame_count += 1
        
        # Detect and track
        detections = self.detect_vehicles(frame)
        self.update_tracking(detections)
        
        # Draw visualization
        frame = self.draw_tracking(frame)
        
        # Add info overlay
        active_count = len([o for o in self.tracked_objects.values() if o['frames_tracked'] >= 3])
        info_text = f"Frame: {self.frame_count}/{self.total_frames} | Cats: {active_count}"
        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Metrics
        speeds = {f"Cat {i}": obj['speed'] 
                  for i, (obj_id, obj) in enumerate(self.tracked_objects.items()) 
                  if obj['frames_tracked'] >= 3}
        
        metrics = {
            'object_count': active_count,
            'frame': self.frame_count,
            'total_frames': self.total_frames,
            'speeds': speeds
        }
        
        return frame_rgb, metrics
    
    def release(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error al guardar archivo: {e}")
        return None


def main():
    st.title("🐈 Sistema de Tracking de Gatos")
    st.markdown("Sube un video para detectar y rastrear gatos con su velocidad estimada")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    
    uploaded_file = st.sidebar.file_uploader(
        "📹 Cargar Video", 
        type=['mp4', 'avi', 'mov', 'mpeg4'],
        help="Sube un video de tráfico para analizar"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Estadísticas")
    
    # Placeholders for stats
    stats_frame = st.sidebar.empty()
    stats_vehicles = st.sidebar.empty()
    
    # Initialize session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("▶️ Iniciar", type="primary", use_container_width=True)
    stop_btn = col2.button("⏹️ Detener", use_container_width=True)
    
    # Main display area
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.empty()
    
    if start_btn and uploaded_file:
        video_path = save_uploaded_file(uploaded_file)
        if video_path:
            tracker = SimpleVehicleTracker(video_path)
            if tracker.initialize():
                st.session_state.tracker = tracker
                st.session_state.is_running = True
                st.success("Video cargado. Iniciando tracking...")
            else:
                st.error(" Error al abrir el video")
    
    if stop_btn:
        st.session_state.is_running = False
        if st.session_state.tracker:
            st.session_state.tracker.release()
            st.session_state.tracker = None
    
    # Processing loop
    if st.session_state.is_running and st.session_state.tracker:
        tracker = st.session_state.tracker
        
        while st.session_state.is_running:
            frame, metrics = tracker.process_frame()
            
            if frame is None:
                st.info(" Video terminado")
                st.session_state.is_running = False
                break
            
            # Display frame
            video_placeholder.image(frame, channels="RGB", use_container_width=True)
            
            # Update stats in sidebar
            stats_frame.metric("Frame Actual", f"{metrics.get('frame', 0)}/{metrics.get('total_frames', 0)}")
            stats_vehicles.metric("Gatos Detectados", metrics.get('object_count', 0))            
            # Progress bar
            if metrics.get('total_frames', 0) > 0:
                progress = metrics.get('frame', 0) / metrics.get('total_frames', 1)
                progress_bar.progress(progress)
            
            # Display speeds
            with metrics_placeholder.container():
                speeds = metrics.get('speeds', {})
                if speeds:
                    cols = st.columns(min(len(speeds), 4))
                    for i, (name, speed) in enumerate(list(speeds.items())[:4]):
                        cols[i].metric(name, f"{speed:.1f} km/h")
            
            # Small delay to control frame rate
            time.sleep(0.03)
    
    elif not uploaded_file:
        st.info(" Sube un video en el panel izquierdo para comenzar")


if __name__ == "__main__":
    main()
