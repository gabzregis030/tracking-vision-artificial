"""
Sistema de Tracking de Gatos - Interfaz Streamlit

Esta versión utiliza YOLO para detección robusta de gatos.
"""

import streamlit as st
import cv2
import tempfile
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import ObjectDetector, ObjectTracker, SpeedCalculator

st.set_page_config(
    page_title="Sistema de Tracking de Gatos",
    page_icon="🐈",
    layout="wide"
)

# Custom CSS
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


class SmartTracker:
    """Smart tracker using YOLO detection and CSRT tracking."""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        
        # Components
        self.detector = None
        self.tracker = None
        self.speed_calculator = None
        
        # State
        self.fps = 30
        self.total_frames = 0
        self.frame_count = 0
        self.detected_objects = []
        self.is_tracking = False
        
    def initialize(self):
        """Initialize video and models."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize YOLO Detector
        # We try to use the tiny model we configured
        try:
            # Lower confidence to 0.15 and increase NMS to 0.5 to allow overlapping cats
            self.detector = ObjectDetector(method="yolo", confidence_threshold=0.15, nms_threshold=0.5)
            print("YOLO Detector initialized with high sensitivity")
        except Exception as e:
            st.error(f"Error cargando detector: {e}")
            return False
            
        # Initialize Tracker
        self.tracker = ObjectTracker("CSRT")
        
        # Initialize Speed Calculator
        self.speed_calculator = SpeedCalculator(fps=self.fps, pixels_per_meter=100.0) # Cat calibration
        
        return True
    
    def process_frame(self):
        """Process a single frame."""
        if self.cap is None:
            return None, {}
            
        ret, frame = self.cap.read()
        if not ret:
            return None, {}
        
        self.frame_count += 1
        
        # Get currently active trackers
        active_count = self.tracker.count_active() if self.tracker else 0
        
        # Detection logic
        # Run detection periodically (every 15 frames) or if we aren't tracking anything
        if self.frame_count % 15 == 0 or active_count == 0:
            detections = self.detector.detect(frame, target_classes=["cat"])
            
            # If we found objects
            if len(detections) > 0:
                # If we aren't tracking anything, OR we found MORE objects than we are tracking
                # we re-initialize to catch the new ones.
                if active_count == 0 or len(detections) > active_count:
                    print(f"Detectados {len(detections)} gatos (Actualizando trackers)")
                    rois = [(x, y, w, h) for x, y, w, h, _, _ in detections]
                    self.tracker.init_multi(frame, rois)
                    
                    # Add to speed calculator
                    for _ in rois:
                        self.speed_calculator.add_object()
        
        # Update Tracking
        success_list, bboxes = self.tracker.update(frame)
        active_count = 0
        
        # Visualization & Metrics
        object_speeds = {}
        
        bright_green = (0, 255, 0)
        
        for i, (bbox, success) in enumerate(zip(bboxes, success_list)):
            if success:
                active_count += 1
                x, y, w, h = bbox
                center = (x + w/2, y + h/2)
                
                # Update speed
                speed = self.speed_calculator.update(i, center)
                speed_kmh = self.speed_calculator.get_speed_kmh(i)
                object_speeds[f"Gato {i+1}"] = speed_kmh
                
                # Draw Box
                cv2.rectangle(frame, (x, y), (x + w, y + h), bright_green, 2)
                
                # Draw Label
                label = f"Gato {i+1} : {speed_kmh:.1f} km/h"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bright_green, 2)
        
        # Convert for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        metrics = {
            'frame': self.frame_count,
            'total_frames': self.total_frames,
            'object_count': active_count,
            'speeds': object_speeds
        }
        
        return frame_rgb, metrics

    def release(self):
        if self.cap:
            self.cap.release()


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory."""
    try:
        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error al guardar archivo: {e}")
        return None


def main():
    st.title("🐈 Sistema de Tracking de Gatos (AI Powered)")
    st.markdown("Sube un video para detectar y rastrear gatos usando **YOLO Artificial Intelligence**.")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    
    uploaded_file = st.sidebar.file_uploader(
        "📹 Cargar Video", 
        type=['mp4', 'avi', 'mov', 'mpeg4'],
        help="Sube un video de gatos"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Estadísticas")
    
    # Placeholders
    stats_frame = st.sidebar.empty()
    stats_cats = st.sidebar.empty()
    
    # Session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
        
    # Controls
    col1, col2 = st.sidebar.columns(2)
    start_btn = col1.button("▶️ Iniciar", type="primary", use_container_width=True)
    stop_btn = col2.button("⏹️ Detener", use_container_width=True)
    
    # Main display
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.empty()
    
    if start_btn and uploaded_file:
        video_path = save_uploaded_file(uploaded_file)
        if video_path:
            # Use our new SmartTracker
            tracker = SmartTracker(video_path)
            if tracker.initialize():
                st.session_state.tracker = tracker
                st.session_state.is_running = True
                st.success("✅ IA Iniciada. Detectando gatos...")
            else:
                st.error("❌ Error al iniciar el motor de IA")
                
    if stop_btn:
        st.session_state.is_running = False
        if st.session_state.tracker:
            st.session_state.tracker.release()
            st.session_state.tracker = None
            
    # Loop
    if st.session_state.is_running and st.session_state.tracker:
        tracker = st.session_state.tracker
        
        while st.session_state.is_running:
            frame, metrics = tracker.process_frame()
            
            if frame is None:
                st.info("🎬 Video terminado")
                st.session_state.is_running = False
                break
                
            # Update UI
            video_placeholder.image(frame, channels="RGB", use_container_width=True)
            stats_frame.metric("Frame", f"{metrics.get('frame', 0)}/{metrics.get('total_frames', 0)}")
            stats_cats.metric("Gatos Detectados", metrics.get('object_count', 0))
            
            if metrics.get('total_frames', 0) > 0:
                progress_bar.progress(metrics.get('frame', 0) / metrics.get('total_frames', 1))
                
            with metrics_placeholder.container():
                speeds = metrics.get('speeds', {})
                if speeds:
                    cols = st.columns(min(len(speeds), 4))
                    for i, (name, speed) in enumerate(list(speeds.items())[:4]):
                        cols[i].metric(name, f"{speed:.1f} km/h")
            
            # Reduce sleep since YOLO processing takes time
            # time.sleep(0.01) 

if __name__ == "__main__":
    main()
