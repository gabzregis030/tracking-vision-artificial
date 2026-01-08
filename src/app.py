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
    page_title="Rastreo de Gatos",
    layout="wide"
)

# Modern UI Styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Elegant Title */
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        background: linear-gradient(120deg, #6c5ce7, #a55eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Metrics Cards */
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #a55eea;
    }
    .stMetric label {
        color: #636e72 !important;
        font-size: 0.9rem !important;
    }
    .stMetric div {
        color: #2d3436 !important;
        font-weight: 700 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #dfe6e9;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6c5ce7, #a55eea);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
    }
    
    /* Header style */
    .header-text {
        font-size: 1.2rem;
        color: #636e72;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)


class SmartTracker:
    """Smart tracker using YOLO detection and CSRT tracking."""
    
    def __init__(self, video_path, model_type="tiny"):
        self.video_path = video_path
        self.model_type = model_type  # Keep parameter for flexibility but default to tiny
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

    def calculate_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
        
    def initialize(self):
        """Initialize video and models."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        # Initialize YOLO Detector
        try:
            # Load detection model
            self.detector = ObjectDetector(
                method="yolo", 
                model_type=self.model_type,
                # Use 0.40 for Standard (reduce noise) and 0.15 for Tiny (catch everything)
                confidence_threshold=0.40 if self.model_type == "standard" else 0.15,
                nms_threshold=0.3 # Lower threshold = Aggressively remove overlapping boxes
            )
            print(f"YOLO Detector initialized ({self.model_type})")
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
        # Run periodically (every 10 frames) OR if zero cats tracked
        if self.frame_count % 10 == 0 or active_count == 0:
            # First pass: Normal sensitivity
            detections = self.detector.detect(frame, target_classes=["cat"])
            
            # Adaptive Sensitivity: If no cats found and we are trusting Tiny YOLO, try harder!
            if len(detections) == 0 and active_count == 0:
                print("No detections, trying high sensitivity mode...")
                detections = self.detector.detect(frame, target_classes=["cat"], confidence_threshold=0.05)
            
            if len(detections) > 0:
                print(f"Detectados {len(detections)} gatos en frame {self.frame_count}")
                
                # Get current trackers to compare
                current_bboxes = self.tracker.get_active_bboxes()
                
                new_objects_to_add = []
                for det in detections:
                    x, y, w, h, _, _ = det
                    det_box = (x, y, w, h)
                    
                    # Check overlap with ANY existing active tracker
                    is_new = True
                    for curr_box in current_bboxes:
                        iou = self.calculate_iou(det_box, curr_box)
                        if iou > 0.3: # If overlaps > 30%, assume it's the same cat
                            is_new = False
                            break
                    
                    if is_new:
                        new_objects_to_add.append(det_box)
                
                # Add ONLY the genuinely new objects
                if len(new_objects_to_add) > 0:
                    print(f"Agregando {len(new_objects_to_add)} nuevos gatos")
                    for box in new_objects_to_add:
                        # Initialize new tracker for this box
                        self.tracker.init(frame, box)
                        self.speed_calculator.add_object()
                    self.is_tracking = True
        
        # Update Tracking
        success_list, bboxes = self.tracker.update(frame)
        active_count = 0
        
        # Visualization & Metrics
        object_speeds = {}
        
        # Vibrant colors for visualization
        color_primary = (234, 94, 165) # Pink/Purple
        
        for i, (bbox, success) in enumerate(zip(bboxes, success_list)):
            if success:
                active_count += 1
                x, y, w, h = bbox
                center = (x + w/2, y + h/2)
                
                # Update speed
                speed = self.speed_calculator.update(i, center)
                speed_kmh = self.speed_calculator.get_speed_kmh(i)
                object_speeds[f"Gato {i+1}"] = speed_kmh
                
                # Draw Box - Stylish corners instead of full box? Let's keep box for clarity but nicer color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_primary, 2)
                
                # Draw Label with background
                label = f"Gato {i+1}: {speed_kmh:.1f} km/h"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-25), (x + w_text + 10, y), color_primary, -1)
                cv2.putText(frame, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
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
    st.title("Sistema de Rastreo de Gatos")
    st.markdown("<p class='header-text'>Análisis inteligente de movimiento y velocidad</p>", unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuración")
    
    uploaded_file = st.sidebar.file_uploader(
        "Seleccionar archivo de video", 
        type=['mp4', 'avi', 'mov', 'mpeg4'],
        help="Formatos soportados: MP4, AVI, MOV"
    )

    st.sidebar.markdown("### Configuración de Detección")
    
    # Simple toggle for "Hard Mode"
    use_enhance = st.sidebar.checkbox(
        "Modo Alta Precisión", 
        value=False,
        help="Actívalo si no detecta gatos (ej. cuando hay personas o poca luz). Usa un modelo más potente pero más lento."
    )
    
    model_type = "standard" if use_enhance else "tiny"

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Métricas en tiempo real")
    
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
    start_btn = col1.button("Iniciar Análisis", type="primary", use_container_width=True)
    stop_btn = col2.button("Detener", use_container_width=True)
    
    # Main display
    video_placeholder = st.empty()
    metrics_placeholder = st.empty()
    progress_bar = st.empty()
    
    if start_btn and uploaded_file:
        video_path = save_uploaded_file(uploaded_file)
        if video_path:
            # Use our new SmartTracker
            tracker = SmartTracker(video_path, model_type)
            if tracker.initialize():
                st.session_state.tracker = tracker
                st.session_state.is_running = True
                st.success("Sistema iniciado correctamente.")
            else:
                st.error("Error al inicializar el sistema de seguimiento.")
                
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
                st.info("Fin del video.")
                st.session_state.is_running = False
                break
                
            # Update UI
            video_placeholder.image(frame, channels="RGB", use_container_width=True)
            stats_frame.metric("Frame Procesado", f"{metrics.get('frame', 0)}/{metrics.get('total_frames', 0)}")
            stats_cats.metric("Objetos en escena", metrics.get('object_count', 0))
            
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
