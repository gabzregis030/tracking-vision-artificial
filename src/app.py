
import streamlit as st
import cv2
import tempfile
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.controllers.app_controller import AppController

st.set_page_config(
    page_title="Vision Tracking System",
    page_icon="🤖",
    layout="wide"
)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def main():
    st.title("🤖 Vision Tracking System")
    
    # Sidebar Configuration
    st.sidebar.title("Configuration")
    
    # Input Source
    source_type = st.sidebar.radio("Input Source", ["Video File", "Webcam"])
    
    video_path = None
    if source_type == "Video File":
        uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if uploaded_file:
            video_path = save_uploaded_file(uploaded_file)
    else:
        video_path = 0  # Webcam index
        
    st.sidebar.markdown("---")
    
    # Tracking Options
    tracker_type = st.sidebar.selectbox(
        "Tracker Algorithm", 
        ["CSRT", "KCF", "MOSSE", "MEDIANFLOW", "MIL", "BOOSTING"],
        index=1  # Default to KCF usually fast
    )
    
    use_kalman = st.sidebar.checkbox("Enable Kalman Filter", value=True)
    
    use_detector = st.sidebar.checkbox("Auto-Detection (YOLO)", value=True)
    
    target_classes = None
    if use_detector:
        target_classes_input = st.sidebar.text_input("Classes to detect (space separated)", "person car sports ball")
        target_classes = target_classes_input.split()
    
    st.sidebar.markdown("---")
    
    # Control Buttons
    if 'controller' not in st.session_state:
        st.session_state.controller = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

    col1, col2 = st.sidebar.columns(2)
    
    start_btn = col1.button("Start Tracking", type="primary")
    stop_btn = col2.button("Stop")
    
    # Main Logic
    if start_btn:
        if video_path is None and source_type == "Video File":
            st.error("Please upload a video file first.")
        else:
            # Initialize Controller
            controller = AppController(
                video_source=video_path,
                tracker_type=tracker_type,
                use_kalman=use_kalman,
                use_detector=use_detector
            )
            
            if controller.initialize():
                st.session_state.controller = controller
                st.session_state.is_running = True
                
                # Initial detection/selection
                if use_detector:
                    with st.spinner("Detecting objects..."):
                        if not controller.detect_and_track(target_classes):
                            st.error("No objects detected! Try adjusting classes or using manual mode (not supported in web yet).")
                            st.session_state.is_running = False
                else:
                    st.warning("Manual selection is not fully supported in web interface yet. Please use Auto-Detection.")
                    # Fallback to center logic or error?
                    # For now just stop
                    st.session_state.is_running = False
            else:
                st.error("Failed to initialize video source.")

    if stop_btn:
        st.session_state.is_running = False
        if st.session_state.controller:
            st.session_state.controller.cleanup()
            st.session_state.controller = None
            
    # Display Area
    placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    if st.session_state.is_running and st.session_state.controller:
        while st.session_state.is_running:
            frame, metrics = st.session_state.controller.process_frame()
            
            if frame is None:
                st.info("Video finished.")
                st.session_state.is_running = False
                break
                
            # Display Frame
            placeholder.image(frame, channels="RGB", use_container_width=True)
            
            # Display Metrics
            with metrics_placeholder.container():
                cols = st.columns(3)
                cols[0].metric("Objects Tracked", metrics.get("object_count", 0))
                
                # Show speed of first object as example
                speeds = metrics.get("speeds", {})
                if speeds:
                    for i, (name, speed) in enumerate(speeds.items()):
                        if i < 2: # Show max 2 speeds to avoid clutter
                            cols[i+1].metric(f"{name} Speed", f"{speed:.1f} px/s")

if __name__ == "__main__":
    main()
