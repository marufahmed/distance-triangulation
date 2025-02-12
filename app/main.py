import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
# from app.stereo_vision import StereoVisionSystem, Point3D
# from app.utils import get_camera_indexes, capture_frame, verify_camera_access
from stereo_vision import StereoVisionSystem, Point3D
from utils import get_camera_indexes, capture_frame, verify_camera_access
import os

# Initialize session state
if 'stereo_system' not in st.session_state:
    st.session_state.stereo_system = StereoVisionSystem()
if 'calibration_images_left' not in st.session_state:
    st.session_state.calibration_images_left = []
if 'calibration_images_right' not in st.session_state:
    st.session_state.calibration_images_right = []
if 'current_points' not in st.session_state:
    st.session_state.current_points = []
if 'disparity_map' not in st.session_state:
    st.session_state.disparity_map = None
if 'rectified_left' not in st.session_state:
    st.session_state.rectified_left = None



default_left = int(os.getenv("LEFT_CAMERA_INDEX", "1"))
default_right = int(os.getenv("RIGHT_CAMERA_INDEX", "2"))


def main():
    st.set_page_config(page_title="Stereo Vision System", layout="wide")
    
    # Debug information
    st.sidebar.subheader("Camera Detection")
    camera_indexes = get_camera_indexes()
    
    if not camera_indexes:
        st.error("No cameras detected!")
        if st.button("Run Diagnostics"):
            for i in range(4):
                st.write(f"Testing camera {i}:")
                st.write(verify_camera_access(i))
        return
    
    st.sidebar.write("Detected cameras:", camera_indexes)
    
    # Camera selection (force different cameras)
    left_cam = st.sidebar.selectbox("Left Camera", camera_indexes, index=0)
    right_options = [idx for idx in camera_indexes if idx != left_cam]
    right_cam = st.sidebar.selectbox("Right Camera", right_options, index=0)

    # Main area
    st.title("Stereo Vision Measurement System")
    
    # Tabs
    tab1, tab2 = st.tabs(["Calibration", "Measurement"])
    
    with tab1:
        st.header("Camera Calibration")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Capture Calibration Pair"):
                left_frame = capture_frame(left_cam)
                right_frame = capture_frame(right_cam)
                if left_frame is not None and right_frame is not None:
                    st.session_state.calibration_images_left.append(left_frame)
                    st.session_state.calibration_images_right.append(right_frame)
                    st.success(f"Captured pair {len(st.session_state.calibration_images_left)}")
        
        with col2:
            if st.button("Perform Calibration"):
                with st.spinner("Calibrating..."):
                    success = st.session_state.stereo_system.calibrate_stereo(
                        st.session_state.calibration_images_left,
                        st.session_state.calibration_images_right
                    )
                    if success:
                        st.success("Calibration successful!")
                        st.session_state.stereo_system.save_calibration("calibration.npz")
                    else:
                        st.error("Calibration failed. Ensure enough valid image pairs.")
        
        # Display calibration images
        if st.session_state.calibration_images_left:
            st.subheader("Latest Calibration Pair")
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state.calibration_images_left[-1], caption="Left Camera")
            with col2:
                st.image(st.session_state.calibration_images_right[-1], caption="Right Camera")
    
    with tab2:
        st.header("Distance Measurement")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Capture Measurement Frame"):
                left_frame = capture_frame(left_cam)
                right_frame = capture_frame(right_cam)
                if left_frame is not None and right_frame is not None:
                    disparity_map, rect_left, rect_right = st.session_state.stereo_system.compute_depth_map(
                        left_frame, right_frame
                    )
                    st.session_state.disparity_map = disparity_map
                    st.session_state.rectified_left = rect_left
                    st.session_state.current_points = []
            if st.button("Test Measurement System"):
                if not st.session_state.stereo_system.is_calibrated():
                    st.error("System not calibrated! Please complete calibration first.")
                else:
                    st.write("Calibration Parameters:")
                    st.write({
                        "Baseline (mm)": st.session_state.stereo_system.baseline,
                        "Focal Length": st.session_state.stereo_system.focal_length,
                        "Resolution": st.session_state.stereo_system.image_size
                    })
        
        if st.session_state.rectified_left is not None:
            st.image(st.session_state.rectified_left, caption="Click points to measure distance")
            
            # Handle click events
            clicked = st.image(st.session_state.rectified_left, caption="Click to select points")
            if clicked is not None:
                point = clicked
                st.session_state.current_points.append(point)
                
                if len(st.session_state.current_points) == 2:
                    p1 = st.session_state.stereo_system.get_3d_point(
                        st.session_state.disparity_map,
                        st.session_state.current_points[0][0],
                        st.session_state.current_points[0][1]
                    )
                    p2 = st.session_state.stereo_system.get_3d_point(
                        st.session_state.disparity_map,
                        st.session_state.current_points[1][0],
                        st.session_state.current_points[1][1]
                    )
                    distance = st.session_state.stereo_system.measure_distance(p1, p2)
                    st.success(f"Distance: {distance:.3f} meters")
                    st.session_state.current_points = []

if __name__ == "__main__":
    main()
