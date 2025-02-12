import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import time
from stereo_vision import StereoVisionSystem, Point3D
from utils import get_camera_indexes, capture_frame, CameraInfo
import os
import logging
import time


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def main():
    st.set_page_config(page_title="Stereo Vision System", layout="wide")
    
    # Debug information
    st.sidebar.subheader("Camera Detection")
    cameras = get_camera_indexes()
    
    if not cameras:
        st.error("No C922 cameras detected!")
        return
    
    st.sidebar.write("Detected cameras:", [(cam.port, cam.index, cam.resolution) for cam in cameras])
    
    # Camera selection (force different cameras)
    left_cam = st.sidebar.selectbox(
        "Left Camera",
        cameras,
        index=0,
        format_func=lambda x: f"Port {x.port} (Index {x.index})"
    )
    right_options = [cam for cam in cameras if cam != left_cam]
    right_cam = st.sidebar.selectbox(
        "Right Camera",
        right_options,
        index=0,
        format_func=lambda x: f"Port {x.port} (Index {x.index})"
    )

    # Main area
    st.title("Stereo Vision Measurement System")
    
    # Tabs
    tab1, tab2 = st.tabs(["Calibration", "Measurement"])
    
    with tab1:
        st.header("Camera Calibration")
        col1, col2 = st.columns(2)
        

        with col1:
            if st.button("Capture Calibration Pairs"):
                captured_pairs = 0
                start_time = time.time()

                while captured_pairs < 20:
                    left_frame = capture_frame(left_cam)
                    right_frame = capture_frame(right_cam)

                    if left_frame is not None and right_frame is not None:
                        st.session_state.calibration_images_left.append(left_frame)
                        st.session_state.calibration_images_right.append(right_frame)
                        captured_pairs += 1
                        st.success(f"Captured pair {captured_pairs}")
                        logging.info(f"Successfully captured calibration pair {captured_pairs}")
                    else:
                        st.error("Failed to capture frames from one or both cameras")
                        logging.error("Failed to capture calibration frames")

                    elapsed_time = time.time() - start_time
                    remaining_time = 1 - elapsed_time
                    if remaining_time > 0:
                        time.sleep(remaining_time / (20 - captured_pairs))  # Adjust sleep time to capture within 1 second


        with col2:
            if st.button("Perform Calibration"):
                if len(st.session_state.calibration_images_left) < 5:
                    st.error("Need at least 5 image pairs for calibration")
                    logging.warning("Attempted calibration with insufficient image pairs")
                else:
                    with st.spinner("Calibrating..."):
                        success = st.session_state.stereo_system.calibrate_stereo(
                            st.session_state.calibration_images_left,
                            st.session_state.calibration_images_right
                        )
                        if success:
                            st.success("Calibration successful!")
                            st.session_state.stereo_system.save_calibration("calibration.npz")
                            logging.info("Calibration completed and saved successfully")
                        else:
                            st.error("Calibration failed. Ensure enough valid image pairs.")
                            logging.error("Calibration failed")
        
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
                if not st.session_state.stereo_system.is_calibrated():
                    st.error("System not calibrated! Please complete calibration first.")
                    logging.warning("Attempted measurement without calibration")
                    return
                    
                left_frame = capture_frame(left_cam)
                right_frame = capture_frame(right_cam)
                if left_frame is not None and right_frame is not None:
                    disparity_map, rect_left, rect_right = st.session_state.stereo_system.compute_depth_map(
                        left_frame, right_frame
                    )
                    if disparity_map is not None:
                        st.session_state.disparity_map = disparity_map
                        st.session_state.rectified_left = rect_left
                        st.session_state.current_points = []
                        logging.info("Successfully captured and processed measurement frames")
                    else:
                        st.error("Failed to compute depth map")
                        logging.error("Depth map computation failed")
                else:
                    st.error("Failed to capture frames")
                    logging.error("Failed to capture measurement frames")
                    
            if st.button("Test Measurement System"):
                if not st.session_state.stereo_system.is_calibrated():
                    st.error("System not calibrated! Please complete calibration first.")
                    logging.warning("Attempted system test without calibration")
                else:
                    st.write("Calibration Parameters:")
                    st.write({
                        "Baseline (mm)": st.session_state.stereo_system.baseline,
                        "Focal Length": st.session_state.stereo_system.focal_length,
                        "Resolution": st.session_state.stereo_system.image_size
                    })
                    logging.info("System test completed successfully")
        
        if st.session_state.rectified_left is not None:
            st.image(st.session_state.rectified_left, caption="Click points to measure distance")
            
            # Handle click events
            clicked = st.image(st.session_state.rectified_left, caption="Click to select points")
            if clicked is not None:
                point = clicked
                st.session_state.current_points.append(point)
                logging.info(f"Point selected at {point}")
                
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
                    
                    if p1 is not None and p2 is not None:
                        distance = st.session_state.stereo_system.measure_distance(p1, p2)
                        if distance is not None:
                            st.success(f"Distance: {distance:.3f} meters")
                            logging.info(f"Distance measurement: {distance:.3f} meters")
                        else:
                            st.error("Failed to calculate distance")
                            logging.error("Distance calculation failed")
                    else:
                        st.error("Failed to compute 3D points")
                        logging.error("3D point computation failed")
                    
                    st.session_state.current_points = []

if __name__ == "__main__":
    main()