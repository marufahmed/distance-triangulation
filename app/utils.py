import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, NamedTuple
import subprocess
import time
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CameraInfo(NamedTuple):
    index: int
    port: str
    resolution: Tuple[int, int]
    is_c922: bool

@contextmanager
def safe_video_capture(index: int, resolution: Optional[Tuple[int, int]] = None, timeout: float = 2.0):
    """
    Safely handle VideoCapture with proper cleanup and initialization
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # Force DirectShow backend
    if resolution:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cap.set(cv2.CAP_PROP_FOCUS, 0)      # Set focus to infinity
    
    try:
        start_time = time.time()
        while not cap.isOpened() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {index} after {timeout} seconds")
            
        yield cap
    finally:
        if cap is not None:
            cap.release()

def verify_camera_capabilities(cap) -> bool:
    """
    Verify if the camera supports the required capabilities for our system
    """
    try:
        # Set and verify resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Read a test frame
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
            
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Verify if we got the requested resolution
        return actual_width == 1280 and actual_height == 720
    except:
        return False

def initialize_cameras() -> Dict[str, CameraInfo]:
    """
    Initialize and map cameras with consistent settings
    """
    camera_map = {}
    target_resolution = (1280, 720)
    
    try:
        # Get USB device listing
        result = subprocess.run(['usbipd', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("Failed to run usbipd list command")
            return {}
            
        c922_ports = []
        
        # Find C922 cameras
        for line in result.stdout.splitlines():
            if '046d:085c' in line and 'Shared' in line:  # Logitech C922
                busid = line.split()[0].strip()
                c922_ports.append(busid)
        
        logging.info(f"Found C922 cameras on ports: {c922_ports}")
        
        # Find available cameras that support our requirements
        available_cameras = []
        for idx in range(4):  # Check first 4 indices to avoid excessive scanning
            try:
                with safe_video_capture(idx, resolution=target_resolution) as cap:
                    if cap.isOpened() and verify_camera_capabilities(cap):
                        resolution = (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        )
                        available_cameras.append((idx, resolution))
                        logging.info(f"Found compatible camera at index {idx} with resolution {resolution}")
            except Exception as e:
                logging.warning(f"Error checking camera {idx}: {str(e)}")
        
        # Match available cameras with C922 ports
        for i, (idx, resolution) in enumerate(available_cameras):
            if i < len(c922_ports):
                port = c922_ports[i]
                camera_map[port] = CameraInfo(
                    index=idx,
                    port=port,
                    resolution=resolution,
                    is_c922=True
                )
                logging.info(f"Mapped port {port} to camera index {idx} with resolution {resolution}")
    
    except Exception as e:
        logging.error(f"Error during camera initialization: {str(e)}")
    
    return camera_map

def get_camera_indexes() -> List[CameraInfo]:
    """
    Get list of available camera information, specifically for shared C922 cameras
    """
    camera_map = initialize_cameras()
    
    if camera_map:
        # Sort by camera index to ensure consistent ordering
        cameras = sorted(camera_map.values(), key=lambda x: x.index)
        logging.info(f"Using cameras: {cameras}")
        return cameras
    
    return []

def capture_frame(camera_info: CameraInfo) -> Optional[np.ndarray]:
    """
    Capture a frame with retries and error handling
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with safe_video_capture(camera_info.index, camera_info.resolution) as cap:
                if cap.isOpened():
                    # Discard first few frames
                    for _ in range(3):
                        cap.read()
                        time.sleep(0.1)
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return frame
                    
                    logging.warning(f"Failed to capture frame from camera {camera_info.port} on attempt {attempt + 1}")
        except Exception as e:
            logging.error(f"Capture attempt {attempt + 1} failed for camera {camera_info.port}: {str(e)}")
            
        time.sleep(0.5)
    
    return None