import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import subprocess
import re
import os
import time
from contextlib import contextmanager

@contextmanager
def safe_video_capture(index: int, timeout: float = 2.0):
    """
    Safely handle VideoCapture with proper cleanup
    """
    cap = cv2.VideoCapture(index)
    try:
        start_time = time.time()
        while not cap.isOpened() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        yield cap
    finally:
        if cap is not None:
            cap.release()

def map_usb_to_camera_index() -> Dict[str, int]:
    """
    Map USB ports to camera indices for Logitech C922 cameras
    """
    camera_map = {}
    try:
        # Get USB device listing
        result = subprocess.run(['usbipd', 'list'], capture_output=True, text=True)
        c922_ports = []
        
        # Find C922 cameras
        for line in result.stdout.splitlines():
            if '046d:085c' in line:  # Logitech C922
                busid = line.split()[0]
                c922_ports.append(busid)
        
        # Try to match ports to camera indices
        for idx in range(10):  # Check reasonable range of indices
            with safe_video_capture(idx) as cap:
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        # Store the working index for each port
                        if len(camera_map) < len(c922_ports):
                            port = c922_ports[len(camera_map)]
                            camera_map[port] = idx
                            
                        if len(camera_map) == len(c922_ports):
                            break
    except Exception as e:
        print(f"Error mapping cameras: {str(e)}")
    
    return camera_map

def get_camera_indexes() -> List[int]:
    """
    Get list of available camera indices, prioritizing C922 cameras
    """
    camera_map = map_usb_to_camera_index()
    
    # If we found our C922 cameras, return their indices
    if camera_map:
        return sorted(camera_map.values())
    
    # Fallback: scan for any available cameras
    available_cameras = []
    for idx in range(10):
        with safe_video_capture(idx) as cap:
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(idx)
    
    return available_cameras

def verify_camera_access(index: int) -> Dict[str, any]:
    """
    Verify camera access with detailed diagnostics
    """
    diagnostics = {
        "index": index,
        "is_accessible": False,
        "can_capture": False,
        "frame_size": None,
        "fps": None,
        "error": None
    }
    
    try:
        with safe_video_capture(index) as cap:
            if cap.isOpened():
                diagnostics["is_accessible"] = True
                diagnostics["fps"] = cap.get(cv2.CAP_PROP_FPS)
                
                # Try to capture a frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    diagnostics["can_capture"] = True
                    diagnostics["frame_size"] = (frame.shape[1], frame.shape[0])
    except Exception as e:
        diagnostics["error"] = str(e)
    
    return diagnostics

def capture_frame(camera_index: int) -> Optional[np.ndarray]:
    """
    Capture a frame with retries and error handling
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with safe_video_capture(camera_index) as cap:
                if cap.isOpened():
                    # Discard first few frames
                    for _ in range(2):
                        cap.read()
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return frame
        except Exception as e:
            print(f"Capture attempt {attempt + 1} failed: {str(e)}")
            
        time.sleep(0.5)
    
    return None