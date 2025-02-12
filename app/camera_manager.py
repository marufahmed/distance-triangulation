import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import time
from contextlib import contextmanager

@contextmanager
def video_capture(index: int, timeout: float = 2.0):
    """Safe context manager for VideoCapture"""
    cap = cv2.VideoCapture(index)
    try:
        start_time = time.time()
        while not cap.isOpened() and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        yield cap
    finally:
        if cap is not None:
            cap.release()

class CameraInfo:
    def __init__(self, index: int, busid: str, position: str):
        self.index = index
        self.busid = busid  # USB Bus ID from usbipd
        self.position = position  # 'left' or 'right'
        self.resolution = None
        self.fps = None

    def to_dict(self):
        return {
            'index': self.index,
            'busid': self.busid,
            'position': self.position,
            'resolution': self.resolution,
            'fps': self.fps
        }

    @staticmethod
    def from_dict(data: dict) -> 'CameraInfo':
        cam = CameraInfo(data['index'], data['busid'], data['position'])
        cam.resolution = data['resolution']
        cam.fps = data['fps']
        return cam

class CameraManager:
    def __init__(self, config_path: str = 'camera_config.json'):
        self.config_path = Path(config_path)
        self.cameras: Dict[str, CameraInfo] = {}
        self.load_config()

    def get_c922_cameras(self) -> List[str]:
        """Get list of Logitech C922 camera BUSIDs using usbipd"""
        c922_cameras = []
        try:
            result = subprocess.run(['usbipd', 'list'], capture_output=True, text=True)
            for line in result.stdout.splitlines():
                if '046d:085c' in line:  # Logitech C922
                    busid = line.split()[0]
                    c922_cameras.append(busid)
        except Exception as e:
            print(f"Error running usbipd: {e}")
        return c922_cameras

    def detect_cameras(self) -> Dict[str, CameraInfo]:
        """Detect and identify connected cameras"""
        cameras = {}
        c922_busids = self.get_c922_cameras()
        
        # Map camera indices to C922 cameras
        for idx in range(10):
            cap = cv2.VideoCapture(idx)
            try:
                if not cap.isOpened():
                    continue
                    
                # Try to capture a frame to verify camera works
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                # Read camera properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # If we have unmapped C922 cameras, associate them with this index
                if c922_busids:
                    busid = c922_busids.pop(0)
                    
                    # Create or update camera info
                    if busid not in self.cameras:
                        # For new cameras, assign position based on existing config
                        position = 'right' if 'left' in [c.position for c in self.cameras.values()] else 'left'
                        cam_info = CameraInfo(idx, busid, position)
                    else:
                        # Use existing position but update index
                        cam_info = self.cameras[busid]
                        cam_info.index = idx
                    
                    cam_info.resolution = (width, height)
                    cam_info.fps = fps
                    cameras[busid] = cam_info
                
            finally:
                cap.release()
        
        return cameras

    def verify_camera(self, index: int) -> bool:
        """Verify camera is accessible and can capture frames"""
        with video_capture(index) as cap:
            if not cap.isOpened():
                return False
            ret, frame = cap.read()
            return ret and frame is not None

    def update_cameras(self):
        """Update camera configuration"""
        self.cameras = self.detect_cameras()
        # Verify cameras are still accessible
        self.cameras = {
            busid: cam 
            for busid, cam in self.cameras.items() 
            if self.verify_camera(cam.index)
        }
        self.save_config()

    def get_camera_pair(self) -> Tuple[Optional[CameraInfo], Optional[CameraInfo]]:
        """Get the configured stereo camera pair"""
        left_cam = next((cam for cam in self.cameras.values() if cam.position == 'left'), None)
        right_cam = next((cam for cam in self.cameras.values() if cam.position == 'right'), None)
        return left_cam, right_cam

    def swap_positions(self):
        """Swap left/right camera positions"""
        for cam in self.cameras.values():
            cam.position = 'right' if cam.position == 'left' else 'left'
        self.save_config()

    def load_config(self):
        """Load camera configuration from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                self.cameras = {
                    busid: CameraInfo.from_dict(cam_data)
                    for busid, cam_data in data.items()
                }
            except Exception as e:
                print(f"Error loading camera config: {e}")
                self.cameras = {}

    def save_config(self):
        """Save camera configuration to file"""
        try:
            data = {
                busid: cam.to_dict()
                for busid, cam in self.cameras.items()
            }
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving camera config: {e}")

    def capture_frame(self, camera_index: int) -> Optional[np.ndarray]:
        """Capture a frame with retries and error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            with video_capture(camera_index) as cap:
                if cap.isOpened():
                    # Discard first few frames
                    for _ in range(2):
                        cap.read()
                    
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return frame
            time.sleep(0.5)
        return None