# test_cameras.py
import cv2
import numpy as np

def test_cameras():
    # Test first camera
    cap1 = cv2.VideoCapture(0)
    ret1, frame1 = cap1.read()
    if ret1:
        print("Camera 1 working")
        print(f"Resolution: {frame1.shape}")
    else:
        print("Camera 1 not accessible")
    cap1.release()

    # Test second camera
    cap2 = cv2.VideoCapture(1)
    ret2, frame2 = cap2.read()
    if ret2:
        print("Camera 2 working")
        print(f"Resolution: {frame2.shape}")
    else:
        print("Camera 2 not accessible")
    cap2.release()

if __name__ == "__main__":
    test_cameras()