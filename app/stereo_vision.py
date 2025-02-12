import numpy as np
import cv2
import glob
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json
import os
from datetime import datetime

@dataclass
class Point3D:
    x: float
    y: float
    z: float

class StereoVisionSystem:
    def __init__(self):
        self.camera_matrix_l = None
        self.dist_coeffs_l = None
        self.camera_matrix_r = None
        self.dist_coeffs_r = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.Q = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.stereo_map_l = None
        self.stereo_map_r = None

    def calibrate_stereo(self, left_images: List[np.ndarray], 
                        right_images: List[np.ndarray], 
                        chess_size: Tuple[int, int]=(9,6), 
                        square_size: float=0.025) -> bool:
        """
        Calibrate stereo camera system using provided image pairs
        """
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chess_size[0], 0:chess_size[1]].T.reshape(-1,2)
        objp = objp * square_size

        objpoints = []
        imgpoints_l = []
        imgpoints_r = []

        for left_img, right_img in zip(left_images, right_images):
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            ret_l, corners_l = cv2.findChessboardCorners(left_gray, chess_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(right_gray, chess_size, None)

            if ret_l and ret_r:
                objpoints.append(objp)
                corners_l2 = cv2.cornerSubPix(left_gray, corners_l, (11,11), (-1,-1), criteria)
                corners_r2 = cv2.cornerSubPix(right_gray, corners_r, (11,11), (-1,-1), criteria)
                imgpoints_l.append(corners_l2)
                imgpoints_r.append(corners_r2)

        if len(objpoints) < 5:
            return False

        ret, self.camera_matrix_l, self.dist_coeffs_l, self.camera_matrix_r, self.dist_coeffs_r, \
        self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints, imgpoints_l, imgpoints_r,
            None, None, None, None,
            left_gray.shape[::-1], None, None,
            cv2.CALIB_FIX_INTRINSIC, criteria)

        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.camera_matrix_l, self.dist_coeffs_l,
            self.camera_matrix_r, self.dist_coeffs_r,
            left_gray.shape[::-1], self.R, self.T)

        self.stereo_map_l = cv2.initUndistortRectifyMap(
            self.camera_matrix_l, self.dist_coeffs_l, self.R1, self.P1,
            left_gray.shape[::-1], cv2.CV_16SC2)
        
        self.stereo_map_r = cv2.initUndistortRectifyMap(
            self.camera_matrix_r, self.dist_coeffs_r, self.R2, self.P2,
            left_gray.shape[::-1], cv2.CV_16SC2)

        return True

    def compute_depth_map(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """Compute depth map from rectified stereo images"""
        if self.stereo_map_l is None or self.stereo_map_r is None:
            return None

        # Rectify images
        left_rect = cv2.remap(left_img, self.stereo_map_l[0], self.stereo_map_l[1], cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.stereo_map_r[0], self.stereo_map_r[1], cv2.INTER_LINEAR)

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16*16,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        return disparity, left_rect, right_rect

    def get_3d_point(self, disparity_map: np.ndarray, x: int, y: int) -> Point3D:
        """Get 3D coordinates for a point in the disparity map"""
        point3d = cv2.reprojectImageTo3D(
            np.array([[[x, y, disparity_map[y, x]]]]), 
            self.Q
        )
        return Point3D(
            x=float(point3d[0][0][0]),
            y=float(point3d[0][0][1]),
            z=float(point3d[0][0][2])
        )

    def measure_distance(self, point1: Point3D, point2: Point3D) -> float:
        """Measure Euclidean distance between two 3D points"""
        return np.sqrt(
            (point1.x - point2.x)**2 + 
            (point1.y - point2.y)**2 + 
            (point1.z - point2.z)**2
        )

    def save_calibration(self, filename: str):
        """Save calibration parameters"""
        np.savez(filename,
                 camera_matrix_l=self.camera_matrix_l,
                 dist_coeffs_l=self.dist_coeffs_l,
                 camera_matrix_r=self.camera_matrix_r,
                 dist_coeffs_r=self.dist_coeffs_r,
                 R=self.R, T=self.T, E=self.E, F=self.F, Q=self.Q,
                 R1=self.R1, R2=self.R2, P1=self.P1, P2=self.P2)

    def load_calibration(self, filename: str) -> bool:
        """Load calibration parameters"""
        try:
            data = np.load(filename)
            self.camera_matrix_l = data['camera_matrix_l']
            self.dist_coeffs_l = data['dist_coeffs_l']
            self.camera_matrix_r = data['camera_matrix_r']
            self.dist_coeffs_r = data['dist_coeffs_r']
            self.R = data['R']
            self.T = data['T']
            self.E = data['E']
            self.F = data['F']
            self.Q = data['Q']
            self.R1 = data['R1']
            self.R2 = data['R2']
            self.P1 = data['P1']
            self.P2 = data['P2']
            return True
        except:
            return False