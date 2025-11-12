import os
import numpy as np
import cv2
from contextlib import contextmanager

@contextmanager
def createWindow(title):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    try:
        yield
    finally:
        cv2.destroyWindow(title)


@contextmanager
def openCapture(camera_index=1, image_width=1280, image_height=720):
    # Open default camera (0). On Windows, CAP_DSHOW often reduces latency when opening.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Optional: request a resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    # Disable auto-focus for stable calibration
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 50)  # set focus (adjust as needed)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1=auto exposure

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    try:
        yield cap
    finally:
        cap.release()


def load_calibration(path="calib.npz"):
    if not os.path.exists(path):
        return None, None
    d = np.load(path)
    return d["camera_matrix"], d["dist_coeffs"]


def save_calibration(camera_matrix, dist_coeffs, path="calib.npz"):
    np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Saved calibration to {path}")


def create_charuco_board():
    # --- ChArUco board parameters (adjust to your printed board) ---
    squares_x = 7   # number of chessboard squares in X direction
    squares_y = 5   # number of chessboard squares in Y direction
    square_length = 0.03  # meters (or any other unit) - used for pose/scale
    marker_length = 0.022  # must be < square_length

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    return board
